#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次 RFFT2 全复用的共享频谱缓存（freqformer 线，2026-11-26）。

设计目标
--------
同一步训练内，v-band 能量损失、低频 TV 结构正则、相位一致性、A1 频带条件向量、
FreqFormer 路由描述符原本各自对同一张量做一次 ``torch.fft.rfft2``。本模块把这
一次 FFT 算完后缓存，所有下游消费方共用同一组 ``x0hat / 掩膜 / 分量 / 占比 /
低频 IFFT / 相位角``，杜绝重复 FFT 占显存、耗算力。

形状与频带口径（与 ``x0_fourier_band_loss.py`` 完全一致）
---------------------------------------------------------
- 输入 ``x0``: 5D ``[B, C, T, H, W]``；rfft2 只作用于最后两维 ``(H, W)``，
  T 维不动。
- 频带（归一化空间频率 cycles/pixel）：
  low ``r <= band_low``（默认 0.05）；mid ``band_low < r <= band_mid_high``
  （默认 0.25）；high ``r > band_mid_high``。
- 能量口径掩膜带半平面乘子（W=0 / Nyquist 列计 1，其余计 2）；空间重建口径
  掩膜不带乘子（irfft2 自己合成 Hermitian 半平面）。

dtype 约定
----------
FFT 在 ROCm/torch 上对 bf16 不保证可靠，输入 bf16 时先升 fp32 再算；输出
``x0hat`` 为 complex64，掩膜/占比/IFFT/相位为 fp32。消费方需要 bf16 时自行
``.to(dtype)``，本模块不主动把复数压回半精度（避免静默丢虚部）。

缓存纪律
--------
- 缓存键 = ``(data_ptr, shape, tag)``。同一张量在同一步内重复取频谱只算 1 次。
- **每步必须显式 ``reset()``**：上一步张量释放后 data_ptr 可能被新张量复用，
  跨步残留条目会拿到陈旧图/陈旧数值。训练循环在每步损失计算前调用一次即可。
- 缓存条目持有参与反传的复数激活；``reset()`` 后旧条目才会被释放。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def build_band_masks(
    h: int,
    w: int,
    *,
    band_low: float = 0.05,
    band_mid_high: float = 0.25,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    half_plane_multiplicity: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 x0_fourier_band_loss._band_masks 同口径的低/中/高频掩膜。

    返回三个 ``[H, W//2+1]`` 掩膜（顺序 low, mid, high）。
    ``half_plane_multiplicity=True`` 用于能量统计；``False`` 用于 irfft2 重建。
    """
    if not (0.0 < band_low < band_mid_high < 0.5 ** 0.5):
        raise ValueError(
            f"invalid band edges: low={band_low}, mid_high={band_mid_high}; "
            "expected 0 < low < mid_high < sqrt(0.5)"
        )
    fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device, dtype=dtype)
    radial = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))

    wf = w // 2 + 1
    if half_plane_multiplicity:
        mult = torch.full((wf,), 2.0, device=device, dtype=dtype)
        mult[0] = 1.0
        if w % 2 == 0:
            mult[-1] = 1.0
    else:
        mult = torch.ones((wf,), device=device, dtype=dtype)

    low = (radial <= float(band_low)).to(dtype) * mult[None, :]
    mid = ((radial > float(band_low)) & (radial <= float(band_mid_high))).to(dtype) * mult[None, :]
    high = (radial > float(band_mid_high)).to(dtype) * mult[None, :]
    return low, mid, high


@dataclass
class SpectrumEntry:
    """一次 FFT 的全部复用产物。字段保持公开，供损失/正则直接取用。"""

    tag: str
    source_shape: Tuple[int, ...]
    x0hat: torch.Tensor                 # complex64 [B,C,T,H,Wf]
    mask_low: torch.Tensor              # fp32 [H,Wf] 能量口径（带乘子）
    mask_mid: torch.Tensor
    mask_high: torch.Tensor
    mask_low_sp: torch.Tensor            # fp32 [H,Wf] 重建口径（无乘子）
    mask_mid_sp: torch.Tensor
    mask_high_sp: torch.Tensor
    comp_low: torch.Tensor              # complex64 [B,C,T,H,Wf]
    comp_mid: torch.Tensor
    comp_high: torch.Tensor
    ratios: torch.Tensor                # fp32 [B,3] 低/中/高能量占比
    low_ifft: torch.Tensor              # fp32 [B,C,T,H,W] 低频分量 IFFT
    phase: torch.Tensor                 # fp32 [B,C,T,H,Wf] 相位角（弧度）
    band_low: float
    band_mid_high: float

    def components_spatial(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """三带空间分量（irfft2 重建，fp32，detach 语义由调用方决定）。"""
        b, c, t, h, wf = self.x0hat.shape
        w = (wf - 1) * 2
        out = []
        for comp, m_sp in (
            (self.comp_low, self.mask_low_sp),
            (self.comp_mid, self.mask_mid_sp),
            (self.comp_high, self.mask_high_sp),
        ):
            out.append(
                torch.fft.irfft2(comp, s=(h, w), dim=(-2, -1), norm="ortho")
            )
        return out[0], out[1], out[2]


class SharedSpectrum:
    """按 ``(data_ptr, shape, tag)`` 缓存的单次 FFT 复用对象。

    典型用法（训练循环，每步一次）::

        spec_cache = SharedSpectrum()
        for step in loader:
            spec_cache.reset()
            spec_pred = spec_cache.compute(x0hat, tag="pred")
            spec_teach = spec_cache.compute(x0, tag="teach")   # 另一张量，第二次 FFT
            loss = band_loss(..., spectrum_pred=spec_pred, spectrum_teach=spec_teach)

    注意：pred 与 teach 是两张不同张量，各算一次 FFT；同一张量的多个下游消费
    只算一次。
    """

    def __init__(self, band_low: float = 0.05, band_mid_high: float = 0.25,
                 max_entries: int = 4):
        self.band_low = float(band_low)
        self.band_mid_high = float(band_mid_high)
        self.max_entries = int(max_entries)
        self._cache: Dict[Tuple[int, Tuple[int, ...], str], SpectrumEntry] = {}
        self._entry: Optional[SpectrumEntry] = None   # 最近一次条目（便捷访问）
        self.fft_calls: int = 0          # 本缓存实际触发的 rfft2 次数（自检用）

    def reset(self) -> None:
        """每步开始时调用：丢弃上一步缓存，释放残留激活图。"""
        self._cache.clear()
        self._entry = None

    def compute(self, x0: torch.Tensor, tag: str = "x0", force: bool = False) -> SpectrumEntry:
        """计算（或命中缓存）x0 的完整频谱产物。

        Args:
            x0: 5D ``[B,C,T,H,W]``，bf16/fp16/fp32 均可；内部升 fp32 做 FFT。
            tag: 同形状同 data_ptr 时区分不同语义来源（如 "pred"/"teach"）。
            force: 跳过缓存强制重算（自检/调试用）。
        """
        if x0.ndim != 5:
            raise ValueError(f"x0 必须是 5D [B,C,T,H,W]，收到 {tuple(x0.shape)}")
        key = (x0.data_ptr(), tuple(x0.shape), tag)
        if not force and key in self._cache:
            return self._cache[key]

        b, c, t, h, w = x0.shape
        xf = x0.float()
        x0hat = torch.fft.rfft2(xf, dim=(-2, -1), norm="ortho")
        self.fft_calls += 1

        device = x0hat.device
        m_low, m_mid, m_high = build_band_masks(
            h, w, band_low=self.band_low, band_mid_high=self.band_mid_high,
            device=device, dtype=torch.float32, half_plane_multiplicity=True,
        )
        m_low_sp, m_mid_sp, m_high_sp = build_band_masks(
            h, w, band_low=self.band_low, band_mid_high=self.band_mid_high,
            device=device, dtype=torch.float32, half_plane_multiplicity=False,
        )

        comp_low = x0hat * m_low_sp
        comp_mid = x0hat * m_mid_sp
        comp_high = x0hat * m_high_sp

        # 三带能量占比（能量口径掩膜带半平面乘子，总和即全 FFT 能量）。
        mag_sq = x0hat.abs().pow(2)
        e_low = (mag_sq * m_low).sum(dim=(-2, -1)).sum(dim=(1, 2))
        e_mid = (mag_sq * m_mid).sum(dim=(-2, -1)).sum(dim=(1, 2))
        e_high = (mag_sq * m_high).sum(dim=(-2, -1)).sum(dim=(1, 2))
        energy = torch.stack([e_low, e_mid, e_high], dim=-1)
        ratios = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)

        low_ifft = torch.fft.irfft2(
            comp_low, s=(h, w), dim=(-2, -1), norm="ortho"
        )
        phase = torch.angle(x0hat)

        entry = SpectrumEntry(
            tag=tag, source_shape=tuple(x0.shape),
            x0hat=x0hat,
            mask_low=m_low, mask_mid=m_mid, mask_high=m_high,
            mask_low_sp=m_low_sp, mask_mid_sp=m_mid_sp, mask_high_sp=m_high_sp,
            comp_low=comp_low, comp_mid=comp_mid, comp_high=comp_high,
            ratios=ratios, low_ifft=low_ifft, phase=phase,
            band_low=self.band_low, band_mid_high=self.band_mid_high,
        )
        # 上限内的小字典缓存；超限时丢弃最早条目（跨步已 reset，不会泄漏）。
        if len(self._cache) >= self.max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = entry
        self._entry = entry
        return entry

    # ---- 便捷访问（语义同 _entry 字段）------------------------------------
    @property
    def entry(self) -> Optional[SpectrumEntry]:
        return self._entry

    def ratios_of(self, x0: torch.Tensor, tag: str = "x0") -> torch.Tensor:
        return self.compute(x0, tag=tag).ratios

    def low_component(self, x0: torch.Tensor, tag: str = "x0") -> torch.Tensor:
        return self.compute(x0, tag=tag).low_ifft

    def phase_of(self, x0: torch.Tensor, tag: str = "x0") -> torch.Tensor:
        return self.compute(x0, tag=tag).phase
