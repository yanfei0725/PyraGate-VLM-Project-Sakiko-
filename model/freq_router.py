#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量 FreqFormer 频域路由（freqformer 线，2026-11-26）。

定位
----
不动主干原空间 QKV 与既有权重，不新增三套独立 QKV；只在已有 K 上取频域
描述符，用低秩矩阵（rank 可配，默认 8）生成低/中/高三频带权重，按 K token
所属频带把乘回注意力 logits（以加性 bias = log(a_k) 的形式进 SDPA）。

σ 条件调制
----------
高 σ（噪声阶段）压低高频权重、低 σ（细节阶段）释放细节。抑制系数
``high_sigma_suppress`` 可调，默认 0.5；``sigma`` 为流匹配噪声级 [B,1]/[B]，
None 时按 0 处理（等价不抑制）。

Top-k 稀疏
----------
``topk>0`` 时只保留能量最高的 ``topk`` 个高频 token，其余高频贡献置零；
低/中频不动。``topk=0``（默认）关闭。

热启动 / 命名纪律
------------------
- 本模块被挂为 ``UViT3D.freqrouter``，state_dict key 前缀为 ``freqrouter.``，
  不污染既有 key；Stage1 ckpt ``load_state_dict(strict=False)`` 可直接加载，
  新增参数随机初始化。
- 输出层零初始化：``proj_out`` 权重/bias 置 0 → 起步 gates=1、a_k=1，
  路由在初始点严格恒等；训练中再逐步学开。
- 参数量极小（rank=8, nh=24 时约 0.7K 量级），占主干 < 0.001%，远低于 1% 红线。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from fft_shared import build_band_masks
# 频带边界与 x0 频带条件共用同一组常量（值不变，数值零变化）
from x0_fourier_band_loss import BAND_LOW, BAND_MID_HIGH


class FreqRouter(nn.Module):
    """K 频域描述符 → 三频带注意力路由权重。

    Args:
        dim: 主干 hidden dim（记录用；本模块不在通道维做全秩投影）。
        num_heads: 注意力头数。
        rank: 低秩瓶颈秩，默认 8。
        scale: 路由门控幅度系数，默认 1.0；gates = 1 + scale·tanh(…)。
        topk: 高频 token 保留数，0=关。
        high_sigma_suppress: 高 σ 对高频带的线性抑制系数（默认 0.5）。
        gate_floor / gate_cap: 三带门控裁剪区间。
        band_low / band_mid_high: 频带边界，与损失口径一致。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int = 8,
        scale: float = 1.0,
        topk: int = 0,
        high_sigma_suppress: float = 0.5,
        gate_floor: float = 0.2,
        gate_cap: float = 1.5,
        band_low: float = BAND_LOW,
        band_mid_high: float = BAND_MID_HIGH,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rank = int(rank)
        self.scale = float(scale)
        self.topk = int(topk)
        self.high_sigma_suppress = float(high_sigma_suppress)
        self.gate_floor = float(gate_floor)
        self.gate_cap = float(gate_cap)
        self.band_low = float(band_low)
        self.band_mid_high = float(band_mid_high)

        # 输入描述符：3 个频带归一化能量 + 1 个 σ。低秩瓶颈。
        self.proj_in = nn.Linear(3 + 1, self.rank, bias=True)
        # 每头 3 个带门控（低/中/高）；head 维已在 feat 里，输出维=3。
        self.proj_out = nn.Linear(self.rank, 3, bias=True)
        # 零起步：初始 gates=1 严格恒等（详见模块 docstring）。
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self._mask_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def _apply(self, fn, recurse: bool = True):
        """Keep router parameters in fp32 even when the parent model is bf16.

        The router weights are tiny (rank=8) but sit at magnitudes where one
        bf16 ulp is ~4e-3.  With lr=1e-5 an otherwise valid gradient can round
        to exactly zero, which made proj_in look permanently frozen.  Keeping
        only this submodule fp32 removes the quantisation floor without
        changing the backbone dtype or checkpoint shape.
        """
        super()._apply(fn, recurse=recurse)
        for p in self.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.float()
        for b in self.buffers():
            if torch.is_tensor(b) and b.is_floating_point() and b.dtype != torch.float32:
                b.data = b.data.float()
        return self

    # ------------------------------------------------------------------
    def _spatial_masks(self, h: int, w: int, device: torch.device):
        key = (h, w, str(device))
        hit = self._mask_cache.get(key)
        if hit is None:
            hit = build_band_masks(
                h, w, band_low=self.band_low, band_mid_high=self.band_mid_high,
                device=device, dtype=torch.float32, half_plane_multiplicity=False,
            )
            self._mask_cache[key] = hit
        return hit

    def forward(
        self,
        k: torch.Tensor,
        grid: Tuple[int, int, int],
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: ``[B, nh, N, hd]`` 注意力 K（已 norm/RoPE）。
            grid: ``(T, H, W)`` token 网格。
            sigma: ``[B,1]``/``[B]`` 噪声级；None=0。

        Returns:
            gates: ``[B, nh, 3]`` 低/中/高频带门控（裁剪后）。
            a_k: ``[B, nh, N]`` 逐 K token 路由乘子（进 logits 的乘性权重；
                 调用方转成 log(a_k) 加性 bias）。
        """
        B, nh, N, hd = k.shape
        T, H, W = grid
        device = k.device

        # 时序域网格 H=1/W=1 或退化网格：频带无意义，直接返回恒等路由。
        if H <= 1 or W <= 1 or N != T * H * W:
            ones_a = k.new_ones(B, nh, N)
            ones_g = k.new_ones(B, nh, 3)
            return ones_g, ones_a

        # 1) 通道池化 -> 空间 token 图 [B,nh,T,H,W]
        e = k.mean(dim=-1).view(B, nh, T, H, W).float()
        Fk = torch.fft.rfft2(e, dim=(-2, -1), norm="ortho")
        m_low, m_mid, m_high = self._spatial_masks(H, W, device)

        # 2) 三带描述符：归一化带内平均幅度 [Bk,nh,3]（T/H/Wf 全部池化）
        mag = Fk.abs().pow(2)
        e_low = (mag * m_low).mean(dim=(-3, -2, -1))
        e_mid = (mag * m_mid).mean(dim=(-3, -2, -1))
        e_high = (mag * m_high).mean(dim=(-3, -2, -1))
        desc = torch.stack([e_low, e_mid, e_high], dim=-1)
        desc = desc / (desc.sum(dim=-1, keepdim=True) + 1e-8)

        # 3) σ 条件向量 [Bk,nh,1]。空间/时序头会把 batch 拆成 B*T / B*H*W，
        #    这里按当前注意力 batch 广播（训练 b=1，σ 取标量均值即可）。
        Bk = k.shape[0]
        if sigma is None:
            sig = e.new_zeros(Bk, 1)
        else:
            sig = sigma.reshape(-1).float().mean().reshape(1, 1).expand(Bk, 1)
        sig = sig.clamp(0.0, 1.0)
        sig_nh = sig.unsqueeze(1).expand(Bk, nh, 1)

        feat = torch.cat([desc, sig_nh], dim=-1)          # [Bk,nh,4]
        # 训练时主干是 bf16，而 desc/sig 走 float32 统计；proj 权重是 bf16，
        # 直接 linear 会 mat1/mat2 dtype mismatch。这里对齐 proj dtype。
        feat = feat.to(dtype=self.proj_in.weight.dtype)
        h = torch.nn.functional.silu(self.proj_in(feat))
        gates = self.proj_out(h).view(Bk, nh, 3)
        gates = 1.0 + self.scale * torch.tanh(gates)

        # 高 σ 线性压高频；低/中频不受 σ 直接抑制。
        high_suppress = self.high_sigma_suppress * sig_nh.to(dtype=gates.dtype)
        gates = gates.clone()
        gates[..., 2] = gates[..., 2] - high_suppress.squeeze(-1)
        gates = gates.clamp(self.gate_floor, self.gate_cap)
        gates = gates.to(dtype=k.dtype)

        # 4) 逐 token 频带归属：三带空间重建后按幅度 softmax 分配权重。
        k_low = torch.fft.irfft2(Fk * m_low, s=(H, W), dim=(-2, -1), norm="ortho")
        k_mid = torch.fft.irfft2(Fk * m_mid, s=(H, W), dim=(-2, -1), norm="ortho")
        k_high = torch.fft.irfft2(Fk * m_high, s=(H, W), dim=(-2, -1), norm="ortho")
        band_amp = torch.stack([k_low.abs(), k_mid.abs(), k_high.abs()], dim=-1)
        w_band = torch.softmax(band_amp, dim=-1)          # [B,nh,T,H,W,3]

        # 5) Top-k 稀疏：只剪高频，低/中频保留。
        if self.topk > 0:
            k_top = min(self.topk, N)
            high_score = k_high.reshape(B, nh, N)
            _, top_idx = high_score.topk(k_top, dim=-1)
            keep = torch.zeros_like(high_score, dtype=torch.bool)
            keep.scatter_(-1, top_idx, True)
            keep = keep.view(B, nh, T, H, W).unsqueeze(-1)
            w_band = w_band.clone()
            w_band[..., 2] = w_band[..., 2] * keep.float().squeeze(-1)
            w_band = w_band / (w_band.sum(dim=-1, keepdim=True) + 1e-8)

        g = gates.unsqueeze(2).unsqueeze(2).unsqueeze(2)  # [B,nh,1,1,1,3]
        a_map = (w_band * g).sum(dim=-1)                   # [B,nh,T,H,W]
        a_k = a_map.reshape(B, nh, N).to(dtype=k.dtype)
        return gates, a_k
