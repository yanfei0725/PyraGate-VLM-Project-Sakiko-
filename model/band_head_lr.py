# -*- coding: utf-8 -*-
"""band_head_lr.py — 多尺度低秩频带头（bandhead 线，2026-11-26）。

把原 A3 三分频头（``full_gated_head.py`` 里的 ``band_head_l/m/h``，三套独立
DW3x3 + SiLU + 1x1 投影）合并为「一次共享投影 + 频域分频 + 每带低秩分解」
的一套头。

设计对照（与 FUSION_CONTRACT.md §1/§2 对齐）：

1. **共享投影只做一次**：``shared_down`` (in_dim -> shared_rank) 与
   ``shared_up`` (shared_rank -> in_dim) 组成一个低秩共享投影，三个频带复用
   同一份投影结果；不再有三套独立头投影。
2. **频域直接分频，不在 latent 空间硬切**：对共享特征做 ``rfft2``，用
   ``x0_fourier_band_loss._band_masks`` 的同一套径向掩膜（low r<=0.05，
   mid 0.05<r<=0.25，high r>0.25）拆出低/中/高三个空间频带，再
   ``irfft2`` 回空间域。分频在 token 空间的原始频谱上完成，不依赖
   unpatchify 之后的 latent 带限。
3. **低频直接预测，中高频残差**：``low`` 输出作为大结构/轮廓的相位基准
   （direct，非残差）；``mid`` / ``high`` 只作为细节残差。本模块只产出
   三分量，**不自行叠加到主速度场**；如何组合由集成线在 ``train_staged.py``
   接线阶段决定。
4. **每带低秩分解（rank 默认 8）**：每个频带子头 = ``u: in_dim -> rank``
   (1x1, 无 bias) + ``v: rank -> out_dim`` (1x1, 带 bias)。``u`` 随机小值、
   ``v`` 零初始化：step-0 输出严格为 0，同时 v.weight 有非零梯度。
5. **高频频域乘掩膜（冻结）**：高频子头输出再在原始频谱上乘 high 掩膜，
   掩膜是 ``requires_grad=False`` 的缓存 buffer；不做 latent 空间硬限制。
6. **结构感知无独立分支**：结构正则直接从低频分量提 Sobel 梯度，返回标量
   正则项（``structure_reg``）；本模块不接训练回路，训练端按需取用。
7. **热启动**：``load_from_legacy`` 从既有 ``band_head_l/m/h`` 的 1x1 投影
   权重做 SVD 截断，映射到新头的 ``u/v``；新增参数 key 前缀 ``bandlr.``
   （挂在 ``FullGatedTokenHead.bandlr`` 下时），Stage1 ckpt ``strict=False``
   可直接加载。

边界：本模块只做头内数学与自检接口；不改主干、不改 ``uvit3d.py`` /
``x0_fourier_band_loss.py`` / ``train_staged.py``。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 包内导入 / 平铺导入两种部署方式都兼容
    from .x0_fourier_band_loss import _band_masks
except ImportError:  # pragma: no cover - 平铺 sys.path 部署
    from x0_fourier_band_loss import _band_masks




class MultiBandLowRankHead(nn.Module):
    """共享投影 + 频域三分频 + 低秩子头的一体化频带头。

    Args:
        in_dim: 主干 token 特征通道数（旧三头输入通道，即 FullGatedTokenHead.Cmax）。
        out_dim: 输出通道数（= patch_t*patch_h*patch_w*out_channels，旧头 out_dim）。
        rank: 每个频带子头的低秩秩，默认 8（契约 §2）。
        shared_rank: 共享投影的低秩秩，默认 8。
        band_low: 低/中边界（cycles/pixel），默认 0.05，与 x0_fourier_band_loss 一致。
        band_mid_high: 中/高边界，默认 0.25。
        high_mask_enabled: 高频输出是否在频谱上再乘 high 掩膜。
        freeze_masks: 掩膜 buffer 是否冻结（恒为 True；掩膜本身无可训参数）。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        shared_rank: int = 8,
        band_low: float = 0.05,
        band_mid_high: float = 0.25,
        high_mask_enabled: bool = True,
        freeze_masks: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.shared_rank = int(shared_rank)
        self.band_low = float(band_low)
        self.band_mid_high = float(band_mid_high)
        self.high_mask_enabled = bool(high_mask_enabled)
        self.freeze_masks = bool(freeze_masks)

        # 1) 一次共享低秩投影（三个频带复用）。
        # 注意：低秩乘积不能 u/v 同时零初始化，否则 v.weight 的梯度=u(x)=0、
        # u.weight 的梯度=v^T grad=0，整个头只有 bias 能训。这里只把
        # 输出侧 v 零初始化（保证 step-0 输出严格为 0），输入侧随机小值。
        self.shared_down = nn.Conv2d(self.in_dim, self.shared_rank, 1, bias=False)
        self.shared_up = nn.Conv2d(self.shared_rank, self.in_dim, 1, bias=True)
        nn.init.normal_(self.shared_down.weight, std=0.02)
        nn.init.normal_(self.shared_up.weight, std=0.02)
        nn.init.zeros_(self.shared_up.bias)

        # 低频软直接门：low = base_low + g*(head_low - base_low)。
        # g=sigmoid(logit)，初始 3.35e-4，step-0 只替换约 0.0335% 低频能量；
        # 关闭时严格等价；开启时也不是“一刀切”，但初梯度按 g 同比变小。
        # 原来 -3.0（4.7%）会明显改低频，A0 前先按 -8.0 收紧。
        self.low_gate_logit = nn.Parameter(torch.tensor(-8.0))

        # 2) 三个低秩子头：l=低频直接预测；m/h=中高频残差。
        self.u_l = nn.Conv2d(self.in_dim, self.rank, 1, bias=False)
        self.v_l = nn.Conv2d(self.rank, self.out_dim, 1, bias=True)
        self.u_m = nn.Conv2d(self.in_dim, self.rank, 1, bias=False)
        self.v_m = nn.Conv2d(self.rank, self.out_dim, 1, bias=True)
        self.u_h = nn.Conv2d(self.in_dim, self.rank, 1, bias=False)
        self.v_h = nn.Conv2d(self.rank, self.out_dim, 1, bias=True)
        for w in (self.u_l.weight, self.u_m.weight, self.u_h.weight):
            nn.init.normal_(w, std=0.02)
        for w in (self.v_l.weight, self.v_m.weight, self.v_h.weight):
            nn.init.zeros_(w)
        for b in (self.v_l.bias, self.v_m.bias, self.v_h.bias):
            nn.init.zeros_(b)

        # 掩膜按 (h, w) 缓存；float32、requires_grad=False。
        self._mask_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # 审计用最后一次前向缓存（detach）。
        self._last_low: Optional[torch.Tensor] = None
        self._last_mid: Optional[torch.Tensor] = None
        self._last_high: Optional[torch.Tensor] = None
        self._last_low_feat: Optional[torch.Tensor] = None
        self._last_structure_reg: Optional[torch.Tensor] = None
        self._last_band_rms: Optional[Tuple[float, float, float]] = None

    # ------------------------------------------------------------------ masks
    def _get_masks(self, h: int, w: int, device: torch.device,
                   dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h, w)
        cached = self._mask_cache.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        masks = _band_masks(
            h, w,
            band_low=self.band_low,
            band_mid_high=self.band_mid_high,
            device=device,
            dtype=torch.float32,
            half_plane_multiplicity=False,
        )
        masks = tuple(m.to(dtype) for m in masks)
        if self.freeze_masks:
            for m in masks:
                m.requires_grad_(False)
        self._mask_cache[key] = masks
        return masks

    # ------------------------------------------------------------- frequency
    @torch.no_grad()
    def split_bands(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 4D [N,C,H,W] 特征在频谱上拆成 low/mid/high 三个空间分量。

        返回 float32、detach；仅用于监控/对齐，需要反传的路径走 ``forward``。
        """
        if x.ndim != 4:
            raise ValueError(f"split_bands expects 4D [N,C,H,W], got {tuple(x.shape)}")
        n, c, h, w = x.shape
        f = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
        masks = self._get_masks(h, w, x.device, torch.float32)
        bands = tuple(
            torch.fft.irfft2(f * m, s=(h, w), dim=(-2, -1), norm="ortho")
            for m in masks
        )
        return bands[0], bands[1], bands[2]

    @staticmethod
    def structure_reg_loss(low_feat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """从低频分量提 Sobel 梯度的结构正则标量（无独立分支、无可训参数）。

        直接对低频特征的通道均值图求 x/y 方向 Sobel，返回梯度幅值均值。
        训练端若启用，自行乘权重；本模块不接优化器回路。
        """
        if low_feat.ndim != 4:
            raise ValueError(f"low_feat must be 4D [N,C,H,W], got {tuple(low_feat.shape)}")
        gray = low_feat.float().mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=low_feat.device, dtype=torch.float32).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(F.pad(gray, [1, 1, 1, 1], mode="replicate"), kx)
        gy = F.conv2d(F.pad(gray, [1, 1, 1, 1], mode="replicate"), ky)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + eps).mean()

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向。

        Args:
            x: [N, in_dim, H, W] token 特征（N = B*T）。

        Returns:
            dict:
              ``low``      : 低频直接预测 [N,out_dim,H,W]（非残差）。
              ``mid``      : 中频残差 [N,out_dim,H,W]。
              ``high``     : 高频残差 [N,out_dim,H,W]（频谱掩膜后）。
              ``low_feat`` : 低频共享特征分量（结构正则源）。
              ``structure_reg``: 结构正则标量。
              ``band_rms``: 三分量 RMS 三元组（监控用）。
        """
        if x.ndim != 4:
            raise ValueError(f"x must be 4D [N,C,H,W], got {tuple(x.shape)}")
        n, c, h, w = x.shape
        if c != self.in_dim:
            raise ValueError(f"x C={c} != in_dim={self.in_dim}")

        # 1) 共享投影（只做一次）。
        shared = self.shared_up(self.shared_down(x))

        # 2) 频谱分频。FFT 在 fp32 上做（bf16 FFT 在部分 ROCm build 上不稳）。
        f = torch.fft.rfft2(shared.float(), dim=(-2, -1), norm="ortho")
        masks = self._get_masks(h, w, x.device, torch.float32)
        f_l, f_m, f_h = f * masks[0], f * masks[1], f * masks[2]
        b_l = torch.fft.irfft2(f_l, s=(h, w), dim=(-2, -1),
                               norm="ortho").to(dtype=x.dtype)
        b_m = torch.fft.irfft2(f_m, s=(h, w), dim=(-2, -1),
                               norm="ortho").to(dtype=x.dtype)
        b_h = torch.fft.irfft2(f_h, s=(h, w), dim=(-2, -1),
                               norm="ortho").to(dtype=x.dtype)

        # 3) 每带低秩子头。
        o_l = self.v_l(self.u_l(b_l)).to(dtype=x.dtype)
        o_m = self.v_m(self.u_m(b_m)).to(dtype=x.dtype)
        o_h_raw = self.v_h(self.u_h(b_h))

        # 4) 高频在原始频谱上再乘冻结 high 掩膜（不做 latent 空间硬限制）。
        if self.high_mask_enabled:
            fh = torch.fft.rfft2(o_h_raw.float(), dim=(-2, -1), norm="ortho")
            o_h = torch.fft.irfft2(fh * masks[2], s=(h, w), dim=(-2, -1),
                                   norm="ortho").to(dtype=x.dtype)
        else:
            o_h = o_h_raw.to(dtype=x.dtype)

        # 5) 结构正则：直接从低频分量提梯度，无独立分支。
        sreg = self.structure_reg_loss(b_l)

        with torch.no_grad():
            self._last_low = o_l.detach()
            self._last_mid = o_m.detach()
            self._last_high = o_h.detach()
            self._last_low_feat = b_l.detach()
            self._last_structure_reg = sreg.detach()
            self._last_band_rms = tuple(
                float(t.detach().float().pow(2).mean().sqrt()) for t in (o_l, o_m, o_h)
            )

        return {
            "low": o_l,
            "mid": o_m,
            "high": o_h,
            "low_feat": b_l,
            "low_gate": torch.sigmoid(self.low_gate_logit),
            "structure_reg": sreg,
            "band_rms": self._last_band_rms,
        }

    # ------------------------------------------------------------ hot start
    @torch.no_grad()
    def load_from_legacy(
        self,
        legacy_l: nn.Module,
        legacy_m: nn.Module,
        legacy_h: nn.Module,
    ) -> Dict[str, object]:
        """从旧 ``band_head_l/m/h`` 热启动。

        旧头结构（见 ``full_gated_head._make_band_head``）::

            Sequential(DW Conv3x3 groups=Cmax, SiLU, Proj Conv1x1 Cmax->out_dim)

        映射方式：取旧 1x1 投影权重 ``W [out,in]``，CPU fp32 SVD 截断到
        ``rank``：``v.weight = U_r * Σ_r``，``u.weight = Vh_r``。旧 depthwise
        是恒等初始化（中心 1、其余 0），无可复用的空间混频权重，跳过并在
        返回报告里记为 skipped。共享投影无旧权重来源，保留默认零初始化。

        Returns:
            dict: ``matched``（已映射参数量）、``unmatched``（新头未被旧权重
            覆盖的参数量）、``skipped_modules``（跳过的旧子模块名）、
            ``svd_rank``、``message``。
        """
        report: Dict[str, object] = {
            "matched": 0,
            "unmatched": 0,
            "skipped_modules": [],
            "svd_rank": self.rank,
            "message": "",
        }
        band_map = (
            ("l", legacy_l, self.u_l, self.v_l),
            ("m", legacy_m, self.u_m, self.v_m),
            ("h", legacy_h, self.u_h, self.v_h),
        )
        for tag, legacy, u_sub, v_sub in band_map:
            proj = legacy[-1]  # Conv1x1 Cmax->out_dim
            if not isinstance(proj, nn.Conv2d):
                report["skipped_modules"].append(f"band_{tag}:last-not-conv")
                continue
            w = proj.weight.detach().float().cpu()  # [out, in, 1, 1]
            b = (proj.bias.detach().float().cpu()
                 if proj.bias is not None else None)
            wo = w.view(w.shape[0], w.shape[1])
            # DW depthwise 恒等初始化，无可复用权重，显式跳过。
            report["skipped_modules"].append(f"band_{tag}.0(dw3x3)")

            try:
                U, S, Vh = torch.linalg.svd(wo, full_matrices=False)
            except Exception as exc:  # noqa: BLE001 - 热启动失败不能炸训练
                report["message"] = f"band_{tag} SVD failed: {exc!r}; kept zero-init"
                continue
            r = min(self.rank, U.shape[1], Vh.shape[0])
            vw = (U[:, :r] * S[:r]).to(v_sub.weight.device)
            uw = Vh[:r, :].to(u_sub.weight.device)
            v_sub.weight.copy_(vw.view_as(v_sub.weight))
            u_sub.weight.copy_(uw.view_as(u_sub.weight))
            if b is not None:
                v_sub.bias.copy_(b.to(v_sub.bias.device))
            report["matched"] += v_sub.weight.numel() + u_sub.weight.numel()
            if b is not None:
                report["matched"] += v_sub.bias.numel()

        total = sum(p.numel() for p in self.parameters())
        report["unmatched"] = total - int(report["matched"])
        report["message"] = (
            f"matched={report['matched']}/{total} params; "
            f"shared_down/shared_up 无旧权重来源，保留零初始化；"
            f"skipped={report['skipped_modules']}"
        )
        return report

    # ------------------------------------------------------------------ audit
    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
