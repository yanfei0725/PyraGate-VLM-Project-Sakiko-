#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x0-only Fourier band loss with 5D [B,C,T,H,W] support.

Design constraints:
  * Only x0 prediction / x0 teacher are used. No velocity-field term.
  * rfft2 is applied to the last two spatial dims (H, W); T is untouched.
  * Band edges use normalized spatial frequency in cycles/pixel:
      low:  r <= 0.05
      mid:  0.05 < r <= 0.25
      high: r > 0.25
  * Energy is normalized by total per-sample energy before log1p, so the
    term does not fight the separately calibrated output amplitude.
  * A per-band complex cosine term prevents a noisy prediction from
    satisfying the loss by merely matching energy.
  * No RMS pre-alignment, no gradient equalization, no multi-loss mixing.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

# 共享频带口径（2026-11-26）：x0 频带损失/条件与 FreqRouter 的注意力频域路由
# 统一引用同一组边界，避免两套默认值日后漂移。数值与改前完全一致。
BAND_LOW: float = 0.05
BAND_MID_HIGH: float = 0.25


def _as_spectrum_entry(spec):
    """把外部传入的 SharedSpectrum / SpectrumEntry 归一为 entry；None 透传。"""
    if spec is None:
        return None
    if hasattr(spec, "x0hat"):
        return spec
    if hasattr(spec, "entry"):
        return spec.entry
    raise TypeError(
        "spectrum 必须是 fft_shared.SharedSpectrum / SpectrumEntry，收到 "
        f"{type(spec)!r}")


def _validate_inputs(x0_pred: torch.Tensor, x0_teacher: torch.Tensor) -> None:
    if x0_pred.ndim != 5 or x0_teacher.ndim != 5:
        raise ValueError(
            f"x0_pred/x0_teacher must be 5D [B,C,T,H,W], got "
            f"{tuple(x0_pred.shape)} and {tuple(x0_teacher.shape)}"
        )
    if x0_pred.shape != x0_teacher.shape:
        raise ValueError(
            f"x0_pred shape {tuple(x0_pred.shape)} != "
            f"x0_teacher shape {tuple(x0_teacher.shape)}"
        )


def _band_masks(
    h: int,
    w: int,
    *,
    band_low: float,
    band_mid_high: float,
    device: torch.device,
    dtype: torch.dtype,
    half_plane_multiplicity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build band masks aligned to rfft2 output [H, W//2+1].

    ``fftfreq``/``rfftfreq`` return cycles-per-pixel, which is the same
    normalized scale used by the existing diagnostic scripts.  Do not use
    ``linspace(0, 1, W//2+1)``: that maps Nyquist to 1.0 and silently doubles
    the frequency scale.

    rfft2 drops the negative-W half-plane.  For a real input the missing
    columns are Hermitian mirrors of the retained columns, so every non-boundary
    W column must be counted twice to reproduce the full-FFT band energy.  The
    W=0 and W=Nyquist columns occur once.

    Set ``half_plane_multiplicity=False`` for irfft2-based spatial band
    reconstruction: irfft2 already reconstructs the missing half-plane, so
    multiplying retained bins by 2 would double the band amplitude.
    """
    if not (0.0 < band_low < band_mid_high < 0.5**0.5):
        raise ValueError(
            f"invalid band edges: low={band_low}, mid_high={band_mid_high}; "
            "expected 0 < low < mid_high < sqrt(0.5)"
        )

    # [H] in [-0.5, 0.5), [W//2+1] in [0, 0.5]
    fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device, dtype=dtype)
    radial = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))

    # Reconstruct the full-FFT counting measure from the half-plane.
    wf = w // 2 + 1
    if half_plane_multiplicity:
        mult = torch.full((wf,), 2.0, device=device, dtype=dtype)
        mult[0] = 1.0
        if w % 2 == 0:
            mult[-1] = 1.0
    else:
        mult = torch.ones((wf,), device=device, dtype=dtype)

    low = (radial <= float(band_low)).to(dtype) * mult[None, :]
    mid = (
        (radial > float(band_low)) & (radial <= float(band_mid_high))
    ).to(dtype) * mult[None, :]
    high = (radial > float(band_mid_high)).to(dtype) * mult[None, :]
    return low, mid, high


def _band_ratios(
    x: torch.Tensor,
    *,
    band_low: float,
    band_mid_high: float,
) -> torch.Tensor:
    """Return per-sample band energy ratios [B, 3] (low, mid, high)."""
    b, c, t, h, w = x.shape
    f = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
    mag_sq = f.abs().pow(2)
    masks = _band_masks(
        h,
        w,
        band_low=band_low,
        band_mid_high=band_mid_high,
        device=x.device,
        dtype=mag_sq.dtype,
    )
    bands = [
        (mag_sq * mask).sum(dim=(-2, -1)).sum(dim=(1, 2))  # [B]
        for mask in masks
    ]
    # Masks carry the half-plane multiplicity, so their sum is the correct
    # full-FFT total; a raw half-plane sum would misweight the W=0 column.
    total = torch.stack(bands, dim=-1).sum(dim=-1)
    ratios = torch.stack([e / (total + 1e-8) for e in bands], dim=-1)
    return ratios


@torch.no_grad()
def x0_band_energy_ratios(
    x0: torch.Tensor,
    *,
    band_low: float = BAND_LOW,
    band_mid_high: float = BAND_MID_HIGH,
) -> torch.Tensor:
    """Public helper for Stage-3 monitoring: [B,3] ratios of low/mid/high."""
    if x0.ndim != 5:
        raise ValueError(f"x0 must be 5D [B,C,T,H,W], got {tuple(x0.shape)}")
    return _band_ratios(x0, band_low=band_low, band_mid_high=band_mid_high)


@torch.no_grad()
def split_frequency_bands(
    x: torch.Tensor,
    *,
    band_low: float = BAND_LOW,
    band_mid_high: float = BAND_MID_HIGH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split 5D [B,C,T,H,W] into low/mid/high spatial bands.

    Uses no half-plane multiplicity: irfft2 synthesizes the Hermitian half,
    so multiplying by 2 would double the reconstructed band amplitude.
    Returned tensors are float32 and detached; use for teacher targets or
    monitoring, not for a path that needs gradients through x.
    """
    if x.ndim != 5:
        raise ValueError(f"x must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
    b, c, t, h, w = x.shape
    f = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
    masks = _band_masks(
        h,
        w,
        band_low=band_low,
        band_mid_high=band_mid_high,
        device=x.device,
        dtype=torch.float32,
        half_plane_multiplicity=False,
    )
    bands = []
    for m in masks:
        bands.append(torch.fft.irfft2(f * m, s=(h, w), dim=(-2, -1), norm="ortho"))
    return bands[0], bands[1], bands[2]




def band_limit_5d(
    x: torch.Tensor,
    band: int,
    *,
    band_low: float = BAND_LOW,
    band_mid_high: float = BAND_MID_HIGH,
) -> torch.Tensor:
    """Band-limit a 5D [B,C,T,H,W] latent tensor with the exact eval masks.

    A3 v3 在 unpatchify 之后使用；频带定义与采样评估完全一致，避免
    patchify/unpatchify 改变频率映射。
    """
    if x.ndim != 5:
        raise ValueError(f"x must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
    b, c, t, h, w = x.shape
    f = torch.fft.rfft2(x.float(), dim=(-2, -1), norm="ortho")
    masks = _band_masks(
        h, w, band_low=band_low, band_mid_high=band_mid_high,
        device=x.device, dtype=torch.float32, half_plane_multiplicity=False)
    y = torch.fft.irfft2(f * masks[int(band)], s=(h, w), dim=(-2, -1), norm="ortho")
    return y.to(dtype=x.dtype)


def x0_fourier_band_loss(
    x0_pred: torch.Tensor,
    x0_teacher: torch.Tensor,
    weight: float = 0.15,
    band_low: float = BAND_LOW,
    band_mid_high: float = BAND_MID_HIGH,
    eps: float = 1e-8,
    spatial_weight: float = 1.0,
    spectrum_pred=None,
    spectrum_teach=None,
    sigma: torch.Tensor | None = None,
    sigma_weighted: bool = False,
) -> torch.Tensor:
    """x0-only Fourier band matching loss.

    Args:
        x0_pred: predicted x0, 5D [B,C,T,H,W].
        x0_teacher: teacher x0, same shape. Detached internally.
        weight: total loss weight; constrained to the accepted 0.1..0.2 range.
        band_low: low/mid boundary in normalized spatial frequency.
        band_mid_high: mid/high boundary in normalized spatial frequency.
        eps: numerical epsilon for ratios and cosine denominators.
        spatial_weight: weight for the band-limited spatial MSE term.  Energy
            ratios + global complex cosine can be satisfied by high-frequency
            noise; comparing the irfft2 band components in pixel space blocks
            that shortcut.  Set 0 to recover the old energy/cosine-only loss.
        spectrum_pred / spectrum_teach: 可选 ``fft_shared.SharedSpectrum`` 或
            ``SpectrumEntry``。传入后复用其已算好的 rfft2/掩膜/分量，不再重复
            FFT；不传时保持旧行为（内部各算一次 rfft2），签名向后兼容。
        sigma: [B]/[B,1] 流匹配噪声级；仅当 ``sigma_weighted=True`` 时使用。
        sigma_weighted: 频域解耦权重。低频 ∝ σ²，中/高频 ∝ (1-σ)²；
            只在 band-limited spatial MSE 上生效，energy/cosine 保持原口径。

    Returns:
        Scalar tensor.
    """
    # 2026-11-26：放宽到 0.02~0.3，第三优先级需要 xf=0.08 对照。
    if not (0.02 <= float(weight) <= 0.3):
        raise ValueError(f"weight must be in [0.02, 0.3], got {weight}")
    if float(spatial_weight) < 0.0:
        raise ValueError(f"spatial_weight must be >= 0, got {spatial_weight}")
    _validate_inputs(x0_pred, x0_teacher)

    # FFT on bf16 is not reliably supported on all torch/ROCm builds.
    pred = x0_pred.float()
    teach = x0_teacher.detach().float()
    b, c, t, h, w = pred.shape

    ep = _as_spectrum_entry(spectrum_pred)
    et = _as_spectrum_entry(spectrum_teach)
    if ep is not None:
        f_pred = ep.x0hat
    else:
        f_pred = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    if et is not None:
        f_teach = et.x0hat
    else:
        f_teach = torch.fft.rfft2(teach, dim=(-2, -1), norm="ortho")
    if ep is not None:
        masks = (ep.mask_low, ep.mask_mid, ep.mask_high)
    else:
        masks = _band_masks(
            h,
            w,
            band_low=band_low,
            band_mid_high=band_mid_high,
            device=pred.device,
            dtype=torch.float32,
        )

    # ---- normalized log-energy term -------------------------------------
    # Aggregate over C,T per sample so the target is the same global band
    # distribution used by the diagnostics (55.30 / 31.77 / 12.93), not a
    # noisy per-channel target.
    mag_pred_sq = f_pred.abs().pow(2)
    mag_teach_sq = f_teach.abs().pow(2)

    e_pred = torch.stack(
        [(mag_pred_sq * m).sum(dim=(-2, -1)).sum(dim=(1, 2)) for m in masks],
        dim=-1,
    )  # [B,3]
    e_teach = torch.stack(
        [(mag_teach_sq * m).sum(dim=(-2, -1)).sum(dim=(1, 2)) for m in masks],
        dim=-1,
    )  # [B,3]

    # Masks carry the rfft half-plane multiplicity; sum them for the total.
    total_pred = e_pred.sum(dim=-1)
    total_teach = e_teach.sum(dim=-1)
    ratio_pred = e_pred / (total_pred[:, None] + float(eps))
    ratio_teach = e_teach / (total_teach[:, None] + float(eps))
    energy_loss = F.mse_loss(
        torch.log1p(ratio_pred), torch.log1p(ratio_teach)
    )

    # ---- per-band complex cosine term ----------------------------------
    # Uses all bins in each band, not just scalar energy, so high-frequency
    # noise cannot satisfy the objective by matching the energy ratio.
    cos_loss = pred.new_zeros(())
    for m in masks:
        p = f_pred * m
        q = f_teach * m
        dot = (p.conj() * q).sum(dim=(-2, -1)).real
        norm_p = p.abs().pow(2).sum(dim=(-2, -1)).add(float(eps)).sqrt()
        norm_q = q.abs().pow(2).sum(dim=(-2, -1)).add(float(eps)).sqrt()
        # Clamp before 1-cos: floating-point roundoff can produce 1+1e-7
        # for identical inputs, which makes the total loss slightly negative.
        cos = torch.clamp(
            dot / (norm_p * norm_q + float(eps)), min=-1.0, max=1.0
        )
        cos_loss = cos_loss + (1.0 - cos).mean()

    # ---- band-limited spatial MSE --------------------------------------
    # Reconstruct low/mid/high components with irfft2 and compare them in
    # pixel space.  The masks here have NO half-plane multiplicity: irfft2
    # already synthesizes the Hermitian half, so multiplying by 2 would
    # double the reconstructed band amplitude.  Each band is normalized by
    # its own teacher RMS to avoid the low band's large DC dominating.
    spatial_loss = pred.new_zeros(())
    if float(spatial_weight) > 0.0:
        if ep is not None:
            p_bands = list(ep.components_spatial())
        else:
            masks_spatial = _band_masks(
                h, w, band_low=band_low, band_mid_high=band_mid_high,
                device=pred.device, dtype=torch.float32, half_plane_multiplicity=False,
            )
            p_bands = [
                torch.fft.irfft2(f_pred * m, s=(h, w), dim=(-2, -1), norm="ortho")
                for m in masks_spatial
            ]
        if et is not None:
            t_bands = list(et.components_spatial())
        else:
            masks_spatial = _band_masks(
                h, w, band_low=band_low, band_mid_high=band_mid_high,
                device=pred.device, dtype=torch.float32, half_plane_multiplicity=False,
            )
            t_bands = [
                torch.fft.irfft2(f_teach * m, s=(h, w), dim=(-2, -1), norm="ortho")
                for m in masks_spatial
            ]
        per_band = []
        for p_band, t_band in zip(p_bands, t_bands):
            scale = (
                t_band.float().pow(2).mean(dim=(1, 2, 3, 4), keepdim=True)
                .sqrt()
                .clamp_min(float(eps))
            )
            per_band.append(
                F.mse_loss(
                    p_band.float() / scale,
                    t_band.float() / scale,
                )
            )
        if sigma_weighted:
            if sigma is None:
                raise ValueError("sigma_weighted=True requires sigma")
            s = sigma.detach().float().reshape(-1, 1).clamp(0.0, 1.0)
            # low: sigma^2 (high noise -> layout); mid/high: (1-sigma)^2
            w = torch.cat([s.pow(2), (1.0 - s).pow(2), (1.0 - s).pow(2)], dim=-1)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            spatial_loss = (torch.stack(per_band, dim=0).unsqueeze(0) * w).sum(-1).mean()
        else:
            spatial_loss = torch.stack(per_band).mean()

    return float(weight) * (
        energy_loss + cos_loss + float(spatial_weight) * spatial_loss
    )


def low_tv_loss(
    x0: torch.Tensor,
    spectrum=None,
    band_low: float = BAND_LOW,
    band_mid_high: float = BAND_MID_HIGH,
) -> torch.Tensor:
    """低频分量空间 TV（相邻 H/W 差分 L1 均值）。

    传入 ``spectrum``（SharedSpectrum / SpectrumEntry）时复用其低频 IFFT 分量，
    不再重复 FFT；不传时内部走 ``split_frequency_bands`` 旧路径，向后兼容。
    """
    entry = _as_spectrum_entry(spectrum)
    if entry is not None:
        low = entry.low_ifft
    else:
        low = split_frequency_bands(
            x0, band_low=band_low, band_mid_high=band_mid_high)[0]
    # low 是 5D [B,C,T,H,W]：H 差分切倒数第二维，W 差分切最后一维。
    # 旧写法 [..., 1:, :, :] 实际切的是 T，[..., :, 1:, :] 实际切的是 H，W 差分丢失。
    tv_h = (low[..., 1:, :] - low[..., :-1, :]).abs().mean()
    tv_w = (low[..., :, 1:] - low[..., :, :-1]).abs().mean()
    return 0.5 * (tv_h + tv_w)


