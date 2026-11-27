# -*- coding: utf-8 -*-
"""Text cross-attention for the Sakiko/PyraGate bottleneck (v3, 2026-11-27).

Design contract
---------------
* Only the 6 bottleneck blocks get this module.  The encoder/decoder and the
  output head stay untouched, so existing FreqFormer / full_head / band heads
  keep working.
* Text tokens are the official Wan2.2 UMT5 sequence after the official
  ``condition_embedder.text_embedder`` projection: [B, L<=128, 3072].  The
  pooled 4096-d vector remains the AdaLN condition, unchanged.
* Q/K/V/O can be copied 1:1 from official ``blocks.N.attn2``.  Official
  WanAttention normalises the *full* inner dim (3072) before reshaping heads;
  this module follows that exactly.
* ``rank>0`` turns Q/O into a structured low-rank linear (BandLR-style
  down/up factorisation).  K/V stay full and frozen by default, because they
  carry the official text->video prior.
* Gate is sigma-adaptive: ``gate = base_gate * sigma`` (clamped 0..0.2).  At
  sigma=0 the module is exactly identity; at sigma=1 it injects the full
  prior scaled by base_gate (0.1 by default).
* If ``text_tokens`` is None the whole module is a no-op.  This is essential
  for old checkpoints / inference paths that only provide pooled text.

This file is intentionally dependency-light: it does not import uvit3d.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma_to_batch(sigma, x: torch.Tensor) -> torch.Tensor:
    """Normalise sigma to float32 [B,1], accepting None / float / scalar / [B] / [B,1].

    The canonical dtype is float32, independent of the activation dtype.  The
    single-entry contract matters here: bf16 rounds 0.9 to 0.8984375 and makes
    entry/adapter sigma gates disagree.  Call sites that multiply sigma into a
    bf16 activation must cast the *resulting gate* at the point of use.
    """
    B = int(x.shape[0])
    if sigma is None:
        return torch.ones(B, 1, device=x.device, dtype=torch.float32)
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(float(sigma), device=x.device, dtype=torch.float32)
    else:
        sigma = sigma.to(device=x.device, dtype=torch.float32)
    if sigma.ndim == 0:
        sigma = sigma.reshape(1, 1)
    elif sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    elif sigma.ndim > 2:
        sigma = sigma.reshape(-1, 1)
    if sigma.shape[0] == 1 and B != 1:
        sigma = sigma.expand(B, 1)
    if sigma.shape[0] != B:
        raise ValueError(f"sigma batch {sigma.shape[0]} != x batch {B}")
    return sigma


# ---------------------------------------------------------------------------
# σ 门控唯一计算入口（架构/门控修复 2026-11-27）
# ---------------------------------------------------------------------------
# 归一化唯一走上面的 sigma_to_batch（P2 入口）；本函数是全模型固定
# 噪声级斜坡门控 g(σ) 的唯一真相来源。所有 head 的固定 σ 斜坡门控都调用它，
# 不再各自重写 ramp。
#   - sigma=None -> 全 1（恒等支路，不进图）
#   - 线性 clamp 斜坡 [0,1]，不做平方/立方等高次多项式捷径（旧 adapter 的
#     g=g·g 二次方已在此拆除）
#   - 模块侧不 detach（与现状梯度口径一致）；损失侧权重仍由训练脚本用 detached σ
def sigma_gate(sigma: Optional[torch.Tensor], ref: Optional[torch.Tensor] = None,
               lo: float = 0.0, hi: float = 1.0,
               batch: Optional[int] = None,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
    """把噪声级 σ 归一化为 [B,1] ∈ [0,1] 的线性斜坡门控 g(σ)。

    ref: 提供 device/dtype/batch 的参考激活张量；为 None 时用 batch/dtype/device
    兜底（adapter 等无激活张量的场景）。lo/hi 为斜坡区间，区间内线性升到 1，
    区间外截断到 [0,1]。
    """
    if torch.is_tensor(ref):
        _ref = ref
    else:
        _batch = int(batch or 1)
        _ref = torch.empty(_batch, 1, device=device,
                           dtype=dtype if dtype is not None else torch.float32)
    s = sigma_to_batch(sigma, _ref)                        # 唯一归一化入口
    g = ((s - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0).to(torch.float32)
    if torch.is_tensor(ref):
        g = g.to(dtype=ref.dtype, device=ref.device)
    elif dtype is not None:
        g = g.to(dtype=dtype, device=device)
    return g


# ---------------------------------------------------------------------------
# Low-rank linear (structured BandLR analogue for token-space projections)
# ---------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    """y = up(down(x)) + bias, with an optional dense->low-rank SVD copy.

    It is *not* a conv-band head; it is the token-space analogue used by the
    cross-attention Q/O projections.  Both factors are trainable; bias is kept.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for LowRankLinear")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        # Match nn.Linear default init closely enough; the copy path overwrites.
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.up.weight, a=5 ** 0.5)
        # Two kaiming factors multiply to 1/(9*in) variance, while a dense
        # kaiming layer is 1/(3*in).  sqrt(3) restores the step-0 activation
        # scale so a full-backbone low-rank swap does not silently shrink the
        # residual stream.  copy_from_dense overwrites this anyway.
        with torch.no_grad():
            self.up.weight.mul_(3.0 ** 0.5)
        if bias:
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    @torch.no_grad()
    def copy_from_dense(self, weight: torch.Tensor,
                        bias: Optional[torch.Tensor] = None) -> None:
        """SVD-truncate a dense [out,in] weight into down/up factors."""
        w = weight.detach().float().cpu()
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"dense weight shape {tuple(w.shape)} != "
                f"{(self.out_features, self.in_features)}")
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        r = min(self.rank, u.shape[1], vh.shape[0])
        up = (u[:, :r] * s[:r]).to(self.up.weight.device, self.up.weight.dtype)
        down = vh[:r, :].to(self.down.weight.device, self.down.weight.dtype)
        self.up.weight.copy_(up)
        self.down.weight.copy_(down)
        if bias is not None and self.up.bias is not None:
            self.up.bias.copy_(bias.to(self.up.bias.device, self.up.bias.dtype))


class _QOLinear(nn.Module):
    """Q/O holder: dense nn.Linear or structured LowRankLinear."""

    def __init__(self, dim: int, rank: int = 0, bias: bool = True):
        super().__init__()
        self.rank = int(rank)
        if self.rank > 0:
            self.proj: nn.Module = LowRankLinear(dim, dim, self.rank, bias=bias)
        else:
            self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @torch.no_grad()
    def copy_from_dense(self, weight: torch.Tensor,
                        bias: Optional[torch.Tensor] = None) -> None:
        if isinstance(self.proj, LowRankLinear):
            self.proj.copy_from_dense(weight, bias)
        else:
            if weight.shape != self.proj.weight.shape:
                raise ValueError(
                    f"dense weight {tuple(weight.shape)} != "
                    f"{tuple(self.proj.weight.shape)}")
            self.proj.weight.copy_(weight.to(self.proj.weight.device,
                                             self.proj.weight.dtype))
            if bias is not None and self.proj.bias is not None:
                self.proj.bias.copy_(bias.to(self.proj.bias.device,
                                             self.proj.bias.dtype))

    def weight_for_svd(self) -> torch.Tensor:
        if isinstance(self.proj, LowRankLinear):
            return (self.proj.up.weight @ self.proj.down.weight).detach()
        return self.proj.weight.detach()


# ---------------------------------------------------------------------------
# Official Wan text projection (4096 UMT5 -> 3072 transformer width)
# ---------------------------------------------------------------------------
class WanTextProjection(nn.Module):
    """Exact reimplementation of Wan ``condition_embedder.text_embedder``.

    PixArtAlphaTextProjection(4096, 3072, act_fn='gelu_tanh'):
        linear_1 -> GELU(tanh) -> linear_2
    """

    def __init__(self, text_dim: int = 4096, dim: int = 3072):
        super().__init__()
        self.linear_1 = nn.Linear(text_dim, dim, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(x)))

    @torch.no_grad()
    def load_from_wan(self, tensor_getter) -> None:
        """Load official ``condition_embedder.text_embedder.*`` weights."""
        for name in ("linear_1", "linear_2"):
            layer = getattr(self, name)
            layer.weight.copy_(
                tensor_getter(f"condition_embedder.text_embedder.{name}.weight")
                .to(layer.weight.device, layer.weight.dtype))
            layer.bias.copy_(
                tensor_getter(f"condition_embedder.text_embedder.{name}.bias")
                .to(layer.bias.device, layer.bias.dtype))


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class TextCrossAttention(nn.Module):
    """Bottleneck text cross-attention with sigma-adaptive gate."""

    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        text_dim: int = 3072,
        rank: int = 0,
        base_gate: float = 0.1,
        text_scale_alpha: float = 0.0,
        eps: float = 1e-5,
        freeze_kv: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.text_dim = int(text_dim)
        self.rank = int(rank)
        self.freeze_kv = bool(freeze_kv)
        self.text_scale_alpha = float(text_scale_alpha)

        self.q = _QOLinear(dim, rank=self.rank, bias=True)
        self.k = nn.Linear(text_dim, dim, bias=True)
        self.v = nn.Linear(text_dim, dim, bias=True)
        self.o = _QOLinear(dim, rank=self.rank, bias=True)
        # Official Wan uses RMSNorm over the full inner dim, before head split.
        self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)

        self.base_gate = nn.Parameter(torch.tensor(float(base_gate)))
        if self.freeze_kv:
            self.k.weight.requires_grad_(False)
            self.k.bias.requires_grad_(False)
            self.v.weight.requires_grad_(False)
            self.v.bias.requires_grad_(False)

    # ------------------------------------------------------------------
    def set_freeze_kv(self, freeze: bool) -> None:
        self.freeze_kv = bool(freeze)
        for p in (self.k.weight, self.k.bias, self.v.weight, self.v.bias):
            p.requires_grad_(not freeze)

    @torch.no_grad()
    def load_from_wan_attn2(self, tensor_getter, src_prefix: str) -> Dict[str, str]:
        """Copy official ``blocks.N.attn2`` into this module 1:1.

        ``tensor_getter(key) -> Tensor`` must resolve keys from the official
        safetensors shards.  Returns a small audit dict.
        """
        report: Dict[str, str] = {"src": src_prefix}
        q_w = tensor_getter(f"{src_prefix}.to_q.weight")
        q_b = tensor_getter(f"{src_prefix}.to_q.bias")
        k_w = tensor_getter(f"{src_prefix}.to_k.weight")
        k_b = tensor_getter(f"{src_prefix}.to_k.bias")
        v_w = tensor_getter(f"{src_prefix}.to_v.weight")
        v_b = tensor_getter(f"{src_prefix}.to_v.bias")
        o_w = tensor_getter(f"{src_prefix}.to_out.0.weight")
        o_b = tensor_getter(f"{src_prefix}.to_out.0.bias")
        nq = tensor_getter(f"{src_prefix}.norm_q.weight")
        nk = tensor_getter(f"{src_prefix}.norm_k.weight")

        # q/o can be low-rank; copy_from_dense does SVD truncation if needed.
        self.q.copy_from_dense(q_w, q_b)
        self.o.copy_from_dense(o_w, o_b)
        # K/V must stay full-rank and exactly official.
        if self.k.weight.shape != k_w.shape or self.v.weight.shape != v_w.shape:
            raise ValueError(
                f"official K/V shapes {tuple(k_w.shape)}/{tuple(v_w.shape)} "
                f"do not match module {tuple(self.k.weight.shape)}/"
                f"{tuple(self.v.weight.shape)}; text_dim must be 3072")
        self.k.weight.copy_(k_w.to(self.k.weight.device, self.k.weight.dtype))
        self.k.bias.copy_(k_b.to(self.k.bias.device, self.k.bias.dtype))
        self.v.weight.copy_(v_w.to(self.v.weight.device, self.v.weight.dtype))
        self.v.bias.copy_(v_b.to(self.v.bias.device, self.v.bias.dtype))
        self.norm_q.weight.copy_(nq.to(self.norm_q.weight.device,
                                      self.norm_q.weight.dtype))
        self.norm_k.weight.copy_(nk.to(self.norm_k.weight.device,
                                      self.norm_k.weight.dtype))
        if self.freeze_kv:
            self.set_freeze_kv(True)
        report.update({
            "q": "lowrank_svd" if self.q.rank > 0 else "dense",
            "o": "lowrank_svd" if self.o.rank > 0 else "dense",
            "kv": "frozen" if self.freeze_kv else "trainable",
        })
        return report

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        text_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x [B,N,C], text_tokens [B,L,text_dim], text_mask [B,L] bool/int.

        ``text_kv`` optionally supplies pre-projected (K,V) [B,L,C] to skip
        the per-step K/V projections.  K/V are frozen in the default recipe.
        """
        if text_tokens is None and text_kv is None:
            return x
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"x C={C} != dim={self.dim}")

        q = self.norm_q(self.q(x))
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 噪声感知文本缩放：高 σ 时把文本 token 幅度放大，补偿
        # xt≈噪声时文本条件相对信号过弱。只在 per-token 路径生效；
        # 预投影 text_kv 路径没有原始 token，不做缩放。
        if (self.text_scale_alpha > 0 and text_tokens is not None
                and sigma is not None):
            _sig = sigma_to_batch(sigma, x).reshape(B, 1, 1).to(dtype=x.dtype)
            text_tokens = text_tokens * (1.0 + self.text_scale_alpha * _sig)

        if text_kv is not None:
            k_flat, v_flat = text_kv
        else:
            if text_tokens is None:
                return x
            if text_tokens.shape[-1] != self.text_dim:
                raise ValueError(
                    f"text token dim {text_tokens.shape[-1]} != {self.text_dim}")
            k_flat = self.k(text_tokens)
            v_flat = self.v(text_tokens)
        # Official norm_k is over the full inner dim [B,L,C] before head split.
        k_flat = self.norm_k(k_flat)
        k = k_flat.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if text_mask is not None:
            m = text_mask.to(device=x.device, dtype=torch.bool)
            if m.ndim == 1:
                m = m.unsqueeze(0)
            attn_mask = m[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).reshape(B, N, C)

        # 自适应门控：低 σ 关、高 σ 开。唯一 σ 斜坡入口 sigma_gate；输出限幅 [0,0.2]。
        g = sigma_gate(sigma, ref=x, lo=0.0, hi=1.0).reshape(B, 1, 1)
        gate = (self.base_gate.to(x.dtype) * g).clamp(0.0, 0.2)
        if text_valid is not None:
            # 全链路统一 null 语义：CFG 的 null 样本即使为了 SDPA 用全 True
            # mask + zero token，也必须在这里把整个文本残差精确关掉。
            gate = gate * text_valid.to(device=x.device, dtype=x.dtype).reshape(B, 1, 1)
        return x + gate * self.o(y)


# ---------------------------------------------------------------------------
# Lightweight AdaLN-Zero global text injection (parallel, never replaces norm1)
# ---------------------------------------------------------------------------
class AdaLNZeroText(nn.Module):
    """Text-pooled AdaLN-Zero residual injection.

    ``x = x + gate(sigma) * (norm(x) * scale(text) + shift(text))``.
    ``text_proj`` is zero-init, so step-0 is exactly identity.  This is added
    *after* the block's normal residual path; it never replaces norm1/norm2.
    """

    def __init__(self, dim: int = 3072, text_dim: int = 3072,
                 gate_hidden: int = 256, rank: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.rank = int(rank)
        if self.rank > 0:
            # Low-rank text->scale/shift: 3072->r->6144.  At rank=128 this is
            # ~1.19M/layer instead of 18.9M/layer, so all 30 blocks fit.
            self.text_down = nn.Linear(text_dim, self.rank, bias=False)
            self.text_up = nn.Linear(self.rank, dim * 2, bias=True)
            nn.init.zeros_(self.text_up.weight)
            nn.init.zeros_(self.text_up.bias)
        else:
            self.text_proj = nn.Linear(text_dim, dim * 2, bias=True)
            nn.init.zeros_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, max(8, int(gate_hidden))),
            nn.SiLU(),
            nn.Linear(max(8, int(gate_hidden)), 1),
        )
        nn.init.zeros_(self.sigma_mlp[-1].weight)
        nn.init.constant_(self.sigma_mlp[-1].bias, -3.0)

    def _project(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            return self.text_up(F.silu(self.text_down(pooled)))
        return self.text_proj(pooled)

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                sigma: Optional[torch.Tensor] = None,
                tier_scale: Optional[torch.Tensor] = None,
                text_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if text_tokens is None:
            return x
        B = x.shape[0]
        tt = text_tokens.to(device=x.device, dtype=x.dtype)
        if text_mask is None:
            pooled = tt.mean(dim=1)
        else:
            m = text_mask.to(device=x.device, dtype=x.dtype)
            if m.ndim == 1:
                m = m.unsqueeze(0)
            w = m.unsqueeze(-1)
            pooled = (tt * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        scale, shift = self._project(pooled).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        if sigma is None:
            sig = x.new_ones(B, 1)
        else:
            sig = sigma_to_batch(sigma, x).reshape(B, 1)
        gate = torch.sigmoid(self.sigma_mlp(sig.to(self.sigma_mlp[0].weight.dtype)))
        gate = gate.to(x.dtype).unsqueeze(-1)  # [B,1,1]
        if tier_scale is not None:
            gate = gate * tier_scale.to(device=x.device, dtype=x.dtype).reshape(B, 1, 1)
        if text_valid is not None:
            gate = gate * text_valid.to(device=x.device, dtype=x.dtype).reshape(B, 1, 1)
        return x + gate * (self.norm(x) * scale + shift)


# ---------------------------------------------------------------------------
# Sigma sliding adapter (shared backbone, no expert) and 4-tier context tokens
# ---------------------------------------------------------------------------
class SigmaSlidingAdapter(nn.Module):
    """Shared σ-gated residual adapter.  No second expert, no hard routing.

    ``g(σ)=clamp((σ-lo)/(hi-lo),0,1)``（线性斜坡，唯一入口 sigma_gate），
    ``h=h+g(σ)*up(silu(down(norm(h))))``.
    ``up`` is zero-init, so step-0 is exact identity.  Low-σ samples get g=0
    and cannot update this branch, which is what removes the 000/002200
    gradient fight without duplicating the 3.3B backbone.
    """

    def __init__(self, dim: int = 3072, rank: int = 64,
                 lo: float = 0.78, hi: float = 0.98):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.lo = float(lo)
        self.hi = float(hi)
        self.norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True)
        self.down = nn.Linear(dim, rank, bias=True)
        self.up = nn.Linear(rank, dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def gate(self, sigma: Optional[torch.Tensor], dtype: torch.dtype,
             device: torch.device, batch: int = 1) -> torch.Tensor:
        # 唯一 σ 斜坡入口 sigma_gate；旧 g=g·g 二次方高次捷径已拆除。
        return sigma_gate(sigma, ref=None, lo=self.lo, hi=self.hi,
                          batch=batch, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None,
                extra_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # extra_gate 为历史残留的二次门控旁路，已拆除（A-04）；形参保留仅兼容
        # 旧调用签名，不参与任何计算。
        B = x.shape[0]
        g = self.gate(sigma, x.dtype, x.device, batch=B)
        if g.shape[0] == 1 and B != 1:
            g = g.expand(B, 1)
        y = self.up(F.silu(self.down(self.norm(x).to(self.down.weight.dtype)))
                    .to(x.dtype))
        return x + g.unsqueeze(1) * y


class _LowRankTokenCross(nn.Module):
    """Cross-attend two token sets through a rank bottleneck.

    q from ``query``, k/v from ``context``; attention runs in rank space.
    Output projection is zero-init, so the whole module starts as identity.
    """

    def __init__(self, dim: int = 3072, rank: int = 128, heads: int = 8):
        super().__init__()
        assert rank % heads == 0
        self.q_in = nn.Linear(dim, rank, bias=False)
        self.k_in = nn.Linear(dim, rank, bias=False)
        self.v_in = nn.Linear(dim, rank, bias=False)
        self.out = nn.Linear(rank, dim, bias=True)
        self.norm_q = nn.RMSNorm(rank)
        self.norm_k = nn.RMSNorm(rank)
        self.mha = nn.MultiheadAttention(rank, heads, batch_first=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q_in(query)).to(self.mha.in_proj_weight.dtype)
        k = self.norm_k(self.k_in(context)).to(self.mha.in_proj_weight.dtype)
        v = self.v_in(context).to(self.mha.in_proj_weight.dtype)
        y, _ = self.mha(q, k, v, need_weights=False)
        return self.out(y.to(query.dtype))


class TierContextTokens(nn.Module):
    """Four learnable pyramid tokens: scene / actor / part / pixel.

    They are NOT inserted into the spatial token sequence (that would break
    ``N == T*H*W``).  They read the bottleneck state once, write global context
    back, and are then injected per tier through the existing ``tier_mod`` path.
    This is both self-conditioning and register tokens in one module.
    """

    def __init__(self, dim: int = 3072, n_tiers: int = 4,
                 rank: int = 128, heads: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.n_tiers = int(n_tiers)
        self.tokens = nn.Parameter(torch.zeros(n_tiers, dim))
        nn.init.normal_(self.tokens, std=0.02)
        self.read = _LowRankTokenCross(dim, rank, heads)
        self.write = _LowRankTokenCross(dim, rank, heads)
        self.cond_proj = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, dim, bias=True),
        )
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, z: torch.Tensor):
        B = z.shape[0]
        tok = self.tokens.unsqueeze(0).expand(B, -1, -1).to(z.dtype)
        tok = tok + self.read(tok, z)
        z = z + self.write(z, tok)
        cond = self.cond_proj(tok.mean(dim=1))
        return z, tok, cond


class CoarseX0Head(nn.Module):
    """Parameter-free coarse x0 head: 2×2 average pool only.

    It deliberately owns no weights.  Coarse supervision reuses the main x0
    prediction/target, so the budget stays 0 trainable parameters; this module
    exists so the coarse path is an explicit structure, not an ad-hoc loss.
    """

    def __init__(self, kernel=(1, 2, 2), stride=(1, 2, 2)):
        super().__init__()
        self.kernel = tuple(int(x) for x in kernel)
        self.stride = tuple(int(x) for x in stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=self.kernel, stride=self.stride)


# ---------------------------------------------------------------------------
# Bottleneck global modulation (cheap, zero-init identity)
# ---------------------------------------------------------------------------
class BottleneckGlobalMLP(nn.Module):
    """Global average-pool -> 2-layer MLP -> per-channel scale.

    ``x = x * (1 + scale(pool(x)))``; the last linear layer is zero-init, so
    step-0 is exactly identity.  Rank=128 keeps it ~0.79M params per block.
    """

    def __init__(self, dim: int = 3072, rank: int = 128):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=True)
        self.up = nn.Linear(rank, dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        scale = self.up(F.silu(self.down(self.norm(pooled)))
                        .to(dtype=self.up.weight.dtype))
        return x * (1.0 + scale.unsqueeze(1).to(dtype=x.dtype))


# ---------------------------------------------------------------------------
# Official checkpoint loader
# ---------------------------------------------------------------------------
def _find_transformer_shards(ckpt_dir: str) -> Iterable[str]:
    pat = os.path.join(ckpt_dir, "transformer", "*.safetensors")
    files = sorted(glob.glob(pat))
    if not files:
        # also accept a direct transformer dir
        files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no transformer safetensors under {ckpt_dir}")
    return files


class WanTensorGetter:
    """Lazy key->tensor resolver across official safetensors shards."""

    def __init__(self, ckpt_dir: str):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.files = list(_find_transformer_shards(ckpt_dir))
        self.key_to_file = {}
        for f in self.files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    self.key_to_file[k] = f

    def __call__(self, key: str) -> torch.Tensor:
        f = self.key_to_file.get(key)
        if f is None:
            raise KeyError(f"official key not found: {key}")
        with self._safe_open(f, framework="pt", device="cpu") as sf:
            return sf.get_tensor(key)


def init_text_cross_from_wan(model: nn.Module, ckpt_dir: str,
                             src_blocks: Tuple[int, ...] = (12, 13, 14, 15, 16, 17),
                             dst_attr: str = "bottleneck",
                             module_attr: str = "text_cross",
                             dst_indices: Optional[Tuple[int, ...]] = None
                             ) -> list[Dict[str, str]]:
    """Initialise ``text_cross`` modules from official attn2.

    ``dst_indices`` selects the actual block indices inside ``dst_attr``.
    Without it, every block with the module is initialised in order (old
    behaviour).  For mid/late-layer mounting, pass the real indices so
    ``src_blocks[k]`` maps to the correct destination block, not to the
    first N blocks.
    """
    getter = WanTensorGetter(ckpt_dir)
    dst_blocks = getattr(model, dst_attr)
    if dst_indices is None:
        dst_indices = tuple(range(len(dst_blocks)))
    reports = []
    for k, i in enumerate(dst_indices):
        blk = dst_blocks[int(i)]
        mod = getattr(blk, module_attr, None)
        if mod is None:
            raise RuntimeError(
                f"{dst_attr}[{i}] has no {module_attr}; dst_indices/src mapping "
                "must point at mounted blocks only")
        src = int(src_blocks[k % len(src_blocks)])
        reports.append(mod.load_from_wan_attn2(getter, f"blocks.{src}.attn2"))
    return reports


__all__ = [
    "LowRankLinear",
    "sigma_to_batch",
    "sigma_gate",
    "WanTextProjection",
    "TextCrossAttention",
    "AdaLNZeroText",
    "BottleneckGlobalMLP",
    "SigmaSlidingAdapter",
    "TierContextTokens",
    "CoarseX0Head",
    "WanTensorGetter",
    "init_text_cross_from_wan",
]
