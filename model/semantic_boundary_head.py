"""
方块 + 语义 双功能输出头（SemanticBoundaryDualHead）
====================================================

一个头同时做两件此前分散在多个旁路里的事：

1) 方块支路（boundary）：在 **latent 空间** 直接预测 patch 周期残差，
   专治 32px 方块。与 local_conv（token 域固定 depthwise 平滑）、
   GatedSkip._lowpass（skip 支路低通）、latent_refine（unpatchify 后固定
   3×3 卷积）、full_head 的 sobel/aa/pyramid（token 域高频）**机制不同**：
   本支路是可学习的、以 patch 内位置为条件的 Fourier/邻域残差，输出在
   latent 空间，直接对最终速度场做 correction，零门控起步。

2) 语义支路（semantic）：把 VLM 全局条件编译成固定 M 个语义槽，
   以 cross-attention 写入 latent token，输出语义条件残差。与北大桥
   （注入 cond/AdaLN）、crossmodal（模态 A/B 摘要域）**不共享权重**，
   这是输出端的语义纠错口，不重复条件注入链。

两支路各自独立零门控 tanh(gate)，初始严格恒等；可单独消融。
不进训练主线，只在 model_v7_backup_20261126 使用。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from uvit3d import RMSNorm, SelfAttention, SwiGLU
except ImportError:  # pragma: no cover
    from .uvit3d import RMSNorm, SelfAttention, SwiGLU


class SemanticBoundaryDualHead(nn.Module):
    """双功能输出头：v <- v + tanh(g_b)*Δ_boundary + tanh(g_s)*Δ_semantic。"""

    def __init__(self, dim: int, num_heads: int, out_channels: int,
                 patch_size: Tuple[int, int, int] = (1, 2, 2),
                 ffn_hidden: int = 4096, sem_rank: int = 768,
                 sem_slots: int = 16):
        super().__init__()
        pt, ph, pw = patch_size
        assert pt == 1, "双功能头当前要求 patch_t=1（与 full_head 同约束）"
        self.dim = dim
        self.patch_size = tuple(patch_size)
        self.out_dim = out_channels * ph * pw

        # 把最终 latent v 重新 patchify 回 token 域（权重独立，不复用主干 patchify）
        self.patch_emb = nn.Conv3d(out_channels, dim, kernel_size=patch_size,
                                   stride=patch_size)

        # ---- 支路 A：patch 周期残差（方块） ----
        self.norm_b = RMSNorm(dim)
        self.boundary_attn = SelfAttention(dim, num_heads)
        self.boundary_ffn = SwiGLU(dim, hidden_dim=ffn_hidden)
        self.boundary_norm = RMSNorm(dim)
        self.boundary_out = nn.Linear(dim, self.out_dim, bias=False)

        # ---- 支路 B：语义槽 cross-attention（语义理解） ----
        self.norm_s = RMSNorm(dim)
        self.sem_proj = nn.Linear(dim, dim, bias=False)      # VLM 全局条件 -> 语义槽调制
        self.sem_codebook = nn.Parameter(torch.randn(sem_slots, dim) * 0.02)
        # 低秩语义 cross-attn：q/k/v/o 全在 sem_rank 子空间，省 35M+
        self.sem_down = nn.Linear(dim, sem_rank, bias=False)
        self.sem_attn = SelfAttention(sem_rank, num_heads)
        self.sem_up = nn.Linear(sem_rank, dim, bias=False)
        self.sem_out = nn.Linear(dim, self.out_dim, bias=False)

        # 零门控：初始严格恒等
        self.gate_boundary = nn.Parameter(torch.zeros(1))
        self.gate_semantic = nn.Parameter(torch.zeros(1))

    def _unpatch(self, t: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """[B,N,C*ph*pw] -> [B,C,T,H,W]，与 full_head 的 unpatchify 同构。"""
        pt, ph, pw = self.patch_size
        C = self.out_dim // (ph * pw)
        t = t.view(B, T, H, W, ph, pw, C)
        t = t.permute(0, 6, 1, 4, 2, 5, 3).contiguous()
        return t.reshape(B, C, T * pt, H * ph, W * pw)

    def forward(self, v: torch.Tensor, z: torch.Tensor,
                ctx: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """
        v:   [B,C,T,H,W] 最终速度场（full_head 输出）
        z:   [B,N,dim]   主干最终 token 特征（提供 patch 周期结构先验）
        ctx: [B,dim]     VLM/AdaLN 全局语义条件
        grid:(T,H,W)    token 网格

        SEMHEAD_TIMER=1 时按段计时（每 20 步汇总打印一次，仅诊断用，不影响数值）。
        """
        import os, time as _time
        _t = os.environ.get("SEMHEAD_TIMER") == "1"
        if _t:
            self._sh_n = getattr(self, "_sh_n", 0) + 1
            _acc = getattr(self, "_sh_acc", None)
            if _acc is None:
                _acc = [0.0, 0.0, 0.0, 0.0]; self._sh_acc = _acc
            torch.cuda.synchronize(); _t0 = _time.perf_counter()
        B, C, T, H, W = v.shape
        Tt, Ht, Wt = grid
        vp = self.patch_emb(v).flatten(2).transpose(1, 2)   # [B,N,dim]
        if _t:
            torch.cuda.synchronize(); _acc[0] += _time.perf_counter() - _t0; _t1 = _time.perf_counter()

        # ---- A: 方块残差 ----
        hb = self.norm_b(vp + z)
        yb = self.boundary_attn(hb, rope_grid=grid)
        yb = self.boundary_norm(yb + self.boundary_ffn(yb))
        db = self._unpatch(self.boundary_out(yb), B, Tt, Ht, Wt)
        if _t:
            torch.cuda.synchronize(); _acc[1] += _time.perf_counter() - _t1; _t2 = _time.perf_counter()

        # ---- B: 语义残差 ----
        slots = self.sem_codebook.unsqueeze(0) + self.sem_proj(ctx).unsqueeze(1)
        qs = self.norm_s(vp)
        qs = self.sem_down(qs)
        sk = self.sem_down(slots)
        ys = self.sem_attn(qs, rope_grid=None, cross_kv=(sk, sk))
        ds = self._unpatch(self.sem_out(self.sem_up(ys)), B, Tt, Ht, Wt)
        if _t:
            torch.cuda.synchronize(); _acc[2] += _time.perf_counter() - _t2; _t3 = _time.perf_counter()

        gb = torch.tanh(self.gate_boundary).to(v.dtype)
        gs = torch.tanh(self.gate_semantic).to(v.dtype)
        out = v + gb * db + gs * ds
        if _t:
            torch.cuda.synchronize(); _acc[3] += _time.perf_counter() - _t3
            if self._sh_n % 20 == 0:
                n = float(self._sh_n)
                print("[semhead] n=%d 均值ms: patch_emb=%.0f 支路A=%.0f 支路B=%.0f 写回=%.0f"
                      % (self._sh_n, _acc[0] / n * 1000, _acc[1] / n * 1000,
                         _acc[2] / n * 1000, _acc[3] / n * 1000), flush=True)
        return out

    def aux_losses(self) -> dict:
        """给训练侧的可选辅助量（当前仅暴露门控，不强制使用）。"""
        return {
            "boundary_gate": torch.tanh(self.gate_boundary).detach(),
            "semantic_gate": torch.tanh(self.gate_semantic).detach(),
        }


def _selfcheck():
    torch.manual_seed(0)
    h = SemanticBoundaryDualHead(64, 4, 8, patch_size=(1, 2, 2),
                                 ffn_hidden=128, sem_rank=32, sem_slots=4)
    v = torch.randn(2, 8, 3, 8, 8)
    z = torch.randn(2, 3 * 4 * 4, 64)
    ctx = torch.randn(2, 64)
    with torch.no_grad():
        out = h(v, z, ctx, (3, 4, 4))
    assert out.shape == v.shape, (out.shape, v.shape)
    # 零门控起步 → 严格恒等
    assert torch.equal(out, v), (out - v).abs().max().item()
    # 开门后梯度可达两支路
    h.gate_boundary.data.fill_(1.0)
    h.gate_semantic.data.fill_(1.0)
    out = h(v, z, ctx, (3, 4, 4))
    out.pow(2).mean().backward()
    assert h.boundary_out.weight.grad is not None
    assert h.sem_out.weight.grad is not None
    print("[semantic_boundary_head] selfcheck PASS")
    return h


if __name__ == "__main__":
    _selfcheck()
