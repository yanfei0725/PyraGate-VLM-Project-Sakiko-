"""
a2 分支 —— 混合线性注意力核心（纯 torch，ROCm / RDNA3 友好）
================================================================

设计目标（对应 a2 规划方案 Track A）
--------------------------------------
1. **复用现有 qkv 投影**：线性注意力只改「聚合方式」，不改 Q/K/V 的 GEMM 排布。
   因此 mode="softmax"→"linear"/"hybrid" 切换时，主干 qkv/o 权重形状完全不变。
2. **三模式**：
   - "softmax"  : 原 FA2 / SDPA 全注意力（基线，逐值等价旧主干）
   - "linear"   : 特征映射线性注意力，O(N) 长程，最省算力
   - "hybrid"   : 分块局部 softmax（关键 token 高保真）+ 全局线性（长程粗粒度），
                  逐 head 可学习门控融合。这才是 ReHyAt / SANA-Video 2.0 的核心。
3. **数值稳定**：bf16 输入在 fp32 下累加再 cast 回去（避免长序列累乘溢出 / 下溢）。
4. **零外部依赖**：不依赖 fla / GLA / CUDA Triton —— 那些生态以 CUDA 为主，
   RDNA3 上 kernel 未验证。本模块在任何 torch 设备（含 CPU）可跑，便于先验证正确性。

数学
----
线性注意力（非因果）：
    φ(·) 为特征映射（默认 elu+1，保证非负）
    S   = φ(K)ᵀ V                      [B,H,D,Dv]
    z   = Σ_n φ(K)_n                   [B,H,D]
    out = φ(Q) S / (φ(Q) · z + eps)    [B,H,N,Dv]

分块局部 softmax：把 N 切成长度 L 的块，只在块内做 O(N·L) softmax，
保留相邻 token 的高保真交互（用户 5 图里「相邻帧局部依赖」的注意力层面实现）。

混合：
    out = y_local_softmax + gate ⊙ y_global_linear
    gate 逐 head、零初始化（tanh 参数化）→ 起步严格等于纯局部 softmax，
    不会一上来就被未训练的线性分支污染。

约定：所有函数输入输出形状为 [B, H, N, D]。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 复用主干的 FA2/SDPA 分发（softmax 基线分支）；单独跑本文件时降级到本地实现。
    try:
        from attn_backend import attention as dispatch_attention
    except ImportError:
        from .attn_backend import attention as dispatch_attention
    _HAS_DISPATCH = True
except Exception:
    _HAS_DISPATCH = False


# ---------------------------------------------------------------------------
# 特征映射
# ---------------------------------------------------------------------------
def feature_map(x: torch.Tensor, kind: str = "elu") -> torch.Tensor:
    """注意力特征映射 φ(·)，在 fp32 下调用以保证线性注意力累加稳定。

    kind:
      "elu"   —— elu(x)+1（Performer / linear transformer 标准，非负，默认）
      "relu"  —— relu(x)
      "relu2" —— relu(x)²（DPR，稀疏锐化）
    """
    if kind == "elu":
        return F.elu(x) + 1.0
    if kind == "relu":
        return F.relu(x)
    if kind == "relu2":
        return F.relu(x).square()
    raise ValueError(f"未知特征映射: {kind}")


# ---------------------------------------------------------------------------
# 核心 1：全局线性注意力（非因果 / 因果）
# ---------------------------------------------------------------------------
def linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    feature_map_kind: str = "elu",
    causal: bool = False,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """特征映射线性注意力。q/k/v: [B,H,N,D] -> [B,H,N,Dv]。

    非因果走矩阵结合律 O(N·D²)；因果用前缀和（N 很小时才调用，如时序头 T=9/13）。
    """
    out_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    phi_q = feature_map(qf, feature_map_kind)   # [B,H,N,D]
    phi_k = feature_map(kf, feature_map_kind)

    if not causal:
        # S = φ(K)ᵀ V   [B,H,D,Dv]
        s = torch.matmul(phi_k.transpose(-1, -2), vf)
        z = phi_k.sum(dim=-2, keepdim=True)      # [B,H,1,D]
        out = torch.matmul(phi_q, s)             # [B,H,N,Dv]
        if normalize:
            denom = torch.matmul(phi_q, z.transpose(-1, -2)) + eps   # [B,H,N,1]
            out = out / denom
    else:
        # 因果前缀和（chunked，架构优化#6）：
        # 原实现物化 [B,H,N,D,Dv]（N=1000,D=128 → 单张量>5GB），改为逐块累积
        # S_block=Σφ(k)vᵀ [B,H,D,Dv]，只保存块边界的 S（块数×D×Dv），内存 O((N/L)·D·Dv)。
        chunk = 128
        Nq = phi_q.shape[2]
        nb = (Nq + chunk - 1) // chunk
        s_cum = phi_q.new_zeros(phi_q.shape[0], phi_q.shape[1], phi_q.shape[-1], vf.shape[-1])
        z_cum = phi_q.new_zeros(phi_q.shape[0], phi_q.shape[1], phi_q.shape[-1])
        outs = []
        for b in range(nb):
            lo, hi = b * chunk, min((b + 1) * chunk, Nq)
            phi_kb = phi_k[:, :, lo:hi]
            vb = vf[:, :, lo:hi]
            # 块内累积（小张量 [B,H,L,D,Dv]，L=128 受控）
            kv_b = phi_kb.unsqueeze(-1) * vb.unsqueeze(-2)
            s_b = kv_b.cumsum(dim=2)[:, :, -1]           # [B,H,D,Dv] 块末状态
            z_b = phi_kb.cumsum(dim=-2)[:, :, -1]        # [B,H,D]
            s_cum = s_cum + s_b                          # 到本块末的全局前缀
            z_cum = z_cum + z_b
            # 块内每个 token 的前缀 = 之前块累积 + 块内累积
            s_within = kv_b.cumsum(dim=2)                 # [B,H,L,D,Dv]
            s_all = s_cum.unsqueeze(2) - s_b.unsqueeze(2) + s_within
            qb = phi_q[:, :, lo:hi]
            ob = torch.einsum("bhld,bhlde->bhle", qb, s_all)
            if normalize:
                z_within = phi_kb.cumsum(dim=-2)          # [B,H,L,D]
                z_all = z_cum.unsqueeze(2) - z_b.unsqueeze(2) + z_within
                denom = (qb * z_all).sum(dim=-1, keepdim=True) + eps
                ob = ob / denom
            outs.append(ob)
        out = torch.cat(outs, dim=2)                      # [B,H,N,Dv]

    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# 核心 2：分块局部 softmax（关键 token 高保真）
# ---------------------------------------------------------------------------
def chunked_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    chunk_size: int = 132,
    causal: bool = False,
    scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    merge: Optional[bool] = None,
    mask_budget_mb: float = 32.0,
) -> torch.Tensor:
    """把 N 切成块，只在块内做 softmax。复杂度 O(N·L)，L=chunk_size。

    这是「相邻 token 高保真交互」的注意力层实现：跨块的长程依赖交给线性分支，
    块内细节保留 softmax 表达力。默认 L=132 ≈ 一块 15×22 空间 token 的局部邻域量级。

    2026-11-24 修复：
      - BUG-6：Dv≠D 时输出用 v 的最后一维（原来 empty_like(q) 会崩）。
      - BUG-9：attn_bias（local_bias 的 2D 位置先验）透传到块内 softmax，
        使 hybrid 与 local_bias 可以合一，不再二选一旁路。
      - 架构优化#2：merge=True 时用「一次 SDPA + block-diagonal mask」替代
        Python 循环里的多次 kernel launch；mask 是 N×N，默认超过
        mask_budget_mb(32MB) 自动回退分块循环（避免 4B 主干 N=990/24头 时显存爆炸）。
    """
    B, H, N, D = q.shape
    Dv = v.shape[-1]
    # 2026-11-24 tensor-index-review：chunk_size<=0 会让 range() 空转，
    # out=empty_like 未初始化直接返回（静默垃圾输出，maxabs 可达 1e37）。显式拒绝。
    if chunk_size < 1:
        raise ValueError(f"chunk_size 必须 >= 1，收到 {chunk_size}")
    if scale is None:
        scale = D ** -0.5
    out = q.new_empty(B, H, N, Dv)

    # merge 决策：mask 内存约 B*H*N*N 字节（bool），超预算自动回退分块循环。
    if merge is None:
        merge = (B * H * N * N) <= int(mask_budget_mb * 1024 * 1024)

    if merge:
        # 一次 SDPA + block-diagonal mask：块内可见、跨块屏蔽。
        pos = torch.arange(N, device=q.device)
        block_id = torch.div(pos, chunk_size, rounding_mode="floor")
        same_block = block_id[:, None] == block_id[None, :]        # [N,N] bool
        if attn_bias is None:
            mask = same_block.view(1, 1, N, N)
        else:
            b = attn_bias
            while b.dim() < 4:
                b = b.unsqueeze(0)
            if b.shape[0] == 1:
                b = b.expand(B, -1, -1, -1)
            if b.shape[1] == 1:
                b = b.expand(-1, H, -1, -1)
            # 块内用加性 bias，跨块 -inf；bool→float 屏蔽统一用 -inf 语义
            mask = torch.where(same_block.view(1, 1, N, N), b,
                               torch.full_like(b, float("-inf")))
            if b.dtype == torch.bool:
                mask = same_block.view(1, 1, N, N) & b
        if causal:
            cm = (pos[:, None] >= pos[None, :]).view(1, 1, N, N)
            if mask.dtype == torch.bool:
                mask = mask & cm
            else:
                mask = mask.masked_fill(~cm.expand_as(mask), float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    # 分块循环（大 N 默认路径；每块独立 kernel，显存 O(B·H·L·N)）。
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        qc = q[:, :, start:end]
        kc = k[:, :, start:end]
        vc = v[:, :, start:end]
        bias_chunk = None
        if attn_bias is not None:
            b = attn_bias
            if b.dim() == 2:
                bias_chunk = b[start:end, start:end]
            elif b.dim() == 3:
                bias_chunk = b[:, start:end, start:end]
            else:
                bias_chunk = b[:, :, start:end, start:end]
        oc = F.scaled_dot_product_attention(
            qc, kc, vc,
            attn_mask=bias_chunk,
            is_causal=(bool(causal) if bias_chunk is None else False),
            scale=scale)
        out[:, :, start:end] = oc
    return out


# ---------------------------------------------------------------------------
# 核心 3：混合注意力（局部 softmax + 全局线性，逐 head 门控）
# ---------------------------------------------------------------------------
def hybrid_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int = 132,
    feature_map_kind: str = "elu",
    causal: bool = False,
    use_softmax: bool = True,
    scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    merge: Optional[bool] = None,
) -> torch.Tensor:
    """混合注意力：y = y_local_softmax + gate ⊙ y_global_linear。

    gate: 可广播到 [B,H,N,1] 的逐 head 门控（通常来自 nn.Parameter，tanh 参数化）。
    gate=0 → 严格等于局部 softmax；gate 增大时逐步引入全局线性长程。
    use_softmax=False → 跳过局部 softmax，纯线性（gate 此时无意义）。
    attn_bias: 可选加性 bias [N,N]/[B,N,N]（local_bias 的 2D 先验），只在块内 softmax 生效。
    """
    y_lin = linear_attention(q, k, v, feature_map_kind, causal=causal)
    if not use_softmax:
        return y_lin
    y_loc = chunked_softmax_attention(q, k, v, chunk_size=chunk_size,
                                      causal=causal, scale=scale,
                                      attn_bias=attn_bias, merge=merge)
    if gate.dim() == 0 or gate.numel() == 1:
        g = gate.view(1, 1, 1, 1)
    elif gate.dim() == 2:
        g = gate.view(1, gate.shape[0], 1, 1)   # [H,?] -> [1,H,1,1]
    else:
        g = gate.view(1, gate.shape[0], 1, 1) if gate.shape[0] == q.shape[1] else gate
    return y_loc + g.to(y_lin.dtype) * y_lin


# ---------------------------------------------------------------------------
# nn.Module 包装：可直接塞进 UViTBlock / SelfAttention
# ---------------------------------------------------------------------------
class HybridLinearAttention(nn.Module):
    """混合线性注意力模块（持有逐 head 门控参数）。

    用法：在 SelfAttention 里根据 attn_mode 决定是否实例化；
    本模块只负责「拿到 q/k/v 之后怎么聚合」，不碰 qkv/o 投影。
    """

    def __init__(
        self,
        num_heads: int,
        chunk_size: int = 132,
        feature_map_kind: str = "elu",
        gate_init: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.chunk_size = int(chunk_size)
        self.feature_map_kind = feature_map_kind
        self.causal = causal
        # 逐 head 门控，零初始化（tanh 参数化在 forward 里做，初始严格 0）。
        # BUG-11 修复：tanh 封顶 ±1 限制线性分支贡献。改为可学习软上限 gate_span
        # （初始 1.0，可训到 >1），门控 = gate_span * tanh(logit)，起步仍严格 0。
        self.gate_logit = nn.Parameter(torch.full((num_heads,), float(gate_init)))
        self.gate_span = nn.Parameter(torch.ones(1))

    def gate_value(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.gate_span * torch.tanh(self.gate_logit)).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                use_softmax: bool = True,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        g = self.gate_value(q.dtype)
        return hybrid_attention(
            q, k, v, gate=g,
            chunk_size=self.chunk_size,
            feature_map_kind=self.feature_map_kind,
            causal=self.causal,
            use_softmax=use_softmax,
            attn_bias=attn_bias,
        )


# ---------------------------------------------------------------------------
# 快速自检（纯 CPU，不占显存）：数值健康 + 梯度存在 + 三模式形状一致
# ---------------------------------------------------------------------------
def _selfcheck(device: str = "cpu"):
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 33, 16
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)
    v = torch.randn(B, H, N, D, device=device)

    y_lin = linear_attention(q, k, v)
    y_loc = chunked_softmax_attention(q, k, v, chunk_size=11)
    mod = HybridLinearAttention(H, chunk_size=11).to(device)
    y_hyb = mod(q, k, v, use_softmax=True)

    assert y_lin.shape == y_loc.shape == y_hyb.shape == (B, H, N, D)
    assert torch.isfinite(y_lin).all() and torch.isfinite(y_loc).all() and torch.isfinite(y_hyb).all()
    # 零门控起步：hybrid 应严格等于局部 softmax
    assert torch.allclose(y_hyb, y_loc, atol=1e-5), "零门控 hybrid 未严格等于局部 softmax"
    # 梯度可达线性分支
    loss = y_hyb.float().pow(2).mean()
    loss.backward()
    assert mod.gate_logit.grad is not None and torch.isfinite(mod.gate_logit.grad).all()

    # 因果路径（小 N）
    y_c = linear_attention(q[:, :, :9], k[:, :, :9], v[:, :, :9], causal=True)
    assert y_c.shape == (B, H, 9, D) and torch.isfinite(y_c).all()

    # bf16 累加稳定
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    yb = linear_attention(qb, kb, vb)
    assert torch.isfinite(yb).all() and yb.dtype == torch.bfloat16

    return {
        "linear_vs_local_cos": float(F.cosine_similarity(
            y_lin.flatten(0, 1).float(), y_loc.flatten(0, 1).float(), dim=-1).mean()),
        "zero_gate_identity": True,
        "grad_ok": True,
        "bf16_stable": True,
        "shapes": list(y_lin.shape),
    }


if __name__ == "__main__":
    info = _selfcheck("cpu")
    print("[linear_attn selfcheck]")
    for k, v in info.items():
        print(f"  {k}: {v}")
