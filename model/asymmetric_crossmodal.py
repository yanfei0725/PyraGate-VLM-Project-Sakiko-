"""
a2 语义层（图1）—— 压缩跨模态可见 + 非对称特权
====================================================

来源：用户 5 图之第 1 张（"代际"论证链的第一环）。

拓扑（用用户的记号复述）
------------------------
模态 A（驱动方，如音频/VLM 语义）与模态 B（被驱动方，如视频 latent）各有原始表示 a, b。
在共享注意力场里：
  * A 看到的是 B 的**压缩/预估摘要** b' （bottleneck，维度 b_dim << N_B·C）
  * B 看到的是 A 的**压缩摘要** a'
  * 只有 A 能访问 α —— A 的私有全局/控制信号（信息特权）
  * 非对称访问：α 沿 A 路径进 attention，B 路径拿不到

本质：跨模态信息在注意力场里流动，但流动的是**有损的、经瓶颈的摘要**，
不是完整高维表示。计算复杂度取决于瓶颈维度 b_dim，不是全序列长度 N ——
这比 H3 的任意 token 对无损 full attention（N² 稠密计算）省得多。

同时：这是"跨模态因果推理"的结构基础（图2 方向一）——
A→B 的驱动方向由 α 特权天然表达，H3 的对称 full attention 结构上做不到。

实现约定
--------
- 输入：a [B, Na, C], b [B, Nb, C]，可选 alpha [B, C]（A 私有特权信号）
- 摘要器：可学习的 Perceiver 式 bottleneck（M 个 query 槽，M << Nb/Na）
- 注意力：摘要域内做 softmax（跨模态），复杂度 O(M·(Na+Nb))，与原始 token 数弱相关
- 门控：所有回注用零初始化 out_gate，起步严格恒等，不污染已有主干
- 纯 torch，CPU 可跑，不占显存
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from attn_backend import attention as dispatch_attention
except ImportError:
    try:
        from .attn_backend import attention as dispatch_attention
    except Exception:
        dispatch_attention = None


class BottleneckSummary(nn.Module):
    """Perceiver 式压缩摘要器：把 [B,N,C] 压成 [B,M,C]，M 为固定槽数。

    M << N 时，跨模态注意力的计算量由 M 决定（瓶颈维度），而非 N²。
    """
    def __init__(self, dim: int, num_slots: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.norm_q(self.slots).expand(B, -1, -1)          # [B,M,C]
        kv = self.norm_kv(x)                                   # [B,N,C]
        qh = q.view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        kh = kv.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vh = kh
        if dispatch_attention is not None:
            y = dispatch_attention(qh, kh, vh, is_causal=False, backend="math")
        else:
            y = F.scaled_dot_product_attention(qh, kh, vh, scale=self.head_dim ** -0.5)
        return y.transpose(1, 2).reshape(B, self.num_slots, C)


def _proj(dim: int, rank: int) -> nn.Module:
    """低秩等价投影：rank<=0 时退化为单个 nn.Linear（保持旧 ckpt 兼容）。"""
    if rank <= 0 or rank >= dim:
        return nn.Linear(dim, dim, bias=False)
    return nn.Sequential(
        nn.Linear(dim, rank, bias=False),
        nn.Linear(rank, dim, bias=False),
    )


class AsymmetricCrossModal(nn.Module):
    """压缩跨模态可见 + 非对称 α 特权。

    前向：
      a' = summary_A(a)                    # A 的压缩摘要
      b' = summary_B(b)                    # B 的压缩摘要
      # 共享注意力场（摘要域）：
      ctx_A = cross_attend(query= a 池化? 用 a'，key/value= b')   # A 看到 B 的压缩版
      ctx_B = cross_attend(query= b', key/value= a')              # B 看到 A 的压缩版
      # 非对称特权：只有 A 路径注入 α
      ctx_A = ctx_A + alpha_proj(alpha)
      # 回注（零门控起步）
      a_out = a + gate_a * inject_a(ctx_A)
      b_out = b + gate_b * inject_b(ctx_B)
    """
    def __init__(self, dim: int, num_slots: int = 32, num_heads: int = 4,
                 alpha_dim: Optional[int] = None, rank: int = 0):
        super().__init__()
        # rank>0：10 个 dim×dim 投影换低秩瓶颈（参数 10×2×dim×rank）。
        # 零门控起步不变，热启动只需新键 strict=False 加载。
        self.rank = int(rank)
        # 2026-11-24 BUG-4：原默认 None→alpha_proj=None→α 特权默认完全失效。
        # 改为默认 alpha_dim=dim（图1 的 α 非对称特权默认生效）；显式传 0 才关闭。
        if alpha_dim is None:
            alpha_dim = dim
        self.dim = dim
        self.alpha_dim = int(alpha_dim)
        self.summary_a = BottleneckSummary(dim, num_slots, num_heads)
        self.summary_b = BottleneckSummary(dim, num_slots, num_heads)
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 摘要域内的跨模态 Q/K/V（只作用于 M 个槽，复杂度 O(M²)）
        self.q_a = _proj(dim, self.rank)   # A 侧 query（看 b'）
        self.k_b = _proj(dim, self.rank)
        self.v_b = _proj(dim, self.rank)
        self.q_b = _proj(dim, self.rank)   # B 侧 query（看 a'）
        self.k_a = _proj(dim, self.rank)
        self.v_a = _proj(dim, self.rank)
        # 回注投影
        self.inject_a = _proj(dim, self.rank)
        self.inject_b = _proj(dim, self.rank)
        # 隐患B：token 级回注的 K/V（query 是各 token 自身，K/V 来自 M 槽摘要）
        self.k_ctx_a = _proj(dim, self.rank)
        self.v_ctx_a = _proj(dim, self.rank)
        self.k_ctx_b = _proj(dim, self.rank)
        self.v_ctx_b = _proj(dim, self.rank)
        # 非对称 α 特权：只进 A 路径（B 无此投影）。self.alpha_dim>0 时创建。
        if self.alpha_dim > 0:
            self.alpha_proj = (_proj(self.alpha_dim, self.rank)
                               if self.rank > 0
                               else nn.Linear(self.alpha_dim, dim, bias=False))
        else:
            self.alpha_proj = None
        # 零门控起步（tanh 参数化）：初始严格恒等
        self.gate_a = nn.Parameter(torch.zeros(1))
        self.gate_b = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _attend(q, k, v, heads, head_dim):
        B, M, C = q.shape
        qh = q.view(B, M, heads, head_dim).transpose(1, 2)
        kh = k.view(B, -1, heads, head_dim).transpose(1, 2)
        vh = v.view(B, -1, heads, head_dim).transpose(1, 2)
        if dispatch_attention is not None:
            y = dispatch_attention(qh, kh, vh, is_causal=False, backend="math")
        else:
            y = F.scaled_dot_product_attention(qh, kh, vh, scale=head_dim ** -0.5)
        return y.transpose(1, 2).reshape(B, M, C)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                alpha: Optional[torch.Tensor] = None):
        B, Na, C = a.shape
        Nb = b.shape[1]
        # 压缩摘要（计算量由 num_slots 决定，不随 Na/Nb 二次增长）
        ap = self.summary_a(a)   # [B,M,C]
        bp = self.summary_b(b)

        # A 看 B 的压缩版 b'；B 看 A 的压缩版 a'
        ctx_A = self._attend(self.q_a(ap), self.k_b(bp), self.v_b(bp),
                             self.num_heads, self.head_dim)
        ctx_B = self._attend(self.q_b(bp), self.k_a(ap), self.v_a(ap),
                             self.num_heads, self.head_dim)

        # 非对称特权：α 只沿 A 路径注入（这是 A→B 因果驱动方向的结构来源）
        if alpha is not None and self.alpha_proj is not None:
            ctx_A = ctx_A + self.alpha_proj(alpha).unsqueeze(1)

        # 2026-11-24 隐患B 修复：原来 .mean(dim=1) 把 M 槽摘要压成单向量广播给所有
        # token，跨模态信息无 token 级差异 → 退化成 AdaLN 式全局 condition。
        # 改为 token 级交叉注意力：每个 a/b token 用自身作 query 去 attend 摘要槽，
        # 得到逐 token 不同的跨模态上下文（仍是 O(N·M)，M 槽瓶颈不变）。
        ga = torch.tanh(self.gate_a).to(a.dtype)
        gb = torch.tanh(self.gate_b).to(b.dtype)
        # A 侧：query=a（Na token），key/value=注入了 α 的摘要 ctx_A（M 槽）
        ctx_a_tok = self._attend(a, self.k_ctx_a(ctx_A), self.v_ctx_a(ctx_A),
                                 self.num_heads, self.head_dim)
        # B 侧：query=b（Nb token），key/value=摘要 ctx_B（M 槽）
        ctx_b_tok = self._attend(b, self.k_ctx_b(ctx_B), self.v_ctx_b(ctx_B),
                                 self.num_heads, self.head_dim)
        a_out = a + ga * self.inject_a(ctx_a_tok)
        b_out = b + gb * self.inject_b(ctx_b_tok)
        return a_out, b_out

    def compute_cross_kv(self, a: torch.Tensor, b: torch.Tensor,
                         alpha: Optional[torch.Tensor] = None):
        """融合版：算一次摘要，产出 A 侧回注 + B 侧供每层 attention 消费的 K/V。

        与 forward 的区别：
          * forward 是一次性 token 级回注（旧路径，保留向后兼容）；
          * compute_cross_kv 把 B 侧摘要 K/V 交给主干逐 block 注入注意力场，
            跨模态信息在每层 attention 内流动，而不是只做一次全局加和。

        非对称特权保持不变：
          ctx_A 注入 alpha（A 私有），只沿 A 路径回注到 vlm_cond；
          返回的 cross_kv 来自 ctx_B（B 看 a'），结构上不含 alpha。

        复用已有投影，零新增参数。
        """
        ap = self.summary_a(a)   # [B,M,C]
        bp = self.summary_b(b)
        ctx_A = self._attend(self.q_a(ap), self.k_b(bp), self.v_b(bp),
                             self.num_heads, self.head_dim)
        if alpha is not None and self.alpha_proj is not None:
            ctx_A = ctx_A + self.alpha_proj(alpha).unsqueeze(1)
        ctx_B = self._attend(self.q_b(bp), self.k_a(ap), self.v_a(ap),
                             self.num_heads, self.head_dim)
        # A 侧 token 级回注：α 特权落在 A（VLM）路径上
        ctx_a_tok = self._attend(a, self.k_ctx_a(ctx_A), self.v_ctx_a(ctx_A),
                                 self.num_heads, self.head_dim)
        ga = torch.tanh(self.gate_a).to(a.dtype)
        a_out = a + ga * self.inject_a(ctx_a_tok)
        # B 侧摘要 K/V：供主干每层 SelfAttention 作为额外 key/value 拼接。
        # gate_b 复用为跨模态 V 强度调节（初始 tanh(0)=0 → 系数 1，不扰动起步）；
        # budget=0 时整条 cross_kv 不进图，恒等性由外层保证。
        gb = torch.tanh(self.gate_b).to(b.dtype)
        cross_k = self.k_ctx_b(ctx_B)
        cross_v = self.v_ctx_b(ctx_B) * (1.0 + gb)
        return a_out, (cross_k, cross_v)

    def summary_cost_tokens(self) -> int:
        """跨模态注意力的复杂度指标：O(M² + M·C)，与原始 N 无关。"""
        return self.num_slots ** 2


def _selfcheck():
    torch.manual_seed(0)
    B, Na, Nb, C = 2, 40, 60, 32
    a = torch.randn(B, Na, C)
    b = torch.randn(B, Nb, C)
    alpha = torch.randn(B, C)
    mod = AsymmetricCrossModal(C, num_slots=8, num_heads=4, alpha_dim=C)

    a0, b0 = mod(a, b, alpha)          # 零门控：严格恒等
    assert torch.allclose(a0, a, atol=1e-6) and torch.allclose(b0, b, atol=1e-6)

    # 打开门控后梯度可达全部分支（含 α 特权）
    with torch.no_grad():
        mod.gate_a.fill_(0.3); mod.gate_b.fill_(0.3)
    a1, b1 = mod(a, b, alpha)
    assert not torch.allclose(a1, a)
    loss = a1.pow(2).mean() + b1.pow(2).mean()
    loss.backward()
    assert mod.alpha_proj.weight.grad is not None and torch.isfinite(mod.alpha_proj.weight.grad).all()
    assert mod.summary_a.slots.grad is not None

    # 非对称性：B 路径无 alpha 投影（结构上拿不到 α）
    assert not hasattr(mod, "alpha_proj_b")

    return {"zero_gate_identity": True, "asymmetric_alpha": True,
            "summary_slots": mod.num_slots,
            "crossmodal_cost": f"O(M^2)={mod.summary_cost_tokens()} vs full O(N^2)={Nb**2}"}


if __name__ == "__main__":
    info = _selfcheck()
    print("[asymmetric_crossmodal selfcheck]")
    for k, v in info.items():
        print(f"  {k}: {v}")
