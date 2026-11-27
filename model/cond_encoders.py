"""
轻量化条件编码器（轻量 B 档）

两个模块，沿用 VLM 文本注入链的「减层 + 低秩瓶颈」方案：

1. HierarchicalStyleEncoder（分层风格条件编码器，6 层 r=384 低秩，≈72M）
   - 输入：风格 token [B, L, dim]（来自 VLM 风格头 / 北大桥输出）
   - 输出：风格条件 [B, dim]（加法注入 AdaLN cond）+ 路由 logits [B, num_experts]
   - 作用：约束 MoE 画风分区、减少串味；接口与「风格 token 输入 / 路由输出」兼容

2. ReferenceResampleAlignEncoder（参考帧特征重采样对齐编码器，6 层 r=384 cross-attn，≈68M）
   - 保留 cross-attn 重采样核心层；只砍冗余特征融合层（10→6）
   - 输入：当前 token [B, N, dim] + 参考 token [B, Nr, dim]（由 ref_latent patchify 得）
   - 输出：对齐偏移 [B, dim]（零门控加法注入 cond，ref=None 时恒等不进图）
   - 作用：参考帧与生成 latent 的特征对齐，角色保形（抗变脸/漂移）

设计红线（对齐 VLM 注入链 / 全局 MoE 纪律）：
- 纯 torch SDPA + 低秩 Linear，**不写自定义 corr kernel**（避开 RDNA3 烂 kernel / 碎片化）
- 输出零门控起步（gate=0 → 严格恒等，逐 bit 回退）
- low_rank>0 且 <dim 时走 dim→r→dim 瓶颈，否则全秩；n_layers 可调（标准/轻量 A/轻量 B）

作者：PyraGate-VLM / Project Sakiko
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from uvit3d import RMSNorm
except ImportError:
    # 独立运行（未被 UViT3D 导入）时的兜底：与 uvit3d.RMSNorm 同语义
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
            self.dim = dim

        def forward(self, x):
            return F.rms_norm(x, (self.dim,), self.weight, eps=self.eps)


def _lowrank_linear(in_dim: int, out_dim: int, rank: int) -> nn.Module:
    """rank>0 且 <min(in,out) 时 dim→r→dim 瓶颈，否则全秩线性。"""
    if rank > 0 and rank < min(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, rank, bias=False),
            nn.Linear(rank, out_dim, bias=False),
        )
    return nn.Linear(in_dim, out_dim, bias=False)


class _LowRankAttnLayer(nn.Module):
    """单层低秩 Transformer（pre-norm + 残差），self / cross 两用。

    - self 模式：q/k/v 均来自主输入 x
    - cross 模式：q 来自主输入 x，k/v 来自 ctx（参考 token）
    每层参数 = 8·dim·rank（qkv,o 低秩） + 2·dim·ffn_hidden（FFN） + 4·dim（norm）
    """
    def __init__(self, dim: int, num_heads: int, rank: int, ffn_hidden: int,
                 eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = int(rank)

        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)
        self.q = _lowrank_linear(dim, dim, self.rank)
        self.k = _lowrank_linear(dim, dim, self.rank)
        self.v = _lowrank_linear(dim, dim, self.rank)
        self.o = _lowrank_linear(dim, dim, self.rank)
        # q/k 归一化（作用在低秩 up 投影后的 dim 维，保持多头结构）
        self.q_norm = RMSNorm(dim, eps)
        self.k_norm = RMSNorm(dim, eps)
        self.ffn_up = nn.Linear(dim, ffn_hidden, bias=False)
        self.ffn_down = nn.Linear(ffn_hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv = x if ctx is None else ctx
        B, N, C = x.shape
        L = kv.shape[1]
        q = self.q_norm(self.q(self.norm1(x))).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k(self.norm1(kv))).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(self.norm1(kv)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)          # [B,nh,N,hd] 纯 torch，无自定义 kernel
        y = y.transpose(1, 2).reshape(B, N, C)
        x = x + self.o(y)
        h = self.norm2(x)
        x = x + self.ffn_down(F.silu(self.ffn_up(h)))
        return x


class HierarchicalStyleEncoder(nn.Module):
    """分层风格条件编码器（轻量 B：6 层 r=384 低秩，≈72M）。

    做「风格 token → 风格条件 + MoE 路由 logits」的轻量映射：
    - 多层 self-attn 沿风格 token 交互 → 池化成 [B,dim] 风格条件
    - 分层缩放（每层贡献可学习权重）→ 保留「分层」语义
    - 路由头输出 [B,num_experts] 风格 logits，直接喂给全局 MoE 的 style_logits 接口
    - 零门控起步：out_gate=0 → 风格条件恒等 0，不改 AdaLN/路由注入逻辑
    """
    def __init__(self, dim: int = 3072, num_heads: int = 24, num_experts: int = 3,
                 n_layers: int = 6, rank: int = 384, ffn_hidden: int = 160,
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.layers = nn.ModuleList([
            _LowRankAttnLayer(dim, num_heads, rank, ffn_hidden, eps)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(dim, eps)
        # 分层可学习权重（每层一个标量，softmax 归一）
        self.layer_scale = nn.Parameter(torch.ones(self.n_layers))
        self.cond_proj = nn.Linear(dim, dim, bias=False)      # 风格条件投影
        self.route_head = nn.Linear(dim, num_experts, bias=False)  # → MoE 路由 logits
        self.out_gate = nn.Parameter(torch.zeros(1))          # 零起步：严格恒等

    def forward(self, style_tokens: torch.Tensor) -> tuple:
        """style_tokens: [B, L, dim] → (style_cond [B,dim], route_logits [B,E])。"""
        h = style_tokens
        feats = []
        for layer in self.layers:
            h = layer(h)
            feats.append(h.mean(dim=1))                        # 分层池化 → [B,dim]
        stacked = torch.stack(feats, dim=1)                    # [B,Ln,dim]
        w = torch.softmax(self.layer_scale, dim=0).to(stacked.dtype)
        fused = torch.einsum("l,blc->bc", w, stacked)         # 分层加权融合
        fused = self.norm(fused)
        style_cond = torch.tanh(self.out_gate) * self.cond_proj(fused)  # gate=0 → 恒等 0
        route_logits = self.route_head(fused)                  # 路由 logits 始终有效（供 MoE）
        return style_cond, route_logits

    def disable(self):
        with torch.no_grad():
            self.out_gate.zero_()


class ReferenceResampleAlignEncoder(nn.Module):
    """参考帧特征重采样对齐编码器（轻量 B：6 层 r=384 cross-attn，≈68M）。

    - 保留 cross-attn 重采样核心：可学习 resample query 从参考 token 里采 K/V
    - 只砍融合深度（原版 10 层 → 6 层），不砍重采样能力
    - 输入：当前 token [B,N,dim] + 参考 token [B,Nr,dim]
    - 输出：对齐偏移 [B,dim]（零门控加法，ref=None 时调用方旁路 → 恒等）
    """
    def __init__(self, dim: int = 3072, num_heads: int = 24, n_layers: int = 6,
                 rank: int = 384, ffn_hidden: int = 64, n_resample: int = 16,
                 eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.n_resample = int(n_resample)
        # 可学习重采样 query（从参考 token 池中抽取对齐特征）
        self.resample_query = nn.Parameter(torch.randn(1, self.n_resample, dim) * 0.02)
        self.resample_layers = nn.ModuleList([
            _LowRankAttnLayer(dim, num_heads, rank, ffn_hidden, eps)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(dim, eps)
        self.align_proj = nn.Linear(dim, dim, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(1))          # 零起步：严格恒等
        # W3 修复：原 forward 里 cur_tokens 只用于取 batch size，当前 latent 完全
        # 没参与"对齐"。补一条零初始化投影，起步恒等（不扰动热启动）。
        self.cur_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.cur_proj.weight)

    def forward(self, cur_tokens: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        """cur_tokens: [B,N,dim]; ref_tokens: [B,Nr,dim] → align_cond [B,dim]。"""
        B = cur_tokens.shape[0]
        q = self.resample_query.expand(B, -1, -1)             # [B, n_resample, dim]
        h = q
        # W3：当前 latent 的全局摘要逐层注入重采样 query，让"对齐"真正同时看
        # 当前帧与参考帧。cur_proj 零初始化 → 起步严格恒等。
        cur_add = (self.cur_proj(cur_tokens.mean(dim=1)).unsqueeze(1)
                   if cur_tokens is not None else None)
        for layer in self.resample_layers:
            h = layer(h, ctx=ref_tokens)                      # cross-attn: q←query, k/v←ref
            if cur_add is not None:
                h = h + cur_add
        fused = self.norm(h.mean(dim=1))                      # [B,dim]
        return torch.tanh(self.out_gate) * self.align_proj(fused)  # gate=0 → 恒等 0

    def disable(self):
        with torch.no_grad():
            self.out_gate.zero_()


if __name__ == "__main__":
    # meta device 精确统计参数量（不分配数据、不碰显卡）
    for name, ctor in [
        ("风格编码器 6层 r384", lambda: HierarchicalStyleEncoder()),
        ("参考对齐 6层 r384", lambda: ReferenceResampleAlignEncoder()),
    ]:
        with torch.device("meta"):
            m = ctor()
        n = sum(p.numel() for p in m.parameters())
        print(f"{name}: {n:,} ({n / 1e6:.2f}M)")
