"""
PyraGate-VLM v2.0 — 3D U-ViT 主干骨架（"底子"）
==================================================
对齐白皮书 v2.0 §3 的 V1 基线设计，纯 torch 实现，无 diffusers/flash-attn 依赖。

设计要点（逐条对应白皮书）：
  §3.2  条件注入 = AdaLN-Zero（时间步 + 文本 emb，移除 token 拼接）
  §3.3  skip = 可学习门控 α（初始 0.1）+ 谱约束（spectral norm），可置 0 退化
  §3.4  注意力 = 2+1D 分解（空间 Attn + 时序 Attn 交替），时序头可选因果掩码
        瓶颈层保留完整 3D 联合注意力（JointAttention）
  §3.5  FFN = SwiGLU（等参数量，隐层 ×2/3）
  §3.6  输出头 = 速度场主头 + 轻量时序辅助头（帧间相似度残差）
  §4    金字塔 = PatchMerge3D / PatchUnmerge3D（时空可分别控制降采样）

运行（纯 CPU，绝不碰显卡）：
  CUDA_VISIBLE_DEVICES="" python selftest_uvit3d.py

注意：此文件只定义"模型长什么样"，不含训练/蒸馏/量化逻辑。
"""

import math
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt      # 梯度检查点（V2-23）

try:
    from attn_backend import attention as dispatch_attention, backend_diagnose
except ImportError:                                  # 作为 package 导入时
    from .attn_backend import attention as dispatch_attention, backend_diagnose

# ---------------------------------------------------------------------------
# sakiko_op 门控算子接入（v2.0 挂载点：GatedSkip）
# ---------------------------------------------------------------------------
# 算子包在 pyragate 父目录的 sakiko_op/，从 model/ 直接跑自检时不在 sys.path。
# 这里做路径兜底 + try/except 降级：算子包缺失时主干退回原生加法，绝不因可选算子崩溃。
import os as _os
import sys as _sys
_PYRAGATE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PYRAGATE_ROOT not in _sys.path:
    _sys.path.insert(0, _PYRAGATE_ROOT)
try:
    from sakiko_op.switch import gated_residual as _gated_residual_op, \
        use_custom_op as _sakiko_use_custom_op, hits as _sakiko_hits
    _SAKIKO_AVAILABLE = True
except Exception:                                    # 算子包缺失/版本不含 switch → 原生降级
    _SAKIKO_AVAILABLE = False
    def _sakiko_use_custom_op(explicit=None):
        return False
    def _sakiko_hits(reset=False):
        return {}
    def _gated_residual_op(x, branch, gate, tag="default", use_op=None):
        # 兜底路径与 switch.py 的 native 分支语义一致（gate=0 严格返回 x）
        if isinstance(gate, torch.Tensor):
            if gate.ndim == 0 and float(gate) == 0.0:
                return x
            return x + gate * branch
        if float(gate) == 0.0:
            return x
        return x + gate * branch


# ---------------------------------------------------------------------------
# 3D RoPE（对齐教师 Wan2.2，wan22/modules/model.py:31-70）
# ---------------------------------------------------------------------------

def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """[兼容/自检保留] 复数相位表 complex128。

    生产前向已改用 rope_cos_sin()（实数 fp32）根治 B1 —— complex buffer 被
    Module.to(bf16) 转换时会静默丢弃虚部，使 RoPE 退化为静态缩放。本函数仅
    保留给 selftest §8f 的「对齐教师 Wan2.2」参考实现使用，不进生产前向。
    """
    assert dim % 2 == 0, "RoPE head_dim 必须偶数"
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)   # [max_seq_len, dim/2] complex


def rope_apply_3d(x: torch.Tensor, grid: Tuple[int, int, int], freqs: torch.Tensor) -> torch.Tensor:
    """[兼容/自检保留] 复数版 3D RoPE，供 selftest 逐值对齐教师用。生产前向不用。"""
    N, n, d = x.shape
    T, H, W = grid
    assert N == T * H * W, f"RoPE token 数 {N} 与 grid {grid} 不符"
    c = d // 2                                  # 每 head 的复数通道数
    f_t, f_h, f_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(N, n, c, 2))
    freqs_i = torch.cat([
        f_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1),
        f_h[:H].view(1, H, 1, -1).expand(T, H, W, -1),
        f_w[:W].view(1, 1, W, -1).expand(T, H, W, -1),
    ], dim=-1).reshape(N, 1, -1)
    y = torch.view_as_real(x_c * freqs_i).flatten(2)   # [N, n, d]
    return y.to(x.dtype)


def rope_cos_sin(max_seq_len: int, dim: int, theta: float = 10000.0
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """[生产路径] 实数 fp32 cos/sin 表（R1/R9 + 根治 B1）。

    - 实数表被 Module.to(bf16) 转换只降精度、不丢信息（complex 会丢虚部 → B1）。
    - fp32 生成、模型级共享一份（R9：不再每块各存一份 complex 表）。
    - 返回 (cos, sin)，各 [max_seq_len, dim]；两半重复以适配 rotate_half 形式。
    """
    assert dim % 2 == 0, "RoPE head_dim 必须偶数"
    c = dim // 2                                        # 复数通道数
    inv_freq = 1.0 / torch.pow(
        theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim))
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    # 返回 [L, c]（每个复数通道一个频率），施加时 repeat_interleave 到 [L, dim]。
    # 注意：不是 cat([cos,cos]) —— 那对应 half-split 配对，与复数版 view_as_complex
    # 的相邻配对不等价（实测差 6.89）。
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 实数旋转：相邻配对 [-x_odd, x_even]，严格匹配 view_as_complex 的
    (x[2j], x[2j+1]) 配对。"""
    x = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)


@lru_cache(maxsize=32)
def _rope_table_factory(max_seq_len: int, head_dim: int, theta: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """R9：模型级共享一份实数 cos/sin 表（lru_cache 按 shape 缓存，不随模型 .to()）。
    返回 fp32 CPU 表；施加前按 q.dtype cast（bf16 走快速路径）。
    """
    return rope_cos_sin(max_seq_len, head_dim, theta)


def rope_apply_real(x: torch.Tensor, grid: Tuple[int, int, int],
                    cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """[生产路径] 实数 3D RoPE，batch 广播无 Python loop（R1）。

    x: [B, N, n_heads, head_dim] 或 [N, n_heads, head_dim]（兼容两种调用）；
    cos/sin: [max_seq_len, head_dim] 模型级共享实数表。
    t/h/w 三段分频与复数版完全一致（数值等价，差 ≤1 ULP）。
    """
    squeezed = x.dim() == 3
    if squeezed:
        x = x.unsqueeze(0)
    B, N, n, d = x.shape
    T, H, W = grid
    assert N == T * H * W, f"RoPE token 数 {N} 与 grid {grid} 不符"
    c = d // 2                                          # 复数通道数
    c3 = c // 3
    # cos/sin 是 [L, c]（每复数通道一频），按 t/h/w 分频（与复数版 rope_apply_3d 同）
    split = [c - 2 * c3, c3, c3]
    ct, ch, cw = cos.split(split, dim=-1)
    st, sh, sw = sin.split(split, dim=-1)
    def grid_table(tab_t, tab_h, tab_w):
        g = torch.cat([
            tab_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1),
            tab_h[:H].view(1, H, 1, -1).expand(T, H, W, -1),
            tab_w[:W].view(1, 1, W, -1).expand(T, H, W, -1),
        ], dim=-1).reshape(1, N, 1, c)                  # [1,N,1,c] 复数通道级
        return g.repeat_interleave(2, dim=-1)           # -> [1,N,1,d] 实数配对级
    cos_g, sin_g = grid_table(ct, ch, cw), grid_table(st, sh, sw)
    # 按 x dtype 计算：x 为 bf16 时走 bf16 快速路径（实测 3.6×，见实现记录 R1）
    xf = x.to(cos_g.dtype) if x.dtype in (torch.bfloat16, torch.float16) else x.float()
    y = xf * cos_g + rotate_half(xf) * sin_g
    y = y.to(x.dtype)
    return y.squeeze(0) if squeezed else y


# ---------------------------------------------------------------------------
# 基础构件
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Wan 风格 RMSNorm（对齐教师，见 wan22/modules/model.py:82）。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L0: F.rms_norm 单融合核替手写 .float() 往返（fp32 diff=0 已验证）。
        # F.rms_norm(x, normalized_shape, weight, eps) 内部 fp32 累加。
        # 注意：weight 需与 x 同 dtype（F.rms_norm 不做隐式提升）。
        return F.rms_norm(x, (self.dim,), self.weight, eps=self.eps)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制：x * (1+scale) + shift。shape 广播对齐 [B,L,C]。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TeacherAdaLN(nn.Module):
    """L1 教师式条件调制（对齐 Wan 的共享 embedder + 每块小表设计）。

    旧版每块一个 Linear(dim,6*dim)=56.6M，26 块 = 1.47B（占全模型 32.5%，
    但只吃池化 cond、算力占比 0.048%，是纯显存死开销）。
    教师做法：共享 embedder MLP (dim->dim) 只算一次，每块一个 (1,6,dim) 小表
    提供逐块独立的 6 段调制偏置 → 参数 1.47B → ~10M，省 ~1.46B / ~2.9GB 权重。

    教师 Wan 的 scale_shift_table 形状 (1,6,3072) 与此处逐块表同构 → 热启动可直拷。
    """
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        # R8: 两层 MLP（教师同款；原单层 Linear(t_emb_dim,dim)+SiLU）
        self.embedder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.embedder[-1].weight)
        nn.init.zeros_(self.embedder[-1].bias)

    def forward(self, cond_raw: torch.Tensor) -> torch.Tensor:
        """cond_raw: [B, cond_dim] -> e: [B, dim]（共享，全块复用）。"""
        return self.embedder(cond_raw)


MLP_RATIO = 8 / 3   # 设计选定：原生 FFN(mlp_ratio=4, 2 矩阵) -> SwiGLU(8/3, 3 矩阵) 参数持平
                    # 校验：原生 2*d*4d=8d²；SwiGLU 3*d*(8/3 d)=8d² ✓

class SwiGLU(nn.Module):
    """Dual-Act 分段激活 SwiGLU（架构图 Udit 4.5B 主干 FFN，V2-24）。

    原生 SwiGLU: w3(silu(w1(x)) * w2(x))。
    Dual-Act（架构图 §4）：|z| < t 段用 SiLU，|z| >= t 段平滑线性压缩，
    逐 token 阈值 t 可由 VLM Slot 特征经北大桥下发（t_cond，阶段2）；
    阶段1 无 VLM，用可学习标量阈值。分段用 where 实现 → torch.compile 友好。

    参数量：3 个矩阵与原生持平（8d²），额外仅 1 个标量阈值/层，不改变 4.5B 校准。
    dual_act=False 时严格退化为标准 SwiGLU（数值等价、自检兼容）。
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False,
                 dual_act: bool = False, init_threshold: float = 1.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = round(dim * MLP_RATIO)
        # L0: w1/w2 合并为单个 GEMM（2*hidden 输出后 chunk），核数减半。
        self.hidden_dim = hidden_dim
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # 输出投影
        self.dual_act = bool(dual_act)
        # F1: 阈值 2.5→1.0（实测 2.5 时 P(|z|>=t)=3e-5 压缩支全死且梯度锁死；
        #     1.0 对应约 8-9% 元素进压缩区，支路存活可学）。
        self.log_threshold = nn.Parameter(torch.tensor(math.log(init_threshold)))
        self.compress_slope = 0.1

    @property
    def w1(self):
        """兼容视图（selftest 直接引用 .w1/.w2）：w12 前半。"""
        return self._w12_view(0)

    @property
    def w2(self):
        return self._w12_view(1)

    def _w12_view(self, idx: int) -> nn.Linear:
        if not hasattr(self, "_w12_views"):
            d, h = self.hidden_dim, self.w12.in_features
            views = []
            for j in range(2):
                sl = nn.Linear(h, d, bias=self.w12.bias is not None)
                sl.weight = torch.nn.Parameter(self.w12.weight[j * d:(j + 1) * d],
                                               requires_grad=False)
                if self.w12.bias is not None:
                    sl.bias = torch.nn.Parameter(self.w12.bias[j * d:(j + 1) * d],
                                                 requires_grad=False)
                views.append(sl)
            self._w12_views = views
        return self._w12_views[idx]

    def forward(self, x: torch.Tensor, t_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, v = self.w12(x).chunk(2, dim=-1)
        if not self.dual_act:
            return self.w3(F.silu(z) * v)
        t = self.log_threshold.exp()
        if t_cond is not None:                       # 阶段2：逐 token 阈值（VLM Slot）
            t = t + t_cond
        lo = F.silu(z)
        hi = F.silu(t) + (z.abs() - t) * self.compress_slope * torch.sign(z)
        act = torch.where(z.abs() < t, lo, hi)
        return self.w3(act * v)


class GlobalConditionalMoE(nn.Module):
    """全局条件空间 MoE 三专家（anime / photo / 3d）——speedrun 定版形态（V2-27）。

    ⚠ 与已废弃的「按块 FFN 替件版」根本不同：
      - 旧版：每个 UViTBlock 内建 3 专家、吃 token 流 [B,N,C] 替换 FFN →
        26 块 × 226.5M = 5.89B + 1.96B 死 ffn，算力 3×FFN/块（均属误装）。
      - 本版：**全模型仅一套**，吃条件向量 e [B,dim]（不是 token），产出同维
        c_expert [B,dim]，再以 α 门控加法注入指定 block（默认 7–12）。
        条件向量长度 dim=3072 ≪ 990 token，算力开销 ≈ 0。

    - 参数预算锁死 ≈1B：3 专家 × n_layers(4) × 2·dim·(hidden_mult·dim)
      = 3 × 4 × 2 × 3072 × 13440 ≈ 991M（+ router/α，可忽略）。
    - Top1 硬路由（argmax one-hot + straight-through 梯度，router 可训）。
    - α 标量零初始化 → c_expert ≡ 0 → 注入后 cond 逐 bit 恒等（α=0 回退红线）。
    - 专家输出经 tanh 有界 → 即便 α=0 也不会因 inf 产生 0×inf=NaN。
    """
    def __init__(self, dim: int, num_experts: int = 3, hidden_mult: float = 4.375,
                 n_layers: int = 4, inject_start: int = 7, inject_end: int = 12):
        super().__init__()
        self.dim = dim
        self.num_experts = int(num_experts)
        self.n_layers = int(n_layers)
        self.hidden = max(32, int(dim * hidden_mult))
        self.inject_start = int(inject_start)
        self.inject_end = int(inject_end)
        self.last_route = None                        # [B] 记录本步 top1 专家（观测用）

        def make_expert():
            layers = []
            for _ in range(self.n_layers):
                layers += [nn.Linear(dim, self.hidden), nn.SiLU(),
                           nn.Linear(self.hidden, dim)]
            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([make_expert() for _ in range(self.num_experts)])
        self.router = nn.Linear(dim, self.num_experts, bias=False)
        nn.init.zeros_(self.router.weight)            # 初始路由 logits 全 0 → 均匀、稳定
        self.alpha = nn.Parameter(torch.zeros(1))     # 零初始化：严格恒等起步

    def forward(self, cond: torch.Tensor,
                style_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """cond: [B,dim] 共享条件 e → c_expert: [B,dim]（α=0 时严格 0）。"""
        logits = style_logits if style_logits is not None else self.router(cond)
        top1 = logits.argmax(dim=-1)                                  # [B] 硬路由
        hard = F.one_hot(top1, self.num_experts).to(cond.dtype)       # [B,E]
        soft = torch.softmax(logits.float(), dim=-1).to(cond.dtype)
        w = hard - soft.detach() + soft                               # STE：前向 top1，梯度走 soft
        outs = torch.stack([torch.tanh(expert(cond)) for expert in self.experts], 0)  # [E,B,dim]
        c = torch.einsum("be,ebd->bd", w, outs)                       # [B,dim] 只取选中专家
        self.last_route = top1.detach()
        return self.alpha * c                                         # α=0 → 严格 0 向量

    def freeze(self) -> "GlobalConditionalMoE":
        """冻结三专家权重，仅保留 router / α 可训（专家热启动后固定）。"""
        for p in self.experts.parameters():
            p.requires_grad_(False)
        return self


# ---------------------------------------------------------------------------
# 注意力：2+1D 分解（§3.4）
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """通用多头自注意力（纯 torch SDPA，CPU 可跑），带可选 3D RoPE。
    输入 x: [B, N, C]，其中 N 为该注意力域内的 token 数。
    rope_grid 给定时，对 q/k 施加 t/h/w 分频 RoPE。
    """
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6,
                 causal: bool = False, use_rope: bool = True, rope_max_len: int = 1024,
                 attn_backend: str = "auto"):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        # "flash2" = 真 FA2 优先（flash_attn 包 / SDPA FLASH backend），
        # CPU 或 fp32 自动降级 math（不伪装成 FA2）。详见 attn_backend.py。
        self.attn_backend = attn_backend
        self.use_rope = use_rope and (self.head_dim % 2 == 0)
        self.rope_max_len = rope_max_len
        # L0: qkv 合并成单个 GEMM（核数 3->1，权重排布变化）；
        # 保留 self.q/self.k/self.v 属性名指向视图切片以兼容 selftest §2 直接引用。
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        # B1 根治：不再注册 complex buffer（.to(bf16) 会静默丢虚部）。
        # 实数 cos/sin 表由 _rope_table_factory 模型级共享（R9），施加前按 q.dtype cast。

    @property
    def q(self):
        """兼容视图：qkv 前三之一（仅供 selftest/外部读取，不用于前向）。"""
        return self._qkv_slice(0)

    @property
    def k(self):
        return self._qkv_slice(1)

    @property
    def v(self):
        return self._qkv_slice(2)

    def _qkv_slice(self, idx: int) -> nn.Linear:
        if not hasattr(self, "_qkv_views"):
            d = self.dim
            w = self.qkv.weight
            views = []
            for j in range(3):
                sl = nn.Linear(d, d, bias=False)
                sl.weight = torch.nn.Parameter(w[j * d:(j + 1) * d], requires_grad=False)
                views.append(sl)
            self._qkv_views = views
        return self._qkv_views[idx]

    def forward(self, x: torch.Tensor, rope_grid: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)   # [B,nh,N,hd]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        # 3D RoPE：实数 fp32 表 -> 按 q.dtype cast，batch 广播无 Python loop（R1/R9）。
        if self.use_rope and rope_grid is not None:
            cos, sin = _rope_table_factory(self.rope_max_len, self.head_dim, 10000.0)
            cos = cos.to(device=q.device, dtype=q.dtype)
            sin = sin.to(device=q.device, dtype=q.dtype)
            # rope_apply_real 期望 [B,N,nh,hd]，这里 q 是 [B,nh,N,hd]
            q = rope_apply_real(q.transpose(1, 2), rope_grid, cos, sin).transpose(1, 2)
            k = rope_apply_real(k.transpose(1, 2), rope_grid, cos, sin).transpose(1, 2)
        y = dispatch_attention(q, k, v, is_causal=self.causal,
                               backend=self.attn_backend)
        y = y.transpose(1, 2).reshape(B, N, C)
        return self.o(y)


class SpatialAttention(nn.Module):
    """空间头：单帧画面构图/纹理。x [B, T*H*W, C], grid=(T,H,W)。
    把 token 重排为 [B*T, H*W, C]，在每帧内部做自注意力。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 attn_backend: str = "auto"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=False,
                                  attn_backend=attn_backend)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = grid
        assert N == T * H * W, f"token 数 {N} 与 grid {grid} 不符"
        y = x.view(B, T, H * W, C).reshape(B * T, H * W, C)
        # 空间域：每帧内 (1,H,W)；RoPE 的 T 维取 1（帧内只用 h/w 分频）
        y = self.attn(y, rope_grid=(1, H, W))
        return y.view(B, T, H * W, C).reshape(B, N, C)


class TemporalAttention(nn.Module):
    """时序头：帧间运动/连续性。x [B, T*H*W, C], grid=(T,H,W)。
    把 token 重排为 [B*H*W, T, C]，在每个空间位置跨帧做注意力。
    causal=True 时加因果掩码（§3.4，抑制长序列跨帧乱跳）。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 causal: bool = False, attn_backend: str = "auto"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=causal,
                                  attn_backend=attn_backend)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = grid
        assert N == T * H * W
        # [B, T*H*W, C] -> [B, T, H, W, C] -> [B, H, W, T, C] -> [B*H*W, T, C]
        y = x.view(B, T, H, W, C).permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        # 时序域：每个空间位置 (T,1,1)；RoPE 只用 t 分频
        y = self.attn(y, rope_grid=(T, 1, 1))
        return y.view(B, H, W, T, C).permute(0, 3, 1, 2, 4).reshape(B, N, C)


class JointAttention(nn.Module):
    """完整 3D 联合注意力：瓶颈层保留，学全局运动（§3.4）。
    更接近教师 Wan 的 Full Attention 范式。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 attn_backend: str = "auto"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=False,
                                  attn_backend=attn_backend)

    def forward(self, x: torch.Tensor, grid: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        # 瓶颈层完整 3D：用 (T,H,W) 全分频 RoPE，最贴近教师
        return self.attn(x, rope_grid=grid)


class GSA(nn.Module):
    """Gated Spatial Attention（另一条成熟路线）。
    ---------------------------------------------------------------
    门控混合时空注意力：空间头与时序头并行计算，用可学习门控逐 token 混合，
    而不同于 §3.4 的"空间/时序交替 Block"。门控由输入自身 + 条件参与，
    让模型自适应决定每个位置偏重"画面构图"还是"帧间运动"。

    混合公式：
        y_s = SpatialAttn(norm(x))
        y_t = TemporalAttn(norm(x))
        g   = sigmoid(gate_proj(cond))          # [B,N,1] 逐 token 门控
        y   = g * y_s + (1 - g) * y_t
        out = x + gate_out * o_proj(y)

    设计参考：GSA 常见于视频扩散/视频理解的门控时空混合（如 Gated Spatio-
    temporal Attention 一族）；本实现为最小可跑骨架，门控条件可换 per-token
    特征。因果性由时序头内部的 causal 开关控制。
    """
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 causal_temporal: bool = False, cond_dim: Optional[int] = None,
                 attn_backend: str = "auto"):
        super().__init__()
        self.spatial = SpatialAttention(dim, num_heads, qk_norm,
                                        attn_backend=attn_backend)
        self.temporal = TemporalAttention(dim, num_heads, qk_norm,
                                          causal_temporal,
                                          attn_backend=attn_backend)
        self.norm = RMSNorm(dim)
        # 逐 token 门控：由 dim 维条件向量投影到 1（可换成 per-token 特征）
        self.gate_proj = nn.Linear(cond_dim or dim, 1)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(1))  # 零初始化：初始近似恒等

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int],
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm(x)
        y_s = self.spatial(h, grid)
        y_t = self.temporal(h, grid)
        # 门控源：优先用 cond（[B,C]），否则用输入池化
        src = cond if cond is not None else h.mean(dim=1)
        g = torch.sigmoid(self.gate_proj(src)).view(B, 1, 1)
        y = g * y_s + (1 - g) * y_t
        return x + torch.tanh(self.out_gate) * self.o_proj(y)


# ---------------------------------------------------------------------------
# FFN 子层检查点包装（ckpt_scope="ffn_only" 用）：只重算 FFN，SelfAttn 不重算。
# ---------------------------------------------------------------------------
class _FFNCheckpoint(nn.Module):
    """包住 FFN，前向时对 FFN 单独套 checkpoint（use_reentrant=False）。
    通过 module 级 flag 由 UViT3D 统一开关，推理/grad 关闭时透明直通。"""
    def __init__(self, ffn: nn.Module):
        super().__init__()
        self.ffn = ffn
        self.enabled = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not (self.enabled and self.training and torch.is_grad_enabled()):
            return self.ffn(x, *args, **kwargs)
        return _ckpt.checkpoint(self.ffn, x, *args,
                                use_reentrant=False, preserve_rng_state=False, **kwargs)


# ---------------------------------------------------------------------------
# U-ViT Block：AdaLN-Zero + 6 段调制（对齐 Wan 的 6 段范式）
# ---------------------------------------------------------------------------

class UViTBlock(nn.Module):
    """单个 Transformer Block。
    - cond_mode='adaln'：6 段调制（shift/scale/gate × 2），§3.2
    - attn_mode：'spatial' / 'temporal' / 'joint'，§3.4 的 2+1D 交替
    - ffn = SwiGLU，§3.5
    """
    def __init__(self, dim: int, num_heads: int, attn_mode: str,
                 ffn_ratio: float = 4.0, causal_temporal: bool = False,
                 cond_mode: str = "adaln", qk_norm: bool = True,
                 attn_backend: str = "auto",
                 dual_act: bool = False, dual_branch_adaln: bool = False):
        super().__init__()
        assert attn_mode in ("spatial", "temporal", "joint", "gsa")
        assert cond_mode == "adaln", "V1 只用 AdaLN-Zero（§3.2，移除 token 拼接）"
        self.dim = dim
        self.attn_mode = attn_mode

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        if attn_mode == "spatial":
            self.attn = SpatialAttention(dim, num_heads, qk_norm,
                                         attn_backend=attn_backend)
        elif attn_mode == "temporal":
            self.attn = TemporalAttention(dim, num_heads, qk_norm, causal_temporal,
                                          attn_backend=attn_backend)
        elif attn_mode == "gsa":
            self.attn = GSA(dim, num_heads, qk_norm, causal_temporal, cond_dim=dim,
                            attn_backend=attn_backend)
        else:
            self.attn = JointAttention(dim, num_heads, qk_norm,
                                       attn_backend=attn_backend)

        hidden = int(dim * ffn_ratio * 2 / 3)
        self.dual_act = bool(dual_act)
        # V2-27：FFN 永远是主干 FFN（Dual-Act SwiGLU）。MoE 已改为全局条件空间
        # 专家（GlobalConditionalMoE，在 UViT3D 主层实例化），不再按块替换 FFN。
        # → 旧 use_moe 分支删除：无 2B 死 ffn、无 3×FFN/块算力、无 top-1 改造需求。
        self.ffn = SwiGLU(dim, hidden_dim=hidden, dual_act=dual_act)

        # L1 教师式 AdaLN：每块只放 (1,6,dim) 小表（18K），共享 embedder 在
        # UViT3D 层只算一次。旧版 Linear(dim,6*dim)=56.6M/块 是纯显存死开销。
        # 教师 Wan 的 scale_shift_table 形状 (1,6,3072) 与本表同构 → 热启动直拷。
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim))
        # 可选：AdaLN 双分支调制（独立 scale/shift 小 MLP，解耦均值/方差控制）。
        # 默认关（定案，本轮选 SNR 分层；保留接口供兜底微调）。
        self.dual_branch_adaln = bool(dual_branch_adaln)
        if self.dual_branch_adaln:
            self.scale_mlp = nn.Linear(dim, 3 * dim)
            self.shift_mlp = nn.Linear(dim, 3 * dim)
            nn.init.zeros_(self.scale_mlp.weight); nn.init.zeros_(self.scale_mlp.bias)
            nn.init.zeros_(self.shift_mlp.weight); nn.init.zeros_(self.shift_mlp.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                grid: Tuple[int, int, int],
                t_cond: Optional[torch.Tensor] = None,
                style_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, C]
        cond: [B, C] 时间步 emb + 文本 emb 的融合向量
        grid: (T, H, W)
        t_cond: 阶段2 VLM Slot 下发的逐 token Dual-Act 阈值（阶段1 None）
        style_logits: 阶段2 VLM 风格头下发的 MoE 路由信号（阶段1 None）
        """
        # L1 教师式：cond 现在是共享 embedder 输出 e [B,C]；每块加 (1,6,C) 小表
        # 得到本块独立的 6 段调制。初始表为 0 → gate≈0 → Block 恒等、梯度稳定。
        # [B,C] + [1,6,C] -> [B,6,C] -> 6 × [B,C]
        mods = cond.unsqueeze(1) + self.scale_shift_table   # [B,6,C]
        if self.dual_branch_adaln:
            mods = mods + torch.stack([
                self.scale_mlp(cond), self.shift_mlp(cond),
                torch.zeros_like(cond), torch.zeros_like(cond),
                torch.zeros_like(cond), torch.zeros_like(cond),
            ], dim=1)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.unbind(dim=1)

        # 注意力子层（AdaLN 调制）；GSA 额外吃条件向量做门控
        h = modulate(self.norm1(x), shift1, scale1)
        if self.attn_mode == "gsa":
            y = self.attn(h, grid, cond=cond)
        else:
            y = self.attn(h, grid)
        x = x + y * gate1.unsqueeze(1)

        # FFN 子层（AdaLN 调制）：主干恒为 Dual-Act SwiGLU。
        # 全局条件 MoE 不在这里替换 FFN，而是在 UViT3D.forward 里加法注入 cond。
        z2 = modulate(self.norm2(x), shift2, scale2)
        if self.dual_act:
            y = self.ffn(z2, t_cond=t_cond)
        else:
            y = self.ffn(z2)
        x = x + y * gate2.unsqueeze(1)
        return x


class SpecializedBlock(nn.Module):
    """V2-29 专项注意力 Block：只保留单一注意力域的轻量块。

    - mode='temporal'：只做帧间时间注意力（治频闪/角色抖动/动作跳变），插主干中段
    - mode='space'    ：只做单帧空间注意力（磨五官/线条），插深层
    - 外层零输出门控 gate=0 → 严格恒等（初始不扰动已训主干，训练中逐步学开）
    - 内部复用 UViTBlock（单注意力 + 标准 FFN），参数 ~113M/块

    为何用标准 FFN 而非缩小：设计要求「参数控制 300M 内」而非「尽量小」；
    两块标准容量 ~227M < 300M，既守住预算又不牺牲容量。
    """
    def __init__(self, dim: int, num_heads: int, mode: str,
                 attn_backend: str = "auto", causal_temporal: bool = False):
        super().__init__()
        assert mode in ("temporal", "spatial"), f"专项块只支持 temporal/spatial，收到 {mode}"
        self.mode = mode
        self.block = UViTBlock(dim, num_heads, mode,
                               attn_backend=attn_backend,
                               causal_temporal=causal_temporal)
        self.out_gate = nn.Parameter(torch.zeros(1))     # 零起步：严格恒等

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                grid: Tuple[int, int, int]) -> torch.Tensor:
        y = self.block(x, cond, grid)
        return x + torch.tanh(self.out_gate) * (y - x)  # gate=0 → 逐 bit 恒等

    def disable(self):
        with torch.no_grad():
            self.out_gate.zero_()


# ---------------------------------------------------------------------------
# Skip：可学习门控 + 谱约束（§3.3）
# ---------------------------------------------------------------------------

class GatedSkip(nn.Module):
    """x_deep = x_deep + alpha * Norm(skip_feat)
    - alpha：可学习门控，初始 0.1（白皮书标注为【工程推断】，可调）
    - 谱约束：对 skip 支路 Linear 做 spectral_norm，抑制高频噪声传递
    - 消融：把 alpha 置 0 → 退化为无 skip 直筒 DiT（§1.3 红线 8）
    """
    def __init__(self, dim: int, alpha_init: float = 0.1, spectral: bool = True):
        super().__init__()
        self.norm = RMSNorm(dim)
        proj = nn.Linear(dim, dim, bias=False)
        if spectral:
            proj = nn.utils.parametrizations.spectral_norm(proj)
        self.proj = proj
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x_deep: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        branch = self.proj(self.norm(skip_feat))
        # 统一走门控入口：默认 SAKIKO_CUSTOM_OP 未设 → native（数值=原 x+α·branch）；
        # 设 SAKIKO_CUSTOM_OP=1 → 走注册的 sakiko.fused_gate_residual 算子。
        # 两条路径在 alpha=0 时都严格返回 x_deep（比 x+0·branch 更抗 inf/nan 污染）。
        return _gated_residual_op(x_deep, branch, self.alpha, tag="gated_skip")

    def disable(self):
        """消融开关：门控置 0，等效关闭 skip。"""
        with torch.no_grad():
            self.alpha.zero_()


# ---------------------------------------------------------------------------
# 金字塔：时空 Patch Merge / Unmerge（§4.1）
# ---------------------------------------------------------------------------

class PatchMerge3D(nn.Module):
    """时空块合并：空间缩 s 倍，时间缩 t 倍。
    grid (T,H,W) -> (T/t, H/s, W/s)，token 数缩 s*s*t 倍。
    §4.1 建议：空间每 stage 缩 2，时间保持 1（t=1），避免运动模糊。
    """
    def __init__(self, dim: int, s: int = 2, t: int = 1):
        super().__init__()
        self.s, self.t = s, t
        self.proj = nn.Linear(dim * s * s * t, dim)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]):
        B, N, C = x.shape
        T, H, W = grid
        assert (T % self.t == 0 and H % self.s == 0 and W % self.s == 0), \
            f"grid {grid} 不能被 s={self.s},t={self.t} 整除"
        x = x.view(B, T // self.t, self.t, H // self.s, self.s, W // self.s, self.s, C)
        # 重排使 t,s,s 三个维度进入通道
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, (T // self.t) * (H // self.s) * (W // self.s), C * self.t * self.s * self.s)
        x = self.proj(x)
        return x, (T // self.t, H // self.s, W // self.s)


class PatchUnmerge3D(nn.Module):
    """金字塔 decoder 端上采样恢复：token 数放大 s*s*t 倍。"""
    def __init__(self, dim: int, s: int = 2, t: int = 1):
        super().__init__()
        self.s, self.t = s, t
        self.proj = nn.Linear(dim, dim * s * s * t)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int]):
        B, N, C = x.shape
        T, H, W = grid
        x = self.proj(x)
        x = x.view(B, T, H, W, self.t, self.s, self.s, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, (T * self.t) * (H * self.s) * (W * self.s), C)
        return x, (T * self.t, H * self.s, W * self.s)


# ---------------------------------------------------------------------------
# 新增轻量算子（默认关，可配置开启）：时序低通 / 运动头 / 频域分支 / 局部卷积
# ---------------------------------------------------------------------------

class TemporalLowPass(nn.Module):
    """教师 flicker 0.0244 的关键算子：depthwise 时序低通卷积（每通道独立 3 帧滑均）。
    参数 3072*3≈9K，算力可忽略。resolve_depth 从 latent 的 T 维取；初始恒等
    （中心=1）可选。输入 z [B,N,C] + grid，按 (T,H,W) reshape 后沿 T 卷积。
    """
    def __init__(self, dim: int, kernel: int = 3, init_identity: bool = False):
        super().__init__()
        self.dim, self.k = dim, kernel
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, groups=dim,
                              padding=kernel // 2, bias=False)
        if init_identity:
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:, :, kernel // 2] = 1.0

    def forward(self, z: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = z.shape
        T, H, W = grid
        # [B,T,H*W,C] -> [B*H*W, C, T]：H*W 当 batch，C 当 channel（depthwise groups=C），T 当长度
        y = z.view(B, T, H * W, C).permute(0, 2, 3, 1).reshape(B * H * W, C, T)
        y = self.conv(y).view(B, H * W, C, T).permute(0, 3, 1, 2).reshape(B, N, C)
        return z + y   # 残差：低通作为修正项，初始权重小 -> 近似恒等


class MotionHead(nn.Module):
    """帧级运动/光流预测辅助头（训练期辅助监督，推理可摘）。
    从 token 特征预测每帧 3 维运动向量。参数 <2M。
    """
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, z: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = z.shape
        T, H, W = grid
        per_tok = self.net(z).view(B, T, H * W, 3)
        return per_tok.mean(dim=2)   # [B, T, 3] 每帧平均运动


class FreqBranch(nn.Module):
    """轻量频域分支：对 token 序列做 FFT，取幅度谱经小 MLP 输出全局频率调制。
    参数 <2M；学全局频率分布、抗伪影/彩噪。零门控起步（初始恒等）。
    """
    def __init__(self, dim: int, rank: int = 256):
        super().__init__()
        self.rank = rank
        self.net = nn.Sequential(nn.Linear(dim, rank), nn.SiLU(), nn.Linear(rank, dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(z.float(), dim=1).abs().to(z.dtype).mean(dim=1)  # [B,dim]
        return z + torch.tanh(self.gate) * self.net(spec).unsqueeze(1)


class LocalConvFFN(nn.Module):
    """depthwise spatial conv 局部增强（ConvNeXt/HunyuanVideo 验证）：
    在 FFN 前对 token 的 (H,W) 做 3x3 depthwise conv，补 Transformer 缺失的
    局部归纳偏置。参数 3072*9≈28K/块，30 块 <1M。零门控起步。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = z.shape
        T, H, W = grid
        y = z.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        y = self.dw(y).view(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, N, C)
        return z + torch.tanh(self.gate) * y


class DetailRefineHead(nn.Module):
    """V2-30 DDT 式高频细节残差头（Motif-Video 2B 验证：低/高频分离解码提质量）。

    - 独立于主速度场头：主头管低频语义/构图，本头只补高频细节（五官/线条/纹理）
    - per-token MLP：dim → hidden → out，作用在最终 token 特征上（不进主干 30 块）
    - 零输出门控 gate=0 → 严格恒等（初始不扰动主头，训练中逐步学开）
    - 算力：2 个 per-token GEMM，对 990 token ≈ 0.06 TFLOP，占步时 <1%
    - 参数：hidden 可调（默认 19456 → ~63.5M，用于把总量补到 ~4.0B）
    """
    def __init__(self, dim: int, patch_size: Tuple[int, int, int],
                 out_channels: int, hidden: int = 19456, eps: float = 1e-6):
        super().__init__()
        self.out_dim = math.prod(patch_size) * out_channels
        self.norm = RMSNorm(dim, eps)
        self.up = nn.Linear(dim, hidden, bias=True)
        self.down = nn.Linear(hidden, self.out_dim, bias=True)
        self.out_gate = nn.Parameter(torch.zeros(1))       # 零起步：严格恒等

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,N,dim] → 细节残差 [B,N,out_dim]（gate=0 时严格 0）。"""
        h = self.norm(x)
        h = F.silu(self.up(h))
        return torch.tanh(self.out_gate) * self.down(h)

    def disable(self):
        with torch.no_grad():
            self.out_gate.zero_()


# ---------------------------------------------------------------------------
# 输出头（§3.6）：速度场主头 + 时序辅助头
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """主头：预测速度场 v（流匹配目标）。
    辅助头：输出帧间相似度残差，训练期加 flicker 正则（§3.6，工程推断）。
    """
    def __init__(self, dim: int, patch_size: Tuple[int, int, int],
                 out_channels: int = 16, aux_head: bool = True, head_adaln: bool = False,
                 detail_head: bool = False, detail_hidden: int = 19456):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_channels)
        # V2-30 DDT 高频细节残差头（零门控起步；默认关，4b5 开）
        self.detail_head = DetailRefineHead(dim, patch_size, out_channels,
                                            hidden=detail_hidden) if detail_head else None
        # V2-31 显存勘察：此前细节头直接调用、激活全程驻留（实测 45MB）；
        # 由 UViT3D.enable_gradient_checkpointing() 同步打开，走 checkpoint 重算。
        self.detail_ckpt: bool = False
        # 时序辅助头：每 token 一个标量残差，用于 flicker 正则
        self.aux_head = nn.Sequential(nn.Linear(dim, dim // 4), nn.SiLU(),
                                      nn.Linear(dim // 4, 1)) if aux_head else None
        # R7: 输出头 AdaLN 调制（教师 head.modulation 有，学生原来缺）。
        # 零初始化 scale/shift -> 初始恒等。+18K 参数，算力近零。
        self.head_adaln = bool(head_adaln)
        if self.head_adaln:
            self.head_mod = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.head_mod.weight)
            nn.init.zeros_(self.head_mod.bias)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int],
                cond_e: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        T, H, W = grid
        h = self.norm(x)  # R10: norm 只算一次，主头/辅助头复用（原来算两遍）
        if self.head_adaln and cond_e is not None:
            scale, shift = self.head_mod(cond_e).chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        v = self.head(h)  # [B,N,prod(p)*out_c]
        # V2-30 高频细节残差：gate=0 时严格 0，主头输出逐 bit 不变
        # V2-31：开启 checkpoint 时重算细节头激活（省 ~45MB 峰值，数值等价）
        if self.detail_head is not None:
            if self.detail_ckpt and torch.is_grad_enabled():
                dv = _ckpt.checkpoint(self.detail_head, x,
                                      use_reentrant=False, preserve_rng_state=False)
            else:
                dv = self.detail_head(x)
            v = v + dv
        # 重排回 [B, out_c, T*p_t, H*p_h, W*p_w]
        pt, ph, pw = self.patch_size
        v = v.view(B, T, H, W, pt * ph * pw, self.out_channels)
        v = v.permute(0, 4, 5, 1, 2, 3).contiguous()
        v = v.view(B, self.out_channels, T * pt, H * ph, W * pw)
        aux = self.aux_head(h).view(B, T, H, W).squeeze(-1) if self.aux_head else None
        return v, aux


# ---------------------------------------------------------------------------
# 主模型：对称 U 型 3D U-ViT
# ---------------------------------------------------------------------------

@dataclass
class UViT3DConfig:
    """模型配置。dim/depth 决定参数量；4.5B 基线给在 v1_baseline() 工厂。"""
    in_channels: int = 16
    out_channels: int = 16
    patch_size: Tuple[int, int, int] = (1, 2, 2)   # (t, h, w)
    dim: int = 3072                                 # 与 Wan teacher hidden 对齐
    num_heads: int = 24
    ffn_ratio: float = 4.0
    # U 型：encoder 层数 = decoder 层数，bottleneck 层数
    enc_depth: int = 6
    dec_depth: int = 6
    bottleneck_depth: int = 2
    # 注意力策略：2+1D 交替（spatial/temporal）或 GSA 门控混合，瓶颈用 joint
    # 设计选定：GSA（Gated Spatial Attention）为另一条成熟路线，与 2+1D 并列可选
    attn_pattern: Tuple[str, ...] = ("spatial", "temporal")
    attn_strategy: str = "alternate"     # "alternate"=2+1D 交替 | "gsa"=门控混合
    causal_temporal: bool = False
    # 注意力后端：设计选定 "flash2"（真 FA2 优先），CPU 自动降级 math。
    # 可选 "auto"/"flash2"/"sdpa"/"math"；详见 attn_backend.py。
    attn_backend: str = "flash2"
    # 梯度检查点：只砍前向激活显存，不减少权重/梯度/优化器状态（V2-23）。
    # 显存紧张时开启；推理恒关闭（推理无梯度，无收益只增开销）。
    grad_checkpoint: bool = False
    # 检查点粒度（V2-25）："block"=整块重算（默认，GPU 实测省 ~1.1GiB）；
    # "ffn_only"=只重算 FFN/CrossAttn、SelfAttn 不重算（省重算开销，但激活显存收益未实测，
    #   因为 FlashAttn 的 SelfAttn 激活本就小，关掉其重算可能反而增峰——谨慎用）。
    ckpt_scope: str = "block"
    # 架构图主干结构（V2-24）：Dual-Act 分段激活 SwiGLU（可训）+ MoE 三专家（阶段1旁路）
    dual_act: bool = True          # 主干 FFN 启用 Dual-Act 分段激活（默认开）
    # V2-27 全局条件空间 MoE（speedrun 定版）：全模型一套、吃 cond [B,dim]、top1、α 注入
    use_moe: bool = False          # 阶段1 不启用；阶段2 设 True 才有 ~1B 全局专家
    moe_num_experts: int = 3
    moe_hidden_mult: float = 4.375 # 专家 hidden = dim × 该值（dim=3072→13440 → 单专家≈330M）
    moe_n_layers: int = 4          # 单专家 MLP 层数（4 层 → 各 ≈330M）
    moe_inject_start: int = 7      # α 门控注入起始 block（全局序号 0..N-1）
    moe_inject_end: int = 12       # 注入结束 block（含），共 6 块
    moe_style_dim: int = 0         # VLM 风格头维度（0=用内部 router；>0=预留外接风格 logits）
    # VAE：用 AMD 蜂鸟本地轻量版（TurboVAED 同源，14×/4×），保持 fp16 不量化
    vae_path: str = _os.environ.get("PYRAGATE_VAE_PATH", "Hummingbird-Wan2.2-VAE_bf16.safetensors")
    # skip
    skip_alpha_init: float = 0.1
    skip_spectral: bool = True
    # 金字塔（V2，默认关；V1 单分辨率）
    pyramid: bool = False
    pyramid_s: int = 2
    pyramid_t: int = 1
    # VLM 多尺度分层注入（北大桥 4096->3072）
    use_vlm: bool = False
    vlm_dim: int = 4096                             # Qwen3-VL-8B hidden
    vlm_stage_ratio: float = 0.5                    # 前 50% block 用全局 cond，后 50% 用 per-token
    # 输出头
    aux_head: bool = True
    cond_dim: int = 768                             # 文本 emb 维度（投影到 dim）
    t_emb_dim: int = 256
    # L1 教师式 adaLN + 可选调制增强
    dual_branch_adaln: bool = False                 # AdaLN 双分支（独立 scale/shift MLP）
    snr_cond: bool = False                          # SNR 分层条件编码（按噪声级调 VLM 粒度）
    ref_align: bool = False                         # 参考帧残差对齐旁路（ref=None 时恒等）
    vlm_low_rank: int = 512                         # VLM cross-injector 低秩 r（0=全秩）
    head_adaln: bool = False                        # R7 输出头 AdaLN 调制（+18K）
    # 轻量算子（定案全上；tiny 自检默认关以免扰动逐 bit 断言）
    temporal_lowpass: bool = False                  # 时序低通 conv（教师 flicker 关键，~9K）
    motion_head: bool = False                       # 帧级运动辅助头（<2M）
    freq_branch: bool = False                       # 频域分支（<2M）
    local_conv: bool = False                        # depthwise spatial conv 局部增强（<1M）
    # V2-28 轻量化条件编码器（6层 r384 低秩，纯 torch SDPA 无 corr kernel）
    style_encoder: bool = False                     # 分层风格条件编码器（≈72M，约束 MoE 画风分区）
    ref_align_encoder: bool = False                 # 参考帧重采样对齐编码器（≈68M，角色保形）
    cond_rank: int = 384                            # 两个编码器的低秩瓶颈 r
    cond_layers: int = 6                            # 两个编码器的层数（与 VLM 减层方案一致）
    # V2-29 专项注意力 Block（纯时序治频闪 / 纯空间磨单帧；各 ~113M，合计 ~227M < 300M）
    temporal_block: bool = False                    # 纯时序块（只时间注意力），插主干中段
    spatial_block: bool = False                     # 纯空间块（只空间注意力），插深层
    # V2-30 DDT 高频细节残差头（~63.5M，补总量到 ~4.0B；per-token 算力 <1% 步时）
    detail_head: bool = False                       # 输出头高频细节残差（低频语义/高频细节分离）
    detail_hidden: int = 19456                      # 细节头 MLP 隐维（19456 → 63.5M）

    @classmethod
    def tiny(cls):
        """CPU 自检用极小配置（不占显存，秒级跑完）。"""
        return cls(dim=64, num_heads=4, enc_depth=2, dec_depth=2,
                   bottleneck_depth=1, cond_dim=32, t_emb_dim=16,
                   patch_size=(1, 2, 2), aux_head=True)

    @classmethod
    def v1_baseline_4b5(cls):
        """白皮书 V1 基线：4.5B 参数。

        2026-11-21 校准：原 enc/dec=7 实测仅 2.790B（用 meta device 零内存统计），
        与标称 4.5B 差 1.7B。改 enc/dec=12 后实测 4.536B（ffn_ratio=4.0，
        SwiGLU 内部 ×2/3 已体现"8/3 持平"语义，见 §3.5）。selftest 第 9 段有硬断言。
        """
        # 2026-11-22：适配本地蜂鸟轻量 VAE（Hummingbird-Wan2.2-VAE_bf16，
        # TurboVAED 同源）。该 VAE latent 为 48 通道 / 空间 16× 压缩（实测
        # 704×480×9f -> [1,48,3,30,44]），不同于通用 Wan 16 通道。
        # 故 in/out_channels=48；参数量增量 ~1.2M（4.536->~4.537B，断言仍通过）。
        #
        # 3.5B 改造：
        # - 教师式 adaLN（共享 embedder + 每块 18K 小表）：省 ~1.46B
        # - bottleneck_depth 2->6（全局 3D 注意力容量 ×3）：+0.45B
        # - cond_dim 768->4096（UMT5-XXL hidden，逐 token 注入前置）
        # - head AdaLN + t_embed 两层 MLP
        # 实测目标 ≈3.52B（meta device 精确统计，selftest §9 硬断言同步）。
        return cls(dim=3072, num_heads=24, enc_depth=12, dec_depth=12,
                   bottleneck_depth=6, pyramid=False,
                   in_channels=48, out_channels=48, cond_dim=4096,
                   dual_branch_adaln=False, snr_cond=False, ref_align=False,
                   vlm_low_rank=512, head_adaln=True,
                   temporal_lowpass=True, motion_head=True, freq_branch=True,
                   local_conv=True,
                   style_encoder=True, ref_align_encoder=True,
                   cond_rank=384, cond_layers=6,
                   temporal_block=True, spatial_block=True,
                   detail_head=True, detail_hidden=19456)


class UViT3D(nn.Module):
    """对称 U 型 3D U-ViT 主干。

    数据流：
      latent [B,C,T,H,W] --Conv3d patchify--> tokens [B,N,dim]
      encoder（逐层 2+1D attn）--> 记录每层 skip
      bottleneck（JointAttention，完整 3D）
      decoder（逐层上采样 + GatedSkip 融合对应 encoder skip）
      OutputHead --> 速度场 v [B,C,T',H',W'] + 时序辅助残差
    """
    def __init__(self, cfg: UViT3DConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_backend_info = backend_diagnose()
        # 梯度检查点开关（训练期可 enable/disable；默认跟 cfg）
        self.grad_checkpointing: bool = bool(cfg.grad_checkpoint)
        self.ckpt_scope: str = getattr(cfg, "ckpt_scope", "block")
        pt, ph, pw = cfg.patch_size
        self.patchify = nn.Conv3d(cfg.in_channels, cfg.dim,
                                  kernel_size=cfg.patch_size,
                                  stride=cfg.patch_size)

        # 条件投影：时间步 emb + 文本 emb -> dim（R8: t_embed 已并入 TeacherAdaLN
        # 的两层 embedder，此处保留单层投影把 t_emb/text_emb 投到 dim 作为融合输入）
        self.t_embed = nn.Sequential(nn.Linear(cfg.t_emb_dim, cfg.dim), nn.SiLU())
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.dim)
        # L1 共享条件 embedder（教师式；全块共用，产出 6 段调制基 e [B,dim]）
        self.ada_embedder = TeacherAdaLN(cfg.dim, cfg.dim)
        # SNR 分层条件编码：按噪声级 sigma∈[0,1] 把条件粒度调制到各块
        # （流匹配 t 天然是噪声水平，无需额外标注；输出 per-block 的门控标量）
        self.snr_cond = bool(cfg.snr_cond)
        if self.snr_cond:
            self.snr_mlp = nn.Sequential(
                nn.Linear(1, cfg.dim // 4), nn.SiLU(),
                nn.Linear(cfg.dim // 4, cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth),
            )
        # 参考帧残差对齐旁路：ref latent 与当前 latent 的特征做轻量 cross-correlation，
        # 输出偏移调制送 AdaLN。ref=None 时整条路恒等（不进计算图），不阻断无参考训练。
        self.ref_align = bool(cfg.ref_align)
        if self.ref_align:
            self.ref_proj = nn.Linear(cfg.dim, cfg.dim)
            self.ref_gate = nn.Parameter(torch.zeros(1))   # 零初始化：初始恒等
            self.ref_norm = RMSNorm(cfg.dim)

        # encoder / decoder blocks
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        def pick_mode(i):
            return "gsa" if cfg.attn_strategy == "gsa" else cfg.attn_pattern[i % len(cfg.attn_pattern)]

        for i in range(cfg.enc_depth):
            mode = pick_mode(i)
            self.enc_blocks.append(UViTBlock(cfg.dim, cfg.num_heads, mode,
                                             cfg.ffn_ratio, cfg.causal_temporal,
                                             attn_backend=cfg.attn_backend,
                                             dual_act=cfg.dual_act,
                                             dual_branch_adaln=cfg.dual_branch_adaln))

        for i in range(cfg.bottleneck_depth):
            self.bottleneck = nn.ModuleList() if i == 0 else self.bottleneck
            self.bottleneck.append(UViTBlock(cfg.dim, cfg.num_heads, "joint",
                                             cfg.ffn_ratio, cfg.causal_temporal,
                                             attn_backend=cfg.attn_backend,
                                             dual_act=cfg.dual_act,
                                             dual_branch_adaln=cfg.dual_branch_adaln))

        for i in range(cfg.dec_depth):
            mode = pick_mode(cfg.dec_depth - 1 - i)
            self.dec_blocks.append(UViTBlock(cfg.dim, cfg.num_heads, mode,
                                             cfg.ffn_ratio, cfg.causal_temporal,
                                             attn_backend=cfg.attn_backend,
                                             dual_act=cfg.dual_act,
                                             dual_branch_adaln=cfg.dual_branch_adaln))
            self.skips.append(GatedSkip(cfg.dim, cfg.skip_alpha_init, cfg.skip_spectral))

        # 金字塔 merge/unmerge（V2，可选）
        if cfg.pyramid:
            self.merges = nn.ModuleList([
                PatchMerge3D(cfg.dim, cfg.pyramid_s, cfg.pyramid_t)
                for _ in range(cfg.enc_depth)
            ])
            self.unmerges = nn.ModuleList([
                PatchUnmerge3D(cfg.dim, cfg.pyramid_s, cfg.pyramid_t)
                for _ in range(cfg.dec_depth)
            ])
        else:
            self.merges = None
            self.unmerges = None

        # ffn_only 粒度：block 全部创建完后，把 FFN 子层包成可 checkpoint 单元。
        # 仅 ckpt_scope=="ffn_only" 生效；block 模式保持原样不动。
        if self.ckpt_scope == "ffn_only":
            for _blk in list(self.enc_blocks) + list(self.bottleneck) + list(self.dec_blocks):
                _blk.ffn = _FFNCheckpoint(_blk.ffn)
                _blk.ffn.enabled = self.grad_checkpointing

        # VLM 多尺度分层注入（北大桥 4096->3072 + 全局/cross-attn 双路径）
        self.vlm_injector = None
        if cfg.use_vlm:
            from vlm_bridge import VLMInjector
            total_blocks = cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth
            self.vlm_injector = VLMInjector(
                dim=cfg.dim, num_heads=cfg.num_heads, n_blocks=total_blocks,
                vlm_dim=cfg.vlm_dim, stage_ratio=cfg.vlm_stage_ratio,
                low_rank=getattr(cfg, "vlm_low_rank", 0),
            )

        # V2-27 全局条件空间 MoE：全模型仅一套、吃 cond [B,dim]、Top1、α 门控注入
        # blocks [inject_start, inject_end]。阶段1 use_moe=False（不实例化、0 参数）；
        # 阶段2 启用后 ≈1B，且 α=0 起步可严格回退到无专家主干。
        self.use_moe = bool(cfg.use_moe)
        self.global_moe = None
        if self.use_moe:
            n_blocks = cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth
            inj_end = min(cfg.moe_inject_end, n_blocks - 1)
            self.global_moe = GlobalConditionalMoE(
                dim=cfg.dim, num_experts=cfg.moe_num_experts,
                hidden_mult=cfg.moe_hidden_mult, n_layers=cfg.moe_n_layers,
                inject_start=cfg.moe_inject_start, inject_end=inj_end,
            )

        # V2-29 专项注意力 Block（零输出门控起步，初始恒等）
        self.temporal_special = None
        if getattr(cfg, "temporal_block", False):
            self.temporal_special = SpecializedBlock(
                cfg.dim, cfg.num_heads, "temporal",
                attn_backend=cfg.attn_backend, causal_temporal=cfg.causal_temporal)
        self.spatial_special = None
        if getattr(cfg, "spatial_block", False):
            self.spatial_special = SpecializedBlock(
                cfg.dim, cfg.num_heads, "spatial",
                attn_backend=cfg.attn_backend)

        # 新增轻量算子（4b5 全开；tiny 默认关）
        self.temporal_lowpass = TemporalLowPass(cfg.dim) if getattr(cfg, "temporal_lowpass", False) else None
        self.motion_head = MotionHead(cfg.dim) if getattr(cfg, "motion_head", False) else None
        self.freq_branch = FreqBranch(cfg.dim) if getattr(cfg, "freq_branch", False) else None

        # V2-28 轻量化条件编码器（定案：低秩 r384 + 6层，纯 torch SDPA）
        # 风格编码器：输出的 route_logits 直接复用全局 MoE 的 style_logits 接口；
        # style_cond 零门控加法注入 cond（gate=0 严格恒等）。
        self.style_encoder = None
        if getattr(cfg, "style_encoder", False):
            from cond_encoders import HierarchicalStyleEncoder
            self.style_encoder = HierarchicalStyleEncoder(
                dim=cfg.dim, num_heads=cfg.num_heads,
                num_experts=getattr(cfg, "moe_num_experts", 3),
                n_layers=getattr(cfg, "cond_layers", 6),
                rank=getattr(cfg, "cond_rank", 384),
            )
        # 参考对齐编码器：替换原 ref_proj 旁路；输入由 ref_latent patchify 得 token，
        # 输出零门控加法注入 cond（ref_latent=None 时整条路不进图 → 恒等）。
        self.ref_align_encoder = None
        if getattr(cfg, "ref_align_encoder", False):
            from cond_encoders import ReferenceResampleAlignEncoder
            self.ref_align_encoder = ReferenceResampleAlignEncoder(
                dim=cfg.dim, num_heads=cfg.num_heads,
                n_layers=getattr(cfg, "cond_layers", 6),
                rank=getattr(cfg, "cond_rank", 384),
            )
        self.local_conv = LocalConvFFN(cfg.dim) if getattr(cfg, "local_conv", False) else None

        self.head = OutputHead(cfg.dim, cfg.patch_size, cfg.out_channels, cfg.aux_head,
                               head_adaln=getattr(cfg, "head_adaln", False),
                               detail_head=getattr(cfg, "detail_head", False),
                               detail_hidden=getattr(cfg, "detail_hidden", 19456))
        # V2-31：细节头 ckpt 开关与主干保持一致（构造时 cfg 可能已开）
        self.head.detail_ckpt = bool(self.grad_checkpointing)

    # ------------------------------------------------------------------
    # 梯度检查点（V2-23）
    # 只削减「前向激活」显存：不减少权重/梯度/优化器状态。
    # 21.46GB 卡上 4.5B 全参的瓶颈是 权重9.07+梯度9.07=18.14GB，
    # 检查点救不了这部分，但能把 token 数上限显著抬高（配合 Adafactor）。
    # ------------------------------------------------------------------
    def _set_ffn_ckpt(self, enabled: bool):
        for _blk in list(self.enc_blocks) + list(self.bottleneck) + list(self.dec_blocks):
            if isinstance(_blk.ffn, _FFNCheckpoint):
                _blk.ffn.enabled = enabled

    def enable_gradient_checkpointing(self) -> "UViT3D":
        self.grad_checkpointing = True
        if self.ckpt_scope == "ffn_only":
            self._set_ffn_ckpt(True)
        self.head.detail_ckpt = True       # V2-31：细节头同步纳入
        return self

    def disable_gradient_checkpointing(self) -> "UViT3D":
        self.grad_checkpointing = False
        self._set_ffn_ckpt(False)
        self.head.detail_ckpt = False      # V2-31：细节头同步关闭
        return self

    def _blk_call(self, blk: nn.Module, z: torch.Tensor,
                  cond: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """block 前向：按 ckpt_scope 决定重算粒度。
          - "block"（默认）：整块套 checkpoint（GPU 实测省 ~1.1GiB 激活）；
          - "ffn_only"：整块不套，改由 block 内 FFN 子层自套（SelfAttn 不重算）。
        均用 use_reentrant=False（支持非 tensor grid / 闭包 cond 梯度 / 冻结分支）。
        """
        if not (self.grad_checkpointing and torch.is_grad_enabled()):
            return blk(z, cond, grid)
        if self.ckpt_scope == "ffn_only":
            return blk(z, cond, grid)   # FFN 子层内部已套 checkpoint
        return _ckpt.checkpoint(
            blk, z, cond, grid,
            use_reentrant=False, preserve_rng_state=False,
        )

    def _cond(self, t_emb: torch.Tensor, text_emb: torch.Tensor,
              vlm_cond: Optional[torch.Tensor] = None,
              sigma: Optional[torch.Tensor] = None,
              ref_latent: Optional[torch.Tensor] = None,
              grid: Optional[Tuple[int, int, int]] = None,
              style_cond: Optional[torch.Tensor] = None,
              cur_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """融合时间步 + 文本 + VLM 全局语义 -> 共享 embedder 输出 e [B,dim]。

        Returns (e, snr_gates):
          e: 全块共享的 6 段调制基（教师式 adaLN）
          snr_gates: [B, n_blocks] 或 None（SNR 分层时逐块缩放条件粒度）
        style_cond / cur_tokens 为 V2-28 两个轻量编码器的可选输入，None 时整条路不进图。
        """
        t = self.t_embed(t_emb)
        c = self.cond_proj(text_emb)
        cond = t + c
        if self.vlm_injector is not None and vlm_cond is not None:
            cond = cond + self.vlm_injector.global_cond(vlm_cond)
        # V2-28 分层风格条件编码器的加法输出（零门控起步，gate=0 严格恒等）
        if style_cond is not None:
            cond = cond + style_cond
        # V2-28 参考帧重采样对齐编码器：替换原 ref_proj 轻量旁路。
        # ref_latent=None 时整条路不进图（恒等）；给定时用 6 层 cross-attn 重采样对齐全。
        if ref_latent is not None:
            ref_tok = self.patchify(ref_latent).flatten(2).transpose(1, 2)  # [B,Nr,dim]
            if self.ref_align_encoder is not None:
                ctx = cur_tokens if cur_tokens is not None else ref_tok
                cond = cond + self.ref_align_encoder(ctx, ref_tok)
            elif self.ref_align:
                # 旧轻量旁路（向后兼容 tiny/无新编码器配置）
                cond = cond + torch.tanh(self.ref_gate) * self.ref_proj(self.ref_norm(ref_tok.mean(dim=1)))
        e = self.ada_embedder(cond)                    # 共享两层 MLP -> [B, dim]
        # SNR 分层：sigma 是流匹配噪声级（t_emb 通常由 sigma 编码），无额外标注。
        # 高噪声阶段给粗粒度条件、低噪声阶段给细粒度：用 sigma 经小 MLP 出逐块 gate。
        snr_gates = None
        if self.snr_cond:
            sig = sigma if sigma is not None else t_emb.mean(dim=-1, keepdim=True)
            if sig.dim() == 1:
                sig = sig.unsqueeze(-1)
            snr_gates = torch.sigmoid(self.snr_mlp(sig))   # [B, n_blocks]
        return e, snr_gates

    def _block_cond(self, e: torch.Tensor, snr_gates: Optional[torch.Tensor],
                    idx: int) -> torch.Tensor:
        """取第 idx 块的实际条件（SNR 分层时逐块缩放）。"""
        if snr_gates is None:
            return e
        g = snr_gates[:, idx:idx + 1]                   # [B,1]
        return e * g

    def _moe_block_cond(self, e: torch.Tensor, snr_gates: Optional[torch.Tensor],
                        idx: int, c_expert: Optional[torch.Tensor]) -> torch.Tensor:
        """V2-27：在 [inject_start, inject_end] 的 block 上加法注入全局专家条件。

        c_expert = α · top1_expert(e)，α=0 起步 → 严格逐 bit 恒等（回退红线）。
        注入是加法而非 FFN 替换：主干 FFN 恒定保留，无死参数、无 3×FFN 算力。
        """
        cond = self._block_cond(e, snr_gates, idx)
        if c_expert is not None and self.global_moe is not None:
            mo = self.global_moe
            if mo.inject_start <= idx <= mo.inject_end:
                cond = cond + c_expert
        return cond

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, text_emb: torch.Tensor,
                vlm_hidden: Optional[torch.Tensor] = None, return_aux: bool = False,
                sigma: Optional[torch.Tensor] = None,
                ref_latent: Optional[torch.Tensor] = None,
                style_logits: Optional[torch.Tensor] = None,
                style_tokens: Optional[torch.Tensor] = None):
        """
        x: [B, C, T, H, W] latent
        t_emb: [B, t_emb_dim]
        text_emb: [B, cond_dim]
        vlm_hidden: [B, L, vlm_dim] Qwen3-VL hidden（可选；给定时走多尺度分层注入）
        sigma: [B,1] 流匹配噪声级（SNR 分层条件用，None 时从 t_emb 推）
        ref_latent: [B,C,T',H',W'] 参考帧 latent（参考残差对齐用，None 时旁路恒等）
        style_logits: [B,E] VLM 风格头下发的 top1 路由 logits（None 时用 MoE 内部 router）
        style_tokens: [B,L,dim] 风格 token（V2-28 风格编码器输入；None 时该路恒等）
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.cfg.patch_size
        assert T % pt == 0 and H % ph == 0 and W % pw == 0, \
            f"输入 {x.shape} 必须能被 patch_size {self.cfg.patch_size} 整除"

        z = self.patchify(x)                          # [B, dim, T/pt, H/ph, W/pw]
        z = z.flatten(2).transpose(1, 2)              # [B, N, dim]
        grid = (T // pt, H // ph, W // pw)
        # 局部归纳偏置：patch token 上做 depthwise spatial conv（零门控起步）
        if self.local_conv is not None:
            z = self.local_conv(z, grid)

        # 北大桥只算一次：4096 -> dim，低分辨率用池化、高分辨率用 per-token
        vlm_cond = None
        if self.vlm_injector is not None and vlm_hidden is not None:
            vlm_cond = self.vlm_injector.bridge(vlm_hidden)   # [B, L, dim]

        # V2-28 分层风格条件编码器：风格 token -> 风格条件(加法入 cond) + 路由 logits(喂 MoE)。
        # style_tokens=None 时整条路不进图；外部若显式传 style_logits 则优先用它（覆盖编码器路由）。
        style_cond = None
        if self.style_encoder is not None and style_tokens is not None:
            style_cond, style_route = self.style_encoder(style_tokens)
            if style_logits is None:
                style_logits = style_route

        e, snr_gates = self._cond(t_emb, text_emb, vlm_cond, sigma, ref_latent, grid,
                                 style_cond=style_cond, cur_tokens=z)

        # V2-27 全局条件 MoE：全模型算一次 c_expert [B,dim]，再按 gidx 加法注入。
        # 处理的是条件向量（长度 dim=3072）而非 990 token，算力开销 ≈ 0。
        c_expert = None
        if self.global_moe is not None:
            c_expert = self.global_moe(e, style_logits=style_logits)

        # 全局 block 序号：encoder -> bottleneck -> decoder
        gidx = 0

        # encoder + skip 收集（低分辨率段：共享 embedder 输出 e 已含全局 cond）
        skip_feats = []
        for i, blk in enumerate(self.enc_blocks):
            z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert), grid)
            skip_feats.append(z)
            gidx += 1
            if self.merges is not None and i < len(self.merges):
                z, grid = self.merges[i](z, grid)

        # V2-29 纯时序专项块（主干中段，encoder 之后）：专治频闪/角色抖动/动作跳变
        if self.temporal_special is not None:
            # 走 _blk_call 纳入 checkpoint（此前直接调用 → 激活全程驻留漏网点）
            z = self._blk_call(self.temporal_special, z,
                               self._moe_block_cond(e, snr_gates, gidx, c_expert), grid)

        # bottleneck：完整 3D 联合注意力
        for blk in self.bottleneck:
            z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert), grid)
            gidx += 1

        # decoder + 加权 skip 融合（高分辨率段：per-token cross-attn 注入）
        for i, blk in enumerate(self.dec_blocks):
            if self.unmerges is not None and i < len(self.unmerges):
                z, grid = self.unmerges[i](z, grid)
            if i < len(skip_feats):
                z = self.skips[i](z, skip_feats[-(i + 1)])
            z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert), grid)
            if vlm_cond is not None:
                z = self.vlm_injector.inject_high(gidx, z, vlm_cond)
            gidx += 1

        # V2-29 纯空间专项块（深层，decoder 之后）：专磨单帧五官/线条，不增加时序计算
        if self.spatial_special is not None:
            # 走 _blk_call 纳入 checkpoint（此前直接调用 → 激活全程驻留漏网点）
            z = self._blk_call(self.spatial_special, z,
                               self._moe_block_cond(e, snr_gates, gidx, c_expert), grid)

        # 新增轻量算子（时序低通 / 频域 / 运动头）——都在最终 token 特征上施加
        if self.freq_branch is not None:
            z = self.freq_branch(z)
        if self.temporal_lowpass is not None:
            z = self.temporal_lowpass(z, grid)

        v, aux = self.head(z, grid, cond_e=e)
        # 运动头：帧级运动向量（辅助监督用，附在 return_aux 里一并返回，不进主速度场）
        if return_aux:
            if self.motion_head is not None:
                motion = self.motion_head(z, grid)
                return v, aux, motion   # 4b5: (v, aux, motion)
            return v, aux               # tiny/无运动头: 保持 2-tuple
        return v


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def disable_all_skips(model: UViT3D):
    """消融：关闭全部 skip 门控（§1.3 红线 8，退化为直筒对照）。"""
    for s in model.skips:
        s.disable()


def freeze_global_moe(model: UViT3D) -> "UViT3D":
    """阶段二：冻结全局 MoE 三专家，仅保留 router / α 可训（热启动后固定）。"""
    if model.global_moe is not None:
        model.global_moe.freeze()
    return model


def disable_global_moe(model: UViT3D) -> "UViT3D":
    """消融红线：把 α 置 0 → c_expert ≡ 0 → 主干严格回退到无专家形态。"""
    if model.global_moe is not None:
        with torch.no_grad():
            model.global_moe.alpha.zero_()
    return model


# ---------------------------------------------------------------------------
# 初始化 / 训练封装（F2 谱初始化 + F5 谱范数冻结）
# ---------------------------------------------------------------------------

def reinit_linear_teacher_spectrum(model: nn.Module, std: float = 0.02,
                                   skip_prefixes: Tuple[str, ...] = ("head_mod", "scale_shift_table",
                                                                      "gate", "log_threshold", "out_gate")):
    """F2: 把学生所有 nn.Linear 权重重初始化为 normal(0, std)。
    教师 M85 实测权重 std 0.017-0.028 (~N(0,0.02))，torch 默认 kaiming-uniform
    在 dim=3072 时 std≈0.0104，新模块起点比教师工作点矮 2×。
    跳过零初始化门控/阈值/ AdaLN 表（保持恒等起步语义）。
    """
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(name.endswith(p) or f".{p}" in name for p in skip_prefixes):
            with torch.no_grad():
                mod.weight.normal_(0, std)
                if mod.bias is not None:
                    mod.bias.zero_()
            n += 1
    return n


def freeze_spectral_power_iteration(model: nn.Module):
    """F5: 预热后冻结 spectral_norm 的 u/v 幂迭代，消除梯度检查点重算的
    不确定性和 torch.compile 断图点。PyTorch 里通过 parametrization 的
    use_parametrization=False 关闭（权重冻结为当前谱归一化值）。
    """
    n = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            try:
                mod.parametrizations.weight[0].use_parametrization = False
                n += 1
            except AttributeError:
                pass
    return n
