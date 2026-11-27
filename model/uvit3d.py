"""
PyraGate-VLM v2.0 — 3D U-DiT 主干骨架（"底子"）
==================================================
架构定位：单分辨率 U 形 DiT（U-DiT）。空间/时间 token 不做金字塔下采样，
但保留对称 encoder/bottleneck/decoder 与可学习 skip。
`uvit3d.py` / `UViT3D` 是历史文件名和类名，仅为兼容既有 checkpoint、
配置和 import 而保留；它们不代表当前模型是 U-ViT。
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
import os
import warnings
from contextlib import nullcontext
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt      # 梯度检查点（V2-23）

# RoPE 基频可切换：默认 10000 与原训练/ckpt 完全一致；A/B 时用环境变量
# SAKIKO_ROPE_THETA=1000/100，训练与采样必须在同一进程环境里设置。
ROPE_THETA = float(os.environ.get("SAKIKO_ROPE_THETA", "10000.0"))

try:
    from attn_backend import attention as dispatch_attention, backend_diagnose
except ImportError:                                  # 作为 package 导入时
    from .attn_backend import attention as dispatch_attention, backend_diagnose

# a2_v2 算力层：混合线性注意力（默认关闭，softmax 逐值等价主线）。
try:
    from linear_attn import HybridLinearAttention as _HybridLinearAttention
except ImportError:
    from .linear_attn import HybridLinearAttention as _HybridLinearAttention

# a2_v2 语义层：压缩跨模态 + 非对称特权（默认不实例化）。
try:
    from asymmetric_crossmodal import AsymmetricCrossModal as _AsymmetricCrossModal
except ImportError:
    from .asymmetric_crossmodal import AsymmetricCrossModal as _AsymmetricCrossModal

try:
    from full_gated_head import FullGatedTokenHead
except ImportError:                                  # 作为 package 导入时
    from .full_gated_head import FullGatedTokenHead

# PAU-DiT 语法平面：金字塔抽象语法层（默认不实例化，零门控恒等）。
try:
    from syntax_compiler import SyntaxCompiler as _SyntaxCompiler, tier_index as _tier_index
except ImportError:
    from .syntax_compiler import SyntaxCompiler as _SyntaxCompiler, tier_index as _tier_index

# VSA 块稀疏注意力（默认关闭；纯 torch 实现，ROCm 可跑，不依赖 CUDA kernel）。
try:
    from vsa_attention import vsa_sparse_attention as _vsa_sparse_attention
except ImportError:
    from .vsa_attention import vsa_sparse_attention as _vsa_sparse_attention

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
            # 不能用“值为 0 就 return x”短路：零初始化门控会因此丢图、永久拿不到梯度。
            # 与 sakiko_op/switch.py 的 native 分支保持一致：x + 0*branch 数值等价但保留梯度。
            return x + gate.to(x.dtype) * branch
        if float(gate) == 0.0:
            return x
        return x + gate * branch


# 小门控/零起步参数：训练时必须 fp32 + 独立优化器，否则会被 bf16 参数舍入
# 和 Adafactor 的 eps2 尺度下限一起冻死（5044 步 local_conv.gate 卡在 1/512）。
_GATE_NAME_SUFFIXES = ("scale_shift_table", "out_gate", ".gate")


def is_gate_param(name: str) -> bool:
    """判断是否属于必须 fp32 / 独立 optimizer 的门控参数。"""
    if any(name.endswith(s) for s in _GATE_NAME_SUFFIXES):
        return True
    # 2026-11-24 低频头新增的极小标量表：bf16 在 -2 / 0.01 量级的量子
    # 会把 Adafactor 的 2e-3 步长直接抹掉，必须 fp32。
    if name.startswith("full_head.high_gate."):
        return True
    # 2026-11-24 内嵌 Dual-Act：Rezero 标量/频带 Norm/Patch Norm 全部 fp32。
    if name.startswith(("full_head.dualact_gate.", "full_head.band_norm.",
                        "full_head.patch_norm.")):
        return True
    if (name.startswith("full_head.channel_gate.") or name.endswith("rpe.gamma")
            or name.endswith("out_gain")):
        return True
    if (name.startswith("skips.") or name.startswith("global_moe.")) and (
            name.endswith(".alpha") or name.endswith(".scale")):
        return True
    return False


def promote_gate_params_fp32(model) -> int:
    """把门控参数提升到 fp32（只改这些标量/小表，不动 4B 主干 bf16）。

    2026-11-26 追加：**所有 0 维参数**也一并升 fp32。
    原因：0 维 fp32 张量在 PyTorch 里按标量参与类型提升（bf16 * 0dim_fp32
    -> bf16，实测），所以升 fp32 不改前向 dtype；但不升的话，`alpha`
    (≈0.98)、`band_head_scale`(0.05)、`low_gate_logit` 这些 0 维量的 bf16
    量子高达 3.8e-3，而 gate/head 组 lr 只有 2.5e-4 → 每步位移不足半个
    量子，被 round-to-nearest 静默吞掉，实测 `alpha=0.00e+00(0.0q)`。
    """
    n = 0
    for name, p in model.named_parameters():
        if (is_gate_param(name) or p.dim() == 0) and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


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
    # 按 x dtype 计算：x 为 bf16 时走 bf16 快速路径（实测 3.6×，见审查报告 R1）
    xf = x.to(cos_g.dtype) if x.dtype in (torch.bfloat16, torch.float16) else x.float()
    y = xf * cos_g + rotate_half(xf) * sin_g
    y = y.to(x.dtype)
    return y.squeeze(0) if squeezed else y


@lru_cache(maxsize=32)
def _rope_grid_table_runtime(max_seq_len: int, head_dim: int, theta: float,
                             grid: Tuple[int, int, int], device_type: str,
                             device_index: int, dtype_name: str):
    """Cache the per-grid, device-resident RoPE tables.

    Before this cache, every attention block rebuilt an [1,N,1,d] cat + a
    repeat_interleave for both cos and sin on every forward step. At N=990,
    30 blocks that is a steady stream of small allocations and copies with no
    numerical effect. The cache is keyed by grid/device/dtype, so switching
    resolution or device cannot silently reuse the wrong table.
    """
    cos_cpu, sin_cpu = _rope_table_factory(max_seq_len, head_dim, theta)
    dtype = getattr(torch, dtype_name)
    device = torch.device(device_type) if device_index < 0 else torch.device(
        device_type, device_index)
    cos = cos_cpu.to(device=device, dtype=dtype)
    sin = sin_cpu.to(device=device, dtype=dtype)
    T, H, W = grid
    N = T * H * W
    c = head_dim // 2
    c3 = c // 3
    ct, ch, cw = cos.split([c - 2 * c3, c3, c3], dim=-1)
    st, sh, sw = sin.split([c - 2 * c3, c3, c3], dim=-1)

    def grid_table(tab_t, tab_h, tab_w):
        return torch.cat([
            tab_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1),
            tab_h[:H].view(1, H, 1, -1).expand(T, H, W, -1),
            tab_w[:W].view(1, 1, W, -1).expand(T, H, W, -1),
        ], dim=-1).reshape(1, N, 1, c).repeat_interleave(2, dim=-1)

    return grid_table(ct, ch, cw), grid_table(st, sh, sw)


def rope_apply_grid(x: torch.Tensor, cos_g: torch.Tensor, sin_g: torch.Tensor):
    """Apply precomputed grid RoPE tables to [B,N,heads,head_dim]."""
    squeezed = x.dim() == 3
    if squeezed:
        x = x.unsqueeze(0)
    xf = x.to(cos_g.dtype) if x.dtype in (torch.bfloat16, torch.float16) else x.float()
    y = (xf * cos_g + rotate_half(xf) * sin_g).to(x.dtype)
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

    教师 Wan 的 blocks.N.modulation 形状 (1,6,3072) 与此处逐块表同构 → 热启动可直拷。
    """
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6, segments: int = 6):
        super().__init__()
        self.dim = dim
        self.segments = segments
        # R8: 两层 MLP（教师同款；原单层 Linear(t_emb_dim,dim)+SiLU）
        # 2026-11-23 结构对齐：教师是 time_projection.1 = Linear(3072, 18432=6*dim)，
        # 每块再叠加 blocks.N.modulation (1,6,dim)。原来只输出 1 个 dim 向量再广播到
        # 6 段，shift/scale/gate 被迫完全相同（教师实测 6 段近乎正交，cos -0.14~0.83）。
        self.embedder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, segments * dim),
        )
        nn.init.zeros_(self.embedder[-1].weight)
        nn.init.zeros_(self.embedder[-1].bias)

    def forward(self, cond_raw: torch.Tensor) -> torch.Tensor:
        """cond_raw: [B, cond_dim] -> e: [B, dim]（共享，全块复用）。"""
        return self.embedder(cond_raw)


MLP_RATIO = 8 / 3   # 原生 FFN(mlp_ratio=4, 2 矩阵) -> SwiGLU(8/3, 3 矩阵) 参数持平
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
        return self.alpha.to(cond.dtype) * c                                         # α=0 → 严格 0 向量

    def freeze(self) -> "GlobalConditionalMoE":
        """冻结三专家权重，仅保留 router / α 可训（专家热启动后固定）。"""
        for p in self.experts.parameters():
            p.requires_grad_(False)
        return self


# ---------------------------------------------------------------------------
# 注意力：2+1D 分解（§3.4）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2026-11-27 v3.3：分组块对角投影（G=2/4）。输入/输出按组切分，组内全秩，
# 组间零交互；用于替代“全局低秩”作为注意力主路的压缩底座。
# ---------------------------------------------------------------------------
class BlockDiagLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int):
        super().__init__()
        if groups <= 0 or in_features % groups or out_features % groups:
            raise ValueError(f"BlockDiagLinear: dim {in_features}->{out_features} "
                             f"not divisible by groups={groups}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        self.blocks = nn.ModuleList([
            nn.Linear(in_features // groups, out_features // groups, bias=False)
            for _ in range(groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.groups, dim=-1)
        return torch.cat([blk(ch) for blk, ch in zip(self.blocks, chunks)],
                         dim=-1)


class GroupedQKVLinear(nn.Module):
    """Q/K/V 各自分组块对角，输出顺序仍为 [Q,K,V]。"""

    def __init__(self, dim: int, groups: int = 2):
        super().__init__()
        self.q_proj = BlockDiagLinear(dim, dim, groups)
        self.k_proj = BlockDiagLinear(dim, dim, groups)
        self.v_proj = BlockDiagLinear(dim, dim, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x),),
                         dim=-1)


class SplitQKVLinear(nn.Module):
    def __init__(self, dim: int, q_rank: int = 0, k_rank: int = 0,
                 v_rank: int = 0):
        super().__init__()
        from text_cross_attn import LowRankLinear

        def _make(rank: int) -> nn.Module:
            if rank > 0:
                return LowRankLinear(dim, dim, min(rank, dim), bias=False)
            return nn.Linear(dim, dim, bias=False)

        self.q_proj = _make(int(q_rank))
        self.k_proj = _make(int(k_rank))
        self.v_proj = _make(int(v_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x),),
                         dim=-1)


class SelfAttention(nn.Module):
    """通用多头自注意力（纯 torch SDPA，CPU 可跑），带可选 3D RoPE。
    输入 x: [B, N, C]，其中 N 为该注意力域内的 token 数。
    rope_grid 给定时，对 q/k 施加 t/h/w 分频 RoPE。
    """
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6,
                 causal: bool = False, use_rope: bool = True, rope_max_len: int = 2048,
                 attn_backend: str = "auto",
                 attn_mode: str = "softmax", hybrid_chunk: int = 132,
                 hybrid_fmap: str = "elu", vsa_keep_ratio: float = 0.25):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        # attn_mode ∈ {softmax, linear, hybrid, vsa}，默认 softmax 逐值等价主线。
        assert attn_mode in ("softmax", "linear", "hybrid", "vsa"), f"未知 attn_mode: {attn_mode}"
        self.vsa_keep_ratio = float(vsa_keep_ratio)
        self.attn_mode = attn_mode
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        # "flash2" = 真 FA2 优先（flash_attn 包 / SDPA FLASH backend），
        # CPU 或 fp32 自动降级 math（不伪装成 FA2）。详见 attn_backend.py。
        self.attn_backend = attn_backend
        self.use_rope = use_rope and (self.head_dim % 2 == 0)
        self.rope_max_len = rope_max_len
        # 固定 2D 局部注意力先验（非参数、不改变 ckpt）：仅由 UViT3D 在最后
        # 2~3 个 decoder block 上打开。softmax 前减去 s*(Δh²+Δw²)，让邻居 token
        # 在 attention 内部真正互相写入，而不是输出端低通。
        self.local_bias_strength = 0.0
        # v3 可选局部窗口：仅 spatial 头设置；0=关。加性 mask，不省 FLOPs，
        # 只改变可见范围（anchor token 仍全局），用于质量消融。
        self.local_window = 0
        # freqformer：外部挂载的 FreqRouter（None=关闭，前向与基线逐位等价）。
        self.router: Optional[nn.Module] = None
        # L0: qkv 合并成单个 GEMM（核数 3->1，权重排布变化）；
        # 保留 self.q/self.k/self.v 属性名指向视图切片以兼容 selftest §2 直接引用。
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        # B1 根治：不再注册 complex buffer（.to(bf16) 会静默丢虚部）。
        # 实数 cos/sin 表由 _rope_table_factory 模型级共享（R9），施加前按 q.dtype cast。
        # a2_v2：复用同一组 qkv 投影，只换聚合方式；softmax 模式不实例化 → 参数零增量。
        self.hybrid = (_HybridLinearAttention(num_heads, chunk_size=hybrid_chunk,
                                              feature_map_kind=hybrid_fmap,
                                              causal=causal, gate_init=0.0)
                       if attn_mode in ("linear", "hybrid") else None)

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
        if isinstance(self.qkv, (SplitQKVLinear, GroupedQKVLinear)):
            return (self.qkv.q_proj, self.qkv.k_proj, self.qkv.v_proj)[idx]
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

    def forward(self, x: torch.Tensor, rope_grid: Optional[Tuple[int, int, int]] = None,
                sigma: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)   # [B,nh,N,hd]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        # 3D RoPE：缓存后的实表直接驻留 device/dtype，避免每块每步重复 cat+repeat。
        if self.use_rope and rope_grid is not None:
            cos_g, sin_g = _rope_grid_table_runtime(
                self.rope_max_len, self.head_dim, ROPE_THETA, tuple(rope_grid),
                q.device.type, q.device.index if q.device.index is not None else -1,
                str(q.dtype).replace("torch.", ""))
            q = rope_apply_grid(q.transpose(1, 2), cos_g, sin_g).transpose(1, 2)
            k = rope_apply_grid(k.transpose(1, 2), cos_g, sin_g).transpose(1, 2)
        # 融合版跨模态：压缩摘要 K/V 拼到自注意力 key/value 尾部，让跨模态
        # 信息在每层注意力场内流动（Q 长度不变，KV 长度 N+M）。
        # cross_kv=None 时整段不进图，逐 bit 等价主线。
        cross_len = 0
        ck = cv = None
        k_self, v_self = k, v          # 供 VSA/hybrid 单独走稀疏/线性分支
        if cross_kv is not None:
            ck, cv = cross_kv
            M = ck.shape[1]
            cross_len = M
            ck = ck.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            cv = cv.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            ck = self.norm_k(ck)
            k = torch.cat([k, ck], dim=2)
            v = torch.cat([v, cv], dim=2)
        # freqformer：K 频域路由乘子 -> logits 加性 bias（None=关闭，不进图）。
        router_bias = None
        if self.router is not None:
            # FreqRouter 只接受 self 的 N=T*H*W token；cross_kv 拼接后长度变 N+M，
            # 直接喂 k 会触发 freq_router.py 的 N!=T*H*W 检查并静默退回恒等路由。
            # 路由乘子只作用于 self 部分，cross 段在后面用 pad(0) 补齐。
            _gates, _a_k = self.router(k_self, rope_grid, sigma)
            router_bias = torch.log(_a_k.clamp_min(1e-6)).to(q.dtype).view(
                B, self.num_heads, 1, N)
            if cross_len:
                router_bias = F.pad(router_bias, (0, cross_len), value=0.0)
        # v3 optional spatial window: anchor every `win` tokens is global;
        # all other queries see only the win x win neighbourhood.  Additive
        # mask only; it does not reduce SDPA FLOPs on ROCm.
        if (self.local_window and rope_grid is not None
                and len(rope_grid) == 3):
            tv_, th_, tw_ = rope_grid
            win = int(self.local_window)
            if th_ > 1 and tw_ > 1 and win > 0:
                ph = torch.arange(th_, device=x.device)
                pw = torch.arange(tw_, device=x.device)
                dh = (ph[:, None] - ph[None, :]).abs()
                dw = (pw[:, None] - pw[None, :]).abs()
                # anchor every win-th token on both axes
                ah = (ph % max(win, 1) == 0)
                aw = (pw % max(win, 1) == 0)
                anchor = ah[:, None] & aw[None, :]           # [H,W]
                anchor_flat = anchor.reshape(-1)
                # Build [N,N] directly: local in h and w independently.
                allowed = (dh[:, None, :, None] <= win // 2) & \
                          (dw[None, :, None, :] <= win // 2)
                allowed = allowed.reshape(th_ * tw_, th_ * tw_)
                allowed = allowed | anchor_flat[:, None]
                # repeat over T if joint grid; spatial head has T=1.
                allowed_t = allowed.unsqueeze(0).expand(tv_, -1, -1).reshape(
                    tv_ * th_ * tw_, tv_ * th_ * tw_)
                wb = torch.where(allowed_t, q.new_zeros(()),
                                 torch.full((), -1e9, device=x.device, dtype=q.dtype))
                wb = wb.view(1, 1, wb.shape[0], wb.shape[1])
                if cross_len:
                    wb = F.pad(wb, (0, cross_len), value=0.0)
                router_bias = wb if router_bias is None else router_bias + wb
        if self.local_bias_strength and rope_grid is not None:
            # 固定 2D 相对位置先验：s>0 时邻居 logits 更高。显式 matmul 是因为
            # flash_attn 不支持任意 attn_bias；只开在最后 3 个 decoder，N=990 可控。
            tv_, th_, tw_ = rope_grid
            s_bias = float(self.local_bias_strength)
            pos_t = torch.arange(tv_, device=x.device).repeat_interleave(th_ * tw_)
            pos_h = torch.arange(th_, device=x.device).repeat_interleave(tw_).repeat(tv_)
            pos_w = torch.arange(tw_, device=x.device).repeat(tv_ * th_)
            dt = (pos_t[:, None] - pos_t[None, :]).float()
            dh = (pos_h[:, None] - pos_h[None, :]).float()
            dw = (pos_w[:, None] - pos_w[None, :]).float()
            bias = -s_bias * (dt.pow(2) + dh.pow(2) + dw.pow(2))   # [N,N]
            b = bias.to(q.dtype).view(1, 1, N, N)
            if cross_len:
                b = F.pad(b, (0, cross_len, 0, cross_len), value=0.0)
            if self.causal:
                # 因果 + 加性 bias：上三角用 -inf 屏蔽
                causal_mask = torch.triu(torch.full((N, N), float("-inf"),
                                                   device=x.device, dtype=q.dtype), diagonal=1)
                if cross_len:
                    causal_mask = F.pad(causal_mask, (0, cross_len, 0, cross_len),
                                        value=float("-inf"))
                b = b + causal_mask.view(1, 1, N + cross_len, N + cross_len)
            if router_bias is not None:
                b = b + router_bias
            if _os.environ.get("SAKIKO_ATTN_MATH"):
                with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                    enable_math=True,
                                                    enable_mem_efficient=False):
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=b, scale=self.head_dim ** -0.5)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=b, scale=self.head_dim ** -0.5)
        elif router_bias is not None:
            # 路由 bias 走显式 SDPA（flash_attn 包不支持任意 attn_bias）。
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=router_bias, scale=self.head_dim ** -0.5)
        elif cross_kv is not None and self.attn_mode == "vsa":
            # 全栈启用修复：VSA 稀疏自注意力 + 摘要 KV 交叉注意力同时进图，
            # 不再因 cross_kv 短路而让加速模块变成死开关。
            if rope_grid is not None and len(rope_grid) == 3:
                y = _vsa_sparse_attention(q, k_self, v_self, tuple(rope_grid),
                                          topk_ratio=self.vsa_keep_ratio)
            else:
                y = _vsa_sparse_attention(q, k_self, v_self, (N, 1, 1),
                                          topk_ratio=self.vsa_keep_ratio)
            y = y + F.scaled_dot_product_attention(
                q, ck, cv, scale=self.head_dim ** -0.5)
        elif cross_kv is not None and self.attn_mode in ("linear", "hybrid"):
            y = self.hybrid(q, k_self, v_self,
                            use_softmax=(self.attn_mode == "hybrid"))
            y = y + F.scaled_dot_product_attention(
                q, ck, cv, scale=self.head_dim ** -0.5)
        elif cross_kv is not None:
            # softmax 模式：Q=N、KV=N+M，显式 SDPA 保证不等长支持。
            y = F.scaled_dot_product_attention(
                q, k, v, scale=self.head_dim ** -0.5)
        elif self.attn_mode == "vsa":
            # VSA 块稀疏注意力：q/k/v 已是 [B,nh,N,hd]；用 RoPE 网格做 3D tile 切分。
            # 无 grid 时退化为 1D tile（topk_ratio=1.0 时严格等价稠密 SDPA）。
            if rope_grid is not None and len(rope_grid) == 3:
                y = _vsa_sparse_attention(q, k, v, tuple(rope_grid),
                                          topk_ratio=self.vsa_keep_ratio)
            else:
                y = _vsa_sparse_attention(q, k, v, (N, 1, 1),
                                          topk_ratio=self.vsa_keep_ratio)
        elif self.attn_mode != "softmax":
            # a2_v2 算力层：非 softmax 模式直接走 hybrid 并 return，绝不再预 dispatch
            # （BUG-1：先算全量 softmax 再覆盖 = 负优化）。local_bias/router_bias 已折进
            # 块内 softmax；这里把 residual bias 作为 attn_bias 传给 hybrid（可选）。
            y = self.hybrid(q, k, v, use_softmax=(self.attn_mode == "hybrid"))
        else:
            y = dispatch_attention(q, k, v, is_causal=self.causal,
                                   backend=self.attn_backend)
        y = y.transpose(1, 2).reshape(B, N, C)
        return self.o(y)


class SpatialAttention(nn.Module):
    """空间头：单帧画面构图/纹理。x [B, T*H*W, C], grid=(T,H,W)。
    把 token 重排为 [B*T, H*W, C]，在每帧内部做自注意力。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 attn_backend: str = "auto", attn_core: str = "softmax",
                 hybrid_chunk: int = 132, hybrid_fmap: str = "elu"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=False,
                                  attn_backend=attn_backend, attn_mode=attn_core,
                                  hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int],
                sigma: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = grid
        assert N == T * H * W, f"token 数 {N} 与 grid {grid} 不符"
        if cross_kv is not None:
            ck, cv = cross_kv
            M = ck.shape[1]
            # 全局摘要 [B,M,C] 复制到每个帧域 [B*T,M,C]
            ck = ck.unsqueeze(1).expand(B, T, M, C).reshape(B * T, M, C)
            cv = cv.unsqueeze(1).expand(B, T, M, C).reshape(B * T, M, C)
            cross_kv = (ck, cv)
        y = x.view(B, T, H * W, C).reshape(B * T, H * W, C)
        # 空间域：每帧内 (1,H,W)；RoPE 的 T 维取 1（帧内只用 h/w 分频）
        y = self.attn(y, rope_grid=(1, H, W), sigma=sigma, cross_kv=cross_kv)
        return y.view(B, T, H * W, C).reshape(B, N, C)


class TemporalAttention(nn.Module):
    """时序头：帧间运动/连续性。x [B, T*H*W, C], grid=(T,H,W)。
    把 token 重排为 [B*H*W, T, C]，在每个空间位置跨帧做注意力。
    causal=True 时加因果掩码（§3.4，抑制长序列跨帧乱跳）。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 causal: bool = False, attn_backend: str = "auto",
                 attn_core: str = "softmax", hybrid_chunk: int = 132,
                 hybrid_fmap: str = "elu"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=causal,
                                  attn_backend=attn_backend, attn_mode=attn_core,
                                  hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int],
                sigma: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = grid
        assert N == T * H * W
        if cross_kv is not None:
            ck, cv = cross_kv
            M = ck.shape[1]
            # 全局摘要 [B,M,C] 复制到每个空间位置域 [B*H*W,M,C]
            ck = ck.unsqueeze(1).expand(B, H * W, M, C).reshape(B * H * W, M, C)
            cv = cv.unsqueeze(1).expand(B, H * W, M, C).reshape(B * H * W, M, C)
            cross_kv = (ck, cv)
        # [B, T*H*W, C] -> [B, T, H, W, C] -> [B, H, W, T, C] -> [B*H*W, T, C]
        y = x.view(B, T, H, W, C).permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        # 时序域：每个空间位置 (T,1,1)；RoPE 只用 t 分频
        y = self.attn(y, rope_grid=(T, 1, 1), sigma=sigma, cross_kv=cross_kv)
        return y.view(B, H, W, T, C).permute(0, 3, 1, 2, 4).reshape(B, N, C)


class JointAttention(nn.Module):
    """完整 3D 联合注意力：瓶颈层保留，学全局运动（§3.4）。
    更接近教师 Wan 的 Full Attention 范式。"""
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True,
                 attn_backend: str = "auto", attn_core: str = "softmax",
                 hybrid_chunk: int = 132, hybrid_fmap: str = "elu"):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qk_norm, causal=False,
                                  attn_backend=attn_backend, attn_mode=attn_core,
                                  hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)

    def forward(self, x: torch.Tensor, grid: Optional[Tuple[int, int, int]] = None,
                sigma: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # 瓶颈层完整 3D：用 (T,H,W) 全分频 RoPE，最贴近教师
        return self.attn(x, rope_grid=grid, sigma=sigma, cross_kv=cross_kv)


class GSA(nn.Module):
    """Gated Spatial Attention（另一条可选路线）。
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
                 attn_backend: str = "auto", attn_core: str = "softmax",
                 hybrid_chunk: int = 132, hybrid_fmap: str = "elu"):
        super().__init__()
        self.spatial = SpatialAttention(dim, num_heads, qk_norm,
                                        attn_backend=attn_backend, attn_core=attn_core,
                                        hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)
        self.temporal = TemporalAttention(dim, num_heads, qk_norm,
                                          causal_temporal,
                                          attn_backend=attn_backend, attn_core=attn_core,
                                          hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)
        self.norm = RMSNorm(dim)
        # 逐 token 门控：由 dim 维条件向量投影到 1（可换成 per-token 特征）
        self.gate_proj = nn.Linear(cond_dim or dim, 1)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(1))  # 零初始化：初始近似恒等

    def forward(self, x: torch.Tensor, grid: Tuple[int, int, int],
                cond: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm(x)
        y_s = self.spatial(h, grid, cross_kv=cross_kv)
        y_t = self.temporal(h, grid, cross_kv=cross_kv)
        # 门控源：优先用 cond（[B,C]，或教师式逐块 6 段调制 [B,6,C]→取均值），
        # 否则用输入池化。2026-11-23：原实现遇到 3D cond 会在 .view(B,1,1) 处崩。
        if cond is None:
            src = h.mean(dim=1)
        else:
            src = cond.mean(dim=1) if cond.dim() == 3 else cond
        g = torch.sigmoid(self.gate_proj(src)).view(B, 1, 1)
        y = g * y_s + (1 - g) * y_t
        return x + torch.tanh(self.out_gate).to(x.dtype) * self.o_proj(y)


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
                 dual_act: bool = False, dual_branch_adaln: bool = False,
                 adaln_rank: int = 0,
                 attn_core: str = "softmax", hybrid_chunk: int = 132,
                 hybrid_fmap: str = "elu", hybrid_layers: str = "all"):
        super().__init__()
        assert attn_mode in ("spatial", "temporal", "joint", "gsa")
        assert cond_mode == "adaln", "V1 只用 AdaLN-Zero（§3.2，移除 token 拼接）"
        self.dim = dim
        self.attn_mode = attn_mode
        # a2_v2：attn_core=softmax/linear/hybrid 作用于注意力聚合；本块是否启用
        # 由 hybrid_layers 按块序号决定（'all' 或逗号分隔的 gidx）。默认 softmax。
        _core = attn_core if (hybrid_layers == "all" or
                              attn_mode in hybrid_layers.split(",")) else "softmax"
        _hk = dict(attn_core=_core, hybrid_chunk=hybrid_chunk, hybrid_fmap=hybrid_fmap)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        if attn_mode == "spatial":
            self.attn = SpatialAttention(dim, num_heads, qk_norm,
                                         attn_backend=attn_backend, **_hk)
        elif attn_mode == "temporal":
            self.attn = TemporalAttention(dim, num_heads, qk_norm, causal_temporal,
                                          attn_backend=attn_backend, **_hk)
        elif attn_mode == "gsa":
            self.attn = GSA(dim, num_heads, qk_norm, causal_temporal, cond_dim=dim,
                            attn_backend=attn_backend, **_hk)
        else:
            self.attn = JointAttention(dim, num_heads, qk_norm,
                                       attn_backend=attn_backend, **_hk)

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
        # 默认关（用户"轻量叠加只选一个"，本轮选 SNR 分层；保留接口供兜底微调）。
        self.dual_branch_adaln = bool(dual_branch_adaln)
        self.adaln_rank = int(adaln_rank)
        if self.dual_branch_adaln:
            if self.adaln_rank > 0 and self.adaln_rank < dim:
                # 低秩双分支：dim->r->3dim，参数从 2×3dim² 降到 2×(dim·r+r·3dim)
                self.scale_mlp = nn.Sequential(
                    nn.Linear(dim, self.adaln_rank), nn.SiLU(),
                    nn.Linear(self.adaln_rank, dim))
                self.shift_mlp = nn.Sequential(
                    nn.Linear(dim, self.adaln_rank), nn.SiLU(),
                    nn.Linear(self.adaln_rank, dim))
                for seq in (self.scale_mlp, self.shift_mlp):
                    nn.init.zeros_(seq[-1].weight); nn.init.zeros_(seq[-1].bias)
            else:
                self.scale_mlp = nn.Linear(dim, dim)
                self.shift_mlp = nn.Linear(dim, dim)
                nn.init.zeros_(self.scale_mlp.weight); nn.init.zeros_(self.scale_mlp.bias)
                nn.init.zeros_(self.shift_mlp.weight); nn.init.zeros_(self.shift_mlp.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                grid: Tuple[int, int, int],
                t_cond: Optional[torch.Tensor] = None,
                style_logits: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        ffn_modulation: Optional[torch.Tensor] = None,
        tier_mod: Optional[torch.Tensor] = None,
        cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        text_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, C]
        cond: [B, C] 时间步 emb + 文本 emb 的融合向量
        grid: (T, H, W)
        t_cond: 阶段2 VLM Slot 下发的逐 token Dual-Act 阈值（阶段1 None）
        style_logits: 阶段2 VLM 风格头下发的 MoE 路由信号（阶段1 None）
        ffn_modulation: a2_v2 语义层下发的 FFN 调制 [B,1,C] 或 [B,N,C]；None 恒等
        tier_mod: PAU 金字塔层级 AdaLN 增量 [B,C]；None 恒等（零初始化起步）
        """
        if tier_mod is not None:
            tm = tier_mod.to(cond.dtype)
            cond = cond + (tm.unsqueeze(1) if cond.dim() == 3 else tm)
            # AST tier strength: scene/actor tiers get a stronger text AdaLN
            # than part/pixel; zero tier_mod -> exactly 1.0 (identity).
            _tier_scale = 1.0 + torch.tanh(
                tier_mod.float().mean(dim=-1, keepdim=True)).to(x.dtype)
        else:
            _tier_scale = None
        # L1 教师式：cond 现在是共享 embedder 输出 e [B,C]（旧）或 [B,6,C]（教师同构）。
        # 每块加 (1,6,C) 小表得到本块独立的 6 段调制。初始表为 0 且 e=0 → gate=0 → 恒等。
        mods = self.scale_shift_table + (cond.unsqueeze(1) if cond.dim() == 2 else cond)
        if self.dual_branch_adaln:
            # 修复：教师式 mods 是 [B,6,dim] 六段（shift1/scale1/gate1/shift2/scale2/gate2）。
            # 原实现把 3*dim 输出与 dim 的 zeros 直接 stack → 形状不一致，从未启用故未暴露。
            # 现在独立 scale/shift 网络各输出 [B,dim]，分别加到两段 scale/shift。
            _c = cond if cond.dim() == 2 else cond.mean(dim=1)
            _sa = self.scale_mlp(_c).to(x.dtype)
            _sh = self.shift_mlp(_c).to(x.dtype)
            _z = torch.zeros_like(_sa)
            mods = mods + torch.stack([_sh, _sa, _z, _sh, _sa, _z], dim=1)
        # 表参数用 fp32 存/更新，进前向时统一 cast 回激活 dtype；避免 fp32 全链路。
        mods = mods.to(x.dtype)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.unbind(dim=1)

        # 注意力子层（AdaLN 调制）；GSA 额外吃条件向量做门控
        h = modulate(self.norm1(x), shift1, scale1)
        if self.attn_mode == "gsa":
            y = self.attn(h, grid, cond=cond, cross_kv=cross_kv)
        else:
            y = self.attn(h, grid, sigma=sigma, cross_kv=cross_kv)
        x = x + y * gate1.unsqueeze(1)

        # v3 text cross-attn：bottleneck 6 层默认全有；v3.1 可扩到
        # dec_blocks[:text_cross_decoder_layers]（decoder 最深层）。
        # 位置严格按方案：self-attn 之后、FFN 之前；中高频/输出头完全不动。
        _tc = getattr(self, "text_cross", None)
        if _tc is not None and text_tokens is not None:
            x = _tc(x, text_tokens, text_mask=text_mask, sigma=sigma,
                    text_valid=text_valid)
        # AdaLN-Zero 全局文本注入：并行 residual，零初始化恒等起步。
        _ta = getattr(self, "text_adaln", None)
        if _ta is not None and text_tokens is not None:
            x = _ta(x, text_tokens, text_mask=text_mask, sigma=sigma,
                    tier_scale=_tier_scale, text_valid=text_valid)
        # v3 bottleneck global modulation：cross-attn 之后、FFN 之前。
        _gm = getattr(self, "bottleneck_global_mlp", None)
        if _gm is not None:
            x = _gm(x)

        # FFN 子层（AdaLN 调制）：主干恒为 Dual-Act SwiGLU。
        # 全局条件 MoE 不在这里替换 FFN，而是在 UViT3D.forward 里加法注入 cond。
        z2 = modulate(self.norm2(x), shift2, scale2)
        if self.dual_act:
            y = self.ffn(z2, t_cond=t_cond)
        else:
            y = self.ffn(z2)
        # a2_v2 语义层：VLM→VLM→FFN 调制，按 (1+mod) 缩放 FFN 输出；None 严格恒等。
        if ffn_modulation is not None:
            m = ffn_modulation.to(x.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(1)
            y = y * (1.0 + m)
        x = x + y * gate2.unsqueeze(1)
        # σ 连续滑动 adapter：高σ才开，低σ g=0。
        # ffn_mod 已在上面作用于 FFN 输出；这里不再重复乘一次，避免同一信号
        # 同时控 FFN 和 adapter 造成双重门控。
        _sa = getattr(self, "sigma_adapter", None)
        if _sa is not None:
            x = _sa(x, sigma=sigma)
        return x


class SpecializedBlock(nn.Module):
    """V2-29 专项注意力 Block：只保留单一注意力域的轻量块。

    - mode='temporal'：只做帧间时间注意力（治频闪/角色抖动/动作跳变），插主干中段
    - mode='space'    ：只做单帧空间注意力（磨五官/线条），插深层
    - 外层零输出门控 gate=0 → 严格恒等（初始不扰动已训主干，训练中逐步学开）
    - 内部复用 UViTBlock（单注意力 + 标准 FFN），参数 ~113M/块

    为何用标准 FFN 而非缩小：用户要求「参数控制 300M 内」而非「尽量小」；
    两块标准容量 ~227M < 300M，既守住预算又不牺牲容量。
    """
    def __init__(self, dim: int, num_heads: int, mode: str,
                 attn_backend: str = "auto", causal_temporal: bool = False,
                 attn_core: str = "softmax", hybrid_chunk: int = 132,
                 hybrid_fmap: str = "elu", shared_ffn: Optional[nn.Module] = None):
        super().__init__()
        assert mode in ("temporal", "spatial"), f"专项块只支持 temporal/spatial，收到 {mode}"
        self.mode = mode
        self.block = UViTBlock(dim, num_heads, mode,
                               attn_backend=attn_backend,
                               causal_temporal=causal_temporal,
                               attn_core=attn_core,
                               hybrid_chunk=hybrid_chunk,
                               hybrid_fmap=hybrid_fmap)
        if shared_ffn is not None:
            # 全栈启用：专项块只保留自己的注意力，FFN 复用主干相邻块。
            # 释放 2×75.5M 参数（≈151M）；零输出门控起步仍严格恒等。
            # （曾试验 blend 可选模式保留本块 FFN，已撤回。）
            self.block.ffn = shared_ffn
        self.out_gate = nn.Parameter(torch.zeros(1))     # 零起步：严格恒等

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                grid: Tuple[int, int, int],
                sigma: Optional[torch.Tensor] = None,
                ffn_modulation: Optional[torch.Tensor] = None,
                tier_mod: Optional[torch.Tensor] = None,
                cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                text_tokens: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                text_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.block(x, cond, grid, sigma=sigma,
                       ffn_modulation=ffn_modulation, tier_mod=tier_mod,
                       cross_kv=cross_kv, text_tokens=text_tokens,
                       text_mask=text_mask, text_valid=text_valid)
        return x + torch.tanh(self.out_gate).to(x.dtype) * (y - x)  # gate=0 → 逐 bit 恒等

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
        # 2026-11-23：skip 支路先做固定 3×3 可分离高斯低通，再乘可学习 scale。
        # 目的：抹平 token 网格上 32px patch 边界的高频跳变，避免 skip 直接把
        # encoder 的块状高频灌进 decoder；scale 初始 1，不破坏原起点。
        self.scale = nn.Parameter(torch.ones(1))
        kh = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        kw = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        self.register_buffer("_lp_h", kh.view(1, 1, 1, 3, 1).repeat(dim, 1, 1, 1, 1))
        self.register_buffer("_lp_w", kw.view(1, 1, 1, 1, 3).repeat(dim, 1, 1, 1, 1))

    def _lowpass(self, z: torch.Tensor, grid) -> torch.Tensor:
        B, N, C = z.shape
        T, H, W = grid
        zz = z.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        # 可分离高斯：先 H 后 W；groups=C，逐通道、不跨通道混色。
        zz = F.conv3d(zz, self._lp_h.to(zz.dtype), None, stride=1, padding=(0, 1, 0), groups=C)
        zz = F.conv3d(zz, self._lp_w.to(zz.dtype), None, stride=1, padding=(0, 0, 1), groups=C)
        return zz.permute(0, 2, 3, 4, 1).reshape(B, N, C)

    def forward(self, x_deep: torch.Tensor, skip_feat: torch.Tensor, grid=None) -> torch.Tensor:
        branch = self.proj(self.norm(skip_feat))
        if grid is not None:
            branch = self._lowpass(branch, grid)
        # 统一走门控入口：默认 SAKIKO_CUSTOM_OP 未设 → native。
        gate = self.alpha.to(x_deep.dtype) * self.scale.to(x_deep.dtype)
        return _gated_residual_op(x_deep, branch, gate, tag="gated_skip")

    def disable(self):
        """消融开关：门控置 0，等效关闭 skip。"""
        with torch.no_grad():
            self.alpha.zero_()


# ---------------------------------------------------------------------------
# 金字塔：时空 Patch Merge / Unmerge（§4.1）
# ---------------------------------------------------------------------------

_PYRAMID_WARNED: set = set()


def _pyramid_warn_once(stage: str, grid, s: int, t: int) -> None:
    """2-stage 金字塔网格不合法时的一次性告警（按 stage+网格去重，不静默吞错）。"""
    key = f"{stage}:{tuple(grid)}:s{s}:t{t}"
    if key in _PYRAMID_WARNED:
        return
    _PYRAMID_WARNED.add(key)
    warnings.warn(
        f"[pyramid] {stage} 网格 {tuple(grid)} 不满足 patch (s={s}, t={t}) 整除/序列长度要求，"
        f"已跳过该 2-stage 降级路径；cfg.pyramid 默认 False，仅显式开启时触发（每条网格仅告警一次）。",
        RuntimeWarning, stacklevel=2)


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
        ok = (T % self.t == 0 and H % self.s == 0 and W % self.s == 0
              and N == T * H * W)
        if not ok:
            # 非常规网格（非整除/奇数分辨率/短序列）：降级跳过本 stage，不硬崩；
            # 默认 cfg.pyramid=False，此路径默认不进图，行为不变。
            _pyramid_warn_once("merge", grid, self.s, self.t)
            return x, grid
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
        if N != T * H * W:
            # 与 merge 对称的兜底：网格/序列不一致时跳过本 stage，告警一次。
            _pyramid_warn_once("unmerge", grid, self.s, self.t)
            return x, grid
        x = self.proj(x)
        x = x.view(B, T, H, W, self.t, self.s, self.s, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, (T * self.t) * (H * self.s) * (W * self.s), C)
        return x, (T * self.t, H * self.s, W * self.s)


# ---------------------------------------------------------------------------
# 新增轻量算子：时序低通 / 运动头 / 频域分支 / 局部卷积
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
        # W5 注释修正：默认 init_identity=False，Conv1d 走常规初始化（3-tap 上界
        # 约 0.58），并非"严格恒等起步"；残差形式只保证量级温和。需要严格恒等
        # 时显式传 init_identity=True。
        return z + y


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
        return z + torch.tanh(self.gate).to(z.dtype) * self.net(spec).unsqueeze(1)


class LocalConvFFN(nn.Module):
    """depthwise spatial conv 局部增强（ConvNeXt/HunyuanVideo 验证）：
    在 FFN 前对 token 的 (H,W) 做 3x3 depthwise conv，补 Transformer 缺失的
    局部归纳偏置。参数 3072*9≈28K/块，30 块 <1M。零门控起步。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        # 2026-11-23 修正：原「中心=1、周围=0」不是低通，是纯恒等，门控打开也只是
        # 把 z 乘 1.02，根本没有邻域混合。改成归一化 3×3 高斯核（和=16），
        # gate 打开时才是真正的温和低通。
        with torch.no_grad():
            k = torch.tensor([[1.0, 2.0, 1.0],
                              [2.0, 4.0, 2.0],
                              [1.0, 2.0, 1.0]]) / 16.0
            self.dw.weight.zero_()
            self.dw.weight[:, 0, :, :] = k
        self.gate = nn.Parameter(torch.tensor([0.02]))

    def forward(self, z: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = z.shape
        T, H, W = grid
        y = z.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        y = self.dw(y).view(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, N, C)
        return z + torch.tanh(self.gate).to(z.dtype) * y

    def smooth(self, z: torch.Tensor, grid: Tuple[int, int, int],
               strength: float) -> torch.Tensor:
        """固定强度的 3×3 高斯邻域混合：z + strength * (Gaussian(z) - z)。

        与 gate 参数无关，不新增参数；用于最终 token 特征输出前的一次性平滑，
        直接压 patch 之间的均值跳变（32px 方块的直接来源），代价只有一次 conv。
        """
        B, N, C = z.shape
        T, H, W = grid
        y = z.view(B, T, H, W, C).permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        y = self.dw(y).view(B, T, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, N, C)
        return z + float(strength) * (y.to(z.dtype) - z)


class DetailRefineHead(nn.Module):
    """V2-30 DDT 式高频细节残差头（Motif-Video 2B 验证：低/高频分离解码提质量）。

    - 独立于主速度场头：主头管低频语义/构图，本头只补高频细节（五官/线条/纹理）
    - per-token MLP：dim → hidden → out，作用在最终 token 特征上（不进主干 30 块）
    - 残差从严格 0 起步，但梯度可达：down 权重/偏置置零，gate=1（tanh=0.762）
      旧版 gate=0 会让 down/up 梯度≈0，detail 永远学不动；这里只把“输出”置零，
      不把“梯度”置零。
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
        # 输出严格 0，但 gate 非零、down 可训练：避免旧版 gate=0 造成的双死梯度。
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.out_gate = nn.Parameter(torch.ones(1))        # tanh(1)=0.762，残差仍为 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,N,dim] → 细节残差 [B,N,out_dim]（初始化时 down=0，严格 0）。"""
        h = self.norm(x)
        h = F.silu(self.up(h))
        return torch.tanh(self.out_gate).to(h.dtype) * self.down(h)

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
                 detail_head: bool = False, detail_hidden: int = 19456,
                 out_lp_init: float = 0.0):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_channels)
        # 2026-11-24：全局输出低通已移除（收益差、副作用致命：压通道方差/对比度）。
        # out_lp_init 仅保留为旧调用签名兼容，不再创建参数或 buffer。
        self.out_lp_init = 0.0
        # 2026-11-23：主头零初始化（AdaLN-Zero 式「零预测起点」）。
        # 原实现用默认 init，再叠加 F2 的 std=0.02：对 3072→192 的 Linear，
        # 输出 std ≈ 0.02*sqrt(3072) ≈ 1.1，第一步 loss 实测 2.76（零预测仅 1.58）——
        # 模型还没开始学，就先自带一个幅值 ~1 的错误预测。
        # 零初始化后 step1 恒等输出 0，loss 恰等于零预测线，梯度仍能通过
        # dL/dW = dL/dout ⊗ h 更新主头（第二步起 trunk 才有梯度），是标准做法。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
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
        if _os.environ.get("SAKIKO_DEBUG_GRAD"):
            print(f"[dbg-head] x_rms={x.float().pow(2).mean().sqrt().item():.3e} "
                  f"h_rms={h.float().pow(2).mean().sqrt().item():.3e} "
                  f"W_max={self.head.weight.abs().max().item():.3e} "
                  f"v_max={v.float().abs().max().item():.3e}", flush=True)
        # V2-30 高频细节残差：gate=0 时严格 0，主头输出逐 bit 不变
        # V2-31：开启 checkpoint 时重算细节头激活（省 ~45MB 峰值，数值等价）
        self._last_detail_v = None
        if self.detail_head is not None:
            if self.detail_ckpt and torch.is_grad_enabled():
                dv = _ckpt.checkpoint(self.detail_head, x,
                                      use_reentrant=False, preserve_rng_state=False)
            else:
                dv = self.detail_head(x)
            v = v + dv
            # 供训练器做 detail 支路独立高频监督；保留计算图，不 detach。
            dv_raw = dv
        # 重排回 [B, out_c, T*p_t, H*p_h, W*p_w]
        # 2026-11-23 修复（实测 0.69% 槽位与原布局一致、且通道不纯）：
        # 旧写法把 pt*ph*pw 合并成一个 pp 轴再 permute 到 C 前面，pp 再也拆不回去，
        # 通道轴与 patch 内空间轴交叉错位 —— view 不报错，所以自检（只查形状）全绿。
        # 与教师 einsum('bfhwpqrc->bcfphqwr') 逐元素对齐的正确顺序：
        pt, ph, pw = self.patch_size
        v = v.view(B, T, H, W, pt, ph, pw, self.out_channels)
        v = v.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        v = v.reshape(B, self.out_channels, T * pt, H * ph, W * pw)
        if 'dv_raw' in locals():
            dv_raw = dv_raw.view(B, T, H, W, pt, ph, pw, self.out_channels)
            dv_raw = dv_raw.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            self._last_detail_v = dv_raw.reshape(
                B, self.out_channels, T * pt, H * ph, W * pw)
        aux = self.aux_head(h).view(B, T, H, W).squeeze(-1) if self.aux_head else None
        return v, aux


class LatentRefine(nn.Module):
    """unpatchify 后的 latent 空间 3×3 联合卷积残差。

    2026-11-24：不用全局低通。卷积核权重零初始化，残差从 0 起步；
    gate 初始 1（tanh≈0.762），保证 conv 权重第一步就能拿到梯度。
    只做空间 (1,3,3)，不混时间；跨 patch 边界做局部联合，不抹整块 latent。
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, kernel, kernel),
                              padding=(0, kernel // 2, kernel // 2), bias=False)
        nn.init.zeros_(self.conv.weight)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v + torch.tanh(self.gate).to(v.dtype) * self.conv(v)


def token_gradient_continuity(f: torch.Tensor,
                              channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Token-Boundary 梯度连续性损失（一阶差分版，无二阶 autograd）。

    f: [N, H, W, C]，N 为 B*T。约束相邻 token 的差分本身连续：
      diff_h[i] = F[i+1] - F[i]
      diff_w[j] = F[j+1] - F[j]
      loss = mean(||diff_h[i+1]-diff_h[i]||_2) + mean(||diff_w[j+1]-diff_w[j]||_2)

    channel_mask: [N,C,1,1]，只有低频通道参与（高频边缘允许跳变）。
    原始 v_tok 在 step200 的 RMS 只有 ~1e-2，192 维 L2 算出来是 1e-3 量级，
    乘 0.08 等于 1e-4，实际不约束。这里按全图 RMS 归一化并按 sqrt(C) 除一次，
    使权重落在可感知量级，且不改变输出幅度。
    RMS 用 detach 且下限 0.02：避免输出接近 0 时 1/rms 把梯度放大成 NaN。
    """
    f = f.float()
    if channel_mask is not None:
        f = f * channel_mask.permute(0, 2, 3, 1).float()
    if f.shape[1] < 3 or f.shape[2] < 3:
        return f.new_zeros(())
    rms = f.pow(2).mean().sqrt().detach().clamp_min(0.02)
    f = f / rms
    dh = f[:, 1:, :, :] - f[:, :-1, :, :]
    dw = f[:, :, 1:, :] - f[:, :, :-1, :]
    dhh = dh[:, 1:, :, :] - dh[:, :-1, :, :]
    dww = dw[:, :, 1:, :] - dw[:, :, :-1, :]
    inv_c = float(f.shape[-1]) ** -0.5
    lh = dhh.pow(2).sum(dim=-1).add(1e-8).sqrt().mean() * inv_c
    lw = dww.pow(2).sum(dim=-1).add(1e-8).sqrt().mean() * inv_c
    return 0.5 * (lh + lw)


# ---------------------------------------------------------------------------
# 主模型：单分辨率 U 形 DiT（U-DiT；历史文件名/类名 UViT3D 保留兼容）
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
    # GSA（Gated Spatial Attention）为另一条可选路线，与 2+1D 并列
    attn_pattern: Tuple[str, ...] = ("spatial", "temporal")
    attn_strategy: str = "alternate"     # "alternate"=2+1D 交替 | "gsa"=门控混合
    causal_temporal: bool = False
    # 注意力后端："flash2"（真 FA2 优先），CPU 自动降级 math。
    # 可选 "auto"/"flash2"/"sdpa"/"math"；详见 attn_backend.py。
    attn_backend: str = "flash2"
    # a2_v2 算力层（默认全关，softmax 逐值等价主线；GPU 门禁 ≥1.5x & cos≥0.9 才启用）。
    # attn_core: softmax / linear / hybrid；hybrid_layers: "all" 或逗号分隔 block 序号。
    attn_core: str = "softmax"
    hybrid_layers: str = "none"
    hybrid_chunk: int = 132          # 局部 softmax 块长（≈一块空间 token 邻域量级）
    hybrid_fmap: str = "elu"         # 线性分支特征映射 elu/relu/relu2
    # a2_v2 语义层：压缩跨模态 + 非对称特权（默认不实例化）。
    a2_crossmodal: bool = False
    a2_crossmodal_slots: int = 32
    # True=压缩摘要 K/V 进每层注意力场（融合版）；False=旧一次性 token 融合（向后兼容）
    a2_crossmodal_in_attn: bool = False
    # PAU-DiT 语法平面（默认关，零门控恒等；开时保 step3200 热启动）。
    pau_syntax_enabled: bool = False
    pau_tiers: int = 4            # scene / actor / part / pixel
    pau_rank: int = 64
    pau_gate_init: float = 0.01   # syntax 主 gate 非零，解锁零初始化 heads
    pau_skip_threshold: float = 0.0
    # VSA 块稀疏注意力（默认关；softmax 逐值等价主线）。
    vsa_keep_ratio: float = 0.25      # 保留的 tile 比例（0.25 = 稀疏 75%）
    # 梯度检查点：只砍前向激活显存，不减少权重/梯度/优化器状态（V2-23）。
    # 显存紧张时开启；推理恒关闭（推理无梯度，无收益只增开销）。
    grad_checkpoint: bool = False
    # 检查点粒度（V2-25）："block"=整块重算（默认，GPU 实测省 ~1.1GiB）；
    # "ffn_only"=只重算 FFN/CrossAttn、SelfAttn 不重算（省重算开销，但激活显存收益未实测，
    #   因为 FlashAttn 的 SelfAttn 激活本就小，关掉其重算可能反而增峰——谨慎用）。
    ckpt_scope: str = "block"
    # v3 精细化梯度检查点：bottleneck / 最后 N 个 decoder 可单独关闭。
    # 默认不关（与旧行为逐位一致）；只有显式设置才改变重算范围。
    ckpt_skip_bottleneck: bool = False
    ckpt_last_dec_off: int = 0
    # v3 全主干结构化低秩：0=关闭（逐位等于全秩）。rank>0 时把
    # self-attn 的 qkv/o 与 SwiGLU 的 w12/w3 换成 LowRankLinear。
    # 注意：rank=64/128 会把 4B 主干砍到 ~0.5B；rank=1536 才对应
    # 约 30~40% 的参数下降。仅从零训练使用，旧 ckpt 不可 strict 载入。
    band_lr_rank: int = 0
    band_lr_scope: str = "all"   # all / attn / ffn
    # 分层 rank: 0=沿用 band_lr_rank；>0 覆盖该段。深层语义层通常需要更高秩。
    band_lr_rank_enc: int = 0
    band_lr_rank_bottleneck: int = 0
    band_lr_rank_dec: int = 0
    # 模块级独立 rank：0=沿用上面的段 rank/全局 rank。
    # 典型配置：attn=1536 锚定稳定性，ffn=512；FFN 再按 enc/bn/dec 递减。
    band_lr_rank_attn: int = 0
    band_lr_rank_ffn: int = 0
    band_lr_rank_ffn_enc: int = 0
    band_lr_rank_ffn_bottleneck: int = 0
    band_lr_rank_ffn_dec: int = 0
    # 2026-11-27 v3.2：Q/K 与 V/O 分开低秩；0=该分支保持 dense。
    # 只有 vo>0 且 qk=0 时才是真正“Q/K 全秩 + V/O 压缩”的省参数方案。
    band_lr_qk_rank: int = 0
    band_lr_vo_rank: int = 0
    # 2026-11-27 v3.3：注意力 Q/K/V/O 分组块对角组数；0=关（走 dense/低秩）。
    band_lr_attn_groups: int = 0
    # 模块级开关：可只低秩注意力或只低秩 FFN。
    band_lr_attn: bool = True
    band_lr_ffn: bool = True
    # 架构图主干结构（V2-24）：Dual-Act 分段激活 SwiGLU（可训）+ MoE 三专家（阶段1旁路）
    dual_act: bool = False         # 阶段1从零打底：先关 Dual-Act（负侧阈值处有≈1.0 跳变），
                                   # 主干收敛后再置 True 做消融对照
    # V2-27 全局条件空间 MoE（speedrun 定版）：全模型一套、吃 cond [B,dim]、top1、α 注入
    use_moe: bool = False          # 阶段1 不启用；阶段2 设 True 才有 ~1B 全局专家
    moe_num_experts: int = 3
    moe_hidden_mult: float = 4.375 # 专家 hidden = dim × 该值（dim=3072→13440 → 单专家≈330M）
    moe_n_layers: int = 4          # 单专家 MLP 层数（4 层 → 各 ≈330M）
    moe_inject_start: int = 7      # α 门控注入起始 block（全局序号 0..N-1）
    moe_inject_end: int = 12       # 注入结束 block（含），共 6 块
    moe_style_dim: int = 0         # VLM 风格头维度（0=用内部 router；>0=预留外接风格 logits）
    # VAE：用 AMD 蜂鸟本地轻量版（TurboVAED 同源，14×/4×），保持 fp16 不量化
    vae_path: str = "./cache/Hummingbird-Wan2.2-VAE_bf16.safetensors"
    # skip
    skip_alpha_init: float = 0.0   # 从零训练：严格恒等起步（0.1 会往 decoder 混 10% 随机 skip）
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
    # 轻量算子（tiny 自检默认关以免扰动逐 bit 断言）
    temporal_lowpass: bool = False                  # 时序低通 conv（教师 flicker 关键，~9K）
    motion_head: bool = False                       # 帧级运动辅助头（<2M）
    freq_branch: bool = False                       # 频域分支（<2M）
    local_conv: bool = False                        # depthwise spatial conv 局部增强（<1M）
    local_every_dec_block: bool = False             # 每个 decoder block 后复用 local_conv（无新参数）
    local_out_strength: float = 0.1                 # 最终 token 特征输出前的固定高斯平滑强度（0=关）
    out_lp_init: float = 0.0                        # 已废弃：全局输出低通已移除，仅保留字段兼容旧配置
    latent_refine: bool = False                     # unpatchify 后 3×3 联合卷积残差（零卷积起步）
    # 固定局部注意力先验：最后 N 个 decoder block 的 softmax logits 减 s*(Δh²+Δw²)，
    # 让 attention 本身偏向邻居 token；非参数、不改 ckpt。
    attn_local_bias: float = 0.0
    attn_local_last_n: int = 3
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
    # 2026-11-24 FullGatedHead-RPE-Sobel-AA-Pyramid：新的 token 空间输出头。
    # 与旧 LinearHead 不共享权重；从 step200 起新开完整实验。
    full_gated_head: bool = False
    full_head_groups: int = 4
    full_head_compress: int = 4
    full_head_ctx_dim: int = 128
    full_head_disable: Tuple[str, ...] = ()
    # 2026-11-26 freqformer 线：轻量频域路由（默认关；关闭时与基线逐位等价）。
    # 不新增三套 QKV，只在已有 K 上做低秩三频带路由；参数前缀 freqrouter.。
    freq_route: bool = False
    freq_route_rank: int = 8
    freq_route_scale: float = 1.0
    freq_route_topk: int = 0
    # 2026-11-27 v3：bottleneck 文本 cross-attn（仅 6 个 bottleneck block）。
    # 默认全关；开启时旧 ckpt strict=False 直载，新增键前缀 bottleneck.N.text_cross.*。
    text_cross_attn: bool = False
    text_cross_rank: int = 0          # 0=全秩 Q/O；>0=BandLR 式 Q/O 低秩（K/V 仍全秩）
    text_cross_base_gate: float = 0.1 # gate=base_gate*sigma，clamp 0..0.2
    text_cross_text_dim: int = 3072   # 官方 text_embedder 投影后的 UMT5 宽度
    text_cross_scale_alpha: float = 0.0  # 噪声感知文本缩放：token*(1+alpha*sigma)
    text_cross_freeze_kv: bool = True
    text_cross_init_wan: str = ""     # 官方 5B diffusers 目录；空=不自动拷贝
    text_cross_src_blocks: Tuple[int, ...] = (12, 13, 14, 15, 16, 17)
    adaln_text_layers: Tuple[int, ...] = ()  # decoder 浅层 AdaLN-Zero 文本注入索引
    adaln_text_gate_hidden: int = 256
    # 2026-11-27 v3.1：把 text cross-attn 扩到最靠近 bottleneck 的 N 个 decoder
    # block（dec_blocks[:N]，因 decoder 顺序镜像官方 29..18，故对应官方 29,28,...）。
    # 0=只保留原 6 层 bottleneck。每个 block +37.75M 全秩 Q/K/V/O。
    text_cross_decoder_layers: int = 0
    text_cross_decoder_start: int = 0
    text_cross_decoder_src_blocks: Tuple[int, ...] = ()  # 空=自动 total-1-start..
    # 2026-11-27 v3：bottleneck 全局平均池化 MLP（零初始化，step-0 恒等）。
    bottleneck_global_mlp: bool = False
    bottleneck_global_rank: int = 128
    # 2026-11-27 v3.2 all-in：AST/金字塔与新模块接线开关。
    # 全层低秩 AdaLN 文本调制（替代旧 3 层全秩；rank128，30 层约 35.6M）。
    adaln_text_all: bool = False
    adaln_text_rank: int = 0
    # σ 连续滑动低秩 adapter（共享主干，不做独立 expert；30 blocks rank64）。
    sigma_adapter: bool = False
    sigma_adapter_rank: int = 64
    sigma_adapter_lo: float = 0.78
    sigma_adapter_hi: float = 0.98
    # 4 个 pyramid tier context/register token（scene/actor/part/pixel）。
    context_tokens: bool = False
    context_token_rank: int = 128
    context_token_heads: int = 8
    # text-cross 扩到 encoder 连续区间；start 默认 0，未来挂中/后段时 src 映射
    # 按真实层号偏移，不会把 blocks.0..N-1 错配到 enc_blocks[start:..]。
    text_cross_encoder_start: int = 0
    text_cross_encoder_layers: int = 0
    text_cross_encoder_rank: int = 0
    # REPA 特征 tap（只出 projector，不含 DINOv2 目标；训练侧另算）。
    repa_dim: int = 0
    repa_layer: int = 8
    # 0 参数 coarse x0 头：只做 2×2 avg-pool，监督复用主 x0。
    coarse_x0_head: bool = False
    # 2026-11-27 v3：可选局部窗口注意力（仅 spatial 头；加性 mask，非块稀疏，
    # 不承诺 ROCm 上省 FLOPs，默认关，仅作质量消融）。
    attn_spatial_window: int = 0
    # 2026-11-26 全栈启用（仅备份版）：把原本全秩的旁路压成低秩，
    # 使"全部接线 + 全部启用"后总参数仍可收敛到 ~3.999B。
    a2_crossmodal_rank: int = 0       # >0 时 10 个 dim×dim 投影换 2×dim×rank
    vlm_bridge_rank: int = 0          # >0 时北大桥中间维=4*rank（可低秩）
    adaln_rank: int = 0               # >0 时双分支 AdaLN 用低秩 MLP
    share_special_ffn: bool = False   # 两个专项块 FFN 复用主干相邻块 FFN（省 ~151M）
    freeze_legacy_head: bool = False  # W2：冻结旧 OutputHead（省 ≈342MB 梯度+优化器状态）。
                                      # 默认关：保证本文件作为 drop-in 替换进主线训练流程时
                                      # 与主线逐位同构（优化器参数名不变、可 resume）；
                                      # 开新训练时由 full_stack() 或显式传参打开。
    dual_sem_head: bool = False       # 方块+语义双功能输出头
    dual_sem_ffn_hidden: int = 4096
    dual_sem_rank: int = 768
    dual_sem_slots: int = 16

    @classmethod
    def tiny(cls):
        """CPU 自检用极小配置（不占显存，秒级跑完）。"""
        return cls(dim=64, num_heads=4, enc_depth=2, dec_depth=2,
                   bottleneck_depth=1, cond_dim=32, t_emb_dim=16,
                   patch_size=(1, 2, 2), aux_head=True)

    @classmethod
    def a2_hybrid(cls, attn_core: str = "hybrid", hybrid_layers: str = "all",
                  a2_crossmodal: bool = False):
        """a2_v2 工厂：基于 v1_baseline_4b5（含完整 v7 full_head 默认值），
        只打开 a2 算力层/语义层开关。默认 softmax 走此工厂仍逐值等价主线。
        GPU 门禁（≥1.5x & cos≥0.9）未过前，hybrid_layers 保持 "none"。
        """
        return cls(dim=3072, num_heads=24, enc_depth=12, dec_depth=12,
                   bottleneck_depth=6, pyramid=False,
                   in_channels=48, out_channels=48, cond_dim=4096,
                   dual_branch_adaln=False, snr_cond=False, ref_align=False,
                   vlm_low_rank=512, head_adaln=True,
                   temporal_lowpass=True, motion_head=True, freq_branch=True,
                   local_conv=True, local_every_dec_block=False,
                   local_out_strength=0.0, out_lp_init=0.0, latent_refine=True,
                   style_encoder=False, ref_align_encoder=False,
                   cond_rank=384, cond_layers=6,
                   temporal_block=True, spatial_block=True,
                   detail_head=True, detail_hidden=19456,
                   skip_alpha_init=1.0,
                   attn_core=attn_core, hybrid_layers=hybrid_layers,
                   a2_crossmodal=a2_crossmodal)

    @classmethod
    def pau_syntax(cls, pau_tiers: int = 4, pau_rank: int = 64):
        """PAU-DiT 工厂：继承 v1_baseline_4b5 全部默认值，只打开语法平面。"""
        return cls(dim=3072, num_heads=24, enc_depth=12, dec_depth=12,
                   bottleneck_depth=6, pyramid=False,
                   in_channels=48, out_channels=48, cond_dim=4096,
                   dual_branch_adaln=False, snr_cond=False, ref_align=False,
                   vlm_low_rank=512, head_adaln=True,
                   temporal_lowpass=True, motion_head=True, freq_branch=True,
                   local_conv=True, local_every_dec_block=False,
                   local_out_strength=0.0, out_lp_init=0.0, latent_refine=True,
                   style_encoder=False, ref_align_encoder=False,
                   cond_rank=384, cond_layers=6,
                   temporal_block=True, spatial_block=True,
                   detail_head=True, detail_hidden=19456,
                   skip_alpha_init=1.0,
                   pau_syntax_enabled=True, pau_tiers=pau_tiers, pau_rank=pau_rank)

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
                   local_conv=True, local_every_dec_block=False,
                   local_out_strength=0.0, out_lp_init=0.0, latent_refine=True,
                   style_encoder=False, ref_align_encoder=False,
                   cond_rank=384, cond_layers=6,
                   temporal_block=True, spatial_block=True,
                   detail_head=True, detail_hidden=19456,
                   skip_alpha_init=1.0)

    @classmethod
    def full_stack(cls):
        """全栈启用工厂（仅备份版 model_v7_backup_20261126）。

        所有架构开关 + 全部旁路 + ASTree/TeaCache 由 fullstack_sampler 接线；
        大模块一律低秩化，专项块 FFN 共享主干，双功能头回填，使总参数 ≈3.999B。
        与 v1_baseline_4b5 的旧 ckpt 严格兼容：新键 strict=False 加载，零门控起步。
        """
        return cls(dim=3072, num_heads=24, enc_depth=12, dec_depth=12,
                   bottleneck_depth=6, pyramid=False,
                   in_channels=48, out_channels=48, cond_dim=4096,
                   attn_backend="flash2",
                   # 训练口径继承
                   full_gated_head=True, head_adaln=True,
                   full_head_disable=("patch_norm", "dualact_gate", "bandnorm_res"),
                   freq_route=True, freq_route_rank=8,
                   freq_route_scale=1.0, freq_route_topk=0,
                   skip_alpha_init=1.0,
                   temporal_lowpass=True, motion_head=True, freq_branch=True,
                   local_conv=True, local_every_dec_block=True, latent_refine=True,
                   # a2 算力层。2026-11-26 AutoClaw 实测后改回 softmax：
                   #   linear/hybrid 与 softmax 的输出余弦只有 0.18~0.70（门禁要求 ≥0.9）；
                   #   VSA 在 ROCm 上比稠密 SDPA 慢 2~4 倍（注意力核对 q 长度 64 的形状效率极低）。
                   # 两项都留在代码里默认不用；需要复现旧的全栈行为时显式传 attn_core="vsa"。
                   attn_core="softmax", hybrid_layers="none",
                   # 语义层：压缩跨模态（低秩）+ PAU 语法平面
                   a2_crossmodal=True, a2_crossmodal_in_attn=True,
                   a2_crossmodal_rank=32, a2_crossmodal_slots=32,
                   pau_syntax_enabled=True, pau_tiers=4, pau_rank=24,
                   # 北大桥 + 条件编码器（全低秩）
                   use_vlm=True, vlm_low_rank=32, vlm_bridge_rank=64,
                   vlm_stage_ratio=0.5,
                   ref_align=True, style_encoder=True, ref_align_encoder=True,
                   cond_rank=16, cond_layers=2,
                   dual_branch_adaln=True, adaln_rank=16, snr_cond=True,
                   # 专项块共享 FFN + 双功能头
                   temporal_block=True, spatial_block=True,
                   share_special_ffn=True,
                   freeze_legacy_head=True,
                   detail_head=True, detail_hidden=19456,
                   dual_sem_head=True, dual_sem_ffn_hidden=3484,
                   dual_sem_rank=768, dual_sem_slots=16)

    @classmethod
    def full_stack_v3(cls):
        """2026-11-27 v3：full_stack + bottleneck text cross-attn + global MLP.

        与 full_stack 的唯一结构差异集中在 6 个 bottleneck block：
          * text_cross_attn: 自适应 σ 门控；Q/O 可低秩；K/V 默认冻结。
          * bottleneck_global_mlp: 全局池化调制，零初始化恒等。
        其余 FreqFormer / full_head / band head / special blocks 全部沿用。
        """
        cfg = cls.full_stack()
        cfg.text_cross_attn = True
        cfg.text_cross_rank = 0
        cfg.text_cross_base_gate = 0.1
        cfg.text_cross_text_dim = 3072
        cfg.text_cross_scale_alpha = 0.0
        cfg.text_cross_freeze_kv = True
        cfg.text_cross_init_wan = (
            "./checkpoints/Wan2.2-TI2V-5B-Diffusers")
        cfg.text_cross_encoder_layers = 6
        cfg.text_cross_encoder_rank = 256
        cfg.adaln_text_all = True
        cfg.adaln_text_rank = 128
        cfg.adaln_text_layers = ()
        cfg.sigma_adapter = True
        cfg.sigma_adapter_rank = 64
        cfg.context_tokens = True
        cfg.context_token_rank = 128
        cfg.repa_dim = 384
        cfg.repa_layer = 8
        cfg.coarse_x0_head = True
        # context tokens replace the old global-MLP branch; keeping both would
        # add two global residuals on the same bottleneck state.
        cfg.bottleneck_global_mlp = False
        cfg.bottleneck_global_rank = 128
        return cfg


class UViT3D(nn.Module):
    """单分辨率 U 形 DiT（U-DiT）主干。

    数据流：
      latent [B,C,T,H,W] --Conv3d patchify--> tokens [B,N,dim]
      encoder（逐层 2+1D attn）--> 记录每层 skip
      bottleneck（JointAttention，完整 3D）
      decoder（逐层上采样 + GatedSkip 融合对应 encoder skip）
      OutputHead --> 速度场 v [B,C,T',H',W'] + 时序辅助残差
    """
    @classmethod
    def a2_hybrid(cls, attn_core: str = "softmax", hybrid_layers: str = "none",
                  a2_crossmodal: bool = False):
        """a2_v2 模型工厂入口（2026-11-26 补，备份版）。

        主线版补在模型类上，但默认参数是 hybrid/all，与主线文件里
        “默认 softmax 逐值等价主线”的注释自相矛盾。备份版这里用安全默认值
        （softmax/none），需要 hybrid 时显式传参。
        """
        cfg = UViT3DConfig.a2_hybrid(attn_core=attn_core,
                                     hybrid_layers=hybrid_layers,
                                     a2_crossmodal=a2_crossmodal)
        return cls(cfg)

    def __init__(self, cfg: UViT3DConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_backend_info = backend_diagnose()
        print(f"[rope] theta={ROPE_THETA:g} (SAKIKO_ROPE_THETA)", flush=True)
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
        # L1 共享条件 embedder（教师式；全块共用，产出 6 段调制基 e [B,6,dim]）
        # 2026-11-23：段数 6 与教师 blocks.N.modulation / time_projection(18432=6*3072) 对齐
        self.cond_segments = 6
        self.ada_embedder = TeacherAdaLN(cfg.dim, cfg.dim, segments=self.cond_segments)
        # a2_v2 语义层：压缩跨模态 + 非对称特权（默认不实例化，零参数增量）。
        self.crossmodal = None
        self.crossmodal_budget = None
        if getattr(cfg, "a2_crossmodal", False):
            self.crossmodal = _AsymmetricCrossModal(
                cfg.dim, num_slots=getattr(cfg, "a2_crossmodal_slots", 32),
                rank=getattr(cfg, "a2_crossmodal_rank", 0))
        # PAU-DiT 语法平面：默认不实例化（零参数增量）；开时输出零门控起步。
        self.syntax_compiler = None
        self._syn_tier = None
        self._n_blocks = (cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth
                          + int(bool(getattr(cfg, "temporal_block", False)))
                          + int(bool(getattr(cfg, "spatial_block", False))))
        if getattr(cfg, "pau_syntax_enabled", False):
            self.syntax_compiler = _SyntaxCompiler(
                cfg.dim, self._n_blocks,
                n_tiers=int(getattr(cfg, "pau_tiers", 4)),
                rank=int(getattr(cfg, "pau_rank", 64)),
                vlm_dim=cfg.vlm_dim if getattr(cfg, "use_vlm", False) else 0,
                gate_init=float(getattr(cfg, "pau_gate_init", 0.01) or 0.01))

        # SNR 分层条件编码：按噪声级 sigma∈[0,1] 把条件粒度调制到各块
        # （流匹配 t 天然是噪声水平，无需额外标注；输出 per-block 的门控标量）
        self.snr_cond = bool(cfg.snr_cond)
        if self.snr_cond:
            # 2026-11-23 修复：gidx 还包含 temporal_special / spatial_special 两个槽位，
            # 原输出只到 enc+bottleneck+dec，snr_cond=True 时最后两次 _block_cond 会越界。
            _n_special = int(bool(getattr(cfg, "temporal_block", False))) + \
                         int(bool(getattr(cfg, "spatial_block", False)))
            self.snr_mlp = nn.Sequential(
                nn.Linear(1, cfg.dim // 4), nn.SiLU(),
                nn.Linear(cfg.dim // 4,
                          cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth + _n_special),
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
                                             dual_branch_adaln=cfg.dual_branch_adaln,
                                             adaln_rank=getattr(cfg, "adaln_rank", 0),
                                             attn_core=cfg.attn_core,
                                             hybrid_layers=cfg.hybrid_layers,
                                             hybrid_chunk=cfg.hybrid_chunk,
                                             hybrid_fmap=cfg.hybrid_fmap))

        self.bottleneck = nn.ModuleList()
        for i in range(cfg.bottleneck_depth):
            self.bottleneck.append(UViTBlock(cfg.dim, cfg.num_heads, "joint",
                                             cfg.ffn_ratio, cfg.causal_temporal,
                                             attn_backend=cfg.attn_backend,
                                             dual_act=cfg.dual_act,
                                             dual_branch_adaln=cfg.dual_branch_adaln,
                                             adaln_rank=getattr(cfg, "adaln_rank", 0),
                                             attn_core=cfg.attn_core,
                                             hybrid_layers=cfg.hybrid_layers,
                                             hybrid_chunk=cfg.hybrid_chunk,
                                             hybrid_fmap=cfg.hybrid_fmap))

        # 2026-11-27 v3：bottleneck 文本 cross-attn / 全局池化 MLP。
        # 只挂 bottleneck，enc/dec、FreqFormer、full_head、band head 不动。
        self.text_cross_modules = None
        if getattr(cfg, "text_cross_attn", False):
            from text_cross_attn import TextCrossAttention, BottleneckGlobalMLP
            self.text_cross_modules = []
            for _blk in self.bottleneck:
                _tc = TextCrossAttention(
                    dim=cfg.dim, num_heads=cfg.num_heads,
                    text_dim=int(getattr(cfg, "text_cross_text_dim", 3072)),
                    rank=int(getattr(cfg, "text_cross_rank", 0)),
                    base_gate=float(getattr(cfg, "text_cross_base_gate", 0.1)),
                    text_scale_alpha=float(getattr(cfg, "text_cross_scale_alpha", 0.0)),
                    freeze_kv=bool(getattr(cfg, "text_cross_freeze_kv", True)),
                )
                _blk.text_cross = _tc
                self.text_cross_modules.append(_tc)
            _wan_dir = str(getattr(cfg, "text_cross_init_wan", "") or "")
            if _wan_dir:
                from text_cross_attn import init_text_cross_from_wan
                _reports = init_text_cross_from_wan(
                    self, _wan_dir,
                    src_blocks=tuple(getattr(cfg, "text_cross_src_blocks",
                                              (12, 13, 14, 15, 16, 17)))
                )
                print(f"[text-cross] init from Wan2.2 attn2: {_reports}", flush=True)
            else:
                print("[text-cross] enabled but no init checkpoint; random init",
                      flush=True)
            # encoder 连续区间也挂 text-cross；start 允许未来挂到中/后段。
            _enc_tc_n = int(getattr(cfg, "text_cross_encoder_layers", 0) or 0)
            if _enc_tc_n > 0:
                _enc_start = max(0, int(getattr(cfg, "text_cross_encoder_start", 0) or 0))
                _enc_end = min(len(self.enc_blocks), _enc_start + _enc_tc_n)
                _enc_n = _enc_end - _enc_start
                if _enc_n <= 0:
                    raise ValueError(
                        f"text_cross_encoder_start={_enc_start} out of range "
                        f"enc_depth={len(self.enc_blocks)}")
                _enc_rank = int(getattr(cfg, "text_cross_encoder_rank", 0) or 0)
                for _blk in self.enc_blocks[_enc_start:_enc_end]:
                    _tc = TextCrossAttention(
                        dim=cfg.dim, num_heads=cfg.num_heads,
                        text_dim=int(getattr(cfg, "text_cross_text_dim", 3072)),
                        rank=_enc_rank,
                        base_gate=float(getattr(cfg, "text_cross_base_gate", 0.1)),
                        text_scale_alpha=float(getattr(cfg, "text_cross_scale_alpha", 0.0)),
                        freeze_kv=bool(getattr(cfg, "text_cross_freeze_kv", True)),
                    )
                    _blk.text_cross = _tc
                    self.text_cross_modules.append(_tc)
                if _wan_dir:
                    _enc_indices = tuple(range(_enc_start, _enc_end))
                    _reports = init_text_cross_from_wan(
                        self, _wan_dir,
                        src_blocks=_enc_indices,
                        dst_attr="enc_blocks",
                        dst_indices=_enc_indices)
                    print(f"[text-cross-enc] encoder blocks {_enc_start}..{_enc_end-1} "
                          f"init from Wan2.2 attn2 {list(_enc_indices)}: {_reports}",
                          flush=True)
        if getattr(cfg, "bottleneck_global_mlp", False):
            from text_cross_attn import BottleneckGlobalMLP
            for _blk in self.bottleneck:
                _blk.bottleneck_global_mlp = BottleneckGlobalMLP(
                    cfg.dim, rank=int(getattr(cfg, "bottleneck_global_rank", 128)))
            print(f"[bottleneck-mlp] enabled on {len(self.bottleneck)} bottleneck blocks "
                  "(zero-init identity)", flush=True)
        # 4-tier context/register tokens：在 bottleneck 做 read/write，不插序列。
        self.context_tokens = None
        if getattr(cfg, "context_tokens", False):
            from text_cross_attn import TierContextTokens
            self.context_tokens = TierContextTokens(
                cfg.dim, n_tiers=int(getattr(cfg, "pau_tiers", 4)),
                rank=int(getattr(cfg, "context_token_rank", 128)),
                heads=int(getattr(cfg, "context_token_heads", 8)))
            print("[context-tokens] 4 pyramid tiers on bottleneck "
                  "(read/write, no sequence insertion)", flush=True)
        self._ctx_tiers = None

        for i in range(cfg.dec_depth):
            mode = pick_mode(cfg.dec_depth - 1 - i)
            self.dec_blocks.append(UViTBlock(cfg.dim, cfg.num_heads, mode,
                                             cfg.ffn_ratio, cfg.causal_temporal,
                                             attn_backend=cfg.attn_backend,
                                             dual_act=cfg.dual_act,
                                             dual_branch_adaln=cfg.dual_branch_adaln,
                                             adaln_rank=getattr(cfg, "adaln_rank", 0),
                                             attn_core=cfg.attn_core,
                                             hybrid_layers=cfg.hybrid_layers,
                                             hybrid_chunk=cfg.hybrid_chunk,
                                             hybrid_fmap=cfg.hybrid_fmap))
            self.skips.append(GatedSkip(cfg.dim, cfg.skip_alpha_init, cfg.skip_spectral))

        # v3.1：text cross-attn 扩展到 decoder 连续区间（默认最深层）。
        # decoder 是镜像顺序：dec_blocks[0] 对应官方 block 29，故 src 也从 29 递减。
        _dec_tc_n = int(getattr(cfg, "text_cross_decoder_layers", 0) or 0)
        if _dec_tc_n > 0:
            if not bool(getattr(cfg, "text_cross_attn", False)):
                raise ValueError("text_cross_decoder_layers>0 requires text_cross_attn=True")
            _dec_start = max(0, int(getattr(cfg, "text_cross_decoder_start", 0) or 0))
            _dec_end = min(len(self.dec_blocks), _dec_start + _dec_tc_n)
            _dec_n = _dec_end - _dec_start
            if _dec_n <= 0:
                raise ValueError(
                    f"text_cross_decoder_start={_dec_start} out of range "
                    f"dec_depth={len(self.dec_blocks)}")
            from text_cross_attn import TextCrossAttention, init_text_cross_from_wan
            if self.text_cross_modules is None:
                self.text_cross_modules = []
            for _blk in self.dec_blocks[_dec_start:_dec_end]:
                _tc = TextCrossAttention(
                    dim=cfg.dim, num_heads=cfg.num_heads,
                    text_dim=int(getattr(cfg, "text_cross_text_dim", 3072)),
                    rank=int(getattr(cfg, "text_cross_rank", 0)),
                    base_gate=float(getattr(cfg, "text_cross_base_gate", 0.1)),
                    text_scale_alpha=float(getattr(cfg, "text_cross_scale_alpha", 0.0)),
                    freeze_kv=bool(getattr(cfg, "text_cross_freeze_kv", True)),
                )
                _blk.text_cross = _tc
                self.text_cross_modules.append(_tc)
            _wan_dir = str(getattr(cfg, "text_cross_init_wan", "") or "")
            if _wan_dir:
                _total_blocks = cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth
                _dec_indices = tuple(range(_dec_start, _dec_end))
                _src = tuple(getattr(cfg, "text_cross_decoder_src_blocks", ()) or ())
                if not _src:
                    _src = tuple(_total_blocks - 1 - i for i in _dec_indices)
                _reports = init_text_cross_from_wan(
                    self, _wan_dir, src_blocks=_src, dst_attr="dec_blocks",
                    dst_indices=_dec_indices)
                print(f"[text-cross-dec] decoder blocks {_dec_start}..{_dec_end-1} "
                      f"init from Wan2.2 attn2 {list(_src)}: {_reports}", flush=True)
            else:
                print(f"[text-cross-dec] decoder blocks {_dec_start}..{_dec_end-1} "
                      f"random init", flush=True)

        # 轻量 AdaLN-Zero 全局文本注入：并行 residual，不替换 norm1/norm2。
        # v3.2：adaln_text_all=True 时 30 个主干 block 全挂低秩 AdaLN；
        # 旧 3 层全秩配置仍保留兼容，但新 run 用全层 rank128。
        self.adaln_text_modules = None
        _ada_all = bool(getattr(cfg, "adaln_text_all", False))
        _ada_rank = int(getattr(cfg, "adaln_text_rank", 0) or 0)
        _ada_layers = tuple(int(x) for x in
                            (getattr(cfg, "adaln_text_layers", ()) or ()))
        if _ada_all or _ada_layers:
            if not bool(getattr(cfg, "text_cross_attn", False)):
                raise ValueError("adaln_text_all/layers requires text_cross_attn=True")
            from text_cross_attn import AdaLNZeroText
            self.adaln_text_modules = []
            if _ada_all:
                _targets = (list(self.enc_blocks) + list(self.bottleneck)
                            + list(self.dec_blocks))
                _tag = "all-30"
            else:
                _targets = [self.dec_blocks[_i] for _i in sorted(set(_ada_layers))]
                _tag = f"decoder {list(_ada_layers)}"
            for _blk in _targets:
                _ta = AdaLNZeroText(
                    dim=cfg.dim,
                    text_dim=int(getattr(cfg, "text_cross_text_dim", 3072)),
                    gate_hidden=int(getattr(cfg, "adaln_text_gate_hidden", 256)),
                    rank=_ada_rank,
                )
                _blk.text_adaln = _ta
                self.adaln_text_modules.append(_ta)
            print(f"[adaln-text] {_tag} zero-init parallel injection "
                  f"rank={_ada_rank}", flush=True)

        # σ 连续滑动 adapter：30 个主干 block，共享主干、不做独立 expert。
        self.sigma_adapter_modules = None
        if getattr(cfg, "sigma_adapter", False):
            from text_cross_attn import SigmaSlidingAdapter
            self.sigma_adapter_modules = []
            for _blk in (list(self.enc_blocks) + list(self.bottleneck)
                         + list(self.dec_blocks)):
                _sa = SigmaSlidingAdapter(
                    cfg.dim, rank=int(getattr(cfg, "sigma_adapter_rank", 64)),
                    lo=float(getattr(cfg, "sigma_adapter_lo", 0.78)),
                    hi=float(getattr(cfg, "sigma_adapter_hi", 0.98)))
                _blk.sigma_adapter = _sa
                self.sigma_adapter_modules.append(_sa)
            _sa_params = sum(p.numel() for _m in self.sigma_adapter_modules
                            for p in _m.parameters())
            print(f"[sigma-adapter] {len(self.sigma_adapter_modules)} blocks "
                  f"rank={int(getattr(cfg, 'sigma_adapter_rank', 64))} "
                  f"params={_sa_params/1e6:.2f}M", flush=True)

        # AST skip 保护：文本条件关键块不能被硬跳过，否则高σ文本路径直接死。
        # 真实 gidx：enc 0..11 / temporal_special 12 / bottleneck 13..18 /
        # decoder 19..30 / spatial_special 31。必须把 temporal 槽算进去。
        self._protected_blocks = set()
        _enc_n, _bn_n, _dec_n = (len(self.enc_blocks), len(self.bottleneck),
                                 len(self.dec_blocks))
        _ts = int(bool(getattr(cfg, "temporal_block", False)))
        _enc_tc_n = int(getattr(cfg, "text_cross_encoder_layers", 0) or 0)
        if _enc_tc_n > 0:
            _p_enc_start = max(0, int(getattr(cfg, "text_cross_encoder_start", 0) or 0))
            _p_enc_end = min(_enc_n, _p_enc_start + _enc_tc_n)
            self._protected_blocks.update(range(_p_enc_start, _p_enc_end))
        if bool(getattr(cfg, "text_cross_attn", False)):
            self._protected_blocks.update(
                range(_enc_n + _ts, _enc_n + _ts + _bn_n))
        _dec_tc_n = int(getattr(cfg, "text_cross_decoder_layers", 0) or 0)
        if _dec_tc_n > 0:
            _p_dec_start = max(0, int(getattr(cfg, "text_cross_decoder_start", 0) or 0))
            _p_dec_end = min(_dec_n, _p_dec_start + _dec_tc_n)
            _dec_gidx_start = _enc_n + _ts + _bn_n + _p_dec_start
            self._protected_blocks.update(
                range(_dec_gidx_start, _dec_gidx_start + (_p_dec_end - _p_dec_start)))
        if self._protected_blocks:
            print(f"[ast-skip] protected blocks {sorted(self._protected_blocks)} "
                  f"(text-cross mounted blocks + full bottleneck)", flush=True)

        # REPA: encoder block tap -> projector to DINOv2 dim; target is computed
        # outside the model (training script) so no ViT lives in the graph.
        self._last_repa_feature = None
        self.repa_proj = None
        _repa_dim = int(getattr(cfg, "repa_dim", 0) or 0)
        if _repa_dim > 0:
            self.repa_proj = nn.Linear(cfg.dim, _repa_dim, bias=True)
            print(f"[repa] tap=enc block {int(getattr(cfg, 'repa_layer', 8))} "
                  f"proj dim={_repa_dim}", flush=True)

        # 0 参数 coarse x0 头：显式结构，预算按 0 trainable 入账。
        self.coarse_x0_head = None
        if getattr(cfg, "coarse_x0_head", False):
            from text_cross_attn import CoarseX0Head
            self.coarse_x0_head = CoarseX0Head()
            print("[coarse-x0] 2×2 avg-pool head, 0 trainable params", flush=True)

        # v3 fine-grained checkpointing: by default every block follows the
        # global switch.  Explicitly disable re-checkpoint for bottleneck / the
        # last N decoder blocks (they are low-resolution and cheap to keep).
        if getattr(cfg, "ckpt_skip_bottleneck", False):
            for _blk in self.bottleneck:
                _blk._ckpt_allowed = False
        _dec_off = int(getattr(cfg, "ckpt_last_dec_off", 0) or 0)
        if _dec_off > 0:
            for _blk in self.dec_blocks[-_dec_off:]:
                _blk._ckpt_allowed = False

        # 固定局部注意力先验：只开最后 N 个 decoder block（默认 3）。
        # 从环境变量读取便于同一 ckpt 零训练扫强度；不新增参数、不改 state_dict。
        s_bias = float(_os.environ.get("SAKIKO_ATTN_LOCAL_BIAS",
                                       getattr(cfg, "attn_local_bias", 0.0)) or 0.0)
        if s_bias > 0:
            last_n = int(getattr(cfg, "attn_local_last_n", 3))
            for blk in self.dec_blocks[-last_n:]:
                for m in blk.modules():
                    if hasattr(m, "local_bias_strength"):
                        m.local_bias_strength = s_bias

        # v3 optional spatial window: only spatial heads; anchors remain global.
        _win = int(getattr(cfg, "attn_spatial_window", 0) or 0)
        if _win > 0:
            for _blk in list(self.enc_blocks) + list(self.dec_blocks):
                if getattr(_blk, "attn_mode", None) == "spatial":
                    if hasattr(_blk.attn, "attn"):
                        _blk.attn.attn.local_window = _win
                elif getattr(_blk, "attn_mode", None) == "gsa":
                    if hasattr(_blk.attn.spatial, "attn"):
                        _blk.attn.spatial.attn.local_window = _win
            print(f"[attn-window] spatial local window={_win}, anchors global",
                  flush=True)

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
            # W1 修复：必须含 temporal_special / spatial_special 两个槽位，
            # 否则 forward 里 gidx 给专项块 +1 后 inject_high 分阶段索引整体后移。
            total_blocks = self._n_blocks
            self.vlm_injector = VLMInjector(
                dim=cfg.dim, num_heads=cfg.num_heads, n_blocks=total_blocks,
                vlm_dim=cfg.vlm_dim, stage_ratio=cfg.vlm_stage_ratio,
                low_rank=getattr(cfg, "vlm_low_rank", 0),
                bridge_rank=getattr(cfg, "vlm_bridge_rank", 0),
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
        _share = bool(getattr(cfg, "share_special_ffn", False))
        self.temporal_special = None
        if getattr(cfg, "temporal_block", False):
            self.temporal_special = SpecializedBlock(
                cfg.dim, cfg.num_heads, "temporal",
                attn_backend=cfg.attn_backend, causal_temporal=cfg.causal_temporal,
                attn_core=cfg.attn_core, hybrid_chunk=cfg.hybrid_chunk,
                hybrid_fmap=cfg.hybrid_fmap,
                shared_ffn=(self.enc_blocks[-1].ffn if _share else None))
        self.spatial_special = None
        if getattr(cfg, "spatial_block", False):
            self.spatial_special = SpecializedBlock(
                cfg.dim, cfg.num_heads, "spatial",
                attn_backend=cfg.attn_backend,
                attn_core=cfg.attn_core, hybrid_chunk=cfg.hybrid_chunk,
                hybrid_fmap=cfg.hybrid_fmap,
                shared_ffn=(self.dec_blocks[-1].ffn if _share else None))

        # 新增轻量算子（4b5 全开；tiny 默认关）
        self.temporal_lowpass = TemporalLowPass(cfg.dim) if getattr(cfg, "temporal_lowpass", False) else None
        self.motion_head = MotionHead(cfg.dim) if getattr(cfg, "motion_head", False) else None
        self.freq_branch = FreqBranch(cfg.dim) if getattr(cfg, "freq_branch", False) else None

        # V2-28 轻量化条件编码器（低秩 r384 + 6 层，纯 torch SDPA）
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
                               detail_hidden=getattr(cfg, "detail_hidden", 19456),
                               out_lp_init=getattr(cfg, "out_lp_init", 0.0))
        # 2026-11-24：新完整门控输出头。旧 LinearHead 保留仅为旧 ckpt 兼容，
        # full_gated_head=True 时前向完全不使用旧头权重。
        self.full_head = None
        self._last_full_head_aux = None
        if getattr(cfg, "full_gated_head", False):
            self.full_head = FullGatedTokenHead(
                in_dim=cfg.dim,
                out_dim=math.prod(cfg.patch_size) * cfg.out_channels,
                t_emb_dim=cfg.t_emb_dim,
                groups=getattr(cfg, "full_head_groups", 4),
                vlm_dim=cfg.vlm_dim,
                ctx_dim=getattr(cfg, "full_head_ctx_dim", 128),
                compress_ratio=getattr(cfg, "full_head_compress", 4),
            )
            self.full_head.set_switches(getattr(cfg, "full_head_disable", ()) or ())
            fh = self.full_head
            print("[full-head] Cmax=%d out_dim=%d groups=%d compress=%d "
                  "cfreq=%s hscale=%s pproj=%s fclamp=%s dual_high=%s cross=%s group=%s time=%s rpe=%s sobel=%s aa=%s pyr=%s faux=%s vmag=%s"
                  % (fh.Cmax, fh.out_dim, cfg.full_head_groups,
                     cfg.full_head_compress, fh.enable_channel_freq,
                     fh.enable_head_scale, fh.enable_pre_proj_scale,
                     fh.enable_final_weight_clamp, fh.enable_dual_high, fh.enable_cross,
                     fh.enable_group_mask, fh.enable_time_mod, fh.enable_rpe,
                     fh.enable_sobel_grad, fh.enable_antialias, fh.enable_pyramid,
                     fh.enable_frame_aux, fh.enable_vmag_aux),
                  flush=True)
            # W2 修复（原为备份版无条件执行，2026-11-26 改为 cfg.freeze_legacy_head 开关）：
            # 旧 OutputHead 共 85,361,538 参数
            # （detail_head 63.5M + head_mod 18.9M + aux_head 2.36M），
            # full_gated_head=True 时永不参与前向。构造期冻结后 train_staged
            # 的 make_optimizers 会因 `if not p.requires_grad: continue` 自动
            # 跳过，省梯度与优化器状态（≈342MB bf16）。
            # 权重仍留在 state_dict，strict=False 热启动逐 key 兼容。
            # 注意：optimizer 参数名列表变化，旧 optimizer_state 不可直接
            # resume —— 开关打开时必须开新训练或 --reset-optim。
            if getattr(cfg, "freeze_legacy_head", False):
                for _p in self.head.parameters():
                    _p.requires_grad_(False)
        # V2-31：细节头 ckpt 开关与主干保持一致（构造时 cfg 可能已开）
        self.head.detail_ckpt = bool(self.grad_checkpointing)
        # 2026-11-24：unpatchify 后的 latent 3×3 联合卷积残差（零卷积、零残差起步）
        self.latent_refine = (LatentRefine(cfg.out_channels)
                              if getattr(cfg, "latent_refine", False) else None)
        # 2026-11-26 双功能头：方块残差 + 语义 cross-attn，零门控起步。
        self.dual_sem_head = None
        if getattr(cfg, "dual_sem_head", False):
            try:
                from .semantic_boundary_head import SemanticBoundaryDualHead
            except ImportError:  # 允许以顶层模块方式构建（审计/脚本直跑）
                from semantic_boundary_head import SemanticBoundaryDualHead
            self.dual_sem_head = SemanticBoundaryDualHead(
                cfg.dim, cfg.num_heads, cfg.out_channels,
                patch_size=cfg.patch_size,
                ffn_hidden=int(getattr(cfg, "dual_sem_ffn_hidden", 4096)),
                sem_rank=int(getattr(cfg, "dual_sem_rank", 768)),
                sem_slots=int(getattr(cfg, "dual_sem_slots", 16)))

        # 2026-11-26 A1：频带 AdaLN 条件。每次前向只对 xt 做一次 rfft2，
        # 把 low/mid/high 占比经 3->64->dim 的零初始化 MLP 加到共享 cond。
        # 只在 band_cond_enabled=True 时进入计算图；零初始化保证 5200 基线不变。
        self.band_cond_enabled = False
        self.band_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, cfg.dim))
        nn.init.zeros_(self.band_mlp[-1].weight)
        nn.init.zeros_(self.band_mlp[-1].bias)

        # 2026-11-26 freqformer：轻量频域路由（默认关）。挂到非 GSA 的
        # SelfAttention；参数前缀 freqrouter.，Stage1 ckpt strict=False 直载。
        self.freqrouter = None
        if getattr(cfg, "freq_route", False):
            from freq_router import FreqRouter
            self.freqrouter = FreqRouter(
                cfg.dim, cfg.num_heads,
                rank=getattr(cfg, "freq_route_rank", 8),
                scale=getattr(cfg, "freq_route_scale", 1.0),
                topk=getattr(cfg, "freq_route_topk", 0),
            )

            def _attach_sa(blk):
                if getattr(blk, "attn_mode", None) == "gsa":
                    return
                sa = getattr(blk.attn, "attn", blk.attn)
                if hasattr(sa, "router"):
                    # 不走 nn.Module.__setattr__：否则 router 被注册成每个 block
                    # 的子模块，state_dict 会按 block 路径复制 30+ 份。顶层
                    # self.freqrouter 是唯一注册点，这里只做普通属性引用。
                    object.__setattr__(sa, "router", self.freqrouter)

            for _blk in list(self.enc_blocks) + list(self.bottleneck) + list(self.dec_blocks):
                _attach_sa(_blk)
            if self.temporal_special is not None:
                _attach_sa(self.temporal_special.block)
            if self.spatial_special is not None:
                _attach_sa(self.spatial_special.block)

        # v3: optional structured low-rank on the whole backbone.  This is
        # deliberately config-gated (default 0).  A from-scratch run can use
        # it because there is no warm-start state_dict to preserve; an old
        # checkpoint cannot be loaded strict after this changes key shapes.
        _blr = int(getattr(cfg, "band_lr_rank", 0) or 0)
        if _blr > 0 or any(int(getattr(cfg, f"band_lr_rank_{s}", 0) or 0) > 0
                          for s in ("enc", "bottleneck", "dec",
                                    "attn", "ffn", "ffn_enc", "ffn_bottleneck", "ffn_dec")) \
                or int(getattr(cfg, "band_lr_qk_rank", 0) or 0) > 0 \
                or int(getattr(cfg, "band_lr_vo_rank", 0) or 0) > 0 \
                or int(getattr(cfg, "band_lr_attn_groups", 0) or 0) > 0:
            self._apply_band_lr()

    def _apply_band_lr(self) -> None:
        """Replace dense qkv/o/w12/w3 with token-space low-rank factors.

        Rank resolution per segment: segment override > global band_lr_rank.
        Module gates: band_lr_attn / band_lr_ffn.  From-scratch only.
        """
        from text_cross_attn import LowRankLinear

        cfg = self.cfg
        g_rank = max(0, int(getattr(cfg, "band_lr_rank", 0) or 0))
        scope = str(getattr(cfg, "band_lr_scope", "all") or "all").lower()
        do_attn = bool(getattr(cfg, "band_lr_attn", True)) and scope in ("all", "attn")
        do_ffn = bool(getattr(cfg, "band_lr_ffn", True)) and scope in ("all", "ffn")

        def _seg_rank(name: str) -> int:
            v = int(getattr(cfg, f"band_lr_rank_{name}", 0) or 0)
            return v if v > 0 else g_rank

        def _rank_for(kind: str, seg: str) -> int:
            """kind-specific rank > kind+segment rank > segment rank > global."""
            v = int(getattr(cfg, f"band_lr_rank_{kind}_{seg}", 0) or 0)
            if v > 0:
                return v
            v = int(getattr(cfg, f"band_lr_rank_{kind}", 0) or 0)
            if v > 0:
                return v
            return _seg_rank(seg)

        before = sum(p.numel() for p in self.parameters())
        seen: set[int] = set()

        def _swap(mod: nn.Module, attr: str, rank: int) -> None:
            old = getattr(mod, attr, None)
            if not isinstance(old, nn.Linear) or rank <= 0:
                return
            if id(mod) in seen:
                return
            r = min(int(rank), old.in_features, old.out_features)
            new = LowRankLinear(old.in_features, old.out_features, r,
                                bias=old.bias is not None)
            new = new.to(device=old.weight.device, dtype=old.weight.dtype)
            setattr(mod, attr, new)

        def _apply_group(modules, attn_rank: int, ffn_rank: int) -> None:
            nonlocal seen
            for m in modules:
                if do_attn and isinstance(m, SelfAttention):
                    _g = max(0, int(getattr(cfg, "band_lr_attn_groups", 0) or 0))
                    _qk = max(0, int(getattr(cfg, "band_lr_qk_rank", 0) or 0))
                    _vo = max(0, int(getattr(cfg, "band_lr_vo_rank", 0) or 0))
                    if _g > 0:
                        # 分组块对角：Q/K/V/O 各 G 组，组内全秩，组间零交互。
                        m.qkv = GroupedQKVLinear(m.dim, groups=_g).to(
                            device=m.qkv.weight.device, dtype=m.qkv.weight.dtype)
                        m.o = BlockDiagLinear(m.dim, m.dim, groups=_g).to(
                            device=m.o.weight.device, dtype=m.o.weight.dtype)
                    elif _qk > 0 or _vo > 0:
                        # Q/K/V 分开：qk=0 表示该分支 dense；vo=0 表示 V dense。
                        m.qkv = SplitQKVLinear(m.dim, q_rank=_qk, k_rank=_qk,
                                               v_rank=_vo).to(
                            device=m.qkv.weight.device, dtype=m.qkv.weight.dtype)
                        if _vo > 0:
                            _swap(m, "o", _vo)
                    else:
                        _swap(m, "qkv", attn_rank)
                        _swap(m, "o", attn_rank)
                if do_ffn and isinstance(m, SwiGLU):
                    _swap(m, "w12", ffn_rank)
                    _swap(m, "w3", ffn_rank)
                seen.add(id(m))

        for _blk in self.enc_blocks:
            _apply_group(_blk.modules(), _rank_for("attn", "enc"),
                         _rank_for("ffn", "enc"))
        for _blk in self.bottleneck:
            _apply_group(_blk.modules(), _rank_for("attn", "bottleneck"),
                         _rank_for("ffn", "bottleneck"))
        for _blk in self.dec_blocks:
            _apply_group(_blk.modules(), _rank_for("attn", "dec"),
                         _rank_for("ffn", "dec"))
        for _special in (self.temporal_special, self.spatial_special):
            if _special is not None:
                _apply_group(_special.modules(), _rank_for("attn", "dec"),
                             _rank_for("ffn", "dec"))

        after = sum(p.numel() for p in self.parameters())
        print(f"[band-lr] attn enc/bn/dec={_rank_for('attn', 'enc')}/"
              f"{_rank_for('attn', 'bottleneck')}/{_rank_for('attn', 'dec')} "
              f"qk={int(getattr(cfg, 'band_lr_qk_rank', 0) or 0)} "
              f"vo={int(getattr(cfg, 'band_lr_vo_rank', 0) or 0)} "
              f"groups={int(getattr(cfg, 'band_lr_attn_groups', 0) or 0)} "
              f"ffn enc/bn/dec={_rank_for('ffn', 'enc')}/"
              f"{_rank_for('ffn', 'bottleneck')}/{_rank_for('ffn', 'dec')} "
              f"attn={do_attn} ffn={do_ffn} params "
              f"{before/1e9:.3f}B -> {after/1e9:.3f}B "
              f"({100.0*(before-after)/max(before,1):.1f}% cut)", flush=True)

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
                  cond: torch.Tensor, grid: Tuple[int, int, int],
                  sigma: Optional[torch.Tensor] = None,
                  ffn_modulation: Optional[torch.Tensor] = None,
                  tier_mod: Optional[torch.Tensor] = None,
                  cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  text_tokens: Optional[torch.Tensor] = None,
                  text_mask: Optional[torch.Tensor] = None,
                  text_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """block 前向：按 ckpt_scope 决定重算粒度。
          - "block"（默认）：整块套 checkpoint（GPU 实测省 ~1.1GiB 激活）；
          - "ffn_only"：整块不套，改由 block 内 FFN 子层自套（SelfAttn 不重算）。
        均用 use_reentrant=False（支持非 tensor grid / 闭包 cond 梯度 / 冻结分支）。
        sigma 透传给频域路由；ffn_modulation 透传给 a2_v2 语义层 FFN 调制（None 恒等）。
        text_tokens 只在 bottleneck 传入；其它 block 为 None，整条路不进图。
        """
        if not ((self.grad_checkpointing
                 and getattr(blk, "_ckpt_allowed", True))
                and torch.is_grad_enabled()):
            return blk(z, cond, grid, sigma=sigma,
                       ffn_modulation=ffn_modulation, tier_mod=tier_mod,
                       cross_kv=cross_kv, text_tokens=text_tokens,
                       text_mask=text_mask, text_valid=text_valid)
        if self.ckpt_scope == "ffn_only":
            return blk(z, cond, grid, sigma=sigma,
                       ffn_modulation=ffn_modulation, tier_mod=tier_mod,
                       cross_kv=cross_kv, text_tokens=text_tokens,
                       text_mask=text_mask, text_valid=text_valid)

        def _blk_fn(_z, _c, _g, _s, _f, _t, _x, _tt, _tm, _tv):
            return blk(_z, _c, _g, sigma=_s, ffn_modulation=_f, tier_mod=_t,
                       cross_kv=_x, text_tokens=_tt, text_mask=_tm,
                       text_valid=_tv)
        return _ckpt.checkpoint(
            _blk_fn, z, cond, grid, sigma, ffn_modulation, tier_mod, cross_kv,
            text_tokens, text_mask, text_valid,
            use_reentrant=False, preserve_rng_state=False,
        )

    def _tier_mod_for(self, tier_mod: Optional[torch.Tensor], gidx: int
                      ) -> Optional[torch.Tensor]:
        """按全局 block 序号取金字塔 tier 增量；None 恒等。

        v3.2: 4 个 context/register tokens 也在这里按 tier 注入，保证它们
        不是挂在 bottleneck 外面的死参数，而是逐层进入 AdaLN 条件。
        """
        base = None
        if tier_mod is not None:
            n = tier_mod.shape[1]
            i = _tier_index(gidx, self._n_blocks, n)
            base = tier_mod[:, i]
        ctx = getattr(self, "_ctx_tiers", None)
        if ctx is not None:
            j = _tier_index(gidx, self._n_blocks, ctx.shape[1])
            c = ctx[:, j]
            base = c if base is None else base + c
        return base

    def _ffn_mod_for(self, ffn_modulation: Optional[torch.Tensor], gidx: int
                     ) -> Optional[torch.Tensor]:
        """从 [B,n_blocks,C] 或 [B,C] 调制中取第 gidx 块；None/N 越界 → None。"""
        if ffn_modulation is None:
            return None
        if ffn_modulation.dim() == 2:
            return ffn_modulation
        if ffn_modulation.dim() == 3 and gidx < ffn_modulation.shape[1]:
            return ffn_modulation[:, gidx]
        return None

    @staticmethod
    def _merge_cond(base: torch.Tensor, *adds: Optional[torch.Tensor]) -> torch.Tensor:
        """四路条件汇流：按给定顺序逐项加法，None 跳过。

        仅做代码合并，不改加法顺序/精度：band_cond → VLM global_cond →
        style_cond → ref_align，与改前逐值等价。
        """
        for a in adds:
            if a is not None:
                base = base + a
        return base

    def _cond(self, t_emb: torch.Tensor, text_emb: torch.Tensor,
              vlm_cond: Optional[torch.Tensor] = None,
              sigma: Optional[torch.Tensor] = None,
              ref_latent: Optional[torch.Tensor] = None,
              grid: Optional[Tuple[int, int, int]] = None,
              style_cond: Optional[torch.Tensor] = None,
              cur_tokens: Optional[torch.Tensor] = None,
              band_cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """融合时间步 + 文本 + VLM 全局语义 -> 共享 embedder 输出 e [B,dim]。

        Returns (e, snr_gates):
          e: 全块共享的 6 段调制基（教师式 adaLN）
          snr_gates: [B, n_blocks] 或 None（SNR 分层时逐块缩放条件粒度）
        style_cond / cur_tokens 为 V2-28 两个轻量编码器的可选输入，None 时整条路不进图。
        """
        t = self.t_embed(t_emb)
        c = self.cond_proj(text_emb)
        # 四路条件统一汇流：band_cond → VLM global_cond → style_cond → ref_align，
        # 加法顺序与改前一致（逐值等价）；None 项整条路不进图。
        vlm_add = None
        if self.vlm_injector is not None and vlm_cond is not None:
            vlm_add = self.vlm_injector.global_cond(vlm_cond)
        ref_add = None
        if ref_latent is not None:
            ref_tok = self.patchify(ref_latent).flatten(2).transpose(1, 2)  # [B,Nr,dim]
            if self.ref_align_encoder is not None:
                ctx = cur_tokens if cur_tokens is not None else ref_tok
                ref_add = self.ref_align_encoder(ctx, ref_tok)
            elif self.ref_align:
                # 旧轻量旁路（向后兼容 tiny/无新编码器配置）
                ref_add = torch.tanh(self.ref_gate) * self.ref_proj(
                    self.ref_norm(ref_tok.mean(dim=1)))
        cond = self._merge_cond(t + c, band_cond, vlm_add, style_cond, ref_add)
        e_raw = self.ada_embedder(cond)                 # 共享两层 MLP -> [B, 6*dim]
        d = self.cfg.dim
        if e_raw.shape[-1] == self.cond_segments * d:
            e = e_raw.unflatten(-1, (self.cond_segments, d))          # [B,6,dim] 教师同构
        else:
            e = e_raw                                   # 兼容旧 1 段表
        # SNR 分层：sigma 是流匹配噪声级（t_emb 通常由 sigma 编码），无额外标注。
        # 高噪声阶段给粗粒度条件、低噪声阶段给细粒度：用 sigma 经小 MLP 出逐块 gate。
        snr_gates = None
        if self.snr_cond:
            from text_cross_attn import sigma_to_batch
            sig = sigma if sigma is not None else t_emb.mean(dim=-1, keepdim=True)
            # σ 归一化唯一走 sigma_to_batch，不在主干内另写 reshape/device/dtype 口径。
            sig = sigma_to_batch(sig, e)
            # sigma_to_batch 的契约是 float32；snr_mlp 权重随模型是 bf16，
            # 这里只在进入 MLP 前对齐权重 dtype，不把 sigma 本身退回 bf16。
            sig = sig.to(dtype=self.snr_mlp[0].weight.dtype)
            snr_gates = torch.sigmoid(self.snr_mlp(sig))   # [B, n_blocks]
        return e, snr_gates

    def _block_cond(self, e: torch.Tensor, snr_gates: Optional[torch.Tensor],
                    idx: int) -> torch.Tensor:
        """取第 idx 块的实际条件（SNR 分层时逐块缩放）。"""
        if snr_gates is None:
            return e
        g = snr_gates[:, idx:idx + 1]                   # [B,1]
        if e.dim() == 3:                                # [B,6,C] 6 段逐样本调制
            return e * g.unsqueeze(-1)
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
                # 2026-11-23：cond 可能是逐块 6 段调制 [B,6,C]，c_expert 是 [B,C]，
                # 直接相加在 batch>1 时会广播崩；沿段轴广播。
                ce = c_expert.unsqueeze(1) if (cond.dim() == 3 and c_expert.dim() == 2) else c_expert
                cond = cond + ce
        return cond

    def _unpatchify_full_head(self, v_tok: torch.Tensor, B0: int,
                              Tt: int, Ht: int, Wt: int) -> torch.Tensor:
        """[B*T, out_dim, Ht, Wt] -> [B, C, T, H, W] for the full head."""
        pt, ph, pw = self.cfg.patch_size
        v = v_tok.permute(0, 2, 3, 1).contiguous()
        v = v.view(B0, Tt, Ht, Wt, pt, ph, pw, self.cfg.out_channels)
        v = v.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return v.reshape(B0, self.cfg.out_channels, Tt * pt, Ht * ph, Wt * pw)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, text_emb: torch.Tensor,
                vlm_hidden: Optional[torch.Tensor] = None, return_aux: bool = False,
                sigma: Optional[torch.Tensor] = None,
                ref_latent: Optional[torch.Tensor] = None,
                style_logits: Optional[torch.Tensor] = None,
                style_tokens: Optional[torch.Tensor] = None,
                backbone_no_grad: bool = False,
                skip_blocks: Optional[List[int]] = None,
                crossmodal_budget: Optional[float] = None,
                ffn_modulation: Optional[torch.Tensor] = None,
                syntax_signals: Optional[dict] = None,
                text_tokens: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                text_valid: Optional[torch.Tensor] = None,
                feature_pool: Optional[List[torch.Tensor]] = None):
        """
        x: [B, C, T, H, W] latent
        t_emb: [B, t_emb_dim]
        text_emb: [B, cond_dim]
        vlm_hidden: [B, L, vlm_dim] Qwen3-VL hidden（可选；给定时走多尺度分层注入）
        sigma: [B,1] 流匹配噪声级（SNR 分层条件用，None 时从 t_emb 推）
        ref_latent: [B,C,T',H',W'] 参考帧 latent（参考残差对齐用，None 时旁路恒等）
        style_logits: [B,E] VLM 风格头下发的 top1 路由 logits（None 时用 MoE 内部 router）
        style_tokens: [B,L,dim] 风格 token（V2-28 风格编码器输入；None 时该路恒等）
        text_tokens: [B,L,3072] 官方 Wan text_embedder 投影后的 UMT5 token 序列；
                     仅 v3 bottleneck text_cross 使用，None 时整条路不进图
        text_mask: [B,L] bool/int 文本 padding mask（True=有效 token）
        backbone_no_grad: 冻结主干且只训 full_head 时，主干不建激活图；Head 照常反传。
        """
        B, C, T, H, W = x.shape
        # σ 统一入口：允许 float/标量/1D/2D；下游 full_head/snr_cond/text gate
        # 一律拿 [B,1]，不再各自假设 sigma 是 tensor。
        if sigma is not None:
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
        if feature_pool is not None:
            feature_pool.clear()
        self._ctx_tiers = None
        self._last_repa_feature = None
        pt, ph, pw = self.cfg.patch_size
        assert T % pt == 0 and H % ph == 0 and W % pw == 0, \
            f"输入 {x.shape} 必须能被 patch_size {self.cfg.patch_size} 整除"

        # 门控时序观测 r_t：原始 latent 相邻帧残差均值，3 帧滑动窗口。
        # 只做监控/门控输入，detach 后不向主干回传梯度。
        frame_res_frames = None
        if self.full_head is not None:
            with torch.no_grad():
                if T > 1:
                    d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(
                        dim=(1, 3, 4)).float()          # [B,T-1]
                    # 把首帧差分复制到 t=0，使长度恢复到 T；再对时间做
                    # kernel=3 的滑动平均，输出仍为 [B,T]。
                    d_full = torch.cat([d[:, :1], d], dim=1)  # [B,T]
                    d3 = F.pad(d_full.unsqueeze(1), (1, 1), mode="replicate")
                    r = F.avg_pool1d(d3, kernel_size=3, stride=1).squeeze(1)  # [B,T]
                else:
                    r = x.new_zeros(B, 1, dtype=torch.float32)
                frame_res_frames = r.reshape(-1).detach()   # [B*T]

        # 双流 Head 的高流输入：raw xt 的 patch 内高通。只在空间 patch
        # (ph,pw) 上做去均值；当前 patch_size=(1,2,2)。通道顺序铺成
        # [ph,pw,C]，与后续 unpatchify 的 [pt,ph,pw,C] 完全一致。
        pixel_hp = None
        if self.full_head is not None:
            if pt != 1:
                raise RuntimeError("dual high-freq path currently requires patch_t=1")
            hp = x.reshape(B, C, T, H // ph, ph, W // pw, pw)
            hp = hp - hp.mean(dim=(4, 6), keepdim=True)
            hp = hp.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
            hp = hp.reshape(B * T, H // ph, W // pw, ph * pw * C)
            pixel_hp = hp.permute(0, 3, 1, 2).contiguous()   # [B*T, ph*pw*C, H/ph, W/pw]

        # 只训 Head 时，主干不需要保存任何反向激活。no_grad 必须只覆盖主干，
        # 不能包住后面的 full_head，否则输出头也会失去梯度。
        backbone_ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with backbone_ctx:
            z = self.patchify(x)                        # [B, dim, T/pt, H/ph, W/pw]
            z = z.flatten(2).transpose(1, 2)            # [B, N, dim]
            grid = (T // pt, H // ph, W // pw)
            # 局部归纳偏置：patch token 上做 depthwise spatial conv（零门控起步）
            if self.local_conv is not None:
                z = self.local_conv(z, grid)

            # 北大桥只算一次：4096 -> dim，低分辨率用池化、高分辨率用 per-token
            vlm_cond = None
            if self.vlm_injector is not None and vlm_hidden is not None:
                vlm_cond = self.vlm_injector.bridge(vlm_hidden)   # [B, L, dim]

            # 融合版跨模态（图1 拓扑）：压缩摘要 K/V 进每层注意力场。
            # A = VLM(vlm_cond) 持 α 特权；B = 视频 latent 只看 a' 压缩摘要。
            # 放在 _cond 之前，使 A 侧 α 回注能进 global_cond；默认整段不进图。
            cross_kv = None
            if (self.crossmodal is not None
                    and getattr(self.cfg, "a2_crossmodal_in_attn", False)
                    and crossmodal_budget is not None
                    and float(crossmodal_budget) > 0 and vlm_cond is not None):
                alpha = vlm_cond.mean(dim=1)                 # A 私有全局信号
                a_out, cross_kv = self.crossmodal.compute_cross_kv(
                    vlm_cond, z, alpha)
                vlm_cond = (vlm_cond + float(crossmodal_budget)
                            * (a_out - vlm_cond))             # α 沿 A 路径生效

            # V2-28 分层风格条件编码器：风格 token -> 风格条件(加法入 cond) + 路由 logits(喂 MoE)。
            # style_tokens=None 时整条路不进图；外部若显式传 style_logits 则优先用它（覆盖编码器路由）。
            style_cond = None
            if self.style_encoder is not None and style_tokens is not None:
                style_cond, style_route = self.style_encoder(style_tokens)
                if style_logits is None:
                    style_logits = style_route

            # A1：频带条件只做一次 rfft2，不在 30 个 Block 里逐块 FFT。
            # 输出零初始化；关闭时 band_cond=None，前向与 5200 基线逐 bit 一致。
            band_cond = None
            self._last_band_cond_norm = None
            if self.band_cond_enabled:
                from x0_fourier_band_loss import x0_band_energy_ratios
                _bd = x0_band_energy_ratios(x.detach()).to(
                    dtype=self.band_mlp[0].weight.dtype)
                band_cond = self.band_mlp(_bd)
                self._last_band_cond_norm = float(
                    band_cond.detach().float().norm(dim=-1).mean())

            e, snr_gates = self._cond(t_emb, text_emb, vlm_cond, sigma, ref_latent, grid,
                                      style_cond=style_cond, cur_tokens=z,
                                      band_cond=band_cond)

            # PAU-DiT 语法平面：默认不实例化 → 全零信号 → 逐值等价主线。
            self._syn_tier = None
            self._last_syntax_aux = None
            if self.syntax_compiler is not None:
                if syntax_signals is None:
                    _cond_vec = e.mean(dim=1) if e.dim() == 3 else e
                    _vlm_intent = (vlm_cond.mean(dim=1)
                                   if vlm_cond is not None else None)
                    syntax_signals = self.syntax_compiler(
                        _cond_vec, sigma=sigma, vlm_intent=_vlm_intent,
                        threshold=getattr(self.cfg, "pau_skip_threshold", 0.0))
                # aux loss 是编译器自己的监督图，必须带梯度取出；其余控制信号
                # 才 detach 后注入主干。aux 不进主干的 detach 字典。
                if isinstance(syntax_signals, dict):
                    self._last_syntax_aux = syntax_signals.pop("aux_loss", None)
                # 辅助控制平面信号：注入主干前 detach，切断编译器 <-> 主干/文本 的
                # 跨路径梯度；编译器参数 requires_grad 不变，专用 aux loss 接入后
                # 仍可学习。非张量值（skip_blocks 整数表等）原样保留。
                syntax_signals = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in syntax_signals.items()}
                self._syn_tier = syntax_signals["tier_mod"]
                if skip_blocks is None:
                    skip_blocks = syntax_signals["skip_blocks"]
                if ffn_modulation is None:
                    ffn_modulation = syntax_signals["ffn_mod"]
                if crossmodal_budget is None:
                    _cb = float(syntax_signals["cross_budget"].mean())
                    crossmodal_budget = _cb if _cb > 0 else None

            # a2_v2 语义层接线（图1/非对称特权）：crossmodal 默认不实例化；
            # 仅当 a2_crossmodal=True 且 ASTree 下发 budget>0 且 VLM 条件存在时消费。
            # 用 vlm_cond 作模态 A（α 特权方）、latent token z 作模态 B。
            if (self.crossmodal is not None and cross_kv is None
                    and crossmodal_budget is not None
                    and float(crossmodal_budget) > 0 and vlm_cond is not None):
                # 旧路径（a2_crossmodal_in_attn=False）：一次性 token 融合，向后兼容。
                alpha = e.mean(dim=1) if e.dim() == 3 else e     # [B,dim] 特权信号
                z_orig = z
                _, z = self.crossmodal(vlm_cond, z, alpha)
                z = z_orig + float(crossmodal_budget) * (z - z_orig)  # budget 插值

            # V2-27 全局条件 MoE：全模型算一次 c_expert [B,dim]，再按 gidx 加法注入。
            # 处理的是条件向量（长度 dim=3072）而非 990 token，算力开销 ≈ 0。
            c_expert = None
            if self.global_moe is not None:
                c_expert = self.global_moe(e.mean(dim=1) if e.dim() == 3 else e,
                                           style_logits=style_logits)

            # 全局 block 序号：encoder -> bottleneck -> decoder
            gidx = 0
            # AST 只能软跳非关键块；文本 cross/AdaLN 关键块强制保护。
            _skip = set(skip_blocks or ()) - set(self._protected_blocks)

            # encoder + skip 收集（低分辨率段：共享 embedder 输出 e 已含全局 cond）
            skip_feats = []
            for i, blk in enumerate(self.enc_blocks):
                if gidx not in _skip:
                    z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert),
                                       grid, sigma=sigma,
                                       ffn_modulation=self._ffn_mod_for(ffn_modulation, gidx),
                                       tier_mod=self._tier_mod_for(self._syn_tier, gidx),
                                       cross_kv=cross_kv,
                                       text_tokens=text_tokens,
                                       text_mask=text_mask,
                                       text_valid=text_valid)
                if (self.repa_proj is not None
                        and i == int(getattr(self.cfg, "repa_layer", 8))):
                    # 只存 tap，不在模型里算 DINOv2；训练侧对 x0/真实帧算 target。
                    self._last_repa_feature = z
                skip_feats.append(z)
                gidx += 1
                if self.merges is not None and i < len(self.merges):
                    z, grid = self.merges[i](z, grid)
            if feature_pool is not None:
                feature_pool.append(z.mean(dim=1))

            # V2-29 纯时序专项块（主干中段，encoder 之后）：专治频闪/角色抖动/动作跳变
            if self.temporal_special is not None:
                # 走 _blk_call 纳入 checkpoint（此前直接调用 → 激活全程驻留漏网点）
                if gidx not in _skip:
                    z = self._blk_call(self.temporal_special, z,
                                       self._moe_block_cond(e, snr_gates, gidx, c_expert),
                                       grid, sigma=sigma,
                                       ffn_modulation=self._ffn_mod_for(ffn_modulation, gidx),
                                       tier_mod=self._tier_mod_for(self._syn_tier, gidx),
                                       cross_kv=cross_kv,
                                       text_tokens=text_tokens,
                                       text_mask=text_mask,
                                       text_valid=text_valid)
                gidx += 1  # 专项块也占一个 block 槽位，避免后续注入索引错位

            # 4-tier context/register tokens：只在 bottleneck 入口做一次。
            # read/write 在序列空间完成；tokens 本身不进 (T,H,W)，不破坏 N==T*H*W。
            if self.context_tokens is not None:
                z, self._ctx_tiers, _ctx_cond = self.context_tokens(z)
                e = e + (_ctx_cond.unsqueeze(1) if e.dim() == 3 else _ctx_cond)

            # bottleneck：完整 3D 联合注意力 + v3 文本 cross-attn
            for blk in self.bottleneck:
                if gidx not in _skip:
                    z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert),
                                       grid, sigma=sigma,
                                       ffn_modulation=self._ffn_mod_for(ffn_modulation, gidx),
                                       tier_mod=self._tier_mod_for(self._syn_tier, gidx),
                                       cross_kv=cross_kv,
                                       text_tokens=text_tokens,
                                       text_mask=text_mask,
                                       text_valid=text_valid)
                gidx += 1
            if feature_pool is not None:
                feature_pool.append(z.mean(dim=1))

            # decoder + 加权 skip 融合（高分辨率段：per-token cross-attn 注入）
            for i, blk in enumerate(self.dec_blocks):
                if self.unmerges is not None and i < len(self.unmerges):
                    z, grid = self.unmerges[i](z, grid)
                if i < len(skip_feats):
                    z = self.skips[i](z, skip_feats[-(i + 1)], grid=grid)
                if gidx not in _skip:
                    z = self._blk_call(blk, z, self._moe_block_cond(e, snr_gates, gidx, c_expert),
                                       grid, sigma=sigma,
                                       ffn_modulation=self._ffn_mod_for(ffn_modulation, gidx),
                                       tier_mod=self._tier_mod_for(self._syn_tier, gidx),
                                       cross_kv=cross_kv,
                                       text_tokens=text_tokens,
                                       text_mask=text_mask,
                                       text_valid=text_valid)
                if vlm_cond is not None and cross_kv is None:
                    # 融合版：摘要 K/V 已在每层 attention 注入，跳过独立 cross-attn，
                    # 避免两套 VLM→latent 路径重复（省 n_high × O(N·L) 开销）。
                    z = self.vlm_injector.inject_high(gidx, z, vlm_cond)
                if self.local_conv is not None and getattr(self.cfg, "local_every_dec_block", False):
                    # 每个 decoder block 后再次做深度 3×3 局部混合。只复用同一组
                    # depthwise 权重/gate，不新增参数；直接压 token 网格的 patch 间差异，
                    # 解决「只在 patchify 入口做一次 2% 平滑、30 个 block 后早已失效」。
                    z = self.local_conv(z, grid)
                gidx += 1
            if feature_pool is not None:
                feature_pool.append(z.mean(dim=1))

            # V2-29 纯空间专项块（深层，decoder 之后）：专磨单帧五官/线条，不增加时序计算
            if self.spatial_special is not None:
                # 走 _blk_call 纳入 checkpoint（此前直接调用 → 激活全程驻留漏网点）
                if gidx not in _skip:
                    z = self._blk_call(self.spatial_special, z,
                                       self._moe_block_cond(e, snr_gates, gidx, c_expert),
                                       grid, sigma=sigma,
                                       ffn_modulation=self._ffn_mod_for(ffn_modulation, gidx),
                                       tier_mod=self._tier_mod_for(self._syn_tier, gidx),
                                       cross_kv=cross_kv,
                                       text_tokens=text_tokens,
                                       text_mask=text_mask,
                                       text_valid=text_valid)
                gidx += 1

            # 新增轻量算子（时序低通 / 频域 / 运动头）——都在最终 token 特征上施加
            if self.freq_branch is not None:
                z = self.freq_branch(z)
            if self.temporal_lowpass is not None:
                z = self.temporal_lowpass(z, grid)

            # 最终 token 特征输出前做一次固定强度高斯邻域混合：直接压 patch 均值跳变。
            # 只在末层做 1 次，不像 per-decoder 那样 12 次拖慢 5 倍；不新增参数。
            if self.local_conv is not None:
                s_raw = _os.environ.get("SAKIKO_LOCAL_OUT_STRENGTH")
                s = float(s_raw) if s_raw not in (None, "") else float(
                    getattr(self.cfg, "local_out_strength", 0.0) or 0.0)
                if s > 0:
                    z = self.local_conv.smooth(z, grid, s)

        if backbone_no_grad:
            # 明确边界：Head 输入不需要梯度，但 Head 自身参数仍正常构建反向图。
            z = z.detach()
            e = e.detach()

        self._last_full_head_aux = None
        self._last_token_bd = None
        self._last_corr_defect = None
        self._last_vmag_loss = None
        self._last_out_rms = None
        self._last_rms_target = None
        self._last_band_heads_latent = None
        self._last_band_base_v_latent = None
        self._last_bandlr_heads_latent = None
        self._last_bandlr_base_bands_latent = None
        self._last_bandlr_low_gate = None
        self._last_bandlr_structure_reg = None
        if self.full_head is not None:
            # Full gated head consumes final token features in a per-frame 2D
            # layout. It replaces the old per-token LinearHead entirely; the
            # old head weights are kept only for checkpoint compatibility and
            # are intentionally not used here.
            B0, N0, C0 = z.shape
            Tt, Ht, Wt = grid
            z_frames = (z.view(B0, Tt, Ht, Wt, C0)
                        .permute(0, 1, 4, 2, 3)
                        .reshape(B0 * Tt, C0, Ht, Wt))
            t_frames = t_emb.repeat_interleave(Tt, dim=0)
            ctx = e.mean(dim=1) if e.dim() == 3 else e
            ctx_frames = ctx.repeat_interleave(Tt, dim=0)
            sigma_frames = None
            if sigma is not None:
                sigma_frames = sigma.view(-1, 1).repeat_interleave(Tt, dim=0)
            v_tok, self._last_full_head_aux = self.full_head(
                z_frames, t_frames, ctx_frames,
                frames=Tt, training=self.training, sigma=sigma_frames,
                pixel_hp=pixel_hp, frame_residual=frame_res_frames,
            )
            if self.training:
                # Token-Boundary 梯度连续性：只算低频通道（高频边缘允许跳变）。
                tbd_src = getattr(self.full_head, "_last_feat", None)
                if tbd_src is None:
                    tbd_src = v_tok
                low_gate = getattr(self.full_head, "_last_low_gate", None)
                self._last_token_bd = token_gradient_continuity(
                    tbd_src.permute(0, 2, 3, 1), channel_mask=low_gate)
                self._last_corr_defect = getattr(self.full_head, "_last_corr_defect", None)
                self._last_vmag_loss = getattr(self.full_head, "_last_vmag_loss", None)
                self._last_out_rms = getattr(self.full_head, "_last_out_rms", None)
                self._last_rms_target = getattr(self.full_head, "_last_rms_target", None)
            else:
                self._last_corr_defect = None
                self._last_vmag_loss = None
                self._last_out_rms = None
                self._last_rms_target = None
            pt, ph, pw = self.cfg.patch_size
            # 修复（2026-11-24）：full head 返回 [B*T, out_dim, Ht, Wt]。必须先
            # permute 成 [B*T, Ht, Wt, out_dim]，再按 [pt,ph,pw,C] 拆通道。
            # 旧代码直接 view 会把 C 轴和 W 轴错位成一个全局置换，head 只能
            # 学到 patch 常量先验，是 32px mosaic / pred_patch_res≈0 的根因。
            v = self._unpatchify_full_head(v_tok, B0, Tt, Ht, Wt)

            # A3-v3：三个 raw 头先 unpatchify 到 latent，再在 latent 空间
            # 做硬带限/缩放/残差相加。频带定义与采样评估完全一致。
            bh = getattr(self.full_head, "_last_band_heads", None)
            bb = getattr(self.full_head, "_last_band_base_v", None)
            self._last_band_head_rms = None
            if bh is not None and bb is not None:
                try:
                    from .x0_fourier_band_loss import band_limit_5d
                except ImportError:
                    from x0_fourier_band_loss import band_limit_5d
                base_v = self._unpatchify_full_head(bb, B0, Tt, Ht, Wt)
                s = float(getattr(self.full_head, "band_head_scale_buf", 0.05))
                heads = []
                for _i, _h in enumerate(bh):
                    _hl = band_limit_5d(
                        self._unpatchify_full_head(_h, B0, Tt, Ht, Wt), _i) * s
                    heads.append(_hl)
                v = v + heads[0] + heads[1] + heads[2]
                self._last_band_base_v_latent = base_v.detach()
                self._last_band_heads_latent = tuple(heads)
                self._last_band_head_rms = tuple(
                    float(x.detach().float().pow(2).mean().sqrt()) for x in heads)
            else:
                self._last_band_heads_latent = None
                self._last_band_base_v_latent = None

            # bandhead 线（2026-11-26）：多尺度低秩频带头。full_head 只在
            # token 空间产出 low/mid/high 三分量；这里统一 unpatchify 到
            # latent，再做与评估完全一致的硬带限。
            #   low  = base_low + g*(head_low - base_low)   （软直接，g 可训）
            #   mid  = base_mid + scale*head_mid            （残差）
            #   high = base_high + scale*head_high          （残差；默认冻结为 0）
            # base 取 A3 三头叠加后的 v，保证 A3 与 bandlr 同时开启时不丢分量。
            blr = getattr(self.full_head, "_last_bandlr", None)
            if blr is not None:
                try:
                    from .x0_fourier_band_loss import band_limit_5d
                except ImportError:
                    from x0_fourier_band_loss import band_limit_5d
                base_v = v
                base_l = band_limit_5d(base_v, 0)
                base_m = band_limit_5d(base_v, 1)
                base_h = band_limit_5d(base_v, 2)
                h_l = band_limit_5d(
                    self._unpatchify_full_head(blr[0], B0, Tt, Ht, Wt), 0)
                h_m = band_limit_5d(
                    self._unpatchify_full_head(blr[1], B0, Tt, Ht, Wt), 1)
                h_h = band_limit_5d(
                    self._unpatchify_full_head(blr[2], B0, Tt, Ht, Wt), 2)
                gate_raw = getattr(
                    getattr(self.full_head, "bandlr", None),
                    "low_gate_logit", None)
                if gate_raw is not None:
                    low_gate = torch.sigmoid(gate_raw.float()).to(dtype=v.dtype)
                else:
                    low_gate = v.new_tensor(0.047)
                s = float(getattr(self.full_head, "band_head_scale_buf", 0.05))
                # loss 用的 low 残差对 base 做 detach：v-band 监督只训头，
                # 不把主干已经学到的低频再拉一遍。
                low_eff_loss = low_gate * (h_l - base_l.detach())
                mid_eff = h_m * s
                high_eff = h_h * s
                v = (base_l + low_gate * (h_l - base_l)
                     + base_m + mid_eff + base_h + high_eff)
                self._last_bandlr_base_bands_latent = (
                    base_l.detach(), base_m.detach(), base_h.detach())
                self._last_bandlr_heads_latent = (low_eff_loss, mid_eff, high_eff)
                self._last_bandlr_low_gate = low_gate.detach()
                self._last_bandlr_structure_reg = getattr(
                    self.full_head, "_last_bandlr_structure_reg", None)
            else:
                self._last_bandlr_heads_latent = None
                self._last_bandlr_base_bands_latent = None
                self._last_bandlr_low_gate = None
                self._last_bandlr_structure_reg = None
            aux = None
        else:
            v, aux = self.head(z, grid, cond_e=(e.mean(dim=1) if e.dim() == 3 else e))
        if self.latent_refine is not None:
            v = self.latent_refine(v)
        # 双功能头（方块 + 语义）：在最终 latent 速度场上做可学习残差纠正。
        # 与 local_conv / GatedSkip._lowpass / latent_refine / full_head 均不共享权重。
        if self.dual_sem_head is not None:
            _ctx = e.mean(dim=1) if e.dim() == 3 else e
            v = self.dual_sem_head(v, z, _ctx, grid)
        # 运动头：帧级运动向量（辅助监督用，附在 return_aux 里一并返回，不进主速度场）
        if return_aux:
            if self.motion_head is not None:
                motion = self.motion_head(z, grid)
                return v, aux, motion   # 4b5: (v, aux, motion)
            return v, aux               # tiny/无运动头: 保持 2-tuple
        return v


# 命名兼容层：新代码应使用 UDiT3D / UDiT3DConfig；旧 checkpoint、配置与
# import 继续使用 UViT3D / UViT3DConfig，二者指向同一组类和 state_dict 键。
UDiT3D = UViT3D
UDiT3DConfig = UViT3DConfig


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def disable_all_skips(model: UViT3D):
    """消融：关闭全部 skip 门控（§1.3 红线 8，退化为直筒对照）。"""
    for s in model.skips:
        s.disable()






# ---------------------------------------------------------------------------
# 初始化 / 训练封装（F2 谱初始化 + F5 谱范数冻结）
# ---------------------------------------------------------------------------

def reinit_linear_teacher_spectrum(model: nn.Module, std: float = 0.02,
                                   skip_prefixes: Tuple[str, ...] = ("head_mod", "scale_shift_table",
                                                                      "gate", "log_threshold", "out_gate"),
                                   skip_roots: Tuple[str, ...] = ("ada_embedder",)):
    """F2: 把学生所有 nn.Linear 权重重初始化为 normal(0, std)。
    教师 M85 实测权重 std 0.017-0.028 (~N(0,0.02))，torch 默认 kaiming-uniform
    在 dim=3072 时 std≈0.0104，新模块起点比教师工作点矮 2×。
    跳过零初始化门控/阈值/ AdaLN 表（保持恒等起步语义）。

    ★ 2026-11-23 实测修复：原先只按名字后缀跳过，导致一批「严格零初始化」的层被
    覆盖成 N(0,0.02)（`ada_embedder.embedder.3`、`freq_branch.net.2`、`local_conv`
    内部 Linear 等），AdaLN-Zero 的恒等起步全部失效 —— 200 步短测 step1 loss
    2.807 直接高于零预测 1.578 就是这个原因。
    现在改为：先记录所有「当前严格为 0」的参数名，重初始化后逐个恢复为 0，
    不依赖命名约定；skip_roots 只作额外保险。
    """
    zero_names = [n for n, p in model.named_parameters()
                  if p.numel() and float(p.abs().max().item()) == 0.0]
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(name.endswith(p) or f".{p}" in name for p in skip_prefixes) \
                and not any(name == r or name.startswith(r + ".") for r in skip_roots):
            with torch.no_grad():
                mod.weight.normal_(0, std)
                if mod.bias is not None:
                    mod.bias.zero_()
            n += 1
    params = dict(model.named_parameters())
    restored = 0
    with torch.no_grad():
        for nm in zero_names:
            p = params.get(nm)
            if p is not None:
                p.zero_()
                restored += 1
    if restored:
        print(f"[init] F2 恢复 {restored} 个严格零初始化参数（恒等起步保护）", flush=True)
    return n


def freeze_spectral_power_iteration(model: nn.Module):
    """F5: 预热后冻结 spectral_norm 的 u/v 幂迭代。

    torch 2.9 的 `_SpectralNorm` 没有 `use_parametrization` 开关（旧代码写它是
    no-op，实际每个 forward 仍在跑幂迭代）。这里把 n_power_iterations 置 0：
    `_power_method` 的循环不执行，u/v 不再更新，sigma 固定为当前估计值。
    """
    n = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            try:
                sn = mod.parametrizations.weight[0]
                if hasattr(sn, "n_power_iterations"):
                    sn.n_power_iterations = 0
                    n += 1
            except (AttributeError, IndexError):
                pass
    return n
