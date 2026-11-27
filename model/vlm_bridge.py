"""
PyraGate-VLM v2.0 — VLM 接入（北大桥 + 多尺度分层注入）
=========================================================
参考本地北大桥（Qwen3-VL hidden -> Wan 条件空间）的精确定义：
  `backups/baseline_v1_20261121/custom_nodes_ComfyUI-WanVLMBridge/wan_vlm_bridge.py`

北大桥结构（逐行对齐 LoomVideo `src/models/transformers/loomvideo.py:30-58`）：
  Qwen3-VL hidden [B,L,4096]
    -> QwenRMSNorm(输入端, fp32 累加)
    -> Linear(4096, 4*3072=12288) -> GELU(tanh)
    -> Linear(12288, 3072)
    -> Wan 条件空间 [B,L,3072]

多尺度分层注入（白皮书 §4.3 / 用户选择）：
  - 低分辨率 stage：桥输出**全局池化** -> AdaLN cond（粗语义：构图/运动）
  - 高分辨率 stage：桥输出 **per-token** -> CrossAttn KV（细粒度：材质/笔触/身份）

权重可加载：gen_model.pth 里 `mllm_embedder.*` 5 个张量（bf16）。
纯 torch，不 import diffusers/comfy/transformers。可纯 CPU 运行。
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 北大桥本体（与本地参考实现逐行同构）
# ---------------------------------------------------------------------------

class QwenRMSNorm(nn.Module):
    """等价于 Qwen3VLTextRMSNorm：fp32 算 rsqrt，乘 weight 后转回输入 dtype。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * xf.to(input_dtype)


class NorthBridge(nn.Module):
    """北大桥：VLM hidden (4096) -> 主干条件空间 (3072)。
    结构与权重键与 LoomVideo gen_model.pth `mllm_embedder.*` 对齐。
    """
    BRIDGE_PREFIX = "mllm_embedder."

    def __init__(self, in_features: int = 4096, hidden_size: int = 3072,
                 out_features: Optional[int] = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.norm = QwenRMSNorm(in_features)
        self.linear_1 = nn.Linear(in_features, 4 * hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh") if act_fn == "gelu_tanh" else nn.SiLU()
        self.linear_2 = nn.Linear(4 * hidden_size, out_features, bias=True)

    def forward(self, vlm_hidden: torch.Tensor) -> torch.Tensor:
        """vlm_hidden: [B, L, 4096] -> [B, L, 3072]"""
        h = self.norm(vlm_hidden)
        h = self.linear_1(h)
        h = self.act_1(h)
        return self.linear_2(h)

    def expected_shapes(self):
        mid = 4 * self.hidden_size
        return {
            "norm.weight": (self.in_features,),
            "linear_1.weight": (mid, self.in_features),
            "linear_1.bias": (mid,),
            "linear_2.weight": (self.out_features, mid),
            "linear_2.bias": (self.out_features,),
        }

    @torch.no_grad()
    def load_from_gen_model(self, ckpt_path: str, dtype=torch.bfloat16, verbose=True):
        """从 gen_model.pth 只取桥的 5 个张量（mmap，避免搬动 9.7GB）。
        Returns report；严格校验形状，不符即报错不静默改结构。
        """
        import os
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        want = self.expected_shapes()
        sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        picked, missing = {}, []
        for k, shape in want.items():
            full = self.BRIDGE_PREFIX + k
            if full not in sd:
                missing.append(full)
                continue
            t = sd[full]
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"桥权重形状不符: {full} 实际 {tuple(t.shape)} 期望 {shape}")
            picked[k] = t
        if missing:
            raise KeyError(f"gen_model.pth 缺少桥权重: {missing}")
        self.load_state_dict({k: v.to(dtype).clone() for k, v in picked.items()}, strict=True)
        self.to(device="cpu", dtype=dtype).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        report = {"keys": {k: tuple(v.shape) for k, v in picked.items()},
                  "n_params": sum(p.numel() for p in self.parameters()), "dtype": str(dtype),
                  "size_mb": sum(p.numel() for p in self.parameters()) * 2 / 1024 / 1024}
        if verbose:
            print(f"[NorthBridge] strict load OK: {report['n_params']:,} params "
                  f"({report['size_mb']:.1f} MB {dtype})")
        del sd
        return report


# ---------------------------------------------------------------------------
# 多尺度条件注入器（白皮书 §4.3：低分辨率全局 / 高分辨率 per-token）
# ---------------------------------------------------------------------------

class CrossAttnInjector(nn.Module):
    """高分辨率注入：U-ViT token 作 Q，VLM per-token 特征作 K/V。
    门控零初始化 -> 初始恒等（延续 v1.2 ref=0 逐 bit 恒等纪律）。

    low_rank>0 时 q/k/v/o 用瓶颈投影 dim->r->dim（参数 4*d*r），
    实测 r=512 每层 12.6M vs 全秩 37.75M；12 层 + 桥 ≈164M。
    """
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 low_rank: int = 0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.low_rank = int(low_rank)
        self.norm_q = QwenRMSNorm(dim, eps)
        self.norm_k = QwenRMSNorm(dim, eps)
        if self.low_rank > 0 and self.low_rank < dim:
            # 低秩：down->up 两段，等效秩 r；注意力内维仍用 dim/num_heads（保持 head 结构）
            r = self.low_rank
            self.q = nn.Sequential(nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False))
            self.k = nn.Sequential(nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False))
            self.v = nn.Sequential(nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False))
            self.o = nn.Sequential(nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False))
        else:
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.o = nn.Linear(dim, dim, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(1))  # 零初始化：初始不注入

    def forward(self, x: torch.Tensor, vlm_cond: torch.Tensor) -> torch.Tensor:
        """x: [B,N,dim] U-ViT token; vlm_cond: [B,L,dim] 桥输出 per-token。"""
        B, N, C = x.shape
        L = vlm_cond.shape[1]
        q = self.norm_q(self.q(x)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(self.k(vlm_cond)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(vlm_cond).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)          # [B,nh,N,hd]
        y = y.transpose(1, 2).reshape(B, N, C)
        return x + torch.tanh(self.out_gate) * self.o(y)

    def disable(self):
        with torch.no_grad():
            self.out_gate.zero_()


class VLMInjector(nn.Module):
    """多尺度分层注入管理器。
    - bridge: 北大桥（4096->3072），产出 per-token 条件
    - global_proj: 桥输出池化后投影到 AdaLN cond（低分辨率用）
    - cross_injectors: 高分辨率各 Block 的 CrossAttn 注入器列表

    stage 划分由 stage_ratio 决定：前 ratio 的 block 用全局（低分辨率），
    其余用 per-token cross-attn（高分辨率）。画风专家只在最高分辨率（白皮书 §5.4）。
    """
    def __init__(self, dim: int, num_heads: int, n_blocks: int,
                 vlm_dim: int = 4096, stage_ratio: float = 0.5,
                 bridge: Optional[NorthBridge] = None, low_rank: int = 0,
                 bridge_rank: int = 0):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.stage_ratio = stage_ratio
        self.low_rank = int(low_rank)
        self.split_idx = int(n_blocks * stage_ratio)   # 前 split_idx 个 block = 低分辨率
        if bridge is not None:
            self.bridge = bridge
        elif int(bridge_rank) > 0:
            # 低秩北大桥：4*rank 中间维，参数 ≈ 4*d_in*rank + 4*rank*d_out。
            # 默认 bridge_rank=0 保持原 NorthBridge（可严格 load gen_model.pth）。
            self.bridge = NorthBridge(vlm_dim, hidden_size=int(bridge_rank),
                                      out_features=dim)
        else:
            self.bridge = NorthBridge(vlm_dim, dim)
        # 全局路径：池化桥输出 -> AdaLN cond 维度（注入到现有 cond）
        self.global_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        nn.init.zeros_(self.global_proj[-1].weight)
        nn.init.zeros_(self.global_proj[-1].bias)
        # per-token 路径：每个高分辨率 block 一个 cross-attn 注入器
        self.cross_injectors = nn.ModuleList([
            CrossAttnInjector(dim, num_heads, low_rank=self.low_rank)
            for _ in range(n_blocks - self.split_idx)
        ])

    def global_cond(self, vlm_cond: torch.Tensor) -> torch.Tensor:
        """低分辨率粗语义：per-token 池化 -> [B,dim]，用于加进 AdaLN cond。"""
        return self.global_proj(vlm_cond.mean(dim=1))

    def inject_high(self, idx: int, x: torch.Tensor, vlm_cond: torch.Tensor) -> torch.Tensor:
        """高分辨率 per-token 注入：idx 为全局 block 序号。"""
        j = idx - self.split_idx
        if 0 <= j < len(self.cross_injectors):
            return self.cross_injectors[j](x, vlm_cond)
        return x

    def is_low_stage(self, idx: int) -> bool:
        return idx < self.split_idx

    def disable_all(self):
        for inj in self.cross_injectors:
            inj.disable()
