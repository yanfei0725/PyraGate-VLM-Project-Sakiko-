#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAU-DiT 语法平面：SyntaxCompiler + 金字塔 tier 调度（默认全关，零门控恒等）。

设计契约（保 step3200 热启动）：
- 所有输出投影 **零初始化** + 主 gate 零初始化 → 起步输出严格为 0。
- 输出 0 时：skip_blocks=None / ffn_modulation=None / crossmodal_budget=None /
  tier_mod=None → 前向逐值等价 a2_v2 融合后的主线（已验证 maxdiff=0）。
- 不 reshape、不改任何现有权重；`strict=False` 直载 step3200 只补新参数。

三层信号：
1. skip_logits [B, n_blocks]：按块跳过调度（ASTree 在环的接口）。
2. ffn_mod    [B, n_blocks, dim]：VLM→VLM→FFN 逐块调制。
3. cross_budget [B, n_blocks]：跨模态非对称特权预算。
4. tier_mod   [B, n_tiers, dim]：金字塔层级（scene/actor/part/pixel）的 AdaLN 增量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

_TIER_NAMES = ("scene", "actor", "part", "pixel")


class SyntaxCompiler(nn.Module):
    """把共享条件 + 噪声级 + VLM 意图编译成逐块/逐层调度信号。

    全部零初始化，forward 起步输出严格为 0（恒等）。
    """

    def __init__(self, dim: int, n_blocks: int, n_tiers: int = 4,
                 rank: int = 64, vlm_dim: int = 0, gate_init: float = 0.01):
        super().__init__()
        self.dim = int(dim)
        self.n_blocks = int(n_blocks)
        self.n_tiers = int(n_tiers)
        self.rank = int(rank)
        self.vlm_dim = int(vlm_dim)
        in_dim = dim + 1 + (vlm_dim if vlm_dim > 0 else dim)
        self.intent = nn.Sequential(nn.Linear(in_dim, rank), nn.SiLU())
        # 四个头全部零初始化
        self.skip_head = nn.Linear(rank, n_blocks)
        self.ffn_head = nn.Linear(rank, n_blocks * dim)
        self.cross_head = nn.Linear(rank, n_blocks)
        self.tier_head = nn.Linear(rank, n_tiers * dim)
        for h in (self.skip_head, self.ffn_head, self.cross_head, self.tier_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        # 主 gate 必须非零：heads 全零时输出仍严格为 0，但 head 梯度
        # = gate * h 非零，训练第一步就能解锁；gate=0+heads=0 会让
        # gate 与四个 head 互锁，数学上永远拿不到梯度（P2-B）。
        # 保持 shape=(1,)：旧 ckpt 的 syntax_compiler.gate 就是 (1,)，
        # 改成标量 () 会触发 train_staged 的形状守卫 fatal。
        self.gate = nn.Parameter(torch.tensor([float(gate_init)]))

    def auxiliary_loss(self, cond: torch.Tensor, sigma: Optional[torch.Tensor],
                       h: torch.Tensor, skip: torch.Tensor, ffn: torch.Tensor,
                       cross: torch.Tensor, tier: torch.Tensor) -> torch.Tensor:
        """Detached-target auxiliary loss for the syntax plane.

        主干消费的是 detach 后的调度信号（保护主干），所以编译器必须有一条
        独立的监督路径。这里用四个自监督目标把所有头拉进图：
          * skip：低 σ 允许更稀疏（target 0.05~0.20），BCE 给 skip_head 梯度；
          * ffn：重建当前条件向量（detach），MSE 给 ffn_head/intent 梯度；
          * cross：目标为 σ（高噪多开跨模态），MSE 给 cross_head 梯度；
          * tier：重建当前条件向量（detach），MSE 给 tier_head 梯度；
          * gate/intent 轻量锚定，保证零初始化头在第一步也有非零梯度。
        所有 target 都 detach；aux 只回传编译器，不穿过主干。
        """
        B = cond.shape[0]
        # 不能直接用 layer_norm(cond) 当 target：初始化阶段共享条件 e 可能恰好
        # 为零，target 全零会让 ffn/tier 两个头 MSE 梯度也为零。加 0.02 正值
        # 锚点，既保持小调制（1+0.02~1.1），又保证两个头在 step0 见梯度。
        target = 0.02 + 0.10 * F.layer_norm(
            cond.detach().float(), (self.dim,)).to(dtype=ffn.dtype)
        ffn_t = target[:, None, :].expand_as(ffn)
        tier_t = target[:, None, :].expand_as(tier)
        loss_ffn = F.mse_loss(ffn.float(), ffn_t.float())
        loss_tier = F.mse_loss(tier.float(), tier_t.float())
        if sigma is None:
            s = cond.new_zeros(B, 1, dtype=torch.float32)
        else:
            s = sigma.detach().float().reshape(-1, 1)
            if s.shape[0] == 1 and B != 1:
                s = s.expand(B, 1)
            s = s[:B].clamp(0.0, 1.0)
        # 低 σ 更允许跳过；target 不取 0.5，避免 logits=0 时 BCE 梯度恰好为 0。
        skip_t = (0.05 + 0.15 * (1.0 - s)).clamp(0.02, 0.20)
        skip_t = skip_t.expand(B, self.n_blocks)
        loss_skip = F.binary_cross_entropy_with_logits(skip.float(), skip_t)
        cross_t = s.expand(B, self.n_blocks)
        loss_cross = F.mse_loss(cross.float(), cross_t)
        # gate 锚到 0.02（当前 init 0.01），intent 的均值锚到 1.0；给零初始化
        # 头之外的两个上游参数在 step0 就有梯度，避免“解锁要等第二步”。
        loss_gate = (self.gate.float() - 0.02).pow(2).mean()
        loss_intent = (h.float().mean(dim=-1) - 1.0).pow(2).mean()
        return (0.5 * (loss_skip + loss_ffn + loss_cross + loss_tier)
                + 0.01 * (loss_gate + loss_intent))

    def forward(self, cond: torch.Tensor, sigma: Optional[torch.Tensor] = None,
                vlm_intent: Optional[torch.Tensor] = None,
                threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """cond: [B,dim]；sigma: [B,1] 或 None；vlm_intent: [B,vlm_dim] 或 None。"""
        B = cond.shape[0]
        dev, dt = cond.device, cond.dtype
        if sigma is None:
            sigma = torch.zeros(B, 1, device=dev, dtype=dt)
        else:
            if not torch.is_tensor(sigma):
                sigma = torch.as_tensor(float(sigma), device=dev, dtype=dt).reshape(1, 1)
            else:
                sigma = sigma.to(device=dev, dtype=dt)
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(-1)
            elif sigma.ndim == 0:
                sigma = sigma.reshape(1, 1)
            if sigma.shape[0] == 1 and B != 1:
                sigma = sigma.expand(B, 1)
        if vlm_intent is None:
            vlm_intent = torch.zeros(B, self.vlm_dim if self.vlm_dim > 0 else self.dim,
                                     device=dev, dtype=dt)
        vlm_intent = vlm_intent.to(device=dev, dtype=dt)
        if vlm_intent.shape[-1] != (self.vlm_dim if self.vlm_dim > 0 else self.dim):
            # 维度不符时不进图：返回全零信号（严格恒等）
            z = torch.zeros(B, self.n_blocks, device=dev, dtype=dt)
            return {
                "skip_logits": z,
                "skip_blocks": None,
                "ffn_mod": torch.zeros(B, self.n_blocks, self.dim, device=dev, dtype=dt),
                "cross_budget": z,
                "tier_mod": torch.zeros(B, self.n_tiers, self.dim, device=dev, dtype=dt),
                "aux_loss": None,
            }
        h = self.intent(torch.cat([cond, sigma, vlm_intent], dim=-1))
        g = self.gate.to(dt)
        skip = self.skip_head(h) * g                       # [B,n_blocks]
        ffn = (self.ffn_head(h) * g).view(B, self.n_blocks, self.dim)
        cross = self.cross_head(h) * g                     # [B,n_blocks]
        tier = (self.tier_head(h) * g).view(B, self.n_tiers, self.dim)
        # 离散 skip 决策：仅当 batch 均值超过阈值才跳过（ASTree 在环接口）
        active = (skip.mean(dim=0) > float(threshold)).nonzero(as_tuple=False).flatten().tolist()
        return {
            "skip_logits": skip,
            "skip_blocks": active if active else None,
            "ffn_mod": ffn,
            "cross_budget": cross,
            "tier_mod": tier,
            # train mode 才建 aux 图；eval/probe 不增加额外计算。
            "aux_loss": (self.auxiliary_loss(cond, sigma, h, skip, ffn, cross, tier)
                         if self.training else None),
        }


def tier_index(gidx: int, n_blocks: int, n_tiers: int) -> int:
    """把全局 block 序号映射到金字塔 tier（等分，粗→细）。"""
    if n_tiers <= 1:
        return 0
    return min(n_tiers - 1, int(gidx * n_tiers) // max(1, n_blocks))




def _selfcheck():
    torch.manual_seed(0)
    c = SyntaxCompiler(64, 8, n_tiers=4, rank=16)
    out = c(torch.randn(2, 64), torch.rand(2, 1), torch.randn(2, 64))
    assert out["skip_blocks"] is None
    for k in ("skip_logits", "ffn_mod", "cross_budget", "tier_mod"):
        assert torch.count_nonzero(out[k]) == 0, k
    print("[syntax_compiler] zero-init identity OK")


if __name__ == "__main__":
    _selfcheck()
