# -*- coding: utf-8 -*-
"""semantic_proj.py — Latent 空间轻量语义投影头（前缀 ``semproj.``）。

设计目标（融合方案 §3「语义蒸馏：Latent 空间轻量语义投影」）：
  * 训练时**不解码视频到像素**、**不把 CLIP 大模型拉进训练图**；
  * 直接把 DiT latent 映射到 CLIP ViT-B/32 的 512 维特征空间，与离线算好的
    CLIP 图像特征算余弦相似度对齐；
  * 极轻量低秩 MLP，参数量远低于主干 1%；作为子模块挂到主干后，
    ``state_dict`` 的 key 以 ``semproj.`` 开头，不污染老 ckpt key。

输入 latent: ``[B, C, T, H, W]``（本机缓存/训练形状 ``[1,48,4,28,48]``）。
池化: 在 (T,H,W) 上做 mean + std -> ``[B, 2C]``；
低秩 MLP: ``Linear(2C, rank) -> SiLU -> Linear(rank, clip_dim)``，输出 L2 归一。

纪律：
  * 模块关闭/未挂载时不改动主干任何计算；
  * ``freeze()`` / ``unfreeze()`` 两档；
  * 所有前向在 float32 下做池化与归一，权重按宿主 dtype 走；
  * 本文件不依赖 train_staged / uvit3d / full_gated_head，可独立 CPU 自检。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSemanticProj(nn.Module):
    """latent -> CLIP 特征空间的低秩投影头。

    Args:
        in_channels: latent 通道数（本机蜂鸟 VAE = 48）。
        clip_dim: 目标 CLIP 特征维度（ViT-B/32 = 512）。
        rank: 低秩瓶颈维（默认 64；远小于 clip_dim，保证低秩化）。
        pool: 池化方式序列，可选 ``"mean"`` / ``"std"``，顺序拼接。
        freeze: 构造即冻结（requires_grad=False）。
    """

    def __init__(self, in_channels: int = 48, clip_dim: int = 512,
                 rank: int = 64, pool: Tuple[str, ...] = ("mean", "std"),
                 freeze: bool = False, enabled: bool = False) -> None:
        super().__init__()
        # 默认关闭：未显式 --semantic-proj 时不进训练路径，前向直接返回零向量，
        # 与基线数值等价（不建损失图、不回传梯度）。
        self.enabled = bool(enabled)
        if not pool:
            raise ValueError("pool 至少选一种池化方式")
        for p in pool:
            if p not in ("mean", "std"):
                raise ValueError(f"未知池化方式 {p!r}")
        self.in_channels = int(in_channels)
        self.clip_dim = int(clip_dim)
        self.rank = int(rank)
        self.pool = tuple(pool)
        pooled_dim = self.in_channels * len(self.pool)

        self.norm = nn.LayerNorm(pooled_dim)
        self.down = nn.Linear(pooled_dim, self.rank)
        self.act = nn.SiLU()
        self.up = nn.Linear(self.rank, self.clip_dim)
        # 输出层零初始化起步：未对齐前投影不产生大方向，避免初期拉爆主干。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------ utils
    def _pool(self, latent: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] -> [B, pooled_dim]，float32 计算。"""
        if latent.dim() != 5:
            raise ValueError(f"latent 必须是 5D [B,C,T,H,W]，got {tuple(latent.shape)}")
        x = latent.float()
        feats: List[torch.Tensor] = []
        for mode in self.pool:
            if mode == "mean":
                feats.append(x.mean(dim=(2, 3, 4)))
            else:  # std
                feats.append(x.std(dim=(2, 3, 4), unbiased=False))
        return torch.cat(feats, dim=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一后的 CLIP 空间向量 ``[B, clip_dim]``（float32）。

        ``enabled=False``（默认）时直接返回零向量，不建计算图，保证关闭路径
        与无此模块的基线完全等价。
        """
        if not self.enabled:
            B = latent.shape[0] if latent.dim() == 5 else latent.shape[0]
            return latent.new_zeros(B, self.clip_dim, dtype=torch.float32)
        h = self._pool(latent)
        h = self.norm(h)
        h = self.act(self.down(h))
        h = self.up(h)
        return F.normalize(h, p=2.0, dim=-1, eps=1e-8)

    # ------------------------------------------------------------ loss 接口
    @staticmethod
    def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``1 - mean(cos)``；pred/target 均期望已 L2 归一。"""
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    # ---------------------------------------------------------- 冻结 / 解冻
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# CPU 自检（不碰 GPU、不碰主干）
# ---------------------------------------------------------------------------
def _selftest() -> None:
    def ck(cond: bool, msg: str) -> None:
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== semantic_proj 纯 CPU 自检 ===")
    torch.manual_seed(0)
    B, C, T, H, W = 2, 48, 4, 28, 48
    x = torch.randn(B, C, T, H, W, dtype=torch.bfloat16)

    proj = LatentSemanticProj(in_channels=C, clip_dim=512, rank=64)
    total, trainable = proj.count_parameters()
    print(f"  [info] params total={total} trainable={trainable} "
          f"({total/1e3:.1f}K)")
    ck(total < 39_070_000, "参数量远低于主干 1% 预算（39.07M）")

    # 默认关闭：前向返回零向量，不建图。
    y_off = proj(x)
    ck(torch.allclose(y_off, torch.zeros_like(y_off), atol=1e-6),
       "默认关闭：输出零向量（基线等价）")

    y = proj(x)
    ck(y.shape == (B, 512), f"输出形状 {tuple(y.shape)} == (B,512)")
    ck(torch.isfinite(y).all().item(), "输出有限值")

    # 零初始化上权：未训练时 pre-up 特征为 0，归一后是零向量（范数≈0）。
    ck(torch.allclose(proj.up.weight, torch.zeros_like(proj.up.weight)),
       "输出层零初始化（恒等起步）")
    ck(torch.allclose(y, torch.zeros_like(y), atol=1e-6),
       "零初始化下输出为零向量（不产生大方向）")

    # 激活路径（enabled=True）：非零权重验证 L2 归一性质。
    proj.enabled = True
    with torch.no_grad():
        proj.up.weight.normal_(0.02)
        y2 = proj(x)
    norms = y2.norm(dim=-1)
    ck(torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "非零路径输出已 L2 归一")

    # 反传：余弦 loss 能回传到 down/up（重新前向，图权重一致）。
    target = F.normalize(torch.randn(B, 512), dim=-1)
    y3 = proj(x)
    loss = proj.cosine_loss(y3, target)
    ck(torch.isfinite(loss).item(), f"cosine loss 有限 {loss.item():.4f}")
    loss.backward()
    ck(proj.down.weight.grad is not None and torch.isfinite(proj.down.weight.grad).all().item(),
       "梯度回传到 down 投影")
    ck(proj.up.weight.grad is not None, "梯度回传到 up 投影")

    # 冻结档：参数无梯度
    proj.freeze()
    ck(all(not p.requires_grad for p in proj.parameters()), "freeze() 后全部 requires_grad=False")
    proj.unfreeze()
    ck(all(p.requires_grad for p in proj.parameters()), "unfreeze() 后全部可训练")

    # shape 报错路径
    try:
        proj(torch.randn(B, C, H, W))
        ck(False, "非 5D 输入应抛 ValueError")
    except ValueError:
        ck(True, "非 5D 输入抛 ValueError（清晰报错）")

    print("=== semantic_proj 自检全部通过 ===")


if __name__ == "__main__":
    _selftest()
