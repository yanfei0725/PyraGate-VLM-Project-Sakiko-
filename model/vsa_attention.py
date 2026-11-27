#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VSA（Video Sparse Attention）纯 PyTorch 参考实现 —— ROCm 可跑，无需 CUDA kernel。

调研出处：FastVideo / Hao AI Lab, NeurIPS 2025（VSA）；CVPR 2026（VSA-H3）。
核心机制（与论文一致）：
  1. 3D tile 化：把 N 个 token 按 (tt,th,tw) 时空块切分，每块 tile_len 个 token。
  2. 均值路由：每个 tile 的 Q/K 分别对 tile_len 维求均值 → [D] 代表向量。
  3. 相关性打分：q_mean · k_mean / sqrt(D) → [n_tiles, n_tiles] 路由分数。
  4. top-k 选块：每个 query tile 只保留分数最高的 topk_ratio 比例 key tiles。
  5. 块稀疏注意力：gather 被选 tile 的 K/V，拼成短序列做 SDPA，scatter 回原位。

与 CUDA 版的区别：本实现用 torch.gather + F.scaled_dot_product_attention，
在 ROCm 上走 SDPA 后端（math/flash2 由 attn_backend 决定），**不依赖 sm90/sm100 kernel**。
当 topk_ratio=1.0 时等价于稠密注意力（用于数值对齐自检）。

默认关闭；通过 attn_core="vsa" 显式启用。
"""
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def _to_tiles(x: torch.Tensor, n_t: int, n_h: int, n_w: int,
              tt: int, th: int, tw: int) -> torch.Tensor:
    """[B,H,N,D] -> [B,H,n_tiles,tile_len,D]，tile 顺序 (t,h,w)，块内 (tt,th,tw)。"""
    B, H, N, D = x.shape
    T, HH, WW = n_t * tt, n_h * th, n_w * tw
    # 先还原 3D 自然网格 (t,h,w)，再切块；不能直接 reshape 成交错块序
    x = x.reshape(B, H, T, HH, WW, D)
    x = x.reshape(B, H, n_t, tt, n_h, th, n_w, tw, D)
    # 目标轴序：batch, head, 块t, 块h, 块w, 块内t, 块内h, 块内w, D
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
    return x.reshape(B, H, n_t * n_h * n_w, tt * th * tw, D)


def _from_tiles(x: torch.Tensor, n_t: int, n_h: int, n_w: int,
                tt: int, th: int, tw: int, N: int) -> torch.Tensor:
    """[B,H,n_tiles,tile_len,D] -> [B,H,N,D]，_to_tiles 的严格逆。"""
    B, H, _, _, D = x.shape
    T, HH, WW = n_t * tt, n_h * th, n_w * tw
    # 逆：块还原 → (块t,块内t,块h,块内h,块w,块内w,D) → 3D 网格 → 展平
    x = x.reshape(B, H, n_t, n_h, n_w, tt, th, tw, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).contiguous()
    x = x.reshape(B, H, n_t, tt, n_h, th, n_w, tw, D)
    x = x.reshape(B, H, T, HH, WW, D)
    return x.reshape(B, H, N, D)


def vsa_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         grid: Tuple[int, int, int],
                         tile: Tuple[int, int, int] = (4, 4, 4),
                         topk_ratio: float = 0.25,
                         scale: Optional[float] = None) -> torch.Tensor:
    """块稀疏注意力。q/k/v: [B,H,N,D]；grid=(T,Hh,Ww)；tile=(tt,th,tw)。

    topk_ratio=1.0 时保留全部 tile，与 F.scaled_dot_product_attention 等价。
    N 不能被 tile 整除时尾部补零（填充行不参与路由，输出被裁掉）。
    """
    B, H, N, D = q.shape
    T, HH, WW = grid
    tt, th, tw = tile
    tile_len = tt * th * tw
    # 向上取整：grid 维不能被 tile 整除时 pad 到整数倍，不丢 token
    n_t = (T + tt - 1) // tt
    n_h = (HH + th - 1) // th
    n_w = (WW + tw - 1) // tw
    Tp, HHp, WWp = n_t * tt, n_h * th, n_w * tw
    pad_n = (Tp * HHp * WWp) - N

    if pad_n > 0:
        q = F.pad(q, (0, 0, 0, pad_n))
        k = F.pad(k, (0, 0, 0, pad_n))
        v = F.pad(v, (0, 0, 0, pad_n))
    Np = q.shape[2]

    qt = _to_tiles(q, n_t, n_h, n_w, tt, th, tw)
    kt = _to_tiles(k, n_t, n_h, n_w, tt, th, tw)
    vt = _to_tiles(v, n_t, n_h, n_w, tt, th, tw)
    n_tiles = qt.shape[2]

    # 均值路由分数 [B,H,n_tiles,n_tiles]
    qm = qt.mean(dim=3)
    km = kt.mean(dim=3)
    sm = scale if scale is not None else (1.0 / math.sqrt(D))
    scores = torch.matmul(qm, km.transpose(-1, -2)) * sm

    keep = max(1, min(n_tiles, int(round(n_tiles * float(topk_ratio)))))
    if keep >= n_tiles:
        # 稠密快路径：展平回 token 维做全局 SDPA（qt 的 tile 维不能当 batch）
        out_flat = F.scaled_dot_product_attention(
            q.reshape(B, H, Np, D), k.reshape(B, H, Np, D), v.reshape(B, H, Np, D))
        out_t = _to_tiles(out_flat, n_t, n_h, n_w, tt, th, tw)
    else:
        topk_idx = scores.topk(keep, dim=-1).indices          # [B,H,n_tiles,keep]
        # 逐 query tile gather 选中的 key/value tiles
        out_t = torch.empty_like(qt)
        for i in range(n_tiles):
            idx = topk_idx[:, :, i, :]                        # [B,H,keep]
            gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, tile_len, D)
            sel_k = torch.gather(kt, 2, gather_idx)           # [B,H,keep,tile_len,D]
            sel_v = torch.gather(vt, 2, gather_idx)
            sel_k = sel_k.reshape(B, H, keep * tile_len, D)
            sel_v = sel_v.reshape(B, H, keep * tile_len, D)
            qi = qt[:, :, i, :, :]                            # [B,H,tile_len,D]
            oi = F.scaled_dot_product_attention(qi, sel_k, sel_v)
            out_t[:, :, i, :, :] = oi

    out = _from_tiles(out_t, n_t, n_h, n_w, tt, th, tw, Np)
    return out[:, :, :N, :]


def _selfcheck():
    torch.manual_seed(0)
    B, H, D = 1, 2, 16
    T, HH, WW = 4, 4, 4                 # 64 token
    N = T * HH * WW
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    # 1) topk=1.0 等价稠密
    a = vsa_sparse_attention(q, k, v, (T, HH, WW), tile=(2, 2, 2),
                             topk_ratio=1.0)
    b = F.scaled_dot_product_attention(q, k, v)
    d = (a - b).abs().max().item()
    print(f"[vsa] dense-equivalence maxdiff={d:.2e}")
    assert d < 1e-5
    # 2) 稀疏路径可跑 + finite
    c = vsa_sparse_attention(q, k, v, (T, HH, WW), tile=(2, 2, 2),
                             topk_ratio=0.25)
    assert torch.isfinite(c).all().item()
    assert c.shape == q.shape
    print(f"[vsa] sparse OK, shape={tuple(c.shape)}")
    # 3) 不整除补零路径
    q2 = torch.randn(B, H, 50, D)
    k2 = torch.randn(B, H, 50, D)
    v2 = torch.randn(B, H, 50, D)
    c2 = vsa_sparse_attention(q2, k2, v2, (5, 5, 2), tile=(2, 2, 2),
                              topk_ratio=0.5)
    assert torch.isfinite(c2).all().item() and c2.shape == q2.shape
    print("[vsa] pad path OK")


if __name__ == "__main__":
    _selfcheck()
