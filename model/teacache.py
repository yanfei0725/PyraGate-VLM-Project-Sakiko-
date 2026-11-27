#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TeaCache：去噪步冗余缓存，采样循环级零训练加速（arXiv 2410.11806 等）。

调研出处：TeaCache (Sun et al.)；FramePack / vLLM-Omni / diffusers 实现。
算法（与本实现一致）：
  - 用便宜代理量判断"该不该跳过"：timestep 调制后的 noisy input（或第一块残差）
    的相对 L1，而不是昂贵的模型输出差。
  - 多项式 rescale 把输入差映射到估计输出差，跨步累积 acc。
  - acc > 阈值 → 全算并清零；否则复用上次缓存的残差，跳过主干。
  - 首步/末步强制全算（保证起止质量）。

设计要点：
  - 缓存逻辑放采样循环层，模型 forward 保持静态 → 避开 torch.compile 动态分支坑。
  - 缓存张量保持模型 dtype（bf16 不转 fp16），避免反复切精度。
  - 系数按模型离线拟合；本实现默认用单位多项式（保守），可按模型标定。
"""
import torch
from typing import Optional, List


class TeaCache:
    """采样循环的缓存判定器。不进模型 forward，不训练，不改权重。"""

    def __init__(self, rel_l1_thresh: float = 0.15,
                 poly_coeffs: Optional[List[float]] = None,
                 start_percent: float = 0.0,
                 end_percent: float = 1.0):
        self.thresh = float(rel_l1_thresh)
        # 默认恒等映射 y=x（保守）；论文按模型拟合，如 FLUX=[498.7,-283.8,55.9,-3.82,0.264]
        # Horner: coeffs=[a,b,c,...] 表示 a*x^n + b*x^(n-1) + ...；[1,0] 即 y=x。
        self.coeffs = list(poly_coeffs) if poly_coeffs else [1.0, 0.0]
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.reset()

    def reset(self):
        self.cnt = 0
        self.acc = 0.0
        self.prev_modulated = None
        self.cached_residual = None
        self.step = 0
        self.compute_steps = 0
        self.skip_steps = 0

    def _rescale(self, delta: float) -> float:
        y = 0.0
        for c in self.coeffs:            # Horner 正序：a*x^n + b*x^(n-1) + ...
            y = y * delta + c
        return y

    def _in_window(self, total_steps: int) -> bool:
        if total_steps <= 0:
            return True
        lo = int(self.start_percent * total_steps)
        hi = int(self.end_percent * total_steps)
        return lo <= self.step < hi

    def should_compute(self, modulated_input: torch.Tensor,
                       total_steps: int, is_first: bool = False,
                       is_last: bool = False) -> bool:
        """判定当前步是否需要完整计算。modulated_input: 任意形状张量（如 ADA 调制输入）。"""
        if is_first or is_last:
            self.step += 1
            self.compute_steps += 1
            return True
        if not self._in_window(total_steps):
            self.step += 1
            self.compute_steps += 1
            return True

        with torch.no_grad():
            cur = modulated_input.detach().float()
            if self.prev_modulated is None:
                self.prev_modulated = cur.clone()
                self.step += 1
                self.compute_steps += 1
                return True
            denom = self.prev_modulated.abs().mean().clamp(min=1e-6)
            rel_l1 = float((cur - self.prev_modulated).abs().mean() / denom)
            self.prev_modulated = cur.clone()
            self.acc += self._rescale(rel_l1)

        compute = self.acc > self.thresh
        if compute:
            self.acc = 0.0
            self.compute_steps += 1
        else:
            self.skip_steps += 1
        self.step += 1
        return compute

    def cache(self, residual: torch.Tensor):
        """全算后缓存残差（保持 dtype，不转精度）。"""
        self.cached_residual = residual.detach().clone()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """跳步时返回缓存残差；调用方负责把它加到当前输入。"""
        if self.cached_residual is None:
            return torch.zeros_like(x)
        r = self.cached_residual.to(dtype=x.dtype, device=x.device)
        if r.shape != x.shape:
            return torch.zeros_like(x)
        return r

    def summary(self) -> str:
        total = self.compute_steps + self.skip_steps
        rate = (self.skip_steps / total) if total else 0.0
        return (f"[teacache] compute={self.compute_steps} skip={self.skip_steps} "
                f"skip_rate={rate:.1%} acc={self.acc:.4f}")


def _selfcheck():
    torch.manual_seed(0)
    # 场景A：变化大 → 应全部全算（判定器不误跳）
    tc_a = TeaCache(rel_l1_thresh=0.15)
    for s in range(20):
        mod = torch.randn(4, 8)
        tc_a.should_compute(mod, 20, is_first=(s == 0), is_last=(s == 19))
    assert tc_a.compute_steps == 20
    print("[teacache] high-change OK", tc_a.summary())

    # 场景B：相邻步同向渐变（贴近真实去噪轨迹）→ 应触发跳步
    tc_b = TeaCache(rel_l1_thresh=0.5)
    base = torch.randn(4, 8)
    drift = torch.randn(4, 8) * 0.002
    cur = base.clone()
    for s in range(20):
        cur = cur + drift
        mod = cur
        need = tc_b.should_compute(mod, 20, is_first=(s == 0), is_last=(s == 19))
        if need:
            tc_b.cache(torch.randn(4, 8))
        else:
            r = tc_b.apply(torch.randn(4, 8))
            assert r.shape == (4, 8) and torch.isfinite(r).all()
    print("[teacache] low-change OK", tc_b.summary())
    assert tc_b.skip_steps > 0 and tc_b.compute_steps < 20
    print("[teacache] selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
