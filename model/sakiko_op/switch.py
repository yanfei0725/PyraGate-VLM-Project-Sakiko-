# -*- coding: utf-8 -*-
"""sakiko_op.switch -- 模型侧切换分支 + 消融开关 + 命中统计（2026-11-21）。

用法（嵌入 Blend 基座旁路，例如 nodes_pyragate 的 attn2_patch 残差处）::

    from sakiko_op.switch import gated_residual
    # 原代码： x = x + state["gate"] * res
    x = gated_residual(x, res, state["gate"], tag="attn2_patch")

开关（优先级：显式参数 > 环境变量 > 默认 False）：
    SAKIKO_CUSTOM_OP=1 时走注册算子（计算图出现 sakiko.fused_gate_residual），
    否则走原生 ``x + gate*branch``（数值逐位等价）。

统计：hit 统计按 tag 计数（采样期/评测日志直接可读），线程安全用 GIL 即可。
"""
from __future__ import annotations

import os
import threading

import torch

from .op import fused_gate_residual

_LOCK = threading.Lock()
_HITS: dict = {}


def use_custom_op(explicit: bool | None = None) -> bool:
    """读开关：explicit 参数优先，其次 SAKIKO_CUSTOM_OP 环境变量（"1"/"true"）。"""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("SAKIKO_CUSTOM_OP", "0").lower() in ("1", "true", "yes")


def _record(tag: str, path: str):
    with _LOCK:
        rec = _HITS.setdefault(tag, {"custom_op": 0, "native": 0})
        rec[path] += 1


def hits(reset: bool = False) -> dict:
    with _LOCK:
        out = {k: dict(v) for k, v in _HITS.items()}
    if reset:
        _HITS.clear()
    return out


def gated_residual(x: torch.Tensor, branch: torch.Tensor, gate, tag: str = "default",
                   use_op: bool | None = None) -> torch.Tensor:
    """门控旁路残差的统一入口（开关分支处）。

    数学上两条路径逐位等价（gate==0 均为严格恒等）；差别只在计算图形态：
    custom_op 路径出现 sakiko.fused_gate_residual 节点，native 路径保留
    aten.mul + aten.add——供 FX 后端匹配/消融对照。
    """
    if use_op is None:
        use_op = use_custom_op()
    if use_op:
        _record(tag, "custom_op")
        return fused_gate_residual(x, branch, gate)
    _record(tag, "native")
    if isinstance(gate, torch.Tensor):
        if gate.ndim == 0 and float(gate) == 0.0:
            return x
        return x + gate * branch
    if float(gate) == 0.0:
        return x
    return x + gate * branch
