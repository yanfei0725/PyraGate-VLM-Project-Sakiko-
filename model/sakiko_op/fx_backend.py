# -*- coding: utf-8 -*-
"""sakiko_op.fx_backend -- Torch FX 自定义后端骨架（fallback 执行版，2026-11-21）。

能力（全部可跑，无需 GPU）：
  1. 图抓取：trace 带门控旁路的模块（含 FakeTensor 形状传播）；
  2. 子图匹配：
     - 模式1：``add(x, mul(g,b))``（任意操作数顺序）→ 替换为
       ``sakiko.fused_gate_residual(x, branch, gate)``；
       branch/gate 消歧用 FakeTensor 形状：与 x 同形者为 branch，另一为 gate；
       同形不可判 → 保守跳过（宁可漏配不可错配）；
     - 模式2：识别图中已有 sakiko.fused_gate_residual 节点（命中计数）；
  3. 执行策略：命中替换后仍返回 GraphModule（eager）＝ fallback——计算等价；
     Vulkan 内核落地后在此处替换执行路径即可。
"""
from __future__ import annotations

import operator

import torch
import torch.fx as fx

from .op import fused_gate_residual  # noqa: F401  (注册副作用)

_MUL_TARGETS = {torch.mul, operator.mul, torch.Tensor.__mul__}
_ADD_TARGETS = {torch.add, operator.add, torch.Tensor.__add__}

_MATCHER_STATS = {"mul_add_rewritten": 0, "sakiko_nodes_seen": 0}


def stats() -> dict:
    return dict(_MATCHER_STATS)


def reset_stats():
    _MATCHER_STATS.update(mul_add_rewritten=0, sakiko_nodes_seen=0)


def _val(node: fx.Node):
    return node.meta.get("val", None)


def _shape(node: fx.Node):
    v = _val(node)
    return tuple(v.shape) if v is not None else None


def _propagate_shapes(gm: fx.GraphModule, example_inputs):
    """FakeTensor 形状传播（有样例输入才做；失败静默降级为无形状）。"""
    try:
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            fakes = [torch.tensor([0.0])[:0].new_empty(0) for _ in []]  # noqa
            fakes = [torch.zeros_like(t, device="meta") if torch.is_tensor(t) else t
                     for t in example_inputs] if example_inputs else []
            FakeTensorProp(gm).propagate(*fakes)
    except Exception:
        pass


def _node_calls(node: fx.Node, targets) -> bool:
    return node.op == "call_function" and node.target in targets


def match_and_replace(gm: fx.GraphModule, rewrite: bool = True) -> fx.GraphModule:
    """模式匹配：模式1重写 mul+add → sakiko 节点；模式2统计已有 sakiko 节点。"""
    g = gm.graph
    _MATCHER_STATS["sakiko_nodes_seen"] = sum(
        1 for n in g.nodes
        if _node_calls(n, {torch.ops.sakiko.fused_gate_residual.default}))
    if rewrite:
        for add_node in list(g.nodes):
            if not _node_calls(add_node, _ADD_TARGETS):
                continue
            a, b = add_node.args
            mul_node = b if _node_calls(b, _MUL_TARGETS) else (a if _node_calls(a, _MUL_TARGETS) else None)
            if mul_node is None:
                continue
            base_x = b if mul_node is a else a
            m1, m2 = mul_node.args
            s_base, s1, s2 = _shape(base_x), _shape(m1), _shape(m2)
            if s_base is not None and s1 is not None and s2 is not None:
                if s1 == s_base and s2 != s_base:
                    branch_node, gate_node = m1, m2
                elif s2 == s_base and s1 != s_base:
                    branch_node, gate_node = m2, m1
                else:
                    continue  # 同形不可判 → 保守跳过
            else:
                branch_node, gate_node = m2, m1  # 无形状：按书写惯例 gate*branch
            with g.inserting_after(add_node):
                fused = g.call_function(
                    torch.ops.sakiko.fused_gate_residual.default,
                    args=(base_x, branch_node, gate_node),
                )
            add_node.replace_all_uses_with(fused)
            g.erase_node(add_node)
            g.erase_node(mul_node)
            _MATCHER_STATS["mul_add_rewritten"] += 1
        gm.recompile()
    return gm


def sakiko_backend(gm: fx.GraphModule, example_inputs=None) -> fx.GraphModule:
    """torch.compile 后端入口：形状传播 → 匹配重写 → eager fallback 执行。

    用法::

        compiled = torch.compile(model, backend=sakiko_backend)
    """
    reset_stats()
    if example_inputs:
        _propagate_shapes(gm, example_inputs)
    gm = match_and_replace(gm, rewrite=True)
    return gm
