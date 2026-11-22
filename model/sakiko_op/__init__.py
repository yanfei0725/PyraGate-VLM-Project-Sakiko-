# -*- coding: utf-8 -*-
"""sakiko_op -- Sakiko 融合算子 + FX 调度后端（参考实现版骨架）。

组件：
  op         schema 注册（torch.ops.sakiko.fused_gate_residual）+ 参考实现 + 自定义反向
  fx_backend FX 图抓取 / mul+add 子图匹配重写 / sakiko 节点识别 / eager fallback 后端
  switch     模型侧开关（SAKIKO_CUSTOM_OP）+ 命中统计

Vulkan SPIR-V 内核 = 后置工程项；schema/FX/开关/消融脚本全部定型，
届时仅替换算子 backend 实现，上层零改动。
"""
from .op import fused_gate_residual, is_registered
from .switch import gated_residual, use_custom_op, hits

__all__ = ["fused_gate_residual", "is_registered", "gated_residual", "use_custom_op", "hits"]
