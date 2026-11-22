# -*- coding: utf-8 -*-
"""sakiko_op.op -- 自定义融合算子 schema + 注册（参考实现版，2026-11-21）。

算子：``sakiko.fused_gate_residual(x, branch, gate) -> out``
数学定义：``out = x + gate * branch``（与门控旁路残差逐位等价）

现状（如实）：
  - 已注册进 ``torch.ops.sakiko``，计算图中出现独立节点；
  - 内部实现为 PyTorch 参考实现（eager fallback），数值完全正确、支持 autograd；
  - Vulkan SPIR-V 内核为后续工程项——注册边界已定型，届时只换 backend 实现，
    schema / FX / 模型侧 / 消融脚本零改动。

口径备注（W10/W11 门控纪律）：gate=0 时必须严格恒等。参考实现里 gate==0
走独立分支直接返回 x（不乘 0），与 nodes_pyragate 的 gate=0 纪律一致，
避免 gate*branch 与 inf/nan 相乘产生 NaN 污染。
"""
from __future__ import annotations

import torch

_LIB = torch.library.Library("sakiko", "FRAGMENT")

_LIB.define(
    "fused_gate_residual(Tensor x, Tensor branch, Tensor gate) -> Tensor"
)


@torch.library.impl(_LIB, "fused_gate_residual", "CPU")
def _fused_gate_residual_cpu(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现（数值正确）。需要梯度时内部走 autograd.Function（自定义反向公式）。"""
    return _autograd_impl(x, branch, gate)


@torch.library.impl(_LIB, "fused_gate_residual", "CUDA")
def _fused_gate_residual_cuda(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """CUDA/ROCm 后端：参考实现是 device-agnostic 的 PyTorch 组合 op，直接复用。

    这不是自定义 HIP/CUDA kernel（无额外加速），只是让 ``SAKIKO_CUSTOM_OP=1``
    在 GPU 上可调用——此前仅注册 CPU/Meta/Autograd，GPU 调用直接 NotImplementedError。
    真 Vulkan/HIP kernel 仍属后续工程项，届时只换 backend 实现，schema 不变。
    """
    return _autograd_impl(x, branch, gate)


@torch.library.impl(_LIB, "fused_gate_residual", "Autograd")
def _fused_gate_residual_autograd(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Autograd key：需要梯度时走 autograd.Function（自定义反向公式）。"""
    return _autograd_impl(x, branch, gate)


@torch.library.impl(_LIB, "fused_gate_residual", "Meta")
def _fused_gate_residual_meta(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Meta 注册：shape/dtype 推导（torch.compile 路径需要）。"""
    torch._check(x.shape == branch.shape, lambda: "x 与 branch 形状必须一致")
    return x.clone()


def _ref_impl(x: torch.Tensor, branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """参考实现本体。gate 可为标量或可广播张量；gate==0 严格恒等（无乘法路径）。"""
    if isinstance(gate, torch.Tensor):
        if gate.ndim == 0:
            if float(gate) == 0.0:
                return x
        # 广播张量 gate：先按 gate!=0 掩码 branch，再乘 gate。
        # 关键：直接 gate*branch 时，gate 的零位遇 branch=inf/nan 会产生 0*inf=nan，
        # 污染本应严格恒等的位置（红线 8）。torch.where 屏蔽后可微且梯度语义不变
        # （gate==0 处 g_branch=grad_out*gate=0 本来就为 0）。
        safe_branch = torch.where(gate != 0, branch, torch.zeros_like(branch))
        return x + gate * safe_branch
    if float(gate) == 0.0:
        return x
    return x + gate * branch


class _FusedGateResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gate):
        ctx.save_for_backward(branch, gate)
        return _ref_impl(x.detach(), branch.detach(), gate.detach() if isinstance(gate, torch.Tensor) else gate)

    @staticmethod
    def backward(ctx, grad_out):
        branch, gate = ctx.saved_tensors
        g_x = grad_out
        g_branch = grad_out * gate if isinstance(gate, torch.Tensor) else grad_out * gate
        if isinstance(gate, torch.Tensor):
            g_gate = (grad_out * branch)
            # 归约到 gate 的形状（标量或广播形状）
            g_gate = _reduce_to_shape(g_gate, gate.shape)
        else:
            g_gate = (grad_out * branch).sum()
        return g_x, g_branch, g_gate


def _reduce_to_shape(g: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """把梯度归约到目标形状（标准广播反向）。"""
    if tuple(g.shape) == tuple(shape):
        return g
    # 先归约掉 g 比目标形状多出的尾部维度。
    # 关键：PyTorch 中 g.sum(dim=()) 是"全维度归约"→标量，不是 no-op。
    # 当 gate 用 (1,1,1)/(8,1,1)/(8,64,1) 这类"同维数但含单例维"的广播形态时，
    # extra_ndim=0，若仍调用 sum(dim=()) 会把张量梯度压成标量导致 backward 崩溃。
    # 因此仅当 extra_ndim>0 时才归约（gate 维数少于 g 的情形，如 gate=(1,)）。
    extra_ndim = g.dim() - len(shape)
    if extra_ndim > 0:
        g = g.sum(dim=tuple(range(extra_ndim)))
    # 再把目标形状里为 1、而被广播出去的维度按 keepdim 归约回去
    dims = [i for i, (gs, ss) in enumerate(zip(g.shape, shape)) if ss == 1 and gs != 1]
    if dims:
        g = g.sum(dim=tuple(dims), keepdim=True)
    return g


def _autograd_impl(x, branch, gate):
    """带 autograd 的入口：需要梯度时走 Function（自定义反向公式），否则直走参考实现。"""
    grad_on = torch.is_grad_enabled() and (
        x.requires_grad or branch.requires_grad
        or (isinstance(gate, torch.Tensor) and gate.requires_grad)
    )
    if grad_on:
        return _FusedGateResidualFn.apply(x, branch, gate)
    return _ref_impl(x, branch, gate)


def _autograd_setup_unused(ctx, *grads):  # noqa: D401  （保留占位：Library 版反向不用此路径）
    raise NotImplementedError


# autograd 路径：CPU impl 内部直接走 _FusedGateResidualFn.apply，
# 无需 torch.library.register_autograd（那种 API 的反向签名是 fn(ctx, grad_out)，
# 且无法在 forward 侧 save_for_backward——参考实现自己带 Function 更干净）。


def fused_gate_residual(x: torch.Tensor, branch: torch.Tensor, gate) -> torch.Tensor:
    """Python 入口：走注册的 schema（计算图出现 sakiko.fused_gate_residual 节点）。"""
    return torch.ops.sakiko.fused_gate_residual(x, branch, gate)


def is_registered() -> bool:
    import torch
    return hasattr(torch.ops.sakiko, "fused_gate_residual")
