# -*- coding: utf-8 -*-
"""gate_decay_forget -- 门控残差 + 时序衰减 + 一阶遗忘（Triton kernel + 参考实现）。

数学（单 kernel 双模式超集，修正了原方案的等价性硬伤）::

    out = wx * x + wb * (gate * decay) * branch

  mode="residual"（默认）: wx=1, wb=alpha
      → alpha=1, decay=1 时**精确回退** ``x + gate*branch``（真等价消融点）；
      → gate=0 严格恒等（out = x）；
      → alpha<1 = 软化注入；decay 压制旧帧门控强度（方案1）；
  mode="blend": wx=alpha, wb=1-alpha
      → 原方案的低通混合公式；alpha=1 = 旁路关闭（out=x），**无**原版等价点。

反向（解析式，kernel 内完成）::

    d/dx      = dout * wx
    d/dbranch = dout * wb * (gate*decay)
    d/gate    = dout * wb * decay * branch
    d/decay   = dout * wb * gate * branch

工程注记：
  - import 顺序硬约束：**先 torch 后 triton**（triton 先导会破坏 torch_hip.dll 加载）；
  - alpha 走运行时标量参数，非 tl.constexpr——α 扫描零重编译；
  - 中间量 g_eff 只存在于寄存器，不写 HBM；
  - Triton 不可用/非 CUDA 张量时自动回退 PyTorch 参考实现（数值一致，可训练）。
"""
from __future__ import annotations

import torch

# 硬约束：torch 先于 triton 导入
TRITON_OK = False
try:
    import triton
    import triton.language as tl
    TRITON_OK = True
except Exception as _e:  # pragma: no cover
    _triton_import_error = _e


if TRITON_OK:

    @triton.jit
    def _fgdf_fwd_kernel(x_ptr, branch_ptr, gate_ptr, decay_ptr, out_ptr,
                         wx, wb, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        b = tl.load(branch_ptr + offs, mask=mask)
        g = tl.load(gate_ptr + offs, mask=mask)
        d = tl.load(decay_ptr + offs, mask=mask)
        g_eff = g * d
        out = wx * x + wb * g_eff * b
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _fgdf_bwd_kernel(dout_ptr, branch_ptr, gate_ptr, decay_ptr,
                         dx_ptr, db_ptr, dg_ptr, ddecay_ptr,
                         wx, wb, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        dout = tl.load(dout_ptr + offs, mask=mask)
        b = tl.load(branch_ptr + offs, mask=mask)
        g = tl.load(gate_ptr + offs, mask=mask)
        d = tl.load(decay_ptr + offs, mask=mask)
        g_eff = g * d
        tl.store(dx_ptr + offs, dout * wx, mask=mask)
        tl.store(db_ptr + offs, dout * wb * g_eff, mask=mask)
        tl.store(dg_ptr + offs, dout * wb * d * b, mask=mask)
        tl.store(ddecay_ptr + offs, dout * wb * g * b, mask=mask)


def ref_gate_decay_forget(x, branch, gate, decay, wx, wb):
    """纯 PyTorch 参考实现（CPU/GPU 皆可，天然可导，数值基准）。"""
    g_eff = gate * decay
    return wx * x + wb * g_eff * branch


def _coefs(alpha: float, mode: str):
    if mode == "residual":
        return 1.0, float(alpha)
    if mode == "blend":
        return float(alpha), 1.0 - float(alpha)
    raise ValueError("mode 必须是 residual|blend")


class _GateDecayForgetFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gate, decay, wx: float, wb: float):
        use_triton = (TRITON_OK and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                      and x.is_contiguous() and branch.is_contiguous()
                      and gate.is_contiguous() and decay.is_contiguous())
        out = torch.empty_like(x)
        if use_triton:
            n = x.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            _fgdf_fwd_kernel[grid](x, branch, gate, decay, out,
                                   float(wx), float(wb), n,
                                   BLOCK_SIZE=1024, num_warps=4)
            ctx.triton_used = True
        else:
            out = ref_gate_decay_forget(x, branch, gate, decay, wx, wb)
            ctx.triton_used = False
        ctx.save_for_backward(branch, gate, decay)
        ctx.wx, ctx.wb = float(wx), float(wb)
        return out

    @staticmethod
    def backward(ctx, dout):
        branch, gate, decay = ctx.saved_tensors
        wx, wb = ctx.wx, ctx.wb
        if ctx.triton_used:
            dx = torch.empty_like(dout)
            db = torch.empty_like(dout)
            dg = torch.empty_like(dout)
            dd = torch.empty_like(dout)
            n = dout.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            _fgdf_bwd_kernel[grid](dout.contiguous(), branch, gate, decay,
                                   dx, db, dg, dd,
                                   wx, wb, n,
                                   BLOCK_SIZE=1024, num_warps=4)
            return dx, db, dg, dd, None, None
        # 参考实现反向（与 kernel 解析式一致）
        g_eff = gate * decay
        with torch.no_grad():
            dx = dout * wx
            db = dout * wb * g_eff
            dg = dout * wb * decay * branch
            dd = dout * wb * gate * branch
        return dx, db, dg, dd, None, None


def gate_decay_forget(x, branch, gate, decay, alpha: float = 1.0,
                      mode: str = "residual", use_triton: bool | None = None):
    """统一入口。见模块 docstring 的数学与模式说明。

    消融对照锚点：mode="residual", alpha=1.0, decay=torch.ones_like(x)
    ⇔ 精确等于 ``x + gate*branch``（原版门控旁路）。
    """
    wx, wb = _coefs(alpha, mode)
    if use_triton is None:
        use_triton = True  # Function 内部会再做环境判定并自动回退
    if not (use_triton and TRITON_OK) or not x.is_cuda:
        # 显式参考实现路径：仍走 Function 以保持反向一致性（Function 内部自动回退）
        pass
    return _GateDecayForgetFn.apply(x, branch, gate, decay, float(wx), float(wb))
