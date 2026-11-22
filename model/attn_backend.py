"""FlashAttention-2 后端适配层（PyraGate 3D U-ViT）。

设计约束
--------
1. 适配层不绑定具体硬件：CPU 上自动降级 math，不冒充 GPU kernel。
2. 真正的 FlashAttention-2 kernel 是 CUDA/ROCm-only，且只吃 fp16/bf16；
   CPU 上不可能"跑真 FA2"。所以这里做的是**显式后端选择与降级链**，
   不把 CPU 上的 math/SDPA 伪装成 FA2。
3. 教师 Wan2.2 的 attention.py 优先 flash_attn_interface(FA3) →
   flash_attn(FA2) → 断言。本机两者都没装，但 torch 2.9.1+rocm7.2.1
   自带 SDPA 的 FLASH_ATTENTION backend（flash_sdp_enabled=True）。
   因此 "flash2" 的实际落地优先级为：
       flash_attn 包 (真 FA2) > SDPA FLASH_ATTENTION > SDPA > math
   在 CPU 上自动降到 math，数值与参考实现对齐（自检验证）。

对外 API
--------
- backend_diagnose(): 环境探测字典（供日志/自检用，不分配 GPU 张量）
- resolve_backend(requested, q=None): (actual, reason)
- attention(q, k, v, is_causal=False, backend="auto", with_info=False):
  q/k/v 形状 [B, H, N, D]，返回 [B, H, N, D]
"""
from __future__ import annotations

import functools
import warnings

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_KERNEL = True
except Exception:                                    # pragma: no cover - 老版本 torch
    SDPBackend = None
    sdpa_kernel = None
    _HAS_SDPA_KERNEL = False

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_WARNED: set = set()


@functools.lru_cache(maxsize=1)
def flash2_package_version():
    """flash_attn 包是否可 import（真 FA2 kernel）。纯 Python 探测，不碰 GPU。"""
    try:
        import flash_attn                            # noqa: F401
        return str(getattr(flash_attn, "__version__", "unknown"))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def flash3_package_version():
    """flash_attn_interface 包是否可 import（教师 Wan2.2 里的 FA3 优先项）。"""
    try:
        import flash_attn_interface                  # noqa: F401
        return "present"
    except Exception:
        return None


def backend_diagnose() -> dict:
    """只读环境画像。注意：不创建 CUDA 张量、不占显存。"""
    return {
        "torch": torch.__version__,
        "flash_attn_pkg": flash2_package_version(),
        "flash_attn_interface_pkg": flash3_package_version(),
        "sdpa_kernel_api": _HAS_SDPA_KERNEL,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "cuda_visible": bool(torch.cuda.is_available()),
    }


def resolve_backend(requested: str, q: torch.Tensor = None):
    """把用户请求解析成 (actual, reason)。

    requested: "flash2" | "auto" | "sdpa" | "math"
    actual:    "fa2_package" | "sdpa_flash" | "sdpa" | "math"
    """
    if requested not in ("flash2", "auto", "sdpa", "math"):
        raise ValueError(f"未知注意力后端: {requested}")

    if requested in ("sdpa", "math"):
        return requested, f"explicit_{requested}"

    device_type = q.device.type if q is not None else (
        "cuda" if torch.cuda.is_available() else "cpu")
    dtype = q.dtype if q is not None else torch.bfloat16

    if device_type != "cuda":
        return "math", "cpu_fallback_no_gpu_kernel"
    if dtype not in _HALF_DTYPES:
        return "math", "flash2_requires_fp16_or_bf16"

    if flash2_package_version() is not None:
        return "fa2_package", "real_flash_attn_package"
    if _HAS_SDPA_KERNEL and torch.backends.cuda.flash_sdp_enabled():
        return "sdpa_flash", "torch_sdpa_flash_backend"
    return "sdpa", "flash2_backend_unavailable_use_sdpa"


def _warn_once(actual: str, reason: str, requested: str):
    key = (requested, actual, reason)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"[attn_backend] 请求 '{requested}' 实际使用 '{actual}' ({reason})；"
        "真 FA2 kernel 仅 CUDA/ROCm + fp16/bf16 可用。",
        RuntimeWarning, stacklevel=3,
    )


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              is_causal: bool = False, backend: str = "auto",
              softmax_scale=None, with_info: bool = False):
    """统一注意力入口。q/k/v: [B, H, N, D] -> [B, H, N, D]。"""
    actual, reason = resolve_backend(backend, q)
    if backend == "flash2":
        _warn_once(actual, reason, backend)

    if actual == "fa2_package":
        import flash_attn
        q2 = q.transpose(1, 2).contiguous()
        k2 = k.transpose(1, 2).contiguous()
        v2 = v.transpose(1, 2).contiguous()
        y = flash_attn.flash_attn_func(
            q2, k2, v2, dropout_p=0.0,
            softmax_scale=softmax_scale, causal=is_causal)
        y = y.transpose(1, 2).contiguous()
    elif actual == "sdpa_flash":
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal, scale=softmax_scale)
        except RuntimeError as exc:                 # 驱动/硬件不支持时降级
            _warn_once("sdpa", f"sdpa_flash_runtime_fallback: {exc}"[:160], backend)
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal, scale=softmax_scale)
            actual, reason = "sdpa", "sdpa_flash_runtime_fallback"
    elif actual == "sdpa":
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=softmax_scale)
    else:                                            # math：显式强制，便于数值对齐
        if _HAS_SDPA_KERNEL:
            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal, scale=softmax_scale)
        else:                                        # pragma: no cover
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal, scale=softmax_scale)

    if with_info:
        return y, {"requested": backend, "actual": actual, "reason": reason}
    return y
