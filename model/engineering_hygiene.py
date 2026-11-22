# -*- coding: utf-8 -*-
"""engineering_hygiene.py — 白皮书 §17.2 工程卫生包的纯 CPU 等价性单测。

§17.2 九项里，7 项能在纯 CPU 上做行为等价验证（本模块实现+自检）：
  1. 循环内同步点削除       -> batched_sync（mock 计数证明批量 < 逐个）
  2. inference_mode 全程    -> assert_inference_mode（无 grad_fn）
  3. bf16 锚定 + NaN guard  -> anchor_dtype / nan_guard
  4. 文本编码 LRU           -> PromptLRU（key = prompt + 模型版本）
  5. 静态常量冻结为 buffer  -> freeze_buffer（不进 parameters）
  7. 视图卫生               -> need_contiguous（多余 contiguous 判定）
  9. 已有内置别重复造       -> no_op_to（同设备 .to() 返回同一对象）

GPU-only 两项（VAE 后处理整链 compile、流优先级分组）不做假测试，仅在
gpu_only_notes() 里列出降级说明与真实验收条件，避免把"没测"当"通过"。

运行（纯 CPU）：
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python engineering_hygiene.py
"""
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn


# --- 1. 循环内同步点削除（mock 计数：批量一次 vs 逐个 N 次）-------------------
class SyncCounter:
    def __init__(self):
        self.count = 0

    def sync(self):
        self.count += 1


def batched_sync(counter: SyncCounter, items: list, batch_size: int = 8) -> int:
    """批量同步：每 batch_size 个只同步一次；返回同步次数。

    真场景里 sync = .item()/.cpu() 这类 host↔device 往返；这里用计数证明削除效果。
    """
    n = 0
    for i in range(0, len(items), batch_size):
        counter.sync()
        n += 1
    return n


# --- 2. inference_mode -------------------------------------------------------
def assert_inference_mode(t: torch.Tensor) -> bool:
    """张量无 grad_fn 即在 inference/no_grad 下产生（纯 CPU 可判）。"""
    return t.grad_fn is None and not t.requires_grad


# --- 3. bf16 锚定 + NaN guard -------------------------------------------------
def anchor_dtype(t: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """把张量锚定到目标 dtype（默认 bf16），模拟 §17.2 bf16 类型锚定。"""
    return t.to(dtype)


def nan_guard(t: torch.Tensor, repl: float = 0.0) -> torch.Tensor:
    """gate 路径 NaN/Inf 兜底：非有限值替换为 repl（§17.2 NaN guard）。"""
    return torch.nan_to_num(t, nan=repl, posinf=repl, neginf=repl)


# --- 4. 文本编码 LRU（key = prompt + 模型版本）-------------------------------
class PromptLRU:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prompt: str, model_version: str) -> tuple:
        return (prompt, model_version)

    def get(self, key: tuple, factory: Callable[[], Any]) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        val = factory()
        self._cache[key] = val
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        self.misses += 1
        return val

    def __len__(self):
        return len(self._cache)


# --- 5. 静态常量冻结为 buffer -------------------------------------------------
def freeze_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """注册为 buffer（随 .to()/.state_dict() 走，但不进 parameters、不参与优化）。"""
    module.register_buffer(name, tensor, persistent=True)


# --- 7. 视图卫生：多余 contiguous 判定 ---------------------------------------
def need_contiguous(t: torch.Tensor) -> bool:
    """只有非连续张量才需要 .contiguous()；连续张量调用属冗余拷贝。"""
    return not t.is_contiguous()


# --- 9. 已有内置：同设备 .to() 是 no-op --------------------------------------
def no_op_to(t: torch.Tensor, device: torch.device) -> bool:
    """同 dtype 同 device 的 .to() 返回同一对象（无需重复造"搬移池"）。"""
    return t.to(device) is t


def gpu_only_notes() -> Dict[str, str]:
    """GPU-only 两项：不假测，给出真实验收条件。"""
    return {
        "VAE 后处理整链 compile": "需 GPU + torch.compile 真编译 VAE decode 链；"
                                 "CPU 上无编译收益且可能更慢，本模块不做假通过。",
        "流优先级分组 + 权重热驻留": "需 CUDA/HIP stream 与显存管理；纯 CPU 无流概念，"
                                   "留待 GPU 阶段用 memory_snapshot 验收。",
    }


def _selftest() -> None:
    def ck(cond, msg):
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== engineering_hygiene 纯 CPU 自检（§17.2 七项）===")
    items = list(range(100))

    # 1) 同步点削除
    c1, c2 = SyncCounter(), SyncCounter()
    for _ in items:
        c1.sync()                    # 逐个同步（旧）
    batched_sync(c2, items, 8)       # 批量同步（新）
    ck(c2.count < c1.count, f"同步点削除：逐个 {c1.count} 次 → 批量 {c2.count} 次")

    # 2) inference_mode
    with torch.inference_mode():
        t = torch.randn(3, 3) + 1
    ck(assert_inference_mode(t), "inference_mode 下张量无 grad_fn（全程推断）")

    # 3) bf16 锚定 + NaN guard
    a = anchor_dtype(torch.randn(4), torch.bfloat16)
    ck(a.dtype == torch.bfloat16, "bf16 类型锚定生效")
    bad = torch.tensor([1.0, float("nan"), float("inf")])
    g = nan_guard(bad)
    ck(torch.isfinite(g).all().item(), "NaN guard：NaN/Inf 全部替换为有限值")

    # 4) 文本编码 LRU
    lru = PromptLRU(capacity=2)
    k = PromptLRU.key("hello", "v1")
    v1 = lru.get(k, lambda: torch.tensor([1.0]))
    v2 = lru.get(k, lambda: torch.tensor([2.0]))
    ck(v1 is v2 and lru.hits == 1 and lru.misses == 1, "LRU 命中复用（key=prompt+版本）")
    ck(PromptLRU.key("hello", "v2") != k, "模型版本变化 → 不同缓存 key")
    lru.get(PromptLRU.key("a", "v1"), lambda: 1)
    lru.get(PromptLRU.key("b", "v1"), lambda: 2)
    ck(len(lru) == 2, "LRU 容量上限生效（淘汰最久未用）")

    # 5) 常量冻结为 buffer
    m = nn.Module()
    freeze_buffer(m, "noise_base", torch.arange(4))
    ck("noise_base" in dict(m.named_buffers()), "常量注册为 buffer")
    ck("noise_base" not in dict(m.named_parameters()), "buffer 不进 parameters（不被优化）")

    # 7) 视图卫生
    contig = torch.randn(3, 4)
    noncontig = contig.t()
    ck(need_contiguous(contig) is False, "连续张量：无需 .contiguous()（删冗余拷贝）")
    ck(need_contiguous(noncontig) is True, "非连续张量：确实需要 .contiguous()")

    # 9) no-op to
    ck(no_op_to(contig, torch.device("cpu")), "同设备 .to() 返回同一对象（no-op）")

    # GPU-only 诚实标注
    notes = gpu_only_notes()
    ck(len(notes) == 2 and all("GPU" in v for v in notes.values()),
       "GPU-only 两项标注降级说明（不冒充通过）")
    for k2, v in notes.items():
        print(f"    [GPU-only] {k2}: {v}")
    print("=== engineering_hygiene 自检全部通过（7 项 CPU + 2 项诚实标注）===")


if __name__ == "__main__":
    _selftest()
