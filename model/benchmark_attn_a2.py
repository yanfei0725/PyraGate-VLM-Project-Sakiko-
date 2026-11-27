"""
a2 注意力 benchmark（Track A 门禁）—— 独立、只读、不加载 ckpt、不碰主干训练
============================================================================

比较三种注意力核心在三个 token 规模下的前向/反向步时、峰值显存、cosine 相似度：
  - softmax （FA2/SDPA 基线）
  - linear  （特征映射线性注意力）
  - hybrid  （分块局部 softmax + 全局线性）

门禁：linear/hybrid 在中长序列上 ≥1.5× 加速 且 与 softmax cosine ≥0.9 才算通过；
否则 Track A 判死、不接主干。

用法（纯 CPU 验证结构；GPU 实测在有卡时把 --device cuda）：
  python benchmark_attn_a2.py --device cpu --sizes 990 3960
"""
from __future__ import annotations

import argparse
import time
import torch
import torch.nn.functional as F

from linear_attn import linear_attention, chunked_softmax_attention, hybrid_attention
from attn_backend import attention as dispatch_attention


def softmax_baseline(q, k, v):
    return dispatch_attention(q, k, v, is_causal=False, backend="math")


def bench_fn(fn, args, warmup=3, repeat=10, device="cpu"):
    for _ in range(warmup):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat, out


def run_size(N, H, D, device, dtype):
    torch.manual_seed(0)
    B = 1
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    fns = {
        "softmax": softmax_baseline,
        "linear": lambda q, k, v: linear_attention(q, k, v),
        "hybrid": lambda q, k, v: hybrid_attention(
            q, k, v, gate=torch.zeros(1, device=device, dtype=dtype),
            chunk_size=132, use_softmax=True),
    }
    results = {}
    outs = {}
    for name, fn in fns.items():
        dt, out = bench_fn(fn, (q, k, v), device=device)
        results[name] = dt
        outs[name] = out
        del out

    # cosine：线性/混合 vs softmax（展平 token 维逐行比）
    ref = outs["softmax"].float().reshape(-1, D)
    cos = {}
    for name in ("linear", "hybrid"):
        cos[name] = float(F.cosine_similarity(
            ref, outs[name].float().reshape(-1, D), dim=-1).mean())

    speedup = {name: results["softmax"] / max(results[name], 1e-9)
               for name in ("linear", "hybrid")}
    return results, cos, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[990, 3960])
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda 不可用，降级 cpu")
        device = "cpu"

    print(f"[a2 attn benchmark] device={device} dtype={args.dtype} "
          f"heads={args.heads} head_dim={args.head_dim}")
    print(f"{'N':>6} | {'softmax(ms)':>12} {'linear(ms)':>11} {'hybrid(ms)':>11} "
          f"| {'cos_lin':>8} {'cos_hyb':>8} | {'spd_lin':>8} {'spd_hyb':>8} | 门禁")
    gate_pass = {}
    for N in args.sizes:
        res, cos, spd = run_size(N, args.heads, args.head_dim, device, dtype)
        # 门禁：中长序列(N>=990) 至少一个模式 ≥1.5× 且 cos≥0.9
        ok = any(spd[m] >= 1.5 and cos[m] >= 0.9 for m in ("linear", "hybrid")) if N >= 990 else None
        gate_pass[N] = ok
        ms = lambda x: x * 1000
        print(f"{N:>6} | {ms(res['softmax']):>12.2f} {ms(res['linear']):>11.2f} "
              f"{ms(res['hybrid']):>11.2f} | {cos['linear']:>8.3f} {cos['hybrid']:>8.3f} "
              f"| {spd['linear']:>8.2f} {spd['hybrid']:>8.2f} | {ok}")

    print()
    print("=== 门禁结论 ===")
    for N, ok in gate_pass.items():
        if ok is None:
            print(f"  N={N}: 仅参考（短序列不做加速门禁）")
        else:
            print(f"  N={N}: {'PASS' if ok else 'FAIL'} "
                  f"（需 ≥1.5× 加速且 cos≥0.9；未过则 Track A 不接主干）")


if __name__ == "__main__":
    main()
