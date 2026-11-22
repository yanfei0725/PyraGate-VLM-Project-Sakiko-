#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R2 教师权重热启动（教师权重热启动）。

教师 Wan5B M85 混血权重（Wan5B_fastwan_m85.safetensors，10GB）
里同构的 embed/head 级张量直接拷进学生，作为训练初值 —— 主干从教师附近
微调，而不是从零训。审计 R2 修正版：**只直拷 embed/head 级 3 类**，
撤回 qkvo/SwiGLU/GSA 直拷（形状/结构不同构，改走按学生形状重拟合）。

直拷映射（全部 meta device 实测形状一致，safe_open 按 key 精确读，不载 10GB 全量）：
    教师 patch_embedding.{weight,bias}  [3072,48,1,2,2] -> 学生 patchify.*
    教师 head.head.{weight,bias}        [192,3072]      -> 学生 head.head.*
    教师 time_embedding.0.{weight,bias} [3072,256]      -> 学生 t_embed.0.*
    教师 text_embedding.0.{weight,bias} [3072,4096]     -> 学生 cond_proj.*
    教师 text_embedding.2.{weight,bias} [3072,3072]     -> 学生 ada_embedder.embedder.1.*（语义主份）
    教师 head.modulation [1,2,3072] 无同构目标（学生 R7 head_mod 是 6 段 6144），跳过。

优先级（与训练脚本 main 对齐）：build -> **warmstart** -> F2 谱初始化 -> resume。
即：热启动先灌教师值；F2 只重设未被热启动覆盖的 Linear；resume 的 checkpoint
权重最后覆盖一切（断点续训语义最高）。

α=0 语义：本模块只改初值，不改结构/门控，任何开关保持默认即可严格回退。
"""
import os
from typing import Dict, List, Tuple

import torch

TEACHER_DEFAULT = os.environ.get("PYRAGATE_TEACHER", "Wan5B_fastwan_m85.safetensors")

# (教师 key, 学生 key, 期望形状) —— 形状不匹配则跳过并报告，绝不 partial copy
WARMSTART_MAP: List[Tuple[str, str, tuple]] = [
    ("patch_embedding.weight", "patchify.weight", (3072, 48, 1, 2, 2)),
    ("patch_embedding.bias", "patchify.bias", (3072,)),
    ("head.head.weight", "head.head.weight", (192, 3072)),
    ("head.head.bias", "head.head.bias", (192,)),
    ("time_embedding.0.weight", "t_embed.0.weight", (3072, 256)),
    ("time_embedding.0.bias", "t_embed.0.bias", (3072,)),
    ("text_embedding.0.weight", "cond_proj.weight", (3072, 4096)),
    ("text_embedding.0.bias", "cond_proj.bias", (3072,)),
    ("text_embedding.2.weight", "ada_embedder.embedder.1.weight", (3072, 3072)),
    ("text_embedding.2.bias", "ada_embedder.embedder.1.bias", (3072,)),
]


def warmstart_from_teacher(model, path: str = TEACHER_DEFAULT,
                           strict_shapes: bool = True) -> Dict:
    """把教师 embed/head 级权重拷进学生（原地修改 model）。

    返回统计 dict：copied / skipped / missing，供训练日志与自检核对。
    strict_shapes=True 时形状不符直接跳过并记入 skipped（不 partial、不报错中断）。
    """
    from safetensors import safe_open

    if not os.path.isfile(path):
        return {"copied": [], "skipped": [f"teacher file missing: {path}"],
                "missing": [], "ok": False}

    state = model.state_dict()
    report = {"copied": [], "skipped": [], "missing": [], "ok": True}

    with safe_open(path, framework="pt", device="cpu") as f:
        teacher_keys = set(f.keys())
        for tkey, skey, expect in WARMSTART_MAP:
            if tkey not in teacher_keys:
                report["missing"].append(tkey)
                continue
            if skey not in state:
                report["skipped"].append(f"{skey}: 学生无此键")
                continue
            tv = f.get_tensor(tkey)
            if tuple(tv.shape) != tuple(expect):
                report["skipped"].append(
                    f"{tkey}: 教师形状 {tuple(tv.shape)} != 期望 {expect}")
                continue
            sv = state[skey]
            if tuple(sv.shape) != tuple(expect):
                report["skipped"].append(
                    f"{skey}: 学生形状 {tuple(sv.shape)} != 期望 {expect}")
                continue
            # dtype 对齐到学生（学生 bf16；教师也是 bf16，此处仅保稳健）
            state[skey] = tv.to(dtype=sv.dtype).clone()
            report["copied"].append(f"{tkey} -> {skey}")

    model.load_state_dict(state, strict=False)
    report["ok"] = len(report["copied"]) > 0 and not report["missing"]
    return report


def format_report(report: Dict) -> str:
    copied = report.get("copied", [])
    skipped = report.get("skipped", [])
    missing = report.get("missing", [])
    lines = [f"[warmstart] 教师热启动：拷贝 {len(copied)} 项"
             f" / 跳过 {len(skipped)} / 缺失 {len(missing)}"]
    for c in copied:
        lines.append(f"    + {c}")
    for s in skipped:
        lines.append(f"    - skip {s}")
    for m in missing:
        lines.append(f"    ! missing {m}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="教师权重热启动自检（CPU，不加载 10GB 全量）")
    ap.add_argument("--teacher", default=TEACHER_DEFAULT)
    args = ap.parse_args()
    from uvit3d import UViT3DConfig, UViT3D
    cfg = UViT3DConfig.v1_baseline_4b5()
    m = UViT3D(cfg)
    rep = warmstart_from_teacher(m, args.teacher)
    print(format_report(rep))
    print("OK" if rep["ok"] else "FAIL")
