# -*- coding: utf-8 -*-
"""quant_plan.py — 混合量化分层清单（白皮书 §7.2 红线层表）的纯 CPU 实现。

职责边界（重要）：
  本模块只做「按模块路径把每个 nn.Module 标注为 fp16 / int8」的**计划与自检**，
  不执行真量化（真 int8 需 GPU + Quark 逐层校准，待显卡）。产出的 plan 可直接
  喂给后续量化器；在这里先保证「哪些层保 fp16」这条红线在代码层可执行、可验证。

分类规则（严格对齐 §7.2）：
  - 时序注意力 / skip 融合与门控 α / AdaLN-Zero 调制 / 输出速度场 Head / 瓶颈层 → fp16
  - 主干 FFN(SwiGLU) / 普通自注意力 / VAE → VAE 保 fp16；FFN 与普通 attn 标 int8
  - KV cache 单独标注 int8_cache（不是权重，是激活选项）

运行（纯 CPU）：
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python quant_plan.py
"""
import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# §7.2 敏感层关键词（路径子串匹配）。顺序敏感：先判到的优先。
# 注：教师式 AdaLN 改造后，调制层 = 共享 embedder（ada_embedder，模块级）+
#     逐块 (1,6,dim) 小表（scale_shift_table，Parameter——模块级清单不单列，
#     量化执行时随宿主块处理并在执行层单独标 fp16）。
SENSITIVE_PATTERNS: Tuple[str, ...] = (
    "temporal",          # 时序注意力（含 causal / rope）
    "skips",             # skip 融合 + 门控 α（GatedSkip）
    "ada_embedder",      # AdaLN 共享条件 embedder（教师式调制）
    "bottleneck",        # 3D 联合注意力瓶颈
    "head",              # 输出预测速度场 Head（OutputHead）
)
# VAE 轻量版：不量化（§0.2/§7.2，保 fp16）
VAE_PATTERNS: Tuple[str, ...] = ("vae", "decoder_vae", "turbo")

FP16 = "fp16"
INT8 = "int8"
KV_INT8 = "int8_cache"

# 注意力类型 -> §7.2 判定。路径里只写 "attn"，具体时空类型靠 type(module) 区分。
# 时序注意力（TemporalAttention）与 GSA（含时序门控）量化会加重闪烁/运动漂移 → fp16。
# 空间注意力（SpatialAttention）与普通自注意力 → int8（§7.2 "普通自注意力权重"）。
_TEMPORAL_TYPES = ("TemporalAttention", "GSA")


def _is_temporal_type(module: nn.Module) -> bool:
    return type(module).__name__ in _TEMPORAL_TYPES


def classify(path: str, module: nn.Module) -> str:
    """按模块类型 + 全路径返回精度标注。纯判定，不碰参数数据。"""
    low = path.lower()
    if any(p.lower() in low for p in VAE_PATTERNS):
        return FP16
    if any(p.lower() in low for p in SENSITIVE_PATTERNS):
        return FP16
    if _is_temporal_type(module):
        return FP16  # §7.2 时序注意力保 fp16
    # norm 层（RMSNorm）参数极小、对分布敏感，归入 fp16（工程稳妥，非 §7.2 明列，标注为保守扩展）
    if isinstance(module, nn.LayerNorm) or "norm" in low:
        return FP16
    return INT8


def build_plan(model: nn.Module, prefix: str = "") -> Dict[str, str]:
    """遍历模型，产出 {module_path: precision} 的完整清单。"""
    plan: Dict[str, str] = {}
    for name, mod in model.named_modules():
        if name == "" and prefix == "":
            continue  # 跳过根模块
        full = f"{prefix}.{name}" if prefix else name
        plan[full] = classify(full, mod)
    return plan


def summarize(plan: Dict[str, str]) -> Dict[str, int]:
    out = {FP16: 0, INT8: 0, KV_INT8: 0}
    for v in plan.values():
        out[v] = out.get(v, 0) + 1
    return out


def assert_redlines(plan: Dict[str, str], model: nn.Module = None) -> List[str]:
    """校验 §7.2 红线：所有敏感层都必须标 fp16；返回违规路径列表（空=通过）。

    model 可选：给定时额外按类型检查时序注意力（路径不含 temporal 关键词）。
    """
    violations: List[str] = []
    for path, prec in plan.items():
        low = path.lower()
        is_sensitive = any(p.lower() in low for p in SENSITIVE_PATTERNS) or \
                       any(p.lower() in low for p in VAE_PATTERNS)
        if is_sensitive and prec != FP16:
            violations.append(path)
    if model is not None:
        for path, mod in model.named_modules():
            if _is_temporal_type(mod) and plan.get(path) != FP16:
                violations.append(path + " (temporal-type)")
    return violations


def _selftest() -> None:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uvit3d import UViT3D, UViT3DConfig

    def ck(cond, msg):
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== quant_plan 纯 CPU 自检（§7.2 红线层表）===")
    cfg = UViT3DConfig.tiny()
    model = UViT3D(cfg)
    plan = build_plan(model)
    s = summarize(plan)
    print(f"  模块总数 {len(plan)}：fp16={s[FP16]} int8={s[INT8]}")

    # 1) 敏感层逐类验证：每类至少命中一个模块且全为 fp16
    for key, need in [("skips", "skip 门控"),
                      ("ada_embedder", "AdaLN 共享 embedder"), ("bottleneck", "瓶颈"),
                      ("head", "输出头")]:
        hits = {p: v for p, v in plan.items() if key.lower() in p.lower()}
        ck(len(hits) > 0, f"{need}：命中 {len(hits)} 个模块")
        ck(all(v == FP16 for v in hits.values()), f"{need}：全部标 fp16")
    # 1b) 教师式 AdaLN 逐块小表是 Parameter（非独立模块）——存在性 + 数量校验
    n_tab = sum(1 for n, _ in model.named_parameters() if "scale_shift_table" in n)
    n_blocks = len(list(model.enc_blocks)) + len(list(model.bottleneck)) + len(list(model.dec_blocks))
    ck(n_tab == n_blocks, f"AdaLN 逐块表：{n_tab}/{n_blocks} 个块各持一个 Parameter（量化执行层单独标 fp16）")
    # 时序注意力靠类型识别（路径只有 attn），验证 TemporalAttention 模块全为 fp16
    temp_mods = [(p, mod) for p, mod in model.named_modules() if _is_temporal_type(mod)]
    ck(len(temp_mods) > 0, f"时序注意力：命中 {len(temp_mods)} 个 TemporalAttention/GSA 模块")
    ck(all(plan[p] == FP16 for p, _ in temp_mods), "时序注意力：全部标 fp16（§7.2）")
    spat_mods = [(p, mod) for p, mod in model.named_modules()
                 if type(mod).__name__ == "SpatialAttention"]
    ck(all(plan[p] == INT8 for p, _ in spat_mods), "空间注意力：全部标 int8（普通自注意力）")

    # 2) 普通 FFN 标 int8（收益主体）；瓶颈内的 FFN 随瓶颈保 fp16（§7.2 瓶颈层优先）
    ffn_hits = {p: v for p, v in plan.items()
                if "ffn" in p.lower() and "bottleneck" not in p.lower()}
    ck(len(ffn_hits) > 0, f"普通 SwiGLU FFN：命中 {len(ffn_hits)} 个模块")
    ck(all(v == INT8 for v in ffn_hits.values()), "普通 SwiGLU FFN：全部标 int8（量化收益主体）")
    bn_ffn = {p: v for p, v in plan.items() if "ffn" in p.lower() and "bottleneck" in p.lower()}
    ck(all(v == FP16 for v in bn_ffn.values()), "瓶颈内 FFN：随瓶颈保 fp16（§7.2 优先级正确）")

    # 3) 红线校验：零违规
    violations = assert_redlines(plan, model)
    ck(len(violations) == 0, f"§7.2 红线：敏感层零违规（违规 {len(violations)} 项）")

    # 4) 分类确定性：同输入同输出
    ck(build_plan(model) == plan, "build_plan 幂等（同模型同清单）")

    # 5) VAE 路径关键词单测（无真权重，纯字符串）
    ck(classify("model.vae.decoder.0", nn.Linear(4, 4)) == FP16, "VAE 路径标 fp16（不量化）")
    ck(classify("model.enc_blocks.0.attn.q", nn.Linear(4, 4)) == INT8, "普通注意力权重标 int8")
    print("=== quant_plan 自检全部通过 ===")


if __name__ == "__main__":
    _selftest()
