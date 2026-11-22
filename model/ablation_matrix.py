# -*- coding: utf-8 -*-
"""ablation_matrix.py — GSA vs 2+1D 交替的消融 run matrix 框架（README 待办 / §3.4.1）。

职责边界：本模块只做「配置矩阵定义 + 指标收集接口 + 纯 CPU 矩阵自检」，
**不跑真训练/真对照实验**（那需要 GPU + 蒸馏收敛后的权重）。用途：
  - 把"GSA 还是 2+1D"这个待裁决项拆成可执行的 run 清单；
  - 每个 run 有稳定 ID、配置、指标字段，结果可落 JSON，接 scorecard 口径；
  - 纯 CPU 验证矩阵完整、ID 唯一、各策略 tiny 模型能按配置构造并前向。

运行（纯 CPU）：
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python ablation_matrix.py
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch


@dataclass(frozen=True)
class AblationRun:
    """单个消融 run 的配置。frozen 保证可作字典 key / 稳定 ID。"""
    run_id: str
    attn_strategy: str = "alternate"   # "alternate"=2+1D 交替 | "gsa"=门控混合
    skip_spectral: bool = True
    pyramid: bool = False
    use_vlm: bool = False
    note: str = ""


# 指标字段（与白皮书 §10.2 / scorecard 对齐；这里只定义容器，真值待实验回填）
METRIC_FIELDS = (
    "psnr", "lpips", "identity_sim", "flicker", "seam_jump",
    "collapse_rate", "vram_peak_gb", "step_ms", "params_b",
)


@dataclass
class RunResult:
    run_id: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    status: str = "pending"   # pending / running / done / failed


def build_matrix() -> List[AblationRun]:
    """生成 GSA vs 2+1D 消融矩阵。

    主线两臂：alternate（2+1D） vs gsa（门控混合），各配 skip 谱约束开/关做正交消融。
    金字塔/VLM 作为附加臂默认关（V2 进阶，先证 V1 基线）。
    """
    runs: List[AblationRun] = []
    for strategy in ("alternate", "gsa"):
        for spectral in (True, False):
            tag = f"{strategy}-{'spec' if spectral else 'nospec'}"
            runs.append(AblationRun(
                run_id=tag,
                attn_strategy=strategy,
                skip_spectral=spectral,
                note=("2+1D 交替" if strategy == "alternate" else "GSA 门控混合")
                     + (" + skip 谱约束" if spectral else " + skip 无谱约束"),
            ))
    return runs


def matrix_ids(runs: List[AblationRun]) -> List[str]:
    return [r.run_id for r in runs]


def new_results(runs: List[AblationRun]) -> Dict[str, RunResult]:
    """为每个 run 建空结果容器（指标 None、状态 pending）。"""
    return {r.run_id: RunResult(run_id=r.run_id,
                                metrics={k: None for k in METRIC_FIELDS})
            for r in runs}


def results_to_json(results: Dict[str, RunResult]) -> str:
    """序列化为 JSON 字符串（可直接落盘接 scorecard）。"""
    payload = {
        "schema": "ablation_matrix_v1",
        "metric_fields": list(METRIC_FIELDS),
        "runs": {rid: {"metrics": r.metrics, "status": r.status}
                 for rid, r in results.items()},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _selftest() -> None:
    from uvit3d import UViT3D, UViT3DConfig

    def ck(cond, msg):
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== ablation_matrix 纯 CPU 自检（GSA vs 2+1D）===")
    runs = build_matrix()
    ids = matrix_ids(runs)
    ck(len(runs) == 4, f"矩阵生成 4 个 run：{ids}")
    ck(len(set(ids)) == len(ids), "run_id 唯一")
    ck({"alternate", "gsa"} == {r.attn_strategy for r in runs},
       "两臂齐全：alternate（2+1D）+ gsa")
    ck({True, False} == {r.skip_spectral for r in runs}, "谱约束开/关正交齐全")

    # 结果容器
    res = new_results(runs)
    ck(set(res.keys()) == set(ids), "结果容器覆盖全部 run_id")
    ck(all(r.status == "pending" for r in res.values()), "初始状态全 pending")
    ck(all(set(r.metrics) == set(METRIC_FIELDS) for r in res.values()),
       f"每个 run 含全部 {len(METRIC_FIELDS)} 个指标字段")
    parsed = json.loads(results_to_json(res))
    ck(parsed["schema"] == "ablation_matrix_v1" and len(parsed["runs"]) == 4,
       "JSON 序列化/反序列化往返正确")

    # 各策略 tiny 模型能按配置构造并前向（不跑训练，只证矩阵配置可注入）
    torch.manual_seed(0)
    _cfg_probe = UViT3DConfig.tiny()
    B, C, T, H, W = 1, _cfg_probe.in_channels, 4, 8, 8
    latent = torch.randn(B, C, T, H, W)
    t_emb = torch.randn(B, _cfg_probe.t_emb_dim)
    text_emb = torch.randn(B, _cfg_probe.cond_dim)
    for r in runs:
        cfg = UViT3DConfig.tiny()
        cfg.attn_strategy = r.attn_strategy
        cfg.skip_spectral = r.skip_spectral
        model = UViT3D(cfg); model.eval()
        with torch.inference_mode():
            v, aux = model(latent, t_emb, text_emb, return_aux=True)
        ck(v.shape == (B, cfg.out_channels, T, H, W),
           f"{r.run_id}: tiny 前向形状正确 {tuple(v.shape)}")
    print("=== ablation_matrix 自检全部通过 ===")


if __name__ == "__main__":
    _selftest()
