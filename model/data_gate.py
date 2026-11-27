# -*- coding: utf-8 -*-
"""训练数据门禁：先筛质量，再按 σ + 场景分组；缺组直接报警。

数据门禁规则：
    所有训练数据必须先筛质量并按场景/σ 分组，不许混池直接训。

用法：
    python model/data_gate.py \
        --cache model/teacher_targets ^
        --out outputs/data_groups.json \
        --sigma-field t --sigma-transform none \
        --min-per-group 20

只读缓存，不删文件、不改训练集；输出分组清单和拒绝原因。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

SIGMA_BINS = [
    ("sigma_005_020", 0.05, 0.20),
    ("sigma_020_050", 0.20, 0.50),
    ("sigma_050_080", 0.50, 0.80),
    ("sigma_080_100", 0.80, 1.00),
]

SCENE_KEYWORDS = {
    # 2026-11-26：动画提至首位——风格词应优先于内容词。旧序中“人物”在前，
    # "anime girl" 类 prompt 首中人物，动画计数被低估（实测低 σ 动画桶 0/20）。
    "动画": ["anime", "cartoon", "animation", "illustration", "动漫", "动画", "卡通", "插画"],
    "人物": ["person", "people", "man", "woman", "girl", "boy", "face", "portrait",
             "人", "男", "女", "人物", "女孩", "男孩", "老人", "孩子"],
    "室内": ["indoor", "room", "kitchen", "bedroom", "office", "restaurant", "cafe",
             "室内", "房间", "厨房", "卧室", "办公室", "餐厅", "咖啡"],
    "室外": ["outdoor", "street", "city", "park", "forest", "beach", "mountain",
             "室外", "街道", "城市", "公园", "森林", "海滩", "山"],
    "物体": ["object", "product", "car", "food", "animal", "cat", "dog", "flower",
             "物体", "产品", "汽车", "食物", "动物", "猫", "狗", "花"],
}


def _load_x0(path: Path, obj: Any = None) -> torch.Tensor:
    if obj is None:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("x0", "latent", "z", "clean"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(f"{path}: no x0/latent/z/clean field, keys={list(obj)}")
    if not torch.is_tensor(obj):
        raise TypeError(f"{path}: x0 is {type(obj)}, not tensor")
    if obj.ndim == 4:
        obj = obj.unsqueeze(1)  # [C,T,H,W] -> [1,C,T,H,W]
    if obj.ndim != 5:
        raise ValueError(f"{path}: x0 shape {tuple(obj.shape)}, expected 5D")
    return obj.float()


def _prompt_of(path: Path, obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("prompt", "caption", "text"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    return ""


def _sigma_from_t(t: float, transform: str, shift: float = 8.0) -> float:
    t = min(max(float(t), 1e-6), 1.0 - 1e-6)
    if transform == "none":
        return t
    if transform == "shift8_forward":
        return shift * t / (1.0 + (shift - 1.0) * t)
    if transform == "shift8_inverse":
        return t / (shift - (shift - 1.0) * t)
    raise ValueError(f"unknown sigma transform: {transform}")


def _sigma_bin(sigma: float) -> str:
    for name, lo, hi in SIGMA_BINS:
        if lo <= sigma < hi or (name == "sigma_080_100" and sigma >= 0.80):
            return name
    return "sigma_out_of_range"


def _scene_of(prompt: str) -> str:
    low = prompt.lower()
    for scene, words in SCENE_KEYWORDS.items():
        if any(w.lower() in low for w in words):
            return scene
    return "其他"


def _quality(x0: torch.Tensor) -> dict[str, float]:
    # x0 [1,C,T,H,W]
    std = float(x0.std())
    spatial = float((x0[..., :, 1:, :] - x0[..., :, :-1, :]).abs().mean())
    temporal = float((x0[:, :, 1:] - x0[:, :, :-1]).abs().mean()) if x0.shape[2] > 1 else 0.0
    flat = float((x0.abs().amax(dim=1) < 1e-4).float().mean())
    # 黑边/纯色边框比例：只看最外一圈像素
    border = torch.cat([
        x0[..., 0, :].reshape(-1), x0[..., -1, :].reshape(-1),
        x0[..., :, 0].reshape(-1), x0[..., :, -1].reshape(-1),
    ])
    border_flat = float((border.abs() < 1e-4).float().mean())
    return {
        "std": round(std, 6),
        "spatial_grad": round(spatial, 6),
        "temporal_motion": round(temporal, 6),
        "flat_ratio": round(flat, 6),
        "border_flat_ratio": round(border_flat, 6),
    }


def _hash_tensor(x0: torch.Tensor) -> str:
    arr = x0.contiguous().numpy().tobytes()
    return hashlib.sha1(arr).hexdigest()


def _sigma_stats_ext(values: list[float]) -> dict[str, Any]:
    """分位数 + 总体标准差 + 四 σ 桶直方图（纯 CPU，statistics.pstdev）。"""
    import statistics
    if not values:
        return {}
    s = sorted(values)

    def _q(p: float) -> float:
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (s[c] - s[f]) * (k - f), 6)

    hist: dict[str, int] = {"sigma_lt_005": 0}
    for name, _, _ in SIGMA_BINS:
        hist[name] = 0
    for v in values:
        if v < 0.05:
            hist["sigma_lt_005"] += 1
        else:
            hist[_sigma_bin(v)] = hist.get(_sigma_bin(v), 0) + 1
    return {
        "min": round(s[0], 6),
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "max": round(s[-1], 6),
        "mean": round(sum(s) / len(s), 6),
        "std": round(statistics.pstdev(s), 6) if len(s) > 1 else 0.0,
        "histogram": hist,
    }


def build_gate(cache_dir: str | Path, sigma_field: str = "t",
               sigma_transform: str = "none", shift: float = 8.0,
               sigma_mode: str = "cache", sigma_min: float = 0.05,
               sigma_max: float = 1.0,
               min_std: float = 0.10, min_grad: float = 0.01,
               max_flat: float = 0.50, max_border_flat: float = 0.80,
               min_per_group: int = 20,
               pair_cache: str | Path | None = None) -> dict[str, Any]:
    cache = Path(cache_dir)
    files = sorted(cache.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt under {cache}")
    pair_dir = Path(pair_cache) if pair_cache else None
    if pair_dir is not None and not pair_dir.is_dir():
        raise FileNotFoundError(f"pair-cache dir missing: {pair_dir}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejects: list[dict[str, Any]] = []
    hashes: Counter[str] = Counter()
    sigma_values: list[float] = []
    scene_counts: Counter[str] = Counter()

    for path in files:
        pair: Any = None
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            x0 = _load_x0(path, obj)
            prompt = _prompt_of(path, obj)
            if pair_dir is not None and not prompt:
                pp = pair_dir / path.name
                if pp.is_file():
                    pair = torch.load(pp, map_location="cpu", weights_only=False)
                    prompt = _prompt_of(path, pair)
        except Exception as e:
            rejects.append({"file": path.name, "reason": f"load/parse failed: {e}"})
            continue
        scene = _scene_of(prompt)

        if sigma_mode == "cache":
            raw: float | None = None
            if isinstance(obj, dict) and sigma_field in obj:
                raw = float(obj[sigma_field])
            elif pair is None and pair_dir is not None:
                pp = pair_dir / path.name
                if pp.is_file():
                    pair = torch.load(pp, map_location="cpu", weights_only=False)
            if raw is None and isinstance(pair, dict) and sigma_field in pair:
                raw = float(pair[sigma_field])
            if raw is None:
                rejects.append({"file": path.name,
                                "reason": f"missing sigma field '{sigma_field}' (cache + pair)"})
                continue
            sigma = _sigma_from_t(raw, sigma_transform, shift)
        elif sigma_mode == "training_uniform":
            # 训练脚本 --p2-resigma 会丢掉 cache t；这里按实际训练区间估算覆盖。
            sigma = random.Random(path.name).uniform(sigma_min, sigma_max)
        else:
            raise ValueError(f"unknown sigma mode: {sigma_mode}")
        sigma_values.append(sigma)
        sbin = _sigma_bin(sigma)
        scene_counts[scene] += 1

        if sbin == "sigma_out_of_range":
            rejects.append({"file": path.name, "sigma": round(sigma, 6),
                            "scene": scene, "reasons": [f"sigma {sigma:.4f} outside [0.05,1.0]"]})
            continue

        q = _quality(x0)
        reasons = []
        if q["std"] < min_std:
            reasons.append(f"std {q['std']:.3f} < {min_std}")
        if q["spatial_grad"] < min_grad:
            reasons.append(f"spatial_grad {q['spatial_grad']:.3f} < {min_grad}")
        if q["flat_ratio"] > max_flat:
            reasons.append(f"flat_ratio {q['flat_ratio']:.3f} > {max_flat}")
        if q["border_flat_ratio"] > max_border_flat:
            reasons.append(f"border_flat {q['border_flat_ratio']:.3f} > {max_border_flat}")

        h = _hash_tensor(x0)
        hashes[h] += 1
        if hashes[h] > 1:
            reasons.append(f"duplicate x0 (hash {h[:10]})")

        if reasons:
            rejects.append({"file": path.name, "sigma": round(sigma, 6),
                            "scene": scene, "reasons": reasons})
            continue
        groups[f"{sbin}/{scene}"].append({
            "file": path.name,
            "sigma": round(sigma, 6),
            "prompt": prompt[:120],
            **q,
        })

    missing = []
    for sbin, _, _ in SIGMA_BINS:
        for scene in list(SCENE_KEYWORDS) + ["其他"]:
            key = f"{sbin}/{scene}"
            if len(groups.get(key, [])) < min_per_group:
                missing.append({"group": key, "count": len(groups.get(key, [])),
                                "required": min_per_group})

    return {
        "cache": str(cache),
        "pair_cache": str(pair_dir) if pair_dir else None,
        "n_files": len(files),
        "n_kept": sum(len(v) for v in groups.values()),
        "n_rejected": len(rejects),
        "sigma_field": sigma_field,
        "sigma_transform": sigma_transform,
        "sigma_mode": sigma_mode,
        "sigma_stats": {
            "min": round(min(sigma_values), 6) if sigma_values else None,
            "max": round(max(sigma_values), 6) if sigma_values else None,
            "mean": round(sum(sigma_values) / len(sigma_values), 6) if sigma_values else None,
        } if sigma_values else {},
        "sigma_stats_ext": _sigma_stats_ext(sigma_values),
        "scene_counts": dict(scene_counts),
        "groups": {k: len(v) for k, v in sorted(groups.items())},
        "missing_or_underfilled": missing,
        "rejects": rejects,
        "kept": {k: v for k, v in sorted(groups.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-field", default="t")
    ap.add_argument("--sigma-transform", default="none",
                    choices=["none", "shift8_forward", "shift8_inverse"])
    ap.add_argument("--sigma-mode", default="cache",
                    choices=["cache", "training_uniform"])
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=8.0)
    ap.add_argument("--min-std", type=float, default=0.10)
    ap.add_argument("--min-grad", type=float, default=0.01)
    ap.add_argument("--max-flat", type=float, default=0.50)
    ap.add_argument("--max-border-flat", type=float, default=0.80)
    ap.add_argument("--min-per-group", type=int, default=20)
    ap.add_argument("--allow-underfilled", action="store_true")
    ap.add_argument("--pair-cache", default="",
                    help="可选：配对教师缓存目录；本目录缺 t/prompt 时按同名文件继承，"
                         "用于无教师 latent 池（如 ds1）做同构分组")
    args = ap.parse_args()

    report = build_gate(
        args.cache, sigma_field=args.sigma_field,
        sigma_transform=args.sigma_transform, shift=args.shift,
        sigma_mode=args.sigma_mode, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        min_std=args.min_std, min_grad=args.min_grad,
        max_flat=args.max_flat, max_border_flat=args.max_border_flat,
        min_per_group=args.min_per_group,
        pair_cache=args.pair_cache or None,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_files", "n_kept", "n_rejected", "sigma_stats",
                       "groups", "missing_or_underfilled")},
                     ensure_ascii=False, indent=2))
    if report["missing_or_underfilled"] and not args.allow_underfilled:
        raise SystemExit("DATA_GATE_FAILED: underfilled groups; see --out")


if __name__ == "__main__":
    main()
