#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyraGate Sakiko - 阶段感知训练器（16000 步总序列，适配当前 7900XT / Adafactor / 离线 CFG-A）

与蓝图的关键差异（按现状修正，不是照搬）：
  1. 优化器保持 Adafactor（4B 全参 AdamW 光动量就要 ~32GB，21GB 卡装不下）；
     只加 weight_decay 参数，不引入 bitsandbytes 依赖。
  2. 阶段/lr/loss/数据集由全局步号调度，不再依赖 --teacher-dir 全局开关；
     蒸馏读 preencode_teacher.py 产出的离线 .pt（v_teacher），教师不进训练进程。
  3. checkpoint 默认不自动删除；只有显式 --prune 才清理，且阶段末版本单独永久保存。
  4. 数据集缺失时直接报错，绝不静默退化成随机 latent。
  5. --until-step 可先只跑当前能跑的段（例如 4000），不必等 DS3/DS4/教师缓存齐备。

全局阶段（步号为 1-based，含端点）：
  P1     1..4000     自监督 DS1        lr 2e-4 -> 1.5e-4（前 400 步 warmup）
  P2  4001..7000     离线蒸馏 DS2      lr 1.5e-4 -> 1.2e-4
  P3  7001..14000    28 轮 x (200 步 DS3 自监督 + 50 步 DS2 蒸馏回炉)
  P4 14001..15000    1000 步 DS3 自监督收尾，lr 6e-5 -> 4e-5
  P5 15001..16000    1000 步 DS4 二次元专项，lr 2e-5 -> 1e-5

退出码：0 正常 / 42 OOM / 43 周期重启（由守护脚本拉起续训）。
"""
import os
# ===== ROCm / MIOpen 调优：必须在 import torch 之前（与 train_backbone.py 一致）=====
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
_MIOPEN_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "miopen_cache")
os.makedirs(_MIOPEN_DB, exist_ok=True)
os.environ.setdefault("MIOpenFindDbPath", _MIOPEN_DB)
os.environ.setdefault("MIOpen_FIND_DB", "1")
os.environ.setdefault("MIOpen_CACHE_MODE", "Normal")
os.environ.setdefault("AMD_LOG_LEVEL", "0")
os.environ.setdefault("HSA_ENABLE_SDMA", "1")
os.environ.setdefault("ROCM_USE_FLUSH_DEFER", "1")
# Flash/Triton/torch.compile 默认把缓存写系统盘、易被编译临时文件写满；
# 训练进程统一把缓存重定向到项目内 cache 目录。
_CACHE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache"))
_cache_dirs = {
    "TRITON_CACHE_DIR": os.path.join(_CACHE_ROOT, "triton"),
    "TORCHINDUCTOR_CACHE_DIR": os.path.join(_CACHE_ROOT, "torchinductor"),
    "TORCH_EXTENSIONS_DIR": os.path.join(_CACHE_ROOT, "torch_extensions"),
    "TEMP": os.path.join(_CACHE_ROOT, "tmp"),
    "TMP": os.path.join(_CACHE_ROOT, "tmp"),
}
for _k, _v in _cache_dirs.items():
    os.makedirs(_v, exist_ok=True)
    os.environ.setdefault(_k, _v)

import argparse
import glob
import gc
import json
import math
import os as _os
import random
import subprocess
import sys
import threading
import time
import zlib
from collections import namedtuple
from collections.abc import Sequence

# 注意：torch 与训练依赖延迟到真正开训前导入；--dry-run 全程不 import torch，
# 遵守项目红线（避免初始化 ROCm context 吞掉数 GB 虚拟地址）。

def list_pt(d):
    return sorted(glob.glob(os.path.join(d, "*.pt"))) if d else []


def _merge_pt(*dirs):
    files = []
    for d in dirs:
        if d:
            files.extend(list_pt(d))
    return files


def _interleave(a, b):
    """按 1:1 交错两个列表（保留兼容）。"""
    out = []
    n = max(len(a), len(b))
    for i in range(n):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


class MixedFiles(Sequence):
    """确定性比例混采序列：每 period 个样本里 a 占 a_per、b 占 b_per。
    按 idx 连续取也能保持比例，不依赖随机状态；9f:13f=1:2 即 a_per=1,b_per=2。"""

    def __init__(self, a, b, a_per=1, b_per=2):
        self.a = list(a)
        self.b = list(b)
        self.a_per = max(1, int(a_per))
        self.b_per = max(0, int(b_per))
        self.period = self.a_per + self.b_per
        self._len = self.period * max(len(self.a), len(self.b), 1)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError(i)
        block, pos = divmod(i, self.period)
        if pos < self.a_per:
            return self.a[(block * self.a_per + pos) % len(self.a)]
        k = pos - self.a_per
        return self.b[(block * self.b_per + k) % len(self.b)]


class SourceBalancedFiles(Sequence):
    """按「源视频」而不是按「文件」配平的非对称采样器（2026-11-23）。

    背景：9f 池 700 条与 13f 池的前 700 个源视频完全重叠，13f 另有 1283 个
    独有源。旧的 1:3 文件级混采让这 700 个源同时吃到 9f+13f（约 1.94 倍暴露），
    另外 1283 个源只有 13f（相对饿死）。

    本采样器每个唯一源每轮固定 total_slots 个槽：
      - 有 9f 的源：前 nine_slots 个槽给 9f，其余给 13f；
      - 只有 13f 的源：全部给 13f。
    默认 3 个 9f + 1 个 13f ⇒ 9f 样本占比 26.5%（接近原 25%，token 占比约 21%），
    但每个源的总曝光严格相等，不再有源被撑死/饿死。
    """

    def __init__(self, a, b, nine_slots=3, total_slots=4, shuffle_seed=0):
        self.nine = {}
        self.thirteen = {}
        for p in a:
            self.nine.setdefault(_src_key(p), []).append(p)
        for p in b:
            self.thirteen.setdefault(_src_key(p), []).append(p)
        self.sources = sorted(set(self.nine) | set(self.thirteen))
        if shuffle_seed:
            random.Random(int(shuffle_seed)).shuffle(self.sources)
        self.shuffle_seed = int(shuffle_seed)
        self._orders = {}
        self.nine_slots = max(0, min(int(nine_slots), int(total_slots)))
        self.total_slots = max(1, int(total_slots))
        self._len = self.total_slots * max(len(self.sources), 1)

    def _order(self, slot: int):
        """每个槽位（即每轮遍历）换一次源顺序，避免固定课程长期复现。"""
        order = self._orders.get(slot)
        if order is None:
            order = list(self.sources)
            rng = random.Random((self.shuffle_seed or 1) + slot * 1000003)
            rng.shuffle(order)
            self._orders[slot] = order
        return order

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError(i)
        slot, pos = divmod(i, len(self.sources))
        src = self._order(slot)[pos]
        if src in self.nine and slot < self.nine_slots:
            return self.nine[src][0]
        return self.thirteen[src][0]



def _parse_ratio(s):
    """'a:b' -> (a,b)，用于 ds1 与 extra 的确定性混采比例。"""
    parts = str(s).split(":")
    if len(parts) != 2:
        raise ValueError(f"ratio 必须是 a:b，收到 {s!r}")
    return max(1, int(parts[0])), max(0, int(parts[1]))


def _src_key(path):
    """按源视频路径做 holdout 分组：同一视频的 9f/14f 片段要一起留出，避免泄漏。"""
    try:
        rec = torch.load(path, map_location="cpu", weights_only=False)
        src = rec.get("src", path) if isinstance(rec, dict) else path
        return str(src)
    except Exception:
        return path


def _split_holdout(files, ratio):
    """确定性 5% 留出：crc32(src) 取模，不依赖随机状态，resume 后划分不变。"""
    if ratio <= 0:
        return list(files), []
    cut = int(round(float(ratio) * 10000))
    keep, hold = [], []
    seen = {}
    for p in files:
        if p not in seen:
            seen[p] = zlib.crc32(_src_key(p).encode("utf-8")) % 10000
        (hold if seen[p] < cut else keep).append(p)
    return keep, hold


class ShuffledFiles(Sequence):
    """固定种子 shuffle 的只读序列：make_batch 仍按 idx % len 取样本，
    但实际顺序由启动时按 seed 生成一次的置换决定，resume 可复现。"""

    def __init__(self, files, seed=0):
        self.files = list(files)
        self.order = list(range(len(self.files)))
        if seed:
            random.Random(int(seed)).shuffle(self.order)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        return self.files[self.order[i]]


class GroupedFiles(Sequence):
    """按 data_gate 的 σ/场景组轮转，避免把 2200 条直接混成一个随机池。

    groups: {group_name: [paths]}；组内用固定 seed 打散，组间按轮转顺序交错。
    这样每个组被等权遍历，batch 内也不会长期被“人物/高 σ”单一组垄断。
    """

    def __init__(self, groups, seed=0):
        self.groups = {str(k): list(v) for k, v in groups.items() if v}
        self.group_names = sorted(self.groups)
        if not self.group_names:
            raise ValueError("GroupedFiles: no non-empty groups")
        for name in self.group_names:
            if seed:
                rng = random.Random(int(seed) + zlib.crc32(name.encode("utf-8")))
                rng.shuffle(self.groups[name])
        self.order = []
        max_len = max(len(self.groups[g]) for g in self.group_names)
        for i in range(max_len):
            for g in self.group_names:
                if i < len(self.groups[g]):
                    self.order.append(self.groups[g][i])
        self._len = len(self.order)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError(i)
        return self.order[i]


def apply_data_gate(files, report_path, strict=False, seed=0):
    """按 data_gate JSON 的 kept 组过滤，并返回 GroupedFiles。"""
    if not report_path or not os.path.isfile(report_path):
        raise FileNotFoundError(f"data-gate report missing: {report_path}")
    report = json.loads(open(report_path, "r", encoding="utf-8").read())
    kept = report.get("kept") or {}
    if not kept:
        raise SystemExit(f"[data-gate] {report_path} 没有 kept 组，拒绝启动")
    by_name = {os.path.basename(p): p for p in files}
    groups, missing = {}, []
    for gname, items in kept.items():
        paths = []
        for it in items:
            name = os.path.basename(str(it.get("file", "")))
            p = by_name.get(name)
            if p is not None:
                paths.append(p)
        if paths:
            groups[gname] = paths
        else:
            missing.append(gname)
    if not groups:
        raise SystemExit(f"[data-gate] {report_path} 的 kept 文件在 ds2 目录里一个都对不上")
    under = report.get("missing_or_underfilled") or []
    print(f"[data-gate] report={report_path} kept={report.get('n_kept')} "
          f"rejected={report.get('n_rejected')} groups={len(groups)} "
          f"matched={sum(len(v) for v in groups.values())}/{len(files)}", flush=True)
    if under:
        msg = f"[data-gate] underfilled_groups={len(under)} (e.g. {under[:3]})"
        if strict:
            raise SystemExit("DATA_GATE_FAILED: " + msg)
        print(msg + "  [warn only; pass --data-gate-strict to fail]", flush=True)
    if missing:
        print(f"[data-gate] kept groups with no matching ds2 files: {missing[:5]}", flush=True)
    return GroupedFiles(groups, seed=seed)


# ---------------------------------------------------------------------------
# 2026-11-26: ds1 每批采样统计落盘（--data-gate-report-dir）；口径复用 data_gate.py
# ---------------------------------------------------------------------------
def _pct_sorted(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 6)


def _merge_hist(dicts):
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _merge_counter(dicts):
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


class DataGateBatchReporter:
    """ds1 每批采样统计：σ 分桶分布 + 质量分分位 + 来源计数，jsonl 追加落盘。

    口径全部复用 model/data_gate.py：σ 分桶边界 ``SIGMA_BINS``/``_sigma_bin``，
    质量分 ``_quality``（std/spatial_grad/temporal_motion/flat_ratio/border_flat_ratio）。
    纯 CPU、不调用任何 RNG；每条记录写完即 flush，不阻塞训练主路径。
    """

    QUALITY_METRICS = ("std", "spatial_grad", "temporal_motion",
                       "flat_ratio", "border_flat_ratio")

    def __init__(self, out_dir, rollup_every=50):
        self.dir = os.path.abspath(str(out_dir))
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "batches.jsonl")
        self.rollup_every = max(1, int(rollup_every))
        self.batch_no = 0
        self._rollup_buf: list[dict] = []
        self._rollup_index = 0
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")

    @staticmethod
    def _import_gate():
        try:
            from . import data_gate as _dg  # type: ignore
        except Exception:
            import data_gate as _dg  # type: ignore
        return _dg

    @staticmethod
    def _src_key(path):
        """best-effort 读 pt 内 src 字段；失败退回文件名。不参与 RNG。"""
        if not path:
            return ""
        try:
            import torch as _torch
            obj = _torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and obj.get("src"):
                return str(obj["src"])
        except Exception:
            pass
        return os.path.basename(str(path))

    def record(self, step, batch_id, dataset, file_path, sigma_vec, x0):
        dg = self._import_gate()
        sigmas = [float(v) for v in
                  sigma_vec.detach().float().reshape(-1).cpu().tolist()]
        sigma_stats = dg._sigma_stats_ext(sigmas) if sigmas else {}
        q_rows: list[dict] = []
        x0c = x0.detach().float().cpu()
        for i in range(x0c.shape[0]):
            q_rows.append(dg._quality(x0c[i:i + 1]))
        quality: dict[str, dict] = {}
        for m in self.QUALITY_METRICS:
            vals = [r[m] for r in q_rows if m in r]
            if vals:
                vs = sorted(vals)
                quality[m] = {
                    "mean": round(sum(vals) / len(vals), 6),
                    "p25": _pct_sorted(vs, 0.25),
                    "p50": _pct_sorted(vs, 0.50),
                    "p75": _pct_sorted(vs, 0.75),
                }
        src = self._src_key(file_path)
        src_id = os.path.splitext(os.path.basename(src))[0].split("_")[0] if src else ""
        record = {
            "step": int(step),
            "batch_id": int(batch_id),
            "dataset": str(dataset),
            "n_samples": len(sigmas),
            "file": os.path.basename(str(file_path)) if file_path else None,
            "sigma": sigma_stats,
            "quality": quality,
            "sources": {src_id or "unknown": 1},
        }
        self._write(record)
        self._rollup_buf.append(record)
        if len(self._rollup_buf) >= self.rollup_every:
            self._flush_rollup()

    def _write(self, record):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _flush_rollup(self):
        if not self._rollup_buf:
            return
        rows = self._rollup_buf
        quality_mean = {}
        for m in self.QUALITY_METRICS:
            vals = [r["quality"][m]["mean"] for r in rows if m in r["quality"]]
            if vals:
                quality_mean[m] = round(sum(vals) / len(vals), 6)
        agg = {
            "rollup_index": self._rollup_index,
            "n_batches": len(rows),
            "batch_range": [rows[0]["step"], rows[-1]["step"]],
            "n_samples": sum(r["n_samples"] for r in rows),
            "sigma_histogram": _merge_hist([r["sigma"].get("histogram", {}) for r in rows]),
            "sources": _merge_counter([r["sources"] for r in rows]),
            "quality_mean": quality_mean,
        }
        self._rollup_index += 1
        self._rollup_buf = []
        rp = os.path.join(self.dir, f"rollup_{self._rollup_index:04d}.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

    def close(self):
        try:
            self._flush_rollup()
            if not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


# 探针：每步参数位移 q/step、master-bf16 差、峰值显存、步耗时（--probe-qlog，默认关）
_PROBE_NAMES = (
    "patchify.weight", "head.head.weight",
    "full_head.final_proj.weight", "full_head.pre_proj_scale",
    "enc_blocks.0.attn.qkv.weight", "enc_blocks.11.attn.qkv.weight",
    "dec_blocks.0.attn.qkv.weight", "dec_blocks.11.attn.qkv.weight",
)


class ProbeQLog:
    """可选遥测：只在 --probe-qlog 给目录时构造；关闭时主路径零开销。"""

    def __init__(self, out_dir):
        self.dir = os.path.abspath(str(out_dir))
        os.makedirs(self.dir, exist_ok=True)
        self._fh = open(os.path.join(self.dir, "probe.jsonl"), "a", encoding="utf-8")
        self.ref: dict[str, torch.Tensor] = {}
        self.ref_dim1: set[str] = set()
        self._audit_written = False

    def snapshot(self, model):
        params = dict(model.named_parameters())
        self.ref = {}
        self.ref_dim1 = set()
        for n in _PROBE_NAMES:
            if n in params:
                self.ref[n] = params[n].detach().float().flatten().clone()
                if params[n].dim() <= 1:
                    self.ref_dim1.add(n)

    @staticmethod
    def _vram_mb():
        try:
            return (round(torch.cuda.max_memory_allocated() / 1e6, 3),
                    round(torch.cuda.max_memory_reserved() / 1e6, 3))
        except Exception:
            return None, None

    @staticmethod
    def _group(d_parts, q_parts, moved, total):
        if d_parts:
            d_all = torch.cat(d_parts)
            q_all = torch.cat(q_parts)
            return {
                "numel": total,
                "delta_l2": float(d_all.norm().item()),
                "delta_rms": float(d_all.pow(2).mean().sqrt().item()),
                "delta_quanta_mean": float(q_all.mean().item()),
                "moved_frac": moved / max(1, total),
            }
        return {"numel": 0, "delta_l2": 0.0, "delta_rms": 0.0,
                "delta_quanta_mean": 0.0, "moved_frac": 0.0}

    def record_step(self, step, step_time, sr_master, model, skip_update):
        try:
            from bf16_sr import bf16_quantum
        except Exception:
            from .bf16_sr import bf16_quantum  # type: ignore
        params = dict(model.named_parameters())
        agg = {"d": [], "q": [], "moved": 0, "total": 0}
        grp = {"dim1": {"d": [], "q": [], "moved": 0, "total": 0},
               "weight": {"d": [], "q": [], "moved": 0, "total": 0}}
        for n, ref in self.ref.items():
            p = params.get(n)
            if p is None:
                continue
            cur = p.detach().float().flatten()
            d = cur - ref
            q = bf16_quantum(ref)
            qd = d.abs() / q.clamp_min(1e-38)
            mv = int((d.abs() > 0).sum().item())
            t = d.numel()
            bucket = grp["dim1"] if n in self.ref_dim1 else grp["weight"]
            for g in (agg, bucket):
                g["d"].append(d)
                g["q"].append(qd)
                g["moved"] += mv
                g["total"] += t
        master_diff = None
        if sr_master is not None and sr_master.enabled:
            md_parts, ma_parts = [], []
            for _, src, master in sr_master.pairs:
                a = master.detach().float().flatten()
                b = src.detach().float().flatten()
                md_parts.append((a - b).abs())
                ma_parts.append(a - b)
            if md_parts:
                md = torch.cat(md_parts)
                ma = torch.cat(ma_parts)
                master_diff = {"max_abs": float(md.max().item()),
                               "mean_abs": float(md.mean().item()),
                               "l2": float(ma.norm().item())}
        rec = {
            "step": int(step),
            "skip_update": bool(skip_update),
            "step_time_s": round(float(step_time), 4),
            "tracked_tensors": len(self.ref),
            "tracked_numel": agg["total"],
            "delta_l2": self._group(agg["d"], agg["q"], agg["moved"], agg["total"])["delta_l2"],
            "delta_rms": self._group(agg["d"], agg["q"], agg["moved"], agg["total"])["delta_rms"],
            "delta_quanta_mean": self._group(agg["d"], agg["q"], agg["moved"], agg["total"])["delta_quanta_mean"],
            "moved_frac": self._group(agg["d"], agg["q"], agg["moved"], agg["total"])["moved_frac"],
            "groups": {k: self._group(v["d"], v["q"], v["moved"], v["total"])
                       for k, v in grp.items()},
            "master_bf16_diff": master_diff,
            "vram_alloc_mb": self._vram_mb()[0],
            "vram_reserved_mb": self._vram_mb()[1],
        }
        if sr_master is not None and sr_master.enabled and not self._audit_written:
            rec["sr_audit"] = sr_master.audit()
            self._audit_written = True
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            if not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


def _load_runtime():
    global torch, build_model, make_batch, _atomic_save
    global load_text_tokens
    global count_parameters, freeze_spectral_power_iteration, reinit_linear_teacher_spectrum
    global is_gate_param, promote_gate_params_fp32
    global sampled_token_corr_loss, patch_mean_pearson_corr_defect, corr_defect_penalty
    global x0_fourier_band_loss, x0_band_energy_ratios, split_frequency_bands
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch as _torch
    from train_backbone import build_model as _build_model
    from train_backbone import make_batch as _make_batch
    from train_backbone import load_text_tokens as _load_text_tokens
    from train_backbone import _atomic_save as _atomic
    from uvit3d import count_parameters as _count
    from uvit3d import freeze_spectral_power_iteration as _freeze
    from uvit3d import reinit_linear_teacher_spectrum as _reinit
    from uvit3d import is_gate_param as _is_gate, promote_gate_params_fp32 as _promote
    from full_gated_head import sampled_token_corr_loss as _corr_scalar
    from full_gated_head import patch_mean_pearson_corr_defect as _patch_corr_defect
    from full_gated_head import corr_defect_penalty as _corr_penalty
    from x0_fourier_band_loss import x0_fourier_band_loss as _x0_fourier
    from x0_fourier_band_loss import x0_band_energy_ratios as _x0_band_ratios
    from x0_fourier_band_loss import split_frequency_bands as _split_bands
    torch = _torch
    build_model, make_batch, _atomic_save = _build_model, _make_batch, _atomic
    load_text_tokens = _load_text_tokens
    count_parameters = _count
    freeze_spectral_power_iteration, reinit_linear_teacher_spectrum = _freeze, _reinit
    is_gate_param, promote_gate_params_fp32 = _is_gate, _promote
    sampled_token_corr_loss = _corr_scalar
    patch_mean_pearson_corr_defect = _patch_corr_defect
    corr_defect_penalty = _corr_penalty
    x0_fourier_band_loss = _x0_fourier
    x0_band_energy_ratios = _x0_band_ratios
    split_frequency_bands = _split_bands
    return torch


def _split_param_groups(items, n_chunks):
    """按 numel 尽量均衡切组；Adafactor foreach 按 group 批量执行，分块可压低临时峰值。"""
    n_chunks = max(1, int(n_chunks))
    if n_chunks == 1 or len(items) <= 1:
        return [[p for _, p in items]]
    total = sum(p.numel() for _, p in items)
    target = max(1, (total + n_chunks - 1) // n_chunks)
    groups, cur, cur_n = [], [], 0
    for _, p in items:
        pn = p.numel()
        if cur and cur_n + pn > target and len(groups) < n_chunks - 1:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append(p)
        cur_n += pn
    if cur:
        groups.append(cur)
    return groups


def make_optimizers(model, args, param_source=None):
    """主参数 Adafactor + 门控参数独立 AdamW（门控 fp32，绕开 eps2/bf16 冻结）。

    param_source: 可选 (name, param) 迭代器。SR-master 模式下由
    ``SelectiveSRMaster.param_source(model)`` 提供，把选中张量换成 fp32 master。
    """
    gate, main, head, head_final, band_cond, band_head, freqrouter, dual_sem = [], [], [], [], [], [], [], []
    new_module = []
    for n, p in (param_source(model) if param_source is not None
                 else model.named_parameters()):
        if not p.requires_grad:
            continue
        if n.startswith("dual_sem_head."):
            # 2026-11-26 实测：dual_sem_head 88.2M/24 张量放进 Adafactor 后
            # opt.step 2.45s（关掉同架构 0.40s）；挪到 AdamW 组。
            dual_sem.append((n, p))
        elif n.startswith("band_mlp."):
            # A1 只改共享 cond，1e-4 一步就能把 x0 从 0.104 抬到 0.146；
            # 必须独立更小的 lr，不能和 A3 带头共用。
            band_cond.append((n, p))
        elif n.startswith("freqrouter."):
            # FreqFormer 路由：独立 AdamW 组，lr=1e-5（比 band_cond 5e-6 略高，
            # 因为门控直接乘在注意力上，需要更快跟上；可调）。
            freqrouter.append((n, p))
        elif any(tag in n for tag in (".text_cross.", ".text_adaln.",
                                      ".sigma_adapter.", ".context_tokens.",
                                      "context_tokens.", "repa_proj.")):
            # v3.2 新模块显式白名单：不能让它们落进主 Adafactor。低秩 Q/O、
            # σ adapter、context token、REPA projector 都是零初始化/小秩，
            # 用独立 AdamW（lr=1e-4, wd=0）保证新前缀一定会被优化器看到。
            new_module.append((n, p))
        elif n.startswith("full_head.band_head_") or n.startswith("full_head.bandlr."):
            # A3 三头 + 新低秩频带头：零初始化残差，走现成 band_lr=1e-4 AdamW。
            band_head.append((n, p))
        elif n.startswith(("full_head.high_gate.", "full_head.dualact_gate.",
                           "full_head.band_norm.", "full_head.patch_norm.")):
            # 2026-11-24 分层解耦 σ 门控 + 内嵌 Dual-Act 全部新增模块：
            # 独立 AdamW + fp32。零初始化 Rezero 放进 Adafactor 会被
            # eps2=1e-3 的 α 下限冻死（实测 4 步只挪 4e-8）。
            gate.append((n, p))
        elif n.startswith("full_head."):
            # final_proj.weight 单独成组、wd=0：全局 wd 会持续把它压回 0，
            # 这是 step600 输出幅度只有 ~3e-3 的直接推手之一。
            # A：pre_proj_scale 也归入幅值组，wd=0；L2 由 head 内的分频正则负责。
            if n in ("full_head.final_proj.weight", "full_head.pre_proj_scale"):
                head_final.append((n, p))
            else:
                head.append((n, p))
        elif is_gate_param(n):
            gate.append((n, p))
        else:
            main.append((n, p))
    use_foreach = bool(getattr(args, "opt_foreach", True))
    chunks = int(getattr(args, "opt_foreach_chunks", 1)) if use_foreach else 1
    main_groups = _split_param_groups(main, chunks)
    groups = [{"params": g} for g in main_groups if g]
    if head:
        groups.append({"params": [p for _, p in head],
                       "lr": args.head_lr,
                       "weight_decay": args.head_wd,
                       "head_group": True})
    # 2026-11-25：final_proj.weight 的 RMS≈5.35e-4，Adafactor 的
    # α=max(eps2,RMS(θ)) 会把它每步压到 ~5e-7；200 步只挪 2.2e-5。
    # 输出投影必须走 AdamW（逐元素归一化），否则 Fourier/x0 梯度
    # 只能推 out_gain，推不动频谱形状。head_final 改挂 opt_gate。
    if not groups:
        # 阶段1只训 gate 模块时 main/head 都可能为空；Adafactor 不接受空
        # parameter list。放一个永不产生梯度的 dummy 参数，保证 opt 接口
        # （zero_grad/step/param_groups）仍然可用且不更新任何模型权重。
        dummy = torch.nn.Parameter(next(model.parameters()).new_zeros(1),
                                   requires_grad=True)
        groups = [{"params": [dummy], "dummy_group": True}]
    opt = torch.optim.Adafactor(groups,
                                lr=2e-4,
                                weight_decay=args.weight_decay,
                                eps=(args.opt_eps1, args.opt_eps2),
                                foreach=use_foreach)
    opt_gate = None
    if gate or head_final or band_cond or band_head or freqrouter or dual_sem or new_module:
        gate_groups = []
        if new_module:
            gate_groups.append({"params": [p for _, p in new_module],
                                "lr": 1e-4,
                                "weight_decay": 0.0,
                                "new_module_group": True})
        if gate:
            gate_groups.append({"params": [p for _, p in gate],
                                "lr": args.gate_lr,
                                "weight_decay": args.gate_wd,
                                "gate_group": True})
        if head_final:
            gate_groups.append({"params": [p for _, p in head_final],
                                "lr": args.head_final_lr,
                                "weight_decay": 0.0,
                                "head_final_group": True})
        if band_cond:
            gate_groups.append({"params": [p for _, p in band_cond],
                                "lr": args.band_cond_lr,
                                "weight_decay": 0.0,
                                "band_cond_group": True})
        if band_head:
            gate_groups.append({"params": [p for _, p in band_head],
                                "lr": args.band_lr,
                                "weight_decay": 0.0,
                                "band_head_group": True})
        if freqrouter:
            gate_groups.append({"params": [p for _, p in freqrouter],
                                "lr": 1e-5,
                                "weight_decay": 0.0,
                                "freqrouter_group": True})
        if dual_sem:
            gate_groups.append({"params": [p for _, p in dual_sem],
                                "lr": 2e-4,
                                "weight_decay": 0.0,
                                "dual_sem_group": True})
        opt_gate = torch.optim.AdamW(gate_groups, lr=args.gate_lr,
                                     weight_decay=args.gate_wd,
                                     betas=(0.9, 0.95), eps=1e-8)
        print(f"[opt-gate] AdamW n_gate={len(gate)} n_head_final={len(head_final)} "
              f"n_band_cond={len(band_cond)} n_band_head={len(band_head)} "
              f"n_freqrouter={len(freqrouter)} n_dual_sem={len(dual_sem)} "
              f"n_new_module={len(new_module)} lr=2.0e-04 "
              f"gate_lr={args.gate_lr:.1e} "
              f"head_final_lr={args.head_final_lr:.1e} "
              f"band_cond_lr={args.band_cond_lr:.1e} band_lr={args.band_lr:.1e} "
              f"gate_wd={args.gate_wd:.1e} "
              f"names={[n for n,_ in gate[:6]]}", flush=True)
    n2d = sum(1 for _, p in main if p.dim() >= 2)
    n1d = len(main) - n2d
    print(f"[opt] Adafactor n={len(main)} (2D={n2d}, 1D/vector={n1d}) "
          f"foreach={use_foreach} chunks={len(main_groups)} "
          f"factorized=internal(2D row/col, 1D scalar) "
          f"wd={args.weight_decay} eps1={args.opt_eps1:.1e} "
          f"eps2={args.opt_eps2:.1e}（门控已从主优化器排除）", flush=True)
    if head:
        print(f"[opt-head] Adafactor n={len(head)} lr={args.head_lr:.1e} "
              f"wd={args.head_wd:.1e}（独立参数组，不随主 lr 调度覆盖）", flush=True)
    if head_final:
        print(f"[opt-head-final] AdamW n={len(head_final)} lr={args.head_final_lr:.1e} "
              f"wd=0（full_head.final_proj.weight + pre_proj_scale 排除 weight decay；"
              f"不经过 Adafactor eps2/RMS 下限）", flush=True)
    return opt, opt_gate


def _build_sr_master(model, args):
    """按 --sr-master / --sr-master-budget-m / --sr-modules 建选择性 fp32 master。

    未显式开启（--sr-master 未给且 budget<=0）直接返回 None：整条 SR 路径不激活，
    训练行为与改动前逐位一致。
    """
    budget = float(getattr(args, "sr_master_budget_m", 0.0) or 0.0)
    flag_on = bool(getattr(args, "sr_master", False))
    if not flag_on and budget <= 0:
        return None
    if budget <= 0:
        budget = 300.0  # 只给了 --sr-master 未给预算时的安全默认（百万参数）
    try:  # 作为包导入（python -m model.train_staged）
        from .bf16_sr import SelectiveSRMaster
    except Exception:  # 直接脚本运行（python model/train_staged.py）
        from bf16_sr import SelectiveSRMaster
    modules = str(getattr(args, "sr_modules", "") or "").strip()
    if modules:
        prefixes = [p.strip() for p in modules.split(",") if p.strip()]
    else:
        prefixes = list(getattr(args, "sr_master_prefixes", None) or [])
    include_dim1 = bool(getattr(args, "sr_include_dim1", True))
    sr = SelectiveSRMaster(model, prefixes=prefixes, budget_m=budget,
                           include_dim1=include_dim1)
    sr.push_every = max(1, int(getattr(args, "sr_every", 1) or 1))
    print(sr.summary(), flush=True)
    if sr.enabled:
        _aud = sr.audit()
        print(f"[sr-master][audit] protected={_aud['n_selected']} "
              f"dim1={len(_aud['dim1_names'])} weight={len(_aud['weight_names'])} "
              f"skipped_dim1={len(_aud['skipped_dim1'])} "
              f"skipped_exclude={len(_aud['skipped_exclude'])} "
              f"already_fp32={len(_aud['already_fp32'])}", flush=True)
        if "full_head.pre_proj_scale" in _aud["dim1_names"]:
            print("[sr-master][audit] full_head.pre_proj_scale 已纳入 SR master 保护", flush=True)
        elif "full_head.pre_proj_scale" in _aud["skipped_dim1"] + _aud["skipped_exclude"]:
            print("[sr-master][audit] full_head.pre_proj_scale 未被保护（dim1/exclude 跳过）",
                  flush=True)
        print("[sr-master] 选中张量：优化器在 fp32 master 上累加，每步随机舍入写回 bf16；"
              "未选中张量仍原位 bf16 更新（行为不变）", flush=True)
        if not bool(getattr(args, "reset_optim", False)):
            print("[sr-master][warn] SR enabled — requires -ResetOptim：未加 --reset-optim 时，"
                  "续训旧 Adafactor/AdamW 的一阶/二阶动量与 fp32 master 参数源不一致，"
                  "启动命令必须补 --reset-optim。", flush=True)
        if sr.push_every > 1:
            print(f"[sr-master] push_every={sr.push_every}：master 每 {sr.push_every} 步 SR 写回 bf16",
                  flush=True)
    return sr


def audit_optimizer_coverage(model, opt, opt_gate, param_source=None):
    """硬校验：每个可训参数必须恰好进入一个优化器，禁止空组。

    param_source 用于 SR-master 模式：此时优化器持有的是 fp32 master，
    校验目标也要切成同一份 (name, param) 视图，否则会把 master 判成
    "unexpected"、把模型参数判成 "missing"。
    """
    src = list(param_source(model)) if param_source is not None else list(model.named_parameters())
    trainable = {id(p): n for n, p in src if p.requires_grad}
    seen, unexpected = {}, []
    for optimizer, label in ((opt, "main"), (opt_gate, "gate")):
        if optimizer is None:
            continue
        for gi, group in enumerate(optimizer.param_groups):
            if not group["params"]:
                raise RuntimeError(f"[opt-audit] empty parameter group: {label}/{gi}")
            for p in group["params"]:
                pid = id(p)
                name = trainable.get(pid)
                if name is None:
                    # 唯一允许的非模型参数是显式标记的 dummy 组。
                    if not group.get("dummy_group"):
                        unexpected.append(f"{label}/{gi}")
                    continue
                seen.setdefault(name, []).append(f"{label}/{gi}")
    missing = sorted(set(trainable.values()) - set(seen))
    duplicate = {n: where for n, where in seen.items() if len(where) != 1}
    if missing or duplicate or unexpected:
        raise RuntimeError(f"[opt-audit] missing={missing[:8]} duplicate={duplicate} "
                           f"unexpected={unexpected[:8]}")
    return {"trainable_tensors": len(trainable), "covered_once": len(seen),
            "main_groups": len(opt.param_groups),
            "gate_groups": 0 if opt_gate is None else len(opt_gate.param_groups)}


OOM_EXIT = 42
CYCLE_EXIT = 43
TOTAL_STEPS = 16000
STAGE_ENDS = {4000, 7000, 14000, 15000, 16000}
SWAP_STEP = 4000
MIN_FREE_GB_DIT = 9.0       # 4.05B bf16 ≈ 8.1GB + 余量；低于此不重载
MIN_FREE_GB_TEACHER = 16.0  # 9.4GB 教师 + 6.3GB UMT5-int8 + 激活余量；低于此不加载
StepSpec = namedtuple("StepSpec", "step stage mode ds lr")

_save_threads = []


def sinusoidal_timestep_embedding(t, dim, max_period=10000.0, t_scale=1000.0):
    """由真实噪声强度 t 生成正弦时间步嵌入 [B,dim]（阶段1修复核心）。

    t_scale=1000（2026-11-23 实测修复）：Wan2.2 教师走 ComfyUI 的
    `ModelSamplingDiscreteFlow.timestep(sigma)=sigma*1000`，
    `sinusoidal_embedding_1d` 直接吃这个 0~1000 的值。学生原先喂 t∈[0,1]，
    实测教师 time_embedding.0 输出的 cos(t=0.1,t=0.9)=0.9997（几乎不可分），
    换成 0~1000 后 cos=0.153 —— 即原先是把时间条件喂成了常数。
    """
    half = dim // 2
    dev = t.device
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=dev, dtype=torch.float32) / half)
    ang = (t.float() * t_scale)[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _cosine(a, b, x):
    x = min(1.0, max(0.0, x))
    return b + 0.5 * (a - b) * (1.0 + math.cos(math.pi * x))


def _linear(a, b, x):
    x = min(1.0, max(0.0, x))
    return a + (b - a) * x


class UpdateWatch:
    """判断「权重到底动没动」的硬指标（2026-11-23 新增）。

    200 步短测实测：596 个张量里 494 个相对变化 <1e-4（中位数精确为 0），
    `patchify.weight` 150 步累计位移只有 0.67 个 bf16 量子 —— 训练实际空转。
    这里固定跟踪两个张量，把位移换算成「bf16 量子」记进日志，作为每次短测的验收列。
    """
    NAMES = ("patchify.weight", "head.head.weight",
             "full_head.final_proj.weight", "full_head.pre_proj_scale",
             "full_head.alpha", "full_head.out_gain")

    def __init__(self, model):
        self.ref = {}
        params = dict(model.named_parameters())
        for name in self.NAMES:
            p = params.get(name)
            if p is not None:
                self.ref[name] = p.detach().flatten()[: 1 << 20].clone()

    def report(self, model):
        params = dict(model.named_parameters())
        parts = []
        for name, ref in self.ref.items():
            p = params.get(name)
            if p is None:
                continue
            cur = p.detach().flatten()[: ref.numel()].float()
            rms = float(ref.float().pow(2).mean().sqrt())
            cur_abs = float(cur.abs().mean())
            # 2026-11-26 修正：bf16 是 1+7 尾数，量子 = 2^(floor(log2|w|)-7)；
            # 旧写法 max(rms,|cur|mean)*2^-8 把量子估小了约 1/3（0.0412 处真实量子
            # 2^-12=2.44e-4，旧公式给 1.6e-4），于是"每步几个量子"被高报 ~1.5 倍。
            # 用张量量级（rms）处的量子做标尺：逐元素求 q 会被大量小权重拉低。
            _lvl = max(rms, 1e-12)
            quantum = float(2.0 ** (math.floor(math.log2(_lvl)) - 7))
            dmax = float((cur - ref.float()).abs().max())
            label = name.split('.')[0]
            if name.startswith("full_head."):
                leaf = name.split('.')[-1]
                if leaf in ("alpha", "out_gain", "pre_proj_scale"):
                    label = leaf
                elif leaf == "weight" and ".final_proj." in name:
                    label = "final_proj.w"
                else:
                    label = "full_head.other"
            parts.append(f"{label}={dmax:.2e}({dmax / quantum:.1f}q)")
        return " ".join(parts)


def latent_dataset_stats(files, n=40, seed=0):
    """读前 n 条 latent 报统计量。零预测基线 = sigma_noise^2 + E[x0^2]，noise 标准正态时
    = 1 + E[x0^2]；不归一化到方差 1（那会把零预测推到 2.0，反而更差）。"""
    import random as _random
    pick = files if len(files) <= n else _random.Random(seed).sample(files, n)
    e2 = m = s = 0.0
    cnt = 0
    for f in pick:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        x = rec.get("latent") if isinstance(rec, dict) else rec
        if x is None:
            continue
        x = x.float()
        e2 += float(x.pow(2).mean()); m += float(x.mean()); s += float(x.std())
        cnt += 1
    if not cnt:
        return None
    e2, m, s = e2 / cnt, m / cnt, s / cnt
    return dict(n=cnt, mean=m, std=s, e2=e2, zero=1.0 + e2)


def build_schedule(until_step=TOTAL_STEPS, p1_until=4000, p2_until=7000):
    """按全局步号生成逐步规格。构建纯 CPU，不碰 GPU。
    p1_until: 把 P1 自监督从 4000 延长到该步（用于 3600→4200 的时序混训实验）。
    p2_until: 把 P2 纯蒸馏窗口从 7000 延长到该步（2026-11-26 新增，用于 6808 起继续纯教师蒸馏，
              避免越过 7001 自动切到 P3 自监督/需 ds3 而中断）。"""
    p1_until = int(p1_until)
    if p1_until > 0:
        p1_until = max(4000, p1_until)
    p2_until = max(p1_until + 1, int(p2_until))
    sch = []
    # P1: 1..p1_until，前 400 步线性 warmup 到 2e-4，随后 cosine 到 1.5e-4（4000 后保持）
    for s in range(1, p1_until + 1):
        if s <= 400:
            lr = 2e-4 * s / 400
        elif s <= 4000:
            lr = _cosine(2e-4, 1.5e-4, (s - 400) / 3600.0)
        else:
            lr = 1.5e-4
        sch.append(StepSpec(s, "P1-self", "self", "ds1", lr))
    # P2: p1_until+1..p2_until，纯离线蒸馏
    for s in range(p1_until + 1, p2_until + 1):
        denom = max(1.0, float(p2_until - (p1_until + 1)))
        sch.append(StepSpec(s, "P2-distill", "distill", "ds2",
                            _cosine(1.5e-4, 1.2e-4, (s - (p1_until + 1)) / denom)))
    # P3: p2_until+1..14000，28 轮 x (200 自监督 + 50 蒸馏回炉)
    for cyc in range(28):
        base = p2_until + 1 + cyc * 250
        self_lr = _linear(1.2e-4, 1.0e-4, cyc / 27.0)
        distill_lr = _linear(4e-5, 3.5e-5, cyc / 27.0)
        for k in range(200):
            sch.append(StepSpec(base + k, "P3-interleave-self", "self", "ds3", self_lr))
        for k in range(200, 250):
            sch.append(StepSpec(base + k, "P3-interleave-distill", "distill", "ds2", distill_lr))
    # P4: 14001..15000，通用高质量收尾
    for s in range(14001, 15001):
        sch.append(StepSpec(s, "P4-polish", "self", "ds3", _linear(6e-5, 4e-5, (s - 14001) / 999.0)))
    # P5: 15001..16000，二次元专项微调（最低 lr + 更大正则）
    for s in range(15001, 16001):
        sch.append(StepSpec(s, "P5-anime", "self", "ds4", _linear(2e-5, 1e-5, (s - 15001) / 999.0)))
    return [x for x in sch if x.step <= until_step]


def print_schedule(sch, args):
    print("[dry-run] 阶段总览（纯 CPU，未初始化 CUDA）")
    groups = []
    for spec in sch:
        stage = spec.stage
        if stage.startswith("P3-interleave-"):
            stage = "P3-" + stage.rsplit("-", 1)[1]  # 28 轮合并显示
        if not groups or groups[-1][0] != stage:
            groups.append([stage, spec.step, spec.step, spec.mode, spec.ds, spec.lr, spec.lr])
        else:
            g = groups[-1]
            g[2] = spec.step
            g[5] = min(g[5], spec.lr)
            g[6] = max(g[6], spec.lr)
    for stage, a, b, mode, ds, lo, hi in groups:
        print(f"  {stage:26s} step {a:5d}..{b:5d}  mode={mode:8s} ds={ds:3s} lr {lo:.2e}..{hi:.2e}")
    print(f"  共 {len(sch)} 步，终点 step {sch[-1].step if sch else 0}")
    for key, path in (("ds1", args.ds1), ("ds2", args.ds2), ("ds3", args.ds3), ("ds4", args.ds4)):
        used = any(x.ds == key for x in sch)
        n = len(list_pt(path)) if path else 0
        flag = "需准备" if used and n == 0 else ("OK" if n else "未使用")
        print(f"  {key}: {path or '(未指定)'}  文件数={n}  [{flag}]")


def validate_data(sch, args):
    """涉及到的阶段数据集必须存在且非空；不静默退化。"""
    cache = {}
    for key in ("ds1", "ds2", "ds3", "ds4"):
        path = getattr(args, key)
        if any(x.ds == key for x in sch):
            _startup_trace(f"validate {key} begin path={path}")
            if not path:
                raise SystemExit(f"[data] 阶段需要 {key}，但 --{key} 未指定")
            files = list_pt(path)
            _startup_trace(f"validate {key} list_pt {len(files)}")
            if not files:
                if key == "ds2" and getattr(args, "p2_autogen", False):
                    cache[key] = []
                    print(f"[data] ds2 暂空：--p2-autogen 会在 step{SWAP_STEP} 现场生成缓存", flush=True)
                    continue
                raise SystemExit(f"[data] 阶段需要 {key}，但目录为空或无 .pt: {path}")
            if key == "ds2" and getattr(args, "data_gate_report", ""):
                files = apply_data_gate(
                    files, args.data_gate_report,
                    strict=bool(getattr(args, "data_gate_strict", False)),
                    seed=getattr(args, "shuffle_seed", 0),
                )
            cache[key] = files
            print(f"[data] {key}={len(files)} 条  {path}", flush=True)
            if key == "ds1":
                base = list_pt(args.ds1)
                extra = list_pt(args.ds1_extra) if args.ds1_extra else []
                _startup_trace(f"validate ds1 base={len(base)} extra={len(extra)}")
                # 用户定稿：ds1:extra 按 --extra-ratio 确定性混采（例如 9f:13f=1:3）
                # source 模式按源视频配平：700 个 9f/13f 重叠源不再双喂，
                # 1283 个 13f 独有源不再相对饿死。
                a_per, b_per = _parse_ratio(args.extra_ratio)
                if extra and getattr(args, "sample_balance", "ratio") == "source":
                    files = SourceBalancedFiles(
                        base, extra,
                        nine_slots=getattr(args, "source_nine_slots", 3),
                        total_slots=4,
                        shuffle_seed=args.shuffle_seed,
                    )
                    print(f"[data] ds1 source-balanced: 9f源={len(files.nine)} "
                          f"13f源={len(files.thirteen)} "
                          f"9f槽={getattr(args,'source_nine_slots',3)}/4", flush=True)
                _ds1_report = getattr(args, "data_gate_report_ds1", "") or ""
                if _ds1_report:
                    # 2026-11-26 修复：门禁此前只挂 ds2，ds1 裸走 MixedFiles 混池。
                    # 接 ds1 专用分组报告（与 ds2 同构；--pair-cache 继承 σ/场景）。
                    # 注意：ds1 无 v_teacher，分组只做质量+场景(σ参考)过滤轮转；
                    # 与 sample_balance=source 互斥，同时给时 gate 优先（只过滤 base）。
                    files = apply_data_gate(
                        base, _ds1_report,
                        strict=bool(getattr(args, "data_gate_strict", False)),
                        seed=getattr(args, "shuffle_seed", 0))
                    print(f"[data] ds1 gate <- {_ds1_report} kept={len(files)}", flush=True)
                else:
                    files = MixedFiles(base, extra, a_per, b_per) if extra else base
                _startup_trace(f"validate ds1 sampler len={len(files)}")
                if args.holdout > 0:
                    files, hold = _split_holdout(files, args.holdout)
                    cache["holdout"] = hold
                    print(f"[data] ds1 holdout={len(hold)} 条（按源视频哈希，ratio={args.holdout}）", flush=True)
                if args.shuffle_seed:
                    files = ShuffledFiles(files, args.shuffle_seed)
                    print(f"[data] ds1 shuffle seed={args.shuffle_seed}", flush=True)
                cache[key] = files
                uniq = len(base) + len(extra)
                sched_name = ("SourceBalancedFiles（每源4槽）"
                              if getattr(args, "sample_balance", "ratio") == "source"
                              else "MixedFiles 周期展开")
                print(f"[data] ds1 unique={uniq} 条（9f={len(base)} + extra={len(extra)}）；"
                      f"schedule_len={len(files)}（{sched_name}，不是唯一池子）", flush=True)
    return cache


_TRAIN_STATE = {}
_TRAIN_CONFIG = {}
# 2026-11-25：夜间节点续训暴露出 ckpt 只存权重、不存 Adafactor/AdamW 状态，
# 每次 resume 动量全部清零。这里在写盘时同步抓取 CPU 快照。
_SAVE_OPTIM = {}
# 2026-11-27：EMA 权重（CPU bf16，避免 GPU 20GB 被 6.2GB EMA 挤爆）。
_EMA_STATE = {"params": None, "last_step": 0}


def _ema_init(model, decay):
    import torch as _t
    params = {n: p.detach().to("cpu", copy=True).to(_t.bfloat16)
              for n, p in model.named_parameters()}
    _EMA_STATE["params"] = params
    _EMA_STATE["last_step"] = 0
    _EMA_STATE["decay"] = float(decay)
    n = sum(v.numel() for v in params.values())
    print(f"[ema] init cpu-bf16 params={len(params)} tensors={n/1e9:.3f}B "
          f"decay={decay}", flush=True)


def _ema_update(model, step, decay, every):
    import torch as _t
    if _EMA_STATE["params"] is None:
        return
    last = int(_EMA_STATE["last_step"])
    if step - last < every:
        return
    k = step - last
    d = float(decay) ** k
    with _t.no_grad():
        for n, p in model.named_parameters():
            e = _EMA_STATE["params"].get(n)
            if e is None:
                # v3.2 resume 时新模块（text_cross/sigma_adapter/context/repa）
                # 在旧 ckpt 的 ema_state 里没有 key。旧逻辑 continue 会让它们
                # 永远不进 EMA；这里用当前权重补种，下一步起正常滑动。
                _EMA_STATE["params"][n] = p.detach().to("cpu", copy=True).to(_t.bfloat16)
                continue
            e.mul_(d).add_(p.detach().to("cpu").to(_t.bfloat16), alpha=1.0 - d)
    _EMA_STATE["last_step"] = step



_REPA_WARNED = set()
_AUX_WARNED = set()


def _repa_target_path(repa_dir, fpath):
    """按与 text_tokens 相同的 basename 约定拼 REPA 教师特征缓存路径。"""
    if not repa_dir or not fpath:
        return None
    return os.path.join(str(repa_dir), os.path.basename(str(fpath)))


def _load_repa_target(path, device, dim):
    """读离线 DINOv2 教师特征缓存，返回 [F,P,D] fp32 张量（无梯度，调用侧再 detach）。

    缓存契约（沿用 model/<cache>/<6位序号>.pt 约定）：dict 含 ``feature`` 张量，
    形状 [F,P,D] / [B,F,P,D] / [P,D]；D 必须等于 repa_proj 输出维。教师侧不进反传。
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feat = obj["feature"] if isinstance(obj, dict) and "feature" in obj else obj
    if not torch.is_tensor(feat):
        raise TypeError(f"repa cache {path}: feature 非张量 {type(feat)}")
    feat = feat.to(torch.float32)
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
    if feat.ndim == 4:  # [B,F,P,D] -> 取第 0 样本（distill B=1 路径）
        feat = feat[0]
    if feat.ndim != 3:
        raise ValueError(f"repa cache {path}: feature 维数 {feat.ndim} != 3")
    if int(feat.shape[-1]) != int(dim):
        raise ValueError(
            f"repa cache {path}: dim {int(feat.shape[-1])} != repa_dim {int(dim)}")
    return feat.to(device=device)


_PSD_EDGES = (0.0, 0.08, 0.16, 0.24, 0.32, 0.64, 1.0, 1.5)
_PSD_CACHE = {}


def _psd_masks(h, w, device):
    key = (h, w, str(device))
    if key in _PSD_CACHE:
        return _PSD_CACHE[key]
    fy = torch.fft.fftfreq(h, d=1.0, device=device).abs() * 2.0
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).abs() * 2.0
    r = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
    masks = []
    for lo, hi in zip(_PSD_EDGES[:-1], _PSD_EDGES[1:]):
        masks.append((r >= lo) & (r < hi))
    _PSD_CACHE[key] = masks
    return masks


def radial_psd_loss(pred, target):
    """按归一化径向频带匹配 pred/target 的功率占比（不含 DC）。

    不做绝对能量匹配，只匹配频谱形状；低频/中频/高频各带同权 MSE。
    pred/target: [B,C,T,H,W]，float32。
    """
    p = torch.fft.rfft2(pred.float(), dim=(-2, -1)).abs().pow(2)
    t = torch.fft.rfft2(target.float().detach(), dim=(-2, -1)).abs().pow(2)
    p = p.mean(dim=(0, 1, 2))
    t = t.mean(dim=(0, 1, 2))
    p[0, 0] = 0.0
    t[0, 0] = 0.0
    masks = _psd_masks(p.shape[0], pred.shape[-1], p.device)
    total_p = p.sum() + 1e-12
    total_t = t.sum() + 1e-12
    fp = torch.stack([p[m].sum() / total_p for m in masks])
    ft = torch.stack([t[m].sum() / total_t for m in masks])
    return torch.nn.functional.mse_loss(fp, ft.detach())


def spectrum_shape_loss(pred, target, high_lo: float = 0.25,
                        high_hi: float = 0.5, slope_weight: float = 0.5):
    """匹配 0.25-0.5 高频能量占比 + 径向 log-log 斜率。

    pred/target: [B,C,T,H,W] float32。只匹配形状，不做绝对幅值压制。
    """
    p = torch.fft.rfft2(pred.float(), dim=(-2, -1)).abs().pow(2)
    t = torch.fft.rfft2(target.float().detach(), dim=(-2, -1)).abs().pow(2)
    p = p.mean(dim=(0, 1, 2))
    t = t.mean(dim=(0, 1, 2))
    p[0, 0] = 0.0
    t[0, 0] = 0.0
    h, w = pred.shape[-2], pred.shape[-1]
    fy = torch.fft.fftfreq(h, d=1.0, device=p.device).abs() * 2.0
    fx = torch.fft.rfftfreq(w, d=1.0, device=p.device).abs() * 2.0
    r = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
    hmask = (r >= high_lo) & (r < high_hi)
    total_p = p.sum() + 1e-12
    total_t = t.sum() + 1e-12
    hp = p[hmask].sum() / total_p
    ht = t[hmask].sum() / total_t
    high_loss = (hp - ht).abs() / (ht.abs() + 1e-3)
    # 8 个等宽径向环的 log-log 斜率，约束整体频谱衰减。
    edges = torch.linspace(0.04, 0.70, 9, device=p.device)
    sp, st = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.any():
            sp.append(p[m].mean())
            st.append(t[m].mean())
    if len(sp) >= 3:
        ep = torch.stack(sp).clamp_min(1e-12).log()
        et = torch.stack(st).clamp_min(1e-12).log()
        x = torch.linspace(-1.0, 1.0, len(sp), device=p.device)
        xm = x - x.mean()
        slope_p = (ep * xm).sum() / (xm.pow(2).sum() + 1e-12)
        slope_t = (et * xm).sum() / (xm.pow(2).sum() + 1e-12)
        slope_loss = (slope_p - slope_t).abs()
    else:
        slope_loss = p.new_zeros(())
    return high_loss + slope_weight * slope_loss


def _optimizer_state_cpu(opt) -> dict:
    """把 optimizer.state_dict() 中的张量同步拷到 CPU，再交给异步写盘线程。"""
    if opt is None:
        return {}
    import torch as _t

    def _cpu(v):
        if _t.is_tensor(v):
            return v.detach().cpu().clone()
        if isinstance(v, dict):
            return {k: _cpu(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_cpu(x) for x in v)
        return v

    return _cpu(opt.state_dict())


def _optimizer_param_names(model, optimizer) -> list[list[str]]:
    """把优化器 param_groups 里的参数映射回模型参数名。

    2026-11-25：Adafactor/AdamW 的 load_state_dict 是按 param_groups 的
    参数顺序映射 state 的；任何新增/删除/重排模块都会让 state 静默错位
    （rpe_table 拿到 1D 的 variance 状态，第一次 step 就 assert）。
    这里把名字顺序写进 ckpt，resume 时逐组比对，不一致直接拒绝。
    """
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    return [[name_by_id.get(id(p), "<non-model>") for p in g["params"]]
            for g in optimizer.param_groups]


def save_ckpt(model, step, loss, out_dir, filename):
    """复用 train_backbone 的原子写盘逻辑；串行化 CPU 拷贝，避免内存翻倍。

    2026-11-23：把 sample_cursor / running / best_loss / n_done / zero_running
    一起写进 ckpt，resume 时不再把 avg/best 和样本进度重置。
    """
    # 2026-11-25: --no-ckpt 此前只存在于 argparse、没有任何调用点，实际仍会写
    # 7.4GB 探针。现在在唯一写盘入口统一短路，保证 smoke 真的不落盘。
    if _TRAIN_CONFIG.get("no_ckpt"):
        print(f"[ckpt] --no-ckpt: skip save step{step} {filename}", flush=True)
        return
    for th in _save_threads:
        if th.is_alive():
            th.join()
    _save_threads.clear()
    os.makedirs(out_dir, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ema_payload = None
    if _EMA_STATE.get("params") is not None:
        _final_only = bool(_TRAIN_CONFIG.get("ema_save_final_only", True))
        if (not _final_only) or int(step) == int(_TRAIN_CONFIG.get("until_step", -1)):
            ema_payload = {k: v.clone() for k, v in _EMA_STATE["params"].items()}
    payload = {"model": state_cpu, "step": step, "cfg": str(model.cfg),
               "loss": float(loss), "train_state": dict(_TRAIN_STATE),
               "train_config": dict(_TRAIN_CONFIG),
               "optimizer_state": {k: _optimizer_state_cpu(v)
                                   for k, v in _SAVE_OPTIM.items()},
               "optimizer_param_names": {k: _optimizer_param_names(model, v)
                                         for k, v in _SAVE_OPTIM.items() if v is not None},
               "model_param_names": [n for n, _ in model.named_parameters()],
               "full_head_disable": list(getattr(model.cfg, "full_head_disable", ()) or ())}
    if ema_payload is not None:
        payload["ema_state"] = ema_payload
        payload["ema_step"] = int(_EMA_STATE.get("last_step", 0))
        payload["ema_decay"] = float(_EMA_STATE.get("decay", 0.0))
    path = os.path.join(out_dir, filename)
    # 轻量 sidecar：resume 在构建 4B 模型之前就能读到结构开关，避免
    # “ckpt 里 disable 了 patch_norm，手动续训却没传 flag”而静默换图。
    cfg_path = path + ".config.json"
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"step": int(step),
                        "full_head_disable": list(getattr(model.cfg, "full_head_disable", ()) or ()),
                        "model": str(getattr(model.cfg, "name", ""))}, f,
                       ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[save][warn] sidecar 写失败 {cfg_path}: {e}", flush=True)
    th = threading.Thread(target=_atomic_save, args=(payload, path), daemon=False)
    th.start()
    _save_threads.append(th)
    print(f"[save] {path} step={step} loss={loss:.5f} (async, serialized)", flush=True)


def prune_old(args):
    """仅当 --prune 显式开启才删除旧 checkpoint；阶段末版本在独立目录，不受影响。"""
    pat = os.path.join(args.out, "backbone_step*.pt")
    files = sorted(glob.glob(pat), key=lambda p: int(os.path.basename(p)[13:-3]))
    if len(files) <= args.keep_recent:
        return
    victims = files[:len(files) - args.keep_recent]
    print(f"[prune] 保留最近 {args.keep_recent} 个；待删除 {len(victims)} 个（阶段末版本不受影响）", flush=True)
    for p in victims:
        try:
            os.remove(p)
            print(f"[prune] 已删除 {p}", flush=True)
        except OSError as e:
            print(f"[prune] 删除失败 {p}: {e}", flush=True)


def sample_hook(spec, step, args):
    if step not in STAGE_ENDS:
        return
    print(f"[sample] 阶段末 step={step} ({spec.stage})，固定种子采样检查时序/服从度/崩坏", flush=True)
    if args.sample_cmd:
        env = dict(os.environ)
        env["SAMPLE_STEP"] = str(step)
        env["SAMPLE_STAGE"] = spec.stage
        try:
            subprocess.run(args.sample_cmd, shell=True, env=env, check=False)
        except Exception as e:
            print(f"[sample] 采样命令执行失败: {e}", flush=True)
    else:
        print("[sample] 未配置 --sample-cmd，请人工采样后再进入下一阶段", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="PyraGate 阶段感知训练器（16000 步）")
    ap.add_argument("--model", choices=["tiny", "4b5", "4b5fs", "4b5v3"], default="4b5")
    ap.add_argument("--full-gated-head", action=argparse.BooleanOptionalAction, default=None,
                    help="启用 FullGatedHead-RPE-Sobel-AA-Pyramid 新输出头；旧LinearHead权重不参与前向。"
                         "三态：不传=按模型工厂默认")
    ap.add_argument("--full-head-groups", type=int, default=4)
    ap.add_argument("--full-head-compress", type=int, default=4)
    ap.add_argument("--full-head-ctx-dim", type=int, default=128)
    ap.add_argument("--full-head-disable", nargs="*", default=[],
                    choices=["cross", "group_mask", "time_mod", "rpe", "sobel",
                             "antialias", "pyramid", "frame_aux", "head_scale", "vmag",
                             "pre_proj_scale", "final_weight_clamp",
                             "dualact_norm", "dualact_gate", "patch_norm",
                             "bandnorm_dc", "bandnorm_res"],
                    help="消融开关：关闭指定子模块，其余保持开启")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS, help="上限步号，默认 16000")
    ap.add_argument("--until-step", type=int, default=TOTAL_STEPS,
                    help="本次跑到第几步停（例如先只跑 4000）")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--mem-frac", type=float, default=0.93)
    ap.add_argument("--no-ckpt", action="store_true",
                    help="不写 checkpoint 文件；梯度检查点由 --no-grad-ckpt 单独控制")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="关闭梯度检查点（用显存换速度；默认开启，适配 21GB 显卡）")
    ap.add_argument("--ckpt-last-dec-off", type=int, default=0,
                    help="最后 N 个 decoder block 不做梯度检查点（显存换速度，默认 0）")
    ap.add_argument("--ckpt-skip-bottleneck", action="store_true",
                    help="bottleneck 6 层不做梯度检查点（显存换速度）")
    ap.add_argument("--out", default="ckpt_resume500")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help=">0 启用 CPU bf16 EMA；200 步续训建议 0.99，长程 0.999")
    ap.add_argument("--ema-every", type=int, default=5,
                    help="每 N 个 optimizer step 更新一次 EMA，减少 6GB PCIe 拷贝")
    ap.add_argument("--ema-save-final-only", action=argparse.BooleanOptionalAction,
                    default=True, help="只在 --until-step 的 ckpt 里写 EMA 权重")
    ap.add_argument("--resume", default="", help="从 checkpoint 续训（恢复 step 号）")
    ap.add_argument("--warmstart", default="", help="M85 warmstart（仅从零开始时有意义）")
    ap.add_argument("--warmstart-mode", choices=["full", "modulation"], default="modulation",
                    help="full=原 42 项（会覆盖 patchify/head/t_embed/cond_proj）；"
                         "modulation=只注入教师 AdaLN 表 + ada_embedder.3（默认，保留学生主头方向）")
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--freeze-spectral-after", type=int, default=100)
    ap.add_argument("--cycle-steps", type=int, default=0, help=">0 时跑满主动 exit(43)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="Adafactor 的 weight_decay；P5 阶段临时用 1.5e-2")
    ap.add_argument("--gate-lr", type=float, default=2.5e-4,
                    help="门控参数独立 AdamW 的 lr；门控 fp32，绕开 Adafactor eps2。"
                         "默认 2.5e-4（与 run_auto_segment.ps1 GateLR=0.00025 对齐）；"
                         "5e-4 实测每步门控+5e-4，600 步会把 local gate 从 0.02 推到 0.32；"
                         "1e-4 更适合「已有 0.02~0.1 初值」的续训。")
    ap.add_argument("--gate-wd", type=float, default=0.0,
                    help="门控参数 weight decay；默认 0，避免把刚打开的门又拉回 0")
    ap.add_argument("--head-lr", type=float, default=2e-3,
                    help="full_gated_head 独立 lr。头参数 bf16 且初始权重 RMS≈2e-4，"
                         "主 lr 1e-4 时 Adafactor 步长落在 bf16 量子以下，头等于冻死；"
                         "2e-3 让 final/pyramid/SSM 支路在 400 步内真正长起来。")
    ap.add_argument("--head-final-lr", type=float, default=0.0,
                    help="final_proj.weight + pre_proj_scale 的独立 AdamW lr；默认 0=冻结，"
                         "避免误用 1e-4 把输出打成高频噪斑。已实测 1e-4 在 20 步内崩坏；"
                         "安全探针从 1e-5 或 3e-5 起。")
    ap.add_argument("--head-wd", type=float, default=1e-2,
                    help="full_gated_head weight decay；方案要求 ≤0.01")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="阶段1热启动：冻结 full_head 之外的全部主干参数，仅训输出头")
    ap.add_argument("--train-mod-only", action="store_true",
                    help="阶段1更严格：只训内嵌 Dual-Act 新增模块"
                         "（band_norm/dualact_gate/patch_norm），其余全冻")
    ap.add_argument("--head-alpha-min", type=float, default=0.08,
                    help="head.alpha 下限；alpha 是 fused 残差缩放，掉到 0.004 等于"
                         "SSM/金字塔/Sobel 全断，每步 optimizer.step 后强制 clamp_min")
    ap.add_argument("--reset-high-out", action="store_true",
                    help="续训前把 full_head.high_out 清零：旧高流是固定 2x2 高通，"
                         "σ=1 时可绕过 final_proj 零初始化直接复制输入高频；清零后"
                         "高流从零残差开始学，不再短路主干。")
    ap.add_argument("--lr-scale", type=float, default=20.0,
                    help="调度表 lr 的整体放大系数。原因：本机 Adafactor 的更新是 "
                         "α=max(eps2,RMS(θ))*lr，等价于「lr×权重RMS」——权重 RMS≈0.02 时"
                         "实际步长只有标称 lr 的 1/50。放大 20× 后有效步长≈标称 lr 的 0.4×，"
                         "仍在 Adam 常规量级之内。")
    ap.add_argument("--opt-eps1", type=float, default=1e-12,
                    help="Adafactor 分母下限 eps1。**不能设 1e-30**：这版实现会把 eps1 平方"
                         "（clamp(min=eps1²)）再 rsqrt，1e-60 在 fp32 下直接下溢成 0，"
                         "rsqrt(0)=inf，inf×grad(0)=NaN —— 实测一步就把 patchify 打成 NaN。"
                         "1e-12 平方后 1e-24 仍可表示，且对正常梯度不生效。")
    ap.add_argument("--opt-eps2", type=float, default=1e-3,
                    help="Adafactor **参数尺度下限** eps2：本机这版实现的更新是 "
                         "α=max(eps2, RMS(θ))*min(lr,1/sqrt(t))，再乘归一化更新。"
                         "**实测**：eps2=1e-30 时零初始化参数的 α=1e-30*lr≈5e-37，"
                         "head 梯度 0.0115 却只挪 7.35e-37 —— 所有零初始化门控/主头"
                         "永久冻结（200 步「权重不动」的真正主因）。"
                         "eps2=1.0 太热（每元素步长≈lr，门控一步推满）；"
                         "取 1e-3（Adafactor 默认）：零参数 α=1e-3*lr 可起步，"
                         "正常参数 α≈RMS*lr≈0.02*lr，再靠 --lr-scale 补偿。")
    ap.add_argument("--opt-var-floor", type=float, default=0.0,
                    help="Adafactor row_var/col_var 绝对下限。默认 0=关闭。"
                         "⚠ 1e-8 会反向压更新：factorized v≈row_mean，sqrt(v)≈1e-4，"
                         "更新被压到 1e-8 级，FFN 200 步只有 1.8e-7 相对位移；"
                         "0 时正常二阶矩≈grad²≈1e-14，sqrt(v)≈1e-7，FFN 200 步位移 1.51%%。"
                         "宁可 0 + weight-health 硬中止，也不要用绝对 floor 把主干锁死。")
    ap.add_argument("--weight-health-max", type=float, default=100.0,
                    help="每 N 步扫描权重绝对值；超过该值立即中止，避免 1e23 坏 ckpt 落盘。0=关闭。")
    ap.add_argument("--weight-check-every", type=int, default=50,
                    help="权重健康扫描间隔（步）；<=0 关闭。")

    ap.add_argument("--opt-foreach", action=argparse.BooleanOptionalAction, default=False,
                    help="Adafactor foreach 批量算子。默认关闭：本机 21GB 实测全量 foreach "
                         "在 accum1/2 均 OOM；chunks4/8 虽能跑但 4.3-5.2s/step，慢于 for-loop "
                         "3.2-3.4s。保留开关仅供实验。")
    ap.add_argument("--opt-foreach-chunks", type=int, default=4,
                    help="foreach 时按 numel 把主参数切成 N 个 param group，逐组 step；"
                         "21GB 卡单组全量 foreach 会 OOM，分块可压低临时峰值。")
    # 2026-11-24：torch 2.9.1+rocm7.2.1 的 Adafactor 构造函数没有 factorized 参数；
    # 源码 _adafactor.py 内部固定：dim>1 走 row/col 因子分解，dim==1 走标量二阶矩。
    # 这不是可开关项，`--opt-factorized` 不存在，避免误以为能关闭。
    ap.add_argument("--cond-mode", choices=["fixed", "random"], default="fixed",
                    help="fixed=真实t正弦嵌入+真文本（若给 text-emb-dir）/零文本+显式sigma（修复版）；"
                         "random=旧随机高斯（复现bug用）")
    ap.add_argument("--text-emb-dir", default="",
                    help="离线 UMT5 真文本缓存目录（与 ds2 teacher 文件同名 .pt）；"
                         "给了就严格读取，缺失/全零直接失败，不再回退 zeros")
    # ---- 噪声档采样（2026-11-23 对齐教师）----
    # 教师 ComfyUI 侧 ModelSamplingDiscreteFlow：sigma = shift·u/(1+(shift-1)·u)，
    # u 为线性步进比例，shift 由 WAN21_T2V/WAN22_T2V 默认 8.0（comfy/supported_models.py:1337）。
    # 故「教师非均匀」= u~U(0,1) 映射后 sigma 向高噪端集中（均值 0.80，而均匀采样是 0.50）。
    # 目标 v=ε−x0 与 sigma 无关，故零预测基线（1+E[x0²]）不变，验收线仍可比。
    ap.add_argument("--t-sched", choices=["teacher", "uniform"], default="teacher",
                    help="teacher=按教师 shift 映射采样 sigma；uniform=旧均匀采样（对照组）")
    ap.add_argument("--t-shift", type=float, default=8.0,
                    help="教师 flow shift（Wan2.1/2.2 T2V 默认 8.0）")
    ap.add_argument("--t-sigma-min", type=float, default=0.0,
                    help="sigma 采样下限；>0 时可做高噪区间专属训练（如 0.5）")
    ap.add_argument("--t-sigma-max", type=float, default=1.0,
                    help="sigma 采样上限；<1 时可做低噪区间专属训练")
    ap.add_argument("--t-high-ratio", type=float, default=0.0,
                    help="混合采样中高噪段占比（0=关闭；0.8=80%% 样本取 [0.5,1]，"
                         "其余取 [0,0.5)）")
    ap.add_argument("--patch-mean-loss", type=float, default=0.0,
                    help="额外对 x0hat 与 x0 的 2x2 patch 均值做 MSE 的权重（>0 启用；"
                         "直接优化 32px 方块对应的 token 均值，不依赖低通平滑）")
    ap.add_argument("--anti-collapse", type=float, default=0.0,
                    help="非对称防偏好/防方块权重：只惩罚 x0hat 的 2x2 patch 内细节方差"
                         "低于目标（欠细节），不惩罚高于目标；0=关闭")
    ap.add_argument("--detail-aux", type=float, default=0.0,
                    help="细节头独立高频监督权重：让 -sigma*dv 直接拟合 x0 的 2x2 patch "
                         "高频残差；0=关闭（只在 anti 长期偏高时启用）")
    ap.add_argument("--frame-aux", type=float, default=0.05,
                    help="FullGatedHead 训练帧间平滑 L1 权重；推理无开销")
    ap.add_argument("--token-bd-loss", type=float, default=0.02,
                    help="Token-Boundary 梯度连续性损失权重：只约束低频通道的"
                         "相邻差分连续；高频通道允许边缘跳变")
    ap.add_argument("--corr-loss", type=float, default=0.05,
                    help="相邻 patch-mean 余弦缺陷(1-corr)的单边损失权重；随机20%%位置采样")
    ap.add_argument("--corr-target-min", type=float, default=0.0,
                    help="已废弃的双向 band 下界，必须为 0；非 0 会拒绝训练，防止反向降低相关性")
    ap.add_argument("--corr-target-max", type=float, default=0.10,
                    help="corr_defect=1-Pearson(相邻 patch mean) 的软上限；"
                         "真实 holdout 参考 corr≈0.904 → defect≈0.096，默认 0.10，"
                         "只有高于它才惩罚")
    ap.add_argument("--token-bd-floor", type=float, default=0.30,
                    help="tbd 软下限；低于它时加小权重把它托回 0.3 左右，"
                         "防止 corr 引导把二阶差分一起压平")
    ap.add_argument("--token-bd-floor-loss", type=float, default=0.01,
                    help="tbd 下限 hinge 权重；只托底，不替代主 tbd 项")
    ap.add_argument("--vmag-loss", type=float, default=0.02,
                    help="D：sigma 感知 RMS 门控权重（1801-2000 由 --rms-warmup-weight 覆盖）；"
                         "推理关闭、不新增可学习参数")
    ap.add_argument("--rms-warmup-weight", type=float, default=0.07,
                    help="D 预热段 RMS 门控权重（默认 0.07）")
    ap.add_argument("--rms-warmup-until", type=int, default=2000,
                    help="RMS 预热权重的截止步（含），之后回落到 --vmag-loss")
    ap.add_argument("--corr-warmup-weight", type=float, default=0.03,
                    help="C 预热段 corr band 基础权重（默认 0.03），之后恢复 --corr-loss")
    ap.add_argument("--corr-amp-gate", type=float, default=3.0,
                    help="corr 与 RMS 联动增益：w=base*sigmoid(g*(rms-target))；"
                         "幅值低于目标时自动给 corr 让路；0=固定权重")
    ap.add_argument("--std-loss", type=float, default=0.0,
                    help="全局 latent std 单边 L2 托底权重：只罚 std 低于目标 0.95 倍的部分，"
                         "防止 corr 把画面压灰；0=关闭")
    ap.add_argument("--x0-target", choices=("real", "teacher"), default="real",
                    help="x0 监督目标：real=真实 latent；teacher=教师去噪估计 noise-v_teacher"
                         "（真蒸馏，软目标带布局）。teacher 需要 ds2 缓存含 v_teacher。")
    ap.add_argument("--x0-loss", type=float, default=0.0,
                    help="配对 x0 重建 MSE 权重：x0hat=xt-σ*pred，直接对同一 x0 监督；"
                         "v-MSE 可被输入高通短路，x0-MSE 不会。建议 0.5。")
    ap.add_argument("--x0-hp-loss", type=float, default=0.0,
                    help="x0 的 2x2 patch 内高通 MSE 权重：专门防 x0hat 退化成 patch 均值/"
                         "色块；建议 0.1~0.2，与 --x0-loss 配套。")
    ap.add_argument("--x0-sigma-weight", action="store_true",
                    help="x0/x0hp 损失按当前 σ 加权（σ=1 全权重，σ=0.1 约 0.1）。"
                         "不开启时低 σ 梯度天然放大 1/σ，会把主干训练偏向低噪细化。")
    ap.add_argument("--x0-high-sigma-weight", type=float, default=0.0,
                    help="高 σ 段 x0 权重下限；0=关闭。hybrid_loss 下 x0 权重为 1-σ，"
                         "σ0.98 只剩 0.02，高 σ 几乎只学 v。设 0.5 可强制高 σ 也直接"
                         "监督 x0 布局。")
    ap.add_argument("--x0-high-sigma-threshold", type=float, default=0.7,
                    help="启用高 σ x0 权重下限的 σ 阈值（默认 0.7）")
    ap.add_argument("--x0-coarse-weight", type=float, default=0.0,
                    help="高σ coarse x0 loss：latent 先 2×2 avg-pool 再算 MSE；"
                         "0=关，不新增参数头")
    ap.add_argument("--x0-coarse-sigma", type=float, default=0.75,
                    help="coarse x0 只对 σ≥该值的样本生效，默认 0.75")
    ap.add_argument("--x0-hi-cos-balance", type=float, default=0.0,
                    help="高σ逐样本方向平衡：按 (1-cos_x0) 的滑动窗口难度放大 x0 权重；"
                         "0=关。只作用于 σ≥threshold 的样本。")
    ap.add_argument("--x0-hi-cos-balance-power", type=float, default=2.0,
                    help="难度倍率幂次；2=低 cos 样本获得更强权重")
    ap.add_argument("--x0-hi-balance-window", type=int, default=20,
                    help="每样本高σ cos 滑动窗口长度")
    ap.add_argument("--x0-hi-balance-max", type=float, default=2.5,
                    help="动态难样本权重上限")
    ap.add_argument("--x0-hi-balance-min", type=float, default=1.0,
                    help="动态难样本权重下限；1.0=不压低易样本")
    ap.add_argument("--x0-hi-balance-threshold", type=float, default=0.8,
                    help="动态均衡只在 σ≥该值时生效")
    ap.add_argument("--hi-grad-clip", type=float, default=0.0,
                    help="高σ x0 分支单独全局范数 clip；0=关。主干仍走总 grad-clip。")
    ap.add_argument("--x0-cos-high-weight", type=float, default=0.0,
                    help="高 σ 段 x0 cosine 方向损失权重；0=关闭。只约束方向，"
                         "不替代 x0 MSE/幅值正则；建议 0.2 起。")
    ap.add_argument("--x0-cos-high-threshold", type=float, default=0.7,
                    help="启用高 σ cosine 损失的 σ 阈值（默认 0.7）")
    ap.add_argument("--predict-x0", action="store_true",
                    help="参数化切换：模型输出直接当 x0（而非 v）。v 由 (xt-x0)/σ 反推。"
                         "高 σ 下强制网络输出画面布局，堵死‘复制噪声刷 cos’的捷径。")
    ap.add_argument("--v-loss-weight", type=float, default=1.0,
                    help="v-MSE 权重；--predict-x0 或 x0 主导训练时可降到 0.1 以下")
    ap.add_argument("--x0-dc-loss", type=float, default=0.0,
                    help="x0hat 的 DC 正则权重：按 (T,H,W) 求每样本每通道均值，罚其平方。"
                         "专门打断“整幅 latent 被拉出全局亮度偏移”的塌缩；建议 0.1~0.3。")
    ap.add_argument("--x0-std-loss", type=float, default=0.0,
                    help="x0hat 幅值匹配权重：每样本每通道 std 与真实 x0 的 std 做 MSE，"
                         "防止 std 虚高/被压平；建议 0.1。")
    ap.add_argument("--x0-fourier-loss", type=float, default=0.0,
                    help="x0-only 5D Fourier 三频带匹配损失权重；0=关闭。"
                         "启用时只接受 [0.1,0.2]，按样本总能量归一化，"
                         "只作用于 x0hat/x0，不作用于 v。")
    ap.add_argument("--flat-penalty", type=float, default=0.01,
                    help="flat_ratio 软惩罚权重：只罚 flat>--flat-target 的部分")
    ap.add_argument("--flat-target", type=float, default=1.2,
                    help="flat_ratio 软惩罚阈值")
    ap.add_argument("--final-grad-boost", type=float, default=4.0,
                    help="final_proj 梯度低于阈值时的 boost 倍数")
    ap.add_argument("--final-grad-min", type=float, default=1e-6,
                    help="final_proj 梯度范数低于此值时触发 boost")
    ap.add_argument("--final-grad-boost-until", type=int, default=2000,
                    help="final_proj 梯度 boost 截止步（含）")
    ap.add_argument("--skip-reg", type=float, default=0.0,
                    help="skip alpha 弱正则权重：单边 hinge，只压 alpha>target 的部分；"
                         "0=关闭")
    ap.add_argument("--skip-reg-target", type=float, default=0.95,
                    help="skip alpha 正则目标上界（默认0.95，仍接近全开，不是关死）")
    ap.add_argument("--boundary-loss", type=float, default=0.0,
                    help="分块边界一致性损失权重：只罚 latent 上 2px（画面32px）patch "
                         "接缝处的周期梯度尖峰；0=关闭")
    ap.add_argument("--boundary-margin", type=float, default=1.25,
                    help="接缝/内部梯度比超过该值才罚（分母 detach，只回传接缝）")
    ap.add_argument("--patch-boundary-loss", type=float, default=0.0,
                    help="32px patch 均值边界超额损失权重：相邻 2x2 token 均值的跳变"
                         "超过干净目标同位置跳变的 margin 倍才罚；直接压 32px 块边界，"
                         "不要求块内平滑，0=关闭")
    ap.add_argument("--patch-boundary-margin", type=float, default=1.15,
                    help="patch 均值边界允许相对目标跳变的倍数（默认1.15）")
    ap.add_argument("--gnorm-skip", type=float, default=0.0,
                    help="预裁剪梯度范数超过该值时跳过本步 optimizer.step（不更新权重），"
                         "用于防单步尖峰把模型推向坏模式；0=关闭（仍保留 clip=1.0）")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="全局梯度裁剪阈值。2026-11-26 新增：实测 gnorm P50=2.11/P90=6.20，"
                         "固定 1.0 时每个 batch 都被缩放 2–6 倍，与 Adafactor 的 "
                         "α=max(eps2,RMS(θ)) 叠加后 patchify 每步只剩 0.29 个 bf16 量子"
                         "（旧 2^-8 估法；按真实 2^-7 量子换算是 0.19）"
                         "默认 1.0 保持旧行为；调到 2.5≈P90 可让主干真实步长抬到 ~0.7 量子。")
    # bf16 随机舍入 + 选择性 fp32 master（2026-11-26 新增；默认全关）
    ap.add_argument("--sr-master-budget-m", type=float, default=0.0,
                    help="随机舍入+fp32 master 的参数预算（百万）。0=关闭（逐位等于旧行为）。"
                         "每 1M 参数多占 4MB 显存：peak 16.4GB 时可开到约 600M（+2.4GB）。"
                         "选中张量的更新在 fp32 累加、按 SR 写回 bf16，绕开 0.29 量子/步的截断。")
    ap.add_argument("--sr-master-prefixes", nargs="*", default=[],
                    help="按前缀优先选择张量（先匹配先占预算）。默认顺序 "
                         "dec_blocks. → patchify. → full_head. → skips. → bottleneck.")
    # 2026-11-26: SR 显式开关组（全部默认 OFF；不传 = 现有训练行为零变化）
    ap.add_argument("--sr-master", action=argparse.BooleanOptionalAction, default=False,
                    help="开启 bf16 随机舍入 + 选择性 fp32 master。默认关闭；"
                         "开启时必须搭配 --reset-optim（旧动量与 fp32 master 参数源会错位）。")
    ap.add_argument("--sr-modules", default="",
                    help="逗号分隔的模块名前缀/正则（正则写 re:^dec_blocks\\.0\\.）；"
                         "空=按内置默认顺序选全合适模块。仅在 --sr-master 开启时生效。")
    ap.add_argument("--sr-dtype", choices=["bf16"], default="bf16",
                    help="master 写回的前向参数 dtype；当前仅支持 bf16（预留位）。")
    ap.add_argument("--sr-every", type=int, default=1,
                    help="每 N 个 optimizer.step 把 fp32 master SR 写回 bf16 一次；1=每步（旧行为）。")
    ap.add_argument("--sr-include-dim1", action=argparse.BooleanOptionalAction, default=True,
                    help="SR 是否纳入 dim<=1 的 scale/bias/向量类参数（如 "
                         "full_head.pre_proj_scale）。默认 on；关闭则这类小 scale 仍原位 bf16。"
                         "注：uvit3d promote_gate_params_fp32 跳过 dim=1，bf16_sr 旧 exclude 也"
                         "曾排除 pre_proj_scale，双重点排除。")
    ap.add_argument("--data-gate-report-dir", default="",
                    help="ds1 每批采样统计落盘目录（σ 分桶/质量分/来源分布，jsonl + 每 50 批 rollup）。"
                         "空=关闭。既有 --data-gate-report 仍是 ds2 门禁 JSON 输入，二者不冲突。")
    ap.add_argument("--probe-qlog", default="",
                    help="可选：把每步参数位移 q/step、master-bf16 差、峰值显存、步耗时写入该目录 jsonl。"
                         "空=关闭（默认），不影响训练行为。")
    # 每素材最多 N 遍
    ap.add_argument("--max-sample-passes", type=int, default=2,
                    help="每条素材最多参与训练几遍。默认 2（硬上限）：用 PassLimitedSampler "
                         "发牌，达到上限的样本永久退出采样池；池子耗尽即 SystemExit，"
                         "绝不回绕复用。0=关闭门禁（旧行为，不建议）。")
    ap.add_argument("--usage-ledger", default="outputs/sample_usage_ledger.json",
                    help="遍数计数账本（JSON）。ds2 用此路径；ds1/ds3/ds4 各自用 "
                         "outputs/sample_usage_ledger_<key>.json。启动读入、每 50 个样本与退出时落盘。")
    ap.add_argument("--assume-history-passes", type=int, default=1,
                    help="ds1/ds3/ds4 没有历史账本时，按该值保守预置已用遍数（默认 1，"
                         "即每条只再允许 1 遍）。0=视为从未训练（仅在确认历史未用时可设）。")
    ap.add_argument("--channel-std-loss", type=float, default=0.0,
                    help="逐通道 std 匹配损失权重：只罚 pred 通道 std 低于 target 的部分")
    ap.add_argument("--channel-std-target", type=float, default=0.85,
                    help="逐通道 std 达到 target 的该比例才不罚（默认0.85）")
    ap.add_argument("--psd-loss", type=float, default=0.0,
                    help="径向频带功率占比匹配损失权重（低频/中频/高频各带同权）")
    ap.add_argument("--local-contrast-loss", type=float, default=0.0,
                    help="patch 局部对比损失权重：匹配相邻 2x2 patch 均值的差分幅值，"
                         "把 32px 块间对比拉回干净 latent；0=关闭")
    ap.add_argument("--spectrum-loss", type=float, default=0.0,
                    help="频谱形状损失权重：匹配 0.25-0.5 高频能量占比 + 径向斜率；0=关闭")
    ap.add_argument("--temporal-loss", type=float, default=0.0,
                    help="时间差分损失权重：匹配 x0hat 相邻帧差与干净 x0 的帧差；0=关闭")
    ap.add_argument("--prune", action="store_true", help="显式开启才自动删除旧 checkpoint")
    ap.add_argument("--keep-recent", type=int, default=2)
    ap.add_argument("--sample-cmd", default="", help="阶段末执行的采样命令（可空）")
    ap.add_argument("--ds1", default="latents_new")
    ap.add_argument("--ds1-extra", default="", help="额外 ds1 目录（高帧数 latent），与 ds1 合并")
    ap.add_argument("--extra-ratio", default="1:2", help="ds1:extra 混采比例，如 1:3")
    ap.add_argument("--sample-balance", choices=["ratio", "source"], default="ratio",
                    help="ratio=按文件比例混采（旧行为）；source=按源视频配平，"
                         "避免 9f/13f 重叠源被双喂、13f 独有源饿死")
    ap.add_argument("--source-nine-slots", type=int, default=3,
                    help="source 采样器每个重叠源每轮的 9f 槽数（默认3，总槽4）")
    ap.add_argument("--resume-sample-cursor", type=int, default=-1,
                    help="resume 时全局样本游标；-1=优先后 ckpt 内 train_state，"
                         "再回退 (start_step-1)*accum；跨旧 ckpt 建议显式指定")
    ap.add_argument("--resume-state", default="",
                    help="旧 ckpt 没有 train_state 时，从该 JSON 恢复 "
                         "sample_cursor/running/best_loss/n_done/zero_running")
    ap.add_argument("--shuffle-seed", type=int, default=0, help=">0 启用固定种子 shuffle")
    ap.add_argument("--seed", type=int, default=0,
                    help=">0 时固定 torch/cuda/python RNG，供 A/B 对齐噪声采样")
    ap.add_argument("--holdout", type=float, default=0.0, help="ds1 按源视频哈希留出的比例（0~0.2）")
    ap.add_argument("--p1-until", type=int, default=4000, help="P1 自监督延长到该步（默认 4000）")
    ap.add_argument("--p2-until", type=int, default=7000, help="P2 纯蒸馏窗口终点（默认 7000；延长可继续纯教师蒸馏而不切入 P3）")
    ap.add_argument("--ds2", default="", help="离线教师目标 .pt 目录（含 v_teacher）")
    ap.add_argument("--ds3", default="", help="高质量通用 latent 目录")
    ap.add_argument("--ds4", default="", help="二次元专项 latent 目录")
    # ---- 4000 步换装（本地直推理教师，不依赖 ComfyUI 服务）----
    ap.add_argument("--p2-autogen", action="store_true",
                    help=f"step{SWAP_STEP} 到点自动换装：暂停 -> 释放 DiT -> 教师直推理生成 DS2 "
                         f"缓存 -> 释放教师 -> 重载 DiT -> 自动接 P2 蒸馏")
    ap.add_argument("--p2-latent-dir", default="latents_new", help="教师输入 x0 的 latent 目录（DS2 样本）")
    ap.add_argument("--p2-prompts", default="teacher_prompts.json", help="latent 文件名 -> prompt 的配对表")
    ap.add_argument("--p2-out", default="teacher_targets", help="教师缓存输出目录（同时作为 ds2）")
    ap.add_argument("--p2-min", type=int, default=2000, help="缓存不足此条数则拒绝启动 P2")
    ap.add_argument("--p2-resigma", action="store_true",
                    help="P2 蒸馏时复用缓存 x0/noise，按 --t-sched/--t-sigma 重新加噪，"
                         "并用精确 target=noise-x0；用于补齐教师缓存缺失的低 σ 段")
    ap.add_argument("--p2-high-min-sigma", type=float, default=0.5,
                    help="--p2-resigma 与 --t-high-ratio>0 时，缓存 t≥该值才复用缓存 t/v_teacher；"
                         "否则用高 σ 重采样 + 精确 noise-x0")
    ap.add_argument("--data-gate-report", default="",
                    help="主机 data_gate.py 产生的分组 JSON；给了就按 kept 过滤并按组轮转采样，"
                         "不再直接 MixedFiles 混池")
    ap.add_argument("--data-gate-report-ds1", default="",
                    help="ds1 专用门禁 JSON（与 ds2 的 --data-gate-report 同构；给了就按 kept "
                         "过滤 ds1 并按组轮转采样，空=关闭）。报告由 data_gate.py 对 ds1 产出")
    ap.add_argument("--data-gate-strict", action="store_true",
                    help="data-gate 报告里 missing_or_underfilled 非空则直接拒绝启动")
    ap.add_argument("--self-renoise", action="store_true",
                    help="自再加噪训练：先用当前模型在 xt 上无梯度前向得到 x0_pred，"
                         "再把 x0_pred 加噪到 --sr-sigma-min/max 训练第二步。"
                         "用于修 teacher-forcing 的 exposure bias，成本约 2x 前向。")
    ap.add_argument("--sr-sigma-min", type=float, default=0.10,
                    help="self-renoise 第二步的 sigma 下限")
    ap.add_argument("--sr-sigma-max", type=float, default=0.50,
                    help="self-renoise 第二步的 sigma 上限")
    ap.add_argument("--rollout2", action="store_true",
                    help="轻量 2 步 rollout：sigma=1 无梯度前向生成 x0_pred，"
                         "再以 (1-t)*x0_pred+t*noise 到 sigma≈0.8，最后一步反传。"
                         "与 --self-renoise 互斥。")
    ap.add_argument("--rollout2-mid-sigma", type=float, default=0.80,
                    help="2 步 rollout 的中间 sigma；默认 0.8")
    ap.add_argument("--rollout2-low-consistency", type=float, default=0.0,
                    help="2 步 rollout 的低频结构一致性权重；两项 x0 的 low band 归一 MSE。")
    ap.add_argument("--low-tv-loss", type=float, default=0.0,
                    help="低频分量总变差正则权重（1e-4~1e-3）；对 x0hat 的 low band "
                         "计算相邻像素差 L1，抑制低频被摊平成纯色块。")
    ap.add_argument("--band-cond", action="store_true",
                    help="A1：把 xt 的 low/mid/high 占比经零初始化 band_mlp 加到共享 cond。"
                         "只做一次 rfft2，不做 30 个 Block 逐块 FFT。")
    ap.add_argument("--band-heads", action="store_true",
                    help="A3：启用 low/mid/high 三个零残差 denoise 头，并冻结旧 final_proj。"
                         "三头以 depthwise 恒等 + 1x1 零初始化起步，step0 与基线逐 bit 一致。")
    ap.add_argument("--band-high", action="store_true",
                    help="A3：启用 high 头。默认关闭并冻结 high 头参数，"
                         "防止主 x0 loss 绕过 v_band_w_high=0 把 high 头训起来。")
    # ---- 2026-11-26 融合管线新开关（默认全关；关闭路径与基线逐位等价）----
    ap.add_argument("--freq-route", action="store_true",
                    help="FreqFormer 轻量频域路由（默认关）。低秩三频带注意力门控，"
                         "state_dict 前缀 freqrouter.；cfg 透传 freq_route*。")
    ap.add_argument("--freq-route-rank", type=int, default=8,
                    help="FreqRouter 低秩瓶颈秩，默认 8（契约 §2）。")
    ap.add_argument("--freq-route-scale", type=float, default=1.0,
                    help="FreqRouter 门控幅度系数 gates=1+scale·tanh(…)，默认 1.0。")
    ap.add_argument("--freq-route-topk", type=int, default=0,
                    help="高频 token Top-k 稀疏保留数；0=关闭。")
    # ---- 2026-11-27 v3：频域解耦稀疏瓶颈 ----
    ap.add_argument("--text-token-dir", default="",
                    help="128-token UMT5 文本 token 缓存目录（text_tokens_*.pt）；"
                         "只有 --text-cross-attn 真正需要。")
    ap.add_argument("--text-dropout", type=float, default=0.0,
                    help="CFG 训练期文本 token 丢弃率；0=关，0.1=10%%。"
                         "丢弃样本用 zero token + 全 True mask，绝不广播 -inf。")
    ap.add_argument("--text-cross-encoder-layers", type=int, default=0,
                    help="encoder 前 N 层 text cross-attn；建议 6，不补 30 层")
    ap.add_argument("--text-cross-encoder-rank", type=int, default=0,
                    help="encoder text-cross Q/O 低秩；0=全秩，256=预算版")
    ap.add_argument("--adaln-text-all", action=argparse.BooleanOptionalAction,
                    default=None, help="30 层全挂低秩 AdaLN 文本调制")
    ap.add_argument("--adaln-text-rank", type=int, default=0,
                    help="全层 AdaLN 低秩；0=全秩，128=预算版")
    ap.add_argument("--sigma-adapter", action=argparse.BooleanOptionalAction,
                    default=None, help="σ 连续滑动低秩 adapter（30 blocks）")
    ap.add_argument("--sigma-adapter-rank", type=int, default=64)
    ap.add_argument("--context-tokens", action=argparse.BooleanOptionalAction,
                    default=None, help="4 个 pyramid tier context/register token")
    ap.add_argument("--context-token-rank", type=int, default=128)
    ap.add_argument("--repa-weight", type=float, default=0.0,
                    help="REPA DINOv2 特征对齐权重；0=关")
    ap.add_argument("--syntax-aux-weight", type=float, default=0.02,
                    help="SyntaxCompiler aux loss 权重；0=关。保留主干 detach，"
                         "只回传 skip/ffn/cross/tier 四头 + gate/intent")
    ap.add_argument("--repa-cache-dir", default="",
                    help="每样本 DINOv2 特征 .pt 缓存目录；REPA 开启时必填")
    ap.add_argument("--repa-warmup", type=int, default=100)
    ap.add_argument("--repa-decay", type=int, default=50)
    ap.add_argument("--text-cross-attn", action=argparse.BooleanOptionalAction,
                    default=None, help="bottleneck 6 层文本 cross-attn（三态："
                    "不传=按模型工厂；4b5v3 默认开）")
    ap.add_argument("--text-cross-rank", type=int, default=0,
                    help="cross-attn Q/O 低秩秩；0=全秩，64/128=BandLR 式低秩")
    ap.add_argument("--text-cross-base-gate", type=float, default=0.1,
                    help="gate=base_gate*sigma，默认 0.1")
    ap.add_argument("--text-cross-text-dim", type=int, default=3072,
                    help="文本 token 宽度；官方 text_embedder 投影后为 3072")
    ap.add_argument("--text-cross-freeze-kv",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="冻结 cross-attn K/V（默认 True，保住官方文本先验）")
    ap.add_argument("--text-cross-init-wan", default=None,
                    help="官方 Wan2.2 5B diffusers 目录；给定则从 "
                         "blocks.12..17.attn2 1:1 初始化")
    ap.add_argument("--text-cross-decoder-layers", type=int, default=0,
                    help="把 text cross-attn 扩到最靠近 bottleneck 的 N 个 decoder "
                         "block（dec_blocks[:N]，对应官方 29,28,...）；0=只保留 6 层 "
                         "bottleneck。每层 +37.75M 全秩参数。")
    ap.add_argument("--text-cross-scale-alpha", type=float, default=0.0,
                    help="噪声感知文本缩放：text_tokens*(1+alpha*sigma)，0=关")
    ap.add_argument("--adaln-text-layers", default="",
                    help="逗号分隔 decoder block 索引（如 0,1,2）；并行 AdaLN-Zero "
                         "文本注入，零初始化恒等起步")
    ap.add_argument("--adaln-text-gate-hidden", type=int, default=256,
                    help="AdaLN-Zero sigma 门控 MLP 隐层宽度")
    ap.add_argument("--flow-consist-weight", type=float, default=0.0,
                    help="高σ流向一致性正则权重；第二遍低σ前向用 no_grad")
    ap.add_argument("--flow-consist-sigma-threshold", type=float, default=0.8,
                    help="仅当 batch 平均 σ≥该值时启用一致性正则")
    ap.add_argument("--flow-consist-ratio", type=float, default=0.95,
                    help="低σ分支 sigma=sigma*ratio")
    ap.add_argument("--flow-consist-every", type=int, default=2,
                    help="每 N 个 optimizer step 做一次一致性前向，默认 2")
    ap.add_argument("--multi-scale-align-weight", type=float, default=0.0,
                    help="多尺度文本特征对齐总权重；0=关")
    ap.add_argument("--align-w-shallow", type=float, default=0.05)
    ap.add_argument("--align-w-middle", type=float, default=0.03)
    ap.add_argument("--align-w-deep", type=float, default=0.01)
    ap.add_argument("--bottleneck-global-mlp",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="bottleneck 全局池化 MLP（零初始化恒等）")
    ap.add_argument("--bottleneck-global-rank", type=int, default=128,
                    help="全局 MLP 低秩瓶颈，默认 128（6 块约 4.7M 参数）")
    ap.add_argument("--band-lr-rank", type=int, default=0,
                    help="全主干结构化低秩 rank；0=关。64/128 会把 4B 砍到 ~0.5B；"
                         "1536 约对应 30~40%% 参数下降。仅从零训练可开。")
    ap.add_argument("--band-lr-scope", choices=["all", "attn", "ffn"], default="all",
                    help="低秩替换范围：all=qkv/o+w12/w3；attn=qkv/o；ffn=w12/w3")
    ap.add_argument("--band-lr-rank-enc", type=int, default=0,
                    help="encoder 段独立 rank；0=沿用 --band-lr-rank")
    ap.add_argument("--band-lr-rank-bottleneck", type=int, default=0,
                    help="bottleneck 段独立 rank；0=沿用 --band-lr-rank")
    ap.add_argument("--band-lr-rank-dec", type=int, default=0,
                    help="decoder 段独立 rank；0=沿用 --band-lr-rank")
    ap.add_argument("--band-lr-rank-attn", type=int, default=0,
                    help="注意力模块独立 rank；0=沿用段/全局 rank。"
                         "典型：1536 锚定稳定性，FFN 另压到 512。")
    ap.add_argument("--band-lr-rank-ffn", type=int, default=0,
                    help="FFN 模块独立 rank；0=沿用段/全局 rank。")
    ap.add_argument("--band-lr-rank-ffn-enc", type=int, default=0,
                    help="浅层 FFN 独立 rank；0=沿用 --band-lr-rank-ffn")
    ap.add_argument("--band-lr-rank-ffn-bottleneck", type=int, default=0,
                    help="bottleneck FFN 独立 rank；0=沿用 --band-lr-rank-ffn")
    ap.add_argument("--band-lr-rank-ffn-dec", type=int, default=0,
                    help="深层 FFN 独立 rank；0=沿用 --band-lr-rank-ffn")
    ap.add_argument("--band-lr-qk-rank", type=int, default=0,
                    help="Q/K 分开低秩 rank；0=Q/K 保持 dense（推荐先只压 V/O）。"
                         "注意 Q/K 低秩 1536 的参数量≈dense，不省参数。")
    ap.add_argument("--band-lr-vo-rank", type=int, default=0,
                    help="V 与 O 分开低秩 rank；0=V/O 保持 dense。"
                         "512 约省 0.378B（30 层 QKV+O 结构）。")
    ap.add_argument("--band-lr-attn-groups", type=int, default=0,
                    help="注意力分组块对角组数 G；0=关。G=2 约省 0.566B"
                         "（Q/K/V/O 各半，组内全秩、组间零交互）。")
    ap.add_argument("--band-lr-attn", action=argparse.BooleanOptionalAction,
                    default=True, help="是否低秩替换 self-attn qkv/o")
    ap.add_argument("--band-lr-ffn", action=argparse.BooleanOptionalAction,
                    default=True, help="是否低秩替换 SwiGLU w12/w3")
    ap.add_argument("--attn-spatial-window", type=int, default=0,
                    help="spatial 头局部窗口边长；0=关。加性 mask，"
                         "不承诺 ROCm 省 FLOPs，只作质量消融")
    ap.add_argument("--hybrid-loss", action="store_true",
                    help="高 σ 偏 v-loss、低 σ 偏 x0-loss 的逐样本混合；"
                         "不是双头参数化，仍是单 x0 输出")
    ap.add_argument("--x0-sigma-power", type=float, default=1.0,
                    help="x0/x0hp 的 σ 权重指数；1=现有 σ，2=σ²；"
                         "仅作消融，direct x0 下没有 1/σ 抵消问题")
    ap.add_argument("--fourier-sigma-weight", action="store_true",
                    help="频域解耦损失：低频 ∝σ²，中/高 ∝(1-σ)²")
    ap.add_argument("--band-head-lr", action="store_true",
                    help="多尺度低秩频带头（默认关）。挂在 full_head.bandlr 下，"
                         "前缀 bandlr.；关闭时前向不调用，与基线逐位等价。")
    ap.add_argument("--bandlr-structure-loss", type=float, default=0.0,
                    help="bandlr 低频共享特征 Sobel 结构正则权重；默认 0（只监控）。"
                         "启用时由 bandlr 的 structure_reg 直接进 loss。")
    ap.add_argument("--reset-bandlr-low-gate", action="store_true",
                    help="resume 后把 full_head.bandlr.low_gate_logit 强制重置为 "
                         "--bandlr-low-gate-init（旧 ckpt 是 -3.0，新代码默认 -8.0）")
    ap.add_argument("--bandlr-low-gate-init", type=float, default=-8.0,
                    help="新建/重置 low_gate_logit 的初始值；sigmoid(-8)=3.35e-4")
    ap.add_argument("--semantic-proj", action="store_true",
                    help="latent 语义投影头占位开关（默认关）。当前无离线 CLIP 帧级"
                         "特征缓存：挂载后冻结、不接损失；前缀 semproj.。")
    ap.add_argument("--latent-distill", action="store_true",
                    help="在线低秩中间层蒸馏占位开关（默认关）。无在线冻结教师挂载"
                         "时显式报错，禁止静默跑空 loss；前缀 ldistill.。")
    ap.add_argument("--v-band-loss", type=float, default=0.1,
                    help="A3 的 v 频带监督总权重；只在 --band-heads 时生效，0=关闭。"
                         "监督对象是 v_teacher=noise-x0 的 low/mid/high 残差，不是 x0。")
    ap.add_argument("--v-band-w-low", type=float, default=1.0)
    ap.add_argument("--v-band-w-mid", type=float, default=0.5)
    ap.add_argument("--v-band-w-high", type=float, default=0.2)
    ap.add_argument("--band-lr", type=float, default=1e-4,
                    help="A3 三个带头的独立 AdamW lr；零初始化 proj 需要逐元素归一化。")
    ap.add_argument("--band-cond-lr", type=float, default=1e-5,
                    help="A1 band_mlp 的独立 AdamW lr。实测 1e-4 一步会把 x0 从 0.104 "
                         "抬到 0.146，A1 必须比 A3 低一个量级。")
    ap.add_argument("--band-head-scale", type=float, default=0.05,
                    help="A3 三头的固定残差缩放。feat RMS≈62，1x1 头权重 7e-5 时"
                         "原始输出 RMS≈0.23；不缩放会 20 步把 v 改掉 1/3。")
    ap.add_argument("--allow-arch-change", action="store_true",
                    help="允许 resume 时模型新增参数（A1 band_mlp / A3 band_head_*）。"
                         "只跳过参数顺序 fatal，不改旧权重。")
    ap.add_argument("--share-special-ffn", action=argparse.BooleanOptionalAction, default=True,
                    help="4b5fs 专项块与 enc 末块是否共享 FFN。热启建议 --no-share-special-ffn，"
                         "两套 FFN 都保留；全新训练再开省 151M。仅 4b5fs 生效。")
    for _flag, _help in (
        ("a2-crossmodal", "4b5fs 跨模态模块（默认按 full_stack 开）"),
        ("a2-crossmodal-in-attn", "4b5fs 摘要 K/V 拼进每层注意力（bias-SDPA 慢路径的开关）"),
        ("dual-branch-adaln", "4b5fs 双支 AdaLN"),
        ("dual-sem-head", "4b5fs 语义边界双功能头"),
        ("snr-cond", "4b5fs SNR 条件调制"),
        ("local-conv", "4b5fs 每块局部卷积"),
        ("pau-syntax", "4b5fs PAU 语法平面"),
    ):
        ap.add_argument(f"--{_flag}", action=argparse.BooleanOptionalAction, default=None,
                        help=_help + "（三态：不传=按 full_stack 默认）")
    ap.add_argument("--dual-sem-ffn-hidden", type=int, default=None,
                    help="4b5fs 语义边界头的 SwiGLU 宽度（full_stack 默认 3484）。"
                         "每步耗时主要来自它：3484→1024 约省 1.6s/步。")
    ap.add_argument("--reset-optim", action="store_true",
                    help="resume 时丢弃旧 optimizer_state，新架构从零动量开始。"
                         "新增参数会改变 param group 布局，旧二阶矩无法安全映射。")
    ap.add_argument("--p2-limit", type=int, default=0, help="0=全量生成；冒烟可设 5")
    ap.add_argument("--p2-cfg", type=float, default=6.0)
    ap.add_argument("--p2-min-free", type=float, default=MIN_FREE_GB_TEACHER,
                    help="载教师前要求的空闲显存（GB），低于此直接拒绝，不硬挤")
    ap.add_argument("--dry-run", action="store_true", help="只打印调度与数据检查，不初始化 CUDA")
    args = ap.parse_args()
    if not (0.0 <= args.t_sigma_min <= args.t_sigma_max <= 1.0):
        raise SystemExit(f"t_sigma 范围非法: [{args.t_sigma_min},{args.t_sigma_max}]")
    if not (0.0 <= args.sr_sigma_min <= args.sr_sigma_max <= 1.0):
        raise SystemExit(f"sr_sigma 范围非法: [{args.sr_sigma_min},{args.sr_sigma_max}]")
    if not (0.0 < args.rollout2_mid_sigma < 1.0):
        raise SystemExit(f"rollout2_mid_sigma 必须在 (0,1): {args.rollout2_mid_sigma}")
    if args.rollout2 and args.self_renoise:
        raise SystemExit("--rollout2 与 --self-renoise 互斥，只能开一个")
    if not (0.0 <= args.t_high_ratio <= 1.0):
        raise SystemExit(f"t_high_ratio 非法: {args.t_high_ratio}")
    if args.freq_route_rank <= 0:
        raise SystemExit(f"freq_route_rank 必须 > 0: {args.freq_route_rank}")
    if args.freq_route_scale < 0.0:
        raise SystemExit(f"freq_route_scale 不能为负: {args.freq_route_scale}")
    if args.freq_route_topk < 0:
        raise SystemExit(f"freq_route_topk 不能为负: {args.freq_route_topk}")
    if args.hybrid_loss and args.v_loss_weight <= 0:
        raise SystemExit("--hybrid-loss 需要 --v-loss-weight > 0；高 σ 端没有 v 项")
    if args.text_cross_rank < 0:
        raise SystemExit(f"text_cross_rank 不能为负: {args.text_cross_rank}")
    if args.text_cross_decoder_layers < 0:
        raise SystemExit(f"text_cross_decoder_layers 不能为负: {args.text_cross_decoder_layers}")
    if args.band_lr_rank < 0:
        raise SystemExit(f"band_lr_rank 不能为负: {args.band_lr_rank}")
    if args.band_lr_qk_rank < 0 or args.band_lr_vo_rank < 0:
        raise SystemExit("band_lr_qk_rank / band_lr_vo_rank 不能为负")
    if args.band_lr_attn_groups < 0:
        raise SystemExit("band_lr_attn_groups 不能为负")
    if args.model == "4b5v3" and not args.text_token_dir:
        raise SystemExit("--model 4b5v3 需要 --text-token-dir（128-token 文本缓存）")
    # 2026-11-26：允许 0.02~0.3，第三优先级需要 xf=0.08 对照；
    # 旧门禁 [0.1,0.2] 只是早期经验区间，不是硬性安全边界。
    if args.x0_fourier_loss != 0.0 and not (0.02 <= args.x0_fourier_loss <= 0.3):
        raise SystemExit(f"x0_fourier_loss 只接受 0 或 [0.02,0.3]: {args.x0_fourier_loss}")
    if args.repa_weight > 0.0 and not str(args.repa_cache_dir or "").strip():
        raise SystemExit("--repa-weight>0 时必须给 --repa-cache-dir（离线 DINOv2 特征缓存目录）")
    return args


def _startup_trace(msg):
    """启动阶段落盘打点：PowerShell 重定向有缓冲，日志看不到时用它定位卡点。"""
    try:
        import time as _t, os as _os
        with open(r"logs/startup_trace.log", "a", encoding="utf-8") as _f:
            _f.write(f"{_t.strftime('%H:%M:%S')} pid={_os.getpid()} {msg}\n")
    except Exception:
        pass


def main():
    args = parse_args()
    if not 0.0 <= float(getattr(args, "text_dropout", 0.0) or 0.0) < 1.0:
        raise SystemExit("[fatal] --text-dropout 必须在 [0,1)；1.0=全 null，训练无文本条件")
    if args.corr_target_min != 0.0:
        raise SystemExit("[fatal] corr loss 已改为单边 defect 上限；--corr-target-min 必须为 0，"
                         "旧双向 band 会主动降低 patch 相关性")
    if not 0.0 <= args.corr_target_max <= 2.0:
        raise SystemExit("[fatal] --corr-target-max 必须在 [0,2]，对应真实 corr 缺陷 1-corr")
    if args.corr_loss < 0:
        raise SystemExit("[fatal] --corr-loss 不能为负")
    _TRAIN_CONFIG.clear()
    _TRAIN_CONFIG.update(vars(args))
    if args.until_step > args.steps:
        args.until_step = args.steps
    if args.mem_frac > 0.95:
        print("[warn] mem_frac 过高，降到 0.95", flush=True)
        args.mem_frac = 0.95
    sch = build_schedule(args.until_step, args.p1_until, getattr(args, "p2_until", 7000))
    if not sch:
        raise SystemExit("[fatal] 调度为空")
    if args.dry_run:
        print_schedule(sch, args)
        return

    _load_runtime()  # 真正开训前才 import torch；dry-run 不触发 ROCm
    _startup_trace("runtime imported")
    # 2026-11-25: seed 设置原先在 _load_runtime() 之前引用 torch，带 --seed 必 NameError。
    if getattr(args, "seed", 0):
        import random as _random
        _random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        print(f"[seed] torch/cuda/python RNG fixed to {args.seed}", flush=True)
    data = validate_data(sch, args)
    _startup_trace("validate_data done")

    # 数据侧审计：E[x0^2] 直接决定零预测基线，先报出来再开训（2026-11-23 新增）
    for key in ("ds1", "ds3", "ds4"):
        if data.get(key):
            _startup_trace(f"stats {key} begin n={len(data[key])}")
            st = latent_dataset_stats(data[key])
            _startup_trace(f"stats {key} done")
            if st:
                print(f"[data-stat] {key} n={st['n']} mean={st['mean']:+.4f} std={st['std']:.4f} "
                      f"E[x2]={st['e2']:.4f} -> 零预测MSE≈{st['zero']:.3f}", flush=True)
    if data.get("ds2"):
        st = latent_dataset_stats(data["ds2"])
        if st:
            print(f"[data-stat] ds2 x0 n={st['n']} E[x2]={st['e2']:.4f}", flush=True)

    assert torch.cuda.is_available(), "需要 GPU（ROCm/CUDA）"

    # 结构开关必须与 ckpt 一致。新 ckpt 有轻量 sidecar，构建 4B 模型前即可继承；
    # 旧 ckpt 没有 sidecar 时，在 load 处做严格 mismatch 拒绝（见下）。
    if args.resume and os.path.isfile(args.resume + ".config.json") and not args.full_head_disable:
        try:
            with open(args.resume + ".config.json", encoding="utf-8") as f:
                side = json.load(f)
            saved = list(side.get("full_head_disable", []) or [])
            if saved:
                args.full_head_disable = saved
                # _TRAIN_CONFIG 在 sidecar 之前已经 update(vars(args))，必须同步，
                # 否则新 ckpt 的 train_config 会记成空 list，下一次 resume 自相矛盾。
                _TRAIN_CONFIG["full_head_disable"] = list(saved)
                print(f"[resume] sidecar full_head_disable={saved}", flush=True)
        except Exception as e:
            print(f"[resume][warn] sidecar 读取失败: {e}", flush=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device, dtype = "cuda", torch.bfloat16
    _startup_trace("cuda fraction set")

    free, total = torch.cuda.mem_get_info()
    _startup_trace(f"mem_get_info free={free/1e9:.2f}")
    print(f"[env] {torch.cuda.get_device_name(0)} free {free/1e9:.2f}/{total/1e9:.2f}GB "
          f"mem_frac={args.mem_frac} bf16 Adafactor accum={args.accum}", flush=True)
    _startup_trace("before model build")

    # 2026-11-23 实测：ROCm/Windows 上在 CPU 构建 4B 模型要 20+ 分钟（参数初始化逐层走 CPU）；
    # 直接 GPU + bf16 构建实测 1.6 秒。默认走 GPU-bf16，失败才回退 CPU。
    old_dtype = torch.get_default_dtype()
    try:
        with torch.device("cuda"):
            torch.set_default_dtype(torch.bfloat16)
            model = build_model(args.model, args)
        model = model.to(device).to(dtype).train()
        # 2026-11-26 fusion 接线（集成线）：惰性模块必须在 resume 加载 / 优化器
        # 创建之前构造，新参数才进 state_dict 顺序比较与优化器分组。
        # 关闭时不注册任何新参数，state_dict 与旧 ckpt 逐 key 一致。
        _fh_early = getattr(model, "full_head", None)
        if args.band_head_lr:
            if _fh_early is None or not hasattr(_fh_early, "set_band_lr"):
                raise SystemExit("[fatal] --band-head-lr 已开，但 full_head 无 set_band_lr 接口"
                                 "（band_head_lr.py 未落地？）")
            _fh_early.set_band_lr(True, enable_high=False)
            # 惰性构造发生在模型 .to(device) 之后，显式把新子模块搬到 GPU/bf16。
            if getattr(_fh_early, "bandlr", None) is not None:
                _fh_early.bandlr.to(device).to(dtype)
            print("[fusion] band_head_lr 已在模型构建后立即启用（惰性构造 bandlr.*，high 冻结）",
                  flush=True)
        model.semproj = None
        if args.semantic_proj:
            try:
                from semantic_proj import LatentSemanticProj
            except ImportError as _e:
                raise SystemExit(f"[fatal] semantic_proj.py 未找到，--semantic-proj 无法启用: {_e}")
            with torch.device("cuda"):
                model.semproj = LatentSemanticProj(
                    in_channels=int(getattr(model.cfg, "in_channels", 48)),
                    clip_dim=512, rank=64, enabled=True)
            model.semproj = model.semproj.to(device).to(dtype)
            # 占位：当前无离线 CLIP 帧级特征缓存，不解码视频、不拉 CLIP 大模型；
            # 挂载后整体冻结，不进任何损失回路。
            model.semproj.freeze()
            print("[semproj] 占位：模块已挂载（前缀 semproj.），无 CLIP 目标缓存，冻结且不接损失",
                  flush=True)
        model.ldistill = None
        if args.latent_distill:
            try:
                from latent_distill import OnlineLowRankDistill  # noqa: F401
            except ImportError as _e:
                raise SystemExit(f"[fatal] latent_distill.py 未找到，--latent-distill 无法启用: {_e}")
            raise SystemExit("[fatal] --latent-distill 需要在线冻结教师同步前向；当前离线 v_teacher "
                             "缓存模式无教师挂载，按契约禁止静默跑空 loss。")
        torch.set_default_dtype(old_dtype)
        _startup_trace("model built on GPU")
    except Exception as e:
        torch.set_default_dtype(old_dtype)
        _startup_trace(f"GPU build FAILED {str(e)[:200]}")
        print(f"[fatal] GPU build failed: {str(e)[:300]}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    args.t_dim = model.cfg.t_emb_dim
    args.cond_dim = model.cfg.cond_dim
    args.in_channels = model.cfg.in_channels

    # 顺序很重要（2026-11-23 修正）：F2 必须**先**做，再 warmstart。
    # 旧代码先 warmstart 再 F2，而 reinit_linear_teacher_spectrum 会遍历所有 Linear，
    # 把刚拷进来的 patch/head/cond_proj/ada.1 又重设成 N(0,0.02) —— 等于 warmstart 白做。
    if not args.resume and args.init_std > 0:
        from uvit3d import reinit_linear_teacher_spectrum
        n = reinit_linear_teacher_spectrum(model, std=args.init_std)
        print(f"[init] F2 谱初始化 std={args.init_std} 重设 {n} 个 Linear（零初始化层保持 0）", flush=True)

    if args.warmstart and not args.resume:
        if args.warmstart_mode == "modulation":
            from teacher_warmstart import warmstart_modulation_only as _warm, format_report
        else:
            from teacher_warmstart import warmstart_from_teacher as _warm, format_report
        rep = _warm(model, args.warmstart)
        print(format_report(rep), flush=True)
        if not rep["ok"]:
            print("[warn] warmstart 未完全成功", flush=True)

    start_step = 1
    resume_train_state = {}
    resume_optim_state = {}
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_train_state = dict(ck.get("train_state") or {})
        resume_optim_state = dict(ck.get("optimizer_state") or {})
        resume_optim_names = dict(ck.get("optimizer_param_names") or {})
        ck_sd, model_sd = ck["model"], model.state_dict()
        # 2026-11-23：strict=False 会「静默」跳过形状不符的权重。结构一改（例如
        # ada_embedder 输出 3072→18432）旧 ckpt 就会丢层且不报错，训练看似继续、
        # 实际条件通路是随机的。这里先显式比对形状，不符直接拒绝续训。
        bad = [k for k, v in ck_sd.items()
               if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)]
        if bad:
            raise SystemExit(
                f"[fatal] resume 形状不匹配 {len(bad)} 项（例：{bad[:3]}），"
                f"结构与 checkpoint 不一致，拒绝续训。请从 step1 重训或换匹配的 ckpt。")
        missing, unexpected = model.load_state_dict(ck_sd, strict=False)
        # P2-A：text_cross_modules / adaln_text_modules / sigma_adapter_modules
        # 已由 ModuleList 改成普通 list，旧 state_dict 里的镜像双份 key 只在 ckpt
        # 侧出现；它们是可安全丢弃的重复镜像，不是当前架构真的缺层。
        _mirror_prefixes = ("text_cross_modules.", "adaln_text_modules.",
                             "sigma_adapter_modules.")
        _deprecated_lost = {"ref_gate", "crossmodal.gate_a", "crossmodal.gate_b"}

        def _is_mirror_or_deprecated(n):
            return n in _deprecated_lost or n.startswith(_mirror_prefixes)

        if missing:
            # missing = 当前模型参数/buffer 在 ckpt 中没有（新架构新增，保持初始化）。
            print(f"[resume][audit] missing_keys={len(missing)}（新参数/buffer，保持初始化）"
                  f" 样例={missing[:5]}", flush=True)
        if unexpected:
            _unexp_mirror = [k for k in unexpected if _is_mirror_or_deprecated(k)]
            _unexp_other = [k for k in unexpected if not _is_mirror_or_deprecated(k)]
            print(f"[resume][audit] unexpected_keys={len(unexpected)} "
                  f"mirror_drop={len(_unexp_mirror)} other={len(_unexp_other)}",
                  flush=True)
            if _unexp_mirror:
                print(f"[resume][audit]   镜像/废弃旧 key 可安全丢弃（前5）: "
                      f"{_unexp_mirror[:5]}", flush=True)
            if _unexp_other:
                print(f"[resume][warn]   非镜像残留旧 key（前5）: {_unexp_other[:5]}",
                      flush=True)
        saved_model_names = list(ck.get("model_param_names") or [])
        now_model_names = [n for n, _ in model.named_parameters()]
        if saved_model_names and saved_model_names != now_model_names:
            extra = [n for n in now_model_names if n not in saved_model_names]
            lost = [n for n in saved_model_names if n not in now_model_names]
            _v32_new_tags = (".text_cross.", ".text_adaln.", ".sigma_adapter.",
                             ".context_tokens.", "context_tokens.", "repa_proj.",
                             "syntax_compiler.")

            def _is_whitelisted_extra(n):
                # 与原 allowed_extra 判定逐字一致：v3.2 子模块用子串识别，
                # 其它新模块用顶层/头内前缀识别；其余一律算未识别新增。
                return (n.startswith(("band_mlp.", "full_head.band_", "full_head.bandlr.",
                                      "freqrouter.", "semproj.", "ldistill."))
                        or any(_t in n for _t in _v32_new_tags))

            _extra_new = [n for n in extra if _is_whitelisted_extra(n)]
            _extra_other = [n for n in extra if not _is_whitelisted_extra(n)]
            _lost_safe = [n for n in lost if _is_mirror_or_deprecated(n)]
            _lost_other = [n for n in lost if not _is_mirror_or_deprecated(n)]
            if extra:
                from collections import Counter as _C
                # v3.2 新模块不是顶层前缀，而是挂在 block 内的子模块：
                # enc_blocks.*.text_cross / bottleneck.*.sigma_adapter 等。
                # 用子串识别，避免 resume 白名单漏掉它们。
                def _pref_of(_n):
                    for _t in _v32_new_tags:
                        if _t in _n:
                            return "NEW:" + _t.strip(".")
                    for _p in ("freqrouter.", "full_head.bandlr.", "semproj.",
                               "ldistill.", "band_mlp.", "full_head.band_"):
                        if _n.startswith(_p):
                            return _p
                    return "OTHER:" + _n.split(".")[0]
                _pref = _C(_pref_of(n) for n in extra)
                print(f"[resume][audit] extra={len(extra)} "
                      f"(whitelist_new={len(_extra_new)} other_new={len(_extra_other)}) "
                      f"lost={len(lost)} (mirror_drop={len(_lost_safe)} "
                      f"other_lost={len(_lost_other)}) prefix={dict(_pref)}",
                      flush=True)
            else:
                print(f"[resume][audit] extra=0 lost={len(lost)} "
                      f"(mirror_drop={len(_lost_safe)} other_lost={len(_lost_other)})",
                      flush=True)
            if _extra_other:
                # 真正未识别的新增可训练参数：不能静默放过，逐项打印名+shape；
                # 仅在 --allow-arch-change --reset-optim 显式分支下才放行。
                _pd = dict(model.named_parameters())
                for _n in _extra_other[:5]:
                    print(f"[resume][warn] 未识别新增参数 name={_n} "
                          f"shape={tuple(_pd[_n].shape) if _n in _pd else '?'}",
                          flush=True)
                print(f"[resume][warn] 未识别新增共 {len(_extra_other)} 项（前5已列）："
                      "不在 v3.2 白名单，必须 --allow-arch-change --reset-optim 显式放行，"
                      "否则拒绝续训。", flush=True)
            if _lost_safe:
                print(f"[resume][audit] lost 镜像/废弃 key 可安全丢弃 {len(_lost_safe)} 项"
                      f"（前5）: {_lost_safe[:5]}", flush=True)
            if _lost_other:
                print(f"[resume][warn] lost 非镜像旧参数 {len(_lost_other)} 项"
                      f"（前5）: {_lost_other[:5]}", flush=True)
            allowed_extra = all(_is_whitelisted_extra(n) for n in extra) if extra else False
            allowed_lost = bool(lost) and all(_is_mirror_or_deprecated(n) for n in lost)
            # 显式 --allow-arch-change --reset-optim 时：优化器状态已清空，
            # ckpt 中当前架构不存在的旧模块权重直接忽略，不做顺序映射校验。
            arch_branch = (bool(getattr(args, "reset_optim", False))
                           and bool(getattr(args, "allow_arch_change", False)))
            if args.allow_arch_change and ((not lost or allowed_lost or arch_branch)
                    and (allowed_extra or getattr(args, "reset_optim", False))):
                _dropped = len(lost) if arch_branch else 0
                print(f"[resume][warn] 架构扩展放行：新增 {len(extra)} 项，"
                      f"丢弃当前架构不存在的旧模块 {_dropped} 项，"
                      f"按 key 加载旧权重、新参数保持初始化"
                      f"{'（已 --reset-optim，旧动量丢弃）' if getattr(args, 'reset_optim', False) else ''}",
                      flush=True)
            else:
                n = min(len(saved_model_names), len(now_model_names))
                first = next((i for i in range(n)
                              if saved_model_names[i] != now_model_names[i]), n)
                raise SystemExit(
                    f"[fatal] 参数注册顺序变化：ckpt 与当前模型第 {first} 项起不一致 "
                    f"(ckpt={saved_model_names[first:first+2]}, "
                    f"当前={now_model_names[first:first+2]}；"
                    f"lost={lost[:2]}, extra={extra[:2]})。"
                    "优化器状态按顺序映射，继续加载会静默错位；请恢复模块注册顺序，"
                    "或显式传 --allow-arch-change --reset-optim 做新架构分支。")
        saved_disable = list((ck.get("train_config", {}) or {}).get("full_head_disable", []) or [])
        if not saved_disable:
            # 兼容 2026-11-26 之前写出的 ckpt：train_config 在 sidecar 继承前生成，
            # 可能是空 list，但顶层 full_head_disable 已是真实值。
            saved_disable = list(ck.get("full_head_disable", []) or [])
        now_disable = list(getattr(model.cfg, "full_head_disable", ()) or [])
        if set(saved_disable) != set(now_disable):
            raise SystemExit(
                f"[fatal] full_head_disable 不一致：ckpt={saved_disable} "
                f"当前={now_disable}。续训必须显式传同样的 --full-head-disable，"
                "否则会静默启用 ckpt 里关闭/随机初始化的子模块。")
        if saved_disable:
            print(f"[resume] full_head_disable 一致 {saved_disable}", flush=True)
        start_step = int(ck.get("step", 0)) + 1
        del ck, ck_sd
        gc.collect()
        print(f"[resume] {args.resume} -> step {start_step}（ck 已释放）", flush=True)
    elif args.resume:
        raise SystemExit(f"[fatal] resume 文件不存在: {args.resume}")

    # 新完整输出头不热启旧 LinearHead；旧头仅保留 key 兼容，显式清零。
    if getattr(model.cfg, "full_gated_head", False):
        with torch.no_grad():
            old = getattr(model, "head", None)
            if old is not None and hasattr(old, "head"):
                old.head.weight.zero_()
                old.head.bias.zero_()
            print("[init] full_gated_head=True：旧 LinearHead 已清零且不参与前向", flush=True)
        with torch.no_grad():
            fh = getattr(model, "full_head", None)
            if fh is not None and hasattr(fh, "alpha"):
                fh.alpha.data.clamp_min_(args.head_alpha_min)
                print(f"[init] full_head.alpha clamp>={args.head_alpha_min:.3f} "
                      f"-> {float(fh.alpha):.4f}", flush=True)
            if fh is not None and hasattr(fh, "out_gain"):
                fh.out_gain.data.clamp_min_(0.08)
                print(f"[init] full_head.out_gain clamp>=0.080 -> "
                      f"{float(fh.out_gain):.4f}", flush=True)
            if fh is not None and args.reset_high_out and hasattr(fh, "high_out"):
                # 旧高流初始化是 high_in/high_out 恒等 + 固定 2x2 高通；它绕开
                # final_proj 的零初始化，σ=1 时几乎复制输入高频。清零 high_out
                # 让高流成为真正的零残差，之后由 x0/hp 损失决定是否长出来。
                fh.high_out.weight.zero_()
                fh.high_out.bias.zero_()
                print("[init] full_head.high_out 清零：高流从零残差开始", flush=True)

    if args.resume_state:
        with open(args.resume_state, "r", encoding="utf-8") as f:
            resume_train_state.update(json.load(f))
        print(f"[resume] train_state <- {args.resume_state}", flush=True)

    # 门控参数必须 fp32：bf16 主干的舍入会把 1e-6 级小步长直接抹掉。
    n_gate_fp32 = promote_gate_params_fp32(model)
    print(f"[gate] 门控参数 fp32 化 {n_gate_fp32} 个（主干仍 bf16）", flush=True)

    # 阶段1热启动冻结：方案要求主干原有参数 requires_grad=False，只训内嵌
    # Rezero/频带 Norm/门控 MLP 新增部分。--train-mod-only 比 --freeze-backbone
    # 更严格：连 3200 已有的 final_proj/alpha/out_gain 也全部冻结。
    if args.freeze_backbone or args.train_mod_only:
        mod_prefixes = ("full_head.band_norm.", "full_head.dualact_gate.",
                        "full_head.patch_norm.")
        frozen = 0
        trainable = 0
        for n, p in model.named_parameters():
            if args.train_mod_only:
                ok = n.startswith(mod_prefixes)
            else:
                ok = n.startswith("full_head.")
            p.requires_grad_(ok)
            if ok:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(f"[freeze] train_mod_only={args.train_mod_only} "
              f"freeze_backbone={args.freeze_backbone} "
              f"trainable={trainable/1e6:.3f}M frozen={frozen/1e6:.3f}M", flush=True)

    # 被诊断开关关闭的新模块不应进入优化器；即使后续 resume 重新设置主干
    # requires_grad，也不能把它们意外复活。结构/state_dict key 仍完整保留。
    fh = getattr(model, "full_head", None)
    if fh is not None and hasattr(fh, "freeze_disabled_modules"):
        n_disabled = fh.freeze_disabled_modules()
        print(f"[freeze] disabled full_head modules frozen={n_disabled/1e6:.6f}M", flush=True)
        if hasattr(fh, "freeze_dead_modules"):
            n_dead = fh.freeze_dead_modules()
            print(f"[freeze] dead reserved full_head modules frozen={n_dead/1e6:.6f}M "
                  f"audit={fh.parameter_audit()}", flush=True)

    # A1/A3：必须在 freeze_disabled/dead 之后设置，避免 set_band_heads 把
    # 被显式冻结的 final_proj 又解冻；band_mlp 的零初始化保证关闭时恒等。
    model.band_cond_enabled = bool(args.band_cond)
    fh = getattr(model, "full_head", None)
    if fh is not None:
        fh.set_band_heads(bool(args.band_heads), enable_high=bool(args.band_high))
        # 2026-11-26：set_band_heads(enabled=True) 会执行
        # `self.final_proj.requires_grad_(False)`（full_gated_head.py:1009，A3 残差头
        # 设计：base 投影冻结、只让 band 头学残差）。但实测该 base 投影的
        # grad_norm=5.1246 是全模型最大项，冻死后 `--head-final-lr` 组里只有一个
        # requires_grad=False 的张量，日志 `final_proj.w` 20 步恒为 0.0q —— 输出方向
        # 与幅值链完全无法调整，只能靠 band 头 0.05 的残差，模型因此退化成
        # “把幅值缩到 0”（rms 0.61→0.22），MSE 反而比全零预测更差。
        # 因此：显式给 --head-final-lr>0 时解除该冻结（默认 0 保持 A3 原设计）。
        if float(getattr(args, "head_final_lr", 0.0)) > 0.0:
            for _p in fh.final_proj.parameters():
                _p.requires_grad_(True)
            print("[unfreeze] final_proj 解冻（head_final_lr>0）："
                  f"{sum(p.numel() for p in fh.final_proj.parameters())} 参数", flush=True)
        fh.band_head_scale_buf.fill_(float(args.band_head_scale))
        # fusion：freeze_disabled/dead 之后再次确认 bandlr 开关状态（high 保持冻结）。
        if hasattr(fh, "set_band_lr"):
            fh.set_band_lr(bool(args.band_head_lr), enable_high=False)
        if (bool(getattr(args, "reset_bandlr_low_gate", False))
                and getattr(fh, "bandlr", None) is not None):
            with torch.no_grad():
                fh.bandlr.low_gate_logit.fill_(float(args.bandlr_low_gate_init))
            print(f"[bandlr] low_gate_logit reset -> {float(args.bandlr_low_gate_init):.3f} "
                  f"(sigmoid={float(torch.sigmoid(fh.bandlr.low_gate_logit)):.6f})", flush=True)
    # 新增模块参数量实测（契约 §2：≤主干 1%，必须实测不报估算）。
    _fusion_params = {"freqrouter": 0, "bandlr": 0, "semproj": 0, "ldistill": 0}
    for _n, _p in model.named_parameters():
        if _n.startswith("freqrouter."):
            _fusion_params["freqrouter"] += _p.numel()
        elif _n.startswith("full_head.bandlr."):
            _fusion_params["bandlr"] += _p.numel()
        elif _n.startswith("semproj."):
            _fusion_params["semproj"] += _p.numel()
        elif _n.startswith("ldistill."):
            _fusion_params["ldistill"] += _p.numel()
    print(f"[fusion] freq_route={args.freq_route} band_head_lr={args.band_head_lr} "
          f"semantic_proj={args.semantic_proj} latent_distill={args.latent_distill} "
          f"params={_fusion_params}", flush=True)
    print(f"[a1a3] band_cond={args.band_cond} band_heads={args.band_heads} "
          f"band_high={args.band_high} "
          f"v_band_loss={args.v_band_loss:.3f} "
          f"band_cond_lr={args.band_cond_lr:.1e} band_lr={args.band_lr:.1e} "
          f"band_head_scale={args.band_head_scale:.4f}", flush=True)
    print(f"[lr] gate_lr={args.gate_lr:.3e} head_lr={args.head_lr:.3e} "
          f"head_final_lr={args.head_final_lr:.3e} "
          f"band_cond_lr={args.band_cond_lr:.3e} band_lr={args.band_lr:.3e} "
          f"freqrouter_lr=1e-5（固定）", flush=True)

    if start_step > args.until_step:
        raise SystemExit(f"[fatal] resume step={start_step} 已超过 --until-step={args.until_step}")

    total_p, train_p = count_parameters(model)
    print(f"[model] {args.model} params={total_p/1e9:.3f}B trainable={train_p/1e9:.3f}B "
          f"dual_act={getattr(model.cfg,'dual_act',None)} use_moe={getattr(model.cfg,'use_moe',None)}",
          flush=True)
    watch = UpdateWatch(model)

    # bf16 随机舍入 + 选择性 fp32 master（默认关闭，budget=0 时行为逐位不变）
    sr_master = _build_sr_master(model, args)
    _sr_src = sr_master.param_source if (sr_master is not None and sr_master.enabled) else None
    if os.environ.get("FREEZE_DUAL") == "1":
        n_fd = 0
        for _n, _p in model.named_parameters():
            if _n.startswith("dual_sem_head.") and _p.requires_grad:
                _p.requires_grad_(False)
                n_fd += _p.numel()
        print(f"[freeze-dual] {n_fd/1e6:.2f}M dual_sem_head params frozen", flush=True)
    opt, opt_gate = make_optimizers(model, args, param_source=_sr_src)
    _SAVE_OPTIM.clear()
    _SAVE_OPTIM.update({"opt": opt, "opt_gate": opt_gate})
    if resume_optim_state:
        if args.reset_optim:
            print("[resume][warn] --reset-optim：丢弃旧 optimizer_state，"
                  "A1/A3 新架构从零动量开始", flush=True)
            resume_optim_state = {}
        for name, obj in (("opt", opt), ("opt_gate", opt_gate)):
            st = resume_optim_state.get(name)
            if obj is None or not st:
                continue
            try:
                saved_names = resume_optim_names.get(name)
                if saved_names:
                    now_names = _optimizer_param_names(model, obj)
                    if saved_names != now_names:
                        # 2026-11-25：head_final 从 Adafactor 迁到 opt_gate 的
                        # 第二组，旧 ckpt 的 opt_gate 只有一组 gate，旧 opt 里
                        # 则含 head_final 组。这里只允许这一种布局迁移，丢弃
                        # 旧动量；其余任何参数顺序变化仍然 fatal。
                        layout_migrated = False
                        _mig_reason = ""
                        if name == "opt":
                            old_has_final = any(
                                "full_head.final_proj.weight" in g for g in saved_names)
                            new_has_final = any(
                                "full_head.final_proj.weight" in g for g in now_names)
                            layout_migrated = old_has_final and not new_has_final
                            _mig_reason = "head_final 迁出 main 组" if layout_migrated else ""
                            # v3.2：text_cross/sigma_adapter 等从旧 main 组
                            # 迁到 new_module AdamW 组，主优化器参数名必然变化。
                            if not layout_migrated and opt_gate is not None:
                                layout_migrated = any(
                                    g.get("new_module_group")
                                    for g in opt_gate.param_groups)
                                if layout_migrated:
                                    _mig_reason = "new_module_group 迁移"
                        elif name == "opt_gate":
                            layout_migrated = any(
                                g.get("head_final_group") or g.get("new_module_group")
                                for g in obj.param_groups)
                            if layout_migrated:
                                _mig_reason = "head_final/new_module 组迁移"
                        if layout_migrated:
                            print(f"[resume][warn] optimizer {name} 组布局变化"
                                  f"（{_mig_reason or 'head_final/new_module -> AdamW'}），"
                                  f"saved_groups={len(saved_names)} "
                                  f"now_groups={len(now_names)}，跳过旧动量", flush=True)
                            continue
                        n = min(len(saved_names), len(now_names))
                        first = next((i for i in range(n)
                                      if saved_names[i] != now_names[i]), n)
                        raise SystemExit(
                            f"[fatal] optimizer {name} 参数顺序变化，第 {first} 组不一致；"
                            "继续 load_state_dict 会让二阶矩/动量错位。"
                            "请恢复模块注册顺序，或从 step1 重训。")
                obj.load_state_dict(st)
                print(f"[resume] optimizer_state <- {name} "
                      f"groups={len(st.get('param_groups', []))} "
                      f"state={len(st.get('state', {}))}", flush=True)
            except Exception as e:
                print(f"[resume][warn] optimizer_state {name} 不兼容，"
                      f"保留新动量：{e}", flush=True)
        # 2026-11-25 修复：load_state_dict 会连同 param_groups 的 lr/weight_decay
        # 一起恢复。旧 ckpt 的 head 组 lr=0 会把 CLI 的 --head-lr 静默覆盖，
        # 导致“解冻 Head”实际仍冻结、训练轨迹与上一轮逐位相同。
        for g in opt.param_groups:
            if g.get("head_group"):
                old_lr = float(g.get("lr", 0.0))
                g["lr"] = float(args.head_lr)
                g["weight_decay"] = (0.0 if g.get("head_final_group")
                                     else float(args.head_wd))
                if old_lr != float(args.head_lr):
                    print(f"[resume] head_group lr {old_lr:.2e} -> "
                          f"{float(args.head_lr):.2e}（以 CLI 为准）", flush=True)
    else:
        print("[resume] optimizer_state 缺失（旧 ckpt）：动量从零开始，"
              "从本节点起的新 ckpt 会保存", flush=True)
    opt_audit = audit_optimizer_coverage(model, opt, opt_gate, param_source=_sr_src)
    print(f"[opt-audit] {opt_audit}", flush=True)

    # --- 2026-11-26: 可选 ds1 每批统计落盘 + 可选 q/step 探针（默认全关）---
    # 注：既有 --data-gate-report / --data-gate-report-ds1 仍是门禁分组 JSON 输入；
    # 本处每批统计落盘走独立的 --data-gate-report-dir，避免两套语义撞名。
    _report_dir = str(getattr(args, "data_gate_report_dir", "") or "").strip()
    data_reporter = DataGateBatchReporter(_report_dir) if _report_dir else None
    if data_reporter is not None:
        print(f"[data-gate-report] ds1 每批统计 -> {data_reporter.dir} "
              f"(rollup 每 {data_reporter.rollup_every} 批)", flush=True)
        import atexit as _atexit
        _atexit.register(data_reporter.close)
    _probe_dir = str(getattr(args, "probe_qlog", "") or "").strip()
    probe_log = ProbeQLog(_probe_dir) if _probe_dir else None
    if probe_log is not None:
        print(f"[probe-qlog] q/step 遥测 -> {probe_log.dir}", flush=True)
        import atexit as _atexit2
        _atexit2.register(probe_log.close)

    _f5_done = False
    _did_swap = False
    if args.freeze_spectral_after > 0 and start_step > args.freeze_spectral_after:
        n = freeze_spectral_power_iteration(model)
        print(f"[init] F5 resume 启动即冻结 {n} 处谱范数幂迭代", flush=True)
        _f5_done = True

    _buf = {"t_emb": None, "text_emb": None}
    try:
        probe = (data.get("ds1") or [None])[0]
        _B0 = 1
        _buf["t_emb"] = torch.empty(_B0, args.t_dim, device=device, dtype=dtype)
        _buf["text_emb"] = torch.empty(_B0, args.cond_dim, device=device, dtype=dtype)
        print("[opt] 条件张量已预分配", flush=True)
    except Exception as e:
        print(f"[opt] 预分配跳过：{e}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "stage_ckpts"), exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    running, best_loss, n_done, zero_running = 0.0, float("inf"), 0, 0.0
    n_applied, running_applied = 0, 0.0
    # 2026-11-23 修复：旧实现用 step+micro 取样本，导致相邻两步重复同一条素材
    # （step1 取 1,2；step2 取 2,3），2644 步只覆盖 1/3 数据池。改为全局样本游标，
    # 每 micro 严格前进 1 条，5288 个样本 = 13f 3966/1983=2.00 遍、9f 1322/700=1.89 遍。
    # 2026-11-23 修复：旧实现 resume 时把 sample_cursor 归零，导致每次续训都
    # 重新喂前缀，前面的源被反复喂、后面的源饿死。默认按全局步号估算，
    # 多次 resume 时用 --resume-sample-cursor 显式给累计游标。
    if args.resume_sample_cursor >= 0:
        sample_cursor = int(args.resume_sample_cursor)
    elif args.resume and resume_train_state.get("sample_cursor") is not None:
        sample_cursor = max(0, int(resume_train_state["sample_cursor"]))
    elif args.resume:
        sample_cursor = max(0, (start_step - 1) * max(1, args.accum))
    else:
        sample_cursor = 0
    running = float(resume_train_state.get("running", 0.0))
    best_loss = float(resume_train_state.get("best_loss", float("inf")))
    n_done = int(resume_train_state.get("n_done", 0))
    zero_running = float(resume_train_state.get("zero_running", 0.0))
    # 旧 ckpt 没有 applied 字段：以前没有 skip，n_done 就是 applied 数。
    n_applied = int(resume_train_state.get("n_applied", n_done))
    running_applied = float(resume_train_state.get("running_applied", running))
    if n_done > 0:
        print(f"[resume] train_state: cursor={sample_cursor} n_done={n_done} "
              f"n_applied={n_applied} avg={running/n_done:.5f} "
              f"opt_avg={running_applied/max(1,n_applied):.5f} "
              f"zero_avg={zero_running/n_done:.5f} best={best_loss:.5f}", flush=True)
    cycle_end = (start_step - 1 + args.cycle_steps) if args.cycle_steps > 0 else args.steps + 1
    print(f"[data] sample_cursor={sample_cursor} (start_step={start_step}, accum={args.accum})",
          flush=True)

    # 每素材最多 N 遍：全阶段硬门禁（ds1 自监督 / ds2 蒸馏 / ds3 / ds4 都管）
    sampler = None
    samplers = {}
    _max_passes = int(getattr(args, "max_sample_passes", 0) or 0)
    if _max_passes > 0:
        try:
            from .sample_budget import PassLimitedSampler, load_ledger
        except Exception:
            from sample_budget import PassLimitedSampler, load_ledger
        _assume = int(getattr(args, "assume_history_passes", 1) or 0)
        for _key in ("ds1", "ds2", "ds3", "ds4"):
            _files = data.get(_key)
            if not _files:
                continue
            if _key == "ds2":
                _lp = os.path.abspath(getattr(args, "usage_ledger", "") or
                                      "outputs/sample_usage_ledger.json")
            else:
                _lp = os.path.abspath(f"outputs/sample_usage_ledger_{_key}.json")
            _fresh = not os.path.isfile(_lp)
            _counts = load_ledger(_lp, _files, _max_passes)
            if _fresh and _assume > 0:
                # ds1/ds3/ds4 的历史遍数没有逐文件重建，按保守假设预置，避免变相放宽
                for _n in list(_counts.keys()):
                    _counts[_n] = min(_assume, _max_passes - 1)
                print(f"[data-budget] {_key} 无历史账本，按 assume_history_passes={_assume} 保守预置",
                      flush=True)
            _s = PassLimitedSampler(_files, _counts, _max_passes,
                                    seed=int(getattr(args, "shuffle_seed", 11) or 11),
                                    ledger_path=_lp, save_every=50,
                                    dataset=os.path.basename(os.path.normpath(getattr(args, _key))),
                                    save_on_init=_fresh)
            print(_s.summary(), flush=True)
            if _s.remaining_budget() <= 0:
                raise SystemExit(f"[data-budget][fatal] {_key} 起始即无可用样本（全部达上限），拒绝开训。")
            samplers[_key] = _s
        import atexit
        for _s in samplers.values():
            atexit.register(_s.save)
        sampler = samplers.get("ds2")



    if args.ema_decay > 0:
        _ema_init(model, args.ema_decay)

    print(f"[run] step {start_step}..{args.until_step}，"
          f"阶段末存档={sorted(STAGE_ENDS)}", flush=True)

    try:
        # 高σ逐样本方向平衡：每样本 cos 滑动窗口 + 跨样本难度基准。
        _hi_cos_hist = {}
        _hi_hard_hist = []
        for step in range(start_step, args.until_step + 1):
            spec = sch[step - 1]
            # LR 调度按“真正执行的 optimizer.step”计数；跳步不推进 warmup/cosine。
            # 否则长程 skip 频繁时，名义步数>有效更新步数，lr 会衰减偏快。
            lr_spec = sch[min(max(n_applied, 0), len(sch) - 1)]
            # 边训边补：每 100 步重扫目录（新素材自动进池）
            if step % 100 == 0:
                pairs = [("ds1", args.ds1), ("ds3", args.ds3), ("ds4", args.ds4)]
                if args.p2_autogen:
                    pairs.append(("ds2", args.p2_out))
                for key, path in pairs:
                    if path and key in data:
                        # ds1 若启用了 extra/holdout/shuffle，池在启动时已定死，避免边训边重排
                        if key == "ds1" and (args.ds1_extra or args.holdout > 0 or args.shuffle_seed):
                            continue
                        fresh = list_pt(path)
                        if fresh:
                            data[key] = fresh
            for g in opt.param_groups:
                if not g.get("head_group"):
                    g["lr"] = lr_spec.lr * args.lr_scale
            if opt_gate is not None:
                # 门控 lr 跟随阶段 lr：始终以 2e-4 为基准按比例缩放。
                # 否则恒定 1e-4 跑满 16000 步会把 local gate 从 0.02 推到 >1。
                gate_scale = max(0.1, lr_spec.lr / 2e-4)
                for g in opt_gate.param_groups:
                    if g.get("head_final_group"):
                        # 输出投影走独立 AdamW，固定 head_final_lr；不能被
                        # gate=0 的调度清成 0，否则 final_proj 永远不更新。
                        g["lr"] = args.head_final_lr
                    elif g.get("band_cond_group"):
                        g["lr"] = args.band_cond_lr
                    elif g.get("band_head_group"):
                        g["lr"] = args.band_lr
                    elif g.get("freqrouter_group"):
                        # 2026-11-26 修复：freqrouter 设计 lr=1e-5（构造处
                        # make_optimizers freqrouter_group），此前落入 else 被
                        # gate_lr*gate_scale 覆盖到 ~1.7e-4（17×）。独立分支固定 1e-5。
                        g["lr"] = 1e-5
                    else:
                        g["lr"] = args.gate_lr * gate_scale
            # P5 单独加大正则，结束后恢复
            want_wd = 1.5e-2 if spec.stage == "P5-anime" else args.weight_decay
            for g in opt.param_groups:
                if not g.get("head_group"):
                    g["weight_decay"] = want_wd

            opt.zero_grad(set_to_none=True)
            if opt_gate is not None:
                opt_gate.zero_grad(set_to_none=True)
            if probe_log is not None:
                probe_log.snapshot(model)
            prof_on = os.environ.get("SAKIKO_PROFILE") == "1"
            prof = {"data": 0.0, "fwd": 0.0, "aux": 0.0, "bwd": 0.0,
                    "clip": 0.0, "opt": 0.0}
            prof_step0 = time.time()
            step_loss, step_zero, step_cos = 0.0, 0.0, 0.0
            step_cos_dc = 0.0
            step_patch = 0.0
            step_anti = 0.0
            step_detail = 0.0
            step_skipreg = 0.0
            step_boundary = 0.0
            step_patchbnd = 0.0
            step_chstd = 0.0
            step_psd = 0.0
            step_frame = 0.0
            step_tokenbd = 0.0
            step_corr = 0.0
            step_corr_pen = 0.0
            step_tbdf = 0.0
            step_vmag = 0.0
            step_corr_w = 0.0
            step_rms_w = 0.0
            step_scale_reg = 0.0
            step_pre_l2 = 0.0
            step_flat = 0.0
            step_std = 0.0
            step_local = 0.0
            step_spec = 0.0
            step_temporal = 0.0
            step_x0 = 0.0
            step_x0cos = 0.0
            step_coarse = 0.0
            step_repa = 0.0
            step_syn = 0.0
            step_hi_mult = 0.0
            step_dc = 0.0
            step_stdreg = 0.0
            step_x0hp = 0.0
            step_x0f = 0.0
            step_x0f_low = 0.0
            step_x0f_mid = 0.0
            step_x0f_high = 0.0
            step_vbl = 0.0
            step_vbm = 0.0
            step_vbh = 0.0
            step_bsreg = 0.0
            step_ltv = 0.0
            step_lc = 0.0
            step_t_sum = 0.0
            step_t_n = 0
            step_t_min = float("inf")
            step_t_max = float("-inf")
            step_boosted = 0
            step_hi_gnorm = 0.0
            skip_update = False
            for micro in range(args.accum):
                prof_t0 = time.time()
                if prof_on:
                    torch.cuda.synchronize()
                sample_idx = sample_cursor
                sample_cursor += 1
                sigma = None
                _fpath = None
                if spec.mode == "distill":
                    _pick = sampler.next_file() if sampler is not None else None
                    _fpath = _pick
                    x0, t_emb, text_emb, teacher_nt, target = make_batch(
                        args, device, dtype, None, sample_idx, data["ds2"], None,
                        teacher_file=_pick)
                    # 缓存里的 9f 仍是 T=3，和 13f 混蒸时同样补齐到 T=4
                    if x0.dim() == 5 and x0.shape[2] != args.T:
                        if x0.shape[2] > args.T:
                            raise RuntimeError(f"distill latent T={x0.shape[2]} > args.T={args.T}")
                        reps = args.T - x0.shape[2]
                        x0 = torch.cat([x0, x0[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
                        if teacher_nt is not None:
                            _nt, _tt = teacher_nt
                            if _nt.dim() == 5 and _nt.shape[2] != args.T:
                                if _nt.shape[2] > args.T:
                                    raise RuntimeError(f"teacher_nt T={_nt.shape[2]} > args.T={args.T}")
                                _nt = torch.cat([_nt, _nt[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
                            teacher_nt = (_nt, _tt)
                        if target is not None and target.shape[2] != args.T:
                            if target.shape[2] > args.T:
                                raise RuntimeError(f"target T={target.shape[2]} > args.T={args.T}")
                            target = torch.cat([target, target[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
                    if args.p2_resigma:
                        # P2 教师缓存 t 分布偏高 σ；但旧实现把 cache t 整个丢掉后
                        # 均匀/Shift 重采样，导致高 σ 段的 v_teacher 也一起丢掉。
                        # 现在：--t-high-ratio>0 时，高 σ 分支优先复用 cache t + v_teacher；
                        # 缓存 t 不够高才回退到高 σ 重采样 + 精确 noise-x0。
                        _nt, _cache_t = teacher_nt
                        _B = x0.shape[0]
                        _cache_t = _cache_t.reshape(-1).float()
                        if _cache_t.numel() == 1 and _B > 1:
                            _cache_t = _cache_t.expand(_B).contiguous()
                        _u = torch.rand(_B, device=device, dtype=torch.float32)
                        if args.t_sched == "teacher":
                            _s = float(args.t_shift)
                            _t = (_s * _u) / (1.0 + (_s - 1.0) * _u)
                        else:
                            _t = _u
                        _high_ratio = float(args.t_high_ratio)
                        if (_high_ratio <= 0.0
                                and float(args.t_sigma_min) >= float(args.p2_high_min_sigma)):
                            # A0 用 --p2-resigma --t-sigma-min 0.5 时，默认就应保留
                            # 缓存 t/v_teacher，不再被“重采样”无条件丢掉。
                            _high_ratio = 1.0
                        if _high_ratio > 0.0:
                            _hi = torch.rand(_B, device=device, dtype=torch.float32) < _high_ratio
                            _cache_ok = ((_cache_t >= float(args.p2_high_min_sigma))
                                         & (_cache_t >= float(args.t_sigma_min))
                                         & (_cache_t <= float(args.t_sigma_max)))
                            _hi_cache = _hi & _cache_ok
                            _hi_t = torch.where(_cache_ok, _cache_t,
                                                0.5 + 0.5 * _u)
                            _lo_t = 0.5 * _u
                            _t = torch.where(_hi, _hi_t, _lo_t)
                            _exact = (_nt.float() - x0.float()).to(dtype)
                            _mask = _hi_cache.reshape(_B, *([1] * (target.dim() - 1)))
                            target = torch.where(_mask, target, _exact)
                        else:
                            target = (_nt.float() - x0.float()).to(dtype)
                        _t.clamp_(min=float(args.t_sigma_min),
                                  max=float(args.t_sigma_max))
                        t_new = _t.to(dtype)
                        t_emb = sinusoidal_timestep_embedding(t_new, args.t_dim).to(dtype)
                        teacher_nt = (_nt, t_new)
                else:
                    _self_pick = samplers[spec.ds].next_file() if spec.ds in samplers else None
                    _fpath = _self_pick
                    if _fpath is None and spec.ds in data and data[spec.ds]:
                        try:
                            _seq = data[spec.ds]
                            _fpath = _seq[sample_idx % len(_seq)]
                        except Exception:
                            _fpath = None
                    x0, _, _, teacher_nt, target = make_batch(
                        args, device, dtype, ([_self_pick] if _self_pick else data[spec.ds]),
                        sample_idx, None, _buf)
                    # 2026-11-23 性能修复：9f 的 T=3 与 13f 的 T=4 交替会让 Triton 每步重编译，
                    # 实测 16s/step。统一把 9f 末帧复制补齐到 T=4，形状固定后再训。
                    if x0.dim() == 5 and x0.shape[2] != args.T:
                        if x0.shape[2] > args.T:
                            raise RuntimeError(f"latent T={x0.shape[2]} > args.T={args.T}")
                        reps = args.T - x0.shape[2]
                        x0 = torch.cat([x0, x0[:, :, -1:].repeat(1, 1, reps, 1, 1)], dim=2)
                    _nb = _buf.get("noise")
                    if _nb is None or _nb.shape != x0.shape or _nb.dtype != dtype:
                        _nb = torch.empty_like(x0)
                        _buf["noise"] = _nb
                    torch.randn(x0.shape, device=device, dtype=dtype, out=_nb)
                    noise = _nb
                    _tb = _buf.get("t")
                    if _tb is None or _tb.shape != (x0.shape[0],) or _tb.dtype != torch.float32:
                        _tb = torch.empty(x0.shape[0], device=device, dtype=torch.float32)
                        _buf["t"] = _tb
                    _u = _buf.get("u")
                    if _u is None or _u.shape != (x0.shape[0],):
                        _u = torch.empty(x0.shape[0], device=device, dtype=torch.float32)
                        _buf["u"] = _u
                    torch.rand(x0.shape[0], device=device, dtype=torch.float32, out=_u)
                    if args.t_sched == "teacher":
                        s = float(args.t_shift)
                        # sigma = s·u/(1+(s-1)·u)，与 comfy time_snr_shift 完全一致
                        _tb.copy_((s * _u) / (1.0 + (s - 1.0) * _u))
                    else:
                        _tb.copy_(_u)
                    if args.t_high_ratio > 0.0:
                        # 混合采样：高噪段 [0.5,1] 占比 p，其余低噪段 [0,0.5)。
                        # 用于验证“高 σ 结构先学”的课程，不改模型/显存。
                        _hi = torch.rand_like(_u) < float(args.t_high_ratio)
                        _tb.copy_(torch.where(_hi, 0.5 + 0.5 * _u, 0.5 * _u))
                    if args.t_sigma_min > 0.0 or args.t_sigma_max < 1.0:
                        _tb.clamp_(min=float(args.t_sigma_min),
                                   max=float(args.t_sigma_max))
                    t = _tb.to(dtype)
                    target = noise - x0
                    teacher_nt = (noise, t)
                noise, t = teacher_nt
                text_tokens = None
                text_mask = None
                _text_valid = None
                if getattr(args, "text_token_dir", "") and _fpath:
                    text_tokens, text_mask = load_text_tokens(
                        args.text_token_dir, os.path.basename(_fpath),
                        device, dtype, token_dim=int(args.text_cross_text_dim))
                # CFG 训练：10% 样本走 null 分支。null 必须是 zero token +
                # 全 True mask；全 False mask 会让 SDPA 产生 NaN。
                if (text_tokens is not None
                        and float(getattr(args, "text_dropout", 0.0) or 0.0) > 0.0):
                    _p = float(args.text_dropout)
                    _drop = torch.rand(x0.shape[0], device=device) < _p
                    if bool(_drop.any()):
                        text_tokens = text_tokens.clone()
                        text_mask = text_mask.clone()
                        text_tokens[_drop] = 0.0
                        text_mask[_drop] = True
                    _text_valid = (~_drop).to(dtype).view(-1, 1)
                if _text_valid is None:
                    _text_valid = torch.ones(x0.shape[0], 1, device=device, dtype=dtype)
                if data_reporter is not None and spec.ds == "ds1":
                    try:
                        data_reporter.record(step, micro, spec.ds, _fpath, t, x0)
                    except Exception as _rep_e:
                        print(f"[data-gate-report][warn] step {step} micro {micro} "
                              f"统计失败: {_rep_e}", flush=True)
                if args.cond_mode == "fixed":
                    # 修复（2026-11-23，两条分支共用）：
                    # 1) 真实 t -> 正弦时间步嵌入（内部乘 1000 对齐教师）；
                    # 2) distill + --text-emb-dir 时保留 make_batch 读到的真文本；
                    #    只有没给文本缓存才回退 zeros（A0 不允许静默回退）。
                    t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
                    _keep_text = bool(getattr(args, "text_emb_dir", "")) and spec.mode == "distill"
                    if _keep_text:
                        if text_emb is None or float(text_emb.abs().max()) == 0.0:
                            raise RuntimeError("text_emb_dir 已设置但本步 text_emb 为零/为空")
                    else:
                        text_emb = torch.zeros(x0.shape[0], args.cond_dim,
                                               device=device, dtype=dtype)
                    sigma = t.float().view(-1, 1).to(dtype)
                tv = t.view(-1, 1, 1, 1, 1)
                xt = (1 - tv) * x0 + tv * noise
                _rollout_x0_pred0 = None
                if args.rollout2:
                    # 轻量 2 步 rollout：σ=1 只做无梯度前向生成中间 x0，
                    # 再以正确的 flow 插值 (1-t)*x0_pred + t*noise 到 σ≈0.8，
                    # 最后一步才反传。显存只多一次无梯度前向，不加反传图。
                    if sigma is None:
                        raise RuntimeError("--rollout2 需要 cond-mode=fixed 且 sigma 非空")
                    _rB = x0.shape[0]
                    _r_t1 = torch.ones(_rB, device=device, dtype=torch.float32)
                    t = _r_t1.to(dtype)
                    t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
                    sigma = t.float().view(-1, 1).to(dtype)
                    tv = t.view(-1, 1, 1, 1, 1)
                    xt = noise  # σ=1：纯噪声起步
                    _was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        pred0 = model(xt, t_emb, text_emb, return_aux=False,
                                      sigma=sigma, backbone_no_grad=True,
                                      text_tokens=text_tokens, text_mask=text_mask,
                                      text_valid=_text_valid)
                        x0_pred0 = (xt.float() - tv.float() * pred0.float()).detach()
                    del pred0
                    model.train(_was_training)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _r_t = torch.full((_rB,), float(args.rollout2_mid_sigma),
                                      device=device, dtype=torch.float32)
                    t = _r_t.to(dtype)
                    t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
                    sigma = t.float().view(-1, 1).to(dtype)
                    tv = t.view(-1, 1, 1, 1, 1)
                    noise = torch.randn_like(x0)
                    xt = ((1.0 - tv.float()) * x0_pred0.float()
                          + tv.float() * noise.float()).to(dtype)
                    target = (noise.float() - x0.float()).to(dtype)
                    _rollout_x0_pred0 = x0_pred0
                    if os.environ.get("SAKIKO_MEM_TRACE") == "1":
                        print(f"[memtrace] step={step} micro={micro} rollout2 "
                              f"mid={float(_r_t.mean().detach()):.4f} "
                              f"x0_pred_std={float(x0_pred0.std().detach()):.4f}", flush=True)
                elif args.self_renoise:
                    # exposure-bias fix：第一步只做无梯度前向拿当前模型的 x0 预测，
                    # 再把 x0_pred 用新噪声加回 --sr-sigma 区间训练第二步。
                    # 训练输入不再是纯教师缓存轨迹，而是模型自己的错误分布。
                    if sigma is None:
                        raise RuntimeError("--self-renoise 需要 cond-mode=fixed 且 sigma 非空")
                    _was_training = model.training
                    model.eval()  # 预前向关掉 full_head 的 training 辅助分支，降峰值
                    with torch.no_grad():
                        pred0 = model(xt, t_emb, text_emb, return_aux=False,
                                      sigma=sigma, backbone_no_grad=True,
                                      text_tokens=text_tokens, text_mask=text_mask,
                                      text_valid=_text_valid)
                        x0_pred0 = (xt.float() - tv.float() * pred0.float()).detach()
                    del pred0
                    model.train(_was_training)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _sr_B = x0.shape[0]
                    _sr_u = torch.rand(_sr_B, device=device, dtype=torch.float32)
                    _sr_t = _sr_u.mul_(float(args.sr_sigma_max - args.sr_sigma_min))
                    _sr_t.add_(float(args.sr_sigma_min))
                    t = _sr_t.to(dtype)
                    t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
                    sigma = t.float().view(-1, 1).to(dtype)
                    tv = t.view(-1, 1, 1, 1, 1)
                    noise = torch.randn_like(x0)
                    # 2026-11-26 修复：原实现是 x0_pred + t*noise，漏了 (1-t)，
                    # 在 σ=0.5 时实际把 x0 幅值放大到 1.5 倍；正确插值是
                    # (1-t)*x0_pred + t*noise。
                    xt = ((1.0 - tv.float()) * x0_pred0.float()
                          + tv.float() * noise.float()).to(dtype)
                    target = (noise.float() - x0.float()).to(dtype)
                    if os.environ.get("SAKIKO_MEM_TRACE") == "1":
                        print(f"[memtrace] step={step} micro={micro} self_renoise "
                              f"sigma_mean={float(_sr_t.mean().detach()):.4f} "
                              f"x0_pred_std={float(x0_pred0.std().detach()):.4f}", flush=True)
                if prof_on:
                    torch.cuda.synchronize()
                    prof["data"] += time.time() - prof_t0
                    prof_t0 = time.time()
                # 记录本 micro 实际生效的 σ（rollout/self-renoise 重采样后的最终 t），
                # 供日志统计 mean/min/max 证明高/低 σ 区间真的落到训练步上。
                _t_flat = t.detach().float().reshape(-1)
                step_t_sum += float(_t_flat.mean())
                step_t_min = min(step_t_min, float(_t_flat.min()))
                step_t_max = max(step_t_max, float(_t_flat.max()))
                step_t_n += 1
                # 双流 Head 的幅值目标：按当前 batch 真实 target v=noise-x0 的
                # per-sample RMS 动态给，废除固定 0.06+0.22*sigmoid 曲线。
                _fh = getattr(model, "full_head", None)
                if _fh is not None:
                    _trms = target.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
                    _fh._rms_target_ext = _trms.detach().repeat_interleave(x0.shape[2])
                _ms_feats = ([] if args.multi_scale_align_weight > 0 else None)
                pred = model(xt, t_emb, text_emb, return_aux=False, sigma=sigma,
                             backbone_no_grad=bool(args.freeze_backbone or args.train_mod_only),
                             text_tokens=text_tokens, text_mask=text_mask,
                             text_valid=_text_valid,
                             feature_pool=_ms_feats)
                # 关键：主前向的 aux 属性必须立刻快照。后面 flow-consist /
                # self-renoise 等无梯度二次前向会进 eval 并把 model 上的
                # _last_full_head_aux / _last_token_bd / _last_vmag_loss 覆盖成
                # None，导致日志 faux/tbd/vmag 恒 0、对应 loss 根本没加。
                _main_aux = {
                    "faux": getattr(model, "_last_full_head_aux", None),
                    "tbd": getattr(model, "_last_token_bd", None),
                    "corr": getattr(model, "_last_corr_defect", None),
                    "vmag": getattr(model, "_last_vmag_loss", None),
                    "rms": getattr(model, "_last_out_rms", None),
                    "rtgt": getattr(model, "_last_rms_target", None),
                    "syntax": getattr(model, "_last_syntax_aux", None),
                }
                # 历史日志里 faux/tbd/vmag 曾整段为 0：不是“损失没启用”，而是
                # 主前向后又被无梯度二次前向覆盖。现在快照已拿走；若仍然 None，
                # 至少打一次明确 warn，避免再次静默跳过。
                for _aux_key, _aux_w in (("faux", args.frame_aux),
                                         ("tbd", args.token_bd_loss),
                                         ("vmag", args.vmag_loss),
                                         ("syntax", args.syntax_aux_weight)):
                    if _aux_w > 0 and _main_aux[_aux_key] is None:
                        _tag = f"missing_{_aux_key}"
                        if _tag not in _AUX_WARNED:
                            _AUX_WARNED.add(_tag)
                            print(f"[aux][warn] {_aux_key} 权重={_aux_w:g} 但主前向未产出；"
                                  f"该项本步不会进 loss（检查模块/帧数/开关）", flush=True)
                if os.environ.get("SAKIKO_MEM_TRACE") == "1":
                    print(f"[memtrace] step={step} micro={micro} after_fwd "
                          f"alloc={torch.cuda.memory_allocated()/2**30:.3f}GB "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.3f}GB "
                          f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GB", flush=True)
                if prof_on:
                    torch.cuda.synchronize()
                    prof["fwd"] += time.time() - prof_t0
                    prof_t0 = time.time()
                _pred_raw = pred.float()
                if args.predict_x0:
                    # x0 参数化：网络输出就是干净 latent。v 用 (xt-x0)/σ 反推，
                    # 高 σ 下不再允许“输出噪声”冒充速度场。
                    x0hat = _pred_raw
                    _sig_safe = tv.float().clamp_min(1e-3)
                    v_pred = (xt.float() - x0hat) / _sig_safe
                else:
                    x0hat = xt.float() - tv.float() * _pred_raw
                    v_pred = _pred_raw
                # v3 hybrid loss：高 σ 偏 v，低 σ 偏 x0；单输出仍是 x0/或纯 v。
                # 这不是“双头参数化”，只是逐样本损失权重，数学上可落地。
                _sig_flat = (sigma.detach().float().reshape(-1)
                             if sigma is not None else torch.ones(
                                 x0.shape[0], device=x0.device))
                _hi_mask = _sig_flat >= float(
                    args.x0_hi_balance_threshold
                    if args.x0_hi_cos_balance > 0
                    else args.x0_high_sigma_threshold)
                if args.hybrid_loss:
                    # 2026-11-27 修复：x0 参数化下 v_pred=(xt-x0hat)/σ，
                    # v_se 自带 1/σ²；v 权重必须用 σ² 抵消，否则 σ=0.001
                    # 时实际梯度放大 1000 倍，会把训练炸出 loss=53.8/gnorm=703。
                    # 非 x0 参数化时 v_pred 无 1/σ，仍按 σ 加权。
                    if args.predict_x0:
                        _hyb_v_w = _sig_flat.clamp(0.0, 1.0).pow(2)
                    else:
                        _hyb_v_w = _sig_flat.clamp(0.0, 1.0)
                    _hyb_x0_w = (1.0 - _sig_flat).clamp(0.0, 1.0)
                    if args.x0_high_sigma_weight > 0:
                        # 只抬高 σ≥threshold 的 x0 权重下限，不动低 σ。
                        _hi_mask = _sig_flat >= float(args.x0_high_sigma_threshold)
                        _hi_floor = torch.full_like(
                            _hyb_x0_w, float(args.x0_high_sigma_weight))
                        _hyb_x0_w = torch.where(
                            _hi_mask, torch.maximum(_hyb_x0_w, _hi_floor),
                            _hyb_x0_w)
                    if args.x0_hi_cos_balance > 0:
                        # 逐样本高σ方向平衡：每个样本维护 cos 滑动窗口，
                        # 难度 = 1 - mean(window cos)；只放大比全局难度更难的样本。
                        # 只在 σ≥x0_hi_balance_threshold（默认0.8）生效。
                        _hi_mask = _sig_flat >= float(args.x0_hi_balance_threshold)
                        if bool(_hi_mask.any()):
                            _cos_s = torch.nn.functional.cosine_similarity(
                                x0hat.flatten(1).float(),
                                x0.flatten(1).float(), dim=1).detach()
                            _hard = (1.0 - _cos_s).clamp_min(0.0)
                            _mult = torch.ones_like(_hyb_x0_w)
                            _valid_hi = 0
                            for _bi in range(int(_sig_flat.shape[0])):
                                if float(_sig_flat[_bi]) < float(args.x0_hi_balance_threshold):
                                    continue
                                _key = f"s{int(sample_idx)}_{_bi}"
                                _hist = _hi_cos_hist.get(_key)
                                if _hist is None:
                                    _hist = []
                                    _hi_cos_hist[_key] = _hist
                                _hist.append(float(_cos_s[_bi]))
                                if len(_hist) > int(args.x0_hi_balance_window):
                                    del _hist[0]
                                _hard_ema = 1.0 - (sum(_hist) / len(_hist))
                                _hi_hard_hist.append(float(_hard_ema))
                                if len(_hi_hard_hist) > 200:
                                    del _hi_hard_hist[0]
                                _valid_hi += 1
                            if _valid_hi > 0 and _hi_hard_hist:
                                _g_hard = max(sum(_hi_hard_hist) / len(_hi_hard_hist), 1e-6)
                                for _bi in range(int(_sig_flat.shape[0])):
                                    if float(_sig_flat[_bi]) < float(args.x0_hi_balance_threshold):
                                        continue
                                    _hist = _hi_cos_hist.get(f"s{int(sample_idx)}_{_bi}", [])
                                    if not _hist:
                                        continue
                                    _hard_ema = 1.0 - (sum(_hist) / len(_hist))
                                    _mult[_bi] = torch.tensor(
                                        (_hard_ema / _g_hard) ** float(
                                            args.x0_hi_cos_balance_power),
                                        device=_mult.device, dtype=_mult.dtype
                                    ).clamp(float(args.x0_hi_balance_min),
                                            float(args.x0_hi_balance_max))
                                _hyb_x0_w = torch.where(
                                    _hi_mask, _hyb_x0_w * _mult, _hyb_x0_w)
                                step_hi_mult += float(_mult[_hi_mask].mean().detach())
                    else:
                        _hi_mask = _sig_flat >= float(args.x0_high_sigma_threshold)
                else:
                    _hyb_v_w = torch.ones_like(_sig_flat)
                    _hyb_x0_w = torch.ones_like(_sig_flat)
                _v_se = (v_pred - target.float()).pow(2).mean(dim=(1, 2, 3, 4))
                if args.hybrid_loss:
                    loss = args.v_loss_weight * (_hyb_v_w * _v_se).mean()
                else:
                    loss = args.v_loss_weight * torch.nn.functional.mse_loss(
                        v_pred, target.float())
                # 高σ流向一致性：同一 batch 的 x0hat(σ) 与低σ分支 x0hat(σ*ratio)
                # 方向一致。低σ分支用 no_grad，省显存；只有高σ分支回传。
                if (args.flow_consist_weight > 0
                        and step % max(1, args.flow_consist_every) == 0
                        and float(_sig_flat.mean()) >= args.flow_consist_sigma_threshold):
                    _s_low = (_sig_flat * float(args.flow_consist_ratio)).clamp(1e-3, 1.0)
                    _tv_low = _s_low.to(dtype).view(-1, 1, 1, 1, 1)
                    _xt_low = ((1.0 - _tv_low.float()) * x0.float()
                               + _tv_low.float() * noise.float()).to(dtype)
                    _t_low = _s_low.to(dtype)
                    _t_emb_low = sinusoidal_timestep_embedding(
                        _t_low, args.t_dim).to(dtype)
                    _was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        _pred_low = model(
                            _xt_low, _t_emb_low, text_emb, return_aux=False,
                            sigma=_s_low.view(-1, 1).to(dtype),
                            backbone_no_grad=True,
                            text_tokens=text_tokens, text_mask=text_mask,
                            text_valid=_text_valid)
                        if args.predict_x0:
                            _x0_low = _pred_low.float()
                        else:
                            _x0_low = (_xt_low.float()
                                       - _tv_low.float() * _pred_low.float())
                    model.train(_was_training)
                    _fc = torch.nn.functional.cosine_similarity(
                        x0hat.flatten(1), _x0_low.detach().flatten(1), dim=1)
                    loss = loss + args.flow_consist_weight * (1.0 - _fc).mean()
                # 多尺度文本特征对齐：encoder/bottleneck/decoder 池化特征与
                # 文本 token 均值方向对齐。text_cls detach，只训视频侧特征。
                if (args.multi_scale_align_weight > 0 and _ms_feats
                        and len(_ms_feats) >= 3 and text_tokens is not None):
                    if text_mask is None:
                        _tcls = text_tokens.float().mean(dim=1)
                    else:
                        _m = text_mask.to(device=text_tokens.device,
                                          dtype=torch.float32)
                        if _m.ndim == 1:
                            _m = _m.unsqueeze(0)
                        _w = _m.unsqueeze(-1)
                        _tcls = ((text_tokens.float() * _w).sum(dim=1)
                                 / _w.sum(dim=1).clamp_min(1.0))
                    _tcls = _tcls.detach()
                    _aws = (float(args.align_w_shallow),
                            float(args.align_w_middle),
                            float(args.align_w_deep))
                    _align = loss.new_zeros(())
                    _wv = _text_valid.squeeze(1).float()
                    for _fi, _aw in zip(_ms_feats[:3], _aws):
                        if _aw <= 0:
                            continue
                        _fv = _fi.float()
                        _cos = torch.nn.functional.cosine_similarity(
                            _fv, _tcls, dim=1)
                        # CFG null 样本的 text token 全零，不能把特征往零向量对。
                        _align = _align + _aw * (
                            ((1.0 - _cos) * _wv).sum()
                            / _wv.sum().clamp_min(1.0))
                    loss = loss + args.multi_scale_align_weight * _align
                if args.repa_weight > 0:
                    # REPA：消费模型独立 tap 槽 _last_repa_feature（不进 feature_pool），
                    # 经 repa_proj 投影到冻结 DINOv2 教师空间算 1-cos。学生侧留在反挂图内，
                    # 教师缓存侧 detach；只挂主 loss，不进 _x0_loss_hi 双回传路径。
                    _repa_tap = getattr(model, "_last_repa_feature", None)
                    _repa_proj = getattr(model, "repa_proj", None)
                    if _repa_tap is None or _repa_proj is None:
                        if "repa_tap_none" not in _REPA_WARNED:
                            _REPA_WARNED.add("repa_tap_none")
                            print(f"[repa][warn] _last_repa_feature/repa_proj 为 None "
                                  f"（repa_dim={getattr(model.cfg, 'repa_dim', 0)}），"
                                  "本步 REPA 跳过", flush=True)
                    else:
                        try:
                            _repa_path = _repa_target_path(args.repa_cache_dir, _fpath)
                            if not _repa_path or not os.path.isfile(_repa_path):
                                raise FileNotFoundError(
                                    _repa_path or "<empty repa_cache_dir>")
                            _repa_tgt = _load_repa_target(
                                _repa_path, device, _repa_proj.out_features)
                            _stu = _repa_proj(_repa_tap)  # [B,N,D]，学生侧不 detach
                            _Br, _Nr, _Dr = _stu.shape
                            _tg = _repa_tgt
                            if _tg.shape[0] == 1 and _Br > 1:
                                _tg = _tg.expand(_Br, -1, -1)
                            if _tg.shape[0] != _Br:
                                _tg = _tg[0:1].expand(_Br, -1, -1)
                            _Pt = int(_tg.shape[1])
                            if _Nr == _Pt:
                                _s_feat = _stu.float()
                                _t_feat = _tg.float()
                            else:
                                pt, ph, pw = model.cfg.patch_size
                                _T0, _H0, _W0 = x0.shape[2], x0.shape[3], x0.shape[4]
                                _Hg, _Wg = _H0 // int(ph), _W0 // int(pw)
                                if _Nr == _T0 * _Hg * _Wg:
                                    _g = _stu.view(_Br, _T0, _Hg, _Wg, _Dr).permute(
                                        0, 1, 4, 2, 3)
                                    _g = _g.reshape(_Br * _T0, _Dr, _Hg, _Wg)
                                    _gs = int(round(_Pt ** 0.5))
                                    _g = torch.nn.functional.interpolate(
                                        _g, size=(_gs, _gs), mode="bilinear",
                                        align_corners=False)
                                    _s_feat = (_g.reshape(_Br, _T0, _Dr, _gs, _gs)
                                               .permute(0, 1, 3, 4, 2)
                                               .reshape(_Br, _T0 * _gs * _gs, _Dr).float())
                                    if _tg.shape[1] != _s_feat.shape[1]:
                                        _tg2 = torch.nn.functional.interpolate(
                                            _tg.permute(0, 2, 1),
                                            size=_s_feat.shape[1], mode="linear",
                                            align_corners=False)
                                        _tg = _tg2.permute(0, 2, 1)
                                    _t_feat = _tg.float()
                                else:
                                    _s_feat = _stu.float().mean(dim=1)
                                    _t_feat = _tg.float().mean(dim=1)
                            _t_feat = _t_feat.detach()
                            _cos = torch.nn.functional.cosine_similarity(
                                _s_feat, _t_feat, dim=-1, eps=1e-6).clamp(-1, 1)
                            _repa_loss = (1.0 - _cos).mean()
                            _warm = max(0, int(args.repa_warmup))
                            _dec = max(1, int(args.repa_decay))
                            if step < _warm:
                                _repa_scale = (float(step) / _warm) if _warm > 0 else 1.0
                            elif step < _warm + _dec:
                                _repa_scale = 0.5 * (1.0 + math.cos(
                                    math.pi * (step - _warm) / _dec))
                            else:
                                _repa_scale = 0.0
                            loss = (loss + float(args.repa_weight)
                                    * _repa_scale * _repa_loss)
                            step_repa += float(_repa_loss.detach())
                        except Exception as _repa_e:
                            if "repa_load" not in _REPA_WARNED:
                                _REPA_WARNED.add("repa_load")
                                print(f"[repa][warn] 目标特征加载/对齐失败，本步跳过 REPA: "
                                      f"{type(_repa_e).__name__}: {_repa_e}", flush=True)
                if args.low_tv_loss > 0:
                    # 低频结构约束：对 low band 的 IFFT 结果算相邻像素 L1。
                    # 目标不是单纯压平，而是用 1e-4~1e-3 的小权重抑制
                    # “把低频摊成纯色块/剧烈跳变”两种偷懒解。
                    _low_tv = split_frequency_bands(x0hat)[0]
                    # _low_tv 是 5D [B,C,T,H,W]：H 差分切倒数第二维，W 差分切最后一维。
                    _tv_h = (_low_tv[..., 1:, :] - _low_tv[..., :-1, :]).abs().mean()
                    _tv_w = (_low_tv[..., :, 1:] - _low_tv[..., :, :-1]).abs().mean()
                    _tv = 0.5 * (_tv_h + _tv_w)
                    loss = loss + args.low_tv_loss * _tv
                    step_ltv += float(_tv.detach())
                if (args.rollout2_low_consistency > 0
                        and _rollout_x0_pred0 is not None):
                    # 两步 rollout 的低频结构一致性：第二部 x0hat 的 low band
                    # 必须贴住第一步自己生成的 low band，防止大轮廓在去噪中跳变。
                    _lc_a = split_frequency_bands(x0hat)[0]
                    _lc_b = split_frequency_bands(_rollout_x0_pred0)[0].detach()
                    _lc_rms = _lc_b.float().pow(2).mean(
                        dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
                    _lc = torch.nn.functional.mse_loss(
                        _lc_a.float() / _lc_rms, _lc_b.float() / _lc_rms)
                    loss = loss + args.rollout2_low_consistency * _lc
                    step_lc += float(_lc.detach())
                _sig_w = 1.0
                if args.x0_sigma_weight or args.x0_sigma_power != 1.0:
                    # 2026-11-27 修复：必须逐样本按 σ 加权，不能取 batch mean。
                    # 旧实现用 mean(σ)^p，σ=0.98 的样本实际只拿到 batch 平均权重，
                    # 高σ x0 梯度被低σ样本稀释，这是高σ方向学不出来的直接原因。
                    # 现在 _sig_w 形状 [B]，高σ样本拿自己的 σ^p，低σ不被抬高。
                    _sig_w = _sig_flat.clamp(0.1, 1.0).pow(
                        float(args.x0_sigma_power))
                _x0_w = _sig_w * _hyb_x0_w
                _x0_ref = x0.float()
                if (args.x0_target == "teacher" and teacher_nt is not None
                        and target is not None):
                    # 真蒸馏：教师在该 t 的去噪估计 noise - v_teacher。
                    # 它是带布局的软目标（实测 cos≈0.59 vs 真实 x0），
                    # 比精确 noise-x0 更能把“画面结构”传给学生。
                    _x0_ref = (teacher_nt[0].float() - target.float()).clamp(-3.0, 3.0)
                _x0_loss_hi = None
                if args.x0_loss > 0:
                    _x0_se = (x0hat - _x0_ref).pow(2).mean(dim=(1, 2, 3, 4))
                    x0_mse = _x0_se.mean()
                    _x0_loss_all = args.x0_loss * (_x0_w * _x0_se).mean()
                    loss = loss + _x0_loss_all
                    if bool(_hi_mask.any()):
                        _x0_loss_hi = args.x0_loss * (
                            (_x0_w * _x0_se)[_hi_mask]).mean()
                    step_x0 += float(x0_mse.detach())
                if args.x0_coarse_weight > 0:
                    # NCR 粗粒度布局：高σ 先把 2×2 均值池化后的全局布局对齐，
                    # 不新增 coarse 参数头；必须复用 _x0_ref，不能改回 real x0。
                    _coarse_mask = _sig_flat >= float(args.x0_coarse_sigma)
                    if bool(_coarse_mask.any()):
                        _coarse_head = getattr(model, "coarse_x0_head", None)
                        if _coarse_head is not None:
                            _xp = _coarse_head(x0hat.float())
                            _rp = _coarse_head(_x0_ref.float())
                        else:
                            _xp = torch.nn.functional.avg_pool3d(
                                x0hat.float(), kernel_size=(1, 2, 2),
                                stride=(1, 2, 2))
                            _rp = torch.nn.functional.avg_pool3d(
                                _x0_ref.float(), kernel_size=(1, 2, 2),
                                stride=(1, 2, 2))
                        _coarse_se = (_xp - _rp).pow(2).mean(
                            dim=(1, 2, 3, 4))
                        _coarse_loss = args.x0_coarse_weight * (
                            (_x0_w * _coarse_se)[_coarse_mask]).mean()
                        loss = loss + _coarse_loss
                        step_coarse += float(_coarse_loss.detach())
                if args.x0_dc_loss > 0:
                    # DC 对齐（不是压零）：x0hat 的每通道(T,H,W)均值必须等于真实 x0 的
                    # 同口径均值。真实 latent 本身允许有正常偏色（每通道均值可达 ±0.4），
                    # 压零会把颜色洗掉；这里只罚“偏离真实 DC”的部分。
                    _dc = x0hat.mean(dim=(2, 3, 4))
                    _dc_t = _x0_ref.mean(dim=(2, 3, 4)).detach()
                    dc_reg = torch.nn.functional.mse_loss(_dc, _dc_t)
                    loss = loss + args.x0_dc_loss * dc_reg
                    step_dc += float(dc_reg.detach())
                if args.x0_std_loss > 0:
                    # 幅值正则：每样本每通道 std 对齐真实 x0 的 std（真实侧 detach）。
                    _sd = x0hat.std(dim=(2, 3, 4), unbiased=False)
                    _sd_t = _x0_ref.std(dim=(2, 3, 4), unbiased=False).detach()
                    std_reg = torch.nn.functional.mse_loss(_sd, _sd_t)
                    loss = loss + args.x0_std_loss * std_reg
                    step_stdreg += float(std_reg.detach())
                if args.x0_cos_high_weight > 0:
                    # 高 σ 方向监督：MSE 可被幅值/DC 项主导，cos 直接罚方向。
                    _cos_x0 = torch.nn.functional.cosine_similarity(
                        x0hat.flatten(1), _x0_ref.flatten(1), dim=1, eps=1e-6)
                    _cos_mask = _sig_flat >= float(args.x0_cos_high_threshold)
                    if bool(_cos_mask.any()):
                        cos_high = (1.0 - _cos_x0[_cos_mask]).mean()
                        loss = loss + args.x0_cos_high_weight * cos_high
                        step_x0cos += float(cos_high.detach())
                if args.x0_hp_loss > 0:
                    # 只盯 2x2 patch 内高频，专门防 x0hat 退化成 32px 色块。
                    _B0, _C0, _T0, _H0, _W0 = x0hat.shape
                    _ph = x0hat.reshape(_B0, _C0, _T0, _H0 // 2, 2, _W0 // 2, 2)
                    _xh = _ph - _ph.mean(dim=(4, 6), keepdim=True)
                    _x0r = _x0_ref.reshape(_B0, _C0, _T0, _H0 // 2, 2, _W0 // 2, 2)
                    _x0h = _x0r - _x0r.mean(dim=(4, 6), keepdim=True)
                    hp_mse = torch.nn.functional.mse_loss(_xh, _x0h.detach())
                    loss = loss + args.x0_hp_loss * _sig_w * hp_mse
                    step_x0hp += float(hp_mse.detach())
                if args.x0_fourier_loss > 0:
                    # 只作用于 x0hat/x0，不使用 v；T 维保持独立，只在 H/W 做 rfft2。
                    # 权重由函数内部强制为 [0.1,0.2]，此处再传一次以便日志可追溯。
                    x0f_loss = x0_fourier_band_loss(
                        x0hat, _x0_ref, weight=float(args.x0_fourier_loss),
                        sigma=sigma, sigma_weighted=bool(args.fourier_sigma_weight))
                    loss = loss + x0f_loss
                    step_x0f += float(x0f_loss.detach())
                    with torch.no_grad():
                        _br = x0_band_energy_ratios(x0hat.detach()).mean(dim=0)
                    step_x0f_low += float(_br[0])
                    step_x0f_mid += float(_br[1])
                    step_x0f_high += float(_br[2])
                if args.band_heads and args.v_band_loss > 0:
                    # A3：监督 v_teacher=noise-x0 的三个频带残差。
                    # base 是未加三头前的完整 v（含高流/低频支路），
                    # 所以 target_res = v_teacher_band - base_band，三头之和精确补到 teacher。
                    _bh = getattr(model, "_last_band_heads_latent", None)
                    _bb = getattr(model, "_last_band_base_v_latent", None)
                    if _bh is None or _bb is None:
                        raise RuntimeError(
                            "--band-heads 已开，但模型没有返回 _last_band_heads_latent；"
                            "检查 FullGatedHead.enable_band_heads 与 UViT3D unpatchify")
                    _vt_l, _vt_m, _vt_h = split_frequency_bands(target.detach())
                    _bb_l, _bb_m, _bb_h = split_frequency_bands(_bb)

                    def _band_mse(pred_b, target_b):
                        rms = target_b.float().pow(2).mean(
                            dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
                        return torch.nn.functional.mse_loss(
                            pred_b.float() / rms, target_b.float() / rms)

                    _bl = _band_mse(_bh[0], _vt_l - _bb_l)
                    _bm = _band_mse(_bh[1], _vt_m - _bb_m)
                    _bhh = _band_mse(_bh[2], _vt_h - _bb_h)
                    _vbl = args.v_band_loss * float(args.v_band_w_low) * _bl
                    _vbm = args.v_band_loss * float(args.v_band_w_mid) * _bm
                    _vbh = args.v_band_loss * float(args.v_band_w_high) * _bhh
                    loss = loss + _vbl + _vbm + _vbh
                    step_vbl += float(_bl.detach())
                    step_vbm += float(_bm.detach())
                    step_vbh += float(_bhh.detach())
                if args.band_head_lr and args.v_band_loss > 0:
                    # bandlr：low 是软直接预测（final_low = base_low + g*(head_low-base_low)），
                    # mid/high 是残差。监督对象仍是 v_teacher=noise-x0 的三频带残差；
                    # base 用 bandlr 叠加前的 v，low 对 base 做 detach，避免重复训主干低频。
                    _blh = getattr(model, "_last_bandlr_heads_latent", None)
                    _blb = getattr(model, "_last_bandlr_base_bands_latent", None)
                    if _blh is None or _blb is None:
                        raise RuntimeError(
                            "--band-head-lr 已开，但模型没有返回 "
                            "_last_bandlr_heads_latent/_last_bandlr_base_bands_latent；"
                            "检查 UViT3D 的 bandlr 接线")
                    _vt_l2, _vt_m2, _vt_h2 = split_frequency_bands(target.detach())

                    def _bl_mse(pred_b, target_b):
                        rms = target_b.float().pow(2).mean(
                            dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
                        return torch.nn.functional.mse_loss(
                            pred_b.float() / rms, target_b.float() / rms)

                    _bl2 = _bl_mse(_blh[0], _vt_l2 - _blb[0])
                    _bm2 = _bl_mse(_blh[1], _vt_m2 - _blb[1])
                    _vbl2 = args.v_band_loss * float(args.v_band_w_low) * _bl2
                    _vbm2 = args.v_band_loss * float(args.v_band_w_mid) * _bm2
                    _vbh2 = _vbl2.new_zeros(())
                    if bool(getattr(getattr(model, "full_head", None),
                                    "enable_band_lr_high", False)):
                        _bh2 = _bl_mse(_blh[2], _vt_h2 - _blb[2])
                        _vbh2 = args.v_band_loss * float(args.v_band_w_high) * _bh2
                        step_vbh += float(_bh2.detach())
                    loss = loss + _vbl2 + _vbm2 + _vbh2
                    step_vbl += float(_bl2.detach())
                    step_vbm += float(_bm2.detach())
                    _sreg = getattr(model, "_last_bandlr_structure_reg", None)
                    if args.bandlr_structure_loss > 0 and _sreg is not None:
                        _sr = args.bandlr_structure_loss * _sreg
                        loss = loss + _sr
                        step_bsreg += float(_sr.detach())
                if (args.patch_mean_loss > 0 or args.anti_collapse > 0
                        or args.boundary_loss > 0 or args.patch_boundary_loss > 0
                        or args.channel_std_loss > 0 or args.psd_loss > 0
                        or args.flat_penalty > 0 or args.corr_loss > 0
                        or args.std_loss > 0 or args.local_contrast_loss > 0
                        or args.spectrum_loss > 0 or args.temporal_loss > 0):
                    # token 均值/方差都按 2x2 patch 统计；patch mean 是 32px 方块的直接来源。
                    pph = int(model.cfg.patch_size[1])
                    ppw = int(model.cfg.patch_size[2])
                    B0, C0, T0, H0, W0 = xt.shape
                    # x0hat 已在上面按参数化分支正确计算：
                    # --predict-x0 时 x0hat=_pred_raw；否则 x0hat=xt-σ*v。
                    # 旧代码在此无条件用 v 公式覆盖，predict-x0 下 aux 全部统计错误对象。
                    if args.std_loss > 0:
                        # 全局 std 单边托底：只罚预测 std 低于干净 latent 的 0.95 倍，
                        # 不压已经偏高的方差，避免为了“匹配 std”把细节一起压灰。
                        sp_std = x0hat.std(dim=(1, 2, 3, 4))
                        st_std = x0.float().std(dim=(1, 2, 3, 4)).detach()
                        std_ratio = sp_std / (st_std + 1e-6)
                        std_loss = torch.relu(0.95 - std_ratio).pow(2).mean()
                        loss = loss + args.std_loss * std_loss
                        step_std += float(std_loss.detach())
                    pm_pred = x0hat.reshape(B0, C0, T0, H0 // pph, pph,
                                            W0 // ppw, ppw).mean(dim=(4, 6))
                    if args.corr_loss > 0:
                        # 与验收 score_latents 完全同一口径：相邻 patch-mean 的
                        # Pearson 相关，而不是旧实现里的通道向量余弦。旧口径
                        # 数学上优化不到 patch_mean_corr_h/w，已废弃。
                        model._last_corr_defect = patch_mean_pearson_corr_defect(pm_pred)
                        # _main_aux 在 fwd 后立刻快照，那时 corr 还没算；
                        # 这里必须回填，否则 3662 读到的永远是 None，corr 静默不进 loss。
                        _main_aux["corr"] = model._last_corr_defect
                    x0_r = x0.float().reshape(B0, C0, T0, H0 // pph, pph,
                                              W0 // ppw, ppw)
                    pm_tgt = x0_r.mean(dim=(4, 6))
                    if args.patch_mean_loss > 0:
                        patch_loss = torch.nn.functional.mse_loss(pm_pred, pm_tgt)
                        loss = loss + args.patch_mean_loss * patch_loss
                        step_patch += float(patch_loss.detach())
                    if args.local_contrast_loss > 0:
                        # patch 均值的一阶差分幅值：直接约束 32px 块之间的局部对比，
                        # 不要求整块均值对齐，避免 patch_mean MSE 把画面拉成常色块。
                        dp_h = (pm_pred[..., 1:, :] - pm_pred[..., :-1, :]).abs()
                        dt_h = (pm_tgt[..., 1:, :] - pm_tgt[..., :-1, :]).abs().detach()
                        dp_w = (pm_pred[..., :, 1:] - pm_pred[..., :, :-1]).abs()
                        dt_w = (pm_tgt[..., :, 1:] - pm_tgt[..., :, :-1]).abs().detach()
                        local_loss = 0.5 * (torch.nn.functional.l1_loss(dp_h, dt_h)
                                            + torch.nn.functional.l1_loss(dp_w, dt_w))
                        loss = loss + args.local_contrast_loss * local_loss
                        step_local += float(local_loss.detach())
                    if args.spectrum_loss > 0:
                        spec_loss = spectrum_shape_loss(x0hat, x0)
                        loss = loss + args.spectrum_loss * spec_loss
                        step_spec += float(spec_loss.detach())
                    if args.temporal_loss > 0 and x0hat.shape[2] > 1:
                        # 时间差分匹配：约束帧间变化幅值和方向，而不是简单压成静态。
                        td_p = x0hat[:, :, 1:] - x0hat[:, :, :-1]
                        td_t = x0[:, :, 1:] - x0[:, :, :-1]
                        temporal_loss = torch.nn.functional.l1_loss(td_p, td_t)
                        loss = loss + args.temporal_loss * temporal_loss
                        step_temporal += float(temporal_loss.detach())
                    if args.anti_collapse > 0:
                        # 非对称防方块：32px 方块的判据不是“patch 均值方差低”，而是
                        # **patch 内细节方差低 + patch 边界跳变**。这里只惩罚前者：
                        # pred 的 patch 内 std 低于 target 才罚，高于 target 不罚。
                        # 不要求对齐 patch 均值，因此不是 patch-mean loss 的重复。
                        x0hat_r = x0hat.reshape(B0, C0, T0, H0 // pph, pph,
                                                W0 // ppw, ppw)
                        det_p = x0hat_r - x0hat_r.mean(dim=(4, 6), keepdim=True)
                        det_t = x0_r - x0_r.mean(dim=(4, 6), keepdim=True)
                        ratio = (det_p.std(dim=(4, 6)) /
                                 (det_t.std(dim=(4, 6)) + 1e-6))
                        anti_loss = torch.relu(0.85 - ratio).pow(2).mean()
                        loss = loss + args.anti_collapse * anti_loss
                        step_anti += float(anti_loss.detach())
                    if args.boundary_loss > 0:
                        # 32px 方块在 latent 上就是 2px patch 边界周期跳变。
                        # 只罚“接缝跳变 / 内部梯度”超过 margin 的部分；分母 detach，
                        # 梯度只经过接缝像素，不会为了降比值去平滑整块 latent。
                        # x0hat: [B,C,T,H,W]；别再把 T/H 当 H/W。
                        d_h = (x0hat[..., 1:, :] - x0hat[..., :-1, :]).abs()   # H 向差分
                        d_w = (x0hat[..., :, 1:] - x0hat[..., :, :-1]).abs()   # W 向差分
                        bh = d_h[..., 1::2, :].mean()
                        ih = d_h[..., 0::2, :].mean()
                        bw = d_w[..., :, 1::2].mean()
                        iw = d_w[..., :, 0::2].mean()
                        inner = (ih + iw).detach()
                        b_ratio = (bh + bw) / (inner + 1e-6)
                        bnd_loss = torch.relu(b_ratio - args.boundary_margin).pow(2)
                        loss = loss + args.boundary_loss * bnd_loss
                        step_boundary += float(bnd_loss.detach())
                    if args.patch_boundary_loss > 0:
                        # 32px 方块的核心是相邻 2x2 token 均值整块跳变，而不是块内像素。
                        # 以干净 x0 的同位置 patch 均值跳变为基准，只罚 pred 超出
                        # baseline×margin 的部分：真实边缘/纹理不会被强行抹平。
                        jp_h = (pm_pred[..., 1:, :] - pm_pred[..., :-1, :]).abs()
                        jp_w = (pm_pred[..., :, 1:] - pm_pred[..., :, :-1]).abs()
                        jt_h = (pm_tgt[..., 1:, :] - pm_tgt[..., :-1, :]).abs().detach()
                        jt_w = (pm_tgt[..., :, 1:] - pm_tgt[..., :, :-1]).abs().detach()
                        pb_h = torch.relu(
                            jp_h - args.patch_boundary_margin * jt_h - 1e-4).pow(2)
                        pb_w = torch.relu(
                            jp_w - args.patch_boundary_margin * jt_w - 1e-4).pow(2)
                        pb_loss = 0.5 * (pb_h.mean() + pb_w.mean())
                        loss = loss + args.patch_boundary_loss * pb_loss
                        step_patchbnd += float(pb_loss.detach())
                    if args.channel_std_loss > 0:
                        # 逐通道 std 匹配：全局 std 正常但通道塌缩时，VAE 解码会变糊/出网格。
                        # 只罚 pred 低于 target 的部分，避免把噪声当细节奖励。
                        ch_p = x0hat.std(dim=(2, 3, 4))
                        ch_t = x0.float().std(dim=(2, 3, 4)).detach()
                        ch_ratio = ch_p / (ch_t + 1e-6)
                        chstd_loss = torch.relu(
                            args.channel_std_target - ch_ratio).pow(2).mean()
                        loss = loss + args.channel_std_loss * chstd_loss
                        step_chstd += float(chstd_loss.detach())
                    if args.psd_loss > 0:
                        psd_loss = radial_psd_loss(x0hat, x0)
                        loss = loss + args.psd_loss * psd_loss
                        step_psd += float(psd_loss.detach())
                    if args.flat_penalty > 0:
                        # 呼应 4：flat_ratio 与 patch_within_std 同源，只罚
                        # “块内/块间 std 比”超过阈值，避免抬幅值时把画面碎成噪声。
                        fr = x0hat.float().reshape(
                            B0, C0, T0, H0 // pph, pph, W0 // ppw, ppw)
                        fr_within = fr.std(dim=(4, 6)).mean()
                        fr_mean = fr.mean(dim=(4, 6))
                        fr_between = fr_mean.std()
                        flat_ratio = fr_within / (fr_between + 1e-6)
                        flat_pen = torch.relu(flat_ratio - args.flat_target)
                        loss = loss + args.flat_penalty * flat_pen
                        step_flat += float(flat_ratio.detach())
                if args.detail_aux > 0:
                    dv = getattr(getattr(model, "head", None), "_last_detail_v", None)
                    if dv is not None:
                        # detail 支路独立监督：detail 的速度贡献 dv 对应 x0 的
                        # -sigma*dv；只监督 patch 内高频残差，不抢主头的低频均值。
                        B1, C1, T1, H1, W1 = x0.shape
                        pph1 = int(model.cfg.patch_size[1])
                        ppw1 = int(model.cfg.patch_size[2])
                        x0r = x0.float().reshape(
                            B1, C1, T1, H1 // pph1, pph1, W1 // ppw1, ppw1)
                        hf_t = x0r - x0r.mean(dim=(4, 6), keepdim=True)
                        dx0 = (-tv.float() * dv.float()).reshape(
                            B1, C1, T1, H1 // pph1, pph1, W1 // ppw1, ppw1)
                        hf_p = dx0 - dx0.mean(dim=(4, 6), keepdim=True)
                        detail_loss = torch.nn.functional.mse_loss(hf_p, hf_t)
                        loss = loss + args.detail_aux * detail_loss
                        step_detail += float(detail_loss.detach())
                if args.frame_aux > 0:
                    faux = _main_aux["faux"]
                    if faux is not None:
                        loss = loss + args.frame_aux * faux
                        step_frame += float(faux.detach())
                if args.syntax_aux_weight > 0:
                    syn_aux = _main_aux.get("syntax")
                    if syn_aux is not None:
                        loss = loss + args.syntax_aux_weight * syn_aux
                        step_syn += float(syn_aux.detach())
                if args.token_bd_loss > 0:
                    tbd = _main_aux["tbd"]
                    if tbd is not None:
                        loss = loss + args.token_bd_loss * tbd
                        step_tokenbd += float(tbd.detach())
                        if args.token_bd_floor > 0 and args.token_bd_floor_loss > 0:
                            tbdf = torch.relu(tbd.new_tensor(args.token_bd_floor) - tbd)
                            loss = loss + args.token_bd_floor_loss * tbdf
                            step_tbdf += float(tbdf.detach())
                if args.corr_loss > 0:
                    corr_defect = _main_aux["corr"]
                    if corr_defect is not None:
                        # P0 回归封死：这里只能单边惩罚“相关性不足”。
                        # 旧实现 relu(target_min - defect) 会给高相关输出正惩罚，
                        # 直接把 patch_between 从 0.978 往错误区间下拉。
                        corr_pen = corr_defect_penalty(corr_defect, args.corr_target_max)
                        base_cw = (args.corr_warmup_weight
                                   if step <= args.rms_warmup_until else args.corr_loss)
                        cw = corr_defect.new_tensor(base_cw)
                        # sigma 门控：sigma=1 时只保留 10%，低 sigma 结构成形
                        # 后才全权重拉 patch 相关。避免高噪声步把 x0hat 往常量压。
                        _sig_frames = sigma.reshape(-1).repeat_interleave(xt.shape[2]).float()
                        cw = cw * (0.10 + 0.90 * (1.0 - _sig_frames).mean())
                        out_rms = _main_aux["rms"]
                        rms_tgt = _main_aux["rtgt"]
                        if (args.corr_amp_gate > 0 and out_rms is not None
                                and rms_tgt is not None):
                            # 呼应 3：幅值低于目标 -> sigmoid<0.5，corr 自动让路；
                            # 幅值达标 -> 权重回到 base 附近，专注拉相关性。
                            cw = cw * torch.sigmoid(
                                args.corr_amp_gate * (out_rms - rms_tgt))
                        loss = loss + cw * corr_pen
                        step_corr += float(corr_defect.detach())
                        step_corr_pen += float(corr_pen.detach())
                        step_corr_w += float(cw.detach())
                if args.vmag_loss > 0:
                    vmag = _main_aux["vmag"]
                    if vmag is not None:
                        rms_w = (args.rms_warmup_weight
                                 if step <= args.rms_warmup_until else args.vmag_loss)
                        loss = loss + rms_w * vmag
                        step_vmag += float(vmag.detach())
                        step_rms_w += float(rms_w)
                fh_reg = getattr(model, "full_head", None)
                if fh_reg is not None:
                    sreg = getattr(fh_reg, "_last_scale_t_reg", None)
                    if sreg is not None:
                        loss = loss + sreg
                        step_scale_reg += float(sreg.detach())
                    pl2 = getattr(fh_reg, "_last_pre_proj_l2", None)
                    if pl2 is not None:
                        loss = loss + pl2
                        step_pre_l2 += float(pl2.detach())
                if args.skip_reg > 0:
                    skips = getattr(model, "skips", None) or []
                    if skips:
                        # 单边弱正则：alpha>target 才罚，目标 0.95；仍允许模型
                        # 在需要时把个别 skip 学得更低，不强行统一关死。
                        a = torch.stack([s.alpha.float().mean() for s in skips])
                        skip_pen = torch.relu(a - args.skip_reg_target).mean()
                        loss = loss + args.skip_reg * skip_pen
                        step_skipreg += float(skip_pen.detach())
                if prof_on:
                    torch.cuda.synchronize()
                    prof["aux"] += time.time() - prof_t0
                    prof_t0 = time.time()
                step_zero += float(torch.nn.functional.mse_loss(
                    torch.zeros_like(target), target).detach())
                with torch.no_grad():
                    # 统一口径：cos 必须比较「同一参数化的预测」和「同一参数化的目标」。
                    # predict-x0 时 pred 是 x0，而 target 是 v=noise-x0；直接比 pred/target
                    # 会得到负 cos（2026-11-27 实测 step200 cos=-0.35，纯属口径错）。
                    # 这里在 x0 模式下改用 v_pred=(xt-x0hat)/σ 与 target(v) 比。
                    _cos_pred = v_pred.float() if args.predict_x0 else pred.float()
                    step_cos += float(torch.nn.functional.cosine_similarity(
                        _cos_pred.flatten(1), target.float().flatten(1), dim=1).mean())
                    # 去 DC 后的 cos：把每样本全局均值减掉再算，专门暴露
                    # “cos 很高但其实只对齐了亮度常数”的假阳性。
                    _pd = _cos_pred - _cos_pred.mean(dim=(1, 2, 3, 4), keepdim=True)
                    _td = target.float() - target.float().mean(dim=(1, 2, 3, 4), keepdim=True)
                    step_cos_dc += float(torch.nn.functional.cosine_similarity(
                        _pd.flatten(1), _td.flatten(1), dim=1).mean())
                if not torch.isfinite(loss):
                    print(f"[EXPLODE] step {step} loss 非有限，跳过 micro", flush=True)
                    opt.zero_grad(set_to_none=True)
                    if opt_gate is not None:
                        opt_gate.zero_grad(set_to_none=True)
                    continue
                if os.environ.get("SAKIKO_MEM_TRACE") == "1":
                    print(f"[memtrace] step={step} micro={micro} before_bwd "
                          f"alloc={torch.cuda.memory_allocated()/2**30:.3f}GB "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.3f}GB "
                          f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GB", flush=True)
                if (args.hi_grad_clip > 0 and _x0_loss_hi is not None
                        and bool(_hi_mask.any())):
                    # 梯度竞争保护：高σ x0 分支先单独 backward 并做一次全局范数
                    # clip，再回传其余损失。主干最终仍受总 grad-clip 约束。
                    # 这样 000000 的高σ大梯度不会在同一个总范数里挤掉 002200。
                    (_x0_loss_hi / args.accum).backward(retain_graph=True)
                    _hi_gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(args.hi_grad_clip))
                    step_hi_gnorm = max(float(step_hi_gnorm), float(_hi_gnorm))
                    ((loss - _x0_loss_hi) / args.accum).backward()
                else:
                    (loss / args.accum).backward()
                if os.environ.get("SAKIKO_MEM_TRACE") == "1":
                    print(f"[memtrace] step={step} micro={micro} after_bwd "
                          f"alloc={torch.cuda.memory_allocated()/2**30:.3f}GB "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.3f}GB "
                          f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GB", flush=True)
                if prof_on:
                    torch.cuda.synchronize()
                    prof["bwd"] += time.time() - prof_t0
                    prof_t0 = time.time()
                if os.environ.get("SAKIKO_DEBUG_GRAD") and micro == 0:
                    pg = dict(model.named_parameters())
                    for nm in ("head.head.weight", "patchify.weight",
                               "enc_blocks.0.attn.qkv.weight"):
                        p = pg.get(nm)
                        gm = None if p is None or p.grad is None else float(p.grad.abs().max())
                        wm = None if p is None else float(p.detach().abs().max())
                        print(f"[dbg-grad] {nm} |w|={wm} |g|={gm}", flush=True)
                    # 2026-11-23：分组梯度范数（定位 gnorm 爆炸来自哪一段）
                    for pref in ("ada_embedder", "cond_proj", "patchify", "head",
                                 "enc_blocks.0.", "enc_blocks.11.", "bottleneck.0.",
                                 "dec_blocks.0.", "dec_blocks.11.", "skips.0.",
                                 "temporal_special", "spatial_special"):
                        gs = [p.grad for n, p in pg.items()
                              if n.startswith(pref) and p.grad is not None]
                        if not gs:
                            continue
                        tot = float(torch.sqrt(sum(g.float().pow(2).sum() for g in gs)))
                        print(f"[dbg-grp] {pref:18s} gnorm={tot:.3e} n={len(gs)}", flush=True)
                step_loss += float(loss.detach())
            # A/C：final_proj 方向梯度若被压到极小值，预热段单层 boost；
            # 只对该层梯度乘系数，不污染主干 DiT。
            fh_boost = getattr(model, "full_head", None)
            if (fh_boost is not None and step <= args.final_grad_boost_until
                    and args.final_grad_boost > 0):
                fp = getattr(fh_boost, "final_proj", None)
                if fp is not None and fp.weight.grad is not None:
                    fg = float(fp.weight.grad.norm())
                    if fg < args.final_grad_min:
                        fp.weight.grad.mul_(args.final_grad_boost)
                        step_boosted = 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(getattr(args, "grad_clip", 1.0)))
            if prof_on:
                torch.cuda.synchronize()
                prof["clip"] += time.time() - prof_t0
                prof_t0 = time.time()
            if not torch.isfinite(grad_norm):
                print(f"[EXPLODE] step {step} gnorm 非有限，跳过更新", flush=True)
                opt.zero_grad(set_to_none=True)
                if opt_gate is not None:
                    opt_gate.zero_grad(set_to_none=True)
                continue
            skip_update = bool(args.gnorm_skip > 0
                               and float(grad_norm) > args.gnorm_skip)
            if skip_update:
                # 硬尖峰防护：这一小步的梯度方向不可信，不写进 Adafactor/AdamW 状态。
                print(f"[GNORM-SKIP] step {step} gnorm={float(grad_norm):.3f} "
                      f"> {args.gnorm_skip:.3f}，跳过 optimizer.step", flush=True)
                opt.zero_grad(set_to_none=True)
                if opt_gate is not None:
                    opt_gate.zero_grad(set_to_none=True)
            else:
                _ot = os.environ.get("OPT_TIMER") == "1"
                if _ot:
                    torch.cuda.synchronize()
                    _o0 = time.time()
                if args.opt_var_floor > 0:
                    # bf16 Adafactor 的 row_var/col_var 在长尾死行会下溢；一旦
                    # 下溢，weight_decay 的梯度 / eps2 会变成指数放大更新。
                    # 只把二阶矩抬到 1e-8，正常行不受影响。
                    for _g in opt.param_groups:
                        for _p in _g["params"]:
                            _st = opt.state.get(_p)
                            if not _st:
                                continue
                            for _k in ("row_var", "col_var"):
                                _v = _st.get(_k)
                                if torch.is_tensor(_v):
                                    _v.clamp_(min=float(args.opt_var_floor))
                if _ot:
                    torch.cuda.synchronize()
                    _o1 = time.time()
                try:
                    if sr_master is not None and sr_master.enabled:
                        # 裁剪已作用在 bf16 梯度上；这里只做 fp32 拷贝，保证 clip 语义不变。
                        sr_master.pull_grad()
                    opt.step()
                except AssertionError as _ada_err:
                    # 2026-11-25: Adafactor 在多维梯度上要求 row_var/col_var；
                    # patch_mean 损失会激活此前无梯度的分支，先打印到底哪个张量
                    # 的状态不完整，避免只看到一行 assert。
                    print(f"[adafactor][fatal] {_ada_err}", flush=True)
                    for _n, _p in model.named_parameters():
                        if _p.grad is None:
                            continue
                        _st = opt.state.get(_p, {})
                        if _p.grad.ndim > 1 and ("row_var" not in _st or "col_var" not in _st):
                            print(f"[adafactor][bad] {_n} p={tuple(_p.shape)} "
                                  f"g={tuple(_p.grad.shape)} state={list(_st.keys())}", flush=True)
                    raise
                if _ot:
                    torch.cuda.synchronize()
                    _o2 = time.time()
                if opt_gate is not None:
                    opt_gate.step()
                if sr_master is not None and sr_master.enabled:
                    # fp32 master -> bf16，随机舍入（无偏），把 0.29 个量子的位移保住
                    if step % sr_master.push_every == 0:
                        sr_master.push()
                fh = getattr(model, "full_head", None)
                if fh is not None and hasattr(fh, "alpha"):
                    with torch.no_grad():
                        fh.alpha.data.clamp_min_(args.head_alpha_min)
                if fh is not None and hasattr(fh, "out_gain"):
                    with torch.no_grad():
                        fh.out_gain.data.clamp_min_(0.08)
                if fh is not None and hasattr(fh, "clamp_weights"):
                    fh.clamp_weights()
                if args.ema_decay > 0:
                    _ema_update(model, step, args.ema_decay, max(1, args.ema_every))
                if probe_log is not None:
                    probe_log.record_step(step, time.time() - prof_step0,
                                          sr_master, model, skip_update)
                if _ot:
                    torch.cuda.synchronize()
                    _o3 = time.time()
                    print(f"[opt-timer] step={step} varfloor={_o1-_o0:.3f}s "
                          f"adafactor={_o2-_o1:.3f}s adamw={_o3-_o2:.3f}s "
                          f"alloc={torch.cuda.memory_allocated()/2**30:.2f}GB "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
                if (args.weight_health_max > 0 and args.weight_check_every > 0
                        and step % args.weight_check_every == 0):
                    _bad = []
                    for _n, _p in model.named_parameters():
                        if not torch.is_tensor(_p):
                            continue
                        _mx = float(_p.detach().abs().max()) if _p.numel() else 0.0
                        if not torch.isfinite(torch.tensor(_mx)) or _mx > args.weight_health_max:
                            _bad.append((_n, _mx))
                    if _bad:
                        for _n, _mx in _bad[:8]:
                            print(f"[WEIGHT-HEALTH][fatal] {_n} max={_mx:.4g}", flush=True)
                        raise SystemExit(
                            f"权重健康检查失败 step={step}: {len(_bad)} 个张量超过 "
                            f"{args.weight_health_max}")
            if prof_on:
                torch.cuda.synchronize()
                prof["opt"] += time.time() - prof_t0
                prof_t0 = time.time()

            if (not _f5_done and args.freeze_spectral_after > 0
                    and step >= args.freeze_spectral_after):
                n = freeze_spectral_power_iteration(model)
                print(f"[init] F5 step {step} 冻结 {n} 处谱范数幂迭代", flush=True)
                _f5_done = True

            step_loss /= args.accum
            zero_running += step_zero / args.accum
            running += step_loss
            n_done += 1
            # 只有真正 optimizer.step 的步才算有效更新；skip 步不进 opt_avg，
            # 也不更新 best_loss，避免把“没更新模型”的采样方差当成进展。
            if not skip_update:
                running_applied += step_loss
                n_applied += 1
                best_loss = min(best_loss, step_loss)
            _TRAIN_STATE.clear()
            _TRAIN_STATE.update({
                "step": step,
                "sample_cursor": sample_cursor,
                "running": running,
                "best_loss": best_loss,
                "n_done": n_done,
                "n_applied": n_applied,
                "running_applied": running_applied,
                "zero_running": zero_running,
            })
            if step == start_step or step % args.log_every == 0 or step == args.until_step:
                peak = torch.cuda.max_memory_allocated() / 1e9
                rms_now = getattr(model, "_last_out_rms", None)
                rms_tgt_now = getattr(model, "_last_rms_target", None)
                _fh_now = getattr(model, "full_head", None)
                hg_now = getattr(_fh_now, "_last_high_gate_mean", None)
                hr_now = getattr(_fh_now, "_last_high_rms", None)
                gb_now = getattr(_fh_now, "_last_gate_base", None)
                gt_now = getattr(_fh_now, "_last_gate_texture", None)
                rt_now = getattr(_fh_now, "_last_frame_residual", None)
                gg_now = getattr(_fh_now, "_last_gate_grid", None)
                dcg_now = getattr(_fh_now, "_last_dualact_dc", None)
                dcr_now = getattr(_fh_now, "_last_dualact_res", None)
                pnrz_now = getattr(_fh_now, "_last_patch_norm_rz", None)
                bcn_now = getattr(model, "_last_band_cond_norm", None)
                bhr_now = getattr(model, "_last_band_head_rms", None)
                if bhr_now is None:
                    bhr_now = getattr(_fh_now, "_last_band_head_rms", None)
                bhr_str = ("-" if bhr_now is None
                           else "/".join("%.2e" % x for x in bhr_now))
                print(f"[step {step:5d}] stage={spec.stage:24s} mode={spec.mode:8s} "
                      f"lr={lr_spec.lr * args.lr_scale:.2e} loss={step_loss:.5f} avg={running/n_done:.5f} "
                      f"opt_avg={running_applied/max(1,n_applied):.5f} applied={n_applied} "
                      f"zero_avg={zero_running/n_done:.5f} best={best_loss:.5f} "
                      f"cos={(step_cos/args.accum):+.4f} "
                      f"cosdc={(step_cos_dc/args.accum):+.4f} "
                      f"gnorm={float(grad_norm):.3f} skipped={int(skip_update)} "
                      f"peak={peak:.2f}GB {time.time()-t_start:.0f}s "
                      f"| pm={step_patch/args.accum:.5f} anti={step_anti/args.accum:.5f} "
                      f"det={step_detail/args.accum:.5f} faux={step_frame/args.accum:.5f} "
                      f"tbd={step_tokenbd/args.accum:.5f} tbdf={step_tbdf/args.accum:.5f} "
                      f"corr_defect={step_corr/args.accum:.5f} cpen={step_corr_pen/args.accum:.5f} "
                      f"cw={step_corr_w/args.accum:.5f} "
                      f"vmag={step_vmag/args.accum:.6f} rw={step_rms_w/args.accum:.4f} "
                      f"rms={(0.0 if rms_now is None else float(rms_now)):.4f} "
                      f"rtgt={(0.0 if rms_tgt_now is None else float(rms_tgt_now)):.4f} "
                      f"hg={(0.0 if hg_now is None else float(hg_now)):.4f} "
                      f"hr={(0.0 if hr_now is None else float(hr_now)):.4f} "
                      f"gb={(0.0 if gb_now is None else float(gb_now)):.4f} "
                      f"gt={(0.0 if gt_now is None else float(gt_now)):.4f} "
                      f"rt={(0.0 if rt_now is None else float(rt_now)):.4f} "
                      f"gg={(0.0 if gg_now is None else float(gg_now)):.4f} "
                      f"sreg={step_scale_reg/args.accum:.6f} pl2={step_pre_l2/args.accum:.6f} "
                      f"flat={step_flat/args.accum:.4f} std={step_std/args.accum:.5f} "
                      f"dc={step_dc/args.accum:.5f} sdreg={step_stdreg/args.accum:.5f} "
                      f"boost={step_boosted} "
                      f"sreg_skip={step_skipreg/args.accum:.5f} "
                      f"bnd={step_boundary/args.accum:.5f} pbnd={step_patchbnd/args.accum:.5f} "
                      f"psd={step_psd/args.accum:.5f} "
                      f"loc={step_local/args.accum:.5f} spt={step_spec/args.accum:.5f} "
                      f"tmp={step_temporal/args.accum:.5f} "
                      f"x0={step_x0/args.accum:.5f} x0hp={step_x0hp/args.accum:.5f} "
                      f"xcos={step_x0cos/args.accum:.5f} "
                      f"coarse={step_coarse/args.accum:.5f} "
                      f"repa={step_repa/args.accum:.5f} "
                      f"syn={step_syn/args.accum:.5f} "
                      f"himul={step_hi_mult/max(1,args.accum):.3f} "
                      f"hig={step_hi_gnorm:.3f} "
                      f"xf={step_x0f/args.accum:.5f} "
                      f"xfr={step_x0f_low/args.accum:.4f}/"
                      f"{step_x0f_mid/args.accum:.4f}/"
                      f"{step_x0f_high/args.accum:.4f} "
                      f"vb={step_vbl/args.accum:.5f}/"
                      f"{step_vbm/args.accum:.5f}/"
                      f"{step_vbh/args.accum:.5f} "
                      f"bsreg={step_bsreg/args.accum:.6f} "
                      f"ltv={step_ltv/args.accum:.6f} lc={step_lc/args.accum:.5f} "
                      f"sig={(step_t_sum/max(1,step_t_n)):.3f}/"
                      f"{step_t_min:.3f}/{step_t_max:.3f} "
                      f"bcn={(0.0 if bcn_now is None else float(bcn_now)):.2e} "
                      f"bhr={bhr_str} "
                      f"dcg={(0.0 if dcg_now is None else float(dcg_now)):.4f} "
                      f"dcr={(0.0 if dcr_now is None else float(dcr_now)):.4f} "
                      f"pnrz={(0.0 if pnrz_now is None else float(pnrz_now)):.4f} "
                      f"chstd={step_chstd/args.accum:.5f} {watch.report(model)}", flush=True)
                if os.environ.get("CLAMP_STAT") == "1":
                    _fh_c = getattr(model, "full_head", None)
                    _sat = getattr(_fh_c, "_last_clamp_sat", None)
                    if _sat is not None:
                        print(f"[clamp] step={step} hi={_sat[0]:.4f} lo={_sat[1]:.4f} "
                              f"wabs={getattr(_fh_c, '_last_w_absmean', 0.0):.4e}", flush=True)
                if prof_on:
                    print(f"[profile] step={step} shape={tuple(xt.shape)} "
                          f"data={prof['data']:.3f}s fwd={prof['fwd']:.3f}s "
                          f"aux={prof['aux']:.3f}s bwd={prof['bwd']:.3f}s "
                          f"clip={prof['clip']:.3f}s opt={prof['opt']:.3f}s "
                          f"total={time.time()-prof_step0:.3f}s", flush=True)

            if args.save_every and step % args.save_every == 0:
                save_ckpt(model, step, step_loss, args.out, f"backbone_step{step}.pt")
                if args.prune:
                    prune_old(args)
            # 终点不是 save_every 整数倍时补一份标准名，保证续训/停机点一定有档
            if (step == args.until_step and step not in STAGE_ENDS
                    and not (args.save_every and step % args.save_every == 0)):
                save_ckpt(model, step, step_loss, args.out, f"backbone_step{step}.pt")
            if step in STAGE_ENDS:
                save_ckpt(model, step, step_loss, os.path.join(args.out, "stage_ckpts"),
                          f"backbone_step{step}_{spec.stage}.pt")
                # 2026-11-23 修复：阶段末（如 4000/7000）通常不是 save_every 的整数倍，
                # 若只写 stage_ckpts，final_master.latest_ckpt() 会看不到 step4000 →
                # 误判「无 4000 checkpoint」而中止 B1（整夜白跑）。故根部再存一份标准名。
                if not (args.save_every and step % args.save_every == 0):
                    save_ckpt(model, step, step_loss, args.out, f"backbone_step{step}.pt")
                sample_hook(spec, step, args)

            # ================= 4000 步换装 =================
            # 用户要求：训练暂停 -> 释放 DiT 权重显存 -> 加载 M5B 教师 -> 遍历 DS2 批量生成
            # 教师缓存 -> 保存完毕 -> 释放教师显存 -> 重载 DiT -> 自动启动 P2 蒸馏。
            # 教师走独立子进程（tools/teacher_generator.py，本地直推理、不依赖 ComfyUI 服务）：
            # 子进程退出后 HIP 显存由驱动整块回收，比进程内 del+empty_cache 干净。
            if (step == SWAP_STEP and args.p2_autogen and not _did_swap
                    and args.until_step > SWAP_STEP):
                _did_swap = True
                print("=" * 78, flush=True)
                print(f"[swap] step {SWAP_STEP} 到点：暂停训练 → 释放 DiT → 本地直推理教师 → 重载 DiT", flush=True)
                for th in _save_threads:
                    th.join()
                _save_threads.clear()
                four_k = os.path.join(args.out, f"backbone_step{SWAP_STEP}.pt")
                if not os.path.isfile(four_k):
                    # 换装后必须能从 4000 步权重续上，落盘不能等后台线程
                    save_ckpt(model, SWAP_STEP, step_loss, args.out, f"backbone_step{SWAP_STEP}.pt")
                    for th in _save_threads:
                        th.join()
                    _save_threads.clear()

                del model, opt, opt_gate, watch, _buf
                model = opt = opt_gate = watch = _buf = None
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                free, total = torch.cuda.mem_get_info()
                print(f"[swap] DiT 已释放 free={free/1e9:.2f}/{total/1e9:.2f}GB", flush=True)
                if free / 1e9 < args.p2_min_free:
                    raise SystemExit(
                        f"[fatal] 释放 DiT 后空闲仍只有 {free/1e9:.2f}GB < "
                        f"{args.p2_min_free:.1f}GB，拒绝加载教师（会挤爆显存）")

                here = os.path.dirname(os.path.abspath(__file__))
                gen = os.path.abspath(os.path.join(here, "..", "tools", "teacher_generator.py"))
                if not os.path.isfile(gen):
                    raise SystemExit(f"[fatal] 找不到教师生成器: {gen}")
                cmd = [sys.executable, gen,
                       "--latent-dir", os.path.abspath(os.path.join(here, args.p2_latent_dir)),
                       "--prompts", os.path.abspath(os.path.join(here, args.p2_prompts)),
                       "--out", os.path.abspath(os.path.join(here, args.p2_out)),
                       "--cfg", str(args.p2_cfg),
                       "--min-samples", str(args.p2_min),
                       "--min-free-gb", str(args.p2_min_free)]
                if args.p2_limit:
                    cmd += ["--limit", str(args.p2_limit)]
                print("[swap] 调用: " + " ".join(cmd), flush=True)
                t_sw = time.time()
                rc = subprocess.call(cmd, cwd=here)
                print(f"[swap] 教师子进程退出 rc={rc}，用时 {time.time()-t_sw:.0f}s", flush=True)
                if rc != 0:
                    raise SystemExit("[fatal] 教师缓存生成失败（rc!=0），拒绝在空/残缺缓存上启动 P2")

                data["ds2"] = list_pt(args.p2_out)
                if len(data["ds2"]) < args.p2_min:
                    raise SystemExit(f"[fatal] ds2 缓存 {len(data['ds2'])} 条 < {args.p2_min}，不启动 P2")
                print(f"[swap] ds2 缓存就绪 {len(data['ds2'])} 条", flush=True)

                free, _ = torch.cuda.mem_get_info()
                print(f"[swap] 教师已释放 free={free/1e9:.2f}GB（重载 DiT 需 ≥{MIN_FREE_GB_DIT:.1f}GB）", flush=True)
                if free / 1e9 < MIN_FREE_GB_DIT:
                    raise SystemExit(
                        f"[fatal] 教师释放后空闲只有 {free/1e9:.2f}GB < {MIN_FREE_GB_DIT:.1f}GB")

                model = build_model(args.model, args).to(device).to(dtype).train()
                ck = torch.load(four_k, map_location="cpu", weights_only=False)
                msd = model.state_dict()
                bad = [k for k, v in ck["model"].items()
                       if k in msd and tuple(v.shape) != tuple(msd[k].shape)]
                if bad:
                    raise SystemExit(f"[fatal] 重载 4000 步权重形状不符 {len(bad)} 项：{bad[:3]}")
                model.load_state_dict(ck["model"], strict=False)
                del ck
                gc.collect()
                torch.cuda.empty_cache()
                n_gate_fp32 = promote_gate_params_fp32(model)
                sr_master = _build_sr_master(model, args)
                _sr_src = sr_master.param_source if (sr_master is not None and sr_master.enabled) else None
                opt, opt_gate = make_optimizers(model, args, param_source=_sr_src)
                _SAVE_OPTIM.clear()
                _SAVE_OPTIM.update({"opt": opt, "opt_gate": opt_gate})
                opt_audit = audit_optimizer_coverage(model, opt, opt_gate, param_source=_sr_src)
                print(f"[opt-audit] {opt_audit}", flush=True)
                print(f"[swap][gate] 门控 fp32 化 {n_gate_fp32} 个", flush=True)
                watch = UpdateWatch(model)
                _buf = {"t_emb": None, "text_emb": None}
                _f5_done = False           # 新模型实例，4001 步重新冻结谱范数
                running, zero_running, n_done = 0.0, 0.0, 0
                best_loss = float("inf")
                torch.cuda.reset_peak_memory_stats()
                print(f"[swap] DiT 重载完成，自动接 P2 蒸馏（step {SWAP_STEP+1} 起）", flush=True)
                print("=" * 78, flush=True)

            if args.cycle_steps > 0 and step >= cycle_end:
                for th in _save_threads:
                    th.join()
                print(f"[cycle] step {step} 周期结束，干净退出清 HIP 碎片", flush=True)
                sys.exit(CYCLE_EXIT)

    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] step {step}: {str(e)[:180]}", flush=True)
        torch.cuda.empty_cache()
        sys.exit(OOM_EXIT)

    for th in _save_threads:
        th.join()
    # Windows + ROCm 的 python.exe 在解释器退出阶段偶发把进程码改成 1，
    # 即使权重已原子落盘、[done] 已打印。给上层一个磁盘上的完成标记，
    # 并在所有写盘线程 join 后强制 0 退出，避免自动链路把成功节点误判成失败。
    try:
        with open(os.path.join(args.out, ".train_done"), "w", encoding="utf-8") as f:
            f.write(f"step={args.until_step}\nloss={loss:.8f}\n")
    except Exception as e:
        print(f"[done][warn] 写完成标记失败：{e}", flush=True)
    print(f"[done] step {args.until_step} 完成，总耗时 {time.time()-t_start:.0f}s", flush=True)
    # 模块级 os._exit(0) 会绕过 atexit；正常退出前显式 close，确保 rollup/探针缓冲落盘。
    try:
        if data_reporter is not None:
            data_reporter.close()
        if probe_log is not None:
            probe_log.close()
    except Exception as _ce:
        print(f"[done][warn] reporter/probe close: {_ce}", flush=True)


if __name__ == "__main__":
    main()
    # 明确覆盖 ROCm/Windows 解释器退出阶段的伪非零码；异常路径不会走到这里。
    os._exit(0)
