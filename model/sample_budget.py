"""每条素材的「最多训几遍」硬门禁。

背景
----
采样器是 `idx = sample_cursor % len(files)`，`files` 由 `list_pt()` 的 sorted()
给出，所以任意历史区间的「哪个文件被取了几次」都可以精确重建。2026-11-26 实测：
`teacher_hi050` 1969 条，a0h 500 + a0i 1000 + a0j 1000 = 2500 个样本，
分布是 **1438 条 1 遍 / 531 条 2 遍 / 0 条未用**。

规矩：**每条素材最多 2 遍**。因此
- 池子生命周期上限 = 1969 × 2 = 3938 个样本；
- 已消耗 2500，**剩余可用 1438 个样本**；
- 531 条已到 2 遍，必须从后续采样中永久排除；
- 1438 条各还剩 1 次，用完后池子耗尽 → 必须加新素材，**不允许回绕复用**。

实现
----
`PassLimitedSampler` 按 shuffle 队列发牌：只发 count<max 的文件；发一次 count+1；
count 到上限即从 pool 永久移除；队列空时从 pool 重建；pool 也空 → 直接
SystemExit（硬停，绝不静默回绕）。计数存 JSON，每 N 步原子落盘 + 退出时落盘。
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# ledger IO
# ---------------------------------------------------------------------------
def load_ledger(path: str, files: Iterable[str], max_passes: int) -> Dict[str, int]:
    """读计数；缺失的按 0 补，避免新增文件被误判为已用满。"""
    names = [os.path.basename(f) for f in files]
    counts: Dict[str, int] = {}
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            counts = {str(k): int(v) for k, v in (obj.get("counts") or {}).items()}
        except Exception as e:  # 坏文件不能让训练静默变成"无限复用"
            raise SystemExit(f"[data-budget][fatal] ledger 读取失败 {path}: {e}")
    for n in names:
        counts.setdefault(n, 0)
    return counts


def save_ledger(path: str, counts: Dict[str, int], meta: Optional[dict] = None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"counts": counts}
    if meta:
        payload.update(meta)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------
class PassLimitedSampler:
    def __init__(self, files: List[str], counts: Dict[str, int], max_passes: int,
                 seed: int = 11, ledger_path: str = "", save_every: int = 50,
                 dataset: str = "", save_on_init: bool = True):
        self.files = list(files)
        self.counts = counts
        self.max_passes = int(max_passes)
        self.seed = int(seed)
        self.ledger_path = ledger_path
        self.save_every = int(save_every)
        self.dataset = dataset
        self._issued = 0
        self._epoch = 0
        self.history: List[str] = []
        self.pool = [f for f in self.files
                     if self.counts.get(os.path.basename(f), 0) < self.max_passes]
        self.queue: List[str] = []
        self._refill()
        if save_on_init:
            self.save()

    # -- 队列 ---------------------------------------------------------------
    def _refill(self) -> None:
        self.queue = list(self.pool)
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(self.queue)
        self._epoch += 1

    def next_file(self) -> str:
        for _attempt in range(2):  # 最多重建两轮，避免无限递归
            if not self.queue:
                if not self.pool:
                    raise SystemExit(
                        "[data-budget][fatal] 数据池已耗尽：所有素材都到达 "
                        f"max_passes={self.max_passes} 上限，剩余可用样本=0。"
                        "按约束不得回绕复用，请补充新素材或显式调高上限后重跑。")
                self._refill()
            while self.queue:
                f = self.queue.pop()
                n = os.path.basename(f)
                if self.counts.get(n, 0) >= self.max_passes:
                    continue  # 上一轮已发满，跳过
                self.counts[n] = self.counts.get(n, 0) + 1
                if self.counts[n] >= self.max_passes and f in self.pool:
                    self.pool.remove(f)
                self._issued += 1
                self.history.append(n)
                if self.save_every > 0 and self._issued % self.save_every == 0:
                    self.save()
                return f
        raise SystemExit("[data-budget][fatal] 队列两次重建后仍无可用样本，硬停。")

    # -- 统计 ---------------------------------------------------------------
    def remaining_budget(self) -> int:
        return sum(max(0, self.max_passes - self.counts.get(os.path.basename(f), 0))
                   for f in self.files)

    def summary(self) -> str:
        dist: Dict[int, int] = {}
        for f in self.files:
            c = self.counts.get(os.path.basename(f), 0)
            dist[c] = dist.get(c, 0) + 1
        at_cap = sum(v for k, v in dist.items() if k >= self.max_passes)
        return (f"[data-budget] dataset={self.dataset or '?'} n={len(self.files)} "
                f"max_passes={self.max_passes} 已发={self._issued} "
                f"剩余可用样本={self.remaining_budget()} "
                f"已达上限={at_cap} 遍数分布={dict(sorted(dist.items()))} "
                f"ledger={self.ledger_path or '(不落盘)'}")

    def save(self) -> None:
        if not self.ledger_path:
            return
        save_ledger(self.ledger_path, self.counts,
                    meta={"dataset": self.dataset, "max_passes": self.max_passes,
                          "n": len(self.files), "issued": self._issued})


# ---------------------------------------------------------------------------
# 历史精确重建：cursor 区间 -> 每文件遍数
# ---------------------------------------------------------------------------
def rebuild_counts(files: List[str], ranges: Iterable[tuple]) -> Dict[str, int]:
    """ranges: [(start_cursor, steps), ...]，按 ``cursor % N`` 的顺序采样。"""
    n = len(files)
    counts: Dict[str, int] = {os.path.basename(f): 0 for f in files}
    for start, steps in ranges:
        for cur in range(int(start), int(start) + int(steps)):
            counts[os.path.basename(files[cur % n])] += 1
    return counts


def _cli() -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="按 cursor 区间重建每条素材的已训遍数")
    ap.add_argument("--dir", required=True, help="素材目录（.pt）")
    ap.add_argument("--ranges", nargs="+", required=True,
                    help="start:steps，例如 10314:500 10814:1000")
    ap.add_argument("--max-passes", type=int, default=2)
    ap.add_argument("--out", default="outputs/sample_usage_ledger.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.pt")))
    rs = [tuple(int(x) for x in r.split(":")) for r in a.ranges]
    counts = rebuild_counts(files, rs)
    dist: Dict[int, int] = {}
    for v in counts.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"[rebuild] n={len(files)} 总样本={sum(counts.values())} 分布={dict(sorted(dist.items()))}")
    print(f"[rebuild] 剩余可用样本={sum(max(0, a.max_passes - v) for v in counts.values())}")
    save_ledger(a.out, counts, meta={"dataset": os.path.basename(os.path.normpath(a.dir)),
                                     "max_passes": a.max_passes, "n": len(files)})
    print(f"[rebuild] ledger -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
