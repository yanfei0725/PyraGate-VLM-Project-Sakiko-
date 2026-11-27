#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练进度条：读 training_daemon.log，显示 step/loss/显存/ETA。

用法：
    python train_progress.py                 # 默认读同目录 training_daemon.log，总步数 3500
    python train_progress.py --total 3500 --log training_daemon.log
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([\d.]+)\s+avg=([\d.]+)\s+best=([\d.]+)"
    r".*?peak=([\d.]+)GB\s+(\d+)s"
)


def parse(log_path):
    pts = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = PAT.search(ln)
            if m:
                pts.append(dict(step=int(m.group(1)), loss=float(m.group(2)),
                                avg=float(m.group(3)), best=float(m.group(4)),
                                peak=float(m.group(5)), sec=int(m.group(6))))
    return pts


def bar(pct, width=40):
    done = int(width * pct)
    return "█" * done + "░" * (width - done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "training_daemon.log"))
    ap.add_argument("--total", type=int, default=3500)
    args = ap.parse_args()

    pts = parse(args.log)
    if not pts:
        print("尚无 step 日志（模型可能还在加载/warmstart）")
        return
    last = pts[-1]
    step, total = last["step"], args.total
    pct = min(1.0, step / total)

    # 稳定步时：用最后两个 log-every 采样点（跳过首步 warmup）
    if len(pts) >= 2:
        a, b = pts[-2], pts[-1]
        sp = (b["sec"] - a["sec"]) / max(1, b["step"] - a["step"])
    else:
        sp = last["sec"] / max(1, step)

    remain_s = max(0, (total - step) * sp)
    eta = datetime.now() + timedelta(seconds=remain_s)

    print()
    print("  PyraGate-VLM 主干训练进度")
    print("  " + "=" * 56)
    print(f"  [{bar(pct)}] {pct*100:5.1f}%")
    print(f"  step {step} / {total}    loss={last['loss']:.4f}  "
          f"avg={last['avg']:.4f}  best={last['best']:.4f}")
    print(f"  步时 {sp:.2f}s/step   peak {last['peak']:.2f}GB   "
          f"剩余 {remain_s/3600:.2f}h")
    print(f"  预计完成 {eta.strftime('%H:%M')}  "
          f"（约 {int(remain_s//3600)}h{int((remain_s%3600)//60)}m 后）")
    print("  " + "=" * 56)
    # loss 趋势（最近 8 个采样点）
    recent = pts[-8:]
    if len(recent) >= 2:
        trend = " → ".join(f"{p['loss']:.2f}" for p in recent)
        print(f"  loss 走势: {trend}")
    print()


if __name__ == "__main__":
    main()
