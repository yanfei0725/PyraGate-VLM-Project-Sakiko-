#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练守护进程（人不在也能跑，崩了自动拉起）— 2026-11-22 加固版

职责：
  1. 循环拉起 train_backbone.py 子进程；
  2. 子进程崩溃 / OOM → 自动重启，最多 MAX_RESTART 次，从最新 checkpoint 续训；
  3. OOM 自动降级：只缩 latent token / 降 accum，绝不升 mem_frac（升=更容易炸）；
  4. ★ 周期重启：子进程每跑 --cycle-steps 步主动干净退出（退出码 43），
     新进程天然销毁全部 HIP 显存 → 强制清空 ROCm 碎片（本机 expandable_segments 不生效）；
  5. 全程写日志，可后台无人值守。

退出码约定：
  0  = 训练正常跑完（steps 达标）→ 守护停止
  42 = OOM → 降级 latent/accum 后重启
  43 = 周期重启（清碎片）→ 原配置原样重启，不算失败、不消耗 restart 配额
  其它 = 崩溃 → 原配置重启

显存策略（吸取 11-22 死机教训）：
  - mem_frac 默认 0.93（21.46G*0.93≈19.96G，留 ~1.5G 吸收 ROCm 碎片；0.99 太极限）；
  - 子进程独立 + set_per_process_memory_fraction 熔断，绝不整机卡死；
  - checkpoint 每 500 步由子进程串行化异步落盘（join 上一线程，最坏 9GB 而非 27GB RAM），
    周期重启前必有存档可续。

用法：
  python run_training_daemon.py --model 4b5 --steps 3000 --T 4 --H 32 --W 32 \
      --mem-frac 0.93 --save-every 500 --cycle-steps 1000 --accum 4
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# 训练子进程须用带 torch 的 python（可用环境变量 PYRAGATE_TRAIN_PYTHON 指定）；
# Python312 拉起，sys.executable 会指向无 ROCm 的解释器 → import torch 失败 →
# 子进程假死 → daemon 反复拉起形成套娃链（2026-11-22 实测事故）。故优先固定
# 未指定时默认继承守护进程自身的解释器。
_VENV_PY = os.environ.get("PYRAGATE_TRAIN_PYTHON", sys.executable)
if os.path.isfile(_VENV_PY):
    PY = _VENV_PY
else:
    PY = sys.executable
TRAIN = os.path.join(HERE, "train_backbone.py")
LOG = os.path.join(HERE, "training_daemon.log")
OOM_EXIT = 42
CYCLE_EXIT = 43
MAX_RESTART_DEFAULT = 8


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    except OSError:
        pass


def find_latest_ckpt(out_dir):
    if not os.path.isdir(out_dir):
        return None
    cks = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
           if f.startswith("backbone_step") and f.endswith(".pt")]
    if not cks:
        return None
    return max(cks, key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "4b5"], default="tiny")
    ap.add_argument("--latent-dir", default="")
    ap.add_argument("--teacher-dir", default="", help="离线教师目标 .pt 目录（含速度场/特征）")
    ap.add_argument("--warmstart", default="",
                    help="教师 M85 权重路径，直拷 embed/head 级 10 项做热启动（R2 修正版）")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--mem-frac", type=float, default=0.93,
                    help="显存熔断比例，默认 0.93（留碎片缓冲）；不要设 >0.95")
    ap.add_argument("--out", default=os.path.join(HERE, "ckpt"))
    ap.add_argument("--save-every", type=int, default=500,
                    help="存档间隔（建议 100→500，降低 9GB state_dict 拷贝频率）")
    ap.add_argument("--cycle-steps", type=int, default=1000,
                    help="子进程每 N 步干净退出一次以清 HIP 碎片；0=禁用周期重启（建议 300→1000）")
    ap.add_argument("--max-restart", type=int, default=MAX_RESTART_DEFAULT)
    ap.add_argument("--restart-wait", type=int, default=15, help="重启前等待秒数")
    args = ap.parse_args()

    # 安全护栏：mem_frac 上限硬性压到 0.95，防止手滑设 0.99 再次死机
    if args.mem_frac > 0.95:
        log(f"WARN 请求 mem_frac={args.mem_frac} > 0.95 太危险，强制降到 0.95")
        args.mem_frac = 0.95

    log("=" * 72)
    log(f"DAEMON START model={args.model} steps={args.steps} "
        f"T/H/W={args.T}/{args.H}/{args.W} mem_frac={args.mem_frac}")
    log(f"  latent dir   = {args.latent_dir or '<合成数据，仅验管线>'}")
    log(f"  teacher dir  = {args.teacher_dir or '<无教师，流匹配自监督>'}")
    log(f"  save-every={args.save_every} cycle-steps={args.cycle_steps} "
        f"accum={args.accum} max-restart={args.max_restart}")

    cur_T, cur_H, cur_W = args.T, args.H, args.W
    cur_frac = args.mem_frac          # OOM 只会保持或下调，绝不上调
    restarts = 0
    cycles = 0
    os.makedirs(args.out, exist_ok=True)

    while True:
        ckpt = find_latest_ckpt(args.out)
        cmd = [PY, TRAIN,
               "--model", args.model,
               "--T", str(cur_T), "--H", str(cur_H), "--W", str(cur_W),
               "--steps", str(args.steps),
               "--lr", str(args.lr),
               "--accum", str(args.accum),
               "--mem-frac", str(cur_frac),
               "--out", args.out,
               "--save-every", str(args.save_every)]
        if args.cycle_steps > 0:
            cmd += ["--cycle-steps", str(args.cycle_steps)]
        if args.latent_dir:
            cmd += ["--latent-dir", args.latent_dir]
        if args.teacher_dir:
            cmd += ["--teacher-dir", args.teacher_dir]
        if args.warmstart:
            cmd += ["--warmstart", args.warmstart]
        if ckpt:
            cmd += ["--resume", ckpt]

        tag = "cycle" if cycles else "attempt"
        log(f"LAUNCH {tag} restarts={restarts}/{args.max_restart} cycles={cycles} "
            f"T/H/W={cur_T}/{cur_H}/{cur_W} frac={cur_frac} resume={bool(ckpt)}")
        t0 = time.time()
        rc = subprocess.call(cmd)
        dur = time.time() - t0

        if rc == 0:
            log(f"SUCCESS 训练正常结束（{dur:.0f}s）。DAEMON STOP。")
            break

        if rc == CYCLE_EXIT:
            # 周期重启：干净退出清碎片，原配置原样拉起，不算失败
            cycles += 1
            log(f"CYCLE 子进程完成一个周期（{dur:.0f}s），清 HIP 碎片后原配置重启 "
                f"（不消耗 restart 配额）")
            time.sleep(args.restart_wait)
            continue

        restarts += 1
        if restarts > args.max_restart:
            log(f"GIVE UP 重启 {args.max_restart} 次仍失败，停止。查 training_daemon.log。")
            sys.exit(1)

        if rc == OOM_EXIT:
            # OOM：只降激活占用。降 mem_frac 只会更早触发熔断，绝不降；升更不可能。
            if cur_T > 4:
                cur_T = max(4, cur_T // 2)
                cur_H = max(16, cur_H // 2)
                cur_W = max(16, cur_W // 2)
                log(f"OOM → 降级 latent 到 T/H/W={cur_T}/{cur_H}/{cur_W}（mem_frac 保持 {cur_frac}）")
            elif args.accum > 1:
                args.accum = max(1, args.accum // 2)
                log(f"OOM → 降梯度累积 accum 到 {args.accum}（mem_frac 保持 {cur_frac}）")
            else:
                log("OOM 且已到最小配置（T=4/accum=1）——全参训练物理上塞不下，停止")
                log("  建议：转 LoRA / 部分层训练，或换 48GB 卡。已存 checkpoint 仍在。")
                sys.exit(1)
        else:
            log(f"CRASH rc={rc}，{args.restart_wait}s 后原配置重启（checkpoint 续训）")

        log(f"WAIT {args.restart_wait}s 后重启...")
        time.sleep(args.restart_wait)


if __name__ == "__main__":
    main()
