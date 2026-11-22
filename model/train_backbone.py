#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyraGate-VLM v2.0 — 裸主干训练循环（2026-11-22 加固版）

阶段1（当前）：流匹配自监督 / 教师蒸馏，只训 4.5B 主干（GSA + Dual-Act SwiGLU）；
VLM / 北大桥 / cond_mod / t阈值 / OPS / Claude断言 / MoE 三专家 全部不实例化、不进图。

显存安全（实测教训）：
  - mem_frac 默认 0.93（留 ~1.5G 吸收 ROCm 碎片；0.99 太极限，已实测会炸）；
  - Adafactor（factored 状态仅 ~11MB），BF16 权重/梯度；
  - 梯度检查点 use_reentrant=False（见 uvit3d.py）；
  - return_aux=False（不建时序辅助头，堵住显存漏点）；
  - 周期重启：--cycle-steps N 跑满 N 步主动 exit(43)，守护脚本拉起新进程清空 HIP 碎片。

教师蒸馏（--teacher-dir）：
  教师 Wan5B M85 权重 10GB，不能与学生 9GB 共驻显存 → 分时加载。
  离线预编码脚本 preencode_teacher.py 先产出每样本 {x0, noise, t, t_emb, text_emb, v_teacher}，
  训练时本脚本只读预编码目标，loss = MSE(pred, v_teacher)，教师不进训练进程。

保存：每 --save-every 步，state_dict 转 CPU 后由后台线程原子写盘（磁盘 IO 释放 GIL，
不阻塞 GPU 计算流）。Windows 下 spawn 子进程传 9GB state_dict 会内存翻倍，故用线程更稳。

退出码：0=正常结束 / 42=OOM / 43=周期重启（均由 run_training_daemon.py 处理）。
"""
import os
# 显存碎片治理必须在 import torch 之前（expandable_segments 在部分平台不生效，保留无害）
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import threading
import time
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uvit3d import UViT3D, UViT3DConfig, count_parameters

C = 16  # latent 通道数（Wan VAE）

OOM_EXIT = 42
CYCLE_EXIT = 43
_save_threads = []


def build_model(kind: str, args) -> UViT3D:
    if kind == "tiny":
        cfg = UViT3DConfig.tiny()
    elif kind == "4b5":
        cfg = UViT3DConfig.v1_baseline_4b5()
    else:
        raise ValueError(f"unknown model kind: {kind}")
    cfg.grad_checkpoint = (not args.no_ckpt)
    model = UViT3D(cfg)
    return model


def _atomic_save(state_cpu, path):
    """后台线程：原子写盘，不碰 GPU。"""
    tmp = path + ".tmp"
    try:
        torch.save(state_cpu, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[save-err] {path}: {e}", flush=True)


def save_ckpt_async(model, step, step_loss, args):
    """state_dict 转 CPU 后交后台线程写盘；训练循环继续跑，不等磁盘 IO。

    安全加固：起新存档前先 join 上一个线程，串行化 9GB state_dict
    的 CPU 拷贝 —— 最坏情况同时只留 1 份 9GB（原来最多 3 份 = 27GB，机器 33GB，
    昨天 RAM 96% 卡死的根因）。save_every 500 默认也降低频率。
    """
    for th in _save_threads:
        if th.is_alive():
            th.join()
    _save_threads.clear()
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = os.path.join(args.out, f"backbone_step{step}.pt")
    payload = {"model": state_cpu, "step": step, "cfg": str(model.cfg),
               "loss": float(step_loss)}
    th = threading.Thread(target=_atomic_save, args=(payload, path), daemon=False)
    th.start()
    _save_threads.append(th)
    print(f"[save] {path} step={step} loss={step_loss:.5f} (async, serialized)", flush=True)


def load_latent_files(d):
    return sorted(glob.glob(os.path.join(d, "*.pt")))


def make_batch(args, device, dtype, files=None, idx=0, teacher_files=None):
    """统一返回 (x0, t_emb, text_emb, teacher_noise_t, target)。
    teacher_noise_t=(noise,t) 或 None；target=v_teacher 或 None。加噪统一在训练循环做。"""
    if teacher_files:
        rec = torch.load(teacher_files[idx % len(teacher_files)], map_location="cpu",
                         weights_only=False)
        x0 = rec["x0"].to(device=device, dtype=dtype)
        noise = rec["noise"].to(device=device, dtype=dtype)
        t = rec["t"].to(device=device, dtype=dtype)
        t_emb = rec["t_emb"].to(device=device, dtype=dtype)
        text_emb = rec["text_emb"].to(device=device, dtype=dtype)
        target = rec["v_teacher"].to(device=device, dtype=dtype)
        return x0, t_emb, text_emb, (noise, t), target
    # 无教师：合成/真实 latent + 流匹配自监督
    if files:
        obj = None
        for _t in range(4):
            try:
                obj = torch.load(files[idx % len(files)], map_location="cpu", weights_only=False)
                break
            except Exception:
                idx += 1  # 跳过正在被编码线程写入的 .pt
        if obj is None:
            latent = torch.randn(1, getattr(args, "in_channels", 48), args.T, args.H, args.W,
                                 device=device, dtype=dtype)
        else:
            latent = obj.get("latent", obj.get("x")) if isinstance(obj, dict) else obj
            latent = latent.to(device=device, dtype=dtype)
    else:
        latent = torch.randn(1, getattr(args, "in_channels", 48), args.T, args.H, args.W,
                             device=device, dtype=dtype)
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    B = latent.shape[0]
    t_emb = torch.randn(B, args.t_dim, device=device, dtype=dtype)
    text_emb = torch.randn(B, args.cond_dim, device=device, dtype=dtype)
    return latent, t_emb, text_emb, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "4b5"], default="tiny")
    ap.add_argument("--latent-dir", default="", help="真实 latent .pt 目录（无教师时）")
    ap.add_argument("--teacher-dir", default="", help="离线教师目标 .pt 目录（含 v_teacher）")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.93, help="显存熔断比例，默认 0.93")
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--cycle-steps", type=int, default=1000,
                    help="跑满 N 步主动 exit(43) 让守护清 HIP 碎片；0=禁用（默认 1000）")
    ap.add_argument("--init-std", type=float, default=0.02,
                    help="F2 谱初始化 std（教师实测 ~0.02；0=禁用改用 torch 默认）")
    ap.add_argument("--freeze-spectral-after", type=int, default=100,
                    help="F5 预热 N 步后冻结 spectral_norm 幂迭代；0=禁用")
    ap.add_argument("--warmstart", default="",
                    help="教师权重热启动：M85 safetensors 路径；空=禁用。"
                         "直拷 embed/head 级 10 项（R2 修正版），优先级 build>warmstart>F2>resume")
    ap.add_argument("--log-every", type=int, default=20, help="打印间隔（步），精简监控")
    args = ap.parse_args()

    if args.mem_frac > 0.95:
        print(f"[warn] mem_frac={args.mem_frac} 太危险，降到 0.95", flush=True)
        args.mem_frac = 0.95

    assert torch.cuda.is_available(), "需要 GPU（ROCm/CUDA）"
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device, dtype = "cuda", torch.bfloat16
    for bn in ("cudnn",):  # ROCm 下可能无此后端，存在才开
        be = getattr(torch.backends, bn, None)
        if be is not None and hasattr(be, "benchmark"):
            try:
                be.benchmark = True
            except Exception:
                pass

    free, total = torch.cuda.mem_get_info()
    print(f"[env] {torch.cuda.get_device_name(0)} free {free/1e9:.2f}/{total/1e9:.2f}GB "
          f"mem_frac={args.mem_frac} bf16 ckpt={not args.no_ckpt} accum={args.accum}", flush=True)

    model = build_model(args.model, args).to(device).to(dtype).train()
    args.t_dim = model.cfg.t_emb_dim
    args.cond_dim = model.cfg.cond_dim
    args.in_channels = model.cfg.in_channels   # 蜂鸟轻量 VAE=48 通道（合成 latent 与真实对齐）
    # 教师热启动（R2 修正版）：build -> warmstart -> F2 -> resume（后者优先级最高）
    if args.warmstart:
        from teacher_warmstart import warmstart_from_teacher, format_report
        ws_rep = warmstart_from_teacher(model, args.warmstart)
        print(format_report(ws_rep), flush=True)
        if not ws_rep["ok"]:
            print("[warn] 教师热启动未完全成功，检查 teacher 路径与形状", flush=True)
    # F2: 教师谱初始化（在 resume 之前，resume 权重会覆盖它）
    if args.init_std > 0:
        from uvit3d import reinit_linear_teacher_spectrum
        n_reinit = reinit_linear_teacher_spectrum(model, std=args.init_std)
        print(f"[init] F2 谱初始化 std={args.init_std} 重设 {n_reinit} 个 Linear", flush=True)
    total_p, train_p = count_parameters(model)
    for p in model.parameters():
        p.requires_grad_(True)
    start_step = 1
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        start_step = int(ck.get("step", 0)) + 1
        print(f"[resume] {args.resume} → step {start_step}", flush=True)
    print(f"[model] {args.model} params={total_p/1e9:.3f}B trainable={train_p/1e9:.3f}B "
          f"dual_act={getattr(model.cfg,'dual_act',None)} use_moe={getattr(model.cfg,'use_moe',None)}",
          flush=True)

    opt = torch.optim.Adafactor(model.parameters(), lr=args.lr)
    teacher_files = load_latent_files(args.teacher_dir) if args.teacher_dir else []
    latent_files = load_latent_files(args.latent_dir) if args.latent_dir else []
    if teacher_files:
        print(f"[data] 教师预编码目标 {len(teacher_files)} 个（蒸馏 loss=MSE vs v_teacher）", flush=True)
    elif latent_files:
        print(f"[data] 真实 latent {len(latent_files)} 个（流匹配自监督）", flush=True)
    else:
        print("[data] 合成 latent（仅验管线，不代表学到东西）", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    running, best_loss = 0.0, float("inf")
    n_done = 0
    cycle_end = (start_step - 1 + args.cycle_steps) if args.cycle_steps > 0 else args.steps + 1

    for step in range(start_step, args.steps + 1):
        # 边训边补：每 100 步重扫 latent 目录，新下/新编码素材自动进池
        if args.latent_dir and not teacher_files and step % 100 == 0:
            _fresh = load_latent_files(args.latent_dir)
            if len(_fresh) != len(latent_files):
                print(f"[data] latent 池刷新 {len(latent_files)} → {len(_fresh)}", flush=True)
                latent_files = _fresh
        try:
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(args.accum):
                x0, t_emb, text_emb, teacher_noise_t, target = make_batch(
                    args, device, dtype,
                    latent_files or None, step + micro,
                    teacher_files or None)
                if teacher_noise_t is not None:
                    # 教师模式：用教师预编码的同款 noise/t 重建 xt，target=v_teacher
                    noise, t = teacher_noise_t
                else:
                    # 流匹配自监督：现场加噪，target=noise-x0
                    noise = torch.randn_like(x0)
                    t = torch.rand(x0.shape[0], device=device, dtype=dtype)
                    target = noise - x0
                tv = t.view(-1, 1, 1, 1, 1)
                xt = (1 - tv) * x0 + tv * noise
                pred = model(xt, t_emb, text_emb, return_aux=False)  # 不建辅助头，省显存
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
                if not torch.isfinite(loss):
                    print(f"[EXPLODE] step {step} loss 非有限 → 跳过 micro", flush=True)
                    opt.zero_grad(set_to_none=True)
                    continue
                (loss / args.accum).backward()
                step_loss += float(loss.detach())   # 标量累加，不保留历史 loss 张量
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                print(f"[EXPLODE] step {step} grad_norm 非有限 → 跳过更新", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            # F5: 预热后冻结 spectral_norm 幂迭代（只冻一次）
            if args.freeze_spectral_after > 0 and step == args.freeze_spectral_after:
                from uvit3d import freeze_spectral_power_iteration
                n_frozen = freeze_spectral_power_iteration(model)
                print(f"[init] F5 step {step} 冻结 {n_frozen} 处谱范数幂迭代"
                      f"（消检查点重算不确定性）", flush=True)
            step_loss /= args.accum
            running += step_loss
            n_done += 1
            if step_loss < best_loss:
                best_loss = step_loss

            if step == start_step or step % args.log_every == 0 or step == args.steps:
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[step {step:5d}] loss={step_loss:.5f} avg={running/n_done:.5f} "
                      f"best={best_loss:.5f} gnorm={float(grad_norm):.3f} "
                      f"peak={peak:.2f}GB {time.time()-t_start:.0f}s", flush=True)

            if args.save_every and step % args.save_every == 0:
                save_ckpt_async(model, step, step_loss, args)

            if args.cycle_steps > 0 and step >= cycle_end:
                # 周期重启：先等所有在途保存线程写完，再干净退出清 HIP 碎片
                for th in _save_threads:
                    th.join()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[cycle] step {step} 周期结束，peak={peak:.2f}GB，"
                      f"干净退出清 HIP 碎片（守护拉起续训）", flush=True)
                sys.exit(CYCLE_EXIT)

        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] step {step}: {str(e)[:180]}", flush=True)
            torch.cuda.empty_cache()
            sys.exit(OOM_EXIT)

    for th in _save_threads:
        th.join()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[done] steps 达标 final_avg={running/max(1,n_done):.5f} "
          f"peak={peak:.2f}GB time={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
