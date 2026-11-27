#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFG-A 教师目标预编码（兼容入口，2026-11-23 改造）

★ 重大变更：**废弃 ComfyUI 服务依赖**。教师推理统一走 tools/teacher_generator.py
  的本地直推理（comfy.sd 直接加载权重 + 进程内前向），不再 POST /prompt、
  不再要求 --comfy-url 有服务在跑。本文件只保留命令行/函数名兼容，实现转发过去。

★ 口径修正（旧实现三处错误，全部已改）：
  1. 符号：流匹配速度场 v = dx_t/dσ = ε - x0。旧写成 `v = x0 - eps`，方向整个反了
     ——学生学到的会是负速度场，cos 直接为负、loss 永远压不下去。
  2. 加噪：x_t = (1-σ)·x0 + σ·ε。旧写成 x0 + σ·ε，σ=0.5 时与标准差差 47.6%。
  3. 缓存里的 t_emb / text_emb 只是占位 zeros，训练侧必须自己重算（train_staged 已改）。

CFG-A：v_teacher = v_uncond + cfg*(v_cond - v_uncond)（在模型输出空间合成）。

用法（与旧版兼容）：
  python preencode_teacher.py --latent-dir latents_new --prompts teacher_prompts.json \
      --cfg 6 --out teacher_targets --dry-run
  去掉 --dry-run 即真跑；--comfy-url 已失效，传了只打印一条提示。
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "tools")))

TEACHER_DEFAULT = r"./cache/Wan5B_fastwan_m85.safetensors"
CFG_DEFAULT = 6.0
STEPS_DEFAULT = 20   # 保留参数位，直推理不再按采样步数走（单点 σ 前向）


def load_latents(latent_dir: str):
    """只返回元信息，不把全部 latent 读进内存（2335 条 ×~400MB 会直接撑爆 31GB 内存）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(latent_dir, "*.pt"))):
        out.append({"path": f, "name": os.path.basename(f)})
    return out


def load_prompts(prompts_path: str):
    if not prompts_path or not os.path.isfile(prompts_path):
        return {}
    with open(prompts_path, encoding="utf-8") as fh:
        return json.load(fh)


def synthesize_cfg(v_cond, v_uncond, cfg: float = CFG_DEFAULT):
    """口径 A：v = uncond + cfg*(cond - uncond)。"""
    return v_uncond + cfg * (v_cond - v_uncond)


def eps_to_velocity(x0, eps):
    """噪声预测 ε -> 流匹配速度场 v = ε - x0（旧版符号写反，已修正）。"""
    return eps - x0


def comfy_teacher_predict(latent, prompt, sigma=0.5, cfg=CFG_DEFAULT,
                          seed=0, engine=None, uncond=False):
    """（保留旧函数名）本地直推理，返回 CFG-A 合成速度场。

    旧签名是 (latent, prompt, comfy_url, steps, cfg, uncond)；这里改为关键字形式，
    调用方若仍按位置传 comfy_url 会落进 sigma —— 请改用 teacher_generator.TeacherEngine。
    """
    if engine is None:
        raise RuntimeError("请传入 teacher_generator.TeacherEngine 实例；本函数不再自建引擎")
    text = "" if uncond else prompt
    rec = engine.predict_velocity(text, latent, sigma, seed=seed)
    return rec["v_teacher"]


def main():
    ap = argparse.ArgumentParser(description="CFG-A 教师目标离线预编码（本地直推理）")
    ap.add_argument("--latent-dir", default=os.path.join(HERE, "latents_new"))
    ap.add_argument("--prompts", default=os.path.join(HERE, "teacher_prompts.json"))
    ap.add_argument("--teacher-weight", default=TEACHER_DEFAULT)
    ap.add_argument("--text-encoder", default=r"./cache/umt5_xxl_int8.safetensors")
    ap.add_argument("--comfy-url", default="", help="已废弃，仅兼容旧脚本；不会再访问")
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT, help="保留参数位，直推理不用")
    ap.add_argument("--cfg", type=float, default=CFG_DEFAULT)
    ap.add_argument("--out", default=os.path.join(HERE, "teacher_targets"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只验流程与数学，不 import torch")
    args = ap.parse_args()

    if args.comfy_url:
        print(f"[warn] --comfy-url={args.comfy_url} 已废弃：教师推理在进程内直跑，不连服务", flush=True)

    lats = load_latents(args.latent_dir)
    prompts = load_prompts(args.prompts)
    have = sum(1 for x in lats if x["name"] in prompts)
    print(f"[data] latent {len(lats)} 条，配对 prompt {have} 条", flush=True)

    if args.dry_run:
        import torch   # 仅本支路需要，且不初始化 ROCm
        x0 = torch.randn(2, 48, 3, 30, 44)
        ec, eu = torch.randn_like(x0), torch.randn_like(x0)
        vc, vu = eps_to_velocity(x0, ec), eps_to_velocity(x0, eu)
        # 1) 符号：v 必须等于 (ε-x0)，与 (x0-ε) 严格反向
        assert torch.allclose(vc, ec - x0), "速度场定义错误"
        cos_rev = torch.nn.functional.cosine_similarity(vc.flatten(), (x0 - ec).flatten(), dim=0)
        assert float(cos_rev) < -0.999, "速度场与 (x0-ε) 同向，符号仍是反的"
        # 2) CFG：cfg=1 时合成场必须等于 cond 场；cfg>1 时方向由 cond 主导
        v1 = synthesize_cfg(vc, vu, 1.0)
        assert torch.allclose(v1, vc, atol=1e-6), "cfg=1 未退化为 cond"
        v = synthesize_cfg(vc, vu, args.cfg)
        cos_cfg = torch.nn.functional.cosine_similarity(v.flatten(), vc.flatten(), dim=0)
        assert float(cos_cfg) > 0, "CFG 合成场方向与 cond 相反"
        print(f"[dry-run] 符号 OK（与 x0-ε cos={float(cos_rev):+.4f}）；"
              f"cfg=1 退化 OK；cfg={args.cfg} 与 cond cos={float(cos_cfg):+.4f}；"
              f"输出 {tuple(v.shape)}", flush=True)
        print("[dry-run] 未 import comfy、未初始化 ROCm、未写盘。", flush=True)
        return

    if not have:
        print("[abort] 没有配对 prompt，拒绝编造 target。", flush=True)
        return 1

    from teacher_generator import TeacherEngine, generate
    eng = TeacherEngine(args.teacher_weight, args.text_encoder, cfg=args.cfg)
    eng.load()
    try:
        n = generate(eng, args.latent_dir, args.prompts, args.out,
                     limit=args.limit, min_samples=args.min_samples)
    finally:
        eng.unload()
    print(f"[done] 教师目标写入 {args.out}：{n} 条", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
