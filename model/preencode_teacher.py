#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFG-A 教师目标离线预编码（CFG-A 合成场口径）。

口径 A：教师按 CFG6 合成速度场作为蒸馏 target
    v_teacher = v_uncond + 6 * (v_cond - v_uncond)
学生单通道学会它 -> 推理 CFG-free，成本砍半，且对齐冠军管线真实采样轨迹。

教师 = Wan5B M85 混血权重（Wan5B_fastwan_m85.safetensors + 时序低通算子，
flicker 0.0244 / 身份 0.9892）。教师 10GB 不能与学生共驻显存 -> 离线预编码，
训练时 train_backbone.py 只读本脚本产出的 {x0,noise,t,t_emb,text_emb,v_teacher}。

★ 数据前置（当前阻塞，如实标注）：
  453 条 latents_real/*.pt 来自 ComfyUI 历史生成，只有 {latent,src}，
  **无配对文本 prompt**。CFG-A 必须对每个 latent 有对应 prompt 才能跑
  cond(带文本)/uncond(空文本) 两遍。数据到位的两种方式：
    1) 提供 prompts.json：{latent文件名: prompt文本}，本脚本直接读；
    2) 从 src 视频反推 caption（需本地 captioner，非本脚本职责）。
  无 prompt 时本脚本 --dry-run 只验流程不产目标，绝不编造 target。

教师推理链路（本地资源已勘察齐全）：
  - 教师权重（Wan5B_fastwan_m85.safetensors，10GB；路径经 --teacher-weight 或 PYRAGATE_TEACHER 指定）
  - ComfyUI Wan 节点（comfy/ldm/wan + comfy_extras/nodes_wan.py）
  - 文本编码器（UMT5-XXL，hidden=4096）
  - 时序低通算子：教师混血定义见白皮书 V2-21
  本脚本通过 ComfyUI 的 WAN 推理 API（--comfy-url）跑 cond/uncond 两遍，
  取 20 步轨迹的噪声预测 epsilon 换算速度场 v = x0 - eps（流匹配口径），再 CFG 合成。

用法：
  # 1) 先 dry-run 验证流程与数据接口（不跑 GPU）
  python preencode_teacher.py --latent-dir latents_real --prompts prompts.json --dry-run
  # 2) prompt 到位 + ComfyUI 起来后跑真预编码
  python preencode_teacher.py --latent-dir latents_real --prompts prompts.json \
      --teacher-weight /path/to/Wan5B_fastwan_m85.safetensors --comfy-url http://127.0.0.1:8188 \
      --steps 20 --cfg 6 --out teachers_real
"""
import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TEACHER_DEFAULT = os.environ.get("PYRAGATE_TEACHER", "Wan5B_fastwan_m85.safetensors")
CFG_DEFAULT = 6.0
STEPS_DEFAULT = 20   # 教师 20 步、学生跟 20 步（对齐教师采样，非少步设定）


def load_latents(latent_dir: str):
    fs = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))
    out = []
    for f in fs:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lat = d.get("latent") if isinstance(d, dict) else d
        if lat is None:
            continue
        out.append({"path": f, "name": os.path.basename(f),
                    "x0": lat.float(), "src": d.get("src", "") if isinstance(d, dict) else ""})
    return out


def load_prompts(prompts_path: str):
    if not prompts_path or not os.path.isfile(prompts_path):
        return {}
    with open(prompts_path, encoding="utf-8") as fh:
        return json.load(fh)


def synthesize_cfg(v_cond: torch.Tensor, v_uncond: torch.Tensor,
                   cfg: float = CFG_DEFAULT) -> torch.Tensor:
    """口径 A：v = uncond + cfg*(cond - uncond)。"""
    return v_uncond + cfg * (v_cond - v_uncond)


def eps_to_velocity(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """噪声预测 eps -> 流匹配速度场 v = x0 - eps。"""
    return x0 - eps


def comfy_teacher_predict(latent: torch.Tensor, prompt: str, comfy_url: str,
                          steps: int, cfg: float, uncond: bool = False):
    """经 ComfyUI WAN 推理 API 取教师噪声预测。★ 需接入本地工作流后实装。

    当前返回 None 并给出明确指引——不编造预测值。接入点：
      POST {comfy_url}/prompt  with Wan2.2 T2V/I2V workflow，
      文本节点喂 prompt（uncond 时喂空串 ""），采样步数=steps，
      从 KSampler/WanSampler 输出取 latent 噪声 -> 转 eps。
    """
    raise NotImplementedError(
        "教师推理需接入本地 ComfyUI Wan 工作流（blueprints/"
        "Text to Video (Wan 2.2).json）。步骤：\n"
        "  1) 启动 ComfyUI API（--comfy-url）\n"
        "  2) 加载 M85 权重 + UMT5-XXL 文本编码器 + 时序低通算子\n"
        "  3) 对每个 latent 跑 cond(prompt) 与 uncond('') 各一遍，steps=20\n"
        "  4) 取两遍 eps，经 eps_to_velocity + synthesize_cfg 合成 v_teacher\n"
        "本脚本拒绝在未接入时编造 target（会毁掉整轮训练）。")


def main():
    ap = argparse.ArgumentParser(description="CFG-A 教师目标离线预编码")
    ap.add_argument("--latent-dir", default=os.path.join(HERE, "latents_real"))
    ap.add_argument("--prompts", default="", help="prompt 映射 JSON：{latent文件名: 文本}")
    ap.add_argument("--teacher-weight", default=TEACHER_DEFAULT)
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--cfg", type=float, default=CFG_DEFAULT)
    ap.add_argument("--out", default=os.path.join(HERE, "teachers_real"))
    ap.add_argument("--dry-run", action="store_true", help="只验流程与数据接口，不跑 GPU")
    args = ap.parse_args()

    lats = load_latents(args.latent_dir)
    prompts = load_prompts(args.prompts)
    have = sum(1 for x in lats if x["name"] in prompts)
    print(f"[data] latent {len(lats)} 条，配对 prompt {have} 条", flush=True)
    if have < len(lats):
        miss = [x["name"] for x in lats if x["name"] not in prompts][:5]
        print(f"[block] 缺 prompt（前5）：{miss}", flush=True)
        print("[block] CFG-A 需要每条 latent 的 prompt 才能跑 cond/uncond 两遍。"
              "无 prompt 时不编造 target。", flush=True)

    if args.dry_run:
        # 用随机 eps 验合成场数学（仅流程自检，不写盘当真 target）
        x0 = torch.randn(2, 8, 4, 4)
        ec, eu = torch.randn_like(x0), torch.randn_like(x0)
        vc, vu = eps_to_velocity(x0, ec), eps_to_velocity(x0, eu)
        v = synthesize_cfg(vc, vu, args.cfg)
        manual = vu + args.cfg * (vc - vu)
        assert torch.allclose(v, manual), "CFG 合成场数学错误"
        print(f"[dry-run] CFG-A 合成场数学 OK：v=uncond+{args.cfg}*(cond-uncond)，"
              f"输出 shape {tuple(v.shape)}。教师 20 步。", flush=True)
        print("[dry-run] 未跑 GPU、未写 target（prompt 数据未到位）。", flush=True)
        return

    if have < len(lats):
        print("[abort] prompt 未配齐，终止真预编码（拒绝编造 target）。", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    for item in lats:
        p = prompts[item["name"]]
        # cond / uncond 两遍教师预测（需接入 ComfyUI Wan 工作流）
        eps_c = comfy_teacher_predict(item["x0"], p, args.comfy_url,
                                      args.steps, args.cfg, uncond=False)
        eps_u = comfy_teacher_predict(item["x0"], "", args.comfy_url,
                                     args.steps, args.cfg, uncond=True)
        v_teacher = synthesize_cfg(
            eps_to_velocity(item["x0"], eps_c),
            eps_to_velocity(item["x0"], eps_u), args.cfg)
        # 采样 noise/t（流匹配），保存学生训练目标
        noise = torch.randn_like(item["x0"])
        t = torch.rand(1)
        t_emb = torch.zeros(256)   # 占位：实际 t_emb 由训练脚本 timestep 编码
        text_emb = torch.zeros(4096)  # 占位：实际 UMT5 hidden 由文本编码器产出
        torch.save({"x0": item["x0"], "noise": noise, "t": t,
                    "t_emb": t_emb, "text_emb": text_emb, "v_teacher": v_teacher},
                   os.path.join(args.out, item["name"]))
    print(f"[done] 教师目标写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
