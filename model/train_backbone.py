#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyraGate-VLM v2.0 — 裸主干训练循环（2026-11-22 加固版）

阶段1（当前）：流匹配自监督 / 教师蒸馏，只训 4.5B 主干（GSA + Dual-Act SwiGLU）；
VLM / 北大桥 / cond_mod / t阈值 / OPS / Claude断言 / MoE 三专家 全部不实例化、不进图。

显存安全（11-22 死机教训）：
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
# ============ ROCm / MIOpen 调优（必须在 import torch 之前）============
# 显存碎片治理（本机 ROCm 实测 expandable_segments 不生效，保留无害）
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# MIOpen 算法缓存：首次搜索最优 conv/norm 算法后落盘，后续重启不再重搜
# （本模型大量 LayerNorm，norm 算法次优时每步多 0.2~0.4s）
_MIOPEN_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "miopen_cache")
os.makedirs(_MIOPEN_DB, exist_ok=True)
os.environ.setdefault("MIOpenFindDbPath", _MIOPEN_DB)
os.environ.setdefault("MIOpen_FIND_DB", "1")
os.environ.setdefault("MIOpen_CACHE_MODE", "Normal")
# 关闭 ROCm 内部调试/同步 flush，减少 Windows 下每步驱动开销
os.environ.setdefault("AMD_LOG_LEVEL", "0")
os.environ.setdefault("HSA_ENABLE_SDMA", "1")
os.environ.setdefault("ROCM_USE_FLUSH_DEFER", "1")

import argparse
import glob
import gc
import math
import threading
import time
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uvit3d import UViT3D, UViT3DConfig, count_parameters

C = 48  # latent 通道数（蜂鸟轻量 Wan2.2 VAE；4b5 走 cfg.in_channels=48）。
# 注：旧值 16 是通用 Wan VAE 口径，与本机 VAE 不符，2026-11-23 更正（本文件内无引用，仅防误读）。

OOM_EXIT = 42
CYCLE_EXIT = 43
_save_threads = []


def build_model(kind: str, args) -> UViT3D:
    if kind == "tiny":
        cfg = UViT3DConfig.tiny()
    elif kind == "4b5":
        cfg = UViT3DConfig.v1_baseline_4b5()
    elif kind == "4b5fs":
        # 2026-11-26 新增（AutoClaw）：全栈架构入口（与备份版同步）。
        # 仅在把备份版 uvit3d.py 换入 model/ 后才有意义；主线 uvit3d.py 无 full_stack 工厂时
        # 此分支会 AttributeError，属显式报错而非静默错构。默认训练流程仍用 "4b5"，行为不变。
        cfg = UViT3DConfig.full_stack()
    elif kind == "4b5v3":
        # 2026-11-27 v3：full_stack + bottleneck text cross-attn + global MLP。
        cfg = UViT3DConfig.full_stack_v3()
    else:
        raise ValueError(f"unknown model kind: {kind}")
    cfg.grad_checkpoint = not bool(getattr(args, "no_grad_ckpt", False))
    cfg.ckpt_skip_bottleneck = bool(getattr(args, "ckpt_skip_bottleneck", False))
    cfg.ckpt_last_dec_off = int(getattr(args, "ckpt_last_dec_off", 0))
    cfg.band_lr_rank = int(getattr(args, "band_lr_rank", 0) or 0)
    cfg.band_lr_scope = str(getattr(args, "band_lr_scope", "all") or "all")
    cfg.band_lr_rank_enc = int(getattr(args, "band_lr_rank_enc", 0) or 0)
    cfg.band_lr_rank_bottleneck = int(getattr(args, "band_lr_rank_bottleneck", 0) or 0)
    cfg.band_lr_rank_dec = int(getattr(args, "band_lr_rank_dec", 0) or 0)
    cfg.band_lr_rank_attn = int(getattr(args, "band_lr_rank_attn", 0) or 0)
    cfg.band_lr_rank_ffn = int(getattr(args, "band_lr_rank_ffn", 0) or 0)
    cfg.band_lr_rank_ffn_enc = int(getattr(args, "band_lr_rank_ffn_enc", 0) or 0)
    cfg.band_lr_rank_ffn_bottleneck = int(getattr(args, "band_lr_rank_ffn_bottleneck", 0) or 0)
    cfg.band_lr_rank_ffn_dec = int(getattr(args, "band_lr_rank_ffn_dec", 0) or 0)
    cfg.band_lr_qk_rank = int(getattr(args, "band_lr_qk_rank", 0) or 0)
    cfg.band_lr_vo_rank = int(getattr(args, "band_lr_vo_rank", 0) or 0)
    cfg.band_lr_attn_groups = int(getattr(args, "band_lr_attn_groups", 0) or 0)
    cfg.band_lr_attn = bool(getattr(args, "band_lr_attn", True))
    cfg.band_lr_ffn = bool(getattr(args, "band_lr_ffn", True))
    _fgh = getattr(args, "full_gated_head", None)
    if _fgh is not None:
        cfg.full_gated_head = bool(_fgh)
    cfg.full_head_groups = int(getattr(args, "full_head_groups", 4))
    cfg.full_head_compress = int(getattr(args, "full_head_compress", 4))
    cfg.full_head_ctx_dim = int(getattr(args, "full_head_ctx_dim", 128))
    cfg.full_head_disable = tuple(getattr(args, "full_head_disable", []) or [])
    # 2026-11-26 4b5fs 热启：share_special_ffn=False 时两套 FFN 都保留，避免双键同参覆盖。
    if hasattr(cfg, "share_special_ffn"):
        cfg.share_special_ffn = bool(getattr(args, "share_special_ffn", cfg.share_special_ffn))
    # 4b5fs 定位开关：None=保留 full_stack 默认；显式 True/False 才覆盖。
    for _n in ("a2_crossmodal", "a2_crossmodal_in_attn", "dual_branch_adaln",
               "dual_sem_head", "snr_cond", "local_conv", "pau_syntax_enabled"):
        if hasattr(cfg, _n):
            _v = getattr(args, _n, None)
            if _v is not None:
                setattr(cfg, _n, bool(_v))
    # train_staged 的 CLI 名是 --pau-syntax（args.pau_syntax），不是
    # pau_syntax_enabled；旧 4b5/4b5fs 路线显式开关必须走这个别名。
    _ps = getattr(args, "pau_syntax", None)
    if _ps is not None and hasattr(cfg, "pau_syntax_enabled"):
        cfg.pau_syntax_enabled = bool(_ps)
    # 采样端从 ckpt 恢复的需要；训练端没有 CLI 的字段保留工厂默认。
    for _n in ("pau_tiers", "pau_rank", "pau_gate_init"):
        _v = getattr(args, _n, None)
        if _v is not None and hasattr(cfg, _n):
            setattr(cfg, _n, int(_v) if _n != "pau_gate_init" else float(_v))
    _fh = getattr(args, "dual_sem_ffn_hidden", None)
    if _fh and hasattr(cfg, "dual_sem_ffn_hidden"):
        cfg.dual_sem_ffn_hidden = int(_fh)
    # 2026-11-26 fusion：FreqFormer 频域路由开关透传到 cfg（uvit3d 构造期读取）。
    cfg.freq_route = bool(getattr(args, "freq_route", False))
    cfg.freq_route_rank = int(getattr(args, "freq_route_rank", 8))
    cfg.freq_route_scale = float(getattr(args, "freq_route_scale", 1.0))
    cfg.freq_route_topk = int(getattr(args, "freq_route_topk", 0))
    # 2026-11-27 v3：bottleneck text cross-attn / global MLP 开关。
    # 三态：不传=保留工厂默认；显式 True/False 覆盖。
    _tc = getattr(args, "text_cross_attn", None)
    if _tc is not None:
        cfg.text_cross_attn = bool(_tc)
    if hasattr(args, "text_cross_rank"):
        cfg.text_cross_rank = int(args.text_cross_rank)
    if hasattr(args, "text_cross_base_gate"):
        cfg.text_cross_base_gate = float(args.text_cross_base_gate)
    if hasattr(args, "text_cross_text_dim"):
        cfg.text_cross_text_dim = int(args.text_cross_text_dim)
    if hasattr(args, "text_cross_scale_alpha"):
        cfg.text_cross_scale_alpha = float(args.text_cross_scale_alpha)
    if hasattr(args, "text_cross_freeze_kv"):
        cfg.text_cross_freeze_kv = bool(args.text_cross_freeze_kv)
    if getattr(args, "text_cross_init_wan", None) is not None:
        cfg.text_cross_init_wan = str(args.text_cross_init_wan)
    if hasattr(args, "text_cross_decoder_layers"):
        cfg.text_cross_decoder_layers = int(args.text_cross_decoder_layers)
    if hasattr(args, "adaln_text_layers"):
        _atl = args.adaln_text_layers
        if isinstance(_atl, str):
            _atl = tuple(int(x) for x in _atl.split(",") if x.strip() != "")
        cfg.adaln_text_layers = tuple(int(x) for x in (_atl or ()))
    if hasattr(args, "adaln_text_gate_hidden"):
        cfg.adaln_text_gate_hidden = int(args.adaln_text_gate_hidden)
    _gm = getattr(args, "bottleneck_global_mlp", None)
    if _gm is not None:
        cfg.bottleneck_global_mlp = bool(_gm)
    if hasattr(args, "bottleneck_global_rank"):
        cfg.bottleneck_global_rank = int(args.bottleneck_global_rank)
    if hasattr(args, "attn_spatial_window"):
        cfg.attn_spatial_window = int(args.attn_spatial_window)
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
    的 CPU 拷贝 —— 最坏情况同时只留 1 份 9GB（原来最多 3 份 = 27GB，易触发内存耗尽）。save_every 500 默认也降低频率。
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


def sinusoidal_timestep_embedding(t, dim, max_period=10000.0, t_scale=1000.0):
    """t(标量噪声强度) -> 正弦时间步嵌入 [B,dim]。

    2026-11-23 修复：本文件原来用 `torch.randn` 当 t_emb，与真实 t 完全无关，
    等于把「污染到什么程度」这个条件从模型输入里删掉——同一张 xt 对应无数条
    流方向，只能收敛到坏解（200 步 loss 卡在零预测上方的主因之一）。
    t_scale=1000 对齐 ComfyUI `ModelSamplingDiscreteFlow.timestep(sigma)=sigma*1000`
    与教师 `sinusoidal_embedding_1d` 的口径。
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = (t.float() * t_scale)[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _load_external_text_emb(path, device, dtype, cond_dim):
    """Read offline UMT5 cache; missing/zero must hard-fail, never silently zero."""
    p = str(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"text_emb cache missing: {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    vec = obj.get("text_emb") if isinstance(obj, dict) else obj
    if not torch.is_tensor(vec):
        raise TypeError(f"text_emb cache {p}: {type(vec)} is not tensor")
    vec = vec.float().reshape(-1)
    if vec.numel() != cond_dim:
        raise ValueError(f"text_emb cache {p}: dim {vec.numel()} != cond_dim {cond_dim}")
    if not torch.isfinite(vec).all() or float(vec.abs().max()) == 0.0:
        raise ValueError(f"text_emb cache {p}: zero/non-finite vector")
    return vec.to(device=device, dtype=dtype).unsqueeze(0)


def load_text_tokens(text_dir, name, device, dtype, token_dim=3072):
    """Load 128-token UMT5 cache for v3 bottleneck cross-attn.

    Returns (tokens [1,L,D], mask [1,L] bool). Missing/zero/non-finite must
    hard-fail; a silent fallback would turn the cross-attn into a dead switch.
    """
    p = os.path.join(str(text_dir), os.path.basename(str(name)))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"text token cache missing: {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "text_tokens" not in obj:
        raise TypeError(f"text token cache {p}: missing text_tokens")
    tok = obj["text_tokens"]
    mask = obj.get("text_mask")
    if not torch.is_tensor(tok) or tok.ndim != 2:
        raise TypeError(f"text token cache {p}: text_tokens must be 2D, got {type(tok)}")
    if tok.shape[-1] != token_dim:
        raise ValueError(f"text token cache {p}: dim {tok.shape[-1]} != {token_dim}")
    if mask is None:
        mask = torch.ones(tok.shape[0], dtype=torch.bool)
    mask = mask.to(dtype=torch.bool).reshape(-1)
    if mask.numel() != tok.shape[0]:
        raise ValueError(f"text token cache {p}: mask {mask.numel()} != L {tok.shape[0]}")
    if int(mask.sum()) <= 0:
        raise ValueError(f"text token cache {p}: empty attention mask")
    if not torch.isfinite(tok).all():
        raise ValueError(f"text token cache {p}: non-finite tokens")
    return (tok.to(device=device, dtype=dtype).unsqueeze(0),
            mask.to(device=device).unsqueeze(0))


def make_batch(args, device, dtype, files=None, idx=0, teacher_files=None, buf=None,
               teacher_file=None):
    # 2026-11-26: external UMT5 text_emb cache support is handled below.
    """统一返回 (x0, t_emb, text_emb, teacher_noise_t, target)。
    teacher_noise_t=(noise,t) 或 None；target=v_teacher 或 None。加噪统一在训练循环做。
    buf: 可选预分配张量字典 {'t_emb','text_emb'}，用 out= 原地生成，避免每步新建。
    teacher_file: 直接指定教师缓存文件（配合 §每素材最多 N 遍 的发牌器；
                  给了它就不再按 idx % len 取，避免绕开门禁）。"""
    if teacher_files:
        _tf = teacher_file if teacher_file else teacher_files[idx % len(teacher_files)]
        rec = torch.load(_tf, map_location="cpu",
                         weights_only=False)
        x0 = rec["x0"].to(device=device, dtype=dtype)
        noise = rec["noise"].to(device=device, dtype=dtype)
        t = rec["t"].to(device=device, dtype=dtype).reshape(-1)
        B0 = x0.shape[0]
        if t.numel() == 1 and B0 > 1:
            t = t.expand(B0).contiguous()
        # 2026-11-23：缓存里的 t_emb 只是 zeros(256) 占位（见 teacher_generator），
        # 直接读等于无时间条件。这里由真实 t 现场重算。
        t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
        _text_dir = getattr(args, "text_emb_dir", "") or ""
        if _text_dir:
            # 2026-11-26：A0 真文本修复。按 teacher 文件名读取离线 UMT5 缓存；
            # 不做 zeros 回退，避免“看起来修了、实际还是零条件”。
            _te = _load_external_text_emb(
                os.path.join(_text_dir, os.path.basename(teacher_files[idx % len(teacher_files)])),
                device, dtype, args.cond_dim)
            text_emb = _te.expand(B0, -1).contiguous()
        else:
            text_emb = torch.zeros(B0, args.cond_dim, device=device, dtype=dtype)
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
    # 占位：真实 t 在训练循环采样后才知道，这里先给零，循环内用 sinusoidal 覆盖。
    # （旧实现喂 randn：与 t 无关的随机条件经 AdaLN 注入每个 block，持续污染流场）
    if (buf is not None and buf.get("t_emb") is not None
            and buf["t_emb"].shape == (B, args.t_dim)
            and buf["text_emb"].shape == (B, args.cond_dim)):
        buf["t_emb"].zero_()
        buf["text_emb"].zero_()
        t_emb, text_emb = buf["t_emb"], buf["text_emb"]
    else:
        t_emb = torch.zeros(B, args.t_dim, device=device, dtype=dtype)
        text_emb = torch.zeros(B, args.cond_dim, device=device, dtype=dtype)
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
    ap.add_argument("--no-ckpt", action="store_true", help="不写 checkpoint 文件（不再影响梯度检查点）")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="关闭梯度检查点（用显存换速度；默认开启以适配 21GB 显卡）")
    ap.add_argument("--mem-frac", type=float, default=0.93, help="显存熔断比例，默认 0.93")
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--cycle-steps", type=int, default=1000,
                    help="跑满 N 步主动 exit(43) 让守护清 HIP 碎片；0=禁用（默认 1000，审查建议）")
    ap.add_argument("--init-std", type=float, default=0.02,
                    help="F2 谱初始化 std（教师实测 ~0.02；0=禁用改用 torch 默认）")
    ap.add_argument("--freeze-spectral-after", type=int, default=100,
                    help="F5 预热 N 步后冻结 spectral_norm 幂迭代；0=禁用")
    ap.add_argument("--warmstart", default="",
                    help="教师权重热启动：M85 safetensors 路径；空=禁用。"
                         "直拷 embed/head 级 10 项（R2 修正版），优先级 build>warmstart>F2>resume")
    ap.add_argument("--log-every", type=int, default=20, help="打印间隔（步），精简监控")
    # 2026-11-23 噪声档采样对齐教师：sigma = shift·u/(1+(shift-1)·u)，u~U(0,1)
    ap.add_argument("--t-sched", choices=["teacher", "uniform"], default="teacher")
    ap.add_argument("--t-shift", type=float, default=8.0, help="Wan2.1/2.2 T2V 默认 flow shift=8.0")
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
          f"mem_frac={args.mem_frac} bf16 grad_ckpt={not args.no_grad_ckpt} accum={args.accum}", flush=True)

    model = build_model(args.model, args).to(device).to(dtype).train()
    args.t_dim = model.cfg.t_emb_dim
    args.cond_dim = model.cfg.cond_dim
    args.in_channels = model.cfg.in_channels   # 蜂鸟轻量 VAE=48 通道（合成 latent 与真实对齐）
    # 2026-11-23 顺序修正：F2 必须先跑，再教师热启动 —— 反过来会把刚热启的
    # cond_proj/head 等层重新随机化（warmstart 的花费全白扔）。
    # F2: 教师谱初始化
    if args.init_std > 0:
        from uvit3d import reinit_linear_teacher_spectrum
        n_reinit = reinit_linear_teacher_spectrum(model, std=args.init_std)
        print(f"[init] F2 谱初始化 std={args.init_std} 重设 {n_reinit} 个 Linear", flush=True)
    # 教师热启动（R2 修正版）：build -> F2 -> warmstart -> resume（后者优先级最高）
    if args.warmstart:
        from teacher_warmstart import warmstart_from_teacher, format_report
        ws_rep = warmstart_from_teacher(model, args.warmstart)
        print(format_report(ws_rep), flush=True)
        if not ws_rep["ok"]:
            print("[warn] 教师热启动未完全成功，检查 teacher 路径与形状", flush=True)
    total_p, train_p = count_parameters(model)
    for p in model.parameters():
        p.requires_grad_(True)
    start_step = 1
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        start_step = int(ck.get("step", 0)) + 1
        # 关键：ck 含 7.5GB state_dict，拷进模型后必须立即释放，否则常驻 CPU 内存
        # （实测未释放时 RSS 17.6GB / 可用内存仅 3.5GB / CPU 内核态 66% / 步时 5.3s vs 2.6s）
        del ck
        gc.collect()
        print(f"[resume] {args.resume} → step {start_step}（ck 已释放）", flush=True)
    print(f"[model] {args.model} params={total_p/1e9:.3f}B trainable={train_p/1e9:.3f}B "
          f"dual_act={getattr(model.cfg,'dual_act',None)} use_moe={getattr(model.cfg,'use_moe',None)}",
          flush=True)

    # 2026-11-23：本机这版 Adafactor 的 eps1 被平方后 clamp，None→bf16.eps=0.0078
    # 会把小梯度钳成 128×grad；eps2 是参数尺度下限，1e-30 会让零初始化参数 α≈1e-30*lr
    # 永久冻结（实测 head 梯度 0.0115 却只挪 7.35e-37）。这里显式给 (1e-12, 1e-3)。
    opt = torch.optim.Adafactor(model.parameters(), lr=args.lr, eps=(1e-12, 1e-3))
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

    # F5 谱范数冻结（resume 兼容）：若起始步已越过阈值，启动即冻结，
    # 否则下面 step == freeze_spectral_after 的分支永远不会命中，
    # 导致 12 处幂迭代全程空转、步时翻倍（5.3s vs 2.6s）。
    _f5_done = False

    # 条件张量预分配（消除每步 4 次小张量分配的 ROCm 驱动开销）
    # shape 以首条真实 latent 为准；形状变化时 make_batch 有 fallback，安全
    _buf = {"t_emb": None, "text_emb": None}
    try:
        _probe = (latent_files or [None])[0]
        _B0 = 1
        _buf["t_emb"] = torch.empty(_B0, args.t_dim, device=device, dtype=dtype)
        _buf["text_emb"] = torch.empty(_B0, args.cond_dim, device=device, dtype=dtype)
        print("[opt] 条件张量已预分配（t_emb/text_emb/noise/t 原地复用）", flush=True)
    except Exception as _e:
        print(f"[opt] 预分配跳过：{_e}", flush=True)

    if args.freeze_spectral_after > 0 and start_step > args.freeze_spectral_after:
        from uvit3d import freeze_spectral_power_iteration
        _n = freeze_spectral_power_iteration(model)
        print(f"[init] F5 resume 启动即冻结 {_n} 处谱范数幂迭代"
              f"（start_step={start_step} > {args.freeze_spectral_after}）", flush=True)
        _f5_done = True

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
                    teacher_files or None, buf=_buf if not teacher_files else None)
                if teacher_noise_t is not None:
                    # 教师模式：用教师预编码的同款 noise/t 重建 xt，target=v_teacher
                    noise, t = teacher_noise_t
                else:
                    # 流匹配自监督：现场加噪，target=noise-x0
                    # 预分配 buffer 复用（形状匹配时原地生成），消除每步分配开销
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
                        _tb.copy_((s * _u) / (1.0 + (s - 1.0) * _u))
                    else:
                        _tb.copy_(_u)
                    t = _tb
                    target = noise - x0
                    # 2026-11-23：用真实 t 生成正弦时间嵌入（乘 1000 对齐教师），
                    # 文本置零（阶段1无文本，随机向量会经 cond_proj 污染 AdaLN）
                    t = t.reshape(-1)
                    t_emb = sinusoidal_timestep_embedding(t, args.t_dim).to(dtype)
                    text_emb = torch.zeros(t.shape[0], args.cond_dim, device=device, dtype=dtype)
                tv = t.view(-1, 1, 1, 1, 1).to(dtype)
                xt = (1 - tv) * x0 + tv * noise
                pred = model(xt, t_emb, text_emb, return_aux=False,
                             sigma=t.view(-1, 1).to(dtype))  # 不建辅助头，省显存
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
            if (not _f5_done and args.freeze_spectral_after > 0
                    and step >= args.freeze_spectral_after):
                from uvit3d import freeze_spectral_power_iteration
                n_frozen = freeze_spectral_power_iteration(model)
                print(f"[init] F5 step {step} 冻结 {n_frozen} 处谱范数幂迭代"
                      f"（消检查点重算不确定性）", flush=True)
                _f5_done = True
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

            if args.save_every and not args.no_ckpt and step % args.save_every == 0:
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
