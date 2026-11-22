#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实视频 → Wan VAE latent 离线预编码（绕开网络，用本机已有视频）— 2026-11-22

背景：从本地已有视频离线预编码 latent。用 ComfyUI 的 comfy.sd.VAE 加载 Wan2.2 VAE，把视频采样成短片段编码成
latent [16,T,H,W]，落盘 .pt，供 train_backbone.py 做真实数据训练 / 教师蒸馏输入。

用法：
  python preencode_latent.py --video-dir <video-dir> \
      --out ./latents_real --limit 200 \
      --frames 9 --height 480 --width 704

输出：out/000000.pt ... 每个 {"latent":[1,16,T,H,W], "src":路径}
"""
import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import os
import sys
import time

import torch

COMFY = os.environ.get("PYRAGATE_COMFY_ROOT", "ComfyUI")
sys.path.insert(0, COMFY)
# 复用本地轻量 VAE（调研报告锁定，TurboVAED 同源，1.4GB bf16）
# 不依赖 ComfyUI API/服务，只借它的 VAE 加载与编码实现
VAE_PATH = os.environ.get("PYRAGATE_VAE_PATH", "Hummingbird-Wan2.2-VAE_bf16.safetensors")


def load_vae(device):
    import comfy.sd
    import comfy.utils
    # 与 ComfyUI VAELoader 节点一致：load_torch_file 处理 metadata，VAE() 不传 device/dtype
    sd, metadata = comfy.utils.load_torch_file(VAE_PATH, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    vae.fs = vae.first_stage_model     # 暴露给直接 encode（绕开 tiled fallback bug）
    vae.fs = vae.fs.to(device=device, dtype=torch.bfloat16).eval()  # 显式搬 GPU
    del sd
    return vae


def read_frames(path, n_frames, height, width):
    """decord 均匀采样 n_frames 帧 resize 到 (H,W)，返回 [T,H,W,3] float [0,1]。
    comfy VAE.encode 期望 [B,H,W,C]（图）/[B,T,H,W,C]（视频），内部 process_input 做 x*2-1。"""
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    if total < 2:
        return None
    n = min(n_frames, total)
    idxs = [int(i * (total - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
    imgs = vr.get_batch(idxs).asnumpy()            # [n,H0,W0,3] uint8
    # 黑片/废片预筛：std 太低没有训练价值，跳过（本机 output 里约 13% 是全黑 demo）
    if float(imgs.std()) < 5.0:
        return None
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0    # [n,3,H0,W0] [0,1]
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()      # [T,H,W,3] [0,1]


def main():
    global VAE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default=os.path.join(COMFY, "output"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "latents_real"))
    ap.add_argument("--vae", default=VAE_PATH)
    ap.add_argument("--limit", type=int, default=200, help="最多编码多少个视频")
    ap.add_argument("--frames", type=int, default=9, help="每视频采样帧数（Wan VAE 需 4k+1）")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    args = ap.parse_args()

    VAE_PATH = args.vae
    assert torch.cuda.is_available(), "需要 GPU"
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    vids = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    vids = vids[:args.limit]
    print(f"[data] 找到 {len(vids)} 个视频，开始加载 VAE ...", flush=True)
    vae = load_vae(device)
    print("[vae] 加载完成", flush=True)

    ok = 0
    t0 = time.time()
    for i, vp in enumerate(vids):
        try:
            frames = read_frames(vp, args.frames, args.height, args.width)
            if frames is None:
                continue
            # 输入 [B,T,H,W,C] -> [B,C,T,H,W]（first_stage_model.encode 直接要的格式）
            frames = frames.unsqueeze(0).movedim(-1, 1).to(device=device, dtype=torch.float32)
            vae.fs.eval()
            with torch.inference_mode():
                # 绕开 comfy.sd.VAE.encode 的 tiled fallback（该路径在本 VAE 上有 out 未赋值 bug）
                # 直接调 first_stage_model.encode，已实测可用
                pi = vae.process_input(frames).to(vae.vae_dtype).to(device)
                lat = vae.first_stage_model.encode(pi)
                latent = lat.detach().float().cpu().contiguous()   # [B,16,T',H/8,W/8]
            outp = os.path.join(args.out, f"{i:06d}.pt")
            torch.save({"latent": latent, "src": vp}, outp)
            ok += 1
            if ok % 10 == 0 or i == 0:
                print(f"[enc {i+1}/{len(vids)}] ok={ok} shape={tuple(latent.shape)} "
                      f"{time.time()-t0:.0f}s", flush=True)
            del frames, pi, lat, latent
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] {vp} 跳过", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[skip] {vp}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
    print(f"[done] 编码成功 {ok}/{len(vids)}，耗时 {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
