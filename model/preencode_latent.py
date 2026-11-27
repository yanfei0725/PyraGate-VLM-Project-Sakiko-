#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实视频 → Wan VAE latent 离线预编码（绕开网络，用本地视频素材）

背景：以本地视频集合为素材（ComfyUI 产出 + 工具 demo），用 ComfyUI 的
comfy.sd.VAE 加载 Wan2.2 VAE，把视频采样成短片段编码成
latent [48,T,H,W]（蜂鸟轻量 Wan2.2 VAE=48 通道，空间 16× / 时间 4× 压缩），
落盘 .pt，供 train_backbone.py 做真实数据训练 / 教师蒸馏输入。

用法（2026-11-24 新默认：768×448 等比缩放+padding，不拉伸）：
  python preencode_latent.py --video-list outputs/dataset_768/video_list_500.txt \
      --out model/latents_13f_768_500 --frames 13 --height 448 --width 768 \
      --pad-mode replicate --workers 4 --prefetch 4

输出：out/000000.pt ... 每个 {"latent":[1,48,T,H,W], "src":路径}
      768×448/13f 实测 (1,48,4,28,48)；CPU 线程池预取与 GPU encode 流水线重叠。
"""
import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import sys
import time

import torch

COMFY = os.environ.get("COMFYUI_DIR", "")
if COMFY:
    sys.path.insert(0, COMFY)
# 复用蜂鸟 XT 轻量 VAE（TurboVAED 同源，1.4GB bf16）
# 不依赖 ComfyUI API/服务，只借它的 VAE 加载与编码实现
VAE_PATH = r"./cache/Hummingbird-Wan2.2-VAE_bf16.safetensors"


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


def _resize_pad(x, height, width, pad_mode="replicate"):
    """等比缩放到目标框内，再中心 padding；绝不各向异性拉伸。

    x: [T,C,H0,W0] 或 [T,C,H,W]，[0,1]；返回 [T,C,height,width]。
    下采样统一 antialias=True；不做锐化/对比度拉伸。
    """
    h0, w0 = int(x.shape[-2]), int(x.shape[-1])
    if (h0, w0) == (height, width):
        return x
    scale = min(height / h0, width / w0)
    nh = max(1, min(height, int(round(h0 * scale))))
    nw = max(1, min(width, int(round(w0 * scale))))
    x = torch.nn.functional.interpolate(
        x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    pl = (width - nw) // 2
    pr = width - nw - pl
    pt = (height - nh) // 2
    pb = height - nh - pt
    if pl or pr or pt or pb:
        if pad_mode == "constant":
            x = torch.nn.functional.pad(x, (pl, pr, pt, pb), mode="constant", value=0.0)
        else:
            x = torch.nn.functional.pad(x, (pl, pr, pt, pb), mode="replicate")
    return x


def read_frames(path, n_frames, height, width, pad_mode="replicate"):
    """decord 均匀采样 n_frames 帧，等比缩放+中心 padding 到 (H,W)，返回 [T,H,W,3] float [0,1]。
    comfy VAE.encode 期望 [B,H,W,C]（图）/[B,T,H,W,C]（视频），内部 process_input 做 x*2-1。

    2026-11-24：不再用 bilinear 直接拉成 704×480（22:15 各向异性）。
    默认 768×448 是 Wan 480p 桶的 16:9 近似；1920×1080 源等比缩到 768×432，上下共补 16px。
    """
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    if total < 2:
        return None
    n = min(n_frames, total)
    idxs = [int(i * (total - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
    imgs = vr.get_batch(idxs).asnumpy()            # [n,H0,W0,3] uint8
    # 黑片/废片预筛：std 太低没有训练价值，跳过
    if float(imgs.std()) < 5.0:
        return None
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0    # [n,3,H0,W0] [0,1]
    x = _resize_pad(x, height, width, pad_mode=pad_mode)
    return x.permute(0, 2, 3, 1).contiguous()      # [T,H,W,3] [0,1]


def main():
    global VAE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="", help="源视频目录；留空时需给 --video-list")
    ap.add_argument("--video-list", default="", help="文本文件，每行一个源视频绝对路径；给定时优先于 video-dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "latents_real"))
    ap.add_argument("--vae", default=VAE_PATH)
    ap.add_argument("--limit", type=int, default=200, help="最多编码多少个视频")
    ap.add_argument("--start", type=int, default=0, help="跳过排序后的前 N 个视频（用于只编码新素材）")
    ap.add_argument("--exclude-src", default="", help="文本文件，每行一个源视频 basename；命中的跳过（用于排除已训素材）")
    ap.add_argument("--frames", type=int, default=9, help="每视频采样帧数（Wan VAE 需 4k+1；14 会被裁到 13）")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=768, help="默认 768（Wan 480p 16:9 近似；旧数据是 704）")
    ap.add_argument("--pad-mode", choices=("replicate", "constant"), default="replicate",
                    help="等比缩放后中心 padding 的模式；默认 replicate，避免黑边")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=4,
                    help="CPU 预取线程数：decord 解码+等比缩放+padding 与 GPU encode 并行")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="CPU 预取队列深度（每条 13f 768×448 float32 约 54MB，别开太大）")
    args = ap.parse_args()

    VAE_PATH = args.vae
    assert torch.cuda.is_available(), "需要 GPU"
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    vids = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    if args.video_list and os.path.isfile(args.video_list):
        with open(args.video_list, "r", encoding="utf-8") as f:
            vids = [line.strip() for line in f if line.strip()]
        vids = [v for v in vids if os.path.isfile(v)]
        print(f"[data] video-list {args.video_list} -> {len(vids)} 个有效视频", flush=True)
    if args.exclude_src and os.path.isfile(args.exclude_src):
        with open(args.exclude_src, "r", encoding="utf-8") as f:
            banned = {line.strip() for line in f if line.strip()}
        before = len(vids)
        vids = [v for v in vids if os.path.basename(v) not in banned]
        print(f"[data] exclude-src {len(banned)} 条，视频 {before} -> {len(vids)}", flush=True)
    vids = vids[args.start:args.start + args.limit]
    print(f"[data] 找到 {len(vids)} 个视频，开始加载 VAE ...", flush=True)
    vae = load_vae(device)
    print("[vae] 加载完成", flush=True)

    ok = 0
    t0 = time.time()
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def _load_task(idx, vp):
        return idx, vp, read_frames(vp, args.frames, args.height, args.width,
                                    pad_mode=args.pad_mode)

    # GPU+CPU 流水线：CPU 线程池提前解码/缩放/补边，GPU 只做 VAE encode。
    # 之前串行跑 100 条 121s；这里把 CPU 帧准备藏在 GPU encode 后面。
    pending = deque()
    submit_i = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        while submit_i < len(vids) or pending:
            while submit_i < len(vids) and len(pending) < max(1, args.prefetch):
                pending.append(ex.submit(_load_task, submit_i, vids[submit_i]))
                submit_i += 1
            fut = pending.popleft()
            i = -1
            try:
                idx, vp, frames = fut.result()
                i = idx
                if frames is None:
                    continue
                if frames.numel() and hasattr(frames, "pin_memory"):
                    try:
                        frames = frames.pin_memory()
                    except Exception:
                        pass
                # 输入 [T,H,W,C] -> [B,C,T,H,W]
                frames = frames.unsqueeze(0).movedim(-1, 1).to(
                    device=device, dtype=torch.float32, non_blocking=True)
                vae.fs.eval()
                with torch.inference_mode():
                    # 绕开 comfy.sd.VAE.encode 的 tiled fallback（该路径有 out 未赋值 bug）
                    pi = vae.process_input(frames).to(vae.vae_dtype).to(device)
                    lat = vae.first_stage_model.encode(pi)
                    latent = lat.detach().float().cpu().contiguous()
                outp = os.path.join(args.out, f"{i:06d}.pt")
                torch.save({"latent": latent, "src": vp}, outp)
                ok += 1
                if ok % 10 == 0 or ok == 1:
                    print(f"[enc {ok}/{len(vids)}] i={i} shape={tuple(latent.shape)} "
                          f"{time.time()-t0:.0f}s", flush=True)
                del frames, pi, lat, latent
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] idx={i} 跳过", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[skip] idx={i}: {type(e).__name__} {str(e)[:100]}", flush=True)
                continue
    print(f"[done] 编码成功 {ok}/{len(vids)}，耗时 {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
