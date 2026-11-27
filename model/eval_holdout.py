import argparse
import glob
import json
import math
import os
import sys

import torch

ROOT = r"."
sys.path.insert(0, os.path.join(ROOT, "model"))
from train_backbone import build_model, sinusoidal_timestep_embedding  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model", "ckpt_v2", "backbone_step5200.pt"))
    ap.add_argument("--holdout", default=os.path.join(ROOT, "model", "latents_13f_holdout"))
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "holdout_eval_5200.json"))
    ap.add_argument("--draws", type=int, default=2, help="每条素材重复抽 t/noise 的次数")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--t-shift", type=float, default=8.0)
    ap.add_argument("--mem-frac", type=float, default=0.90)
    return ap.parse_args()


def main():
    a = parse()
    files = sorted(glob.glob(os.path.join(a.holdout, "*.pt")))
    if not files:
        print(f"[eval] no holdout files in {a.holdout}")
        return
    print(f"[eval] ckpt={a.ckpt} holdout={len(files)} draws={a.draws}")
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    device, dtype = "cuda", torch.bfloat16

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = argparse.Namespace(no_ckpt=True)
    model = build_model("4b5", args)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).to(dtype).eval()
    torch.set_grad_enabled(False)

    t_dim = model.cfg.t_emb_dim
    cond_dim = model.cfg.cond_dim
    per_file = []
    tot_loss = tot_zero = tot_cos = 0.0
    n = 0
    for fi, fp in enumerate(files):
        rec = torch.load(fp, map_location="cpu", weights_only=False)
        x0 = rec["latent"].to(device=device, dtype=dtype)
        fl = fz = fc = 0.0
        for d in range(a.draws):
            g = torch.Generator(device="cpu").manual_seed(a.seed + fi * 97 + d)
            u = torch.rand(1, generator=g)
            sigma = (a.t_shift * u) / (1.0 + (a.t_shift - 1.0) * u)
            t = sigma.to(device=device, dtype=torch.float32)
            noise = torch.randn(x0.shape, generator=g).to(device=device, dtype=dtype)
            tv = t.view(-1, 1, 1, 1, 1).to(dtype)
            xt = (1 - tv) * x0 + tv * noise
            target = noise - x0
            t_emb = sinusoidal_timestep_embedding(t, t_dim).to(dtype)
            text_emb = torch.zeros(x0.shape[0], cond_dim, device=device, dtype=dtype)
            pred = model(xt, t_emb, text_emb, return_aux=False, sigma=t.view(-1, 1))
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            zero = torch.nn.functional.mse_loss(torch.zeros_like(target).float(), target.float())
            cos = torch.nn.functional.cosine_similarity(
                pred.float().flatten(1), target.float().flatten(1), dim=1).mean()
            fl += float(loss); fz += float(zero); fc += float(cos); n += 1
        fl /= a.draws; fz /= a.draws; fc /= a.draws
        tot_loss += fl; tot_zero += fz; tot_cos += fc
        per_file.append({"file": os.path.basename(fp), "loss": fl, "zero": fz, "cos": fc})
        if (fi + 1) % 20 == 0:
            print(f"[eval] {fi+1}/{len(files)} loss={fl:.4f} zero={fz:.4f}", flush=True)

    res = {
        "ckpt": a.ckpt,
        "n_files": len(files),
        "draws": a.draws,
        "avg_loss": tot_loss / len(files),
        "zero_avg": tot_zero / len(files),
        "cos": tot_cos / len(files),
        "gap_vs_zero": (tot_zero - tot_loss) / len(files),
        "t_shift": a.t_shift,
        "per_file": per_file,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[eval] RESULT avg_loss={res['avg_loss']:.5f} zero_avg={res['zero_avg']:.5f} "
          f"gap={res['gap_vs_zero']:.5f} cos={res['cos']:+.4f} n={res['n_files']} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
