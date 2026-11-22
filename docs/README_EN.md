# PyraGate-VLM · 3D U-ViT Backbone (Udit) — English Overview

> 中文详见 [README.md](../README.md)。This file is a condensed English summary of the Chinese README.

Pure-PyTorch video flow-matching DiT backbone. No diffusers / flash-attn hard dependencies; instantiates and self-tests on pure CPU; GPU forward and training pipeline verified on AMD Radeon RX 7900 XT (ROCm 7.2.1, Windows).

## Model at a glance

- **Scale**: 4.0B parameters (meta-device verified: 3,999,944,893). Symmetric U-shape: encoder 12 / bottleneck 6 / decoder 12 (30 blocks), dim 3072, 24 heads (head_dim 128).
- **Conditioning**: teacher-style AdaLN — a shared two-layer embedder plus per-block (1, 6, dim) zero-init tables (replaces per-block `Linear(dim, 6*dim)`; saves ~1.45B parameters / ~2.9GB of weights).
- **Attention**: 2+1D spatial/temporal decomposition (alternating), full-3D joint bottleneck; optional GSA gated-mixing strategy (`attn_strategy="gsa"`); 3D RoPE with t/h/w frequency split using a real-valued fp32 shared table (batched, no per-sample loop).
- **FFN**: Dual-Act piecewise-activation SwiGLU (`|z| < t` → SiLU, `|z| ≥ t` → smooth compression, learnable threshold; parameter parity with a native ratio-4 GELU MLP). w1/w2 fused into one GEMM; q/k/v fused into one GEMM.
- **Output head**: velocity main head + temporal aux head + DDT-style high-frequency detail head (all zero-gated, exactly identity at init).
- **Condition extension**: VLM north bridge (4096→3072, strict-load pretrained weights supported) + low-rank per-token CrossAttn injection chain; hierarchical style-condition encoder (72M) and reference-frame resample-alignment encoder (68.5M).
- **MoE (stage 2, optional)**: global conditional-space experts `GlobalConditionalMoE` (3 experts × ~330M = 0.991B), Top1 hard routing with straight-through gradients, α-gated additive injection into blocks 7–12; α=0 is a bit-exact fallback to the expert-free trunk.
- **Distillation setup**: flow-matching velocity MSE against a 20-step teacher pipeline (temporal anchors: flicker 0.0244 / identity 0.9892); four-term loss (velocity + layer-feature alignment + temporal consistency + flicker regularizer).

## Repository layout

| Path | Content |
|---|---|
| `model/uvit3d.py` | Full backbone structure (AdaLN / 2+1D / GSA / SwiGLU / GatedSkip / pyramid / output heads / global MoE / real-form RoPE) |
| `model/vlm_bridge.py` | VLM north bridge (4096→3072) + multi-scale layered injector |
| `model/cond_encoders.py` | Style-condition encoder + reference-alignment encoder |
| `model/attn_backend.py` | Attention backend chain: flash_attn (Triton) → SDPA FLASH → SDPA → math |
| `model/distill_loss.py` | Four-term distillation loss (differentiable) |
| `model/quant_plan.py` | Mixed-precision quantization layer plan (sensitive layers fp16, others int8) |
| `model/train_backbone.py` | Training loop (Adafactor, bf16, checkpointing, pre-encoded teacher-target mode) |
| `model/run_training_daemon.py` | Training daemon (crash restart, OOM degrade ladder, fragmentation-clearing cycles) |
| `model/preencode_latent.py` | Video → VAE latent offline pre-encoding |
| `model/preencode_teacher.py` | CFG-A teacher velocity-target pre-encoding scaffold |
| `model/teacher_warmstart.py` | Shape-exact embed/head warm-start from teacher checkpoint |
| `model/selftest_uvit3d.py` | Pure-CPU selftest (139 checks incl. bitwise regressions) |
| `model/sakiko_op/` | Optional gated-residual custom operator package (graceful fallback if absent) |
| `docs/TECHNICAL_WHITEPAPER.md` | Full technical whitepaper (CN) |

## Quick start (CPU selftest)

```bash
cd model
python selftest_uvit3d.py       # 139 structural checks
python distill_loss.py
python quant_plan.py
python ablation_matrix.py
python engineering_hygiene.py
```

CI runs the same suite on every push (see `.github/workflows/selftest.yml`).

## Platform notes (ROCm / Windows, measured)

- **Attention backend**: default `flash2`. With community Triton-backend wheels (`flash_attn 2.8.4 win-rocm` + `amd_aiter`) it resolves to the Triton FA2 implementation — on RDNA3 mid/long sequences it measured 1.8–2.3× faster than SDPA; without the package it falls back to torch SDPA FLASH (3.45× vs math); CPU/fp32 degrades to math explicitly.
- **torch.compile**: measured harmful on this platform (first compile 89–210s, then 2.4–5.6× slower — inductor lacks complex-op support); not recommended.
- **Training memory**: bf16 full-parameter (~14.3GB weights+grads at 4.0B) + activation checkpointing + Adafactor; `mem_frac` fuse default 0.93, hard-capped 0.95; periodic clean restart clears HIP fragmentation; checkpoint saving is serialized to avoid multi-threaded 9GB state_dict RAM spikes.

## Verification status

- **CPU selftest: 139 checks passing** — gate=0 bitwise identity regressions (`torch.equal`), real-vs-complex RoPE equivalence (≤1 ULP), F.rms_norm parity, checkpoint forward/backward equivalence, 4.0B parameter hard assertion, zero-gate identities, MoE α=0 fallback.
- **GPU verified (RX 7900 XT 21.46GB)**: 4.0B bf16 forward (peak 11.91GB / 3.8s at T=16, 64×64 latent); Triton kernel forward error 2.7e-3 with aligned gradients; Triton-backend FA2 forward+backward parity across 5 configurations.

## Honest boundaries

1. **No completed distillation training yet**: losses and pipeline are defined and self-tested; the training line depends on data and the pre-encoded teacher-target flow.
2. **Forward pass ≠ quality target met**: structure, red lines and numerics are automated-verified; quality metrics require rendered-video evaluation.
3. **GSA is a skeleton**: gate source currently uses the pooled condition vector (per-token features are a drop-in future change); GSA-vs-2+1D needs a real ablation.
4. **VAE is path configuration only**; decode/tiled-decode belongs to the deployment chain, not this backbone.
5. Deployment-side mechanisms (scheduling / caching / verification chains / output post-processing) are described in the whitepaper, not part of this repository.

## License

[BSL 1.1](../LICENSE): source-available, copy/modify/redistribute allowed, non-production use permitted; converts to GPL-3.0 on 2030-01-01 (Change Date). Production/commercial licensing: contact the authors.

## Citation

See [CITATION.cff](../CITATION.cff).
