# PyraGate-VLM · 3D U-ViT Backbone (Project Sakiko) — English Overview

> 中文详见 [README.md](../README.md)。Condensed English overview of the project.

Pure-PyTorch video flow-matching DiT backbone. No diffusers / flash-attn hard dependencies; instantiates and self-tests on CPU; GPU forward and staged training validated on AMD Radeon RX 7900 XT (ROCm 7.2.1, Windows).

## Model at a glance

- **Scale**: 4.0B parameters (meta-device verified: 3,999,944,893). Symmetric U-shape: encoder 12 / bottleneck 6 / decoder 12 (30 blocks), dim 3072, 24 heads (head_dim 128).
- **Conditioning**: teacher-style AdaLN — shared two-layer embedder plus per-block (1, 6, dim) zero-init tables (replaces per-block `Linear(dim, 6*dim)`, saves ~1.45B parameters).
- **Attention**: 2+1D spatial/temporal decomposition (alternating), full-3D joint bottleneck; optional GSA gated-mixing strategy; 3D RoPE with t/h/w frequency split using a real-valued fp32 shared table (batched).
- **FFN**: Dual-Act piecewise SwiGLU (`|z|<t` SiLU, `|z|>=t` compressed slope, learnable threshold; parameter parity with native ratio-4 MLP). w1/w2 and q/k/v fused into single GEMMs.
- **Full-gated output head**: velocity main head + temporal aux head + high-frequency detail residual head (zero-gated at init); frequency-band residual heads (low/mid/high, band-limited in latent space), embedded dual-act gate, 2D relative position embedding, anti-aliasing, pyramid shortcut, per-frame high-pass with sigma attenuation.
- **Conditional MoE (stage 2, optional)**: global conditional-space 3 experts (~330M each, 0.991B total), Top1 hard routing with straight-through gradients, alpha-gated additive injection into blocks 7-12; alpha=0 is a bit-exact fallback.
- **v3 architecture extensions** (all off by default, identity-at-init):
  - `text_cross`: low-rank UMT5 token cross-attention on selected blocks, sigma-gated via a single shared sigma-gate entry; optional K/V freeze and init from official cross-attention weights.
  - `AdaLNZeroText`: low-rank text-pooled AdaLN parallel residual, zero-initialized.
  - `SigmaSlidingAdapter`: shared low-rank residual active at high sigma.
  - `TierContextTokens`: 4-tier register tokens (scene/actor/part/pixel) injected via AdaLN without changing the token-count grid.
  - `SyntaxCompiler`: PAU-style scheduling compiler (skip / ffn-mod / cross-budget / tier-mod); its auxiliary loss keeps a differentiable path while control signals are detached before backbone injection.
  - `FreqRouter`: FFT-descriptor per-head frequency-band routing bias on attention keys; feeds only self tokens (cross segments padded with zero bias), identity fallback on degenerate grids.
- **Training**: staged self-supervised / distillation loop with teacher target swap, low-sigma backfill, multiple auxiliary losses (frame / token-boundary / velocity-magnitude / spectrum / boundary / REPA / syntax), optimizer groups for gates and new modules, resume shape/name guards.

## Repository layout

| Path | Content |
|---|---|
| `model/uvit3d.py` | Full backbone (AdaLN / 2+1D / GSA / SwiGLU / GatedSkip / pyramid / output head / global MoE / real RoPE / v3 hooks) |
| `model/full_gated_head.py` | Full gated token head (band heads, dual-act gate, RPE, anti-alias, pyramid, high-pass) |
| `model/text_cross_attn.py` | v3 text cross-attn, text AdaLN, sigma adapter, tier tokens, shared sigma gate |
| `model/syntax_compiler.py` | PAU syntax compiler + tier scheduling |
| `model/freq_router.py`, `fft_shared.py`, `x0_fourier_band_loss.py` | Frequency router, shared FFT cache, Fourier-band losses |
| `model/train_staged.py`, `train_backbone.py` | Staged trainer and baseline trainer |
| `model/preencode_latent.py`, `preencode_teacher.py` | Offline latent pre-encoding and CFG-A teacher target pre-encoding |
| `model/data_gate.py` | Quality/scene/sigma grouping gate |
| `model/attn_backend.py` | Attention backend chain: flash_attn (Triton) > SDPA FLASH > SDPA > math |
| `model/distill_loss.py` | Distillation loss (differentiable) |
| `model/selftest_uvit3d.py` | Pure-CPU self test |
| `model/sakiko_op/` | Optional gated-residual custom operator package (graceful fallback if absent) |
| `docs/TECHNICAL_WHITEPAPER.md` | Full technical whitepaper (CN) |

Runtime paths for data / checkpoints / teacher weights come from CLI arguments or environment variables (e.g. `COMFYUI_DIR`, `PYTHON`); no machine-specific absolute paths are stored.

## Quick start (CPU selftest)

```bash
cd model
python selftest_uvit3d.py
python distill_loss.py
python quant_plan.py
python ablation_matrix.py
python engineering_hygiene.py
```

CI runs the same suite on every push (see `.github/workflows/selftest.yml`).

## Platform notes (ROCm / Windows, measured)

- **torch.compile**: measured harmful on this platform (first compile 89-210s, then 2.4-5.6x slower; inductor lacks complex-op support); not recommended.
- **Training memory**: bf16 full parameters (~14.3GB weights+grads at 4.0B) + activation checkpointing + Adafactor; `mem_frac` fuse default 0.93, hard cap 0.95; periodic clean restart clears fragmentation; checkpoint saving serialized.

## Honest boundaries

1. Model/training scaffolding and loss definitions are included; end-to-end quality depends on data and teacher pre-encoding.
2. Forward passing is not the same as image quality; perceptual/temporal metrics need rendered-video evaluation.
3. GSA is a skeleton implementation; contrast against 2+1D needs a real ablation.
4. VAE decode / tiled decode belongs to the deployment chain, not this backbone.

## License

[BSL 1.1](../LICENSE): source-available, copy/modify/redistribute allowed, non-production use permitted; converts to GPL-3.0 on 2030-01-01.
