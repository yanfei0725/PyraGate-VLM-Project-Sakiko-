# PyraGate-VLM · 3D U-ViT Backbone（Project Sakiko）

> English overview: [docs/README_EN.md](docs/README_EN.md) · Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · Technical whitepaper: [docs/TECHNICAL_WHITEPAPER.md](docs/TECHNICAL_WHITEPAPER.md)

Pure-PyTorch video flow-matching DiT backbone. No hard dependency on diffusers or flash-attn; instantiable and self-testable on CPU. GPU forward/training validated on AMD Radeon RX 7900 XT (ROCm 7.2.1, Windows).

## Model overview

- **Scale**: 4.0B parameters (meta-device count 3,999,944,893). Symmetric U shape: encoder 12 / bottleneck 6 / decoder 12 (30 blocks), dim 3072, 24 heads (head_dim 128).
- **Conditioning**: teacher-style AdaLN — a shared two-layer embedder plus a per-block `(1, 6, dim)` zero-init table (replaces per-block `Linear(dim, 6*dim)`, saving ~1.45B parameters).
- **Attention**: 2+1D spatial/temporal decomposition (alternating) with full-3D joint bottleneck; optional GSA gated-mixing strategy; real-form fp32 3D RoPE split by t/h/w, model-level shared table, batched (complex-buffer `to(bf16)` corruption fixed).
- **FFN**: Dual-Act piecewise SwiGLU (`|z|<t` SiLU, `|z|>=t` compressed slope, learnable threshold); `mlp_ratio=8/3` matches vanilla FFN parameter count. w1/w2 and q/k/v fused into single GEMMs.
- **Output head**: velocity main head + temporal aux head + DDT-style high-frequency detail head, all zero-gated at init. The full-gated head adds frequency-band residual heads, anti-aliasing, pyramid shortcut, and per-frame high-pass path.
- **Conditional MoE (stage 2, optional)**: global conditional-space three-expert MoE (~330M each, 0.991B total), Top1 hard routing with straight-through gradients, alpha-gated additive injection into blocks 7-12; strict `alpha=0` identity fallback.
- **v3 architecture extensions** (all off by default, identity-at-init):
  - `text_cross`: low-rank UMT5 token cross-attention on selected blocks, sigma-gated, with a single shared sigma-gate entry point.
  - `AdaLNZeroText`: low-rank text-pooled AdaLN parallel residual, zero-initialized.
  - `SigmaSlidingAdapter`: high-sigma shared low-rank residual adapter.
  - `SyntaxCompiler`: PAU-style tier scheduler producing skip / FFN-modulation / cross-budget / tier-modulation signals; compiler outputs are detached for backbone injection while its own auxiliary loss keeps a differentiable path.
  - `FreqRouter`: FFT-descriptor based per-head frequency-band routing bias on attention keys; falls back to identity on degenerate grids and pads cross-attention segments with zero bias.

## Files

| File | Content |
|---|---|
| `uvit3d.py` | Full backbone structure (AdaLN / 2+1D / GSA / SwiGLU / GatedSkip / pyramid / output head / global MoE / RoPE real path) |
| `full_gated_head.py` | Full gated token head: band heads, dual-act gate, RPE, anti-alias, pyramid, high-frequency branch |
| `text_cross_attn.py` | v3 text cross-attention, text AdaLN, sigma sliding adapter, tier context tokens, shared sigma gate |
| `syntax_compiler.py` | PAU syntax compiler + tier scheduling |
| `freq_router.py`, `fft_shared.py`, `x0_fourier_band_loss.py` | Frequency router, shared FFT cache, Fourier-band losses |
| `train_staged.py`, `train_backbone.py` | Staged trainer (self / distill, teacher swap, aux losses, resume guards) and baseline trainer |
| `preencode_latent.py`, `preencode_teacher.py` | Offline VAE latent pre-encoding and CFG-A teacher target pre-encoding |
| `data_gate.py` | Quality/scene/sigma grouping gate for training data |
| `attn_backend.py` | Attention backend adapter: flash_attn (Triton) > SDPA FLASH > SDPA > CPU math, honest degradation |
| `distill_loss.py` | Distillation loss (velocity MSE / feature alignment / temporal consistency / flicker regularization) |
| `selftest_uvit3d.py` | Pure-CPU self test (bitwise regressions, numeric equivalence, parameter assertions) |

## Quick start (CPU self test)

```bash
cd model
python selftest_uvit3d.py
python distill_loss.py
python quant_plan.py
python ablation_matrix.py
python engineering_hygiene.py
```

Expect all `[PASS]`.

Paths for data / checkpoints / teacher weights are supplied at runtime via CLI arguments or environment variables (e.g. `COMFYUI_DIR`, `PYTHON`); no machine-specific absolute paths are stored in this repository.

## Platform notes (ROCm / Windows validated)

- Backend: on RDNA3 + Windows ROCm the Triton backend is 1.8-2.3x faster than SDPA on mid/long sequences; default is fine.
- `torch.compile`: first compile 89-210s and compiled run 2.4-5.6x slower on this platform (inductor lacks complex operators); not recommended.
- Training memory: bf16 full parameters (weights + grads ~14.3GB @4.0B) + gradient checkpointing + Adafactor; `mem_frac` circuit breaker defaults to 0.93, hard cap 0.95; periodic clean restart (exit 43) clears fragmentation; checkpoint saving is serialized to avoid multi-thread state_dict memory spikes.

## Honest boundaries

1. This repository contains model/training scaffolding and loss definitions; end-to-end quality requires data and teacher pre-encoding.
2. Forward passing is not the same as image quality: structure, invariants and numeric equivalence are auto-verified; perceptual/temporal metrics need rendered-video evaluation.
3. GSA is a skeleton implementation; its contrast against 2+1D needs a real ablation.
4. VAE decode / tiled decode belongs to the deployment chain and is outside this skeleton.
5. No machine-specific checkpoints, logs, notes, or absolute paths are included.

## License

[BSL 1.1](LICENSE) (Business Source License): source is viewable, modifiable and redistributable; research and non-production use is open.
