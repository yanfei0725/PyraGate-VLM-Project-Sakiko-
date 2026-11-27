# Changelog

All notable changes to PyraGate-VLM are documented here.

## [3.0]

### Architecture extensions (uvit3d.py / full_gated_head.py / text_cross_attn.py / syntax_compiler.py)
- Full-gated token head: frequency-band residual heads (low/mid/high, band-limited in latent space), embedded dual-act gate, 2D relative position embedding, anti-aliasing, pyramid shortcut, per-frame high-pass path with sigma attenuation.
- `text_cross`: low-rank UMT5 token cross-attention attached to selected bottleneck/encoder/decoder blocks, sigma-gated via a single shared `sigma_gate` entry; text K/V optional freeze; initialization from official cross-attention weights supported.
- `AdaLNZeroText`: low-rank text-pooled AdaLN parallel residual, zero-initialized.
- `SigmaSlidingAdapter`: shared low-rank residual active at high sigma.
- `TierContextTokens`: 4-tier register tokens (scene/actor/part/pixel) injected via AdaLN without altering the token-count grid.
- `SyntaxCompiler`: PAU-style scheduling compiler (skip / ffn-mod / cross-budget / tier-mod); auxiliary loss keeps a differentiable path while control signals are detached before backbone injection.
- `FreqRouter`: FFT-descriptor per-head frequency-band routing bias; feeds only self-attention tokens (cross-attention segments padded with zero bias); identity fallback on degenerate grids.
- Structural low-rank (`band_lr`) for attention Q/K/V/O and FFN, per-section ranks; grouped block-diagonal attention option.
- Spectral reinitialization and spectral-norm power-iteration freeze helpers, with strict preservation of zero-init identity parameters.

### Training (train_staged.py)
- Staged self-supervised / distillation loop with teacher target swap and low-sigma backfill.
- Multiple auxiliary losses (frame / token-boundary / velocity-magnitude / spectrum / boundary / patch-boundary / REPA / syntax) with immediate snapshot of main-forward aux tensors to avoid overwrite by no-grad secondary forwards; one-shot missing-aux warning when a weighted aux is not produced.
- Optimizer groups: main Adafactor plus independent AdamW groups for gates / head final / band cond / band heads / freq router / dual-sem head / new modules, keeping zero-init small tables trainable.
- Resume guards: shape mismatch rejection, architecture-change allowlist, safe-lost classification for mirrored legacy keys, optimizer param-name migration warnings.
- Selective fp32 master (stochastic-rounding write-back) option for bf16 stability.

### Data (data_gate.py / preencode_latent.py / preencode_teacher.py)
- Quality/scene/sigma grouping gate with configurable thresholds; sigma transform from cached fields.
- Offline latent pre-encoding and CFG-A teacher target pre-encoding (in-process teacher inference, no service dependency).

### Verification & hygiene
- All active Python modules compile cleanly; CPU self tests for syntax zero-init identity, detach/aux gradient path, and head gradient unlock.
- No machine-specific absolute paths, no internal notes, no absolute checkpoints or logs included; runtime paths come from CLI arguments or environment variables.

## [2.0] - 2026-11-22

### Backbone (uvit3d.py)
- Symmetric U-shaped 3D U-ViT, 4.0B parameters (meta-device verified 3,999,944,893): encoder 12 / bottleneck 6 / decoder 12, dim 3072, 24 heads.
- Teacher-style AdaLN: shared two-layer embedder + per-block (1,6,dim) zero-init tables (replaces per-block Linear(dim, 6*dim); saves ~1.45B parameters).
- 2+1D spatial/temporal attention (alternating) with full-3D joint bottleneck; optional GSA gated-mixing strategy; real-form fp32 3D RoPE (t/h/w split, model-level shared table, batched).
- Dual-Act SwiGLU FFN (piecewise SiLU / compressed-slope, learnable threshold; w1/w2 fused; q/k/v fused GEMM).
- Output head: velocity main head + temporal aux head + DDT-style high-frequency detail head (zero-gated).
- Global conditional-space MoE (V2-27 form): 3 experts (~330M each, 0.991B total), Top1 hard routing with straight-through gradients, alpha-gated additive injection into blocks 7-12, strict alpha=0 identity fallback.
- Attention backend: flash_attn (Triton backend) > SDPA FLASH > SDPA > math, honest degradation chain.

### Training
- Adafactor + bf16 + gradient checkpointing, OOM exit-code contract with daemon degrade ladder, periodic clean restart to clear fragmentation.
- Serialized checkpoint saving, CFG-A teacher-target mode, teacher warm-start with shape-exact tensor copy.

### License
- BSL 1.1 (Change License: GPL-3.0, Change Date 2030-01-01).
