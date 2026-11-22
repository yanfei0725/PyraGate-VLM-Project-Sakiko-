# Changelog

All notable changes to PyraGate-VLM are documented here. Dates are 2026.

## [2.0] - 2026-11-22

### Backbone (uvit3d.py)
- Symmetric U-shaped 3D U-ViT, 4.0B parameters (meta-device verified 3,999,944,893): encoder 12 / bottleneck 6 / decoder 12, dim 3072, 24 heads.
- Teacher-style AdaLN: shared two-layer embedder + per-block (1,6,dim) zero-init tables (replaces per-block Linear(dim, 6*dim); saves ~1.45B parameters).
- 2+1D spatial/temporal attention (alternating) with full-3D joint bottleneck; optional GSA gated-mixing strategy; real-form fp32 3D RoPE (t/h/w split, model-level shared table, batched — complex-buffer `to(bf16)` corruption root-fixed).
- Dual-Act SwiGLU FFN (piecewise SiLU / compressed-slope, learnable threshold; w1/w2 fused into a single GEMM; q/k/v fused GEMM).
- Output head: velocity main head + temporal aux head + DDT-style high-frequency detail head (zero-gated).
- Global conditional-space MoE (V2-27 form): one set of 3 experts (~330M each, 0.991B total), Top1 hard routing with straight-through gradients, alpha-gated additive injection into blocks 7-12, strict alpha=0 identity fallback.
- Lightweight condition encoders: hierarchical style encoder (72M) + reference-frame resample-alignment encoder (68.5M); specialized temporal/spatial blocks (113M each); all zero-gated identity-at-init.
- Attention backend: flash_attn (Triton backend) > SDPA FLASH > SDPA > math, honest degradation chain.

### Training (train_backbone.py / run_training_daemon.py)
- Adafactor + bf16 + gradient checkpointing (block scope), OOM exit-code contract (42) with daemon degrade ladder, periodic clean restart (43) to clear HIP fragmentation.
- Serialized checkpoint saving (RAM-spike safety), CFG-A teacher-target mode (pre-encoded velocity targets).
- Teacher warm-start: shape-exact embed/head-level tensor copy from the teacher checkpoint (10-item map, shape-verified, never partial).

### Data
- Offline VAE latent pre-encoding from local videos (preencode_latent.py).
- Prompt pairing for the OpenVidHD-derived latent set: 1932/1936 (99.8%) matched against the on-disk OpenVidHD metadata (prompts.json).
- CFG-A teacher pre-encoding scaffold (preencode_teacher.py) with dry-run math validation.

### Verification
- Pure-CPU selftest: 139 checks passing (gate=0 bitwise regressions, RoPE real-vs-complex equivalence <= 1 ULP, checkpoint forward/backward equivalence, 4.0B parameter hard assertion, zero-gate identities, MoE alpha=0 fallback).
- GPU-verified on Radeon RX 7900 XT (ROCm 7.2.1): 4.0B bf16 forward (peak 11.91GB at T=16/64x64), Triton kernel parity, Triton-backend FA2 parity.

### License
- BSL 1.1 (Change License: GPL-3.0, Change Date 2030-01-01).
