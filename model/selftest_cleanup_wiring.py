"""清理 + 接线修复后的回归自检（CPU-only，不碰 GPU）。

覆盖：默认逐值哈希、频带常量统一、cond 汇流保序、W1 专项块索引口径、
W3 参考对齐 cur_tokens 真参与且零起步、死符号已移除。
"""
import os
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import torch
import uvit3d
from uvit3d import UViT3D, UViT3DConfig
from vlm_bridge import NorthBridge
import x0_fourier_band_loss as xb
from freq_router import FreqRouter
from cond_encoders import ReferenceResampleAlignEncoder

BASE = "f28b7da26065663dc2df4ef70b443809dbcb61f1becde69cf3bc2f4a4887c193"
_ok = 0
_fail = 0


def check(c, m):
    global _ok, _fail
    if c:
        _ok += 1; print(f"  [PASS] {m}")
    else:
        _fail += 1; print(f"  [FAIL] {m}")


def lift(m):
    with torch.no_grad():
        for b in list(m.enc_blocks) + list(m.bottleneck) + list(m.dec_blocks):
            b.scale_shift_table[0, 2, :].fill_(1.0)
            b.scale_shift_table[0, 5, :].fill_(1.0)
        for s in m.skips:
            if hasattr(s, "alpha"):
                s.alpha.fill_(0.3)
        torch.nn.init.normal_(m.head.head.weight, std=0.02)
        torch.nn.init.normal_(m.head.head.bias, std=0.001)


def sha(y):
    return hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()


print("=== 1. 默认逐值等价 ===")
torch.manual_seed(0)
cfg = UViT3DConfig.tiny()
m = UViT3D(cfg).eval(); lift(m)
x = torch.randn(2, cfg.in_channels, 4, 8, 8)
t = torch.randn(2, cfg.t_emb_dim); c = torch.randn(2, cfg.cond_dim)
with torch.no_grad():
    y = m(x, t, c)
check(sha(y) == BASE, f"默认哈希 {BASE[:12]}… 不变")

print("=== 2. 频带口径统一 ===")
fr = FreqRouter(64, 4)
check(xb.BAND_LOW == 0.05 and xb.BAND_MID_HIGH == 0.25
      and fr.band_low == xb.BAND_LOW and fr.band_mid_high == xb.BAND_MID_HIGH,
      "FreqRouter 与 x0 频带条件同一组常量 0.05/0.25（值未变）")
check(xb.x0_band_energy_ratios(torch.randn(1, 3, 2, 8, 8)).shape == (1, 3),
      "x0 频带能量比仍返回 [B,3]")

print("=== 3. cond 四路汇流保序 ===")
base = torch.zeros(1, 4)
a = torch.ones(1, 4); b = torch.ones(1, 4) * 2; cc = torch.ones(1, 4) * 3
got = UViT3D._merge_cond(base, a, None, b, cc)
want = base + a + b + cc
check(torch.equal(got, want), "None 跳过、三项按原顺序加")

print("=== 4. W1 专项块索引口径 ===")
cfg2 = UViT3DConfig.tiny()
cfg2.use_vlm = True; cfg2.vlm_dim = 32; cfg2.vlm_stage_ratio = 0.5
cfg2.temporal_block = True; cfg2.spatial_block = True
m2 = UViT3D(cfg2).eval()
m2.vlm_injector.bridge = NorthBridge(32, cfg2.dim)
expected = cfg2.enc_depth + cfg2.bottleneck_depth + cfg2.dec_depth + 2
check(m2.vlm_injector.n_blocks == expected == m2._n_blocks,
      f"VLMInjector.n_blocks={m2.vlm_injector.n_blocks} 含 2 个专项块（与 gidx 一致）")
xh = torch.randn(2, 6, 32)
with torch.no_grad():
    ok = m2(torch.randn(2, cfg2.in_channels, 4, 8, 8),
            torch.randn(2, cfg2.t_emb_dim), torch.randn(2, cfg2.cond_dim),
            vlm_hidden=xh)
check(torch.isfinite(ok).all(), "开专项块 + VLM 前向正常（无越界/NaN）")

print("=== 5. W3 参考对齐 cur_tokens 真参与 ===")
enc = ReferenceResampleAlignEncoder(dim=32, num_heads=4, n_layers=1,
                                    rank=16, ffn_hidden=16, n_resample=4)
enc.eval()
cur = torch.randn(2, 10, 32); ref = torch.randn(2, 12, 32)
cur2 = cur + 5.0
with torch.no_grad():
    z0 = enc(cur, ref); z1 = enc(cur2, ref)
check(float(enc.cur_proj.weight.abs().max()) == 0.0, "cur_proj 零初始化 → 起步恒等")
check(torch.equal(z0, z1), "起步时改 cur_tokens 不改变输出（不扰动热启动）")
enc.train()
z = enc(cur, ref); z.pow(2).mean().backward()
check(enc.cur_proj.weight.grad is not None and torch.isfinite(enc.cur_proj.weight.grad).all(),
      "cur_proj 梯度可达（cur_tokens 真进计算图）")

print("=== 6. 死符号已移除 / 待接线资产保留 ===")
check(not hasattr(uvit3d, "freeze_global_moe")
      and not hasattr(uvit3d, "disable_global_moe"), "MoE 死函数已移除")
check(not hasattr(xb, "band_limit_4d")
      and not hasattr(xb, "phase_consistency_loss"), "x0 频带零调用函数已移除")
for f in ("distill_loss.py", "astree_denoise_loop.py", "teacache.py",
          "preencode_teacher.py", "data_gate.py"):
    check(os.path.isfile(f"model/{f}") if False else os.path.isfile(f),
          f"待接线资产保留：{f}")

print(f"\n[selftest_cleanup_wiring] {_ok} PASS / {_fail} FAIL")
raise SystemExit(1 if _fail else 0)
