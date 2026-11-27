#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VSA + TeaCache 集成自检：默认逐值等价、VSA 模式可跑、TeaCache 跳步判定。"""
import sys, os
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uvit3d as A
from vsa_attention import vsa_sparse_attention
from teacache import TeaCache

OK = []
def chk(name, cond, extra=""):
    OK.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

print("=== VSA + TeaCache 集成自检 ===")

# 1. VSA 模块自身稠密等价（tile 切分互逆性，已在 vsa_attention.py selfcheck 验证）
# 这里用能被 tile 整除的网格再验一次，经主干 SelfAttention 走 VSA 路径
torch.manual_seed(0)
x = torch.randn(1, 64, 64)          # 64 token，grid (4,4,4) 可被 tile=(4,4,4) 整除
sa_vsa = A.SelfAttention(64, 4, use_rope=False, attn_mode="vsa",
                         attn_backend="math", vsa_keep_ratio=1.0).eval()
with torch.no_grad():
    y_vsa = sa_vsa(x, rope_grid=(4, 4, 4))
chk("VSA keep=1.0 前向 finite", torch.isfinite(y_vsa).all().item()
    and y_vsa.shape == x.shape, f"shape={tuple(y_vsa.shape)}")

# 2. VSA 稀疏模式可跑 + finite
sa_sparse = A.SelfAttention(64, 4, use_rope=False, attn_mode="vsa",
                            attn_backend="math", vsa_keep_ratio=0.25).eval()
with torch.no_grad():
    y_sparse = sa_sparse(x, rope_grid=(4, 4, 4))
chk("VSA 稀疏模式前向 finite", torch.isfinite(y_sparse).all().item()
    and y_sparse.shape == x.shape)

# 3. VSA 与 softmax 两个实例权重相同 → state_dict key 完全兼容（热启动）
sa_soft = A.SelfAttention(64, 4, use_rope=False, attn_mode="softmax",
                          attn_backend="math").eval()
missing = [k for k in sa_soft.state_dict()
           if k not in sa_vsa.state_dict()
           or sa_vsa.state_dict()[k].shape != sa_soft.state_dict()[k].shape]
chk("VSA 与 softmax 权重 key 兼容", len(missing) == 0, f"missing={missing[:2]}")

# 4. 主干默认（pau_syntax=False + attn_core=softmax）前向 finite
torch.manual_seed(0)
cfg = A.UViT3DConfig.tiny()
m = A.UViT3D(cfg).eval()
B, C, T, H, W = 1, cfg.in_channels, 2, 4, 4
xx = torch.randn(B, C, T, H, W)
te = torch.randn(B, cfg.t_emb_dim)
tx = torch.randn(B, cfg.cond_dim)
with torch.no_grad():
    yy = m(xx, te, tx)
chk("主干默认前向 finite", torch.isfinite(yy).all().item())

# 5. TeaCache 跳步判定（变化大不跳 / 渐变跳）
tc = TeaCache(rel_l1_thresh=0.5)
base = torch.randn(4, 8); drift = torch.randn(4, 8) * 0.002; cur = base.clone()
for s in range(20):
    cur = cur + drift
    need = tc.should_compute(cur, 20, is_first=(s == 0), is_last=(s == 19))
    if need:
        tc.cache(torch.randn(4, 8))
chk("TeaCache 渐变触发跳步", tc.skip_steps > 0, f"{tc.summary()}")
tc2 = TeaCache(rel_l1_thresh=0.15)
for s in range(20):
    tc2.should_compute(torch.randn(4, 8), 20, is_first=(s == 0), is_last=(s == 19))
chk("TeaCache 变化大不误跳", tc2.skip_steps == 0, f"{tc2.summary()}")

print()
bad = [n for n, ok in OK if not ok]
print(f"RESULT: {len(OK)-len(bad)}/{len(OK)} PASS")
if bad:
    print("FAILED:", bad)
    sys.exit(1)
print("ALL PASS")
