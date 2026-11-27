#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAU-DiT 集成自检：默认全关逐值等价主线 + 语法平面零门控热启动。全程 no_grad。"""
import sys, os, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "model")
sys.path.insert(0, HERE)
import uvit3d as A
from syntax_compiler import SyntaxCompiler, tier_index

OK = []
def chk(name, cond, extra=""):
    OK.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

print("=== PAU-DiT 集成自检 ===")

torch.manual_seed(0)
cfg = A.UViT3DConfig.tiny()
m = A.UViT3D(cfg).eval()
B, C, T, H, W = 1, cfg.in_channels, 2, 4, 4
x = torch.randn(B, C, T, H, W)
te = torch.randn(B, cfg.t_emb_dim)
tx = torch.randn(B, cfg.cond_dim)
with torch.no_grad():
    y = m(x, te, tx)
chk("tiny 默认前向 finite", torch.isfinite(y).all().item())

# 1. 与主线逐值等价（默认 pau_syntax=False）
spec = importlib.util.spec_from_file_location("main_uvit", os.path.join(MAIN, "uvit3d.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
torch.manual_seed(0)
m2 = M.UViT3D(M.UViT3DConfig.tiny()).eval()
m2.load_state_dict(m.state_dict(), strict=False)
with torch.no_grad():
    y2 = m2(x, te, tx)
d = (y - y2).abs().max().item()
chk("默认逐值等价主线", d < 1e-5, f"maxdiff={d:.2e}")

# 2. 语法平面零初始化 = 严格恒等
torch.manual_seed(0)
cfg3 = A.UViT3DConfig.tiny(); cfg3.pau_syntax_enabled = True; cfg3.pau_rank = 16
m3 = A.UViT3D(cfg3).eval()
chk("语法编译器实例化", m3.syntax_compiler is not None)
chk("主 gate 严格为 0", m3.syntax_compiler.gate.detach().item() == 0.0)
with torch.no_grad():
    syn = m3.syntax_compiler(torch.randn(B, cfg.dim), None, None)
chk("skip_blocks=None（全零）", syn["skip_blocks"] is None)
for k in ("skip_logits", "ffn_mod", "cross_budget", "tier_mod"):
    chk(f"{k} 全零", torch.count_nonzero(syn[k]).item() == 0)

# 3. 打开语法平面但 gate=0 → 与主线逐值等价（热启动核心）
m3.load_state_dict(m2.state_dict(), strict=False)
with torch.no_grad():
    y3 = m3(x, te, tx)
d3 = (y3 - y2).abs().max().item()
chk("语法平面零门控热启动逐值等价", d3 < 1e-5, f"maxdiff={d3:.2e}")

# 4. state_dict key 兼容：主线所有 key 都在 PAU 模型里且形状一致
sd_main, sd_pau = m2.state_dict(), m3.state_dict()
missing = [k for k in sd_main if k not in sd_pau or sd_pau[k].shape != sd_main[k].shape]
chk("主线权重 key 全部可直载", len(missing) == 0, f"missing={missing[:3]}")
new_keys = [k for k in sd_pau if k.startswith("syntax_compiler.")]
chk("新参数仅语法编译器", len(new_keys) > 0 and all(
    k.startswith("syntax_compiler.") for k in set(sd_pau) - set(sd_main)),
    f"new={len(new_keys)}")

# 5. tier 映射
chk("tier 映射 4 级", [tier_index(i, 30, 4) for i in (0, 7, 15, 29)] == [0, 0, 2, 3])

# 6. tier_mod 到达 block（hook 计数）
seen = []
def mk_hook(i):
    def h(mod, inp, out):
        seen.append(i)
    return h
import torch.nn as nn
fm = torch.zeros(B, cfg.dim)
with torch.no_grad():
    # 用语法编译器生成的零 tier_mod，走完整路径
    m4 = A.UViT3D(cfg3).eval()
    m4.load_state_dict(m2.state_dict(), strict=False)
    handles = [b.register_forward_hook(mk_hook(i)) for i, b in enumerate(
        list(m4.enc_blocks) + list(m4.bottleneck) + list(m4.dec_blocks))]
    m4(x, te, tx)
    for h in handles: h.remove()
chk("语法平面开启后 block 全程执行", len(set(seen)) == len(
    list(m4.enc_blocks) + list(m4.bottleneck) + list(m4.dec_blocks)),
    f"seen={len(set(seen))}")

# 7. 工厂
cfgp = A.UViT3DConfig.pau_syntax()
chk("pau_syntax 工厂", cfgp.dim == 3072 and cfgp.pau_syntax_enabled and cfgp.pau_tiers == 4)

print()
bad = [n for n, ok in OK if not ok]
print(f"RESULT: {len(OK)-len(bad)}/{len(OK)} PASS")
if bad:
    print("FAILED:", bad)
    sys.exit(1)
print("ALL PASS")
