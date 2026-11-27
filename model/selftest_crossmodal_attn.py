"""
融合版跨模态注意力场自检（CPU-only，不碰显卡）。

验证：
  1. 默认配置逐值等价（基线哈希不变）
  2. 非对称特权结构：α 只影响 A 侧，B 侧 cross_kv 结构上拿不到 α
  3. 摘要 K/V 真正进入每层 block 的注意力场（hook 计数 == n_blocks）
  4. 融合版跳过重复 inject_high（CrossAttnInjector 调用次数 == 0）
  5. budget=0 时严格恒等
  6. 梯度可达全部分支
  7. 参数量零新增（in_attn 开关不改参数）

运行：
  python selftest_crossmodal_attn.py
"""
import os
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import torch

from uvit3d import UViT3D, UViT3DConfig, SelfAttention, count_parameters
from vlm_bridge import NorthBridge, CrossAttnInjector
from asymmetric_crossmodal import AsymmetricCrossModal, _selfcheck as cm_selfcheck

BASELINE_SHA256 = "f28b7da26065663dc2df4ef70b443809dbcb61f1becde69cf3bc2f4a4887c193"

_ok = 0
_fail = 0


def check(cond, msg):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  [PASS] {msg}")
    else:
        _fail += 1
        print(f"  [FAIL] {msg}")


def banner(s):
    print("\n=== " + s + " ===")


def lift_gates(m):
    """抬起 block gate 与输出头，使前向非平凡（哈希/梯度才有意义）。"""
    with torch.no_grad():
        for blk in list(m.enc_blocks) + list(m.bottleneck) + list(m.dec_blocks):
            blk.scale_shift_table[0, 2, :].fill_(1.0)
            blk.scale_shift_table[0, 5, :].fill_(1.0)
        for sk in m.skips:
            if hasattr(sk, "alpha"):
                sk.alpha.fill_(0.3)
        if getattr(m, "head", None) is not None and hasattr(m.head, "head"):
            torch.nn.init.normal_(m.head.head.weight, std=0.02)
            torch.nn.init.normal_(m.head.head.bias, std=0.001)


def make_inputs(cfg, B=2):
    # 不重置 RNG：测试 2 必须与基线脚本的 RNG 消耗顺序完全一致
    return (torch.randn(B, cfg.in_channels, 4, 8, 8),
            torch.randn(B, cfg.t_emb_dim),
            torch.randn(B, cfg.cond_dim),
            torch.randn(B, 6, 32))


def cfg_fused():
    cfg = UViT3DConfig.tiny()
    cfg.use_vlm = True
    cfg.vlm_dim = 32
    cfg.vlm_stage_ratio = 0.5
    cfg.a2_crossmodal = True
    cfg.a2_crossmodal_slots = 8
    cfg.a2_crossmodal_in_attn = True
    return cfg


def build(cfg):
    torch.manual_seed(0)
    m = UViT3D(cfg).eval()
    if m.vlm_injector is not None:
        m.vlm_injector.bridge = NorthBridge(cfg.vlm_dim, cfg.dim)
    lift_gates(m)
    return m


def sha(y):
    return hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- 1
banner("1. 模块自检：零门控恒等 + 非对称结构")
info = cm_selfcheck()
check(info["zero_gate_identity"], "零门控严格恒等")
check(info["asymmetric_alpha"], "结构上不存在 alpha_proj_b（α 特权只进 A）")

# ---------------------------------------------------------------- 2
banner("2. 默认配置逐值等价（回归红线）")
cfg0 = UViT3DConfig.tiny()
m0 = build(cfg0)
x, t, c, vh = make_inputs(cfg0)
with torch.no_grad():
    y0 = m0(x, t, c)
check(sha(y0) == BASELINE_SHA256,
      f"默认前向哈希 == 基线 {BASELINE_SHA256[:12]}…")
check(m0.crossmodal is None and m0.syntax_compiler is None,
      "默认 crossmodal / syntax_compiler 均未实例化（零参数增量）")

# ---------------------------------------------------------------- 3
banner("3. budget=0 严格恒等（开关不扰动主干）")
cfg = cfg_fused()
m = build(cfg)
x, t, c, vh = make_inputs(cfg)
# 必须用同一个模型对比：不同模型实例随机权重不同，不能用来判恒等
with torch.no_grad():
    y_off = m(x, t, c, vlm_hidden=vh, crossmodal_budget=0.0)
    y_none = m(x, t, c, vlm_hidden=vh, crossmodal_budget=None)
check(torch.equal(y_off, y_none), "budget=0 与不下发 budget 逐 bit 一致")

# ---------------------------------------------------------------- 4
banner("4. 摘要 K/V 进入每层注意力场 + 跳过重复 inject_high")
hits = []
hooks = []
for mod in m.modules():
    if isinstance(mod, SelfAttention):
        hooks.append(mod.register_forward_hook(
            lambda _m, _a, kw, _o: hits.append(kw.get("cross_kv")), with_kwargs=True))
inj_hits = []
for mod in m.modules():
    if isinstance(mod, CrossAttnInjector):
        hooks.append(mod.register_forward_hook(
            lambda _m, _a, _kw, _o: inj_hits.append(1), with_kwargs=True))
with torch.no_grad():
    y_on = m(x, t, c, vlm_hidden=vh, crossmodal_budget=1.0)
for h in hooks:
    h.remove()
n_blocks = (cfg.enc_depth + cfg.bottleneck_depth + cfg.dec_depth)
check(len(hits) == n_blocks, f"SelfAttention 调用次数 {len(hits)} == n_blocks {n_blocks}")
check(all(kv is not None for kv in hits),
      f"全部 {len(hits)} 次注意力都收到非空 cross_kv（摘要在注意力场内流动）")
check(all(kv[0].shape[1] == cfg.a2_crossmodal_slots for kv in hits),
      f"摘要槽数 = {cfg.a2_crossmodal_slots}（瓶颈维度 << VLM token 数 6）")
check(len(inj_hits) == 0,
      "CrossAttnInjector 零调用（融合版跳过重复 VLM→latent cross-attn）")
check(not torch.equal(y_on, y_off), "budget>0 时输出确实改变（跨模态真正生效）")
check(torch.isfinite(y_on).all(), "开启融合路径后输出有限（无 NaN/Inf）")

# ---------------------------------------------------------------- 5
banner("5. 非对称特权：α 只影响 A 路径，不影响 B 侧 cross_kv")
cm = AsymmetricCrossModal(32, num_slots=8, num_heads=4, alpha_dim=32)
with torch.no_grad():
    cm.gate_a.fill_(1.0)
    cm.gate_b.fill_(1.0)
a = torch.randn(2, 6, 32)
b = torch.randn(2, 10, 32)
alpha1 = torch.randn(2, 32)
alpha2 = alpha1 + 3.0
a_out1, kv1 = cm.compute_cross_kv(a, b, alpha1)
a_out2, kv2 = cm.compute_cross_kv(a, b, alpha2)
check(torch.equal(kv1[0], kv2[0]) and torch.equal(kv1[1], kv2[1]),
      "B 侧 cross_kv 与 α 无关（B 结构上拿不到信息特权）")
check(not torch.allclose(a_out1, a_out2),
      "A 侧回注随 α 改变（α 特权沿 A 路径生效）")

# ---------------------------------------------------------------- 6
banner("6. 梯度可达 + 参数量零新增")
m_train = build(cfg).train()
x, t, c, vh = make_inputs(cfg)
y = m_train(x, t, c, vlm_hidden=vh, crossmodal_budget=1.0)
y.pow(2).mean().backward()
grads = [(n, p.grad) for n, p in m_train.crossmodal.named_parameters() if p.grad is not None]
no_grad = [n for n, p in m_train.crossmodal.named_parameters() if p.grad is None]
# inject_b 只服务旧的一次性 token 回注路径；融合版由每层 attention 的
# k_ctx_b/v_ctx_b 取代，故该权重在本路径上是冗余参数（零计算开销）。
check(set(no_grad) <= {"inject_b.weight"},
      f"融合路径无梯度参数仅旧路径专用项：{no_grad}")
check(all(torch.isfinite(g).all() for _, g in grads), "所有可达梯度有限")
p_on, _ = count_parameters(build(cfg_fused()))
cfg_off = cfg_fused()
cfg_off.a2_crossmodal_in_attn = False
p_off, _ = count_parameters(build(cfg_off))
check(p_on == p_off, f"in_attn 开关不改变参数量（{p_on} == {p_off}）")

print(f"\n[selftest_crossmodal_attn] {_ok} PASS / {_fail} FAIL")
raise SystemExit(1 if _fail else 0)
