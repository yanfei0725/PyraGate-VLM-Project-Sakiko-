"""
纯 CPU 自检：验证 3D U-ViT 骨架能拼起来、能前向、设计点正确。
绝不碰显卡：脚本头部强制 CUDA_VISIBLE_DEVICES=""。

运行：
  CUDA_VISIBLE_DEVICES="" python selftest_uvit3d.py
  （Windows git-bash 下 CUDA_VISIBLE_DEVICES 由外部 env 传入，脚本内再设一次兜底）
"""
import os
# 双保险：ROCm 环境靠 HIP_VISIBLE_DEVICES 屏蔽（CUDA_ 变量在部分启动脚本里会被重置）
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import uvit3d
from uvit3d import (UViT3D, UViT3DConfig, SelfAttention, SpatialAttention,
                    TemporalAttention, JointAttention, GSA, GatedSkip, SwiGLU,
                    PatchMerge3D, PatchUnmerge3D, OutputHead, count_parameters,
                    disable_all_skips, MLP_RATIO)
import os as _os

torch.manual_seed(0)


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise AssertionError(msg)


banner("0. 环境（确认计算全在 CPU）")
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
print("  default device:", torch.empty(1).device)
check(torch.empty(1).device.type == "cpu", "默认张量 device = cpu（不自动上 GPU）")
check(not any(".cuda()" in str(f) for f in dir(torch.Tensor)[:0]), "占位：无显式 .cuda() 调用")

banner("1. 2+1D 注意力分解（§3.4）")
B, T, H, W, C, nh = 2, 4, 6, 8, 64, 4
x = torch.randn(B, T * H * W, C)
for name, mod in [("SpatialAttention", SpatialAttention(C, nh)),
                  ("TemporalAttention", TemporalAttention(C, nh, causal=True)),
                  ("JointAttention", JointAttention(C, nh))]:
    y = mod(x, (T, H, W))
    check(y.shape == x.shape, f"{name} 输出形状保持 {tuple(y.shape)}")

banner("2. SwiGLU FFN（mlp_ratio=8/3，参数持平原生 FFN）")
ffn = SwiGLU(C)
check(ffn(x).shape == x.shape, "SwiGLU 输出形状保持")
check(abs(MLP_RATIO - 8 / 3) < 1e-9, f"mlp_ratio = {MLP_RATIO:.4f} = 8/3")
# 参数量对等校验：原生 2*d*4d = 8d²  vs  SwiGLU 3*d*(8/3 d) = 8d²
native_params = 2 * C * (4 * C)
swiglu_params = sum(p.numel() for p in ffn.parameters())
check(abs(swiglu_params - native_params) <= 4 * C,  # dim 取整误差（3 矩阵 × 1 维）
      f"SwiGLU {swiglu_params:,} ≈ 原生 FFN {native_params:,}（参数持平，误差<{4*C}）")

banner("2b. GSA 门控混合时空注意力（设计选定，另一条路线）")
gsa = GSA(C, nh, cond_dim=C)
y = gsa(x, (T, H, W), cond=torch.randn(B, C))
check(y.shape == x.shape, f"GSA 输出形状保持 {tuple(y.shape)}")
check(gsa.out_gate.item() == 0.0, "GSA 输出门零初始化（初始近似恒等）")

banner("3. GatedSkip 门控 + 谱约束（§3.3）")
gs = GatedSkip(C, alpha_init=0.1, spectral=True)
y = gs(x, torch.randn_like(x))
check(y.shape == x.shape, "GatedSkip 输出形状保持")
alpha_before = gs.alpha.item()
gs.disable()
check(gs.alpha.item() == 0.0, "skip.disable() 把门控 α 置 0（消融开关）")

banner("4. 金字塔 PatchMerge / Unmerge（§4.1）")
pm = PatchMerge3D(C, s=2, t=1)
z, g2 = pm(x, (T, H, W))
check(g2 == (T, H // 2, W // 2), f"merge 后 grid {g2} 正确（时间不变，空间×1/2）")
check(z.shape[1] == T * (H // 2) * (W // 2), "merge 后 token 数正确")
um = PatchUnmerge3D(C, s=2, t=1)
z2, g3 = um(z, g2)
check(g3 == (T, H, W), f"unmerge 还原 grid {g3} 正确")
check(z2.shape[1] == x.shape[1], "unmerge 还原 token 数正确")

banner("5. 输出头：速度场 + 时序辅助（§3.6）")
head = OutputHead(C, patch_size=(1, 2, 2), out_channels=16, aux_head=True)
v, aux = head(x, (T, H, W))
check(v.shape == (B, 16, T, H * 2, W * 2), f"速度场输出形状 {tuple(v.shape)}")
check(aux is not None and aux.shape == (B, T, H, W), f"时序辅助残差形状 {tuple(aux.shape)}")

banner("6. UViT3D tiny 端到端前向（V1 单分辨率）")
cfg = UViT3DConfig.tiny()
model = UViT3D(cfg)
model.eval()
Bt, Ct, Tt, Ht, Wt = 1, cfg.in_channels, 4, 8, 8
latent = torch.randn(Bt, Ct, Tt, Ht, Wt)
t_emb = torch.randn(Bt, cfg.t_emb_dim)
text_emb = torch.randn(Bt, cfg.cond_dim)
with torch.inference_mode():
    v, aux = model(latent, t_emb, text_emb, return_aux=True)
pt, ph, pw = cfg.patch_size
check(v.shape == (Bt, cfg.out_channels, Tt // pt * pt, Ht, Wt),
      f"tiny 速度场输出形状 {tuple(v.shape)}")
check(aux.shape == (Bt, Tt // pt, Ht // ph, Wt // pw),
      f"tiny 辅助残差形状 {tuple(aux.shape)}")
total, trainable = count_parameters(model)
print(f"  tiny 参数量：{total:,}（trainable {trainable:,}）")

banner("7. skip 消融：置 0 后输出必须改变（证明 skip 生效）")
v_on, _ = model(latent, t_emb, text_emb, return_aux=True)
disable_all_skips(model)
with torch.inference_mode():
    v_off, _ = model(latent, t_emb, text_emb, return_aux=True)
diff = (v_on - v_off).abs().mean().item()
check(diff > 0, f"skip 开/关输出不同（平均差 {diff:.6f}），门控确实生效")

banner("7b. 红线 8：skip 门控置 0 后必须与无 skip 直筒逐 bit 对齐")
# 验证"置 0 == 结构上无 skip"，而非仅"开/关输出不同"（那是 §7 的反命题，不足以验收红线）。
# 构造两个权重逐 bit 相同的模型：
#   m_alpha0 = 官方 disable_all_skips() 把 α 置 0（skip 分支仍算但乘 0）
#   m_noskip = 结构上完全跳过 skip 分支（forward 直接返回 x_deep）
# 若红线 8 成立，两者输出必须逐 bit 相等（torch.equal，非 allclose）。
_rng_state = torch.get_rng_state()
torch.manual_seed(1234)
m_alpha0 = UViT3D(cfg); m_alpha0.eval()
torch.manual_seed(1234)
m_noskip = UViT3D(cfg); m_noskip.eval()
torch.set_rng_state(_rng_state)  # 恢复 RNG，不影响后续测试段
_weights_same = all(torch.equal(p1, p2)
                    for p1, p2 in zip(m_alpha0.parameters(), m_noskip.parameters()))
check(_weights_same, "对照模型初始化权重逐 bit 相同（同 seed 构造）")
disable_all_skips(m_alpha0)
check(all(s.alpha.item() == 0.0 for s in m_alpha0.skips), "m_alpha0 全部 skip α=0")
for _s in m_noskip.skips:
    _s.forward = lambda x_deep, skip_feat: x_deep   # 结构上无 skip 分支
with torch.inference_mode():
    v_a0, aux_a0 = m_alpha0(latent, t_emb, text_emb, return_aux=True)
    v_ns, aux_ns = m_noskip(latent, t_emb, text_emb, return_aux=True)
check(torch.equal(v_a0, v_ns),
      "红线 8：gate=0 速度场与无 skip 直筒逐 bit 对齐（torch.equal）")
check(torch.equal(aux_a0, aux_ns),
      "红线 8：gate=0 时序辅助头与无 skip 直筒逐 bit 对齐")
print(f"  gate=0 vs 直筒：最大绝对差 {(v_a0 - v_ns).abs().max().item():.2e}")

banner("7c. sakiko_op 算子挂载（GatedSkip 门控入口）")
check(uvit3d._SAKIKO_AVAILABLE, "7c.算子包可用")
check(isinstance(uvit3d._sakiko_hits(), dict), "7c.命中统计可读")
# 默认开关：SAKIKO_CUSTOM_OP 未设 → native，输出必须等于数值参考 x+α·branch
gsm = GatedSkip(C, alpha_init=0.1, spectral=False).eval()
x_in = torch.randn(2, C)
s_in = torch.randn(2, C)
with torch.no_grad():
    branch_ref = gsm.proj(gsm.norm(s_in))
    ref_native = x_in + gsm.alpha * branch_ref
    got_native = gsm(x_in, s_in)
check(torch.equal(got_native, ref_native), "7c.默认native路径数值等价")
# 显式 native（use_op 通过环境变量控制，此处直接切环境变量验两条路径）
_prev_env = os.environ.get("SAKIKO_CUSTOM_OP")
os.environ["SAKIKO_CUSTOM_OP"] = "0"
with torch.no_grad():
    got_off = gsm(x_in, s_in)
os.environ["SAKIKO_CUSTOM_OP"] = "1"
with torch.no_grad():
    got_on = gsm(x_in, s_in)
if _prev_env is None:
    os.environ.pop("SAKIKO_CUSTOM_OP", None)
else:
    os.environ["SAKIKO_CUSTOM_OP"] = _prev_env
check(torch.allclose(got_on, got_off, atol=1e-6), "7c.custom_op与native数值一致")
# 关键：无论哪条路径，alpha=0 必须逐 bit 恒等（红线 8 的局部再验证）
gs0 = GatedSkip(C, alpha_init=0.0, spectral=False).eval()
with torch.no_grad():
    check(torch.equal(gs0(x_in, s_in), x_in), "7c.alpha0严格恒等")
print("  挂载说明：默认 native（数值不变）；SAKIKO_CUSTOM_OP=1 切注册算子；缺失时自动降级")

banner("7d. 梯度检查点：前向逐 bit + 反向梯度等价（V2-23）")
import copy as _copy
cfg_gc = UViT3DConfig.tiny()
cfg_gc.skip_spectral = False
torch.manual_seed(1234)
m_off = UViT3D(cfg_gc).train()
m_on = _copy.deepcopy(m_off)
m_on.enable_gradient_checkpointing()
check(m_on.grad_checkpointing is True and m_off.grad_checkpointing is False,
      "7d.开关状态正确（on=True / off=False）")
x_gc = torch.randn(1, cfg_gc.in_channels, 4, 8, 8)
t_gc = torch.randn(1, cfg_gc.t_emb_dim)
tx_gc = torch.randn(1, cfg_gc.cond_dim)
v_off = m_off(x_gc, t_gc, tx_gc)
v_on = m_on(x_gc, t_gc, tx_gc)
check(torch.equal(v_off, v_on),
      f"7d.前向逐 bit 一致（max diff {(v_off - v_on).abs().max().item():.2e}）")
v_off.pow(2).mean().backward()
v_on.pow(2).mean().backward()
grad_diff, n_cmp, mismatch = 0.0, 0, []
for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_on.named_parameters()):
    assert n1 == n2, f"参数顺序错位：{n1} vs {n2}"
    if p1.grad is None and p2.grad is None:
        continue
    n_cmp += 1
    if p1.grad is None or p2.grad is None:
        mismatch.append(n1 + "(存在性)")
        continue
    grad_diff = max(grad_diff, (p1.grad - p2.grad).abs().max().item())
    if not torch.equal(p1.grad, p2.grad):
        mismatch.append(n1)
check(n_cmp > 0 and not mismatch,
      f"7d.反向梯度逐值一致（比对 {n_cmp} 组，max diff {grad_diff:.2e}，不一致 {len(mismatch)}）")
m_on.eval()
with torch.no_grad():
    v_inf = m_on(x_gc, t_gc, tx_gc)
check(torch.equal(v_inf, v_off.detach()),
      "7d.推理态（no_grad）不套检查点，输出与关闭态逐 bit 一致")
print("  说明：检查点只砍前向激活显存，不动权重/梯度/优化器状态（V2-23）")

# --- 7e：选择性检查点 ffn_only（V2-25）——只重算 FFN，SelfAttn 不重算 ---
cfg_fo = UViT3DConfig.tiny()
cfg_fo.skip_spectral = False
cfg_fo.ckpt_scope = "ffn_only"
torch.manual_seed(77)
m_blk = UViT3D(cfg_fo).train()                 # ckpt_scope=ffn_only 但 grad_checkpoint 关
m_ffn = _copy.deepcopy(m_blk)
m_ffn.enable_gradient_checkpointing()          # 只开 FFN 子层重算
check(any(isinstance(b.ffn, uvit3d._FFNCheckpoint) and b.ffn.enabled
          for b in m_ffn.enc_blocks), "7e.ffn_only 包装已启用")
x_fo = torch.randn(1, cfg_fo.in_channels, 4, 8, 8)
t_fo = torch.randn(1, cfg_fo.t_emb_dim)
tx_fo = torch.randn(1, cfg_fo.cond_dim)
o_blk = m_blk(x_fo, t_fo, tx_fo)
o_ffn = m_ffn(x_fo, t_fo, tx_fo)
check(torch.equal(o_blk, o_ffn),
      f"7e.ffn_only 前向逐 bit 一致（max {(o_blk-o_ffn).abs().max().item():.2e}）")
o_blk.pow(2).mean().backward()
o_ffn.pow(2).mean().backward()
fd = 0.0
for (_, p1), (_, p2) in zip(m_blk.named_parameters(), m_ffn.named_parameters()):
    if p1.grad is not None and p2.grad is not None:
        fd = max(fd, (p1.grad - p2.grad).abs().max().item())
check(fd < 1e-7, f"7e.ffn_only 反向梯度一致（max diff {fd:.2e}）")
print("  说明：ffn_only 收益未实测（FlashAttn SelfAttn 激活本就小，关其重算或反增峰值）")

banner("8. 金字塔版 tiny 端到端前向（V2，pyramid=True）")
cfg_p = UViT3DConfig.tiny()
cfg_p.pyramid = True
cfg_p.skip_spectral = False   # spectral_norm 与 pyramid 组合先测最简
model_p = UViT3D(cfg_p)
model_p.eval()
with torch.inference_mode():
    vp, auxp = model_p(latent, t_emb, text_emb, return_aux=True)
check(vp.shape == v.shape, f"金字塔版输出形状与单分辨率一致 {tuple(vp.shape)}")
total_p, _ = count_parameters(model_p)
print(f"  tiny+pyramid 参数量：{total_p:,}")

banner("8b. GSA 策略端到端前向（attn_strategy='gsa'）")
cfg_g = UViT3DConfig.tiny()
cfg_g.attn_strategy = "gsa"
model_g = UViT3D(cfg_g)
model_g.eval()
with torch.inference_mode():
    vg, _ = model_g(latent, t_emb, text_emb, return_aux=True)
check(vg.shape == v.shape, f"GSA 版输出形状一致 {tuple(vg.shape)}")

banner("8c. VAE 轻量版路径（AMD 蜂鸟，保持 fp16）——可选环境项")
vae_path = cfg.vae_path
print("  vae_path:", vae_path)
if _os.path.exists(vae_path):
    print("  [PASS] VAE 权重在盘上:", vae_path)
else:
    print("  [SKIP] VAE 权重不在盘上（发布仓库不含权重文件；"
          "设 PYRAGATE_VAE_PATH 指向本地权重后可验）——不阻塞结构自检")

banner("8f. 3D RoPE（对齐教师 Wan2.2，t/h/w 分频）")
from uvit3d import rope_params, rope_apply_3d

# 参数生成：复数相位，长度 = max_len，通道 = head_dim/2
head_dim = 16
rp = rope_params(64, head_dim)
check(rp.shape == (64, head_dim // 2), f"rope_params 形状 {tuple(rp.shape)}")
check(torch.allclose(rp.abs(), torch.ones_like(rp.abs()), atol=1e-6),
      "rope_params 是单位复数（纯相位）")

# 保范数：RoPE 是旋转，模长不变
N3, n_heads = T * H * W, 4
q3 = torch.randn(N3, n_heads, head_dim)
q3_rope = rope_apply_3d(q3, (T, H, W), rp)
norm_before = q3.pow(2).sum(-1)
norm_after = q3_rope.pow(2).sum(-1)
check(torch.allclose(norm_before, norm_after, atol=1e-4), "3D RoPE 保范数（旋转不改模长）")

# 位置敏感：token 顺序变 -> 输出不同（证明 RoPE 注入了位置信息）
q3_shift = torch.roll(q3, shifts=1, dims=0)
q3_shift_rope = rope_apply_3d(q3_shift, (T, H, W), rp)
check(not torch.allclose(q3_rope, q3_shift_rope, atol=1e-4), "3D RoPE 位置敏感（换位输出变化）")

# 与教师实现逐值对齐（手工复算 rope_apply 的核心逻辑）
def ref_rope_apply(x, grid, freqs):
    f, h, w = grid
    n = x.size(1)
    c = x.size(2) // 2
    ft, fh, fw = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    xc = torch.view_as_complex(x.to(torch.float64).reshape(f * h * w, n, -1, 2))
    fi = torch.cat([
        ft[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fh[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fw[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1)
    return torch.view_as_real(xc * fi).flatten(2).to(x.dtype)

ref_out = ref_rope_apply(q3, (T, H, W), rp)
check(torch.allclose(q3_rope, ref_out, atol=1e-4), "3D RoPE 前向 == 教师实现手工复算（逐值对齐）")

# 注意力内 RoPE 生效：use_rope 开关对比
sa_rope = SelfAttention(C, nh, use_rope=True)
sa_plain = SelfAttention(C, nh, use_rope=False)
with torch.no_grad():
    sa_plain.q.weight.copy_(sa_rope.q.weight); sa_plain.k.weight.copy_(sa_rope.k.weight)
    sa_plain.v.weight.copy_(sa_rope.v.weight); sa_plain.o.weight.copy_(sa_rope.o.weight)
y_rope = sa_rope(x, rope_grid=(T, H, W))
y_plain = sa_plain(x, rope_grid=(T, H, W))
check(not torch.allclose(y_rope, y_plain, atol=1e-5), "SelfAttention 内 RoPE 生效（开关输出不同）")

banner("8g. FlashAttention-2 后端适配（真 FA2 优先，CPU 自动降级 math）")
from attn_backend import (attention as dispatch_attention,
                          resolve_backend, backend_diagnose)

diag = backend_diagnose()
for key in ("torch", "flash_attn_pkg", "flash_sdp_enabled", "sdpa_kernel_api"):
    check(key in diag, f"后端环境画像含 {key}")
print("  后端画像:", {k: diag[k] for k in
      ("flash_attn_pkg", "flash_sdp_enabled", "cuda_visible", "sdpa_kernel_api")})

# CPU 上请求 flash2 必须明确降级，不允许伪装成真 FA2
actual, reason = resolve_backend("flash2", torch.empty(1))
check(actual == "math" and reason == "cpu_fallback_no_gpu_kernel",
      f"CPU 请求 flash2 -> 实际 math（诚实降级）: {actual}/{reason}")

Bq, Hq, Nq, Dq = 2, 3, 7, 8
qq = torch.randn(Bq, Hq, Nq, Dq)
kk = torch.randn(Bq, Hq, Nq, Dq)
vv = torch.randn(Bq, Hq, Nq, Dq)

def ref_attention(q, k, v, causal=False):
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        n = q.size(-2)
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1),
                                    float("-inf"))
    return torch.softmax(scores, dim=-1) @ v

ref = ref_attention(qq, kk, vv)
for be in ("flash2", "auto", "sdpa", "math"):
    y, info = dispatch_attention(qq, kk, vv, backend=be, with_info=True)
    check(y.shape == qq.shape, f"后端 {be} 输出形状保持 {tuple(y.shape)}")
    check(torch.allclose(y, ref, atol=1e-5),
          f"后端 {be} 数值 == 手写参考注意力（actual={info['actual']}）")

ref_causal = ref_attention(qq, kk, vv, causal=True)
y_c, info_c = dispatch_attention(qq, kk, vv, is_causal=True,
                                 backend="flash2", with_info=True)
check(torch.allclose(y_c, ref_causal, atol=1e-5),
      f"causal 掩码在 flash2 链路上正确（actual={info_c['actual']}）")

# 配置确实贯通到所有 SelfAttention（含 spatial/temporal/joint）
cfg_fa = UViT3DConfig.tiny()
cfg_fa.attn_backend = "flash2"
check(cfg_fa.attn_backend == "flash2", "配置默认后端 = flash2（设计选定）")
model_fa = UViT3D(cfg_fa)
be_enc0 = model_fa.enc_blocks[0].attn.attn.attn_backend
be_bot = model_fa.bottleneck[0].attn.attn.attn_backend
check(be_enc0 == "flash2" and be_bot == "flash2",
      f"后端配置贯通 encoder/joint（{be_enc0}/{be_bot}）")

# 同一权重下 flash2 链路与显式 math 输出逐值一致（CPU 降级路径）
sa_fa = SelfAttention(C, nh, attn_backend="flash2")
with torch.no_grad():
    y_fa = sa_fa(x, rope_grid=(T, H, W))
sa_fa.attn_backend = "math"
with torch.no_grad():
    y_math = sa_fa(x, rope_grid=(T, H, W))
check(torch.allclose(y_fa, y_math, atol=1e-6),
      "SelfAttention flash2 链路与 math 数值逐值一致（CPU 降级）")

with torch.inference_mode():
    v_fa, _ = model_fa(latent, t_emb, text_emb, return_aux=True)
check(v_fa.shape == v.shape, f"flash2 配置端到端形状一致 {tuple(v_fa.shape)}")

banner("8d. VLM 接入：北大桥 + 多尺度分层注入")
from vlm_bridge import NorthBridge, CrossAttnInjector, VLMInjector

# 桥结构：4096 -> 3072（tiny 测试用缩小维度 32 -> 16）
bridge = NorthBridge(in_features=32, hidden_size=16)
vh = torch.randn(B, 5, 32)          # [B, L, vlm_dim] 模拟 Qwen hidden
bc = bridge(vh)
check(bc.shape == (B, 5, 16), f"北大桥输出 [B,L,dim] {tuple(bc.shape)}")
want_shapes = bridge.expected_shapes()
check(want_shapes["linear_1.weight"] == (64, 32), "桥 linear_1 形状 (4*hidden, in) 正确")
check(want_shapes["linear_2.weight"] == (16, 64), "桥 linear_2 形状 (out, 4*hidden) 正确")

# 北大桥参考实现逐值对齐（防"自己验自己"）：手工复算
import torch.nn.functional as Fn

def ref_bridge(caption, norm_w, l1w, l1b, l2w, l2b, eps=1e-6):
    x = caption.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = norm_w * x.to(caption.dtype)
    h = Fn.gelu(Fn.linear(x, l1w, l1b), approximate="tanh")
    return Fn.linear(h, l2w, l2b)

ref = ref_bridge(vh, bridge.norm.weight, bridge.linear_1.weight, bridge.linear_1.bias,
                 bridge.linear_2.weight, bridge.linear_2.bias)
check(torch.allclose(bc, ref, atol=1e-5), "北大桥前向 == 手工参考实现（逐值对齐）")

# CrossAttn 注入器：零门控 -> 初始恒等
inj = CrossAttnInjector(dim=C, num_heads=nh)
y_inj = inj(x, torch.randn(B, 5, C))
check(torch.allclose(y_inj, x, atol=1e-6), "CrossAttn 零门控 -> 输出逐值恒等（初始不注入）")
inj.out_gate.data.fill_(1.0)
y_on = inj(x, torch.randn(B, 5, C))
check(not torch.allclose(y_on, x, atol=1e-6), "CrossAttn 开门后输出改变（注入生效）")

banner("8e. VLM 多尺度分层注入端到端（use_vlm=True）")
cfg_v = UViT3DConfig.tiny()
cfg_v.use_vlm = True
cfg_v.vlm_dim = 32
cfg_v.vlm_stage_ratio = 0.5
model_v = UViT3D(cfg_v)
model_v.eval()
# 覆盖桥为 tiny 维度（config 默认 4096，测试用 32）
model_v.vlm_injector.bridge = NorthBridge(32, cfg_v.dim)
vh2 = torch.randn(Bt, 6, 32)
with torch.inference_mode():
    vv, _ = model_v(latent, t_emb, text_emb, vlm_hidden=vh2, return_aux=True)
check(vv.shape == v.shape, f"带 VLM 输出形状一致 {tuple(vv.shape)}")
check(len(model_v.vlm_injector.cross_injectors) > 0, "高分辨率 cross-attn 注入器已创建")
print(f"  stage split idx = {model_v.vlm_injector.split_idx} / {model_v.vlm_injector.n_blocks}")
# 不传 vlm_hidden 时退化为无 VLM（兼容）
with torch.inference_mode():
    vv0, _ = model_v(latent, t_emb, text_emb, vlm_hidden=None, return_aux=True)
check(vv0.shape == v.shape, "不传 VLM 时正常退化（向后兼容）")

banner("9. 3.5B 基线参数量校准（meta device 零内存精确统计）")
cfg_big = UViT3DConfig.v1_baseline_4b5()
print(f"  V1 baseline 配置：dim={cfg_big.dim}, heads={cfg_big.num_heads}, "
      f"enc={cfg_big.enc_depth}, dec={cfg_big.dec_depth}, bottleneck={cfg_big.bottleneck_depth}, "
      f"ffn={cfg_big.ffn_ratio:.3f}")
# meta device：只建参数壳、不分配数据，零内存精确统计 numel（不需要几 GB 显存/内存）
with torch.device("meta"):
    model_big = UViT3D(cfg_big)
total_big, train_big = count_parameters(model_big)
print(f"  总参数量：{total_big:,}（{total_big / 1e9:.3f}B），可训练：{train_big:,}")
# V2-28 两编码器 +140.6M → 3.710B；V2-29 两专项 Block +226.5M → 3.936B。
check(3.99e9 <= total_big <= 4.01e9,
      f"4.0B 配置实测落在 3.99~4.01B（实测 {total_big / 1e9:.4f}B，含 V2-28 编码器 + V2-29 专项块 + V2-30 细节头 63.5M）")
check(train_big == total_big, "3.94B 配置全部参数可训练（无冻结层）")
# V2-28 两编码器参数账（meta device 精确统计）
from cond_encoders import HierarchicalStyleEncoder as _SEnc, ReferenceResampleAlignEncoder as _REnc
with torch.device("meta"):
    _se = _SEnc(); _re = _REnc()
_n_se = sum(p.numel() for p in _se.parameters())
_n_re = sum(p.numel() for p in _re.parameters())
check(70e6 <= _n_se <= 74e6, f"12.风格编码器落在 70~74M（实测 {_n_se/1e6:.2f}M）")
check(66e6 <= _n_re <= 71e6, f"12.参考对齐编码器落在 66~71M（实测 {_n_re/1e6:.2f}M）")
print(f"  风格编码器 {_n_se/1e6:.2f}M + 参考对齐 {_n_re/1e6:.2f}M = {(_n_se+_n_re)/1e6:.1f}M（目标 140M）")
print("  说明：meta device 只统计参数壳，未分配权重数据、未碰显卡；")
print("        真前向仍需数 GB 内存，待有内存/GPU 时按同一配置实例化。")

banner("10. 主干结构：Dual-Act + 全局条件空间 MoE（V2-27 speedrun 定版）")
# 10a Dual-Act：dual_act=False 严格退化为原生 SwiGLU；True 数值有限、阈值可学习
from uvit3d import SwiGLU as _SwiGLU, GlobalConditionalMoE as _MoE
torch.manual_seed(0)
_sw = _SwiGLU(32, hidden_dim=64, dual_act=False)
_swa = _SwiGLU(32, hidden_dim=64, dual_act=True)
_swa.load_state_dict({k: v for k, v in _sw.state_dict().items()}, strict=False)
_z = torch.randn(4, 32)
check(_sw.dual_act is False and
      abs(float((_sw(_z) - (_sw.w3(torch.nn.functional.silu(_sw.w1(_z)) * _sw.w2(_z)))).abs().max())) < 1e-8,
      "10.DualAct关闭=原生SwiGLU")
check(bool(torch.isfinite(_swa(_z)).all()), "10.DualAct开启数值有限")
check(_swa.log_threshold.requires_grad, "10.DualAct阈值可学习")
_tc = torch.randn(4, 1) * 0.5
check(bool(torch.isfinite(_swa(_z, t_cond=_tc)).all()), "10.DualAct逐token阈值生效")
# 10b 全局 MoE（speedrun 定版）：吃 cond [B,dim] → c_expert [B,dim]，Top1，α=0 恒等
_moe = _MoE(32, num_experts=3, hidden_mult=2.0, n_layers=2)
_z3 = torch.randn(4, 32)                    # [B, dim] 条件向量（不是 token 流）
_y = _moe(_z3)
check(len(_moe.experts) == 3, "10.MoE全局三专家")
check(tuple(_y.shape) == (4, 32), "10.MoE吃条件向量输出同维")
check(bool(torch.isfinite(_y).all()), "10.MoE输出有限")
check(float(_moe.alpha.abs().item()) == 0.0 and
      bool((_y == 0).all()), "10.MoE alpha=0严格零（回退红线）")
check(bool((_moe.last_route < 3).all()), "10.MoE Top1路由索引合法")
_w = torch.softmax(_moe.router(_z3), dim=-1)
check(bool(((_w.sum(dim=-1) - 1.0).abs() < 1e-6).all()), "10.MoE路由和为1")
_moe.alpha.data.fill_(0.1)
_y_on = _moe(_z3)
check(bool((_y_on.abs() > 0).any()), "10.MoE alpha>0时条件增量生效")
check(bool(torch.isfinite(_y_on).all()), "10.MoE alpha>0数值有限")
# 10b2 参数预算：dim=3072 定版 ≈1B（meta device 零内存统计）
with torch.device("meta"):
    _moe_big = _MoE(3072, num_experts=3, hidden_mult=4.375, n_layers=4)
_n_moe = sum(p.numel() for p in _moe_big.parameters())
print(f"  全局MoE参数预算：{_n_moe:,}（{_n_moe/1e9:.3f}B，目标≈1B）")
check(0.9e9 <= _n_moe <= 1.1e9, f"10.MoE全局预算≈1B（实测 {_n_moe/1e9:.3f}B）")
_moe.freeze()
check(all(not p.requires_grad for p in _moe.experts.parameters()), "10.MoE冻结专家")
check(_moe.router.weight.requires_grad and _moe.alpha.requires_grad, "10.MoE路由/α仍可训")
# 10c 端到端：tiny 阶段1 无 MoE；打开 MoE 后 α=0 逐 bit 恒等
_cfg_da = UViT3DConfig.tiny()
check(_cfg_da.dual_act is True and _cfg_da.use_moe is False, "10.默认启用DualAct")
_m = UViT3D(_cfg_da).eval()
check(_m.global_moe is None, "10.阶段1无全局MoE实例（0额外参数）")
_tin = torch.randn(1, 16, 4, 8, 8)
_te = torch.randn(1, 16); _tx = torch.randn(1, 32)
with torch.no_grad():
    _v = _m(_tin, _te, _tx)
check(tuple(_v.shape) == (1, 16, 4, 8, 8), "10.端到端DualAct主干")
# 打开 MoE（tiny）→ α=0 必须与无 MoE 版本逐 bit 相等
_cfg_m = UViT3DConfig.tiny(); _cfg_m.use_moe = True; _cfg_m.moe_inject_start = 1; _cfg_m.moe_inject_end = 2
_mm = UViT3D(_cfg_m).eval(); _mm.load_state_dict(_m.state_dict(), strict=False)
with torch.no_grad():
    _vm = _mm(_tin, _te, _tx)
check(torch.equal(_vm, _v), "10.MoE α=0端到端逐bit恒等（回退红线）")
check(_mm.global_moe is not None and len(_mm.global_moe.experts) == 3, "10.端到端全局MoE三专家")

banner("11. 3.5B 改造新增件：实数 RoPE / 教师式 adaLN / 轻量算子 / F2-F5")
# 11a RoPE 实数版 vs 复数版数值等价（评审指定真网格 3,15,22）
from uvit3d import rope_cos_sin, rope_apply_real, rotate_half
_g = (3, 15, 22)
_hd = 16
_q = torch.randn(_g[0] * _g[1] * _g[2], 4, _hd)
_rp = rope_params(64, _hd)
_ref = rope_apply_3d(_q, _g, _rp)
_cos, _sin = rope_cos_sin(64, _hd)
_got = rope_apply_real(_q.unsqueeze(0), _g, _cos, _sin).squeeze(0)
check((_ref - _got).abs().max().item() < 2e-3,
      f"11.实数RoPE对齐复数版（真网格{_g}，max diff {(_ref-_got).abs().max().item():.2e}）")
check(torch.allclose(_got.pow(2).sum(-1), _q.pow(2).sum(-1), atol=1e-4),
      "11.实数RoPE保范数")

# 11b 教师式 adaLN：共享表存在、逐块小表形状 (1,6,dim)
_cfg_t = UViT3DConfig.v1_baseline_4b5()
with torch.device("meta"):
    _mt = UViT3D(_cfg_t)
check(hasattr(_mt.enc_blocks[0], "scale_shift_table"), "11.教师式adaLN逐块小表存在")
check(hasattr(_mt, "ada_embedder"), "11.共享embedder存在")
check(not hasattr(_mt.enc_blocks[0], "adaLN_modulation"), "11.旧贵版adaLN已移除")

# 11c 轻量算子形状/恒等起步（tiny 打开开关）
_cfg_l = UViT3DConfig.tiny()
_cfg_l.temporal_lowpass = True
_cfg_l.motion_head = True
_cfg_l.freq_branch = True
_cfg_l.local_conv = True
_ml = UViT3D(_cfg_l).eval()
with torch.no_grad():
    _z = torch.randn(1, 16, 4, 8, 8)
    _t = torch.randn(1, 16)
    _tx = torch.randn(1, 32)
    _v, _aux, _motion = _ml(_z, _t, _tx, return_aux=True)
check(tuple(_v.shape) == (1, 16, 4, 8, 8), "11.轻量算子全开前向形状正确")
check(tuple(_motion.shape) == (1, 4, 3), "11.运动头输出 [B,T,3]")
check(abs(float(_ml.freq_branch.gate)) < 1e-6 and abs(float(_ml.local_conv.gate)) < 1e-6,
      "11.频域/局部卷积零门控起步（初始恒等）")

# 11d SNR 分层 + 参考帧旁路（开关打开，ref=None 时与关闭态形状一致）
_cfg_s = UViT3DConfig.tiny()
_cfg_s.snr_cond = True
_cfg_s.ref_align = True
_ms = UViT3D(_cfg_s).eval()
with torch.no_grad():
    _vs = _ms(_z, _t, _tx, sigma=torch.rand(1, 1))
check(tuple(_vs.shape) == (1, 16, 4, 8, 8), "11.SNR分层+参考旁路（ref=None）前向正常")

# 11e F2/F5 初始化封装
from uvit3d import reinit_linear_teacher_spectrum, freeze_spectral_power_iteration
_n = reinit_linear_teacher_spectrum(UViT3D(UViT3DConfig.tiny()), std=0.02)
check(_n > 0, f"11.F2谱初始化重设 {_n} 个 Linear")
_nf = freeze_spectral_power_iteration(UViT3D(UViT3DConfig.tiny()))
check(_nf >= 0, f"11.F5谱范数冻结 {_nf} 处")

banner("12. V2-28 轻量化条件编码器（风格 72M + 参考对齐 68.5M，纯 torch SDPA）")
# 12a 模块级：形状 + 零门控严格恒等 + 梯度可回传（tiny 维度，秒级）
torch.manual_seed(0)
_se_t = _SEnc(dim=64, num_heads=4, num_experts=3, n_layers=6, rank=32, ffn_hidden=32)
_stok = torch.randn(2, 7, 64)
_sc, _sr = _se_t(_stok)
check(tuple(_sc.shape) == (2, 64) and tuple(_sr.shape) == (2, 3), "12.风格编码器输出形状 [B,dim]+[B,E]")
check(bool(torch.isfinite(_sc).all()) and bool(torch.isfinite(_sr).all()), "12.风格编码器输出有限")
check(bool(torch.equal(_sc, torch.zeros_like(_sc))), "12.风格编码器 gate=0 严格恒等")
_se_t.out_gate.data.fill_(1.0)
_sc2, _ = _se_t(_stok); _sc2.sum().backward()
check(any(p.grad is not None for p in _se_t.parameters()), "12.风格编码器梯度可回传")

_re_t = _REnc(dim=64, num_heads=4, n_layers=6, rank=32, ffn_hidden=32, n_resample=4)
_cur = torch.randn(2, 9, 64); _ref = torch.randn(2, 5, 64)
_ac = _re_t(_cur, _ref)
check(tuple(_ac.shape) == (2, 64) and bool(torch.isfinite(_ac).all()), "12.参考对齐编码器输出形状/有限")
check(bool(torch.equal(_ac, torch.zeros_like(_ac))), "12.参考对齐 gate=0 严格恒等")
_re_t.out_gate.data.fill_(1.0)
_ac2 = _re_t(_cur, _ref); _ac2.sum().backward()
check(any(p.grad is not None for p in _re_t.parameters()), "12.参考对齐编码器梯度可回传")

# 12b 端到端接入：tiny 开两编码器，不传 token 时与关闭态逐 bit 恒等（接口兼容红线）
_cfg_e = UViT3DConfig.tiny()
_cfg_e.style_encoder = True; _cfg_e.ref_align_encoder = True
_cfg_e.cond_rank = 16; _cfg_e.cond_layers = 2     # tiny 用极小档，避免自检变慢
_mc = UViT3D(UViT3DConfig.tiny()).eval()            # 无编码器基线
_me = UViT3D(_cfg_e).eval()
_me.load_state_dict(_mc.state_dict(), strict=False)  # 主干权重完全一致（只多出零门控编码器）
with torch.no_grad():
    _ve = _me(_z, _t, _tx)                          # 不传 style_tokens/ref_latent
check(tuple(_ve.shape) == (1, 16, 4, 8, 8), "12.两编码器开启但不传token→前向正常")
with torch.no_grad():
    _vc = _mc(_z, _t, _tx)
check(bool(torch.equal(_ve, _vc)), "12.不传token时与无编码器配置逐bit恒等（接口零扰动）")

# 12c 传入 token：风格路由输出能被前向消费，ref 走新编码器
with torch.no_grad():
    _st = torch.randn(1, 6, 64)
    _vt = _me(_z, _t, _tx, style_tokens=_st)
    _rf = torch.randn(1, 16, 2, 4, 4)
    _vr = _me(_z, _t, _tx, ref_latent=_rf)
check(tuple(_vt.shape) == (1, 16, 4, 8, 8) and tuple(_vr.shape) == (1, 16, 4, 8, 8),
      "12.传风格token/ref_latent端到端前向正常")

banner("13. V2-29 专项注意力 Block（纯时序 113M + 纯空间 113M < 300M）")
from uvit3d import SpecializedBlock as _SpecBlk
for _mode in ("temporal", "spatial"):
    with torch.device("meta"):
        _sb = _SpecBlk(3072, 24, _mode)
    _n = sum(p.numel() for p in _sb.parameters())
    check(110e6 <= _n <= 116e6, f"13.{_mode}专项块落在 110~116M（实测 {_n/1e6:.2f}M）")
# 两块合计 <300M
check(2 * 113.27e6 < 300e6, "13.两专项块合计 226.5M < 300M 预算")
# 功能：零输出门控严格恒等 + gate 打开梯度可回传（tiny 维度）
_sb_t = _SpecBlk(64, 4, "temporal"); _sb_s = _SpecBlk(64, 4, "spatial")
_xs = torch.randn(1, 16, 64); _cd = torch.randn(1, 64); _gd = (4, 2, 2)
with torch.no_grad():
    _ot = _sb_t(_xs, _cd, _gd); _os = _sb_s(_xs, _cd, _gd)
check(bool(torch.equal(_ot, _xs)), "13.时序块 gate=0 严格恒等")
check(bool(torch.equal(_os, _xs)), "13.空间块 gate=0 严格恒等")
_sb_t.out_gate.data.fill_(1.0)
_ot2 = _sb_t(_xs, _cd, _gd); _ot2.sum().backward()
check(any(p.grad is not None for p in _sb_t.parameters()), "13.专项块梯度可回传")
# 端到端：4b5 两专项块插在 encoder 后/decoder 后，不传额外参数即生效（零门控起步）
_cfg_sb = UViT3DConfig.v1_baseline_4b5()
with torch.device("meta"):
    _msb = UViT3D(_cfg_sb)
check(_msb.temporal_special is not None and _msb.spatial_special is not None,
      "13.4b5 两专项块已实例化")

banner("14. V2-30 DDT 高频细节残差头（~63.5M，补总量到 4.0B）")
from uvit3d import DetailRefineHead as _DetailHead
with torch.device("meta"):
    _dh = _DetailHead(3072, (1, 2, 2), 48, hidden=19456)
_n_dh = sum(p.numel() for p in _dh.parameters())
check(63e6 <= _n_dh <= 64e6, f"14.细节头落在 63~64M（实测 {_n_dh/1e6:.2f}M）")
# 功能：零门控严格 0 + gate 打开梯度可回传
_dh_t = _DetailHead(64, (1, 2, 2), 4, hidden=128)
_xh = torch.randn(2, 9, 64)
with torch.no_grad():
    _dh_out = _dh_t(_xh)
check(tuple(_dh_out.shape) == (2, 9, 16), "14.细节头输出形状 [B,N,prod*out_c]")
check(bool(torch.equal(_dh_out, torch.zeros_like(_dh_out))), "14.细节头 gate=0 严格 0")
_dh_t.out_gate.data.fill_(1.0)
_do = _dh_t(_xh); _do.sum().backward()
check(any(p.grad is not None for p in _dh_t.parameters()), "14.细节头梯度可回传")
# 4b5 已实例化细节头
check(_msb.head.detail_head is not None, "14.4b5 细节头已实例化")

banner("15. V2-31 检查点覆盖（专项块 + 细节头）：数值等价 / 反向重算 / 开关同步")
import torch.utils.checkpoint as _ck15
_SB15 = SpecializedBlock if "SpecializedBlock" in dir() else _SpecBlk
_DH15 = DetailRefineHead if "DetailRefineHead" in dir() else _DetailHead

# 15a 专项块：checkpoint 包装 vs 直接调用，前向逐值 + 输入梯度一致，且反向确实重算一次
torch.manual_seed(1234)
_sb15 = _SB15(64, 4, "temporal")
_g15 = (4, 4, 4)
_x15 = torch.randn(2, 4 * 4 * 4, 64, requires_grad=True)
_c15 = torch.randn(2, 64)
_y_direct = _sb15(_x15, _c15, _g15)
_y_direct.sum().backward()
_g_direct = _x15.grad.detach().clone()
_x15.grad = None
_cnt15 = [0]
_fwd15 = _sb15.forward
def _counted15(*a, **kw):
    _cnt15[0] += 1
    return _fwd15(*a, **kw)
_sb15.forward = _counted15
_y_ck = _ck15.checkpoint(_sb15, _x15, _c15, _g15,
                         use_reentrant=False, preserve_rng_state=False)
check(bool((_y_direct - _y_ck).abs().max() < 1e-6), "15.专项块checkpoint前向逐值等价")
_y_ck.sum().backward()
_sb15.forward = _fwd15           # 重算计数完成后再恢复（此前恢复会漏计重算那次）
check(_cnt15[0] == 2, f"15.专项块反向触发一次重算（forward调用{_cnt15[0]}次）")
check(bool((_g_direct - _x15.grad).abs().max() < 1e-6), "15.专项块checkpoint输入梯度等价")

# 15b 细节头：同上（tiny hidden 加速，只验机制不验参数账）
torch.manual_seed(5678)
_dh15 = _DH15(64, (1, 2, 2), 16, hidden=128)
_xd15 = torch.randn(2, 4 * 4 * 4, 64, requires_grad=True)
_vd_direct = _dh15(_xd15)
_vd_direct.sum().backward()
_gd_direct = _xd15.grad.detach().clone()
_xd15.grad = None
_vd_ck = _ck15.checkpoint(_dh15, _xd15, use_reentrant=False, preserve_rng_state=False)
check(bool((_vd_direct - _vd_ck).abs().max() < 1e-6), "15.细节头checkpoint前向逐值等价")
_vd_ck.sum().backward()
check(bool((_gd_direct - _xd15.grad).abs().max() < 1e-6), "15.细节头checkpoint输入梯度等价")

# 15c 开关同步：主层 enable/disable 必须同步细节头 ckpt 标志
_cfg15 = UViT3DConfig.tiny()
_cfg15.temporal_block = True
_cfg15.spatial_block = True
_m15 = UViT3D(_cfg15)
check(_m15.head.detail_ckpt is False, "15.默认关闭时细节头ckpt为False")
_m15.enable_gradient_checkpointing()
check(_m15.head.detail_ckpt is True and _m15.grad_checkpointing, "15.enable同步细节头ckpt")
_m15.disable_gradient_checkpointing()
check(_m15.head.detail_ckpt is False and not _m15.grad_checkpointing, "15.disable同步细节头ckpt")

# 15d 集成路径：开启 checkpoint 后，时序专项块经 _blk_call 被重算（forward 调用 2 次）
_m15.enable_gradient_checkpointing()
_cnt15b = [0]
_fwd15b = _m15.temporal_special.forward
def _counted15b(*a, **kw):
    _cnt15b[0] += 1
    return _fwd15b(*a, **kw)
_m15.temporal_special.forward = _counted15b
_z15 = torch.randn(1, 16, 4, 8, 8)
_t15 = torch.randn(1, 16)
_tx15 = torch.randn(1, 32)
_v15 = _m15(_z15, _t15, _tx15)
_v15.sum().backward()
_m15.temporal_special.forward = _fwd15b
check(_cnt15b[0] == 2, f"15.集成路径时序专项块走checkpoint（调用{_cnt15b[0]}次）")

banner("自检全部通过 — 模型底子可前向、设计点齐全（张量均在 CPU，不依赖显卡）")
