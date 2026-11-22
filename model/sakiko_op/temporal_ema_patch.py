# -*- coding: utf-8 -*-
"""temporal_ema_patch.py -- DiT self-attn 输出的时序低通 patch（推理时开关，零训练）。

数学：对每个 block 的 self_attn 输出特征，沿时间轴做一阶滑动平均::

    feat'_t = beta * feat'_t-1 + (1-beta) * feat_t      （因果 EMA，t 从 0 递增）
    beta = 1.0 → 恒等（关闭）；beta 越小 → 平滑越强（画面易糊/迟缓）。

落点：comfy Wan 的 ``attn1_patch``（self_attn 输出级钩子，model.py:64-70 原生支持）。
不进入注意力数学、不改权重、不动 block 内部公式——纯输出后处理，
随时可关（beta=1 或不挂 patch）。

token 布局契约：Wan 序列 = (T_lat × H_lat × W_lat) 展平，帧索引 = idx // (H*W)。
patch 工厂需要 (T, H, W) 来拆时间轴；调用方（节点侧）持有 latent 网格信息。
"""
from __future__ import annotations

import torch


def make_temporal_ema_patch(beta: float, T: int, H: int, W: int):
    """构造 attn1_patch 兼容的时序 EMA 钩子。

    beta: 保留系数（1.0=关闭）；T/H/W: latent 网格（帧数×高×宽 token 数）。
    返回 fn({"x": ..., "transformer_options": ...}) -> x'。
    """
    assert 0.0 <= beta <= 1.0
    S = H * W
    assert T >= 1 and S >= 1
    state = {"prev": None}  # EMA 状态跨 denoise 步要重置：按采样步判定

    def patch(d):
        x = d["x"]  # (b, s, heads*dim)
        b, s, c = x.shape
        if s != T * S:
            return x  # 网格不匹配（异常输入）→ 原样放行
        to = d.get("transformer_options", {}) or {}
        step_sig = to.get("sample_step_sig", id(d.get("transformer_options")))
        if state["prev"] is not None and state["prev_sig"] != step_sig:
            state["prev"] = None  # 新的采样步序列：EMA 状态清零
        state["prev_sig"] = step_sig

        xb = x.view(b, T, S, c)
        if beta >= 1.0:
            return x
        out = torch.empty_like(xb)
        if state["prev"] is None or state["prev"].shape != xb.shape:
            prev = xb[:, 0:1].clone()
        else:
            prev = state["prev"].to(xb.dtype)
        # 因果 EMA 沿 T 逐帧（T 最多 31，python 循环开销可忽略）
        cur = prev[:, 0]
        for t in range(T):
            cur = beta * cur + (1.0 - beta) * xb[:, t]
            out[:, t] = cur
        state["prev"] = cur.unsqueeze(1).detach().clone()
        return out.reshape(b, s, c)

    return patch


def register_temporal_ema(model, beta: float, T: int, H: int, W: int,
                          blocks: str = "all"):
    """把 EMA patch 注册到 Wan 模型的 attn1_patch（全部或指定 block）。"""
    patch = make_temporal_ema_patch(beta, T, H, W)

    def wrapped(d):
        idx = d.get("transformer_options", {}).get("block_index", None)
        if blocks != "all" and (idx is None or not (idx in blocks)):
            return d["x"]
        return patch(d)

    patched = model.clone()
    patched.set_model_patch(wrapped, "attn1_patch")
    patched._temporal_ema = {"beta": beta, "T": T, "H": H, "W": W, "blocks": blocks}
    return patched
