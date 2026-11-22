# -*- coding: utf-8 -*-
"""distill_loss.py — Wan 5B → 4.5B U-ViT 蒸馏损失（白皮书 §8.2）的纯 CPU 实现。

四项损失（对齐 §8.2 蒸馏课程 / §3.7）：
  1. velocity_mse      : 学生预测速度场 v_pred 与教师速度场 v_teacher 的 MSE（主目标）
  2. feature_align     : 教师-学生层特征对齐（§3.7 层特征对齐 loss，弥补与教师不同构）
  3. temporal_consist  : 时序一致性（相邻帧速度场变化平滑，配合 2+1D）
  4. flicker_reg       : 帧间闪烁正则（§3.6 时序辅助头 / §8.2 flicker 正则）

职责边界：本模块只定义 loss 与其 CPU 自检，**不做真训练**（真训练需 GPU + 数据）。
设计要点：所有 loss 返回标量、可反传；开关项关闭时严格为 0；维度不匹配给出清晰报错。

运行（纯 CPU）：
  CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python distill_loss.py
"""
import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class DistillLossConfig:
    """各项权重。0.0 表示关闭该项（关闭时 loss 严格为 0，便于消融）。"""
    w_velocity: float = 1.0
    w_feature: float = 0.1
    w_temporal: float = 0.05
    w_flicker: float = 0.05
    # 特征对齐时若教师/学生通道数不同，用 1x1 投影对齐（训练期可学习；这里只算 loss）
    feature_proj: bool = True


def velocity_mse(v_pred: torch.Tensor, v_teacher: torch.Tensor) -> torch.Tensor:
    """速度场 MSE。v: [B, C, T, H, W]。"""
    if v_pred.shape != v_teacher.shape:
        raise ValueError(f"速度场形状不匹配：pred {tuple(v_pred.shape)} vs teacher {tuple(v_teacher.shape)}")
    return F.mse_loss(v_pred, v_teacher)


def feature_align(feat_student: List[torch.Tensor], feat_teacher: List[torch.Tensor],
                  proj: bool = True) -> torch.Tensor:
    """层特征对齐：逐层比较，通道数不同时用自适应投影（平均池化到同形状后 MSE）。

    真训练里可用可学习 1x1 conv 投影（REPA 风格）；这里用纯 CPU 可实现的方式：
    把两侧特征各自全局平均池化到 [B, C] 再对齐 C（取较小通道数切片），避免引入参数。
    """
    if len(feat_student) != len(feat_teacher):
        raise ValueError(f"特征层数不匹配：student {len(feat_student)} vs teacher {len(feat_teacher)}")
    if not feat_student:
        return torch.tensor(0.0)
    losses = []
    for fs, ft in zip(feat_student, feat_teacher):
        # [B, C, ...] -> [B, C] 全局平均
        ps = fs.mean(dim=tuple(range(2, fs.dim())))
        pt = ft.mean(dim=tuple(range(2, ft.dim())))
        c = min(ps.shape[-1], pt.shape[-1])
        losses.append(F.mse_loss(ps[..., :c], pt[..., :c]))
    return torch.stack(losses).mean()


def temporal_consist(v: torch.Tensor) -> torch.Tensor:
    """时序一致性：相邻帧速度场的差分应平滑（惩罚帧间突变）。v: [B,C,T,H,W]。"""
    if v.dim() != 5 or v.shape[2] < 2:
        return torch.tensor(0.0)
    dv = v[:, :, 1:, :, :] - v[:, :, :-1, :, :]
    return dv.pow(2).mean()


def flicker_reg(v: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
    """帧间闪烁正则：优先用时序辅助头输出（§3.6），否则退化到速度场一阶差分。"""
    if aux is not None:
        # aux: [B, T', H', W'] 逐 token 标量残差；惩罚相邻帧残差跳变
        if aux.dim() == 4 and aux.shape[1] >= 2:
            da = aux[:, 1:, :, :] - aux[:, :-1, :, :]
            return da.pow(2).mean()
    return temporal_consist(v)


@dataclass
class DistillLossOutput:
    total: torch.Tensor
    parts: Dict[str, float]


def compute_distill_loss(
    v_pred: torch.Tensor,
    v_teacher: torch.Tensor,
    feat_student: Optional[List[torch.Tensor]] = None,
    feat_teacher: Optional[List[torch.Tensor]] = None,
    aux: Optional[torch.Tensor] = None,
    cfg: Optional[DistillLossConfig] = None,
) -> DistillLossOutput:
    """计算总蒸馏损失。任一项权重为 0 则严格跳过（返回 0，不引入图节点）。"""
    cfg = cfg or DistillLossConfig()
    parts: Dict[str, float] = {}
    total = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)

    if cfg.w_velocity > 0:
        lv = velocity_mse(v_pred, v_teacher)
        total = total + cfg.w_velocity * lv
        parts["velocity"] = lv.item()
    if cfg.w_feature > 0 and feat_student and feat_teacher:
        lf = feature_align(feat_student, feat_teacher, proj=cfg.feature_proj)
        total = total + cfg.w_feature * lf
        parts["feature"] = lf.item()
    if cfg.w_temporal > 0:
        lt = temporal_consist(v_pred)
        total = total + cfg.w_temporal * lt
        parts["temporal"] = lt.item()
    if cfg.w_flicker > 0:
        lf = flicker_reg(v_pred, aux)
        total = total + cfg.w_flicker * lf
        parts["flicker"] = lf.item()

    return DistillLossOutput(total=total, parts=parts)


def _selftest() -> None:
    def ck(cond, msg):
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== distill_loss 纯 CPU 自检（§8.2 蒸馏四项）===")
    torch.manual_seed(7)
    B, C, T, H, W = 1, 4, 5, 6, 6
    v_pred = torch.randn(B, C, T, H, W, requires_grad=True)
    v_teacher = torch.randn(B, C, T, H, W)
    aux = torch.randn(B, T, H, W)
    fs = [torch.randn(B, 8, T, H, W), torch.randn(B, 8, T, H, W)]
    ft = [torch.randn(B, 16, T, H, W), torch.randn(B, 16, T, H, W)]

    # 1) 全开：有限值、可反传、四项齐全
    out = compute_distill_loss(v_pred, v_teacher, fs, ft, aux)
    ck(torch.isfinite(out.total).item(), f"全开总 loss 有限值 {out.total.item():.4f}")
    ck(set(out.parts) == {"velocity", "feature", "temporal", "flicker"},
       f"四项 loss 齐全 {list(out.parts)}")
    out.total.backward()
    ck(v_pred.grad is not None and torch.isfinite(v_pred.grad).all().item(),
       "反向传播产生有限梯度（可训练）")
    ck(v_pred.grad.abs().sum().item() > 0, "梯度非零（loss 对预测有约束）")

    # 2) 单项开关：关闭时该项不出现在 parts，总 loss 等于其余项
    v_pred2 = torch.randn(B, C, T, H, W, requires_grad=True)
    cfg_off = DistillLossConfig(w_velocity=0.0, w_feature=0.0,
                                w_temporal=0.0, w_flicker=0.0)
    out_off = compute_distill_loss(v_pred2, v_teacher, fs, ft, aux, cfg_off)
    ck(out_off.total.item() == 0.0, "四项全关时总 loss 严格为 0")
    ck(out_off.parts == {}, "四项全关时 parts 为空（无残项）")

    # 3) 仅速度场：总 loss == w_velocity * velocity_mse
    v_pred3 = torch.randn(B, C, T, H, W, requires_grad=True)
    cfg_v = DistillLossConfig(w_velocity=2.0, w_feature=0.0,
                              w_temporal=0.0, w_flicker=0.0)
    out_v = compute_distill_loss(v_pred3, v_teacher, fs, ft, aux, cfg_v)
    expect = 2.0 * velocity_mse(v_pred3, v_teacher)
    ck(torch.allclose(out_v.total, expect), "仅速度场：总 loss = 权重 × MSE（精确）")

    # 4) 形状不匹配要清晰报错
    try:
        velocity_mse(torch.randn(1, 4, 5, 6, 6), torch.randn(1, 8, 5, 6, 6))
        ck(False, "形状不匹配应抛 ValueError")
    except ValueError:
        ck(True, "速度场形状不匹配抛 ValueError（清晰报错）")

    # 5) 时序一致性：常数序列为 0，随机序列 > 0
    const_v = torch.ones(1, 2, 4, 3, 3)
    ck(temporal_consist(const_v).item() == 0.0, "常数速度场时序一致性 loss = 0")
    ck(temporal_consist(v_teacher).item() > 0.0, "随机速度场时序一致性 loss > 0")
    print("=== distill_loss 自检全部通过 ===")


if __name__ == "__main__":
    _selftest()
