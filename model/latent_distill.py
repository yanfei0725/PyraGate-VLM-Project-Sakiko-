# -*- coding: utf-8 -*-
"""latent_distill.py — 在线零缓存中间层蒸馏（前缀 ``ldistill.``）。

设计目标（融合方案 §4「中间层特征蒸馏：在线低秩蒸馏，零特征缓存」）：
  * 教师冻结、推理态、``no_grad``，与学生**同步前向**，一步算一步，不预存
    2200 条教师中间特征；
  * 只挂 2 层（默认全局第 8、16 层）；学生/教师各挂一层一对 ``rank=16`` 低秩
    降维投影，在低维空间算 MSE + 余弦；
  * 零缓存：只保留当前步瞬时激活，loss 算完立即清空，不留历史张量；
  * **没有挂载教师时前向必须显式抛错**，不许静默跑空 loss 假装蒸馏。

挂接方式（第二阶段集成到 train_staged.py，本文件不改训练脚本）：
  1. 构造：``dist = OnlineLowRankDistill(student_dim=3072, teacher_dim=3072,
     layer_indices=(8, 16), rank=16)``
  2. ``dist.attach(student_blocks, teacher_blocks)``：两个等长的 block 列表，
     顺序与主干前向一致；hook 自动注册。
  3. 训练步：学生、教师各前向一次后 ``loss, parts = dist.forward()``，
     把 ``loss * w`` 加进总 loss 反传；教师侧激活已 detach。
  4. 卸载：``dist.remove_hooks()``。

学生 block 全局序号映射（UViT3D 4b5，见 ``ordered_student_blocks``）：
  enc_blocks 0..11 -> temporal_special(12) -> bottleneck 13..18 ->
  dec_blocks 19..30 -> spatial_special(31)。
  默认层 8 = enc_blocks[8]；层 16 = bottleneck[3]。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_token_tensor(out) -> torch.Tensor:
    """block 输出可能是 tensor 或 tuple/list；取第一个 token 张量 [B,N,D]。"""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"无法识别的 block 输出类型: {type(out).__name__}")


def ordered_student_blocks(model: nn.Module) -> List[nn.Module]:
    """按 UViT3D 前向顺序返回 block 列表（encoder -> temporal -> bottleneck
    -> decoder -> spatial）。专项块为 None 时自动跳过。

    注意：本函数只读取模型子模块，不修改 uvit3d.py 任何代码。
    """
    blocks: List[nn.Module] = []
    blocks.extend(list(getattr(model, "enc_blocks", [])))
    if getattr(model, "temporal_special", None) is not None:
        blocks.append(model.temporal_special)
    blocks.extend(list(getattr(model, "bottleneck", [])))
    blocks.extend(list(getattr(model, "dec_blocks", [])))
    if getattr(model, "spatial_special", None) is not None:
        blocks.append(model.spatial_special)
    return blocks


class OnlineLowRankDistill(nn.Module):
    """在线低秩中间层蒸馏。

    Args:
        student_dim: 学生 block 输出 token 维度（4b5 = 3072）。
        teacher_dim: 教师 block 输出 token 维度（Wan5B 按实际隐藏维传入）。
        layer_indices: 挂接的全局层序号（默认 (8, 16)）。
        rank: 低秩降维瓶颈（契约默认 16）。
        w_mse: 低维 MSE 权重；w_cos: 低维余弦权重。
    """

    def __init__(self, student_dim: int, teacher_dim: int,
                 layer_indices: Sequence[int] = (8, 16), rank: int = 16,
                 w_mse: float = 1.0, w_cos: float = 1.0,
                 enabled: bool = False) -> None:
        super().__init__()
        # 默认关闭：未显式 --latent-distill 时前向返回 0 loss、空 parts，
        # 不注册/不依赖教师，训练路径与基线等价。
        self.enabled = bool(enabled)
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.rank = int(rank)
        self.w_mse = float(w_mse)
        self.w_cos = float(w_cos)

        n = len(self.layer_indices)
        # 学生/教师各自一份降维投影；低维空间对齐。
        self.s_down = nn.ModuleList(
            [nn.Linear(self.student_dim, self.rank) for _ in range(n)])
        self.t_down = nn.ModuleList(
            [nn.Linear(self.teacher_dim, self.rank) for _ in range(n)])

        self._student_blocks: Optional[List[nn.Module]] = None
        self._teacher_blocks: Optional[List[nn.Module]] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._s_act: List[Optional[torch.Tensor]] = [None] * n
        self._t_act: List[Optional[torch.Tensor]] = [None] * n

    # ------------------------------------------------------------ 挂接管理
    def attach(self, student_blocks: Sequence[nn.Module],
               teacher_blocks: Sequence[nn.Module]) -> None:
        """注册 forward hook。两侧 block 数必须等于 layer_indices 长度。"""
        self.remove_hooks()
        s = list(student_blocks)
        t = list(teacher_blocks)
        if len(s) != len(self.layer_indices) or len(t) != len(self.layer_indices):
            raise ValueError(
                f"挂接 block 数不匹配: student={len(s)} teacher={len(t)} "
                f"layers={len(self.layer_indices)}")
        self._student_blocks, self._teacher_blocks = s, t

        for i, blk in enumerate(s):
            self._handles.append(blk.register_forward_hook(self._make_hook(i, "s")))
        for i, blk in enumerate(t):
            self._handles.append(blk.register_forward_hook(self._make_hook(i, "t")))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._clear_acts()

    def _clear_acts(self) -> None:
        self._s_act = [None] * len(self.layer_indices)
        self._t_act = [None] * len(self.layer_indices)

    @property
    def has_teacher(self) -> bool:
        return self._teacher_blocks is not None

    def _make_hook(self, i: int, side: str) -> Callable:
        def _hook(module, _inp, out):
            t = _as_token_tensor(out)
            if side == "t":
                # 教师侧 detach：教师 no_grad 推理，激活不进学生反传图。
                t = t.detach()
            if side == "s":
                self._s_act[i] = t
            else:
                self._t_act[i] = t
        return _hook

    # ------------------------------------------------------------ 前向 loss
    @staticmethod
    def _pool_tokens(z: torch.Tensor) -> torch.Tensor:
        """[B,N,D] -> [B,D]（token 维全局均值池化）。"""
        if z.dim() == 3:
            return z.mean(dim=1)
        if z.dim() == 2:
            return z
        raise ValueError(f"token 特征必须是 2D/3D，got {tuple(z.shape)}")

    def forward(self) -> Tuple[torch.Tensor, dict]:
        """计算中间层蒸馏 loss。

        ``enabled=False``（默认）：返回 0 loss + 空 parts，不碰教师、不建图。
        ``enabled=True`` 但未挂载教师：显式抛 RuntimeError，不许静默跑空 loss。
        """
        if not self.enabled:
            self._clear_acts()
            return torch.zeros((), dtype=torch.float32), {}
        if not self.has_teacher:
            raise RuntimeError(
                "[ldistill] 未挂载教师模型：在线中间层蒸馏需要冻结教师同步前向。"
                "当前离线 v_teacher 缓存模式不支持在线层蒸馏；"
                "如需层对齐，请先 attach() 教师 block，不要把本 loss 当 0 跑。")
        n = len(self.layer_indices)
        parts: dict = {}
        total = torch.zeros((), dtype=torch.float32)
        any_finite = False

        for i in range(n):
            sa, ta = self._s_act[i], self._t_act[i]
            if sa is None or ta is None:
                raise RuntimeError(
                    f"[ldistill] 层 {self.layer_indices[i]} 缺激活："
                    f"student_act={'None' if sa is None else 'ok'} "
                    f"teacher_act={'None' if ta is None else 'ok'}；"
                    "请确认学生/教师都已完成本次前向。")
            ps = self._pool_tokens(sa).float()
            pt = self._pool_tokens(ta).float()
            zs = self.s_down[i](ps)
            zt = self.t_down[i](pt).detach()

            l_mse = F.mse_loss(zs, zt)
            l_cos = 1.0 - F.cosine_similarity(zs, zt, dim=-1).mean()
            li = self.w_mse * l_mse + self.w_cos * l_cos
            total = total + li
            parts[f"layer{self.layer_indices[i]}_mse"] = float(l_mse.detach())
            parts[f"layer{self.layer_indices[i]}_cos"] = float(l_cos.detach())
            any_finite = True

        if not any_finite:
            raise RuntimeError("[ldistill] 没有任何挂接层产生 loss")
        # 零缓存：loss 取完立即清空本轮激活。
        self._clear_acts()
        return total, parts

    # ------------------------------------------------------------ 参数统计
    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# CPU 自检（不碰 GPU、不碰主干）
# ---------------------------------------------------------------------------
def _selftest() -> None:
    def ck(cond: bool, msg: str) -> None:
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        if not cond:
            raise AssertionError(msg)

    print("=== latent_distill 纯 CPU 自检 ===")
    torch.manual_seed(0)
    B, N, Ds, Dt, R = 2, 16, 32, 48, 16

    class _FakeBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            return self.proj(x)

    student = nn.ModuleList([_FakeBlock(Ds) for _ in range(2)])
    teacher = nn.ModuleList([_FakeBlock(Dt) for _ in range(2)])
    # 教师冻结
    for p in teacher.parameters():
        p.requires_grad_(False)

    dist = OnlineLowRankDistill(student_dim=Ds, teacher_dim=Dt,
                                layer_indices=(8, 16), rank=R)
    total, trainable = dist.count_parameters()
    print(f"  [info] params total={total} trainable={trainable} ({total/1e3:.1f}K)")
    ck(total < 39_070_000, "参数量远低于主干 1% 预算（39.07M）")

    # 默认关闭：前向返回 0 loss + 空 parts，不依赖教师、不建图。
    loss_off, parts_off = dist.forward()
    ck(float(loss_off) == 0.0 and parts_off == {}, "默认关闭：0 loss + 空 parts（基线等价）")

    # 开启但未挂教师：前向必须报错
    dist.enabled = True
    try:
        dist.forward()
        ck(False, "开启且无教师挂载应抛 RuntimeError")
    except RuntimeError as e:
        ck("未挂载教师" in str(e), "开启无教师抛 RuntimeError 且文案明确")

    dist.attach(student, teacher)
    ck(dist.has_teacher, "attach 后 has_teacher=True")

    xs = torch.randn(B, N, Ds, requires_grad=True)
    xt = torch.randn(B, N, Dt)
    with torch.no_grad():
        for blk in teacher:
            _ = blk(xt)
    for blk in student:
        xs = blk(xs)

    loss, parts = dist.forward()
    ck(torch.isfinite(loss).item(), f"蒸馏 loss 有限 {loss.item():.4f}")
    ck(len(parts) == 4, f"两层各出 mse/cos 共 4 项: {list(parts)}")
    # 零缓存：算完激活清空
    ck(all(a is None for a in dist._s_act) and all(a is None for a in dist._t_act),
       "loss 算完本轮激活已清空（零缓存）")

    loss.backward()
    ck(student[0].proj.weight.grad is not None, "学生 block 收到梯度")
    ck(all(p.grad is None for p in teacher.parameters()), "教师参数无梯度（冻结/no_grad）")

    # 挂接数量校验
    try:
        dist.attach(student[:1], teacher)
        ck(False, "block 数不匹配应抛 ValueError")
    except ValueError:
        ck(True, "block 数不匹配抛 ValueError")

    dist.remove_hooks()
    ck(not dist._handles, "remove_hooks 后句柄清空")
    print("=== latent_distill 自检全部通过 ===")


if __name__ == "__main__":
    _selftest()
