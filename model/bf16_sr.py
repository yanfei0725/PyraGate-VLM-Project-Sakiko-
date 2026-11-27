"""bf16 随机舍入（stochastic rounding, SR）+ 选择性 fp32 master 权重。

为什么需要它
------------
本机训练的主干权重是 bf16，且**没有 fp32 master**（train_staged.py:1382 起），
优化器直接把更新写回 bf16 参数，每一步都被 round-to-nearest 抹一次。实测
patchify.weight（RMS≈0.0412）的 bf16 量子 q≈max(rms,|cur|mean)·2^-8≈1.6e-4，
而 Adafactor 的实际有效步长只有 4.6e-5 ≈ **0.29 个量子**，于是每 3~4 步才有
一次真实位移，5200 步累计上限 ~0.24 —— 这就是"名义 lr 1.12e-3、实际空转"。

两条正交的解法：

1. **随机舍入**：把 fp32 结果按"距下端点的比例"随机取到相邻两个 bf16 值。
   期望值严格等于 fp32 结果，因此 0.29q 的更新在以 29% 概率进 1 个量子，
   长期平均位移不再被截断。只改舍入方式，不改步长，不引入额外发散风险。

2. **选择性 fp32 master**：把少数高杠杆张量（输出头/最后的 decoder 块/
   patchify）提升为 fp32 主权重，优化器在 fp32 上累加，再 SR 写回 bf16。
   这一步才真正解决"小于一个量子的位移被吃掉"。全量 fp32 master 需
   +15.7GB（3.93B×4B），20GB 卡不可行；所以按前缀 + 预算选择，默认 0=关闭。

注意
----
- SR 只在**写回**这一处生效，前向/反向仍是 bf16，算力与显存不变。
- 未进入 master 的张量保持原来的 bf16 原位更新，行为与关闭时逐位一致。
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# 1. 随机舍入
# ---------------------------------------------------------------------------
def bf16_quantum(x: torch.Tensor) -> torch.Tensor:
    """bf16 相邻可表示值的间距：q = 2^(floor(log2|x|) - 7)。

    两个容易踩的坑（都在自检里被抓到过）：
    1. 不能对 bf16 张量直接 torch.nextafter：ROCm/CPU 上会先上采成 fp32，
       于是量出来的是 fp32 量子（3.7e-9），不是 bf16 的 1.2e-4。
    2. bf16 是 **1+7 位**尾数（8 位有效数），不是 1+8。间距是 2^(e-7)；
       写成 e-8 会让 lo 落在不可表示的格点上，`.to(bf16)` 时再被 RNE 拉走，
       造成 E[out] != x 的系统偏置（实测 -0.19~-0.44 量子）。
    """
    a = x.detach().to(torch.float32).abs().clamp_min(1e-38)
    e = torch.floor(torch.log2(a))
    return torch.pow(2.0, e - 7.0)


def stochastic_round_to_bf16(x: torch.Tensor,
                             generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """把 fp32 张量随机舍入到 bf16，返回 bf16 张量（期望值 = 输入）。

    做法：按 bf16 的量化格点 q=2^(floor(log2|x|)-8) 取「格点下沿」lo，
    frac=(x-lo)/q∈[0,1)；以 frac 的概率进到 lo+q，否则留在 lo。
    数学上 E[out] = x（无偏），因此 0.29q 的小步长会以 29% 概率真实前进 1 格。

    仅在 fp32 输入上有意义；已是 bf16 的值 frac=0，恒等。
    """
    if not torch.is_tensor(x):
        raise TypeError("stochastic_round_to_bf16 需要 torch.Tensor")
    src = x.detach().to(torch.float32)
    a = src.abs()
    zero = a == 0
    a_safe = torch.where(zero, torch.ones_like(a), a)
    q = bf16_quantum(a_safe)
    lo = torch.floor(a_safe / q) * q
    frac = ((a_safe - lo) / q).clamp(0.0, 1.0)
    if generator is None:
        u = torch.rand_like(frac)
    else:
        u = torch.rand(frac.shape, generator=generator, device=frac.device, dtype=frac.dtype)
    mag = torch.where(u < frac, lo + q, lo)
    out = torch.sign(src) * mag
    out = torch.where(zero, torch.zeros_like(out), out)
    return out.to(torch.bfloat16)


def measure_sr_gain(dtype: torch.dtype = torch.bfloat16) -> Dict[str, float]:
    """CPU 自检：固定 0.3 个量子的位移，比较 RNE 与 SR 的累计位移。"""
    torch.manual_seed(0)
    base = torch.full((1 << 16,), 0.0412, dtype=torch.float32)  # ≈ patchify RMS
    q = float(bf16_quantum(base)[0])
    delta = 0.3 * q
    w_rne = base.clone()
    w_sr = base.clone()
    for _ in range(200):
        w_rne = (w_rne + delta).to(dtype).float()              # 纯 RNE 写回
        w_sr = stochastic_round_to_bf16(w_sr + delta).float()   # SR 写回
    rne_total = float((w_rne - base).mean())
    sr_total = float((w_sr - base).mean())
    return {"quantum": q, "delta": delta,
            "rne_mean_shift_200": rne_total, "sr_mean_shift_200": sr_total,
            "rne_q_per_step": rne_total / (200 * q), "sr_q_per_step": sr_total / (200 * q)}


# ---------------------------------------------------------------------------
# 2. 选择性 fp32 master
# ---------------------------------------------------------------------------
class SelectiveSRMaster:
    """给一部分 bf16 参数挂 fp32 主权重；优化器更新 master，再 SR 写回 bf16。

    生命周期（每步）：
        pull()   梯度裁剪后，把 bf16 的 grad 拷成 fp32 挂到 master
        [opt.step()]                     优化器只动 master
        push()   把 master 随机舍入写回 bf16 参数
        zero_grad()  清 master 的 grad

    参数选择：按 ``prefixes`` 顺序（先匹配的优先），累计 numel 不超过 ``budget_m``
    （百万参数）；budget_m<=0 表示不启用。
    """

    def __init__(self, model: torch.nn.Module,
                 prefixes: Sequence[str] = (),
                 budget_m: float = 0.0,
                 exclude: Iterable[str] = ("full_head.final_proj.weight",
                                           "full_head.alpha", "full_head.out_gain"),
                 include_dim1: bool = False):
        self.model = model
        self.pairs: List[Tuple[str, torch.nn.Parameter, torch.nn.Parameter]] = []
        self.selected_numel = 0
        self.skipped_budget = 0
        # 每多少个 optimizer.step 把 master SR 写回 bf16 一次；1=每步（旧行为）。
        self.push_every = 1
        # --- 生效面审计名单（2026-11-26 追加）---
        self.include_dim1 = bool(include_dim1)
        self.dim1_names: List[str] = []          # 被保护的 dim<=1 scale/bias/向量
        self.skipped_dim1: List[str] = []        # 被 include_dim1=False 跳过的 dim<=1
        self.skipped_exclude: List[str] = []    # 被 exclude 名单跳过
        self.already_fp32: List[str] = []       # 已是 fp32，不需要 master
        self.skipped_budget_names: List[str] = []
        # 历史硬排除：pre_proj_scale 曾在 exclude 列表；include_dim1=False 时按 dim1 跳过。
        legacy_dim1_exclude = {"full_head.pre_proj_scale"}
        if budget_m <= 0:
            return
        budget = int(budget_m * 1e6)
        exclude = set(exclude)
        if not self.include_dim1:
            exclude |= legacy_dim1_exclude
        order = list(prefixes) if prefixes else ["dec_blocks.", "patchify.",
                                                "full_head.", "skips.", "bottleneck."]
        matchers = [self._compile_matcher(p) for p in order]
        params = dict(model.named_parameters())
        chosen: Dict[str, torch.nn.Parameter] = {}
        used = 0
        for matcher in matchers:
            for n, p in params.items():
                if n in chosen:
                    continue
                if not self._match(n, matcher):
                    continue
                if n in exclude:
                    self.skipped_exclude.append(n)
                    continue
                if p.dtype != torch.bfloat16:
                    self.already_fp32.append(n)  # 已是 fp32 的门控，不需要 master
                    continue
                is_dim1 = p.dim() <= 1
                if is_dim1 and not self.include_dim1:
                    self.skipped_dim1.append(n)
                    continue
                if used + p.numel() > budget:
                    self.skipped_budget += p.numel()
                    self.skipped_budget_names.append(n)
                    continue
                master = torch.nn.Parameter(p.detach().float().clone(),
                                            requires_grad=True)
                chosen[n] = master
                used += p.numel()
                if is_dim1:
                    self.dim1_names.append(n)
        self.selected_numel = used
        self.pairs = [(n, params[n], m) for n, m in chosen.items()]

    def audit(self) -> dict:
        """SR 生效面覆盖审计：保护名单 / dim1 分组 / 各跳过原因名单。"""
        weight_names = [n for n, _, _ in self.pairs if n not in self.dim1_names]
        return {
            "enabled": self.enabled,
            "include_dim1": self.include_dim1,
            "n_selected": len(self.pairs),
            "selected_numel": self.selected_numel,
            "selected_names": [n for n, _, _ in self.pairs],
            "dim1_names": list(self.dim1_names),
            "weight_names": weight_names,
            "skipped_dim1": list(self.skipped_dim1),
            "skipped_exclude": list(self.skipped_exclude),
            "already_fp32": list(self.already_fp32),
            "skipped_budget_names": list(self.skipped_budget_names),
            "skipped_budget_numel": self.skipped_budget,
        }

    @staticmethod
    def _compile_matcher(pref: str) -> Union[str, "re.Pattern[str]"]:
        """`re:正则` 走 re.search；其余按模块名前缀 startswith 匹配。"""
        pref = str(pref)
        if pref.startswith("re:"):
            return re.compile(pref[3:])
        return pref

    @staticmethod
    def _match(name: str, matcher: Union[str, "re.Pattern[str]"]) -> bool:
        if isinstance(matcher, re.Pattern):
            return matcher.search(name) is not None
        return name.startswith(matcher)

    # -- 供 make_optimizers 使用：同名参数表里把选中的换成 master ----------
    def param_source(self, model: torch.nn.Module):
        by_name = {n: m for n, _, m in self.pairs}
        for n, p in model.named_parameters():
            yield n, by_name.get(n, p)

    @property
    def enabled(self) -> bool:
        return bool(self.pairs)

    def pull(self) -> None:
        """bf16 参数 -> master（前向之前若被外部改动，用它同步；每步梯度用 pull_grad）。"""
        with torch.no_grad():
            for _, src, master in self.pairs:
                master.copy_(src.detach().float())

    def pull_grad(self) -> None:
        """把 bf16 上的梯度拷到 master（裁剪之后调用，保证 clip 语义一致）。"""
        for _, src, master in self.pairs:
            if src.grad is None:
                master.grad = None
            else:
                master.grad = src.grad.detach().to(torch.float32)

    def push(self, generator: Optional[torch.Generator] = None) -> None:
        """master -> bf16，随机舍入。"""
        with torch.no_grad():
            for _, dst, master in self.pairs:
                dst.copy_(stochastic_round_to_bf16(master.detach(), generator))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, _, master in self.pairs:
            master.grad = None if set_to_none else torch.zeros_like(master)

    def summary(self) -> str:
        if not self.pairs:
            return "[sr-master] disabled"
        names = [n for n, _, _ in self.pairs]
        return (f"[sr-master] tensors={len(self.pairs)} dim1={len(self.dim1_names)} "
                f"params={self.selected_numel/1e6:.3f}M "
                f"fp32_extra={self.selected_numel*4/2**30:.3f}GB "
                f"skipped_by_budget={self.skipped_budget/1e6:.3f}M "
                f"first={names[:4]}")


# ---------------------------------------------------------------------------
# 3. 自检
# ---------------------------------------------------------------------------
def _selftest() -> int:
    print("== SR 单步无偏性 ==")
    torch.manual_seed(1)
    base = torch.full((1 << 14,), 0.0412, dtype=torch.float32)
    q = float(bf16_quantum(base)[0])
    print(f"bf16 量子 q={q:.3e}（0.0412 在 [2^-5,2^-4)、1+7 尾数 → 应为 2^-12=2.4414e-04）")
    for frac in (0.05, 0.3, 0.7):
        d = frac * q
        vals = [(base + d) for _ in range(50)]
        acc = torch.stack([stochastic_round_to_bf16(v).float() for v in vals]).mean(0)
        err = float((acc - (base + d)).abs().mean()) / q
        print(f"  frac={frac:.2f}q  SR 均值误差={err:.4f} q  "
              f"(RNE 会固定误差 {abs(round(frac)-frac):.4f} q)")

    print("== SR 累计位移（0.3q/步 × 200 步）==")
    print("  " + str(measure_sr_gain()))

    print("== SelectiveSRMaster（tiny net）==")
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patchify = torch.nn.Conv3d(4, 8, 1)
            self.dec_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(8, 8, bias=False) for _ in range(2)])
            self.head = torch.nn.Linear(8, 4)
        def forward(self, x):
            x = self.patchify(x)
            x = x.flatten(1)[:, :8]
            for b in self.dec_blocks:
                x = b(x)
            return self.head(x)

    m = Toy().to(torch.bfloat16)
    master = SelectiveSRMaster(m, prefixes=["patchify.", "dec_blocks.", "head."], budget_m=1.0)
    print("  " + master.summary())
    assert master.enabled and master.selected_numel > 0

    opt = torch.optim.AdamW([p for _, _, p in master.pairs], lr=1e-3)
    x = torch.randn(2, 4, 1, 2, 3, dtype=torch.bfloat16)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for _ in range(3):
        out = m(x).float().pow(2).mean()
        out.backward()
        master.pull_grad()
        opt.step()
        opt.zero_grad(set_to_none=True)
        master.push()
    changed = sum(1 for n, p in m.named_parameters()
                  if n in before and not torch.equal(p.detach(), before[n]))
    print(f"  3 步后变化张量数={changed}（master 覆盖 {len(master.pairs)} 个）")
    assert changed >= 1, "SR master 路径没有产生任何权重变化"

    print("== 未选中张量必须逐位不变 ==")
    m2 = Toy().to(torch.bfloat16)
    mt2 = SelectiveSRMaster(m2, prefixes=["patchify."], budget_m=1.0)
    ref = m2.head.weight.detach().clone()
    opt2 = torch.optim.AdamW([p for _, _, p in mt2.pairs], lr=1e-2)
    out = m2(x).float().pow(2).mean()
    out.backward()
    mt2.pull_grad(); opt2.step(); opt2.zero_grad(set_to_none=True); mt2.push()
    print(f"  head.weight 未选中，逐位一致={torch.equal(m2.head.weight.detach(), ref)}")
    assert torch.equal(m2.head.weight.detach(), ref)
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
