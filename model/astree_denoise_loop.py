"""
a2 语义层（图3-5）—— ASTree 在环去噪循环 + VLM→VLM→FFN 调用链
==================================================================

来源：5 图之第 3/4/5 张。"ASTree 在环"= 在**去噪循环内**按内容动态改计算图，
而不是把调度塞进单次 forward。本文件提供包装器，驱动任意带 block 列表的主干。

在环语义（图3）
---------------
每步去噪：
  1. VLM（协核）读当前 latent 状态 → 输出高层语义意图（SemanticIntent，只内容不计算）
  2. 意图过 TemporalSmoother（山2：稳定，防抖动重构）
  3. ASTree.compile() → ComputeSchedule（跳过哪些 block / 跨模态预算多少）
  4. SemanticFFNController：把意图二次编译成逐 block 的 FFN 调制（VLM→VLM→FFN）
  5. 按调度执行主干前向（skip 指定 block + FFN 调制），产出速度场 v
  6. Flow scheduler 用 v 更新 x（x ← x + (σ_next-σ)·v），绝不把 v 当 x 喂回（BUG 流匹配）

VLM→VLM→FFN 调用链
------------------------------------------------
  第一层 VLM：理解内容 → SemanticIntent（高层语义，红线：不含块号/预算）
  第二层"VLM"（语义控制头）：SemanticFFNController 读意图 + 步号 → 生成逐 block
             的 FFN 调制向量 [B,n_blocks,dim]，经零初始化门控起步。
  主干 FFN：UViTBlock.forward(ffn_modulation=...) 按 (1+mod) 缩放 FFN 输出。
  → VLM 的语义判断真正落到主干 FFN，而不只是记在 ASTree 的账本上。
  红线保持：第二层消费的仍是「高层语义意图」，不是低层块号/预算数字。

2026-11-24 修复
----------------
  BUG-3：_call_backbone 不再 except TypeError 吞异常 + 二次调用；用 inspect.signature 一次判定。
  BUG-5：vlm_call_interval 真正落地——每 K 步才重新采意图/编译，其余步复用缓存。
  BUG-12：主干传完整 (x, t_emb, text_emb, ...) 参数，不再只传 x。
  BUG-13：crossmodal_budget 写入主干可写属性（主干 forward 暴露 self.crossmodal_budget）。
  流匹配：run_segmented_trial 用 EulerScheduler 更新 x，不把速度场 v 直接喂回。
  三座山：VLM 延迟 / 调度稳定性 / 可验证任务，验收钩子全部保留。

纯 torch，CPU 可跑，不加载真实 4B 权重，不碰训练。
"""
from __future__ import annotations

import inspect
from typing import Optional, Callable, Dict, Any, List, Tuple

import torch
import torch.nn as nn

from astree_scheduler import (ASTreeScheduler, SemanticIntent,
                              InteractionGranularity, ComputeSchedule)


# ---------------------------------------------------------------------------
# 第二层"VLM"：语义→FFN 控制头（VLM→VLM→FFN 调用链的中枢）
# ---------------------------------------------------------------------------
class SemanticFFNController(nn.Module):
    """把 VLM 高层语义意图二次编译成逐 block 的 FFN 调制。

    输入：SemanticIntent.to_vector() [4] + 归一化步号 + 调度预算
    输出：ffn_modulation [B, n_blocks, dim]（经零门控起步）

    设计：
      - 一个小的 MLP（语义→每块一个标量强度），再广播到 dim 维；
      - 独立 nn.Parameter gate 零初始化 → 起步严格恒等（VLM 链路可插拔）；
      - 只吃高层语义（granularity/drive/conf/unc）与预算，不碰块号指令（红线）。
    """
    def __init__(self, dim: int, n_blocks: int, hidden: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.n_blocks = int(n_blocks)
        # 输入 4(意图向量) + 1(步号进度) + 1(预算) = 6
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, self.n_blocks),
        )
        # 零初始化门控：起步 VLM→FFN 链路完全不影响主干
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, intent: SemanticIntent, step: int, total_steps: int,
                budget: float, batch: int = 1) -> torch.Tensor:
        vec = intent.to_vector().float()                      # [4]
        prog = torch.tensor([step / max(1, total_steps)], dtype=vec.dtype)
        bud = torch.tensor([float(budget)], dtype=vec.dtype)
        feat = torch.cat([vec, prog, bud]).unsqueeze(0)        # [1,6]
        per_block = torch.tanh(self.net(feat))                 # [1,n_blocks] ∈ (-1,1)
        g = torch.tanh(self.gate)
        # [1,n_blocks] → transpose [n_blocks,1] → unsqueeze [1,n_blocks,1] → [B,n_blocks,dim]
        mod = (g * per_block).transpose(0, 1).unsqueeze(0).expand(batch, -1, self.dim)
        return mod


# ---------------------------------------------------------------------------
# 流匹配 scheduler（图3 在环正确性）：用速度场 v 更新 x
# ---------------------------------------------------------------------------
class EulerFlowScheduler:
    """流匹配欧拉更新：x ← x + (sigma_next - sigma) · v。

    真实主干返回速度场 v（或 (v, aux[, motion])），绝不能把 v 当 latent 喂回。
    sigma 从 1→0 线性排布，与训练口径 v=noise-x0 一致。
    """
    def __init__(self, total_steps: int = 20):
        self.total_steps = int(total_steps)

    def sigma(self, step: int) -> float:
        return 1.0 - step / max(1, self.total_steps - 1)

    def update(self, x: torch.Tensor, v: torch.Tensor, step: int) -> torch.Tensor:
        s_now = self.sigma(step)
        s_next = self.sigma(step + 1)
        return x + (s_next - s_now) * v

    @staticmethod
    def extract_velocity(out: Any) -> torch.Tensor:
        """从主干返回值里取速度场：v / (v,aux) / (v,aux,motion)。"""
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


# ---------------------------------------------------------------------------
# ASTree 在环去噪循环包装器
# ---------------------------------------------------------------------------
class ASTreeDenoiseLoop:
    """ASTree 在环去噪循环包装器。

    主干需提供 forward(x, t_emb, text_emb, ...)，可选支持 skip_blocks /
    ffn_modulation / crossmodal_budget。本包装器用 inspect.signature 一次判定支持项
    （BUG-3：不再靠异常重试），并传入完整参数（BUG-12）。
    """

    def __init__(self, backbone, scheduler: Optional[ASTreeScheduler] = None,
                 vlm_call_interval: int = 4, backbone_step_ms: float = 240.0,
                 ffn_controller: Optional[SemanticFFNController] = None):
        self.backbone = backbone
        self.scheduler = scheduler or ASTreeScheduler()
        self.vlm_call_interval = max(1, int(vlm_call_interval))
        self.backbone_step_ms = float(backbone_step_ms)
        self.flow = EulerFlowScheduler(self.scheduler.total_steps)
        self.ffn_controller = ffn_controller
        self._intent_history: List[SemanticIntent] = []
        self._schedule_history: List[ComputeSchedule] = []
        self._current_intent: SemanticIntent = SemanticIntent()
        # BUG-5：缓存「上次 VLM 编译结果」，每 K 步才刷新
        self._cached_schedule: Optional[ComputeSchedule] = None
        self._cached_ffn_mod: Optional[torch.Tensor] = None
        # 主干参数（由调用方通过 configure_backbone_inputs 提供）
        self._t_emb = None
        self._text_emb = None
        self._extra_kwargs: Dict[str, Any] = {}
        # 签名能力探测（BUG-3：只在初始化探测一次）
        self._backbone_supports = self._probe_backbone()

    # ---- 主干接口契约 ----
    def configure_backbone_inputs(self, t_emb: torch.Tensor, text_emb: torch.Tensor,
                                  **extra_kwargs):
        """提供主干必填条件（t_emb/text_emb）与可选参数（vlm_hidden/sigma 等）。"""
        self._t_emb = t_emb
        self._text_emb = text_emb
        self._extra_kwargs = dict(extra_kwargs)

    def _probe_backbone(self) -> Dict[str, bool]:
        """一次探测主干 forward 支持哪些 a2 契约参数（不吞异常、不二次调用主干）。"""
        fn = getattr(self.backbone, "forward", self.backbone)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            # DummyBackbone 用 __call__(x, skip_blocks=None)
            try:
                params = inspect.signature(self.backbone.__call__).parameters
            except (TypeError, ValueError):
                params = {}
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return {
            "skip_blocks": ("skip_blocks" in params) or has_kwargs,
            "ffn_modulation": ("ffn_modulation" in params) or has_kwargs,
            "crossmodal_budget": ("crossmodal_budget" in params) or has_kwargs,
            "t_emb": "t_emb" in params,
            "text_emb": "text_emb" in params,
        }

    def set_intent(self, intent: SemanticIntent):
        """VLM 每 K 步调用：只下发高层语义意图（红线：类型系统挡住低层指令）。"""
        if not isinstance(intent, SemanticIntent):
            raise TypeError("VLM 只能下发 SemanticIntent（高层语义意图），"
                            "不能直接下发计算指令（block/budget）——这是 a2 红线")
        self._current_intent = intent
        self._intent_history.append(intent)

    def _call_backbone(self, x, schedule, ffn_mod):
        """按调度执行主干前向：传完整参数，能力探测决定哪些契约参数可传。"""
        kwargs = dict(self._extra_kwargs)
        if self._t_emb is not None and self._backbone_supports.get("t_emb", False):
            kwargs["t_emb"] = self._t_emb
        if self._text_emb is not None and self._backbone_supports.get("text_emb", False):
            kwargs["text_emb"] = self._text_emb
        if self._backbone_supports.get("skip_blocks", False):
            kwargs["skip_blocks"] = schedule.skip_blocks
        if self._backbone_supports.get("ffn_modulation", False) and ffn_mod is not None:
            kwargs["ffn_modulation"] = ffn_mod
        if self._backbone_supports.get("crossmodal_budget", False):
            kwargs["crossmodal_budget"] = schedule.crossmodal_budget
        return self.backbone(x, **kwargs)

    def _need_replan(self, step_idx: int, forced: bool) -> bool:
        """BUG-5：每 vlm_call_interval 步才重编译；首步 / 强制 / 无缓存时必须重编。"""
        if forced or self._cached_schedule is None:
            return True
        return (step_idx % self.vlm_call_interval) == 0

    def step(self, x: torch.Tensor, step_idx: int, *,
             vlm_intent: Optional[SemanticIntent] = None,
             force_replan: bool = False,
             advance_flow: bool = True) -> torch.Tensor:
        """执行一步去噪：意图→编译→FFN调制→主干前向→流匹配更新。"""
        if vlm_intent is not None:
            self.set_intent(vlm_intent)

        # VLM 调用频率落地（BUG-5）：非计划步复用缓存调度 + 缓存 FFN 调制
        if self._need_replan(step_idx, force_replan):
            schedule = self.scheduler.compile(self._current_intent, step=step_idx)
            self._cached_schedule = schedule
            if self.ffn_controller is not None:
                self._cached_ffn_mod = self.ffn_controller(
                    self._current_intent, step_idx, self.scheduler.total_steps,
                    schedule.crossmodal_budget, batch=x.shape[0])
            self._schedule_history.append(schedule)
        else:
            schedule = self._cached_schedule
            if self.ffn_controller is not None and self._cached_ffn_mod is None:
                self._cached_ffn_mod = self.ffn_controller(
                    self._current_intent, step_idx, self.scheduler.total_steps,
                    schedule.crossmodal_budget, batch=x.shape[0])

        # 跨模态预算下发（BUG-13）：双重保险——主干属性 + 遍历子模块
        self._apply_budget(schedule.crossmodal_budget)

        out = self._call_backbone(x, schedule, self._cached_ffn_mod)
        v = self.flow.extract_velocity(out)
        if advance_flow:
            return self.flow.update(x, v, step_idx)
        return out

    def _apply_budget(self, budget: float):
        """把跨模态预算下发到主干及带 crossmodal_budget 属性的子模块。"""
        if hasattr(self.backbone, "crossmodal_budget"):
            try:
                self.backbone.crossmodal_budget = float(budget)
            except Exception:
                pass
        for m in getattr(self.backbone, "modules", lambda: [])():
            if hasattr(m, "crossmodal_budget"):
                try:
                    m.crossmodal_budget = float(budget)
                except Exception:
                    pass

    # ---- 三座山验收钩子 ----
    def schedule_jitter(self) -> float:
        """山2：调度抖动 = 相邻步跨模态预算变化的均值（越小越稳）。"""
        if len(self._schedule_history) < 2:
            return 0.0
        bs = [s.crossmodal_budget for s in self._schedule_history]
        diffs = [abs(bs[i] - bs[i - 1]) for i in range(1, len(bs))]
        return sum(diffs) / len(diffs)

    def plan_latency_report(self) -> Dict[str, Any]:
        """山1：核对 VLM 延迟预算与调用频率。"""
        budget = self.scheduler.vlm_latency_budget_ms(
            backbone_step_ms=self.backbone_step_ms,
            max_fraction=0.05, plan_interval=self.vlm_call_interval)
        return {
            "vlm_call_interval": self.vlm_call_interval,
            "backbone_step_ms": self.backbone_step_ms,
            "vlm_latency_budget_ms": budget,
            "verdict": ("OK: VLM 延迟需 < %.1fms（每 %d 步一次）"
                        % (budget, self.vlm_call_interval)),
            "note": "需实测 1B Qwen3-VL W4A8 前向延迟填入对比；超预算则加大 interval",
        }

    def run_segmented_trial(self, x_init: torch.Tensor, steps_per_segment: int = 3):
        """山3：可验证任务 —— 分三段（细节/协调/因果）跑对照。

        修正流匹配语义：每段内用 flow scheduler 更新 x，不把速度场 v 当 latent 喂回。
        """
        segments = ["detail", "coordination", "causal"]
        x = x_init
        trace = []
        global_step = 0
        for seg in segments:
            intent = self.scheduler.plan_segment(seg)
            self.set_intent(intent)
            for _ in range(steps_per_segment):
                x = self.step(x, global_step, force_replan=True)
                global_step += 1
            trace.append({"segment": seg, "intent": intent.granularity.value,
                          "budget": self._schedule_history[-1].crossmodal_budget})
        return {"trace": trace, "jitter": self.schedule_jitter(),
                "latency": self.plan_latency_report(),
                "x_final_shape": list(x.shape),
                "metrics_note": "填入实测指标后对照 H3 固定 full attention"}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
class _DummyBackbone(nn.Module):
    """假主干：返回速度场（恒等 v=x），记录调用参数。"""
    def __init__(self):
        super().__init__()
        self.calls = []
        self.crossmodal_budget = None

    def forward(self, x, skip_blocks=None, ffn_modulation=None, crossmodal_budget=None):
        self.calls.append({"skip": list(skip_blocks or []),
                           "ffn": None if ffn_modulation is None else tuple(ffn_modulation.shape),
                           "budget": crossmodal_budget})
        return x   # v = x（恒等速度场）


def _selfcheck():
    dummy = _DummyBackbone()
    sch = ASTreeScheduler(num_blocks=6, total_steps=20)
    ffn = SemanticFFNController(dim=16, n_blocks=6)
    loop = ASTreeDenoiseLoop(dummy, sch, vlm_call_interval=3, ffn_controller=ffn)
    loop.configure_backbone_inputs(t_emb=None, text_emb=None)  # dummy 不需要
    x = torch.randn(1, 16)

    # 红线：VLM 不能下发低层指令
    try:
        loop.set_intent({"block": 3, "budget": 0.3})   # type: ignore
        raise AssertionError("红线失效：接受了低层指令")
    except TypeError:
        pass

    # 在环跑一步（FINE 意图 → full_joint 调度）
    loop.set_intent(SemanticIntent(InteractionGranularity.FINE, "bidirectional", 0.9, 0.2))
    out = loop.step(x, 0, force_replan=True)
    assert out.shape == x.shape
    assert len(loop._schedule_history) == 1
    assert dummy.calls[-1]["ffn"] == (1, 6, 16), f"FFN 调制形状错: {dummy.calls[-1]}"
    assert dummy.calls[-1]["budget"] is not None, "预算未下发"

    # BUG-5：vlm_call_interval=3，step 1/2 应复用缓存（不新增 schedule）
    loop.step(x, 1); loop.step(x, 2)
    assert len(loop._schedule_history) == 1, f"interval 未生效: {len(loop._schedule_history)}"
    loop.step(x, 3)
    assert len(loop._schedule_history) == 2, "第 3 步应重编译"

    # 流匹配：x 被 scheduler 更新（不是原样返回 v）
    x0 = torch.ones(1, 16) * 0.5
    out = loop.step(x0, 4, vlm_intent=SemanticIntent(InteractionGranularity.COARSE, "A_to_B", 0.9, 0.1))
    # dummy v=x，sigma 递减 → x_new = x + (neg)*x < x
    assert not torch.equal(out, x0), "流匹配未更新 x（把 v 当 x 了？）"

    # 分段任务
    trial = loop.run_segmented_trial(x, steps_per_segment=2)
    assert len(trial["trace"]) == 3
    assert trial["jitter"] >= 0.0

    return {"redline_ok": True, "in_loop_ok": True,
            "vlm_to_ffn_shape": list(dummy.calls[0]["ffn"]),
            "vlm_interval_ok": True, "flow_match_ok": True,
            "segments": [t["segment"] for t in trial["trace"]],
            "jitter": round(trial["jitter"], 4),
            "latency_budget_ms": trial["latency"]["vlm_latency_budget_ms"]}


if __name__ == "__main__":
    info = _selfcheck()
    print("[astree_denoise_loop selfcheck]")
    for k, v in info.items():
        print(f"  {k}: {v}")
