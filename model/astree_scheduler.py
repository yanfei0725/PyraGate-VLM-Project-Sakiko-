"""
a2 语义层（图2-5）—— ASTree 在环调度 + VLM 高层意图编译器
==================================================================

来源：用户 5 图之第 2/3/4/5 张（"代际领先"论证链的后半段）。

核心红线（图5，必须遵守）
--------------------------
VLM 调控 ASTree 有两种可能，代际含义完全不同：
  ✅ 正确：VLM 输出**高层语义意图**（"这段需要强音画同步" / "这段只需粗粒度语义协调"），
          ASTree 负责把意图**编译**成具体计算图改变。意图层可训练、稳定，VLM 不需要
          理解计算细节，只需理解内容。
  ❌ 错误：VLM 直接输出**低层调度指令**（"第7步 block 3-5 跨模态注意力预算设为0.3"）。
          这需要 VLM 理解主干内部计算，训练难度极高、信号维度爆炸 —— 这不是落地架构。

本模块严格走第一条：VLM 的出口是 SemanticIntent（内容语义层），
ASTree 是"意图 → 计算图"的编译器，二者职责分离、可分别训练验证。

图2 三方向（全部覆盖）
----------------------
1. 跨模态因果推理：intent 里的 drive_direction（A→B / B→A / 双向）表达因果方向，
   H3 的对称 full attention 结构上无法表达。
2. 自适应模态路由：intent 里的 interaction_depth 按语义不确定度动态决定
   "这一步 A/B 交互多深"（压缩瓶颈 vs 全连接），H3 的固定瓶颈做不到。
3. VLM 在线规划：VLM 每步读 latent 状态 → 输出高层规划意图 → ASTree 编译，
   H3 里 VLM 是冻结的文本编码器，不参与去噪循环。

图3：ASTree 在环
----------------
在去噪循环内按当前 latent / 语义不确定度 / 模态对齐置信度，动态决定：
  * 哪些模态在哪些时间步以什么粒度交互（compressed_bottleneck / full_joint / none）
  * 哪些 block 激活 / 跳过
  * 跨模态注意力的预算如何分配
这回答 H3 结构上无法回答的问题："计算图本身应该是内容相关的"。

图4：三座硬问题（本模块给出可验收的接口/机制）
------------------------------------------------
  山1 VLM 延迟预算：vlm_latency_budget_ms() 给出主干单步耗时约束下的 VLM 延迟上限；
     VLM 每 K 步调用一次（plan interval），K 可配，避免逐步调用成瓶颈。
  山2 信号稳定可训练：意图经 EMA 低通 + 置信度门控（TemporalSmoother），
     防止抖动导致计算图频繁重构；训练时把"调度质量"纳入 loss 钩子。
  山3 可验证任务：evaluate_segmented() 实现"分段交互深度"评测框架 ——
     同一段生成分三段（细节对齐 / 粗粒度协调 / 因果驱动），
     对照 H3 固定策略，指标=同步精度/语义一致性/计算效率。

纯 torch，CPU 可跑，不占显存，不碰运行中训练。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 意图层（VLM 的唯一出口）—— 内容语义层，不含任何块号/预算数字
# ---------------------------------------------------------------------------
class InteractionGranularity(str, Enum):
    """交互粒度：内容语义描述，不是计算指令。"""
    FINE = "fine"          # 细粒度（音画逐帧对齐：口型/鼓点）
    COARSE = "coarse"      # 粗粒度（语义协调即可）
    CAUSAL = "causal"      # 因果驱动（A 事件触发 B 变化）
    NONE = "none"          # 本段无需跨模态


@dataclass
class SemanticIntent:
    """VLM 输出的高层语义意图（图5 红线：只有内容语义，没有计算指令）。

    字段都是内容层面的（"需要多细"、"什么方向"、"有多不确定"），
    ASTree 负责把它们翻译成块调度。禁止出现 block_id / budget=0.3 这类低层字段。
    """
    granularity: InteractionGranularity = InteractionGranularity.COARSE
    drive_direction: str = "A_to_B"       # 因果方向：A_to_B / B_to_A / bidirectional
    alignment_confidence: float = 1.0     # 模态对齐置信度 [0,1]
    semantic_uncertainty: float = 0.0     # 语义不确定度 [0,1]

    def to_vector(self) -> torch.Tensor:
        """把意图编码成 4 维向量（供编译器/稳定性模块用），仍不含低层指令。

        2026-11-24 BUG-7：VLM 是外部产出，任何字段拼写/类型偏差都不该崩去噪循环。
        用显式校验 + 带字段名的 ValueError，把 trust boundary 前移到意图层。
        """
        # granularity：接受 Enum 或其 value 字符串
        gv = self.granularity.value if isinstance(self.granularity, InteractionGranularity) \
             else str(self.granularity).lower()
        g_map = {"fine": 1.0, "coarse": 0.5, "causal": 0.75, "none": 0.0}
        if gv not in g_map:
            raise ValueError(f"SemanticIntent.granularity 非法值 {gv!r}，"
                             f"允许 {list(g_map)}")
        d_map = {"A_to_B": 1.0, "B_to_A": -1.0, "bidirectional": 0.0}
        if self.drive_direction not in d_map:
            raise ValueError(f"SemanticIntent.drive_direction 非法值 {self.drive_direction!r}，"
                             f"允许 {list(d_map)}")
        # 数值字段：强制 float 并夹到 [0,1]，非法字符串给带字段名的错误
        try:
            conf = float(self.alignment_confidence)
            unc = float(self.semantic_uncertainty)
        except (TypeError, ValueError) as e:
            raise ValueError(f"SemanticIntent 数值字段非法: confidence={self.alignment_confidence!r} "
                             f"uncertainty={self.semantic_uncertainty!r}") from e
        return torch.tensor([g_map[gv], d_map[self.drive_direction],
                             max(0.0, min(1.0, conf)), max(0.0, min(1.0, unc))])


# ---------------------------------------------------------------------------
# 稳定性（图4 山2）：EMA 低通 + 置信度门控，防止抖动重构计算图
# ---------------------------------------------------------------------------
class TemporalSmoother(nn.Module):
    """对意图向量做时间低通 + 置信度门控。

    低通：ema = (1-a)*ema + a*new，a 越小越平滑（默认 0.2）。
    置信度门控：对齐置信度低时，缩小意图跳变幅度（少折腾计算图）。
    """
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = float(alpha)
        self.register_buffer("ema", None)

    def forward(self, intent_vec: torch.Tensor) -> torch.Tensor:
        if self.ema is None or self.ema.shape != intent_vec.shape:
            self.ema = intent_vec.detach().clone()
            return intent_vec
        conf = intent_vec[:, 2:3]                      # 对齐置信度
        gate = torch.clamp(conf, 0.0, 1.0)             # 置信低→少变
        new_ema = (1 - self.alpha) * self.ema + self.alpha * intent_vec
        out = self.ema + gate * (new_ema - self.ema)
        # 2026-11-24 BUG-2 修复：原来存未门控的 new_ema，低置信跳变时输出
        # 被保护(0.796)但内部 ema 已被污染(0.4)，再喂回稳定值恢复不了。
        # 改为存门控后的 out —— 状态与输出一致，跳变不再永久带偏基准。
        self.ema = out.detach()
        return out


# ---------------------------------------------------------------------------
# ASTree 节点：计算图的一个可调度决策
# ---------------------------------------------------------------------------
@dataclass
class ScheduleNode:
    block_range: Tuple[int, int]        # 作用 block 区间（编译器产出，非 VLM 输出）
    modality: str                       # "cross" / "A" / "B"
    granularity: str                    # compressed_bottleneck / full_joint / none
    active: bool = True


@dataclass
class ComputeSchedule:
    """一帧去噪的完整计算图调度（ASTree 的产出）。"""
    step: int
    nodes: List[ScheduleNode] = field(default_factory=list)
    crossmodal_budget: float = 1.0      # 跨模态注意力预算 [0,1]（编译器算，非 VLM 给）
    skip_blocks: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (f"step{self.step}: cross={self.crossmodal_budget:.2f} "
                f"nodes={len(self.nodes)} skip={self.skip_blocks}")


# ---------------------------------------------------------------------------
# ASTree：意图 → 计算图 的编译器（图3 核心）
# ---------------------------------------------------------------------------
class ASTreeScheduler(nn.Module):
    """内容驱动的动态计算图调度器。

    输入：SemanticIntent（VLM 高层意图）+ 当前步号 + 总步数
    输出：ComputeSchedule（具体块调度）—— 这是 ASTree 的职责，VLM 不碰。

    设计要点（图3）：
      * 按步分段：早期细粒度对齐 / 中期粗粒度协调 / 后期因果驱动（可验证任务的分段）
      * 按内容调粒度：intent.granularity 决定用 compressed_bottleneck 还是 full_joint
      * 按不确定度调预算：semantic_uncertainty 高 → 加大跨模态预算
      * 按对齐置信度调 skip：置信度高且粗粒度 → 跳过后段部分 block（省算力）
    """
    def __init__(self, num_blocks: int = 30, total_steps: int = 20,
                 smoother_alpha: float = 0.2, skip_enabled: bool = True,
                 backbone_step_ms: float = 2500.0):
        super().__init__()
        self.num_blocks = num_blocks
        self.total_steps = total_steps
        self.smoother = TemporalSmoother(smoother_alpha)
        self.skip_enabled = skip_enabled
        # 隐患D：本项目口径——704×480/990 token，主干单步约 2500ms（训练实测）。
        # 不再用旧 README 的 240ms（那是 T=16/64×64 小图口径，会误导延迟预算）。
        self.backbone_step_ms = float(backbone_step_ms)

    def vlm_latency_budget_ms(self, backbone_step_ms: Optional[float] = None,
                              max_fraction: float = 0.05,
                              plan_interval: int = 4) -> float:
        """图4 山1：VLM 延迟预算。

        隐患D 修复：默认值不再是"T=16/64×64 README 的 240ms"（与本项目 704×480/
        990 token 口径不符）。默认改用 PROJECT_BACKBONE_STEP_MS（本项目主干单步
        实测约 2500ms），调用方也可显式传 backbone_step_ms 覆盖。
        VLM 每 plan_interval 步调一次，延迟须 << 主干单步；预算=单步×占比×间隔。
        超预算则加大 plan_interval（降调用频率）。
        """
        if backbone_step_ms is None:
            backbone_step_ms = self.backbone_step_ms
        return float(backbone_step_ms) * max_fraction * plan_interval

    def compile(self, intent: SemanticIntent, step: int,
                smoothed: bool = True) -> ComputeSchedule:
        """把高层意图编译成计算图调度（ASTree 的核心职责）。"""
        vec = intent.to_vector()
        if smoothed:
            vec = self.smoother(vec.unsqueeze(0)).squeeze(0)
        g, drive, conf, unc = vec.tolist()

        # 预算：不确定度越高、粒度越细 → 跨模态预算越大（内容驱动，不是固定策略）
        budget = float(torch.clamp(torch.tensor(0.3 + 0.5 * g + 0.4 * unc), 0.0, 1.0))
        # 粒度：fine→full_joint（保细节），coarse→compressed_bottleneck（省算力），
        #        causal→compressed + A 驱动特权
        gran = ("full_joint" if intent.granularity == InteractionGranularity.FINE
                else "compressed_bottleneck" if intent.granularity == InteractionGranularity.COARSE
                else "causal_driven" if intent.granularity == InteractionGranularity.CAUSAL
                else "none")
        active = intent.granularity != InteractionGranularity.NONE

        n = self.num_blocks
        nodes = [
            ScheduleNode((0, n // 3), "cross", gran, active),
            ScheduleNode((n // 3, 2 * n // 3), "cross", gran, active),
            ScheduleNode((2 * n // 3, n), "cross", gran, active),
        ]
        # 因果方向：A_to_B 时把 A 特权节点插入（结构支撑因果推理）
        if drive > 0.5:
            nodes.append(ScheduleNode((0, n), "A", "privileged", True))
        elif drive < -0.5:
            nodes.append(ScheduleNode((0, n), "B", "privileged", True))

        # 隐患C：统一「哪些块激活」的两种表达——inactive 节点的 block_range
        # 自动并入 skip_blocks（下游只需消费一个字段）。
        skip: List[int] = []
        for nd in nodes:
            if not nd.active:
                skip.extend(range(nd.block_range[0], nd.block_range[1]))
        # skip：置信度高 + 粗粒度 + 预算低 → 跳过后段部分 block（内容决定跳过）
        if self.skip_enabled and conf > 0.8 and budget < 0.5 and active:
            skip.extend(list(range(int(n * 0.7), n, 3)))
        skip = sorted(set(int(i) for i in skip))

        return ComputeSchedule(step=step, nodes=nodes,
                               crossmodal_budget=budget, skip_blocks=skip)

    def plan_segment(self, segment: str) -> SemanticIntent:
        """可验证任务的分段意图（图4 山3）：
        前 1/3 细节对齐 / 中 1/3 粗粒度协调 / 后 1/3 因果驱动。
        """
        if segment == "detail":
            return SemanticIntent(InteractionGranularity.FINE, "bidirectional", 0.9, 0.1)
        if segment == "coordination":
            return SemanticIntent(InteractionGranularity.COARSE, "bidirectional", 0.95, 0.05)
        if segment == "causal":
            return SemanticIntent(InteractionGranularity.CAUSAL, "A_to_B", 0.85, 0.3)
        raise ValueError(f"未知分段: {segment}")


# ---------------------------------------------------------------------------
# 可验证评测框架（图4 山3）：分段策略 vs H3 固定策略
# ---------------------------------------------------------------------------
def evaluate_segmented(scheduler: ASTreeScheduler, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """对照实验框架：ASTree 内容驱动分段 vs H3 固定 full attention。

    metrics 是外部实测填入的指标字典（同步精度 / 语义一致性 / 计算效率），
    本函数负责组织分段调度并给出对照结构，不伪造数值。
    """
    segments = ["detail", "coordination", "causal"]
    plans = []
    for i, seg in enumerate(segments):
        intent = scheduler.plan_segment(seg)
        sched = scheduler.compile(intent, step=i, smoothed=False)
        plans.append({"segment": seg, "intent": intent.granularity.value,
                      "schedule": sched.summary()})
    return {
        "astree_plans": plans,
        "h3_baseline": "fixed full attention, 所有段同一策略（无内容自适应）",
        "metrics": metrics or {"note": "需外部实测填入: sync_acc / sem_consistency / compute_eff"},
        "verdict": "需填入实测指标后，按三指标分段评测是否优于 H3 固定策略",
    }


def _selfcheck():
    torch.manual_seed(0)
    sch = ASTreeScheduler(num_blocks=30, total_steps=20)

    # 山1：VLM 延迟预算（主干 240ms，占比 5%，每 4 步一次 → 48ms 上限）
    budget = sch.vlm_latency_budget_ms(240.0, 0.05, 4)
    assert abs(budget - 48.0) < 1e-3

    # 图5 红线：SemanticIntent 不含任何低层字段
    it = SemanticIntent(InteractionGranularity.FINE, "A_to_B", 0.9, 0.3)
    assert not any(k in it.__dict__ for k in ("block_id", "budget", "layer", "head"))

    # 编译：意图 → 具体调度（粒度/预算/因果节点/skip 都在 ASTree 侧产出）
    sched = sch.compile(it, step=3, smoothed=False)
    assert sched.nodes and sched.crossmodal_budget > 0
    assert any(n.modality == "A" and n.granularity == "privileged" for n in sched.nodes)

    # 稳定性：抖动输入被低通削弱
    sm = TemporalSmoother(alpha=0.2)
    v1 = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    sm(v1); out = sm(v2)
    assert 0.0 < out[0, 0].item() < 1.0, "低通未生效"

    # 分段评测框架
    ev = evaluate_segmented(sch)
    assert len(ev["astree_plans"]) == 3

    return {"vlm_latency_budget_ms": budget, "redline_ok": True,
            "compile_ok": sched.summary(), "smoother_ok": True,
            "segments": [p["segment"] for p in ev["astree_plans"]]}


if __name__ == "__main__":
    info = _selfcheck()
    print("[astree_scheduler selfcheck]")
    for k, v in info.items():
        print(f"  {k}: {v}")
