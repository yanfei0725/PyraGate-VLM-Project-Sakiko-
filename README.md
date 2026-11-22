# PyraGate-VLM · 3D U-ViT Backbone（Udit）

> English overview: [docs/README_EN.md](docs/README_EN.md) · [模型卡 MODEL_CARD](docs/MODEL_CARD.md)（含结构图与教师参考管线示例图）

纯 PyTorch 实现的视频流匹配 DiT 主干骨架。无 diffusers / flash-attn 硬依赖，可在纯 CPU 上实例化与自检；GPU 前向与训练管线已在 AMD Radeon RX 7900 XT（ROCm 7.2.1，Windows）上实测。

## 模型概览

- **规模**：4.0B（meta device 精确统计 3,999,944,893）。对称 U 型：encoder 12 / bottleneck 6 / decoder 12（全局序号 30 块），dim=3072、24 heads（head_dim 128）。
- **条件注入**：教师式 AdaLN——共享两层 embedder + 每块 (1,6,dim) 零初始化小表（替代逐块 Linear(dim,6dim)，省 ~1.45B 参数 / ~2.9GB 权重）。
- **注意力**：2+1D 时空分解（空间/时序交替），瓶颈层 full 3D joint；可选 GSA 门控混合路线（`attn_strategy="gsa"`）；3D RoPE 按 t/h/w 三段分频，实数 fp32 共享表（batch 广播、无逐样本循环）。
- **FFN**：Dual-Act 分段激活 SwiGLU（`|z|<t` 用 SiLU、`|z|≥t` 平滑压缩，可学习阈值；`mlp_ratio=8/3` 与原生 FFN 参数持平）。w1/w2 合并为单 GEMM，q/k/v 合并为单 GEMM。
- **输出头**：速度场主头 + 时序辅助头 + DDT 式高频细节残差头（均零门控起步，初始严格恒等）。
- **条件扩展**：VLM 北大桥（4096→3072，支持 strict 加载预训练权重）+ 低秩 per-token CrossAttn 注入链；分层风格条件编码器（72M）与参考帧重采样对齐编码器（68.5M）。
- **MoE（阶段 2 可选）**：全局条件空间三专家 `GlobalConditionalMoE`（单专家 330M，合计 0.991B），Top1 硬路由（straight-through 梯度）+ α 门控加法注入 blocks 7–12；α=0 时逐 bit 恒等，主干可严格回退。**不是**按块 FFN 替件形态。
- **蒸馏设定**：流匹配速度场 MSE 为主目标；教师为 20 步采样管线（时序锚点 flicker 0.0244 / identity 0.9892）；配套蒸馏 loss（速度场 + 层特征对齐 + 时序一致性 + flicker 正则）见 `distill_loss.py`。

## 文件

| 文件 | 内容 |
|---|---|
| `uvit3d.py` | 模型骨架全部结构（AdaLN / 2+1D / GSA / SwiGLU / GatedSkip / 金字塔 / 输出头 / 全局 MoE / RoPE 实数路径） |
| `vlm_bridge.py` | VLM 接入：北大桥（4096→3072）+ 多尺度分层注入器（全局池化 → AdaLN；per-token → CrossAttn） |
| `cond_encoders.py` | 分层风格条件编码器 + 参考帧重采样对齐编码器（低秩 + 减层方案） |
| `attn_backend.py` | 注意力后端适配：flash_attn 包（Triton 后端）→ SDPA FLASH → SDPA → CPU math，逐级诚实降级 |
| `distill_loss.py` | 蒸馏损失四项（速度场 MSE / 层特征对齐 / 时序一致性 / flicker 正则），可反传 |
| `quant_plan.py` | 混合量化分层清单（敏感层 fp16、普通层 int8）与红线校验，只产出计划不执行量化 |
| `ablation_matrix.py` | GSA vs 2+1D 消融框架（配置矩阵 + 指标容器） |
| `engineering_hygiene.py` | 工程卫生包（等价性单测） |
| `selftest_uvit3d.py` | 纯 CPU 自检（139 项，含逐 bit 回归 / 数值等价 / 参数量硬断言） |
| `train_backbone.py` | 训练循环（Adafactor + bf16 + 梯度检查点 + 教师预编码目标模式 + OOM/周期重启退出码契约） |
| `run_training_daemon.py` | 训练守护进程（崩溃自动重启、OOM 降级阶梯、周期重启清显存碎片） |
| `preencode_latent.py` | 视频 → VAE latent 离线预编码 |

## 快速开始（CPU 自检）

```bash
cd model
python selftest_uvit3d.py       # 主自检（默认强制 CPU）
python distill_loss.py          # 蒸馏 loss 自检
python quant_plan.py            # 量化清单校验
python ablation_matrix.py       # 消融矩阵自检
python engineering_hygiene.py   # 工程卫生自检
```

期望全部 `[PASS]`（合计 139+ 项）。

## 平台注意事项（ROCm / Windows 实测）

- **注意力后端**：`attn_backend="flash2"` 默认。本机安装社区 Triton 后端 wheel（`flash_attn 2.8.4 win-rocm` + `amd_aiter`，纯 Triton kernel 无需 MSVC）后自动走 `fa2_package`；未安装时降级 torch SDPA FLASH（2048 序列实测 3.45× vs math）；CPU/fp32 明确降级 math，不伪装。
- **后端实测**：本平台（RDNA3 + Windows ROCm）上 Triton 后端在中长序列比 SDPA 快 1.8–2.3×，保持默认即可。
- **torch.compile**：本平台实测首次编译 89–210s 且编译后慢 2.4–5.6×（inductor 不支持 complex 算子），不建议启用。
- **训练显存**：bf16 全参（权重+梯度 ~14.3GB @4.0B）+ 梯度检查点 + Adafactor；`mem_frac` 熔断默认 0.93、硬上限 0.95；周期重启（exit 43）清 HIP 碎片；存档串行化避免多线程同时驻留大 state_dict 的内存尖峰。

## 验证结果

**CPU 自检**（139 项全 PASS）：结构前向 / gate=0 逐 bit 恒等回归（`torch.equal`）/ RoPE 实数版与复数参考版逐值对齐（差 ≤1 ULP）/ F.rms_norm 与手写版等价 / F2 谱初始化与 F5 谱范数冻结封装 / 全局 MoE α=0 回退 / 各零门控模块恒等起步 / 梯度检查点前后数值与梯度逐值一致 / 4.0B 参数量硬断言。

**GPU 实测**（RX 7900 XT 21.46GB，ROCm 7.2.1）：

- 4.0B bf16 真前向：小输入峰值 9.21GB / 56ms，最大配置（T=16、64×64 latent）11.91GB / 3.8s，输出全有限值；
- Triton kernel（`gate_decay_forget`）：fp16 前向误差 2.7e-3、反向四路梯度对齐；
- FA2（Triton 后端）：5 组配置前向+反向数值对齐全部通过（bf16 精度内）；
- 时序算子节点直测 10/10。

## 诚实边界

1. **不含已执行的蒸馏训练**：`distill_loss.py` 只定义 loss 与自检；真训练需数据与教师预编码管线。训练层脚本（`train_backbone.py` / `run_training_daemon.py`）已就绪但尚未跑完正式训练。
2. **前向跑通 ≠ 画质达标**：结构、红线、数值等价均有自动化验证；画质/时序指标需成片评测。
3. **GSA 为骨架实现**：门控源当前用条件向量，可换 per-token 特征；与 2+1D 的对照需真消融。
4. **VAE 仅路径配置**：解码/tiled decode 属部署链，不在本骨架内。
5. 部署侧机制（调度 / 缓存 / 校验链 / 输出后处理）见技术白皮书，不在本仓库代码范围内。

## 技术文档

设计细节、参数账、量化红线、蒸馏课程与实现记录见 [`docs/TECHNICAL_WHITEPAPER.md`](docs/TECHNICAL_WHITEPAPER.md)。

## License

[BSL 1.1](LICENSE)（Business Source License）：源码可查、可修改、可再分发，研究与非生产用途开放；**2030-01-01（Change Date）后自动转为 GPL-3.0 开源**。生产/商用授权请联系作者。
