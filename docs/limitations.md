# 限制与后续工作

## 结论边界

Final-200 的结果支持：成功轨迹的 Action-only SFT 是本实验的主要能力增益；Process 选择没有优于 Outcome；在 `lr=1e-6`、LoRA `r=16`、`n=4` 的配方下，GRPO step100 相对 M2 提升 4.5pp（p=.093），呈正向趋势但未达到 0.05 显著性水平。冻结合同是 `total_training_steps=500`、`save_freq=50`，本次由内部 exact-step barrier 在 100 optimizer steps 后受控停止并导出 step100；这里的 GRPO 结论是配方和评测协议限定的观察，不是对所有 GRPO 训练的普遍判断。

## 已知限制

1. **统计重复有限。** M0–M3 各只在固定 Final-200 上运行一次，未估计随机种子方差，也没有跨数据切分的外部验证。exact McNemar 和配对 bootstrap 描述本次 task-level 配对不确定性，不能替代多 seed 实验。
2. **环境外推有限。** ShopSimulator 是模拟购物站点；任务、商品目录、工具 schema 和终止规则不等同于真实电商流量。对真实网站、开放目录和支付安全的结论不在本项目范围内。
3. **基础设施质量未达目标。** 四模型的 infrastructure invalid 为 1–2%，超过预设 `<1%` 门槛。invalid 任务保留在 200 题分母，Judge 侧标记 `not_judged`，因此不能把有效 Judge 均分理解为覆盖全部任务。
4. **训练搜索空间很窄。** GRPO 合同是 500 total training steps、`save_freq=50`，本次在 100 optimizer steps 受控停止；实验只考察一个学习率、LoRA rank 16 和每 prompt 四条 rollout，M3 为 step100 导出。当前正向差异仍未达到统计显著，不说明更长训练、更大学习率或全参数训练的结果。
5. **Judge 不是主指标。** Rubric/Judge 用于解释搜索、候选利用、证据核验、决策和终止等过程；strict success 仍由确定性 Reward/终局检查定义。Judge 分数可能受 rubric 和模型偏差影响。
6. **公开发布边界。** Gold ASIN、私有 TaskFacts、完整环境事实、模型权重与逐题轨迹不是 README 或公开复现输入。读者可按公开代码和身份 hash 审查流程，但无法仅凭仓库重建全部私有评测上下文。

## 后续实验建议

- 在保持 split、Reward 和评测协议不变的前提下，先测量 LoRA delta 是否超过 BF16 表示精度，再扩大学习率/步数或比较全参数更新。
- 为 Process selector 增加与最终 task success 对齐的标签审计，并报告选择分歧、过程指标与成功率的关系。
- 降低 session/release 等 infrastructure invalid，使失败率回到 `<1%` 门槛内。
- 运行多个随机种子和独立留出集，分别报告 seed 方差与跨 split 稳定性。
