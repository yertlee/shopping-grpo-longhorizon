# 文档索引

这些公开文档按“理解系统 → 理解数据 → 读结果 → 复现 → 评估边界”的顺序组织。v1 的冻结结论只以 [结果报告](results-v1.md) 为准。

1. [端到端架构](architecture.md)：环境、数据、训练和评测的组件边界。
2. [v1 数据流水线](data-pipeline-v1.md)：任务切分、Teacher 轨迹验收和 Outcome/Process 清理。
3. [v1 Final-200 结果](results-v1.md)：M0–M3 主指标、配对统计和解释。
4. [v1 复现指南](reproducibility-v1.md)：冻结身份、公共 hash、入口和检查顺序。
5. [限制与后续工作](limitations.md)：统计、环境、基础设施和训练配方的边界。

## 现有实现参考

- [Reward v3 设计](reward-v3.md)：确定性终局奖励和验收规则。
- [评测实现细节](evaluation.md)：统一 driver、rubric/judge 隔离和固定分母汇总。
- [SFT 实现说明](sft.md)
- [GRPO 实现说明](grpo.md)
- [原始数据采集说明](data-collection.md)
- [Final-200 数据集说明](evaluation-dataset.md)

README：[中文](../README.md) · [English](../README.en.md)
