# v1 Final-200 结果

## 评测口径

M0、M1、M2、M3 使用同一份 Final-200 Clean，每题一次 rollout，固定分母为 200。所有 run 共享环境、Reward、工具/Observation 协议和 protocol hash；当前 M3 记录更新为 optimizer step100 导出。严格成功只计完整 `gold_purchase` 且 `reward_valid=true` 的终局。

## 主结果

| 模型 | strict gold success | infrastructure invalid | mean steps | mean terminal utility |
|---|---:|---:|---:|---:|
| M0 Base | 2/200（1.0%）| 2 | 5.40 | −0.100 |
| M1 Outcome SFT | 137/200（68.5%）| 3 | 12.05 | +0.584 |
| M2 Process SFT | 130/200（65.0%）| 4 | 11.50 | +0.553 |
| M3 GRPO step100 export | **139/200（69.5%）** | 2 | 11.25 | +0.602 |

## 配对统计

统计以 task_id 配对，使用双侧 exact McNemar；utility 差异使用配对 bootstrap 95% CI。

| 比较 | strict Δ | 回退 / 改善 | p-value | utility 差 CI |
|---|---:|---:|---:|---|
| M0 → M1 | +67.5pp | 0 / 135 | <0.0001 | [+0.589, +0.776] |
| M1 → M2 | −3.5pp | 16 / 9 | .230 | [−0.100, +0.038] |
| M2 → M3 | +4.5pp | 7 / 16 | .093 | — |

M2→M3 的双侧 exact McNemar p-value 由 16 个改善和 7 个回退计算为 `0.093139648438`。当前聚合更新未提供逐题 terminal utility，因此不沿用旧 checkpoint 的 utility bootstrap CI。

## 过程面板

LLM Judge 只解释通过脱敏和硬检查的轨迹，不改变 strict success。有效 Judge 维度均分（0–2）为：

| 模型 | decision | search | evidence | candidate | termination | judge coverage | reward valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.130 | 0.960 | 0.280 | 0.710 | 0.040 | 0.985 | 0.185 |
| M1 | 1.604 | 1.635 | 1.462 | 1.665 | 1.533 | 0.985 | 0.975 |
| M2 | 1.587 | 1.668 | 1.490 | 1.673 | 1.556 | 0.980 | 0.950 |

当前 step100 更新没有附带新的 Judge 聚合，因此不把其他 checkpoint 的 M3 Judge 数字混入本表。基础设施 invalid 为 1–2%，高于项目 `<1%` 门槛；invalid 任务仍在分母中，Judge 侧记为 `not_judged`。

## 结论

1. SFT 是主要增益来源：M0→M1 提升 67.5pp，且配对差异显著。
2. Process 选择未优于 Outcome：M1→M2 下降 3.5pp，统计上不显著。
3. 在 `lr=1e-6`、LoRA `r=16`、每 prompt `n=4` 的配方下，GRPO step100 相对 M2 提升 4.5pp，方向正向但未达到 0.05 显著性水平。冻结合同为 `total_training_steps=500`、`save_freq=50`；本次通过内部 exact-step barrier 在 100 optimizer steps 后受控停止并导出 step100，不应将这一结果外推到其他 GRPO 配方。

完整的逐题轨迹、Rubric/Judge 输出和运行 manifest 属于实验审计产物；本页只发布可复核的聚合结果。
