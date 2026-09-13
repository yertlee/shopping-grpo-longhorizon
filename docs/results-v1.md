# v1 Final-200 结果

## 评测口径

M0、M1、M2、M3 在同一份 Final-200 Clean 上各运行一次，每题一次 rollout，固定分母为 200。所有 run 共享环境、Reward、工具/Observation 协议和 protocol hash。严格成功只计完整 `gold_purchase` 且 `reward_valid=true` 的终局。

## 主结果

| 模型 | strict gold success | infrastructure invalid | mean steps | mean terminal utility |
|---|---:|---:|---:|---:|
| M0 Base | 2/200（1.0%）| 2 | 5.40 | −0.100 |
| M1 Outcome SFT | **137/200（68.5%）** | 3 | 12.05 | +0.584 |
| M2 Process SFT | 130/200（65.0%）| 4 | 11.50 | +0.553 |
| M3 GRPO step50 export | 131/200（65.5%）| 2 | 11.05 | +0.563 |

## 配对统计

统计以 task_id 配对，使用双侧 exact McNemar；utility 差异使用配对 bootstrap 95% CI。

| 比较 | strict Δ | 回退 / 改善 | p-value | utility 差 CI |
|---|---:|---:|---:|---|
| M0 → M1 | +67.5pp | 0 / 135 | <0.0001 | [+0.589, +0.776] |
| M1 → M2 | −3.5pp | 16 / 9 | .230 | [−0.100, +0.038] |
| M2 → M3 | +0.5pp | 6 / 7 | 1.000 | [−0.040, +0.061] |

M2→M3 的置信区间包含 0；这里的 exact McNemar p-value 按冻结报告记为 `1.000`。

## 过程面板

LLM Judge 只解释通过脱敏和硬检查的轨迹，不改变 strict success。有效 Judge 维度均分（0–2）为：

| 模型 | decision | search | evidence | candidate | termination | judge coverage | reward valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.130 | 0.960 | 0.280 | 0.710 | 0.040 | 0.985 | 0.185 |
| M1 | 1.604 | 1.635 | 1.462 | 1.665 | 1.533 | 0.985 | 0.975 |
| M2 | 1.587 | 1.668 | 1.490 | 1.673 | 1.556 | 0.980 | 0.950 |
| M3 | 1.551 | 1.636 | 1.394 | 1.636 | 1.515 | 0.990 | 0.935 |

这些面板显示 M2 的部分过程指标更高，但并未转化为更高 strict success；M3 在 evidence、termination、reward-valid 等指标上略低于 M2。基础设施 invalid 为 1–2%，高于项目 `<1%` 门槛；invalid 任务仍在分母中，Judge 侧记为 `not_judged`。

## 结论

1. SFT 是主要增益来源：M0→M1 提升 67.5pp，且配对差异显著。
2. Process 选择未优于 Outcome：M1→M2 下降 3.5pp，统计上不显著。
3. 在 `lr=1e-6`、LoRA `r=16`、每 prompt `n=4` 的配方下，GRPO 未产生可检测增益。冻结合同为 `total_training_steps=500`、`save_freq=50`；本次通过内部 exact-step barrier 在 100 optimizer steps 后受控停止，M3 选择 step50 导出，不应将这一结果外推到其他 GRPO 配方。

完整的逐题轨迹、Rubric/Judge 输出和运行 manifest 属于实验审计产物；本页只发布可复核的聚合结果。
