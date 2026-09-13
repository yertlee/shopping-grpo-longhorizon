# v1 数据流水线与清理

## 数据规模与切分

上游压缩商品/任务文件解压后导出 **23,421** 条 task facts。切分在 Teacher API 调用前冻结，并由 component-aware stratified builder 生成：

| Split | Tasks | 用途 |
|---|---:|---|
| Teacher pool | 1,300 | 预冻结的 Teacher 候选池 |
| GRPO train | 1,000 | 在线训练 prompt |
| GRPO validation | 50 | checkpoint/日常验证 |
| Final-200 Clean | 200 | 唯一正式留出测试集 |

跨 split 的 leakage component 同时考虑 task ID、共享目标商品、归一化 Query、数字泛化模板、显式型号 token 和商品 family。Final-200 所在 component 优先隔离；训练和评测 task 不允许重叠。

## Teacher 采集与验收

每个 Teacher task 有 3 个策略 attempt。API 超时、连接中断或环境未创建成功属于 infrastructure retry，沿用 `(task_id, attempt_index, contract_hash)` identity，不作为新策略样本。策略轨迹必须真实执行环境动作。

终局验收要求：

- 环境完成 `done=true` 或 `over=true`；
- Reward v3 给出 `reward_valid=true`；
- 终局包含完整、可核验的 `gold_purchase`；
- 工具调用、消息结构、动作 schema 和轨迹字段均通过硬检查。

不满足条件的轨迹保留 reject reason 供审计，但不进入 SFT 或 GRPO 的有效策略数据。冻结采集的审计计数为 2,100 个有效策略 attempts、2,370 条 append-only raw rows、1,343 条 strict-gold trajectories。硬验收后剩余 1,265 条轨迹，覆盖 512 个可用 task。curated split 明确为 train 400、dev 100、reserve 12；对 shared union 应用 24576-token length gate，并做 difficulty-matched reserve 后，真正的 SFT-ready 输入为 train 398 行、dev 100 行，Outcome/Process task sets 相同。400 是 curated split 计数，而非最终训练行数。

## Outcome 与 Process

两套 SFT 数据使用相同 task 集，差别仅在同 task 内的轨迹选择：

1. **Outcome**：取第一条满足严格终局和结构验收的成功轨迹。
2. **Process**：只使用 Actor 实际可见的过程特征排序成功轨迹；选择顺序固定为 guard rejection、malformed/schema rejection、缺失详情类型、重复动作、无进展动作、decision-ready 后的额外步数和 attempt index。

最终样本渲染为 action-only supervision：只对 Assistant action token 计算 loss，用户 Query 和环境 Observation mask 掉。Reward、Gold 私有字段和隐藏环境状态不写入模型输入。

## Reachability 与 fail-closed

在采集前，Reward 同源的 option normalizer 检查目标商品是否真实暴露每个 required option。不可达任务标记为 `dataset_unreachable`，不通过增加 attempt、替换 Reward 或按结果挑题来补齐。当前扫描显示四个 split 合计 37 个不可达任务：Teacher 19、GRPO train 16、GRPO validation 2、Final-200 0；Final-200 因此保持 200 题。

## Provenance hashes

下列值记录数据来源与当前公开运行合同；它们描述内容身份，不是本机路径：

| 对象 | SHA-256 / identity |
|---|---|
| 上游 commit | `4ed73020e1d7d07eb93e7375a4606b0901d3cded` |
| 商品源（压缩） | `f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125` |
| 商品源（解压） | `57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f` |
| TaskFacts 导出 | `72f72ab2a470c3cf02ec7ea7ecc5e8d9cf6b96909fb824225f238982e5ccd8e5` |
| Reachability manifest | `d26ad0ddc4f5524c6b56c7d197f46e423357262ac74f7db9490131b66e7eb10f` |
| 当前代码 Runtime contract | `5f0967e9dd0cad5f8041484bdb70a6baaf6ee14e8df81f966bf77adbce6e1bac` |
| Final-200 task file | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |

私有 TaskFacts 和 Gold ASIN 不应发布；公开 manifest 只需暴露 task identity、计数、输入 hash、算法版本和输出 hash。
