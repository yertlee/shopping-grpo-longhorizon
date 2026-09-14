# v1 复现指南

本页给出公开可复核的身份和执行顺序。Teacher API、模型服务、训练和 200 题评测都可能产生显著成本；只在确认资源和授权后执行。复现时不要把凭据写入仓库。

## 冻结身份

| 对象 | 值 |
|---|---|
| Upstream | `YYHDBL/shopping-grpo-longhorizon@4ed73020e1d7d07eb93e7375a4606b0901d3cded` |
| ShopSimulator | `9ecba272963960ab4a10e1a781bd05cd7634ce20` |
| Product source, compressed | `f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125` |
| Product source, decompressed | `57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f` |
| 当前代码 Runtime contract | `5f0967e9dd0cad5f8041484bdb70a6baaf6ee14e8df81f966bf77adbce6e1bac` |
| Final-200 task split | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |
| Final-200 运行记录的 Evaluation protocol | `0986526cecc9b1a9770c7786b049689a95528c4c6f72a11be78f6400b5049cec` |

模型身份使用 canonical weights hash：M0 `1cf67d8e5f23e10f337ad6fab4dfccc50784d1656dbe00a5ea41faec662cc607`、M1 `d4a3f835d31dab31bdd77d5b96a259d816901e090d1d6624f674a941c328d7ac`、M2 `7869e6c71e64565dfe38dd9a4c0d167f781a09f3fde6698bbdc75c8d7e540750`。当前 M3 记录对应同配方的 optimizer step100 导出。

## 推荐顺序

1. 创建 Python 3.10+ 环境并安装 `uv`、CUDA 兼容的 PyTorch 与项目依赖。
2. 运行 `bash scripts/setup.sh`；它准备 ShopSimulator、搜索索引和固定版本依赖。
3. 先做无外部调用的 smoke/contract 检查：

   ```bash
   PYTHONPATH=src python -m pytest -q
   bash scripts/grpo.sh --dry-run
   ```

4. 启动环境：`bash scripts/start_environment.sh`；另开终端启动 `Qwen/Qwen3.5-2B` 服务。
5. 先运行 Base：`bash scripts/baseline.sh`。
6. 运行 SFT：`bash scripts/sft.sh`，启动导出模型后执行 `bash scripts/evaluate.sh sft`。
7. 运行 GRPO 前先确认初始化模型为 M2、rollout 数为 4、学习率为 `1e-6`，并保留冻结合同 `total_training_steps=500`、`save_freq=50`；本次运行由内部 exact-step barrier 在 100 optimizer steps 后受控停止，从 step100 导出 adapter 并合并，再执行 `bash scripts/evaluate.sh grpo`。

评测命令必须引用冻结的 Final-200 split 和同一协议；不要用 Final-200 选择数据、调阈值或选择 checkpoint。

## 入口与验证

| 目的 | 入口 |
|---|---|
| 环境 | `scripts/start_environment.sh` |
| Baseline | `scripts/baseline.sh` |
| SFT | `scripts/sft.sh`、`scripts/train_lora_sft.py` |
| GRPO | `scripts/grpo.sh`、`scripts/train_grpo.py` |
| LoRA merge | `scripts/merge_lora_adapter.py`、`scripts/merge_grpo_lora.py` |
| Evaluation | `scripts/evaluate.sh`、`scripts/evaluate_student.py` |
| Reports | `scripts/report.sh`、`scripts/report_all.sh` |

每次正式运行都应保留 run manifest、resolved config、输入文件 hash、模型身份、协议 hash、退出状态和输出摘要。契约/协议或输入 hash 不一致时应 fail closed。

## 结果复核

复核时应确认：四个模型各有 200 个 task-level 结果；strict success 使用固定分母；infrastructure invalid 未剔除；M0→M1、M1→M2、M2→M3 的配对统计分别为 +67.5pp、−3.5pp、+4.5pp，并对应 `<0.0001`、`.230`、`.093`。当前结果见 [results-v1.md](results-v1.md)。

## 不可公开输入

私有 TaskFacts、Gold ASIN、完整商品事实、服务凭据、逐题原始轨迹以及任何本机/服务器路径都不是公开复现输入。公开文档只引用 split/contract/protocol 与内容 hash；如果缺少某个身份文件，应停止并补齐 provenance，而不是猜测或改写冻结值。
