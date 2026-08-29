#!/usr/bin/env bash
# LEGACY historical entry — do NOT use for the commerce-agent-posttrain experiment.
#
# 本脚本指向历史的 data/sft/*.jsonl 数据语义，与当前实验的冻结数据
# （outputs/teacher-sft-ready-v1/{process,outcome}）和 Runbook recipe 不同。
# 当前实验的唯一 SFT 入口：docs/SFT_PREFLIGHT_AND_TRAINING_RUNBOOK.md 阶段 C/D。
# 如确需复用此脚本调试历史管线，必须显式设置 ALLOW_LEGACY_SFT=1。
set -euo pipefail

if [[ "${ALLOW_LEGACY_SFT:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
[sft.sh] This is the LEGACY upstream SFT entry (data/sft/*.jsonl semantics).
[sft.sh] The commerce-agent-posttrain experiment does NOT use it.
[sft.sh] Use docs/SFT_PREFLIGHT_AND_TRAINING_RUNBOOK.md (Stage C/D) instead,
[sft.sh] or set ALLOW_LEGACY_SFT=1 explicitly to run the historical pipeline.
MSG
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
# 冻结的 Qwen3.5-2B revision（refs/main）；train 与 merge 两个调用都必须显式传入。
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --revision "$BASE_MODEL_REVISION" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --revision "$BASE_MODEL_REVISION" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
