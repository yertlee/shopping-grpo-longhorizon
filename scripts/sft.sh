#!/usr/bin/env bash
# Train and merge an SFT model from explicitly supplied external JSONL data.
# Prepare and validate the inputs with docs/reproducibility-v1.md and docs/sft.md,
# then set SFT_TRAIN_FILE and SFT_VALIDATION_FILE to the authorized local paths.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
# 冻结的 Qwen3.5-2B revision（refs/main）；train 与 merge 两个调用都必须显式传入。
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:?Set SFT_TRAIN_FILE to an authorized external JSONL}"
SFT_VALIDATION_FILE="${SFT_VALIDATION_FILE:?Set SFT_VALIDATION_FILE to an authorized external JSONL}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --revision "$BASE_MODEL_REVISION" \
  --train "$SFT_TRAIN_FILE" \
  --validation "$SFT_VALIDATION_FILE" \
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
