#!/usr/bin/env python3
"""Fail-closed validation gate for frozen curated SFT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.curation.pipeline import (  # noqa: E402
    SFT_DATASET_VERSION,
    TERMINAL_TOOL_CONTENT,
    count_jsonl_rows,
    manifest_hash,
    sha256_file,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402

FORBIDDEN_KEYS = {
    "environment_result_private",
    "terminal_environment_result",
    "reward_detail",
    "reasoning_content",
    "provider",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/manifests/split_manifest.json",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=ROOT / "configs/runtime/system_prompt.txt",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_keys(child)


def validate_sft_file(path: Path, *, prompt: str) -> set[int]:
    task_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SFT_DATASET_VERSION:
                raise ValueError(f"{path}:{line_number}: bad SFT schema")
            task_id = int(row["task_id"])
            if task_id in task_ids:
                raise ValueError(f"{path}:{line_number}: duplicate task {task_id}")
            task_ids.add(task_id)
            if row.get("tools") != SHOP_TOOL_SCHEMAS:
                raise ValueError(f"{path}:{line_number}: tool schema drift")
            forbidden = FORBIDDEN_KEYS & set(nested_keys(row))
            if forbidden:
                raise ValueError(
                    f"{path}:{line_number}: private/audit keys leaked: {sorted(forbidden)}"
                )
            messages = row.get("messages") or []
            if len(messages) < 4:
                raise ValueError(f"{path}:{line_number}: incomplete chat")
            if messages[0] != {"role": "system", "content": prompt}:
                raise ValueError(f"{path}:{line_number}: system prompt drift")
            if messages[1].get("role") != "user" or not messages[1].get("content"):
                raise ValueError(f"{path}:{line_number}: missing user instruction")
            history = messages[2:]
            if len(history) % 2:
                raise ValueError(f"{path}:{line_number}: unpaired assistant/tool history")
            for index in range(0, len(history), 2):
                assistant, tool = history[index : index + 2]
                calls = assistant.get("tool_calls") or []
                if assistant.get("role") != "assistant" or len(calls) != 1:
                    raise ValueError(f"{path}:{line_number}: non-serial assistant action")
                if tool.get("role") != "tool":
                    raise ValueError(f"{path}:{line_number}: missing tool observation")
                call = calls[0]
                if tool.get("tool_call_id") != call.get("id"):
                    raise ValueError(f"{path}:{line_number}: tool_call_id mismatch")
                function = call.get("function") or {}
                arguments = function.get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(arguments) if isinstance(arguments, str) else arguments
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed arguments"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(f"{path}:{line_number}: arguments not object")
            final_function = (history[-2]["tool_calls"][0].get("function") or {})
            if final_function.get("name") != "buy_now":
                raise ValueError(f"{path}:{line_number}: terminal action is not buy_now")
            if history[-1].get("content") != TERMINAL_TOOL_CONTENT:
                raise ValueError(f"{path}:{line_number}: terminal result was not sanitized")
    return task_ids


def main() -> int:
    args = parse_args()
    manifest_path = args.curated_dir / "curation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest_hash(manifest)
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("curation manifest content hash mismatch")
    for record in manifest["outputs"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {path}")
        if count_jsonl_rows(path) != int(record["rows"]):
            raise ValueError(f"output row count mismatch: {path}")

    prompt = args.system_prompt.read_text(encoding="utf-8")
    datasets = {
        "outcome_train": args.curated_dir / "outcome" / "train.jsonl",
        "outcome_dev": args.curated_dir / "outcome" / "dev.jsonl",
        "process_train": args.curated_dir / "process" / "train.jsonl",
        "process_dev": args.curated_dir / "process" / "dev.jsonl",
    }
    task_sets = {
        name: validate_sft_file(path, prompt=prompt)
        for name, path in datasets.items()
    }
    if task_sets["outcome_train"] != task_sets["process_train"]:
        raise ValueError("Outcome/Process train task sets differ")
    if task_sets["outcome_dev"] != task_sets["process_dev"]:
        raise ValueError("Outcome/Process dev task sets differ")
    train_ids = task_sets["outcome_train"]
    dev_ids = task_sets["outcome_dev"]
    if train_ids & dev_ids:
        raise ValueError("train/dev task leakage")

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    held_out = set()
    for name, split in split_manifest["splits"].items():
        if name != "teacher_pool":
            held_out.update(int(task_id) for task_id in split["task_ids"])
    if (train_ids | dev_ids) & held_out:
        raise ValueError("SFT task leaks into GRPO/Final split")

    report = {
        "schema_version": "commerce-curated-validation-v1",
        "ready": True,
        "curation_manifest_sha256": manifest["manifest_sha256"],
        "curation_manifest_file_sha256": sha256_file(manifest_path),
        "train_tasks": len(train_ids),
        "dev_tasks": len(dev_ids),
        "train_dev_overlap": 0,
        "held_out_overlap": 0,
        "outcome_process_train_equal": True,
        "outcome_process_dev_equal": True,
        "private_key_leakage": 0,
        "tool_schema_drift": 0,
        "terminal_sanitization_failures": 0,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output = args.output or args.curated_dir / "validation_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
