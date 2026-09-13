#!/usr/bin/env python3
"""Exact Qwen chat-template and assistant-label gate before GPU rental."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(sum(values) / len(values), 4),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=ROOT / "outputs/teacher-curated-v1",
    )
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--expected-train-count", type=int, default=400)
    parser.add_argument("--expected-dev-count", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_components(model_path: Path):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    kwargs = {"trust_remote_code": True, "local_files_only": True}
    config = AutoConfig.from_pretrained(model_path, **kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        return config, processor.tokenizer, processor, True
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return config, tokenizer, tokenizer, False


def inspect_file(
    path: Path,
    *,
    tokenizer,
    chat_template,
    normalize_messages,
    build_supervised_example,
    ignore_index: int,
    max_length: int,
) -> tuple[dict, set[int]]:
    lengths: list[int] = []
    supervised: list[int] = []
    assistant_turns: list[int] = []
    failures = []
    failure_counts = Counter()
    task_ids = set()
    total = 0
    started = time.time()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in task_ids:
                raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id}")
            task_ids.add(task_id)
            messages = row["messages"]
            tools = row.get("tools") or []
            assistant_turns.append(
                sum(message.get("role") == "assistant" for message in messages)
            )
            normalized = normalize_messages(messages)
            reason = None
            length = None
            example = None
            if normalized is None:
                reason = "arguments_normalization_failed"
            else:
                try:
                    rendered = chat_template.apply_chat_template(
                        normalized,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    length = len(
                        tokenizer(rendered, add_special_tokens=False)["input_ids"]
                    )
                    lengths.append(length)
                except Exception as exc:
                    reason = f"template_render_error:{exc.__class__.__name__}"
            if reason is None and length is not None and length > max_length:
                reason = "over_max_length"
            if reason is None:
                example = build_supervised_example(
                    messages=messages,
                    tools=tools,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    chat_template=chat_template,
                )
                if example is None:
                    reason = "assistant_label_boundary_failed"
            if reason is not None:
                failure_counts[reason] += 1
                failures.append(
                    {
                        "line_number": line_number,
                        "task_id": task_id,
                        "trajectory_id": row.get("trajectory_id"),
                        "reason": reason,
                        "token_length": length,
                    }
                )
            else:
                supervised.append(
                    sum(label != ignore_index for label in example["labels"])
                )
            if total % 50 == 0:
                elapsed = time.time() - started
                print(
                    f"{path.name}: {total} rows, failures={len(failures)}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "total": total,
            "kept": total - len(failures),
            "dropped": len(failures),
            "failure_counts": dict(sorted(failure_counts.items())),
            "failures": failures,
            "token_length": distribution(lengths),
            "supervised_assistant_tokens": distribution(supervised),
            "assistant_turns": distribution(assistant_turns),
        },
        task_ids,
    )


def main() -> int:
    args = parse_args()
    if args.max_length < 1:
        raise SystemExit("--max-length must be positive")
    upstream_dataset = (
        args.upstream_root / "src/shopping_grpo/training/sft/dataset.py"
    )
    upstream_train = args.upstream_root / "scripts/train_lora_sft.py"
    if not upstream_dataset.is_file() or not upstream_train.is_file():
        raise SystemExit(f"locked upstream SFT runtime missing: {args.upstream_root}")
    sys.path.insert(0, str(args.upstream_root / "src"))
    from shopping_grpo.training.sft.dataset import (  # noqa: E402
        IGNORE_INDEX,
        build_supervised_example,
        normalize_messages_for_chat_template,
    )

    config, tokenizer, chat_template, is_multimodal = load_components(args.model)
    files = {
        "outcome_train": args.curated_dir / "outcome/train.jsonl",
        "outcome_dev": args.curated_dir / "outcome/dev.jsonl",
        "process_train": args.curated_dir / "process/train.jsonl",
        "process_dev": args.curated_dir / "process/dev.jsonl",
    }
    reserve_files = {
        "outcome_reserve": args.curated_dir / "outcome/reserve.jsonl",
        "process_reserve": args.curated_dir / "process/reserve.jsonl",
    }
    if all(path.exists() for path in reserve_files.values()):
        files.update(reserve_files)
    results = {}
    task_sets = {}
    for name, path in files.items():
        print(f"\n=== {name}: {path} ===", flush=True)
        results[name], task_sets[name] = inspect_file(
            path,
            tokenizer=tokenizer,
            chat_template=chat_template,
            normalize_messages=normalize_messages_for_chat_template,
            build_supervised_example=build_supervised_example,
            ignore_index=IGNORE_INDEX,
            max_length=args.max_length,
        )

    expected = {
        "outcome_train": args.expected_train_count,
        "outcome_dev": args.expected_dev_count,
        "process_train": args.expected_train_count,
        "process_dev": args.expected_dev_count,
    }
    if "outcome_reserve" in results:
        # 两臂 Reserve 各自显式读取 total；任何一臂缺行都应在这里暴露，
        # 而不是被"拿另一臂的数量当期望值"的写法掩盖。
        expected["outcome_reserve"] = results["outcome_reserve"]["total"]
        expected["process_reserve"] = results["process_reserve"]["total"]
        if expected["outcome_reserve"] != expected["process_reserve"]:
            raise SystemExit(
                "Outcome / Process Reserve 行数不一致："
                f"outcome={expected['outcome_reserve']} process={expected['process_reserve']}"
            )
    count_gate = all(results[name]["kept"] == count for name, count in expected.items())
    equality_gate = (
        task_sets["outcome_train"] == task_sets["process_train"]
        and task_sets["outcome_dev"] == task_sets["process_dev"]
        and (
            "outcome_reserve" not in task_sets
            or task_sets["outcome_reserve"] == task_sets["process_reserve"]
        )
    )
    disjoint_gate = not (
        task_sets["outcome_train"] & task_sets["outcome_dev"]
    )
    ready = count_gate and equality_gate and disjoint_gate
    config_path = args.model / "config.json"
    tokenizer_config = args.model / "tokenizer_config.json"
    report = {
        "schema_version": "commerce-sft-preflight-v1",
        "ready_for_gpu_smoke": ready,
        "max_length": args.max_length,
        "model": {
            "path": str(args.model),
            "model_type": getattr(config, "model_type", None),
            "is_multimodal": is_multimodal,
            "config_sha256": sha256_file(config_path) if config_path.exists() else None,
            "tokenizer_config_sha256": sha256_file(tokenizer_config)
            if tokenizer_config.exists()
            else None,
            "chat_template_sha256": sha256_text(
                str(getattr(chat_template, "chat_template", ""))
            ),
        },
        "upstream_runtime": {
            "root": str(args.upstream_root),
            "dataset_py_sha256": sha256_file(upstream_dataset),
            "train_lora_sft_py_sha256": sha256_file(upstream_train),
        },
        "files": results,
        "gates": {
            "all_expected_rows_kept": count_gate,
            "outcome_process_task_sets_equal": equality_gate,
            "train_dev_disjoint": disjoint_gate,
        },
    }
    clean = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(clean).hexdigest()
    output = args.output or args.curated_dir / "sft_preflight_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
