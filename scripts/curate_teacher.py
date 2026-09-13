#!/usr/bin/env python3
"""Build frozen Outcome/Process SFT datasets from append-only Teacher raw."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.collection.schema import strict_gold_success  # noqa: E402
from commerce_posttrain.curation.pipeline import (  # noqa: E402
    CURATION_VERSION,
    DEFAULT_CURATION_SEED,
    assign_stratified_splits,
    build_action_only_sft_row,
    count_by_split_and_difficulty,
    count_jsonl_rows,
    hard_acceptance_reasons,
    iter_latest_trajectories,
    latest_trajectory_offsets,
    manifest_hash,
    select_outcome_and_process,
    sha256_file,
)
from commerce_posttrain.curation.process_analyzer import (  # noqa: E402
    PROCESS_ANALYZER_VERSION,
    analyze_trajectory,
    attach_first_divergence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data/private/task_facts.jsonl"
    )
    parser.add_argument(
        "--reachability",
        type=Path,
        default=ROOT / "data/manifests/task_reachability_manifest.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/manifests/split_manifest.json",
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "data/manifests/runtime_contract.json",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=ROOT / "configs/runtime/system_prompt.txt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collected-task-count", type=int, default=700)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--dev-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_CURATION_SEED)
    return parser.parse_args()


def read_selected_tasks(path: Path, selected: set[int]) -> dict[int, dict]:
    tasks = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in selected:
                tasks[task_id] = row
    missing = sorted(selected - tasks.keys())
    if missing:
        raise ValueError(f"selected task facts missing: {missing[:20]}")
    return tasks


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def feature_metrics(feature: dict) -> dict[str, int]:
    values = feature["features"]
    return {
        "guard_rejections": int(values["legality"]["guard_rejections"]),
        "malformed_calls": int(values["legality"]["malformed_calls"]),
        "schema_rejections": int(values["legality"]["schema_rejections"]),
        "missing_required_detail_types": len(
            values["evidence"]["missing_required_detail_types"]
        ),
        "repeat_actions": int(values["termination"]["repeat_actions"]),
        "no_progress_actions": int(values["termination"]["no_progress_actions"]),
        "steps_after_decision_ready": int(
            values["termination"]["steps_after_decision_ready"]
        ),
        "projection_truncations": int(values["context"]["projection_truncations"]),
    }


def paired_metric_report(pairs: list[tuple[dict, dict]]) -> dict:
    if not pairs:
        return {}
    names = feature_metrics(pairs[0][0]).keys()
    report = {}
    for name in names:
        outcome = [feature_metrics(left)[name] for left, _ in pairs]
        process = [feature_metrics(right)[name] for _, right in pairs]
        deltas = [right - left for left, right in zip(outcome, process)]
        report[name] = {
            "outcome_mean": round(sum(outcome) / len(outcome), 4),
            "process_mean": round(sum(process) / len(process), 4),
            "process_minus_outcome_mean": round(sum(deltas) / len(deltas), 4),
            "process_better_count": sum(delta < 0 for delta in deltas),
            "equal_count": sum(delta == 0 for delta in deltas),
            "process_worse_count": sum(delta > 0 for delta in deltas),
        }
    return report


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 4),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def main() -> int:
    args = parse_args()
    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    reachability = json.loads(args.reachability.read_text(encoding="utf-8"))
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    prompt = args.system_prompt.read_text(encoding="utf-8")

    teacher_eligible = [
        int(task_id)
        for task_id in reachability["splits"]["teacher_pool"]["eligible_task_ids"]
    ]
    if args.collected_task_count > len(teacher_eligible):
        raise ValueError("collected task count exceeds frozen eligible pool")
    selected_ids = set(teacher_eligible[: args.collected_task_count])
    tasks = read_selected_tasks(args.tasks, selected_ids)

    other_split_ids = set()
    for name, split in split_manifest["splits"].items():
        if name != "teacher_pool":
            other_split_ids.update(int(task_id) for task_id in split["task_ids"])
    leakage = sorted(selected_ids & other_split_ids)
    if leakage:
        raise ValueError(f"Teacher tasks leak into held-out splits: {leakage[:20]}")

    latest_index, raw_rows = latest_trajectory_offsets(
        args.raw, selected_task_ids=selected_ids
    )
    expected_attempts = args.collected_task_count * int(runtime["attempts_per_task"])
    if len(latest_index) != expected_attempts:
        raise ValueError(
            f"effective attempts {len(latest_index)} != expected {expected_attempts}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_features_path = args.output_dir / "process_features.jsonl"
    accepted_by_task = defaultdict(list)
    rejection_counts = Counter()
    strict_gold_attempt_count = 0
    effective_task_count = 0
    trajectory_iterator = iter_latest_trajectories(
        args.raw, selected_task_ids=selected_ids
    )
    with process_features_path.open("w", encoding="utf-8", newline="\n") as output:
        for task_id, group in groupby(
            trajectory_iterator, key=lambda row: int(row["task_id"])
        ):
            effective_task_count += 1
            trajectories = list(group)
            if len(trajectories) != int(runtime["attempts_per_task"]):
                raise ValueError(
                    f"task {task_id} has {len(trajectories)} effective attempts"
                )
            features = [
                analyze_trajectory(row, query=tasks[task_id]["query"])
                for row in trajectories
            ]
            features = attach_first_divergence(features, trajectories)
            for trajectory, feature in zip(trajectories, features):
                output.write(
                    json.dumps(feature, ensure_ascii=False, sort_keys=True) + "\n"
                )
                strict_gold_attempt_count += int(strict_gold_success(trajectory))
                reasons = hard_acceptance_reasons(
                    trajectory,
                    expected_contract_sha256=runtime["contract_sha256"],
                )
                if reasons:
                    rejection_counts.update(reasons)
                    continue
                accepted_by_task[task_id].append(
                    (
                        {
                            "request_id": trajectory["request_id"],
                            "attempt_index": int(trajectory["attempt_index"]),
                        },
                        feature,
                    )
                )
    if effective_task_count != args.collected_task_count:
        raise ValueError(
            f"effective tasks {effective_task_count} != expected {args.collected_task_count}"
        )
    usable_ids = sorted(task_id for task_id, rows in accepted_by_task.items() if rows)
    required = args.train_count + args.dev_count
    if len(usable_ids) < required:
        raise ValueError(
            f"only {len(usable_ids)} hard-accepted tasks; {required} required"
        )

    assignment = assign_stratified_splits(
        [tasks[task_id] for task_id in usable_ids],
        train_count=args.train_count,
        dev_count=args.dev_count,
        seed=args.seed,
    )
    selections = []
    selected_request_targets = defaultdict(list)
    paired_features = {"all": [], "train_dev": []}
    for task_id in usable_ids:
        outcome, process = select_outcome_and_process(accepted_by_task[task_id])
        split = assignment[task_id]
        task = tasks[task_id]
        features_by_request = {
            trajectory["request_id"]: feature
            for trajectory, feature in accepted_by_task[task_id]
        }
        pair = (
            features_by_request[outcome["request_id"]],
            features_by_request[process["request_id"]],
        )
        paired_features["all"].append(pair)
        if split in {"train", "dev"}:
            paired_features["train_dev"].append(pair)
        selections.append(
            {
                "task_id": task_id,
                "difficulty": task.get("difficulty", "unknown"),
                "split": split,
                "successful_attempt_count": len(accepted_by_task[task_id]),
                "outcome_request_id": outcome["request_id"],
                "outcome_attempt_index": int(outcome["attempt_index"]),
                "process_request_id": process["request_id"],
                "process_attempt_index": int(process["attempt_index"]),
                "different_selection": outcome["request_id"] != process["request_id"],
            }
        )
        selected_request_targets[outcome["request_id"]].append(("outcome", split))
        selected_request_targets[process["request_id"]].append(("process", split))

    paths = {
        "features": process_features_path,
        "selections": args.output_dir / "selections.jsonl",
        "outcome_train": args.output_dir / "outcome" / "train.jsonl",
        "outcome_dev": args.output_dir / "outcome" / "dev.jsonl",
        "process_train": args.output_dir / "process" / "train.jsonl",
        "process_dev": args.output_dir / "process" / "dev.jsonl",
        "outcome_reserve": args.output_dir / "outcome" / "reserve.jsonl",
        "process_reserve": args.output_dir / "process" / "reserve.jsonl",
    }
    write_jsonl(paths["selections"], selections)
    dataset_paths = {
        ("outcome", "train"): paths["outcome_train"],
        ("outcome", "dev"): paths["outcome_dev"],
        ("process", "train"): paths["process_train"],
        ("process", "dev"): paths["process_dev"],
        ("outcome", "reserve"): paths["outcome_reserve"],
        ("process", "reserve"): paths["process_reserve"],
    }
    for path in dataset_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    written = Counter()
    sft_shape = {
        "outcome": {"assistant_turns": [], "assistant_action_characters": []},
        "process": {"assistant_turns": [], "assistant_action_characters": []},
    }
    with ExitStack() as stack:
        handles = {
            target: stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
            for target, path in dataset_paths.items()
        }
        for trajectory in iter_latest_trajectories(
            args.raw, selected_task_ids=selected_ids
        ):
            targets = selected_request_targets.get(trajectory["request_id"], [])
            if not targets:
                continue
            row = build_action_only_sft_row(
                trajectory,
                system_prompt=prompt,
                user_instruction=tasks[int(trajectory["task_id"])]["query"],
            )
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for target in targets:
                handles[target].write(encoded)
                written[target] += 1
                arm = target[0]
                if target[1] == "reserve":
                    continue
                assistant_messages = [
                    message
                    for message in row["messages"]
                    if message.get("role") == "assistant"
                ]
                sft_shape[arm]["assistant_turns"].append(len(assistant_messages))
                sft_shape[arm]["assistant_action_characters"].append(
                    sum(
                        len(
                            json.dumps(
                                message,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        for message in assistant_messages
                    )
                )
    expected_written = {
        ("outcome", "train"): args.train_count,
        ("outcome", "dev"): args.dev_count,
        ("process", "train"): args.train_count,
        ("process", "dev"): args.dev_count,
        ("outcome", "reserve"): len(usable_ids) - args.train_count - args.dev_count,
        ("process", "reserve"): len(usable_ids) - args.train_count - args.dev_count,
    }
    if dict(written) != expected_written:
        raise ValueError(f"SFT row counts do not match frozen split: {dict(written)}")

    split_counts = Counter(assignment.values())
    success_histogram = Counter(len(rows) for rows in accepted_by_task.values() if rows)
    different = sum(row["different_selection"] for row in selections)
    manifest = {
        "schema_version": CURATION_VERSION,
        "seed": args.seed,
        "inputs": {
            "raw_path": str(args.raw),
            "raw_sha256": sha256_file(args.raw),
            "raw_append_only_rows": raw_rows,
            "effective_request_count": len(latest_index),
            "superseded_retry_rows": raw_rows - len(latest_index),
            "runtime_contract_sha256": runtime["contract_sha256"],
            "reachability_manifest_sha256": reachability["manifest_sha256"],
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "task_facts_sha256": sha256_file(args.tasks),
        },
        "collection": {
            "selected_task_count": len(selected_ids),
            "attempts_per_task": runtime["attempts_per_task"],
            "strict_gold_attempt_count": strict_gold_attempt_count,
            "hard_accepted_attempt_count": sum(map(len, accepted_by_task.values())),
            "usable_task_count": len(usable_ids),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "successful_attempts_per_usable_task": {
                str(key): success_histogram[key] for key in sorted(success_histogram)
            },
        },
        "frozen_splits": {
            "train": split_counts["train"],
            "dev": split_counts["dev"],
            "reserve": split_counts["reserve"],
            "difficulty_counts": count_by_split_and_difficulty(selections),
        },
        "selection_audit": {
            "multi_success_tasks": sum(
                row["successful_attempt_count"] >= 2 for row in selections
            ),
            "train_dev_multi_success_tasks": sum(
                row["successful_attempt_count"] >= 2
                for row in selections
                if row["split"] in {"train", "dev"}
            ),
            "outcome_process_different_tasks": different,
            "outcome_process_different_rate": round(different / len(selections), 4),
            "train_dev_different_tasks": sum(
                row["different_selection"]
                for row in selections
                if row["split"] in {"train", "dev"}
            ),
            "paired_process_metrics_all_usable": paired_metric_report(
                paired_features["all"]
            ),
            "paired_process_metrics_train_dev": paired_metric_report(
                paired_features["train_dev"]
            ),
            "sft_shape_train_dev": {
                arm: {
                    name: distribution(values)
                    for name, values in metrics.items()
                }
                for arm, metrics in sft_shape.items()
            },
        },
        "process_analyzer_version": PROCESS_ANALYZER_VERSION,
        "leakage": {
            "other_split_overlap_count": 0,
            "outcome_process_task_sets_equal": True,
        },
        "outputs": {},
    }
    for name, path in paths.items():
        manifest["outputs"][name] = {
            "path": str(path),
            "rows": count_jsonl_rows(path),
            "sha256": sha256_file(path),
        }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    manifest_path = args.output_dir / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
