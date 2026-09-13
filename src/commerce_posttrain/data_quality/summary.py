"""Quality-adjusted reporting without mutating raw collection artifacts."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from commerce_posttrain.collection.schema import canonical_json_bytes, strict_gold_success

QUALITY_SUMMARY_VERSION = "commerce-collection-quality-summary-v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def reachability_by_task(manifest: Mapping[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for split in manifest.get("splits", {}).values():
        for task_id in split.get("eligible_task_ids", []):
            task_id = int(task_id)
            if task_id in result:
                raise ValueError(f"duplicate reachability classification: {task_id}")
            result[task_id] = "eligible"
        for row in split.get("unreachable", []):
            task_id = int(row["task_id"])
            if task_id in result:
                raise ValueError(f"duplicate reachability classification: {task_id}")
            result[task_id] = "unreachable"
    return result


def summarize_collection_quality(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    reachability_manifest: Mapping[str, Any],
    raw_sha256: str,
) -> dict:
    classification = reachability_by_task(reachability_manifest)
    unknown = sorted(
        {
            int(row["task_id"])
            for row in trajectories
            if int(row["task_id"]) not in classification
        }
    )
    if unknown:
        raise ValueError(f"unclassified trajectory tasks: {unknown}")
    eligible = [
        row
        for row in trajectories
        if classification[int(row["task_id"])] == "eligible"
    ]
    unreachable = [
        row
        for row in trajectories
        if classification[int(row["task_id"])] == "unreachable"
    ]
    raw_tasks = {int(row["task_id"]) for row in trajectories}
    eligible_tasks = {int(row["task_id"]) for row in eligible}
    unreachable_tasks = {int(row["task_id"]) for row in unreachable}
    gold_by_task: dict[int, int] = defaultdict(int)
    for row in eligible:
        gold_by_task[int(row["task_id"])] += int(strict_gold_success(row))
    raw_gold = sum(strict_gold_success(row) for row in trajectories)
    eligible_gold = sum(strict_gold_success(row) for row in eligible)
    summary = {
        "schema_version": QUALITY_SUMMARY_VERSION,
        "inputs": {
            "raw_sha256": raw_sha256,
            "reachability_manifest_sha256": reachability_manifest[
                "manifest_sha256"
            ],
        },
        "raw_fixed_denominator": {
            "attempt_count": len(trajectories),
            "task_count": len(raw_tasks),
            "strict_gold_count": raw_gold,
            "strict_gold_rate": round(raw_gold / len(trajectories), 4)
            if trajectories
            else 0.0,
            "status_counts": dict(
                sorted(Counter(row.get("status", "missing") for row in trajectories).items())
            ),
        },
        "dataset_unreachable": {
            "attempt_count": len(unreachable),
            "task_count": len(unreachable_tasks),
            "task_ids": sorted(unreachable_tasks),
        },
        "eligible_policy_denominator": {
            "attempt_count": len(eligible),
            "task_count": len(eligible_tasks),
            "strict_gold_count": eligible_gold,
            "strict_gold_rate": round(eligible_gold / len(eligible), 4)
            if eligible
            else 0.0,
            "task_any_gold_count": sum(value > 0 for value in gold_by_task.values()),
            "task_any_gold_rate": round(
                sum(value > 0 for value in gold_by_task.values()) / len(eligible_tasks),
                4,
            )
            if eligible_tasks
            else 0.0,
        },
    }
    summary["summary_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    return summary
