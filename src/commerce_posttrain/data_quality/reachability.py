"""Compile task option reachability using the frozen Reward v3 normalizer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

REACHABILITY_SCHEMA_VERSION = "commerce-task-reachability-manifest-v1"
REACHABILITY_VERIFIER_VERSION = "commerce-option-reachability-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def classify_task(
    item: Mapping[str, Any],
    *,
    task_id: int,
    normalize_option_text: Callable[[object], str],
) -> dict:
    """Classify whether every required option exists in the actual UI options.

    Reward v3 falls back to exact normalized selected-option matching when it
    cannot resolve an option axis. Therefore an option is unreachable exactly
    when its normalized required value is absent from every customization
    option exposed by the target product.
    """

    instructions = item.get("instructions") or []
    if len(instructions) != 1:
        return {
            "task_id": int(task_id),
            "status": "unreachable",
            "reason_codes": ["instruction_cardinality"],
            "required_option_count": 0,
            "unmatched_requirement_count": 0,
            "evidence_sha256": sha256_bytes(canonical_json_bytes(instructions)),
        }
    required = [
        str(value)
        for value in instructions[0].get("instruction_options") or []
        if str(value or "").strip()
    ]
    available = [
        str(entry.get("value"))
        for entries in (item.get("customization_options") or {}).values()
        for entry in entries or []
        if isinstance(entry, Mapping) and str(entry.get("value") or "").strip()
    ]
    normalized_available = {normalize_option_text(value) for value in available}
    unmatched = [
        value
        for value in required
        if normalize_option_text(value) not in normalized_available
    ]
    evidence = {
        "required_normalized": sorted(
            normalize_option_text(value) for value in required
        ),
        "available_normalized": sorted(normalized_available),
        "unmatched_normalized": sorted(
            normalize_option_text(value) for value in unmatched
        ),
    }
    return {
        "task_id": int(task_id),
        "status": "unreachable" if unmatched else "eligible",
        "reason_codes": ["required_option_not_exposed"] if unmatched else [],
        "required_option_count": len(required),
        "unmatched_requirement_count": len(unmatched),
        # Keep private option text out of the public manifest while retaining
        # content-addressed evidence for audit/reproduction.
        "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
    }


def build_reachability_manifest(
    *,
    products: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    normalize_option_text: Callable[[object], str],
    upstream_repository: str,
    upstream_commit: str,
    product_source_path: str,
    product_source_compressed_sha256: str,
    product_source_decompressed_sha256: str,
    reward_features_path: str,
    reward_features_sha256: str,
    reward_feature_version: str,
    verifier_sha256: str,
    task_facts_sha256: str,
    split_manifest_sha256: str,
) -> dict:
    split_records = {}
    all_selected_ids: set[int] = set()
    for split_name, split in sorted(split_manifest["splits"].items()):
        task_ids = [int(value) for value in split["task_ids"]]
        all_selected_ids.update(task_ids)
        classifications = [
            classify_task(
                products[task_id],
                task_id=task_id,
                normalize_option_text=normalize_option_text,
            )
            for task_id in task_ids
        ]
        eligible = [row["task_id"] for row in classifications if row["status"] == "eligible"]
        unreachable = [row for row in classifications if row["status"] == "unreachable"]
        split_records[split_name] = {
            "input_count": len(task_ids),
            "eligible_count": len(eligible),
            "unreachable_count": len(unreachable),
            "eligible_task_ids": eligible,
            "unreachable": unreachable,
        }

    selected_classifications = [
        classify_task(
            products[task_id],
            task_id=task_id,
            normalize_option_text=normalize_option_text,
        )
        for task_id in sorted(all_selected_ids)
    ]
    reason_counts = Counter(
        reason
        for row in selected_classifications
        for reason in row["reason_codes"]
    )
    manifest = {
        "schema_version": REACHABILITY_SCHEMA_VERSION,
        "verifier_version": REACHABILITY_VERIFIER_VERSION,
        "verifier_sha256": verifier_sha256,
        "upstream": {
            "repository": upstream_repository,
            "commit": upstream_commit,
            "product_source_path": product_source_path,
            "product_source_compressed_sha256": product_source_compressed_sha256,
            "product_source_decompressed_sha256": product_source_decompressed_sha256,
            "reward_features_path": reward_features_path,
            "reward_features_sha256": reward_features_sha256,
            "reward_feature_version": reward_feature_version,
        },
        "inputs": {
            "task_facts_sha256": task_facts_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        },
        "selected_task_count": len(all_selected_ids),
        "eligible_task_count": sum(
            row["status"] == "eligible" for row in selected_classifications
        ),
        "unreachable_task_count": sum(
            row["status"] == "unreachable" for row in selected_classifications
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "splits": split_records,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def validate_reachability_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split_manifest_sha256: str | None = None,
) -> dict:
    value = json.loads(json.dumps(manifest))
    recorded_hash = value.pop("manifest_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json_bytes(value)):
        raise ValueError("reachability manifest content hash mismatch")
    value["manifest_sha256"] = recorded_hash
    if value.get("schema_version") != REACHABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported reachability manifest schema")
    if value.get("verifier_version") != REACHABILITY_VERIFIER_VERSION:
        raise ValueError("unsupported reachability verifier version")
    if expected_split_manifest_sha256 is not None and (
        value.get("inputs", {}).get("split_manifest_sha256")
        != expected_split_manifest_sha256
    ):
        raise ValueError("reachability manifest uses a different split manifest")
    classified: set[int] = set()
    eligible_total = 0
    unreachable_total = 0
    for split_name, split in value.get("splits", {}).items():
        eligible = [int(task_id) for task_id in split.get("eligible_task_ids", [])]
        unreachable = split.get("unreachable", [])
        unreachable_ids = [int(row["task_id"]) for row in unreachable]
        current = eligible + unreachable_ids
        if len(current) != len(set(current)):
            raise ValueError(f"duplicate task classification in split {split_name}")
        if classified.intersection(current):
            raise ValueError("task classified in more than one split")
        classified.update(current)
        if split.get("input_count") != len(current):
            raise ValueError(f"input count mismatch in split {split_name}")
        if split.get("eligible_count") != len(eligible):
            raise ValueError(f"eligible count mismatch in split {split_name}")
        if split.get("unreachable_count") != len(unreachable):
            raise ValueError(f"unreachable count mismatch in split {split_name}")
        if any(row.get("status") != "unreachable" for row in unreachable):
            raise ValueError(f"invalid unreachable record in split {split_name}")
        eligible_total += len(eligible)
        unreachable_total += len(unreachable)
    if value.get("selected_task_count") != len(classified):
        raise ValueError("selected task count mismatch")
    if value.get("eligible_task_count") != eligible_total:
        raise ValueError("eligible task total mismatch")
    if value.get("unreachable_task_count") != unreachable_total:
        raise ValueError("unreachable task total mismatch")
    return value
