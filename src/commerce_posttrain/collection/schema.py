"""Content-addressed raw Teacher trajectory schema and validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

RAW_SCHEMA_VERSION = "commerce-teacher-raw-v1"
TERMINAL_STATUSES = {
    "done",
    "assistant_final",
    "invalid_action_limit",
    "max_steps",
    "policy_error",
    "infrastructure_failure",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def request_identity(
    *,
    task_id: int,
    attempt_index: int,
    runtime_contract_sha256: str,
    teacher: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": RAW_SCHEMA_VERSION,
        "task_id": int(task_id),
        "attempt_index": int(attempt_index),
        "runtime_contract_sha256": runtime_contract_sha256,
        "teacher": dict(teacher),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def strict_gold_success(trajectory: Mapping[str, Any]) -> bool:
    terminal = trajectory.get("terminal_environment_result") or {}
    reward = terminal.get("reward_detail") or {}
    return (
        trajectory.get("status") == "done"
        and trajectory.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
        and reward.get("reward_version") == "shopsimulator-reward-v3"
        and reward.get("reward_type") == "gold_purchase"
        and reward.get("reward_valid") is True
        and reward.get("purchase_success") is True
        and reward.get("termination_reason") == "gold_purchase"
    )


def validate_raw_trajectory(trajectory: Mapping[str, Any]) -> dict:
    value = dict(trajectory)
    if value.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("unsupported raw trajectory schema")
    try:
        task_id = int(value["task_id"])
        attempt_index = int(value["attempt_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("raw trajectory requires task_id and attempt_index") from exc
    if task_id < 0 or attempt_index not in {0, 1, 2}:
        raise ValueError("attempt_index must be one of 0, 1, 2")
    runtime_hash = value.get("runtime_contract_sha256")
    if not isinstance(runtime_hash, str) or len(runtime_hash) != 64:
        raise ValueError("raw trajectory requires a full runtime contract hash")
    teacher = value.get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("model"):
        raise ValueError("raw trajectory requires frozen Teacher settings")
    expected_id = request_identity(
        task_id=task_id,
        attempt_index=attempt_index,
        runtime_contract_sha256=runtime_hash,
        teacher=teacher,
    )
    if value.get("request_id") != expected_id:
        raise ValueError("raw trajectory request identity mismatch")
    status = value.get("status")
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"raw trajectory has non-terminal status: {status!r}")
    events = value.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("raw trajectory requires an event stream")
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("event_index") != index:
            raise ValueError("raw event indexes must be contiguous")
        if event.get("type") in {"environment_reset", "environment_step"}:
            observation = event.get("actor_visible_observation")
            if not isinstance(observation, str):
                raise ValueError("environment events require actor-visible observation")
    if value.get("released") is not True and not (
        status == "infrastructure_failure" and value.get("release_error")
    ):
        raise ValueError("ShopSimulator lease was not released")
    if status == "infrastructure_failure" and not value.get("error"):
        raise ValueError("infrastructure failure requires an error record")
    recorded_hash = value.pop("trajectory_sha256", None)
    expected_hash = sha256_bytes(canonical_json_bytes(value))
    if recorded_hash != expected_hash:
        raise ValueError("raw trajectory content hash mismatch")
    value["trajectory_sha256"] = recorded_hash
    return value


def finalize_trajectory(trajectory: Mapping[str, Any]) -> dict:
    value = dict(trajectory)
    value.pop("trajectory_sha256", None)
    value["trajectory_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return validate_raw_trajectory(value)
