"""Memory-bounded, deterministic Teacher trajectory curation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from commerce_posttrain.collection.schema import (
    canonical_json_bytes,
    strict_gold_success,
    validate_raw_trajectory,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

CURATION_VERSION = "commerce-teacher-curation-v1"
SFT_DATASET_VERSION = "commerce-action-only-sft-v1"
DEFAULT_CURATION_SEED = 20260829
TERMINAL_TOOL_CONTENT = "购买已完成。"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl_rows(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def latest_trajectory_offsets(
    raw_path: str | Path,
    *,
    selected_task_ids: Iterable[int] | None = None,
) -> tuple[dict[str, dict[str, int]], int]:
    """Index the latest append-only row without retaining full trajectories."""

    selected = (
        {int(task_id) for task_id in selected_task_ids}
        if selected_task_ids is not None
        else None
    )
    latest: dict[str, dict[str, int]] = {}
    raw_rows = 0
    with Path(raw_path).open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            task_id = int(row["task_id"])
            if selected is not None and task_id not in selected:
                continue
            latest[str(row["request_id"])] = {
                "offset": offset,
                "task_id": task_id,
                "attempt_index": int(row["attempt_index"]),
            }
    return latest, raw_rows


def iter_latest_trajectories(
    raw_path: str | Path,
    *,
    selected_task_ids: Iterable[int] | None = None,
) -> Iterator[dict]:
    """Yield validated effective trajectories in task/attempt order."""

    latest, _ = latest_trajectory_offsets(
        raw_path, selected_task_ids=selected_task_ids
    )
    ordered = sorted(
        latest.values(), key=lambda row: (row["task_id"], row["attempt_index"])
    )
    with Path(raw_path).open("rb") as handle:
        for record in ordered:
            handle.seek(record["offset"])
            yield validate_raw_trajectory(json.loads(handle.readline()))


def hard_acceptance_reasons(
    trajectory: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
) -> list[str]:
    """Return deterministic reasons a trajectory cannot enter SFT."""

    reasons: list[str] = []
    if not strict_gold_success(trajectory):
        reasons.append("not_strict_gold")
    if trajectory.get("runtime_contract_sha256") != expected_contract_sha256:
        reasons.append("runtime_contract_mismatch")
    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("release_error"):
        reasons.append("release_error")
    if trajectory.get("released") is not True:
        reasons.append("lease_not_released")

    events = trajectory.get("events") or []
    buy_steps = [
        event
        for event in events
        if event.get("type") == "environment_step"
        and event.get("tool_name") == "buy_now"
        and event.get("done") is True
    ]
    if not buy_steps:
        reasons.append("missing_executed_terminal_buy")

    known_tools = {
        schema["function"]["name"]
        for schema in SHOP_TOOL_SCHEMAS
        if isinstance(schema, dict) and isinstance(schema.get("function"), dict)
    }
    for event in events:
        if event.get("type") == "assistant":
            if event.get("dropped_tool_calls"):
                reasons.append("parallel_tool_calls_dropped")
            message = event.get("message") or {}
            calls = message.get("tool_calls") or []
            if len(calls) > 1:
                reasons.append("multiple_tool_calls")
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name")
                if name not in known_tools:
                    reasons.append("unknown_tool")
                arguments = function.get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(arguments) if isinstance(arguments, str) else arguments
                    )
                except json.JSONDecodeError:
                    reasons.append("malformed_tool_arguments")
                    continue
                if not isinstance(arguments, dict):
                    reasons.append("tool_arguments_not_object")
        elif event.get("type") == "guard_rejection":
            reason = str(event.get("reason") or "")
            if reason.startswith(("malformed_tool_call", "schema_")):
                reasons.append("damaged_tool_structure")

    return sorted(set(reasons))


def build_action_only_sft_row(
    trajectory: Mapping[str, Any],
    *,
    system_prompt: str,
    user_instruction: str | None = None,
) -> dict:
    """Reconstruct only Actor-visible chat messages from the event stream."""

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    previous_call: dict | None = None
    reset_seen = False
    terminal_seen = False

    for event in trajectory.get("events") or []:
        event_type = event.get("type")
        if event_type == "environment_reset":
            if reset_seen:
                raise ValueError("trajectory contains multiple environment resets")
            messages.append(
                {
                    "role": "user",
                    "content": str(
                        user_instruction
                        if user_instruction is not None
                        else event.get("actor_visible_observation") or ""
                    ),
                }
            )
            reset_seen = True
            continue
        if event_type == "assistant":
            message = event.get("message") or {}
            clean = {
                key: deepcopy(message[key])
                for key in ("role", "content", "tool_calls")
                if key in message
            }
            clean["role"] = "assistant"
            messages.append(clean)
            calls = clean.get("tool_calls") or []
            previous_call = calls[0] if len(calls) == 1 else None
            continue
        if event_type not in {"guard_rejection", "environment_step"}:
            continue
        if previous_call is None:
            raise ValueError(f"{event_type} has no preceding serial tool call")
        function = previous_call.get("function") or {}
        name = str(function.get("name") or event.get("tool_name") or "")
        content = str(event.get("actor_visible_observation") or "")
        is_terminal = (
            event_type == "environment_step"
            and event.get("tool_name") == "buy_now"
            and event.get("done") is True
        )
        if is_terminal:
            content = TERMINAL_TOOL_CONTENT
            terminal_seen = True
        messages.append(
            {
                "role": "tool",
                "tool_call_id": previous_call.get("id"),
                "name": name,
                "content": content,
            }
        )
        previous_call = None

    if not reset_seen:
        raise ValueError("trajectory has no Actor-visible reset")
    if not terminal_seen:
        raise ValueError("trajectory has no executed terminal buy")
    if previous_call is not None:
        raise ValueError("trajectory ends with an unmatched tool call")
    return {
        "schema_version": SFT_DATASET_VERSION,
        "trajectory_id": trajectory["request_id"],
        "task_id": int(trajectory["task_id"]),
        "messages": messages,
        "tools": deepcopy(SHOP_TOOL_SCHEMAS),
    }


def _stable_order(seed: int, namespace: str, task_id: int) -> str:
    return sha256_bytes(f"{seed}:{namespace}:{task_id}".encode("utf-8"))


def _hamilton_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total < 0 or total > available:
        raise ValueError("requested quota exceeds available tasks")
    if available == 0:
        return {key: 0 for key in counts}
    exact = {key: total * value / available for key, value in counts.items()}
    quotas = {key: min(counts[key], int(exact[key])) for key in counts}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - int(exact[key])), key),
    )
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("unable to allocate deterministic strata quotas")
    return quotas


def assign_stratified_splits(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    train_count: int,
    dev_count: int,
    seed: int = DEFAULT_CURATION_SEED,
) -> dict[int, str]:
    """Assign tasks using only intrinsic difficulty, task ID and a fixed seed."""

    by_difficulty: dict[str, list[int]] = defaultdict(list)
    for row in task_rows:
        by_difficulty[str(row.get("difficulty") or "unknown")].append(
            int(row["task_id"])
        )
    for difficulty, task_ids in by_difficulty.items():
        task_ids.sort(key=lambda task_id: _stable_order(seed, difficulty, task_id))

    train_quota = _hamilton_quotas(
        {key: len(value) for key, value in by_difficulty.items()}, train_count
    )
    assignment: dict[int, str] = {}
    remaining: dict[str, list[int]] = {}
    for difficulty, task_ids in sorted(by_difficulty.items()):
        cut = train_quota[difficulty]
        for task_id in task_ids[:cut]:
            assignment[task_id] = "train"
        remaining[difficulty] = task_ids[cut:]

    dev_quota = _hamilton_quotas(
        {key: len(value) for key, value in remaining.items()}, dev_count
    )
    for difficulty, task_ids in sorted(remaining.items()):
        cut = dev_quota[difficulty]
        for task_id in task_ids[:cut]:
            assignment[task_id] = "dev"
        for task_id in task_ids[cut:]:
            assignment[task_id] = "reserve"
    return assignment


def select_outcome_and_process(
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    outcome = min(candidates, key=lambda pair: int(pair[0]["attempt_index"]))
    process = min(
        candidates,
        key=lambda pair: (tuple(pair[1]["selection_tuple"]), pair[0]["request_id"]),
    )
    return outcome[0], process[0]


def count_by_split_and_difficulty(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row["split"])][str(row["difficulty"])] += 1
    return {
        split: dict(sorted(counter.items()))
        for split, counter in sorted(counts.items())
    }


def manifest_hash(value: Mapping[str, Any]) -> str:
    clean = dict(value)
    clean.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(clean))
