"""Deterministic oracle replay against the locked ShopSimulator HTTP service."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any, Mapping, Sequence

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.observation import render_structured_observation

REPLAY_SCHEMA_VERSION = "commerce-shopsimulator-replay-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def semantic_result(result: Mapping[str, Any]) -> dict:
    state = result.get("observation_state")
    visible = render_structured_observation(state) if state is not None else str(
        result.get("instruction", result.get("observation", ""))
    )
    reward = result.get("reward_detail") or {}
    purchase = result.get("purchase") or {}
    return {
        "visible_observation": visible,
        "reward": float(result.get("reward", 0.0)),
        "done": bool(result.get("done", False)),
        "over": bool(result.get("over", False)),
        "reward_detail": {
            key: reward.get(key)
            for key in (
                "reward_version",
                "reward_type",
                "reward_valid",
                "purchase_success",
                "termination_reason",
                "terminal_utility",
            )
            if key in reward
        },
        "purchase": {
            key: purchase.get(key)
            for key in ("asin", "price", "options")
            if key in purchase
        },
    }


def _strict_success(result: Mapping[str, Any]) -> bool:
    reward = result.get("reward_detail") or {}
    return (
        result.get("done") is True
        and result.get("over") is True
        and reward.get("reward_version") == "shopsimulator-reward-v3"
        and reward.get("reward_type") == "gold_purchase"
        and reward.get("reward_valid") is True
        and reward.get("purchase_success") is True
        and reward.get("termination_reason") == "gold_purchase"
    )


def oracle_actions(task: Mapping[str, Any]) -> list[str]:
    title = str(task.get("target_title") or "").strip()
    targets = task.get("target_product_ids") or []
    if not title or len(targets) != 1:
        raise ValueError("oracle replay requires target_title and one target product")
    actions = [f"search[{title}]", f"click[{targets[0]}]"]
    actions.extend(f"click[{value}]" for value in task.get("target_options") or [])
    actions.append("click[Buy Now]")
    return actions


def run_oracle_case(
    task: Mapping[str, Any],
    *,
    base_url: str,
    barrier: Barrier | None = None,
    env_factory=ShopAgentEnv,
) -> dict:
    env = env_factory(base_url=base_url)
    task_id = int(task["task_id"])
    steps = []
    env_idx = None
    try:
        reset = env.reset(task_id)
        env_idx = reset.get("env_idx")
        if barrier is not None:
            barrier.wait(timeout=60)
        for action in oracle_actions(task):
            result = env.step(action)
            steps.append({"action": action, "result": semantic_result(result)})
            if result.get("done"):
                break
        terminal = steps[-1]["result"] if steps else {}
        raw_terminal = result if steps else {}
        if not _strict_success(raw_terminal):
            raise AssertionError(
                f"task {task_id} oracle replay did not reach strict gold success: {terminal}"
            )
        semantic = {
            "task_id": task_id,
            "initial_instruction": str(reset.get("instruction") or ""),
            "steps": steps,
        }
        return {
            "task_id": task_id,
            "leased_env_idx": env_idx,
            "semantic": semantic,
            "semantic_sha256": sha256_bytes(canonical_json_bytes(semantic)),
        }
    finally:
        env.release()


def run_replay_suite(
    tasks: Sequence[Mapping[str, Any]],
    *,
    base_url: str,
    workers: int,
    repeats: int,
    runtime_contract_hash: str,
    reference: Mapping[int, str] | None = None,
) -> dict:
    if workers < 1 or repeats < 1:
        raise ValueError("workers and repeats must be positive")
    if workers > 1 and len(tasks) < workers:
        raise ValueError("concurrent replay needs at least one task per worker")
    cases = []
    for repeat_index in range(repeats):
        barrier = Barrier(workers) if workers > 1 else None
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_oracle_case,
                    task,
                    base_url=base_url,
                    barrier=barrier,
                )
                for task in tasks
            ]
            batch = [future.result() for future in futures]
        if workers > 1:
            leases = [case["leased_env_idx"] for case in batch]
            if len(set(leases)) != len(leases):
                raise AssertionError("concurrent replays shared a ShopSimulator lease")
        for case in batch:
            case["repeat_index"] = repeat_index
            cases.append(case)

    signatures: dict[int, set[str]] = {}
    for case in cases:
        signatures.setdefault(case["task_id"], set()).add(case["semantic_sha256"])
    unstable = sorted(task_id for task_id, values in signatures.items() if len(values) != 1)
    if unstable:
        raise AssertionError(f"non-deterministic replay tasks: {unstable}")
    resolved = {task_id: next(iter(values)) for task_id, values in signatures.items()}
    if reference is not None:
        mismatched = sorted(
            task_id
            for task_id, signature in resolved.items()
            if reference.get(task_id) != signature
        )
        if mismatched:
            raise AssertionError(f"replay differs from single-worker reference: {mismatched}")
    report = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "runtime_contract_sha256": runtime_contract_hash,
        "base_url": base_url,
        "workers": workers,
        "repeats": repeats,
        "task_count": len(tasks),
        "case_count": len(cases),
        "all_strict_gold_success": True,
        "all_deterministic": True,
        "reference_match": reference is not None,
        "signatures": {str(key): value for key, value in sorted(resolved.items())},
        "cases": cases,
    }
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def load_private_tasks(path: str | Path) -> dict[int, dict]:
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["task_id"])] = row
    return rows
