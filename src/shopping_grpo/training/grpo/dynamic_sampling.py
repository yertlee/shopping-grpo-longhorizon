"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_STOP_CHECKPOINT_ARTIFACT_GLOBS = (
    "actor/model_world_size_*_rank_*.pt",
    "actor/optim_world_size_*_rank_*.pt",
    "actor/extra_state_world_size_*_rank_*.pt",
)


def parse_stop_after_step(raw: str | None) -> int | None:
    """Parse the launcher-provided natural-stop target without accepting ambiguity."""
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("SHOPPING_GRPO_STOP_AFTER_STEP must be a positive integer") from exc
    if value <= 0 or str(value) != raw.strip():
        raise ValueError("SHOPPING_GRPO_STOP_AFTER_STEP must be a positive integer")
    return value


def validate_stop_after_step(target_step: int | None, save_freq: int) -> int | None:
    """Require an exact checkpoint boundary; never silently stop without a checkpoint."""
    if target_step is None:
        return None
    if save_freq <= 0 or target_step % save_freq:
        raise ValueError(
            "stop-after-step must be an exact positive checkpoint boundary: "
            f"target={target_step}, save_freq={save_freq}"
        )
    return target_step


def should_controlled_stop(
    *, target_step: int | None, current_step: int, optimizer_updated: bool, save_freq: int
) -> bool:
    """Return true only after an optimizer update at an exact saved checkpoint step."""
    target = validate_stop_after_step(target_step, save_freq)
    return bool(
        target is not None
        and optimizer_updated
        and current_step == target
        and current_step % save_freq == 0
    )


def validate_saved_checkpoint(
    output: str | Path,
    *,
    global_step: int,
    artifact_globs: Sequence[str] = DEFAULT_STOP_CHECKPOINT_ARTIFACT_GLOBS,
) -> dict[str, object]:
    """Prove the synchronous veRL checkpoint is complete before emitting a stop marker."""
    root = Path(output).expanduser().resolve()
    checkpoint = root / f"global_step_{int(global_step)}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"controlled stop checkpoint directory is missing: {checkpoint}")
    missing: list[str] = []
    files: list[str] = []
    for pattern in artifact_globs:
        matches = sorted(
            item for item in checkpoint.glob(pattern) if item.is_file()
        )
        if not matches:
            missing.append(pattern)
        files.extend(str(item.relative_to(root)) for item in matches)
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"controlled stop checkpoint tracker is missing: {tracker}")
    try:
        tracker_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"controlled stop checkpoint tracker is unreadable: {tracker}") from exc
    if tracker_step != int(global_step):
        raise RuntimeError(
            "controlled stop checkpoint tracker does not match target: "
            f"tracker={tracker_step}, target={global_step}"
        )
    if missing:
        raise RuntimeError(
            "controlled stop checkpoint is incomplete; missing artifacts: "
            + ", ".join(missing)
        )
    return {"directory": str(checkpoint), "tracker": str(tracker), "files": files}


def record_controlled_stop_after_checkpoint(
    marker_path: str | Path,
    diagnostics_path: str | Path | None,
    *,
    global_step: int,
    target_step: int,
    save_freq: int,
) -> dict[str, object]:
    """Persist the natural-stop marker and diagnostic after synchronous checkpoint save."""
    destination = Path(marker_path)
    checkpoint = validate_saved_checkpoint(destination.parent, global_step=global_step)
    record: dict[str, object] = {
        "schema_version": 1,
        "event": "controlled_stop_after_checkpoint",
        "global_step": int(global_step),
        "target_step": int(target_step),
        "save_freq": int(save_freq),
        "checkpoint_boundary": True,
        "checkpoint": checkpoint,
    }
    append_training_diagnostic(
        diagnostics_path,
        "controlled_stop_after_checkpoint",
        int(global_step),
        target_step=int(target_step),
        save_freq=int(save_freq),
        checkpoint_boundary=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return record


def build_rollout_diagnostics(
    uids: Sequence[Hashable], shopping_infos: Sequence[object]
) -> list[dict[str, Any]]:
    """Attach stable group/rollout identities to public AgentLoop diagnostics."""
    if len(uids) != len(shopping_infos):
        raise ValueError("uids and shopping_infos must have equal length")
    rollout_counts: dict[Hashable, int] = {}
    records = []
    for index, (uid, info) in enumerate(zip(uids, shopping_infos, strict=True)):
        if not isinstance(info, Mapping):
            raise ValueError(f"shopping extra field at index {index} is not an object")
        rollout_index = rollout_counts.get(uid, 0)
        rollout_counts[uid] = rollout_index + 1
        records.append({"uid": uid, "rollout_index": rollout_index, **dict(info)})
    return records


def append_training_diagnostic(
    path: str | Path | None,
    event: str,
    global_step: int,
    **payload: object,
) -> None:
    """Append one driver-side training event; an unset path disables persistence."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def scalar(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    record = {
        "schema_version": 1,
        "event": str(event),
        "global_step": int(global_step),
        **payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=scalar))
        handle.write("\n")


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    overlong = []
    repeat_loop = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        overlong.append(float(bool(info.get("overlong"))))
        repeat_loop.append(float(info.get("reward_type") == "repeat_loop"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )

    def mean(values):
        return sum(values) / len(values)

    return {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/overlong_rate": mean(overlong),
        "trajectory/repeat_loop_rate": mean(repeat_loop),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return terminal utility, success metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        if bool(info.get("overlong")):
            reasons.append("overlong")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append("reward_sampling_invalid")
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    utility_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            utility_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        utilities = group["terminal_utilities"]
        utility_min = min(utilities)
        utility_max = max(utilities)
        utility_varying = utility_max - utility_min > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        if has_sampling_invalid:
            drop_reason = "sampling_invalid"
        elif not utility_varying:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(utilities),
                "purchase_success": tuple(group["purchase_success"]),
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_varying": utility_varying,
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats



def cuda_memory_snapshot(stage: str) -> dict[str, Any]:
    """Return driver-process CUDA allocator state for a named trainer stage."""
    base = {"stage": str(stage), "process_scope": "driver"}
    try:
        import torch
    except ImportError:
        return {**base, "cuda_available": False, "status": "torch_unavailable"}
    if not torch.cuda.is_available():
        return {**base, "cuda_available": False, "status": "cuda_unavailable"}
    device = torch.cuda.current_device()
    return {
        **base, "cuda_available": True, "status": "ok", "device": int(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def record_actor_batch_pre_update(
    batch: object, path: str | Path | None, global_step: int,
    *, stage: str = "actor_batch_pre_update",
) -> None:
    """Append batch structure/token counts without token contents; never raise."""
    if not path:
        return
    try:
        tensor_batch = getattr(batch, "batch", {})
        shapes = {}
        for name in ("input_ids", "attention_mask", "responses", "response_mask", "prompt_mask"):
            value = tensor_batch.get(name) if hasattr(tensor_batch, "get") else None
            if value is not None and hasattr(value, "shape"):
                shapes[name] = [int(dim) for dim in value.shape]
        attention = tensor_batch.get("attention_mask") if hasattr(tensor_batch, "get") else None
        responses = tensor_batch.get("responses") if hasattr(tensor_batch, "get") else None
        response_mask = tensor_batch.get("response_mask") if hasattr(tensor_batch, "get") else None
        prompt_mask = tensor_batch.get("prompt_mask") if hasattr(tensor_batch, "get") else None
        reason = None
        if attention is None:
            reason = "missing_attention_mask"
        elif responses is None:
            reason = "missing_responses"
        elif attention.ndim != 2 or responses.ndim < 2 or attention.shape[0] != responses.shape[0]:
            reason = "incompatible_attention_responses_rows"
        elif response_mask is None:
            width = int(responses.shape[-1])
            if attention.shape[1] < width:
                reason = "responses_width_exceeds_attention_width"
            else:
                response_mask = attention[:, -width:]
                response_mask_source = "attention_mask_suffix"
        else:
            response_mask_source = "response_mask"
        if reason is None and response_mask is not None and response_mask.shape[0] != attention.shape[0]:
            reason = "incompatible_response_mask_rows"
        if reason is None:
            attention_counts = [int(row.sum().item()) for row in attention.detach().cpu()]
            response_counts = [int(row.sum().item()) for row in response_mask.detach().cpu()]
            if prompt_mask is not None and prompt_mask.shape[0] != attention.shape[0]:
                reason = "incompatible_prompt_mask_rows"
                prompt_counts = []
            else:
                prompt_counts = ([int(row.sum().item()) for row in prompt_mask.detach().cpu()]
                                 if prompt_mask is not None else
                                 [max(a - r, 0) for a, r in zip(attention_counts, response_counts, strict=True)])
        else:
            attention_counts = response_counts = prompt_counts = []
        non_tensor = getattr(batch, "non_tensor_batch", {})
        def vals(name: str) -> list[object]:
            value = non_tensor.get(name) if hasattr(non_tensor, "get") else None
            return [] if value is None else (value.tolist() if hasattr(value, "tolist") else list(value))
        uids = vals("uid")
        task_ids = vals("task_id")
        if not task_ids:
            task_ids = [item.get("task_id") if isinstance(item, Mapping) else None for item in vals("shopping")]
        rows = [
            {"attention_tokens": a, "prompt_tokens": p, "response_tokens": r}
            for a, p, r in zip(attention_counts, prompt_counts, response_counts, strict=True)
        ]
        payload = {
            "batch_shapes": shapes,
            "accepted_uids": uids,
            "accepted_task_ids": task_ids,
            "rows": rows,
            "summary": {
                "batch_size": len(rows),
                "attention_tokens_total": sum(attention_counts),
                "prompt_tokens_total": sum(prompt_counts),
                "response_tokens_total": sum(response_counts),
            },
            "availability": "available" if reason is None else "unavailable",
            "reason": reason,
            "response_mask_source": None if reason else response_mask_source,
            "memory": cuda_memory_snapshot(stage),
        }
        append_training_diagnostic(path, "actor_batch_pre_update", global_step, **payload)
    except Exception as exc:
        append_training_diagnostic(
            path, "actor_batch_pre_update", global_step,
            availability="unavailable", reason=f"diagnostic_error:{type(exc).__name__}",
            memory=cuda_memory_snapshot(stage),
        )
