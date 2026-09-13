"""Deterministic process features computed only from Actor-visible events."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from commerce_posttrain.collection.schema import (
    canonical_json_bytes,
    strict_gold_success,
    validate_raw_trajectory,
)
from commerce_posttrain.curation.process_contract import (
    EVIDENCE_SOURCE,
    PROCESS_CONTRACT_VERSION,
    PROCESS_SELECTION_ORDER,
)
from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE

PROCESS_ANALYZER_VERSION = "commerce-process-analyzer-v1"
PROCESS_FEATURES_VERSION = "commerce-process-features-v1"
_PAGE_TYPE = re.compile(r"(?m)^page_type:\s*([^\n]+)")
_SUBPAGE = re.compile(r"(?m)^subpage:\s*([^\n]+)")
_PRICE = re.compile(r"(?m)^price:\s*(?!none\b|null\b|unknown\b)(\S.+)$", re.I)
_SELECTED_OPTIONS = re.compile(r"(?m)^selected_options:\s*(\{.*\})$")
_MODEL_OR_SPEC = re.compile(
    r"型号|规格|尺寸|尺码|材质|颜色|容量|净含量|版本|款式|套装|套餐|\b[a-z]+[-+._]?\d+",
    re.I,
)
_FUNCTION = re.compile(r"功能|支持|防水|续航|性能|功率|接口|兼容|适合|用于")
_REVIEW = re.compile(r"评价|评论|口碑|体验|反馈|耐用|噪音|舒适|好用")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def compile_required_detail_types(query: str) -> tuple[str, ...]:
    """Compile evidence pages only from the same user query the Actor receives."""

    required = {"product_detail"}
    if _MODEL_OR_SPEC.search(query):
        required.add("attributes")
    if _FUNCTION.search(query):
        required.add("features")
    if _REVIEW.search(query):
        required.add("reviews")
    return tuple(sorted(required))


def _action_signature(event: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "tool_name": event.get("tool_name"),
                "parameters": event.get("parameters") or {},
            }
        )
    )


def _visible_event_digest(events: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {
            "type": event.get("type"),
            "tool_name": event.get("tool_name"),
            "parameters": event.get("parameters") or {},
            "reason": event.get("reason"),
            "actor_visible_observation": event.get("actor_visible_observation"),
            "projection": event.get("projection") or {},
            "input_tokens": event.get("input_tokens"),
        }
        for event in events
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def _detail_type(observation: str) -> str | None:
    page = _PAGE_TYPE.search(observation)
    page_type = page.group(1).strip().casefold() if page else ""
    if page_type == "product_detail":
        return "product_detail"
    if page_type == "information_subpage":
        subpage = _SUBPAGE.search(observation)
        return subpage.group(1).strip().casefold() if subpage else "information_subpage"
    return None


def _selected_option_count(observation: str) -> int:
    match = _SELECTED_OPTIONS.search(observation)
    if not match:
        return 0
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return 0
    return len(value) if isinstance(value, dict) else 0


def _strict_outcome(trajectory: Mapping[str, Any]) -> dict:
    terminal = trajectory.get("terminal_environment_result") or {}
    reward = terminal.get("reward_detail") or {}
    return {
        "strict_gold_success": strict_gold_success(trajectory),
        "reward_type": reward.get("reward_type", "none"),
        "reward_valid": reward.get("reward_valid") is True,
        "status": trajectory.get("status"),
    }


def analyze_trajectory(
    trajectory: Mapping[str, Any],
    *,
    query: str,
) -> dict:
    """Analyze process without reading hidden results; outcome is a separate field."""

    trajectory = validate_raw_trajectory(trajectory)
    events = trajectory["events"]
    step_events = [event for event in events if event.get("type") == "environment_step"]
    guard_events = [event for event in events if event.get("type") == "guard_rejection"]
    assistant_events = [event for event in events if event.get("type") == "assistant"]

    action_signatures = [_action_signature(event) for event in step_events]
    action_counts = Counter(action_signatures)
    repeat_actions = sum(count - 1 for count in action_counts.values() if count > 1)
    search_queries = [
        normalize_text((event.get("parameters") or {}).get("query"))
        for event in step_events
        if event.get("tool_name") == "search_products"
    ]
    search_queries = [query for query in search_queries if query]
    unique_queries = len(set(search_queries))
    duplicate_queries = len(search_queries) - unique_queries
    opened = [
        normalize_text((event.get("parameters") or {}).get("asin"))
        for event in step_events
        if event.get("tool_name") == "open_product"
    ]
    opened = [asin for asin in opened if asin]
    observations = [str(event.get("actor_visible_observation") or "") for event in step_events]
    visible_candidates = {
        asin
        for observation in observations
        for asin in re.findall(rf"(?m)^\d+\|({PRODUCT_ID_CAPTURE})\|", observation)
    }

    required_types = set(compile_required_detail_types(query))
    detail_types_seen: set[str] = set()
    maximum_option_axes = 0
    final_price_visible = False
    observation_hashes = []
    no_progress_actions = 0
    decision_ready_step = None
    for step_index, observation in enumerate(observations):
        detail = _detail_type(observation)
        if detail:
            detail_types_seen.add(detail)
        maximum_option_axes = max(maximum_option_axes, _selected_option_count(observation))
        price_visible = bool(_PRICE.search(observation))
        final_price_visible = final_price_visible or price_visible
        digest = sha256_bytes(observation.encode("utf-8"))
        if observation_hashes and digest == observation_hashes[-1]:
            no_progress_actions += 1
        observation_hashes.append(digest)
        buy_visible = "\"Buy Now\"" in observation or "\"buy now\"" in observation.casefold()
        if (
            decision_ready_step is None
            and buy_visible
            and price_visible
            and required_types <= detail_types_seen
        ):
            decision_ready_step = step_index
    if decision_ready_step is None:
        steps_after_ready = 0
    else:
        steps_after_ready = sum(
            1
            for event in step_events[decision_ready_step + 1 :]
            if event.get("tool_name") != "buy_now"
        )

    projection_truncations = sum(
        bool((event.get("projection") or {}).get("truncated")) for event in step_events
    )
    contract_violations = sum(
        not bool((event.get("projection") or {}).get("critical_footer_preserved", True))
        or int((event.get("projection") or {}).get("visible_tokens", 0))
        > int((event.get("projection") or {}).get("token_budget", 10**12))
        for event in step_events
    )
    input_tokens = [
        int(event["input_tokens"])
        for event in assistant_events
        if event.get("input_tokens") is not None
    ]
    malformed_calls = sum(
        str(event.get("reason", "")).startswith("malformed_tool_call")
        for event in guard_events
    )
    schema_rejections = sum(
        str(event.get("reason", "")).startswith("schema_")
        for event in guard_events
    )
    missing = sorted(required_types - detail_types_seen)
    tool_names = [str(event.get("tool_name") or "") for event in step_events]
    features = {
        "legality": {
            "guard_rejections": len(guard_events),
            "malformed_calls": malformed_calls,
            "schema_rejections": schema_rejections,
        },
        "search": {
            "unique_queries": unique_queries,
            "duplicate_queries": duplicate_queries,
            "visible_candidates": len(visible_candidates),
        },
        "candidate_use": {
            "opened_candidates": len(set(opened)),
            "compared_candidates": len(set(opened)),
        },
        "evidence": {
            "required_detail_types_seen": sorted(required_types & detail_types_seen),
            "missing_required_detail_types": missing,
            "option_axes_resolved": maximum_option_axes,
            "final_price_visible": final_price_visible,
        },
        "termination": {
            "repeat_actions": repeat_actions,
            "no_progress_actions": no_progress_actions,
            "steps_after_decision_ready": steps_after_ready,
            "decision_ready_step": decision_ready_step,
        },
        "context": {
            "projection_truncations": projection_truncations,
            "max_input_tokens": max(input_tokens, default=0),
            "contract_violations": contract_violations,
        },
        "diversity": {
            "tool_sequence_signature": sha256_bytes(canonical_json_bytes(tool_names)),
            "search_query_signature": sha256_bytes(canonical_json_bytes(search_queries)),
            "visited_product_signature": sha256_bytes(canonical_json_bytes(sorted(set(opened)))),
            "first_divergence_turn": None,
        },
    }
    flat_selection = {
        **features["legality"],
        "missing_required_detail_types": len(missing),
        **features["termination"],
        "attempt_index": int(trajectory["attempt_index"]),
    }
    selection_tuple = [flat_selection[name] for name in PROCESS_SELECTION_ORDER]
    result = {
        "schema_version": PROCESS_FEATURES_VERSION,
        "analyzer_version": PROCESS_ANALYZER_VERSION,
        "process_contract_version": PROCESS_CONTRACT_VERSION,
        "evidence_source": EVIDENCE_SOURCE,
        "request_id": trajectory["request_id"],
        "trajectory_sha256": trajectory["trajectory_sha256"],
        "task_id": int(trajectory["task_id"]),
        "attempt_index": int(trajectory["attempt_index"]),
        "actor_visible_input_sha256": _visible_event_digest(events),
        "features": features,
        "selection_tuple": selection_tuple,
        "outcome_private_verifier": _strict_outcome(trajectory),
    }
    result["process_features_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def attach_first_divergence(
    records: Sequence[dict],
    trajectories: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Attach same-task behavioral divergence without changing other features."""

    if not records:
        return []
    if len(records) != len(trajectories):
        raise ValueError("records and trajectories must have equal length")
    sequences = []
    for trajectory in trajectories:
        sequences.append(
            [
                _action_signature(event)
                for event in trajectory.get("events") or []
                if event.get("type") == "environment_step"
            ]
        )
    divergence = None
    for turn in range(max((len(sequence) for sequence in sequences), default=0)):
        values = {
            sequence[turn] if turn < len(sequence) else "<missing>"
            for sequence in sequences
        }
        if len(values) > 1:
            divergence = turn
            break
    updated = []
    for record in records:
        value = json.loads(json.dumps(record))
        value["features"]["diversity"]["first_divergence_turn"] = divergence
        value.pop("process_features_sha256", None)
        value["process_features_sha256"] = sha256_bytes(canonical_json_bytes(value))
        updated.append(value)
    return updated
