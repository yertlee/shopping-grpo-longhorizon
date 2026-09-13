"""Shared, resumable ShopSimulator actor loop for Teacher collection."""

from __future__ import annotations

import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from commerce_posttrain.collection.schema import (
    RAW_SCHEMA_VERSION,
    finalize_trajectory,
    request_identity,
)
from shopping_grpo.environment.actions import (
    action_guard_tool_message,
    action_reject_reason,
)
from shopping_grpo.environment.client import (
    ShopAgentEnv,
    ShopEnvironmentError,
    ShopHttpError,
)
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action

MAX_BLOCKED_CALLS = 3


def _tool_name_arguments(tool_call: Mapping[str, Any]) -> tuple[str, dict]:
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments") or "{}"
    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(name, str) or not name:
        raise ValueError("tool call has no function name")
    return name, arguments


def _visible_observation(result: Mapping[str, Any]) -> str:
    state = result.get("observation_state")
    if state is not None:
        return render_structured_observation(state)
    return str(result.get("instruction", result.get("observation", "")))


def _is_infrastructure_error(exc: BaseException) -> bool:
    if isinstance(exc, (ShopHttpError, TimeoutError, OSError)):
        return True
    return isinstance(exc, ShopEnvironmentError) and (
        "Unable to get available environment resource" in str(exc)
    )


def _add_event(trajectory: dict, event: dict) -> None:
    event = dict(event)
    event["event_index"] = len(trajectory["events"])
    trajectory["events"].append(event)


def collect_attempt(
    task: Mapping[str, Any],
    *,
    attempt_index: int,
    runtime_contract: Mapping[str, Any],
    client,
    base_url: str,
    system_prompt: str,
    env_factory=ShopAgentEnv,
) -> dict:
    task_id = int(task["task_id"])
    teacher = dict(runtime_contract["teacher"])
    request_id = request_identity(
        task_id=task_id,
        attempt_index=attempt_index,
        runtime_contract_sha256=runtime_contract["contract_sha256"],
        teacher=teacher,
    )
    trajectory = {
        "schema_version": RAW_SCHEMA_VERSION,
        "request_id": request_id,
        "execution_id": f"{request_id}:{datetime.now(timezone.utc).isoformat()}",
        "task_id": task_id,
        "attempt_index": int(attempt_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_contract_sha256": runtime_contract["contract_sha256"],
        "teacher": teacher,
        "status": "running",
        "events": [],
        "terminal_environment_result": {},
        "done": False,
        "released": False,
        "error": None,
        "release_error": None,
    }
    env = env_factory(base_url=base_url)
    messages = [{"role": "system", "content": system_prompt}]
    latest_observation = ""
    blocked_count = 0
    executed_steps = 0
    try:
        initial = env.reset(task_id)
        latest_observation = _visible_observation(initial)
        messages.append({"role": "user", "content": str(initial.get("instruction") or "")})
        _add_event(
            trajectory,
            {
                "type": "environment_reset",
                "actor_visible_observation": latest_observation,
                "environment_result_private": deepcopy(initial),
            },
        )
        while executed_steps < int(runtime_contract["max_steps"]):
            input_tokens = client.count_chat_tokens(messages, SHOP_TOOL_SCHEMAS)
            if (
                input_tokens
                + int(runtime_contract["max_generated_tokens_per_turn"])
                + int(runtime_contract["context_safety_margin"])
                > int(runtime_contract["context_window"])
            ):
                raise ValueError(f"context budget exceeded before step {executed_steps}")
            assistant = client.complete(messages, SHOP_TOOL_SCHEMAS)
            tool_calls = assistant.get("tool_calls") or []
            dropped = deepcopy(tool_calls[1:])
            if len(tool_calls) > 1:
                assistant = dict(assistant)
                assistant["tool_calls"] = tool_calls[:1]
                tool_calls = tool_calls[:1]
            _add_event(
                trajectory,
                {
                    "type": "assistant",
                    "executed_step_index": executed_steps,
                    "message": deepcopy(assistant),
                    "dropped_tool_calls": dropped,
                    "input_tokens": input_tokens,
                    "provider": deepcopy(getattr(client, "last_metadata", {})),
                },
            )
            messages.append(assistant)
            if not tool_calls:
                trajectory["status"] = "assistant_final"
                break
            tool_call = tool_calls[0]
            try:
                name, arguments = _tool_name_arguments(tool_call)
                reject_reason = action_reject_reason(name, arguments, latest_observation)
            except Exception as exc:
                name, arguments = "invalid", {}
                reject_reason = f"malformed_tool_call:{exc.__class__.__name__}"
            if reject_reason:
                blocked_count += 1
                guard_message = action_guard_tool_message(
                    tool_call, reject_reason, latest_observation
                )
                _add_event(
                    trajectory,
                    {
                        "type": "guard_rejection",
                        "executed_step_index": executed_steps,
                        "reason": reject_reason,
                        "tool_call": deepcopy(tool_call),
                        "actor_visible_observation": guard_message["content"],
                    },
                )
                messages.append(guard_message)
                if blocked_count >= MAX_BLOCKED_CALLS:
                    trajectory["status"] = "invalid_action_limit"
                    break
                continue
            blocked_count = 0
            action = tool_call_to_action(name, arguments)
            result = env.step(action)
            raw_observation = _visible_observation(result)
            latest_observation, projection = project_observation(
                name,
                raw_observation,
                count_tokens=client.count_text_tokens,
                token_budget=runtime_contract["observation_search_tokens"],
                detail_token_budget=runtime_contract["observation_detail_tokens"],
                generic_token_budget=runtime_contract["observation_generic_tokens"],
                parameters=arguments,
                search_top_k=runtime_contract["observation_search_top_k"],
            )
            _add_event(
                trajectory,
                {
                    "type": "environment_step",
                    "executed_step_index": executed_steps,
                    "tool_call": deepcopy(tool_call),
                    "tool_name": name,
                    "parameters": deepcopy(arguments),
                    "env_action": action,
                    "environment_observation_raw": raw_observation,
                    "actor_visible_observation": latest_observation,
                    "projection": projection.to_dict(),
                    "environment_result_private": deepcopy(result),
                    "reward": float(result.get("reward", 0.0)),
                    "done": bool(result.get("done", False)),
                },
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": latest_observation,
                }
            )
            executed_steps += 1
            if result.get("done"):
                trajectory["status"] = "done"
                trajectory["done"] = True
                trajectory["terminal_environment_result"] = deepcopy(result)
                break
        else:
            trajectory["status"] = "max_steps"
    except Exception as exc:
        trajectory["status"] = (
            "infrastructure_failure" if _is_infrastructure_error(exc) else "policy_error"
        )
        trajectory["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _add_event(
            trajectory,
            {
                "type": "attempt_error",
                "stage": "actor_loop",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            },
        )
    finally:
        try:
            env.release()
            trajectory["released"] = True
        except Exception as exc:
            trajectory["status"] = "infrastructure_failure"
            trajectory["release_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            trajectory["error"] = trajectory["error"] or trajectory["release_error"]
    return finalize_trajectory(trajectory)
