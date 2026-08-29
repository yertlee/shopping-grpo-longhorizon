"""Paired Baseline/SFT/GRPO diagnostics without a composite score."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION, JUDGE_DIMENSIONS
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION
from shopping_grpo.evaluation.rubric import stable_hash


COMPARISON_SCHEMA_VERSION = "shopping-paired-model-comparison-v1"
PAIRED_STATISTICS_SCHEMA_VERSION = "shopping-paired-statistics-v1"

# 二元端点：exact McNemar 检验的默认指标（Reward 面板，固定分母语义）。
BINARY_ENDPOINTS = (
    "strict_gold_success",
    "purchase_success",
    "reward_valid",
)
# 次要连续端点：paired bootstrap 95% CI；与二元端点一一分开报告，禁止合成总分。
CONTINUOUS_ENDPOINTS = (
    "final_reward",
    "terminal_utility",
    "executed_tool_steps",
    "guard_rejections",
    "duplicate_canonical_actions",
) + tuple(f"dimension_{name}" for name in JUDGE_DIMENSIONS)


def _index_run(
    *,
    label: str,
    evaluations: Iterable[Mapping],
    expected: set[int],
) -> dict[int, Mapping]:
    result = {}
    for record in evaluations:
        if record.get("schema_version") != EVALUATION_RESULT_VERSION:
            raise ValueError(f"{label} contains an unsupported evaluation schema")
        task_id = int(record["task_id"])
        if task_id not in expected:
            raise ValueError(f"{label} contains unexpected task_id {task_id}")
        if task_id in result:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        result[task_id] = record
    return result


def _hard_violation_count(record: Mapping) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    rubric = record["requirement_rubric"]
    hardness = {
        item["rubric_id"]: item["hardness"]
        for item in rubric["rubrics"]
    }
    return sum(
        assessment["status"] == "violated"
        and hardness.get(assessment["rubric_id"]) == "hard"
        for assessment in rubric["assessments"]
    )


def _dimension_score(record: Mapping, name: str) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    return int(quality["dimension_scores"][name]["score"])


def _delta_summary(deltas: list[float], *, lower_is_better: bool) -> dict:
    if lower_is_better:
        improved = sum(delta < 0 for delta in deltas)
        worsened = sum(delta > 0 for delta in deltas)
    else:
        improved = sum(delta > 0 for delta in deltas)
        worsened = sum(delta < 0 for delta in deltas)
    return {
        "paired_tasks": len(deltas),
        "mean_delta_target_minus_source": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "improved_tasks": improved,
        "unchanged_tasks": sum(delta == 0 for delta in deltas),
        "worsened_tasks": worsened,
    }


def _pairwise(
    source_label: str,
    source: Mapping[int, Mapping],
    target_label: str,
    target: Mapping[int, Mapping],
) -> dict:
    paired_ids = sorted(set(source) & set(target))
    strict_transitions = Counter()
    reward_type_transitions = Counter()
    disagreement_transitions = Counter()
    hard_deltas = []
    dimension_deltas = {name: [] for name in JUDGE_DIMENSIONS}
    step_deltas = []
    guard_deltas = []
    duplicate_action_deltas = []

    for task_id in paired_ids:
        left = source[task_id]
        right = target[task_id]
        left_reward = left["reward_and_terminal"]["metrics"]
        right_reward = right["reward_and_terminal"]["metrics"]
        left_success = bool(left_reward.get("strict_gold_success"))
        right_success = bool(right_reward.get("strict_gold_success"))
        strict_transitions[
            f"{'success' if left_success else 'failure'}_to_"
            f"{'success' if right_success else 'failure'}"
        ] += 1
        reward_type_transitions[
            f"{left_reward.get('reward_type', 'unknown')} -> "
            f"{right_reward.get('reward_type', 'unknown')}"
        ] += 1

        left_disagreement = bool(
            left["requirement_rubric"]["reward_rubric_disagreement"]
        )
        right_disagreement = bool(
            right["requirement_rubric"]["reward_rubric_disagreement"]
        )
        disagreement_transitions[
            f"{'disagreement' if left_disagreement else 'aligned'}_to_"
            f"{'disagreement' if right_disagreement else 'aligned'}"
        ] += 1

        left_hard = _hard_violation_count(left)
        right_hard = _hard_violation_count(right)
        if left_hard is not None and right_hard is not None:
            hard_deltas.append(right_hard - left_hard)
        for name in JUDGE_DIMENSIONS:
            left_score = _dimension_score(left, name)
            right_score = _dimension_score(right, name)
            if left_score is not None and right_score is not None:
                dimension_deltas[name].append(right_score - left_score)

        left_deterministic = left["deterministic"]
        right_deterministic = right["deterministic"]
        step_deltas.append(
            right_deterministic["actions_and_efficiency"][
                "executed_tool_steps"
            ]
            - left_deterministic["actions_and_efficiency"][
                "executed_tool_steps"
            ]
        )
        guard_deltas.append(
            right_deterministic["legality"]["guard_rejection_count"]
            - left_deterministic["legality"]["guard_rejection_count"]
        )
        duplicate_action_deltas.append(
            right_deterministic["repetition"][
                "duplicate_canonical_action_count"
            ]
            - left_deterministic["repetition"][
                "duplicate_canonical_action_count"
            ]
        )

    return {
        "source": source_label,
        "target": target_label,
        "paired_tasks": len(paired_ids),
        "paired_task_ids": paired_ids,
        "reward_and_terminal": {
            "strict_success_transitions": dict(
                sorted(strict_transitions.items())
            ),
            "reward_type_transitions": dict(
                sorted(reward_type_transitions.items())
            ),
        },
        "requirement_rubric": {
            "hard_violation_delta": _delta_summary(
                hard_deltas,
                lower_is_better=True,
            ),
            "reward_rubric_disagreement_transitions": dict(
                sorted(disagreement_transitions.items())
            ),
        },
        "trajectory_quality": {
            name: _delta_summary(deltas, lower_is_better=False)
            for name, deltas in dimension_deltas.items()
        },
        "deterministic": {
            "executed_tool_steps": _delta_summary(
                step_deltas,
                lower_is_better=True,
            ),
            "guard_rejections": _delta_summary(
                guard_deltas,
                lower_is_better=True,
            ),
            "duplicate_canonical_actions": _delta_summary(
                duplicate_action_deltas,
                lower_is_better=True,
            ),
        },
    }

def compare_evaluation_runs(
    *,
    expected_task_ids: Iterable[int],
    runs: Mapping[str, Iterable[Mapping]],
) -> dict:
    """Compare each model pair on identical task IDs, section by section."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    if len(runs) < 2:
        raise ValueError("at least two model runs are required")
    expected_set = set(expected)
    indexed = {
        str(label): _index_run(
            label=str(label),
            evaluations=evaluations,
            expected=expected_set,
        )
        for label, evaluations in runs.items()
    }
    labels = list(indexed)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": len(expected),
        "models": {
            label: {
                "completed_evaluations": len(indexed[label]),
                "missing_task_ids": sorted(
                    expected_set - set(indexed[label])
                ),
            }
            for label in labels
        },
        "pairwise": {
            f"{source}_to_{target}": _pairwise(
                source,
                indexed[source],
                target,
                indexed[target],
            )
            for source, target in combinations(labels, 2)
        },
    }


# ----------------------------------------------------------------------
# WP1 paired statistics：exact McNemar、Holm 校正与 paired bootstrap CI。
# 这里只做配对统计，不把四个面板压成单一 composite score。
# ----------------------------------------------------------------------


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Exact McNemar 双侧 p 值：b/c 为两个不一致格子的计数，不依赖 scipy。"""

    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError("McNemar counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def exact_mcnemar(*, b: int, c: int, both_successes: int = 0) -> dict:
    """由配对计数构造 exact McNemar 检验结果。

    b = source 成功而 target 失败的任务数；c = source 失败而 target 成功的任务数；
    both_successes = 双方都成功的任务数，仅用于报告配对总数。
    """

    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError("McNemar counts must be non-negative")
    return {
        "b": b,
        "c": c,
        "paired_tasks": int(both_successes) + b + c,
        "statistic": min(b, c),
        "p_value": exact_mcnemar_p_value(b, c),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down 校正；返回与输入顺序一致的 adjusted p 值（单调、封顶 1）。"""

    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    seed,
    iterations: int = 2000,
    confidence_level: float = 0.95,
) -> dict:
    """次要连续指标的 paired bootstrap 百分位 CI（均值），可用种子复现。"""

    values = [float(value) for value in deltas]
    if not values:
        return {
            "paired_tasks": 0,
            "estimate": None,
            "lower": None,
            "upper": None,
            "confidence_level": confidence_level,
            "iterations": iterations,
            "seed": seed,
        }
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    rng = random.Random(_derived_seed(seed))
    count = len(values)
    means = []
    for _ in range(int(iterations)):
        sample_total = 0.0
        for _ in range(count):
            sample_total += values[rng.randrange(count)]
        means.append(sample_total / count)
    means.sort()
    alpha = 1.0 - confidence_level
    lower_index = int(math.floor(alpha / 2.0 * iterations))
    upper_index = int(math.ceil((1.0 - alpha / 2.0) * iterations)) - 1
    return {
        "paired_tasks": count,
        "estimate": sum(values) / count,
        "lower": means[min(lower_index, iterations - 1)],
        "upper": means[max(upper_index, 0)],
        "confidence_level": confidence_level,
        "iterations": int(iterations),
        "seed": seed,
    }


def _derived_seed(seed) -> int:
    """把任意种子稳定地映射为 128 位整数，保证逐端点可复现。"""

    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return int(stable_hash(seed)[:32], 16)


def _binary_value(record: Mapping, endpoint: str) -> bool:
    reward = record["reward_and_terminal"]["metrics"]
    return bool(reward.get(endpoint))


def _continuous_value(record: Mapping, endpoint: str) -> float | None:
    if endpoint.startswith("dimension_"):
        quality = record["trajectory_quality"]
        if quality.get("judge_status") != "valid":
            return None
        name = endpoint[len("dimension_"):]
        return float(quality["dimension_scores"][name]["score"])
    if endpoint in {"final_reward", "terminal_utility"}:
        return float(record["reward_and_terminal"]["metrics"].get(endpoint) or 0.0)
    if endpoint == "executed_tool_steps":
        return float(record["deterministic"]["actions_and_efficiency"]["executed_tool_steps"])
    if endpoint == "guard_rejections":
        return float(record["deterministic"]["legality"]["guard_rejection_count"])
    if endpoint == "duplicate_canonical_actions":
        return float(
            record["deterministic"]["repetition"]["duplicate_canonical_action_count"]
        )
    raise ValueError(f"unknown continuous endpoint {endpoint!r}")


def _fully_paired_index(
    *,
    label: str,
    evaluations: Iterable[Mapping],
    expected: set[int],
) -> dict[int, Mapping]:
    """统计比较要求完全配对：缺失 / duplicate / unexpected task 直接抛错。"""

    indexed = _index_run(
        label=label,
        evaluations=evaluations,
        expected=expected,
    )
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"{label} is missing expected task_ids: {missing}")
    return indexed


def compare_paired_statistics(
    *,
    expected_task_ids: Iterable[int],
    runs: Mapping[str, Iterable[Mapping]],
    families: Mapping[str, tuple[str, str]],
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260829,
    confidence_level: float = 0.95,
) -> dict:
    """按注册的两族比较（如 M2-vs-M1 与 M3-vs-M2）输出配对统计报告。

    每族内：二元端点用 exact McNemar 并做 Holm 校正；连续端点用 paired
    bootstrap 95% CI。所有任务必须完全配对；不输出 composite score。
    """

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    if not families:
        raise ValueError("at least one comparison family is required")
    expected_set = set(expected)
    indexed = {
        str(label): _fully_paired_index(
            label=str(label),
            evaluations=evaluations,
            expected=expected_set,
        )
        for label, evaluations in runs.items()
    }
    family_results = {}
    for family_label, (target_label, source_label) in families.items():
        target_label = str(target_label)
        source_label = str(source_label)
        for label in (target_label, source_label):
            if label not in indexed:
                raise ValueError(f"family {family_label!r} references unknown run {label!r}")
        task_ids = sorted(expected_set)
        binary = {}
        for endpoint in BINARY_ENDPOINTS:
            b = 0
            c = 0
            both = 0
            for task_id in task_ids:
                source_value = _binary_value(indexed[source_label][task_id], endpoint)
                target_value = _binary_value(indexed[target_label][task_id], endpoint)
                if source_value and not target_value:
                    b += 1
                elif not source_value and target_value:
                    c += 1
                elif source_value and target_value:
                    both += 1
            binary[endpoint] = exact_mcnemar(
                b=b, c=c, both_successes=both,
            )
        p_values = [binary[endpoint]["p_value"] for endpoint in BINARY_ENDPOINTS]
        adjusted = holm_adjust(p_values)
        for endpoint, holm_p in zip(BINARY_ENDPOINTS, adjusted):
            binary[endpoint]["holm_adjusted_p_value"] = holm_p
        continuous = {}
        for endpoint in CONTINUOUS_ENDPOINTS:
            deltas = []
            for task_id in task_ids:
                left = _continuous_value(indexed[source_label][task_id], endpoint)
                right = _continuous_value(indexed[target_label][task_id], endpoint)
                if left is None or right is None:
                    continue
                deltas.append(right - left)
            ci = paired_bootstrap_ci(
                deltas,
                seed=(bootstrap_seed, family_label, endpoint),
                iterations=bootstrap_iterations,
                confidence_level=confidence_level,
            )
            continuous[endpoint] = {
                "mean_delta_target_minus_source": ci["estimate"],
                "bootstrap_ci": ci,
            }
        family_results[family_label] = {
            "target": target_label,
            "source": source_label,
            "paired_tasks": len(task_ids),
            "binary_endpoints": binary,
            "continuous_endpoints": continuous,
        }
    return {
        "schema_version": PAIRED_STATISTICS_SCHEMA_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": len(expected),
        "bootstrap": {
            "iterations": int(bootstrap_iterations),
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "families": family_results,
        "notes": [
            "exact McNemar + Holm 校正按族进行（如 M2-vs-M1 与 M3-vs-M2）",
            "次要连续指标使用 paired bootstrap CI；不合成单一 composite score",
        ],
    }
