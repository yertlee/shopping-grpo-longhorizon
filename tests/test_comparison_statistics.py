"""comparison.py 配对统计扩展的数值正确性测试（WP1）。

用构造数据验证 exact McNemar / Holm / paired bootstrap CI 的已知答案，并确认
缺失 / duplicate / unexpected task ID 直接抛错、不出现 composite score、
既有 compare_evaluation_runs() 行为不变。
"""

from __future__ import annotations

import unittest

from shopping_grpo.evaluation.comparison import (
    BINARY_ENDPOINTS,
    COMPARISON_SCHEMA_VERSION,
    CONTINUOUS_ENDPOINTS,
    PAIRED_STATISTICS_SCHEMA_VERSION,
    compare_evaluation_runs,
    compare_paired_statistics,
    exact_mcnemar_p_value,
    holm_adjust,
    paired_bootstrap_ci,
)
from shopping_grpo.evaluation.contracts import JUDGE_DIMENSIONS
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION


def _record(
    task_id,
    *,
    strict=False,
    purchase=False,
    reward_valid=True,
    final_reward=0.0,
    terminal_utility=0.0,
    steps=5,
    guards=0,
    duplicate_actions=0,
    judge_status="valid",
    dims=(1, 1, 1, 1, 1),
):
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "task_id": int(task_id),
        "trajectory_id": f"t-{task_id}",
        "reward_and_terminal": {
            "metrics": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase" if strict else "graceful_stop",
                "reward_valid": reward_valid,
                "purchase_success": purchase,
                "strict_gold_success": strict,
                "final_reward": final_reward,
                "terminal_utility": terminal_utility,
                "weighted_score": 0.0,
            },
            "terminal": {},
        },
        "requirement_rubric": {
            "rubric_version": "v1",
            "rubrics": [],
            "assessments": [],
            "reward_rubric_disagreement": False,
            "disagreement_reasons": [],
        },
        "trajectory_quality": {
            "judge_status": judge_status,
            "dimension_scores": {
                name: {"score": dims[index], "reason": "r", "evidence_event_ids": []}
                for index, name in enumerate(JUDGE_DIMENSIONS)
            },
            "errors": {"primary": None, "secondary": [], "evidence_event_ids": []},
            "overall_diagnosis": "ok",
        },
        "deterministic": {
            "actions_and_efficiency": {"executed_tool_steps": steps},
            "legality": {"guard_rejection_count": guards},
            "repetition": {"duplicate_canonical_action_count": duplicate_actions},
            "context": {},
            "validity": {"infrastructure_invalid": False},
        },
        "artifacts": {},
    }


def _run(task_ids, *, strict=(), purchase=(), rewards=None, dims=None):
    rewards = rewards or {}
    dims = dims or {}
    return [
        _record(
            task_id,
            strict=task_id in strict,
            purchase=task_id in purchase,
            reward_valid=rewards.get(task_id, True),
            final_reward=rewards.get("final", {}).get(task_id, 0.0),
            steps=5,
            dims=dims.get(task_id, (1, 1, 1, 1, 1)),
        )
        for task_id in task_ids
    ]


TASK_IDS = [1, 2, 3, 4, 5]


class ExactMcNemarTest(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        # n=5, min=0：p = 2 * (1/2)^5 = 0.0625
        self.assertEqual(exact_mcnemar_p_value(0, 5), 0.0625)
        # n=6, min=1：p = 2 * (C(6,0)+C(6,1)) / 2^6 = 2 * 7/64 = 0.21875
        self.assertEqual(exact_mcnemar_p_value(1, 5), 0.21875)
        # n=3, min=0：p = 2 * 1/8 = 0.25
        self.assertEqual(exact_mcnemar_p_value(3, 0), 0.25)
        # n=10, min=0：p = 2/1024
        self.assertEqual(exact_mcnemar_p_value(0, 10), 2 / 1024)
        # n=4, min=2：2 * (1+4+6)/16 = 1.375 → 封顶 1.0
        self.assertEqual(exact_mcnemar_p_value(2, 2), 1.0)

    def test_rejects_negative_counts(self):
        with self.assertRaises(ValueError):
            exact_mcnemar_p_value(-1, 2)


class HolmAdjustTest(unittest.TestCase):
    def test_known_answer_preserves_order(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_caps_at_one_and_is_monotone(self):
        self.assertEqual(holm_adjust([0.5, 0.5, 1.0]), [1.0, 1.0, 1.0])
        self.assertEqual(holm_adjust([0.9, 0.2]), [0.9, 0.4])

    def test_empty_family(self):
        self.assertEqual(holm_adjust([]), [])


class PairedBootstrapCiTest(unittest.TestCase):
    def test_constant_deltas_give_degenerate_ci(self):
        ci = paired_bootstrap_ci([2.0] * 10, seed=1, iterations=500)
        self.assertEqual(ci["estimate"], 2.0)
        self.assertEqual(ci["lower"], 2.0)
        self.assertEqual(ci["upper"], 2.0)
        self.assertEqual(ci["paired_tasks"], 10)

    def test_ci_bounds_contain_estimate(self):
        deltas = [0.0] * 9 + [10.0]
        ci = paired_bootstrap_ci(deltas, seed=42, iterations=2000)
        self.assertEqual(ci["estimate"], 1.0)
        self.assertGreaterEqual(ci["lower"], 0.0)
        self.assertLessEqual(ci["upper"], 10.0)
        self.assertLessEqual(ci["lower"], ci["upper"])

    def test_same_seed_reproduces_exactly(self):
        deltas = [float(value) for value in range(12)]
        first = paired_bootstrap_ci(deltas, seed=7, iterations=300)
        second = paired_bootstrap_ci(deltas, seed=7, iterations=300)
        self.assertEqual(first, second)

    def test_empty_and_invalid_inputs(self):
        ci = paired_bootstrap_ci([], seed=1)
        self.assertEqual(ci["paired_tasks"], 0)
        self.assertIsNone(ci["lower"])
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0], seed=1, iterations=0)
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1.0], seed=1, confidence_level=1.5)


class ComparePairedStatisticsTest(unittest.TestCase):
    """m1/m2/m3 在 5 个 task 上的构造数据，全部 b/c/p 已手工算出。"""

    def setUp(self):
        self.m1 = _run(TASK_IDS)
        self.m2 = _run(
            TASK_IDS,
            strict={1, 2},
            purchase={1, 2, 3, 4},
            rewards={
                "final": {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0},
            },
        )
        self.m3 = _run(
            TASK_IDS,
            strict={1, 2, 3},
            purchase={1, 2, 3, 4, 5},
            rewards={
                "final": {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.5},
            },
        )

    def _compare(self, **overrides):
        kwargs = dict(
            expected_task_ids=TASK_IDS,
            runs={"m1": self.m1, "m2": self.m2, "m3": self.m3},
            families={"m2_vs_m1": ("m2", "m1"), "m3_vs_m2": ("m3", "m2")},
        )
        kwargs.update(overrides)
        return compare_paired_statistics(**kwargs)

    def test_schema_and_no_composite_score(self):
        report = self._compare()
        self.assertEqual(report["schema_version"], PAIRED_STATISTICS_SCHEMA_VERSION)
        self.assertEqual(report["expected_tasks"], 5)
        self.assertEqual(set(report["families"]), {"m2_vs_m1", "m3_vs_m2"})
        rendered = str(sorted(report))
        self.assertNotIn("composite", rendered)
        self.assertNotIn("total_score", rendered)

    def test_m2_vs_m1_mcnemar_counts_and_holm(self):
        report = self._compare()
        family = report["families"]["m2_vs_m1"]

        strict = family["binary_endpoints"]["strict_gold_success"]
        self.assertEqual((strict["b"], strict["c"]), (0, 2))
        self.assertEqual(strict["p_value"], 0.5)
        # Holm step-down（族内 3 个端点，升序 0.125/0.5/1.0）：
        # purchase 0.125*3=0.375；strict max(0.375, 0.5*2)=1.0；reward 1.0。
        self.assertEqual(strict["holm_adjusted_p_value"], 1.0)

        # m1 从不购买；m2 购买 1–4：b=0, c=4 → p = 2/2^4 = 0.125。
        purchase = family["binary_endpoints"]["purchase_success"]
        self.assertEqual((purchase["b"], purchase["c"]), (0, 4))
        self.assertEqual(purchase["p_value"], 0.125)
        self.assertEqual(purchase["holm_adjusted_p_value"], 0.375)

        reward_valid = family["binary_endpoints"]["reward_valid"]
        self.assertEqual((reward_valid["b"], reward_valid["c"]), (0, 0))
        self.assertEqual(reward_valid["p_value"], 1.0)
        self.assertEqual(reward_valid["holm_adjusted_p_value"], 1.0)
        self.assertEqual(family["paired_tasks"], 5)

    def test_m3_vs_m2_mcnemar_counts_and_holm(self):
        report = self._compare()
        family = report["families"]["m3_vs_m2"]
        strict = family["binary_endpoints"]["strict_gold_success"]
        self.assertEqual((strict["b"], strict["c"]), (0, 1))
        self.assertEqual(strict["p_value"], 1.0)
        purchase = family["binary_endpoints"]["purchase_success"]
        self.assertEqual((purchase["b"], purchase["c"]), (0, 1))
        reward_valid = family["binary_endpoints"]["reward_valid"]
        self.assertEqual((reward_valid["b"], reward_valid["c"]), (0, 0))
        for endpoint in BINARY_ENDPOINTS:
            self.assertEqual(
                family["binary_endpoints"][endpoint]["holm_adjusted_p_value"], 1.0
            )

    def test_continuous_bootstrap_ci_known_answers(self):
        report = self._compare(bootstrap_iterations=200, bootstrap_seed=11)
        family = report["families"]["m2_vs_m1"]
        final = family["continuous_endpoints"]["final_reward"]
        # m2 final - m1 final = [1, 1, 0.5, 0.5, 0]，均值 0.6。
        self.assertEqual(final["mean_delta_target_minus_source"], 0.6)
        ci = final["bootstrap_ci"]
        self.assertEqual(ci["paired_tasks"], 5)
        self.assertLessEqual(ci["lower"], 0.6)
        self.assertGreaterEqual(ci["upper"], 0.6)
        self.assertEqual(ci["iterations"], 200)

        m3 = report["families"]["m3_vs_m2"]["continuous_endpoints"]["final_reward"]
        # m3 final - m2 final = [0, 0, 0.5, 0, 0.5]，均值 0.2。
        self.assertEqual(m3["mean_delta_target_minus_source"], 0.2)

        # 维度分数全为 1 → 退化 CI [0, 0]。
        dimension = family["continuous_endpoints"]["dimension_search_strategy"]
        self.assertEqual(dimension["bootstrap_ci"]["lower"], 0.0)
        self.assertEqual(dimension["bootstrap_ci"]["upper"], 0.0)
        self.assertEqual(
            sorted(family["continuous_endpoints"]),
            sorted(CONTINUOUS_ENDPOINTS),
        )

    def test_report_is_reproducible_with_same_seed(self):
        first = self._compare(bootstrap_seed=20260829)
        second = self._compare(bootstrap_seed=20260829)
        self.assertEqual(
            _jsonable(first), _jsonable(second), "同种子必须逐字节可复现"
        )

    def test_rejects_missing_duplicate_and_unexpected_tasks(self):
        with self.assertRaises(ValueError):
            self._compare(
                runs={"m1": self.m1[:-1], "m2": self.m2, "m3": self.m3}
            )
        with self.assertRaises(ValueError):
            self._compare(
                runs={"m1": self.m1 + [self.m1[0]], "m2": self.m2, "m3": self.m3}
            )
        with self.assertRaises(ValueError):
            self._compare(
                runs={
                    "m1": self.m1,
                    "m2": self.m2,
                    "m3": self.m3 + [_record(99, strict=True)],
                }
            )
        with self.assertRaises(ValueError):
            self._compare(expected_task_ids=[1, 1, 2, 3, 4, 5])

    def test_rejects_unknown_labels_and_empty_families(self):
        with self.assertRaises(ValueError):
            self._compare(families={"bad": ("m2", "m0")})
        with self.assertRaises(ValueError):
            self._compare(families={})

    def test_requires_supported_schema(self):
        broken = _record(6, strict=True)
        broken["schema_version"] = "unsupported-schema"
        with self.assertRaises(ValueError):
            compare_paired_statistics(
                expected_task_ids=[1, 6],
                runs={"a": [_record(1), broken], "b": [_record(1), _record(6)]},
                families={"a_vs_b": ("b", "a")},
            )


class CompareEvaluationRunsBackwardCompatTest(unittest.TestCase):
    def test_existing_compare_evaluation_runs_unchanged(self):
        evaluations_a = [_record(1, strict=True), _record(2)]
        evaluations_b = [_record(1), _record(2, strict=True)]
        report = compare_evaluation_runs(
            expected_task_ids=[1, 2],
            runs={"a": evaluations_a, "b": evaluations_b},
        )
        self.assertEqual(report["schema_version"], COMPARISON_SCHEMA_VERSION)
        self.assertEqual(report["expected_tasks"], 2)
        pairwise = report["pairwise"]["a_to_b"]
        self.assertEqual(
            pairwise["reward_and_terminal"]["strict_success_transitions"],
            {
                "failure_to_success": 1,
                "success_to_failure": 1,
            },
        )


def _jsonable(payload):
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
