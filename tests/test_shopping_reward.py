"""Shopping GRPO terminal reward tests for the Environment v2.1/Reward v3 contract."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    validate_reward,
)


def reward_detail(
    *,
    reward_type="gold_purchase",
    reward_valid=True,
    terminal_utility=1.0,
    weighted_score=1.0,
    evidence_coverage=1.0,
):
    """Build the public terminal payload emitted by ShopSimulator Reward v3."""
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": reward_type == "gold_purchase",
        "terminal_utility": terminal_utility,
        "purchase_success": reward_type
        in {"gold_purchase", "valid_alternative_purchase"},
        "sampling_invalid": not reward_valid,
        "hard_gates": {
            "category": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "category_leaf_ancestor_chain",
                "source_field": "category",
            },
            "budget": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
                "comparator": "variant_price_budget_v1",
                "source_field": "variant_price",
            },
        },
        "weighted_score": weighted_score,
        "evidence_coverage": evidence_coverage,
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
    }


def terminal_state(*, steps=8, detail=None, native_reward=1.0):
    detail = detail or reward_detail(terminal_utility=native_reward)
    state = make_runtime_state(task_id=1, max_steps=35)
    state["steps"] = [{"index": index} for index in range(steps)]
    state.update(
        {
            "done": True,
            "terminal_result": {
                "done": True,
                "over": True,
                "reward_detail": detail,
            },
            "final_reward": native_reward,
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_unverifiable": not detail["reward_valid"],
            "termination_reason": detail["termination_reason"],
            "reward_detail": detail,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_validate_reward_accepts_and_minimizes_a_reward_v3_payload(self):
        raw = reward_detail()
        raw["private_evidence"] = {"hidden": "must not enter training diagnostics"}

        validated = validate_reward(raw)

        self.assertEqual(validated["reward_version"], "shopsimulator-reward-v3")
        self.assertEqual(validated["reward_type"], "gold_purchase")
        self.assertEqual(validated["terminal_utility"], 1.0)
        self.assertNotIn("private_evidence", validated)
        self.assertEqual(
            set(validated["dimension_scores"]),
            {"brand", "model", "core_functions", "key_options"},
        )

    def test_validate_reward_rejects_inconsistent_reward_v3_fields(self):
        missing_version = reward_detail()
        del missing_version["reward_version"]
        with self.assertRaisesRegex(ValueError, "unsupported reward_version"):
            validate_reward(missing_version)

        invalid_sampling = reward_detail()
        invalid_sampling["sampling_invalid"] = True
        with self.assertRaisesRegex(ValueError, "sampling_invalid"):
            validate_reward(invalid_sampling)

        invalid_gate = reward_detail()
        invalid_gate["hard_gates"]["category"]["status"] = "fail"
        with self.assertRaisesRegex(ValueError, "inconsistent passed"):
            validate_reward(invalid_gate)

    def test_gold_purchase_breakdown_uses_native_terminal_utility(self):
        result = reward_breakdown(terminal_state(steps=8))

        self.assertEqual(result["full"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["semantic"], 1.0)
        self.assertEqual(result["terminal_utility"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertFalse(result["sampling_invalid"])
        self.assertFalse(result["infrastructure_invalid"])

    def test_valid_alternative_purchase_keeps_utility_and_success_separate(self):
        detail = reward_detail(
            reward_type="valid_alternative_purchase",
            terminal_utility=0.55,
            weighted_score=0.8,
            evidence_coverage=0.75,
        )
        result = reward_breakdown(terminal_state(detail=detail, native_reward=0.55))

        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["semantic"], 1.0)
        self.assertEqual(result["terminal_utility"], 0.55)
        self.assertEqual(result["total"], 0.55)
        self.assertEqual(result["purchase_success"], 1.0)

    def test_unverifiable_reward_is_invalid_for_sampling_but_not_infrastructure(self):
        detail = reward_detail(
            reward_type="reward_unverifiable",
            reward_valid=False,
            terminal_utility=0.0,
            weighted_score=0.0,
            evidence_coverage=0.0,
        )
        result = reward_breakdown(terminal_state(detail=detail, native_reward=0.0))

        self.assertEqual(result["terminal_utility"], 0.0)
        self.assertEqual(result["total"], 0.0)
        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])
        self.assertFalse(result["infrastructure_invalid"])

    def test_nonterminal_state_does_not_create_a_reward(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["final_reward"] = 1.0

        result = reward_breakdown(state)

        self.assertEqual(result["terminal_utility"], 0.0)
        self.assertEqual(result["total"], 0.0)
        self.assertTrue(result["sampling_invalid"])

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
