import json
import unittest

from test_collection_runtime import collected

from commerce_posttrain.collection.schema import finalize_trajectory
from commerce_posttrain.curation.process_analyzer import (
    analyze_trajectory,
    attach_first_divergence,
)


class ProcessAnalyzerTest(unittest.TestCase):
    def test_process_features_use_only_actor_visible_stream(self):
        original = collected()
        changed = json.loads(json.dumps(original))
        changed["events"][0]["environment_result_private"]["goal"] = {
            "asin": "DIFFERENT_PRIVATE_GOLD"
        }
        changed["events"][-1]["environment_result_private"]["goal"] = {
            "secret": "DIFFERENT"
        }
        changed["terminal_environment_result"]["goal"] = {"secret": "DIFFERENT"}
        changed = finalize_trajectory(changed)
        left = analyze_trajectory(original, query="买一个保温杯")
        right = analyze_trajectory(changed, query="买一个保温杯")
        self.assertEqual(left["actor_visible_input_sha256"], right["actor_visible_input_sha256"])
        self.assertEqual(left["features"], right["features"])
        self.assertEqual(left["selection_tuple"], right["selection_tuple"])

    def test_private_reward_changes_outcome_but_not_process(self):
        original = collected()
        changed = json.loads(json.dumps(original))
        reward = changed["terminal_environment_result"]["reward_detail"]
        reward["reward_type"] = "wrong_purchase"
        reward["purchase_success"] = False
        reward["termination_reason"] = "wrong_purchase"
        changed = finalize_trajectory(changed)
        left = analyze_trajectory(original, query="买一个保温杯")
        right = analyze_trajectory(changed, query="买一个保温杯")
        self.assertEqual(left["features"], right["features"])
        self.assertTrue(left["outcome_private_verifier"]["strict_gold_success"])
        self.assertFalse(right["outcome_private_verifier"]["strict_gold_success"])

    def test_first_divergence_reports_actual_environment_turn(self):
        first = collected()
        second = json.loads(json.dumps(first))
        environment_steps = [
            event for event in second["events"] if event["type"] == "environment_step"
        ]
        environment_steps[1]["parameters"] = {"asin": "87654321"}
        second = finalize_trajectory(second)
        records = [
            analyze_trajectory(first, query="买一个保温杯"),
            analyze_trajectory(second, query="买一个保温杯"),
        ]
        updated = attach_first_divergence(records, [first, second])
        self.assertEqual(
            [row["features"]["diversity"]["first_divergence_turn"] for row in updated],
            [1, 1],
        )


if __name__ == "__main__":
    unittest.main()
