import unittest

from commerce_posttrain.collection.schema import finalize_trajectory, request_identity
from commerce_posttrain.data_quality.summary import summarize_collection_quality


def trajectory(task_id, attempt, *, gold=False):
    terminal = {}
    status = "assistant_final"
    done = False
    if gold:
        status = "done"
        done = True
        terminal = {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        }
    teacher = {"model": "teacher"}
    runtime_hash = "a" * 64
    return finalize_trajectory(
        {
            "schema_version": "commerce-teacher-raw-v1",
            "request_id": request_identity(
                task_id=task_id,
                attempt_index=attempt,
                runtime_contract_sha256=runtime_hash,
                teacher=teacher,
            ),
            "execution_id": f"execution-{task_id}-{attempt}",
            "task_id": task_id,
            "attempt_index": attempt,
            "created_at": "2026-08-27T00:00:00+00:00",
            "runtime_contract_sha256": runtime_hash,
            "teacher": teacher,
            "status": status,
            "events": [
                {
                    "type": "environment_reset",
                    "event_index": 0,
                    "actor_visible_observation": "visible",
                }
            ],
            "terminal_environment_result": terminal,
            "done": done,
            "released": True,
            "error": None,
            "release_error": None,
        }
    )


class QualitySummaryTest(unittest.TestCase):
    def test_reports_raw_and_eligible_denominators(self):
        rows = [
            trajectory(1, 0, gold=True),
            trajectory(1, 1),
            trajectory(2, 0),
        ]
        reachability = {
            "manifest_sha256": "b" * 64,
            "splits": {
                "teacher_pool": {
                    "eligible_task_ids": [1],
                    "unreachable": [{"task_id": 2}],
                }
            },
        }
        summary = summarize_collection_quality(
            rows, reachability_manifest=reachability, raw_sha256="c" * 64
        )
        self.assertEqual(summary["raw_fixed_denominator"]["attempt_count"], 3)
        self.assertEqual(summary["raw_fixed_denominator"]["strict_gold_rate"], 0.3333)
        self.assertEqual(summary["dataset_unreachable"]["task_ids"], [2])
        self.assertEqual(summary["eligible_policy_denominator"]["attempt_count"], 2)
        self.assertEqual(summary["eligible_policy_denominator"]["strict_gold_rate"], 0.5)
        self.assertEqual(summary["eligible_policy_denominator"]["task_any_gold_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
