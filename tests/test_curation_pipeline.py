import json
import tempfile
import unittest
from pathlib import Path

from test_collection_runtime import collected

from commerce_posttrain.collection.schema import finalize_trajectory
from commerce_posttrain.curation.pipeline import (
    TERMINAL_TOOL_CONTENT,
    assign_stratified_splits,
    build_action_only_sft_row,
    iter_latest_trajectories,
    select_outcome_and_process,
)


class CurationPipelineTest(unittest.TestCase):
    def test_latest_effective_row_supersedes_infrastructure_retry(self):
        complete = collected()
        failed = json.loads(json.dumps(complete))
        failed["status"] = "infrastructure_failure"
        failed["done"] = False
        failed["terminal_environment_result"] = {}
        failed["error"] = {"type": "ConnectionError", "message": "temporary"}
        failed = finalize_trajectory(failed)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(
                json.dumps(failed) + "\n" + json.dumps(complete) + "\n",
                encoding="utf-8",
            )
            rows = list(iter_latest_trajectories(raw))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "done")

    def test_action_only_row_contains_no_private_environment_result(self):
        row = build_action_only_sft_row(collected(), system_prompt="system")
        self.assertEqual(row["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(row["messages"][-1]["content"], TERMINAL_TOOL_CONTENT)
        encoded = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("environment_result_private", encoded)
        self.assertNotIn("reward_detail", encoded)

    def test_stratified_split_is_deterministic_and_exact(self):
        tasks = [
            {"task_id": index, "difficulty": ("easy", "medium", "hard")[index % 3]}
            for index in range(19)
        ]
        first = assign_stratified_splits(tasks, train_count=12, dev_count=5, seed=7)
        second = assign_stratified_splits(tasks, train_count=12, dev_count=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("train"), 12)
        self.assertEqual(list(first.values()).count("dev"), 5)
        self.assertEqual(list(first.values()).count("reserve"), 2)

    def test_outcome_and_process_select_within_same_task(self):
        first = {"task_id": 1, "attempt_index": 0, "request_id": "first"}
        second = {"task_id": 1, "attempt_index": 1, "request_id": "second"}
        outcome, process = select_outcome_and_process(
            [
                (first, {"selection_tuple": [2, 0]}),
                (second, {"selection_tuple": [0, 1]}),
            ]
        )
        self.assertEqual(outcome["request_id"], "first")
        self.assertEqual(process["request_id"], "second")


if __name__ == "__main__":
    unittest.main()
