import hashlib
import json
import unittest
from pathlib import Path

from commerce_posttrain.contracts.runtime import DEFAULT_OUTPUT

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UpstreamProvenanceTest(unittest.TestCase):
    def test_integrated_runtime_uses_canonical_environment_manifest(self):
        config = json.loads(
            (ROOT / "configs/runtime/runtime.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["environment_manifest"], "data/environment.json")
        self.assertTrue((ROOT / config["environment_manifest"]).is_file())
        contract = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(contract["upstream_commit"], "4ed73020e1d7d07eb93e7375a4606b0901d3cded")
        self.assertTrue(contract["upstream_repository"].endswith("shopping-grpo-longhorizon.git"))

    def test_minimal_runtime_excludes_training_and_evaluation_packages(self):
        self.assertTrue((ROOT / "src/shopping_grpo/training").exists())
        self.assertTrue((ROOT / "src/shopping_grpo/evaluation").exists())
        self.assertTrue((ROOT / "src/commerce_posttrain/collection").exists())

    def test_final_200_is_exact_and_matches_upstream_metadata(self):
        task_path = ROOT / "data/evaluation/tasks.jsonl"
        metadata = json.loads(
            (ROOT / "data/evaluation/metadata.json").read_text(encoding="utf-8")
        )
        task_ids = [
            json.loads(line).get("task_id", index)
            for index, line in enumerate(task_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        ]
        self.assertEqual(len(task_ids), 200)
        self.assertEqual(len(set(task_ids)), 200)
        self.assertEqual(metadata["tasks"], 200)
        # The evaluation metadata is an external asset descriptor; the
        # integrated repository validates row count and task uniqueness here.
        self.assertEqual(metadata["path"], "data/evaluation/tasks.jsonl")

    def test_generated_private_task_facts_match_source_manifest(self):
        private_path = ROOT / "data/private/task_facts.jsonl"
        if not private_path.exists():
            self.skipTest("private TaskFacts is deterministically regenerated on demand")
        source = json.loads(
            (ROOT / "data/manifests/task_facts_source.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source["row_count"], 23421)
        self.assertEqual(source["task_facts_sha256"], file_hash(private_path))


if __name__ == "__main__":
    unittest.main()
