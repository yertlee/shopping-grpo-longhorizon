"""Final-200 eval facts builder 的单元测试（注入 fake loader，无真实数据/环境）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_final_eval_facts import build_final_eval_facts


def _write_ids(path: Path, ids: list[int]) -> None:
    path.write_text(
        "".join(json.dumps({"task_id": i}, separators=(",", ":")) + "\n" for i in ids),
        encoding="utf-8",
    )


class BuildFinalEvalFactsTest(unittest.TestCase):
    def _run(self, tmp: Path, *, canonical=None, dev=None, expected_count=200):
        task_ids_path = tmp / "final_task_ids.jsonl"
        ids = list(range(1000, 1000 + expected_count))
        _write_ids(task_ids_path, ids)
        product_gzip = tmp / "products.json.gz"
        product_gzip.write_bytes(b"fake")
        facts_output = tmp / "final-200-evaluation-facts.json"
        manifest_output = tmp / "final-200-evaluation-facts.manifest.json"

        products = [{"asin": f"A{i}"} for i in range(5)]
        goals = [{"asin": f"A{i % 5}", "instruction_text": f"q{i}", "attributes": [],
                  "goal_options": []} for i in range(1200)]
        product_item_dict = {f"A{i}": {"asin": f"A{i}", "title": f"t{i}"} for i in range(5)}

        manifest = build_final_eval_facts(
            task_ids_path=task_ids_path,
            product_gzip=product_gzip,
            shopsim_root=tmp,
            facts_output=facts_output,
            manifest_output=manifest_output,
            expected_task_count=expected_count,
            expected_product_count=5,
            expected_product_sha256="",
            product_loader=lambda *_: (products, product_item_dict, {}),
            goal_builder=lambda *_: goals,
            facts_builder=lambda tids, g, p: [
                {"task_id": int(t), "schema_version": "shopping-task-facts-v1"} for t in tids
            ],
            canonical_ids_provider=lambda: set(canonical if canonical is not None else ids),
            dev_ids_provider=lambda: set(dev if dev is not None else []),
            code_hashes={"builder": "deadbeef"},
        )
        return manifest, facts_output

    def test_builds_ordered_facts_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, facts_output = self._run(Path(tmp))
            doc = json.loads(facts_output.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], "shopping-task-facts-source-v1")
            self.assertEqual([r["task_id"] for r in doc["facts"]], list(range(1000, 1200)))
            self.assertEqual(manifest["task_count"], 200)
            self.assertTrue(manifest["canonical_blind_asset_match"])
            self.assertEqual(manifest["dev_overlap_count"], 0)
            self.assertEqual(manifest["outputs"]["facts"]["row_count"], 200)

    def test_rejects_dev_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = set(range(1000, 1200))
            with self.assertRaises(ValueError):
                self._run(Path(tmp), dev={1000, 1001})

    def test_rejects_canonical_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._run(Path(tmp), canonical={9999})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
