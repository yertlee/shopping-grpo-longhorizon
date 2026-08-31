from __future__ import annotations

import hashlib
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_dev_eval_assets as builder


def _jsonl(path: Path, ids: list[int]) -> bytes:
    data = b"".join(json.dumps({"task_id": i}).encode() + b"\n" for i in ids)
    path.write_bytes(data)
    return data


class DevEvalAssetBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.process = root / "process-dev.jsonl"
        self.outcome = root / "outcome-dev.jsonl"
        self.products = root / "products.json.gz"
        self.shopsim = root / "shopsim"
        ids = list(range(100))
        process_bytes = _jsonl(self.process, ids)
        outcome_bytes = _jsonl(self.outcome, ids)
        self.expected = {
            "process": hashlib.sha256(process_bytes).hexdigest(),
            "outcome": hashlib.sha256(outcome_bytes).hexdigest(),
            "products": hashlib.sha256(b"products").hexdigest(),
        }
        self.products.write_bytes(b"products")

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, **kwargs):
        root = Path(self.tmp.name)
        options = dict(
            process_path=self.process,
            outcome_path=self.outcome,
            product_gzip=self.products,
            shopsim_root=self.shopsim,
            ids_output=root / "dev-ids.jsonl",
            facts_output=root / "facts.json",
            manifest_output=root / "manifest.json",
            expected_hashes=self.expected,
            expected_product_count=100,
            code_hashes={"fake_loader": "f" * 64},
            final_ids_provider=lambda: {999},
            product_loader=lambda *_: (
                [{"asin": str(i), "title": "A"} for i in range(100)],
                {str(i): {"asin": str(i), "title": "A"} for i in range(100)},
                [],
            ),
            goal_builder=lambda products, prices: [
                {"asin": p["asin"], "instruction_text": f"q{i}", "goal_options": []}
                for i, p in enumerate(products)
            ],
            facts_builder=lambda ids, goals, products: [
                {"schema_version": "shopping-task-facts-v1", "task_id": i}
                for i in ids
            ],
        )
        options.update(kwargs)
        return builder.build_dev_eval_assets(**options)

    def test_happy_path_outputs_ids_only_and_manifest(self):
        manifest = self._build()
        root = Path(self.tmp.name)
        rows = [json.loads(line) for line in (root / "dev-ids.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[:3], [{"task_id": 0}, {"task_id": 1}, {"task_id": 2}])
        facts = json.loads((root / "facts.json").read_text())
        self.assertEqual(facts["schema_version"], "shopping-task-facts-source-v1")
        self.assertEqual(manifest["task_count"], 100)
        self.assertEqual(manifest["final_overlap_count"], 0)
        self.assertNotIn("title", (root / "manifest.json").read_text())

    def test_process_outcome_mismatch_and_duplicate_fail(self):
        _jsonl(self.outcome, list(range(99)) + [100])
        with self.assertRaises(ValueError):
            self._build()
        _jsonl(self.outcome, [0] * 100)
        with self.assertRaises(ValueError):
            self._build()

    def test_outcome_order_may_differ_without_changing_process_order(self):
        _jsonl(self.outcome, list(reversed(range(100))))
        outcome_bytes = self.outcome.read_bytes()
        manifest = self._build(
            expected_hashes={
                **self.expected,
                "outcome": hashlib.sha256(outcome_bytes).hexdigest(),
            }
        )
        self.assertFalse(manifest["source_sequences_equal"])
        rows = [
            json.loads(line)
            for line in (Path(self.tmp.name) / "dev-ids.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["task_id"] for row in rows], list(range(100)))

    def test_hash_tamper_and_final_overlap_fail(self):
        self.products.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self._build()
        self.products.write_bytes(b"products")
        with self.assertRaises(ValueError):
            self._build(final_ids_provider=lambda: {1})

    def test_product_range_and_facts_order_fail(self):
        with self.assertRaises(ValueError):
            self._build(expected_product_count=101)
        with self.assertRaises(ValueError):
            self._build(facts_builder=lambda ids, goals, products: [{"task_id": 1}] * 100)

    def test_official_loader_receives_decompressed_temporary_json(self):
        root = Path(self.tmp.name)
        compressed = root / "official-products.json.gz"
        payload = json.dumps([{"asin": "A"}]).encode()
        with gzip.open(compressed, "wb") as stream:
            stream.write(payload)
        calls = []

        def fake_load_products(*, filepath, human_goals):
            path = Path(filepath)
            calls.append((path.read_bytes(), human_goals, path.exists()))
            return ([{"asin": "A"}], {"A": {"asin": "A"}}, {"A": 1.0}, {})

        modules = {
            "web_agent_site": types.ModuleType("web_agent_site"),
            "web_agent_site.engine": types.ModuleType("web_agent_site.engine"),
            "web_agent_site.engine.engine": types.SimpleNamespace(
                load_products=fake_load_products
            ),
        }
        with patch.dict(sys.modules, modules):
            products, items, prices = builder._load_products(compressed, self.shopsim)

        self.assertEqual(calls, [(payload, True, True)])
        self.assertEqual(products[0]["asin"], "A")
        self.assertEqual(items["A"]["asin"], "A")
        self.assertEqual(prices["A"], 1.0)

    def test_different_existing_output_is_rejected_and_same_is_idempotent(self):
        self._build()
        self._build()
        ids_path = Path(self.tmp.name) / "dev-ids.jsonl"
        ids_path.write_text('{"task_id": 99}\n')
        with self.assertRaises(FileExistsError):
            self._build()

    @patch("shopping_grpo.evaluation.blind_guard.validate_canonical_blind_asset")
    def test_default_blind_provider_is_used(self, guard):
        guard.return_value = ({}, {1})
        with self.assertRaises(ValueError):
            self._build(final_ids_provider=None)


if __name__ == "__main__":
    unittest.main()
