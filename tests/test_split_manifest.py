import json
import unittest

from commerce_posttrain.splits.manifest import (
    SplitLeakageError,
    SplitSpec,
    audit_split_leakage,
    build_split_manifest,
    normalize_task_facts,
    validate_split_manifest,
)


def task(
    task_id,
    *,
    query=None,
    product=None,
    family=None,
    template=None,
    model_tokens=None,
    difficulty="medium",
):
    return {
        "task_id": task_id,
        "query": query or f"购买商品{task_id}",
        "target_product_ids": [product or f"asin-{task_id}"],
        "product_family": family or f"family-{task_id}",
        "template_id": template or f"template-{task_id}",
        "model_tokens": model_tokens or [],
        "difficulty": difficulty,
    }


def small_spec():
    return SplitSpec(
        seed=7,
        counts={
            "teacher_pool": 4,
            "grpo_train": 3,
            "grpo_validation": 1,
            "final": 2,
        },
        allocation_order=("teacher_pool", "grpo_train", "grpo_validation"),
    )


def task_pool():
    rows = []
    for task_id in range(20):
        difficulty = ("easy", "medium", "hard")[task_id % 3]
        rows.append(task(task_id, difficulty=difficulty))
    # This task is connected to Final task 0 and must be excluded from training.
    rows[10] = task(
        10,
        query="购买商品0，预算200元",
        product="asin-10",
        family="family-0",
        template="template-0",
    )
    return rows


def build(rows=None):
    return build_split_manifest(
        tasks=rows or task_pool(),
        final_task_ids=[0, 1],
        spec=small_spec(),
        task_source_hash="a" * 64,
        runtime_contract_hash="b" * 64,
        task_source_label="tests/task_facts.jsonl",
    )


class SplitManifestTest(unittest.TestCase):
    def test_split_manifest_is_deterministic_and_exact_sized(self):
        first = build()
        self.assertEqual(first, build())
        self.assertEqual(first["splits"]["teacher_pool"]["count"], 4)
        self.assertEqual(first["splits"]["grpo_train"]["count"], 3)
        self.assertEqual(first["splits"]["grpo_validation"]["count"], 1)
        self.assertEqual(first["splits"]["final"]["task_ids"], [0, 1])
        selected = {
            task_id
            for split in first["splits"].values()
            for task_id in split["task_ids"]
        }
        self.assertNotIn(10, selected)
        self.assertEqual(
            first["leakage_audit"]["final_component_excluded_count"], 1
        )

    def test_valid_manifest_round_trips(self):
        rows = task_pool()
        manifest = build(rows)
        self.assertEqual(validate_split_manifest(manifest, tasks=rows), manifest)

    def test_manifest_content_tampering_is_rejected(self):
        rows = task_pool()
        manifest = json.loads(json.dumps(build(rows)))
        manifest["seed"] += 1
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_split_manifest(manifest, tasks=rows)

    def test_rehashed_internal_count_tampering_is_rejected(self):
        rows = task_pool()
        manifest = json.loads(json.dumps(build(rows)))
        manifest["splits"]["teacher_pool"]["count"] += 1
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256")
        from commerce_posttrain.splits.manifest import canonical_json_bytes, sha256_bytes

        manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(unhashed))
        with self.assertRaisesRegex(ValueError, "count does not match"):
            validate_split_manifest(manifest, tasks=rows)

    def test_each_leakage_axis_fails_closed(self):
        cases = (
            ("target_product_ids", ["same-product"], ["same-product"]),
            ("query_cluster", "same-query", "same-query"),
            ("template_id", "same-template", "same-template"),
            ("model_tokens", ["rtx4060"], ["rtx4060"]),
            ("product_family", "same-family", "same-family"),
        )
        for field, left, right in cases:
            with self.subTest(field=field):
                left_task = task(100)
                right_task = task(101)
                left_task[field] = left
                right_task[field] = right
                normalized = [
                    normalize_task_facts(left_task),
                    normalize_task_facts(right_task),
                ]
                tasks_by_id = {row["task_id"]: row for row in normalized}
                with self.assertRaises(SplitLeakageError):
                    audit_split_leakage(
                        {"teacher_pool": [100], "final": [101]}, tasks_by_id
                    )

    def test_near_duplicate_queries_with_different_budgets_share_template(self):
        left = normalize_task_facts(
            {
                "task_id": 1,
                "query": "帮我买一台 RTX 4060 笔记本，预算 6000 元",
                "target_product_ids": ["asin-1"],
            }
        )
        right = normalize_task_facts(
            {
                "task_id": 2,
                "query": "帮我买一台RTX4060笔记本预算7000元",
                "target_product_ids": ["asin-2"],
            }
        )
        self.assertEqual(left["query_template"], right["query_template"])
        with self.assertRaises(SplitLeakageError):
            audit_split_leakage(
                {"teacher_pool": [1], "final": [2]},
                {1: left, 2: right},
            )

    def test_normalized_task_facts_round_trip(self):
        original = normalize_task_facts(task(7, template="budget-laptop"))
        self.assertEqual(normalize_task_facts(original), original)

    def test_task_facts_require_private_target_product_reference(self):
        with self.assertRaisesRegex(ValueError, "target_product_ids"):
            normalize_task_facts({"task_id": 1, "query": "买手机"})


if __name__ == "__main__":
    unittest.main()
