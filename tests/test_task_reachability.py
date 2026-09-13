import json
import re
import unicodedata
import unittest

from commerce_posttrain.data_quality.reachability import (
    build_reachability_manifest,
    classify_task,
    validate_reachability_manifest,
)


def reward_normalize(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text.replace("/", "|"))


def product(required, available):
    return {
        "instructions": [{"instruction_options": required}],
        "customization_options": {
            "颜色分类": [{"value": value} for value in available]
        },
    }


class TaskReachabilityTest(unittest.TestCase):
    def test_nfkc_damage_is_unreachable(self):
        row = classify_task(
            product(
                ["190系列2.0MM 3轨带纱门重型推拉门/?m"],
                ["190系列2.0MM 3轨带纱门重型推拉门/²m"],
            ),
            task_id=20,
            normalize_option_text=reward_normalize,
        )
        self.assertEqual(row["status"], "unreachable")
        self.assertEqual(row["reason_codes"], ["required_option_not_exposed"])
        self.assertEqual(row["unmatched_requirement_count"], 1)

    def test_format_only_differences_are_reachable(self):
        row = classify_task(
            product(["黑色 / XL"], ["黑色|xl"]),
            task_id=1,
            normalize_option_text=reward_normalize,
        )
        self.assertEqual(row["status"], "eligible")

    def test_manifest_is_content_addressed_and_fail_closed(self):
        products = [
            product([], []),
            product(["黑色"], ["黑色"]),
            product(["?m"], ["²m"]),
        ]
        split = {
            "splits": {
                "teacher_pool": {"task_ids": [1, 2]},
                "final": {"task_ids": [0]},
            }
        }
        manifest = build_reachability_manifest(
            products=products,
            split_manifest=split,
            normalize_option_text=reward_normalize,
            upstream_repository="https://example.test/upstream.git",
            upstream_commit="a" * 40,
            product_source_path="products.json.gz",
            product_source_compressed_sha256="b" * 64,
            product_source_decompressed_sha256="f" * 64,
            reward_features_path="reward_features.py",
            reward_features_sha256="c" * 64,
            reward_feature_version="reward-v1",
            verifier_sha256="1" * 64,
            task_facts_sha256="d" * 64,
            split_manifest_sha256="e" * 64,
        )
        validated = validate_reachability_manifest(
            manifest, expected_split_manifest_sha256="e" * 64
        )
        self.assertEqual(validated["selected_task_count"], 3)
        self.assertEqual(validated["eligible_task_count"], 2)
        self.assertEqual(validated["unreachable_task_count"], 1)
        self.assertEqual(
            validated["splits"]["teacher_pool"]["unreachable"][0]["task_id"],
            2,
        )

        tampered = json.loads(json.dumps(manifest))
        tampered["eligible_task_count"] += 1
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_reachability_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
