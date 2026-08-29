"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_grpo_runtime
from scripts.check_grpo_runtime import (
    PATCH_MARKER,
    compose_runtime_config,
    validate_dynamic_sampling,
    validate_training_memory_budget,
)


class DynamicSamplingConfigTest(unittest.TestCase):
    def test_training_memory_budget_enforces_real_micro_batch_one(self):
        config = compose_runtime_config([])
        validate_training_memory_budget(config)
        self.assertEqual(config.data.max_response_length, 20480)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 24576)
        self.assertFalse(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertTrue(config.actor_rollout_ref.actor.calculate_entropy)
        self.assertEqual(
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(
            config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
        )
        self.assertEqual(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz)
        self.assertEqual(
            config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, 1
        )

    def test_training_memory_budget_rejects_unsafe_overrides(self):
        unsafe_response = compose_runtime_config(["data.max_response_length=24576"])
        with self.assertRaisesRegex(SystemExit, "unsafe GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        dynamic_actor = compose_runtime_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be false"):
            validate_training_memory_budget(dynamic_actor)

        dynamic_rollout_log_prob = compose_runtime_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be false"
        ):
            validate_training_memory_budget(dynamic_rollout_log_prob)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=3",
                "shopping_dynamic_sampling.max_consecutive_skipped_updates=10",
                "shopping_dynamic_sampling.reward_tolerance=1e-8",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 3)
        self.assertEqual(
            config.shopping_dynamic_sampling.max_consecutive_skipped_updates, 10
        )
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 1.0e-8)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_enabled_config_requires_installed_patch_marker(self):
        config = compose_runtime_config(["shopping_dynamic_sampling.enable=true"])
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "patch marker is missing"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            # marker 之外还必须比对最终 patched SHA256（audit item 5）：
            # 带 marker 但 SHA 不匹配的文件同样拒绝。
            trainer_source.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            actual = hashlib.sha256(trainer_source.read_bytes()).hexdigest()
            with patch.object(check_grpo_runtime, "EXPECTED_PATCHED_SHA256", actual):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

    def test_ref_must_not_override_actor_model_path(self):
        """冻结 ref = actor 同一个 M2 merged；ref.model 覆盖必须被拒绝。"""
        config = compose_runtime_config([])
        self.assertNotIn("model", config.actor_rollout_ref.ref)
        check_grpo_runtime.validate_actor_ref_model_contract(
            {"actor_rollout_ref": dict(config.actor_rollout_ref)}
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
