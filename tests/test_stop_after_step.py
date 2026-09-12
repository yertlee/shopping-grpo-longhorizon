"""Pure tests for the internal exact-checkpoint stop barrier."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_grpo
from scripts.check_grpo_runtime import validate_synchronous_checkpoint_save
from shopping_grpo.training.grpo.dynamic_sampling import (
    parse_stop_after_step,
    record_controlled_stop_after_checkpoint,
    should_controlled_stop,
    validate_stop_after_step,
)


class StopAfterStepTests(unittest.TestCase):
    def test_no_env_preserves_normal_500_step_recipe(self):
        self.assertIsNone(parse_stop_after_step(None))
        args = train_grpo.parse_args([])
        self.assertIsNone(args.stop_after_step)
        self.assertFalse(any("stop_after" in item for item in train_grpo.hydra_overrides(args)))

    def test_invalid_values_are_rejected(self):
        for value in ("0", "-1", "1.5", "01", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_stop_after_step(value)
        with self.assertRaises(ValueError):
            validate_stop_after_step(100, 30)
        valid = train_grpo.parse_args(["--stop-after-step", "100"])
        train_grpo._validate_stop_after_step(valid)
        invalid_boundary = train_grpo.parse_args(["--stop-after-step", "101"])
        with self.assertRaises(SystemExit):
            train_grpo._validate_stop_after_step(invalid_boundary)
        short_contract = train_grpo.parse_args(
            ["--stop-after-step", "100", "--", "trainer.total_training_steps=2"]
        )
        with self.assertRaises(SystemExit):
            train_grpo._validate_stop_after_step(short_contract)

    def test_stop_barrier_rejects_explicit_async_checkpoint_save(self):
        with patch.dict("os.environ", {"SHOPPING_GRPO_STOP_AFTER_STEP": "100"}, clear=False):
            with self.assertRaises(SystemExit):
                validate_synchronous_checkpoint_save(
                    {
                        "actor_rollout_ref": {
                            "actor": {"checkpoint": {"async_save": True}}
                        }
                    }
                )
            validate_synchronous_checkpoint_save(
                {"actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}}}
            )
            with self.assertRaises(SystemExit):
                validate_synchronous_checkpoint_save(
                    {"actor_rollout_ref": {"actor": {"checkpoint": {"async_save": "unknown"}}}}
                )

    def test_actor_checkpoint_async_save_is_explicitly_fail_closed(self):
        default_args = train_grpo.parse_args(["--stop-after-step", "100"])
        evidence = train_grpo._actor_checkpoint_async_save_evidence(default_args)
        self.assertFalse(evidence["value"])
        self.assertEqual(evidence["source"], "veRL-0.8-default")

        false_args = train_grpo.parse_args(
            ["--stop-after-step", "100", "--", train_grpo.ACTOR_CHECKPOINT_ASYNC_SAVE_PATH + "=false"]
        )
        self.assertFalse(train_grpo._actor_checkpoint_async_save_evidence(false_args)["value"])
        true_args = train_grpo.parse_args(
            ["--stop-after-step", "100", "--", train_grpo.ACTOR_CHECKPOINT_ASYNC_SAVE_PATH + "=true"]
        )
        with self.assertRaises(SystemExit):
            train_grpo._actor_checkpoint_async_save_evidence(true_args)
        unknown_args = train_grpo.parse_args(
            [
                "--stop-after-step", "100", "--",
                train_grpo.ACTOR_CHECKPOINT_ASYNC_SAVE_PATH + "=${oc.env:ASYNC_SAVE}",
            ]
        )
        with self.assertRaises(SystemExit):
            train_grpo._actor_checkpoint_async_save_evidence(unknown_args)
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "null.yaml"
            config.write_text(
                "actor_rollout_ref:\n  actor:\n    checkpoint:\n      async_save: null\n",
                encoding="utf-8",
            )
            null_args = train_grpo.parse_args(
                ["--config", str(config), "--stop-after-step", "100"]
            )
            with self.assertRaises(SystemExit):
                train_grpo._actor_checkpoint_async_save_evidence(null_args)

    def test_launcher_passes_stop_target_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = root / "train.parquet"
            val = root / "validation.parquet"
            train.write_bytes(b"train")
            val.write_bytes(b"val")
            config = root / "grpo.yaml"
            shutil.copy2(train_grpo.DEFAULT_CONFIG, config)
            args = train_grpo.parse_args(
                [
                    "--model", str(model), "--train-data", str(train),
                    "--val-data", str(val), "--config", str(config),
                    "--output", str(root / "out"), "--stop-after-step", "100",
                ]
            )
            _, environment = train_grpo.build_command(args)
            self.assertEqual(environment["SHOPPING_GRPO_STOP_AFTER_STEP"], "100")
            normal = train_grpo.parse_args(
                [
                    "--model", str(model), "--train-data", str(train),
                    "--val-data", str(val), "--config", str(config),
                    "--output", str(root / "out-normal"),
                ]
            )
            _, normal_environment = train_grpo.build_command(normal)
            self.assertNotIn("SHOPPING_GRPO_STOP_AFTER_STEP", normal_environment)

    def test_skipped_update_does_not_count_as_a_step(self):
        self.assertFalse(
            should_controlled_stop(
                target_step=100, current_step=100, optimizer_updated=False, save_freq=50
            )
        )
        self.assertFalse(
            should_controlled_stop(
                target_step=100, current_step=99, optimizer_updated=True, save_freq=50
            )
        )

    def test_step_100_is_the_first_natural_stop_boundary(self):
        self.assertFalse(
            should_controlled_stop(
                target_step=100, current_step=99, optimizer_updated=True, save_freq=50
            )
        )
        self.assertTrue(
            should_controlled_stop(
                target_step=100, current_step=100, optimizer_updated=True, save_freq=50
            )
        )
        self.assertFalse(
            should_controlled_stop(
                target_step=100, current_step=150, optimizer_updated=True, save_freq=50
            )
        )

    def test_marker_and_diagnostic_are_written_after_save(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actor = root / "global_step_100" / "actor"
            actor.mkdir(parents=True)
            for name in (
                "model_world_size_1_rank_0.pt",
                "optim_world_size_1_rank_0.pt",
                "extra_state_world_size_1_rank_0.pt",
            ):
                (actor / name).write_bytes(b"checkpoint")
            (root / "latest_checkpointed_iteration.txt").write_text("100\n", encoding="utf-8")
            marker = root / "controlled_stop_after_checkpoint.json"
            diagnostic = root / "training_diagnostics.jsonl"
            record = record_controlled_stop_after_checkpoint(
                marker,
                diagnostic,
                global_step=100,
                target_step=100,
                save_freq=50,
            )
            self.assertEqual(json.loads(marker.read_text())["event"], "controlled_stop_after_checkpoint")
            self.assertEqual(json.loads(diagnostic.read_text())["global_step"], 100)
            self.assertTrue(record["checkpoint_boundary"])

    def test_save_failure_cannot_emit_marker_or_success_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "controlled_stop_after_checkpoint.json"
            diagnostic = root / "training_diagnostics.jsonl"
            with self.assertRaises(RuntimeError):
                record_controlled_stop_after_checkpoint(
                    marker, diagnostic, global_step=100, target_step=100, save_freq=50
                )
            self.assertFalse(marker.exists())
            self.assertFalse(diagnostic.exists())

    def test_manifest_records_controlled_stop_and_launcher_keeps_exit_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actor = root / "global_step_100" / "actor"
            actor.mkdir(parents=True)
            for name in (
                "model_world_size_1_rank_0.pt",
                "optim_world_size_1_rank_0.pt",
                "extra_state_world_size_1_rank_0.pt",
            ):
                (actor / name).write_bytes(b"checkpoint")
            (root / "latest_checkpointed_iteration.txt").write_text("100\n", encoding="utf-8")
            marker = root / "controlled_stop_after_checkpoint.json"
            record_controlled_stop_after_checkpoint(
                marker,
                root / "training_diagnostics.jsonl",
                global_step=100,
                target_step=100,
                save_freq=50,
            )
            checkpoint_manifest = train_grpo.build_checkpoint_manifest(root, exit_code=0)
            self.assertEqual(checkpoint_manifest["exit_code"], 0)
            self.assertEqual(checkpoint_manifest["latest_global_step"], 100)
            self.assertEqual(json.loads(marker.read_text())["global_step"], 100)
            self.assertTrue(json.loads(marker.read_text())["checkpoint_boundary"])

    def test_runtime_patch_has_internal_barrier_after_update_and_before_loop_end(self):
        v4_patch_text = (
            Path(__file__).resolve().parents[1]
            / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
        ).read_text(encoding="utf-8")
        patch_text = v4_patch_text
        self.assertIn("SHOPPING_GRPO_STEP_BARRIER_V1", patch_text)
        self.assertIn("SHOPPING_GRPO_STOP_AFTER_STEP", patch_text)
        self.assertIn("record_controlled_stop_after_checkpoint", patch_text)
        self.assertIn("should_controlled_stop", patch_text)
        self.assertNotIn("SIGTERM", patch_text)
        self.assertIn("actor_output = self._update_actor(batch)", patch_text)
        self.assertIn("# The synchronous save block above ran at this exact save boundary.", patch_text)
        self.assertLess(
            patch_text.rindex("record_controlled_stop_after_checkpoint"),
            patch_text.rindex("if is_last_step:"),
        )


if __name__ == "__main__":
    unittest.main()
