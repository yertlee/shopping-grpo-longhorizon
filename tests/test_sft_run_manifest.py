"""SFT run manifest 构建器的纯逻辑测试（不依赖 torch / transformers）。"""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.sft.run_manifest import (
    ManifestSecretError,
    CHECKPOINT_OWNER_FILE,
    build_run_manifest,
    compute_run_id,
    finalize_run_manifest,
    hash_weight_files,
    load_run_manifest,
    resolve_successful_run_manifest_path,
    write_run_manifest,
    write_checkpoint_owner,
    validate_checkpoint_owner,
)


def _make_manifest(tmpdir: Path, *, train_name="train.jsonl", recipe_extra=None):
    train = tmpdir / train_name
    train.write_text('{"task_id": 1}\n', encoding="utf-8")
    weights = {
        "weights_sha256": "a" * 64,
        "weight_files": {"model.safetensors": "b" * 64},
        "weights_sha256_note": "test",
    }
    recipe = {
        "max_length": 24576,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
        "seed": 42,
    }
    recipe.update(recipe_extra or {})
    return build_run_manifest(
        stage="smoke",
        model_path="/models/Qwen3.5-2B",
        model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        weights=weights,
        train_path=train,
        validation_path=None,
        sft_ready_manifest_path=None,
        preflight_report_path=None,
        recipe=recipe,
        code_hashes={"scripts/train_lora_sft.py": "c" * 64},
        dependency_versions={"torch": "2.12.1"},
        gpu={"available": False, "name": None, "device_count": 0, "cuda_runtime": None},
        command=["train_lora_sft.py", "--stage", "smoke"],
        resume_from_checkpoint=None,
    )


class SftRunManifestTest(unittest.TestCase):
    def test_resolver_selects_latest_successful_resume_for_interrupted_canonical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = _make_manifest(root)
            write_run_manifest(root / "run_manifest.json", canonical)

            failed = json.loads(json.dumps(canonical))
            finalize_run_manifest(
                failed,
                train_examples=1,
                validation_examples=0,
                train_loss=None,
                eval_loss=None,
                peak_gpu_memory_gib=None,
                checkpoint_paths=["checkpoint-1"],
                exit_code=1,
            )
            write_run_manifest(root / "run_manifest.resume.1.json", failed)

            successful = json.loads(json.dumps(canonical))
            finalize_run_manifest(
                successful,
                train_examples=1,
                validation_examples=0,
                train_loss=0.25,
                eval_loss=None,
                peak_gpu_memory_gib=1.0,
                checkpoint_paths=["checkpoint-2"],
                exit_code=0,
            )
            write_run_manifest(root / "run_manifest.resume.2.json", successful)

            self.assertEqual(
                resolve_successful_run_manifest_path(root),
                root / "run_manifest.resume.2.json",
            )

    def test_resolver_rejects_resume_from_different_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = _make_manifest(root)
            write_run_manifest(root / "run_manifest.json", canonical)
            other = _make_manifest(root, train_name="other.jsonl", recipe_extra={"seed": 7})
            finalize_run_manifest(
                other,
                train_examples=1,
                validation_examples=0,
                train_loss=0.25,
                eval_loss=None,
                peak_gpu_memory_gib=1.0,
                checkpoint_paths=["checkpoint-2"],
                exit_code=0,
            )
            write_run_manifest(root / "run_manifest.resume.1.json", other)

            with self.assertRaisesRegex(ValueError, "run_id mismatch"):
                resolve_successful_run_manifest_path(root)

    def test_manifest_records_inputs_and_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest(Path(tmpdir))
            self.assertEqual(manifest["schema_version"], "shopping-sft-run-manifest-v1")
            self.assertEqual(manifest["stage"], "smoke")
            self.assertEqual(manifest["model"]["revision"], "15852e8c16360a2fea060d615a32b45270f8a8fc")
            self.assertIsNotNone(manifest["data"]["train_sha256"])
            self.assertEqual(manifest["execution"]["exit_code"], None)
            self.assertEqual(manifest["result"]["train_loss"], None)
            self.assertRegex(manifest["run_id"], r"^[0-9a-f]{64}$")

            rebuilt = _make_manifest(Path(tmpdir))
            self.assertEqual(manifest["run_id"], rebuilt["run_id"])

    def test_run_id_changes_when_data_changes_but_not_when_result_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = _make_manifest(Path(tmpdir), train_name="train.jsonl")
            second = _make_manifest(Path(tmpdir), train_name="other.jsonl")
            self.assertNotEqual(first["run_id"], second["run_id"])

            finalized = finalize_run_manifest(
                dict(first),
                train_examples=8,
                validation_examples=0,
                train_loss=0.5,
                eval_loss=None,
                peak_gpu_memory_gib=12.3,
                checkpoint_paths=["checkpoint-2"],
                exit_code=0,
            )
            self.assertEqual(first["run_id"], compute_run_id(finalized))
            self.assertEqual(finalized["execution"]["exit_code"], 0)
            self.assertEqual(finalized["result"]["train_loss"], 0.5)

    def test_double_finalize_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest(Path(tmpdir))
            finalize_run_manifest(
                manifest,
                train_examples=1,
                validation_examples=0,
                train_loss=None,
                eval_loss=None,
                peak_gpu_memory_gib=None,
                checkpoint_paths=[],
                exit_code=0,
            )
            with self.assertRaises(ValueError):
                finalize_run_manifest(
                    manifest,
                    train_examples=1,
                    validation_examples=0,
                    train_loss=None,
                    eval_loss=None,
                    peak_gpu_memory_gib=None,
                    checkpoint_paths=[],
                    exit_code=0,
                )

    def test_secret_keys_are_rejected_everywhere(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ManifestSecretError):
                _make_manifest(Path(tmpdir), recipe_extra={"api_key": "should-not-appear"})
            with self.assertRaises(ManifestSecretError):
                _make_manifest(Path(tmpdir), recipe_extra={"run_config": {"DEEPSEEK_TOKEN_NAME": "x"}})
            # 环境变量只允许记录名字；包含 "vars" 这类普通键名不触发。
            manifest = _make_manifest(Path(tmpdir), recipe_extra={"env_var_names": ["CUDA_VISIBLE_DEVICES"]})
            self.assertEqual(manifest["recipe"]["env_var_names"], ["CUDA_VISIBLE_DEVICES"])

    def test_write_load_roundtrip_and_unknown_top_level_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest(Path(tmpdir))
            path = Path(tmpdir) / "run_manifest.json"
            write_run_manifest(path, manifest)
            loaded = load_run_manifest(path)
            self.assertEqual(loaded["run_id"], manifest["run_id"])

            bogus = dict(loaded)
            bogus["surprise"] = True
            with self.assertRaises(ValueError):
                load_run_manifest.__module__  # noqa: B018 - 占位避免误读
                from shopping_grpo.training.sft.run_manifest import validate_top_level_keys

                validate_top_level_keys(bogus)

    def test_hash_weight_files_local_dir_and_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "config.json").write_text("{}", encoding="utf-8")
            result = hash_weight_files(root)
            self.assertIn("model.safetensors", result["weight_files"])
            self.assertNotIn("config.json", result["weight_files"])
            self.assertRegex(result["weights_sha256"], r"^[0-9a-f]{64}$")

            missing = hash_weight_files(root / "nope")
            self.assertIsNone(missing["weights_sha256"])
            self.assertIn("not a local directory", missing["weights_sha256_note"])

    def test_json_payload_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest(Path(tmpdir))
            first = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            second = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
            self.assertEqual(first, second)

    def test_checkpoint_owner_binds_recursive_exact_file_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text('{"global_step": 4}', encoding="utf-8")
            (checkpoint / "optimizer").mkdir()
            (checkpoint / "optimizer" / "state.pt").write_bytes(b"state")
            sidecar = write_checkpoint_owner(
                checkpoint,
                output_dir=root,
                run_id="a" * 64,
                global_step=4,
            )
            owner = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual([item["path"] for item in owner["files"]], ["optimizer/state.pt", "trainer_state.json"])
            validate_checkpoint_owner(
                owner,
                run_id="a" * 64,
                checkpoint_path="checkpoint-4",
                global_step=4,
                checkpoint_dir=checkpoint,
            )
            (checkpoint / "unexpected.bin").write_bytes(b"new")
            with self.assertRaises(ValueError):
                validate_checkpoint_owner(
                    owner,
                    run_id="a" * 64,
                    checkpoint_path="checkpoint-4",
                    global_step=4,
                    checkpoint_dir=checkpoint,
                )

    def test_checkpoint_owner_rejects_sidecar_tamper_and_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text('{"global_step": 4}', encoding="utf-8")
            write_checkpoint_owner(checkpoint, output_dir=root, run_id="b" * 64, global_step=4)
            owner_path = checkpoint / "checkpoint_owner.json"
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            owner["run_id"] = "c" * 64
            owner_path.write_text(json.dumps(owner), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_checkpoint_owner(
                    owner,
                    run_id="b" * 64,
                    checkpoint_path="checkpoint-4",
                    global_step=4,
                    checkpoint_dir=checkpoint,
                )

    def test_checkpoint_owner_rejects_content_edit_and_internal_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            state = checkpoint / "trainer_state.json"
            state.write_text('{"global_step": 4}', encoding="utf-8")
            write_checkpoint_owner(checkpoint, output_dir=root, run_id="d" * 64, global_step=4)
            state.write_text('{"global_step": 4, "tampered": true}', encoding="utf-8")
            owner = json.loads((checkpoint / CHECKPOINT_OWNER_FILE).read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                validate_checkpoint_owner(
                    owner,
                    run_id="d" * 64,
                    checkpoint_path="checkpoint-4",
                    global_step=4,
                    checkpoint_dir=checkpoint,
                )

            copied = root / "checkpoint-5"
            import shutil

            shutil.copytree(checkpoint, copied)
            copied_owner = json.loads((copied / CHECKPOINT_OWNER_FILE).read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                validate_checkpoint_owner(
                    copied_owner,
                    run_id="d" * 64,
                    checkpoint_path="checkpoint-5",
                    global_step=4,
                    checkpoint_dir=copied,
                )

    def test_checkpoint_owner_rejects_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint-4"
            checkpoint.mkdir()
            state = checkpoint / "trainer_state.json"
            state.write_text('{"global_step": 4}', encoding="utf-8")
            (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
            write_checkpoint_owner(checkpoint, output_dir=root, run_id="e" * 64, global_step=4)
            owner = json.loads((checkpoint / CHECKPOINT_OWNER_FILE).read_text(encoding="utf-8"))
            (checkpoint / "scheduler.pt").unlink()
            with self.assertRaises(ValueError):
                validate_checkpoint_owner(
                    owner,
                    run_id="e" * 64,
                    checkpoint_path="checkpoint-4",
                    global_step=4,
                    checkpoint_dir=checkpoint,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
