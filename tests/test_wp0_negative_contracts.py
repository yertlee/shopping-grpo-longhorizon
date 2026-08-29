"""Independent negative probes for the frozen SFT/merge contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_lora_sft import _validate_resume_checkpoint
from scripts.verify_merged_checkpoint import verify_merged_dir
from scripts.verify_sft_adapter import verify_run_dir
from shopping_grpo.training.sft.reload_check import ForwardResult
from shopping_grpo.training.sft.run_manifest import FROZEN_MODEL_REVISION, compute_run_id, hash_weight_files


def _loaders(calls):
    def load_model(path, dtype="bf16", revision=None):
        calls.append(("model", str(path), revision))
        return {"path": str(path), "dtype": dtype}

    def load_tokenizer(path, revision=None):
        calls.append(("tokenizer", str(path), revision))
        return {"path": str(path)}

    return {
        "load_model": load_model,
        "load_tokenizer": load_tokenizer,
        "load_peft_adapter": lambda base, adapter: base,
        "adapter_weight_keys": lambda path: ["x.lora_A.q_proj", "x.lora_B.q_proj"],
        "trainable_stats": lambda model: (2, 100),
        "short_forward": lambda model, tokenizer, device: ForwardResult(True, device, [1, 1, 1], True),
    }


class Wp0NegativeContractsTest(unittest.TestCase):
    def test_revision_is_not_pairwise_metadata_only(self):
        self.assertNotEqual(FROZEN_MODEL_REVISION, "deadbeef")

    def test_loader_receives_frozen_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            (base / "model.safetensors").write_bytes(b"base")
            # tokenizer 身份校验要求 base 与 run 目录的 tokenizer 文件一致。
            (base / "tokenizer_config.json").write_text("{}")
            run = root / "run"
            run.mkdir()
            (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base), "target_modules": ["q_proj"]}))
            (run / "adapter_model.safetensors").write_bytes(b"adapter")
            (run / "tokenizer_config.json").write_text("{}")
            (run / "train_summary.json").write_text(json.dumps({"train_loss": 1.0, "metrics": {}, "lora": {"trainable_parameters": 2, "total_parameters": 100}}))
            manifest = {"schema_version": "shopping-sft-run-manifest-v1", "stage": "smoke", "model": {"path_or_repo": str(base), "revision": FROZEN_MODEL_REVISION, "weights_sha256": hash_weight_files(base)["weights_sha256"]}, "data": {"train_sha256": "x"}, "recipe": {"r": 1}, "runtime": {}}
            manifest["run_id"] = compute_run_id(manifest)
            manifest.update({"execution": {"exit_code": 0}, "result": {}})
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            calls = []
            _, passed = verify_run_dir(run, loaders=_loaders(calls))
            self.assertTrue(passed)
            self.assertIn(("model", str(base), FROZEN_MODEL_REVISION), calls)

    def test_pseudo_run_id_is_rejected_for_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"; base.mkdir()
            (base / "model.safetensors").write_bytes(b"base")
            run = root / "run"; run.mkdir()
            for name, content in (("adapter_config.json", json.dumps({"base_model_name_or_path": str(base), "target_modules": ["q_proj"]})), ("adapter_model.safetensors", "adapter"), ("tokenizer_config.json", "{}"), ("train_summary.json", json.dumps({"train_loss": 1.0, "metrics": {}, "lora": {"trainable_parameters": 2, "total_parameters": 100}}))):
                (run / name).write_text(content) if not name.endswith("safetensors") else (run / name).write_bytes(content.encode())
            manifest = {"schema_version": "shopping-sft-run-manifest-v1", "run_id": "dead" * 16, "stage": "smoke", "model": {"path_or_repo": str(base), "revision": FROZEN_MODEL_REVISION, "weights_sha256": hash_weight_files(base)["weights_sha256"]}, "data": {"train_sha256": "x"}, "recipe": {"r": 1}, "runtime": {}, "execution": {"exit_code": 0}, "result": {}}
            (run / "run_manifest.json").write_text(json.dumps(manifest))
            _, passed = verify_run_dir(run, loaders=_loaders([]))
            self.assertFalse(passed)

    def test_merge_input_addition_is_rejected(self):
        from scripts.verify_merged_checkpoint import _hash_dir_matching
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); (directory / "known").write_bytes(b"x"); (directory / "attacker").write_bytes(b"x")
            result = _hash_dir_matching(directory, {"known": "2d711642b726b04401627ca9fbac32f5da7e5f2f6f0f1c5fca6f3a5e3f8f7f3e"})
            self.assertIn("attacker", result["unexpected"])

    def test_resume_missing_and_foreign_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / "run"; output.mkdir(); foreign = root / "foreign"; foreign.mkdir()
            manifest = {"execution": {"checkpoint_paths": []}}
            with self.assertRaises(SystemExit): _validate_resume_checkpoint(str(output / "checkpoint-1"), output, manifest)
            checkpoint = foreign / "checkpoint-1"; checkpoint.mkdir(); (checkpoint / "trainer_state.json").write_text("{}")
            with self.assertRaises(SystemExit): _validate_resume_checkpoint(str(checkpoint), output, manifest)


if __name__ == "__main__":
    unittest.main()
