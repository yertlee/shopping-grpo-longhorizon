"""GRPO LoRA tensor-merge CLI 的单元测试（无网络、无真实模型）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.merge_grpo_lora import main, merge_state_dicts
from shopping_grpo.training.sft.run_manifest import FROZEN_MODEL_REVISION

try:
    from scripts.verify_merged_checkpoint import check_manifest_contract
except Exception:  # pragma: no cover - verifier 依赖缺失时跳过契约断言
    check_manifest_contract = None


def _write_fixture(root: Path) -> tuple[Path, Path]:
    base_dir = root / "base"
    adapter_dir = base_dir / "lora_adapter"
    base_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)

    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.visual.blocks.0.attn.qkv.weight": torch.full((2, 2), 0.5, dtype=torch.float32),
        "model.embed_tokens.weight": torch.ones((3, 2), dtype=torch.float32),
    }
    save_file(base, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    (base_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "dtype": "float32"}), encoding="utf-8"
    )
    (base_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    adapter = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.full((2, 2), 0.1, dtype=torch.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.eye(2, dtype=torch.float32),
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight":
            torch.full((2, 2), 0.3, dtype=torch.float32),
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight":
            torch.zeros((2, 2), dtype=torch.float32),
    }
    save_file(adapter, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "task_type": "CAUSAL_LM"}), encoding="utf-8"
    )
    return base_dir, adapter_dir


class MergeGrpoLoraTest(unittest.TestCase):
    def test_tensor_merge_adds_scaled_delta_and_keeps_other_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir, adapter_dir = _write_fixture(Path(tmp))
            output = Path(tmp) / "out"
            rc = main([
                "--base-dir", str(base_dir),
                "--output", str(output),
                "--source-run-id", "grpo-pilot-100/global_step_50",
            ])
            self.assertEqual(rc, 0)

            base = load_file(str(base_dir / "model.safetensors"))
            merged = load_file(str(output / "model.safetensors"))
            # delta = (I @ 0.1-block) * alpha/r = 0.1 * 2 = 0.2 per entry
            expected = (base["model.layers.0.self_attn.q_proj.weight"].float() + 0.2).to(torch.bfloat16)
            self.assertTrue(torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], expected))
            # non-LoRA weights untouched
            self.assertTrue(torch.equal(
                merged["model.embed_tokens.weight"], base["model.embed_tokens.weight"]
            ))

    def test_manifest_satisfies_verifier_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir, adapter_dir = _write_fixture(Path(tmp))
            output = Path(tmp) / "out"
            main([
                "--base-dir", str(base_dir), "--output", str(output),
                "--source-run-id", "grpo-pilot-100/global_step_50",
            ])
            manifest = json.loads((output / "merge_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["merge"]["base_revision"], FROZEN_MODEL_REVISION)
            self.assertEqual(manifest["merge"]["dtype"], "bfloat16")
            self.assertEqual(manifest["lora"]["applied_pairs"], 2)
            self.assertEqual(manifest["lora"]["missing_pairs"], 0)
            self.assertTrue(manifest["lora"]["vision_delta_all_zero"])
            # output config dtype must be normalized to the merged dtype
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["dtype"], "bfloat16")
            self.assertNotIn("torch_dtype", config)
            # output_files_sha256 must be a closed set over live files
            live = {p.name for p in output.iterdir() if p.is_file() and p.name != "merge_manifest.json"}
            self.assertEqual(set(manifest["output_files_sha256"]), live)
            if check_manifest_contract is not None:
                result = check_manifest_contract(manifest)
                self.assertTrue(result["passed"], result["detail"])

    def test_missing_base_target_fails_closed(self):
        base = {"model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2))}
        adapter = {
            "base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight": torch.zeros((2, 2)),
            "base_model.model.model.layers.9.self_attn.q_proj.lora_B.weight": torch.zeros((2, 2)),
        }
        with self.assertRaises(SystemExit):
            merge_state_dicts(base, adapter, scale=2.0, dtype=torch.bfloat16)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
