"""验证 LoRA 合并入口的纯配置逻辑，不需要本地下载模型。"""

import tempfile
import unittest
from pathlib import Path

from scripts.merge_lora_adapter import (
    _manifest_path_identity,
    build_merge_manifest,
    choose_model_class,
)


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


class MergeLoraAdapterTest(unittest.TestCase):
    def test_manifest_identity_resolves_local_paths_but_preserves_repo_ids(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            relative = Path(tmpdir).relative_to(Path.cwd())
            self.assertEqual(
                _manifest_path_identity(relative, local_required=True),
                str(Path(tmpdir).resolve()),
            )
        self.assertEqual(
            _manifest_path_identity("Qwen/Qwen3.5-2B", local_required=False),
            "Qwen/Qwen3.5-2B",
        )

    def test_qwen35_uses_multimodal_model_class(self):
        self.assertEqual(choose_model_class(_Config("qwen3_5"), "causal", "multimodal"), "multimodal")
        self.assertEqual(choose_model_class(_Config("qwen3"), "causal", "multimodal"), "causal")

    def test_merge_manifest_is_auditable(self):
        manifest = build_merge_manifest(
            base_model="Qwen/Qwen3.5-2B",
            adapter_path="checkpoints/sft",
            output_path="checkpoints/sft_merged",
            model_type="qwen3_5",
        )
        self.assertEqual(manifest["operation"], "peft_merge_and_unload")
        self.assertEqual(manifest["source"]["adapter"], "checkpoints/sft")
        self.assertEqual(manifest["output"], "checkpoints/sft_merged")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
