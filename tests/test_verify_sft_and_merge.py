"""adapter / merged checkpoint 验收器与 merge manifest 的低成本测试。

全部使用注入的 fake loader，不下载模型、不 import torch。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.merge_lora_adapter import build_merge_manifest, choose_model_class
from scripts.train_lora_sft import _resume_execution_manifest_path
from scripts.verify_merged_checkpoint import verify_merged_dir
from scripts.verify_sft_adapter import verify_run_dir
from shopping_grpo.training.sft.reload_check import (
    ForwardResult,
    check_tokenizer_identity,
    default_reload_loaders,
)
from shopping_grpo.training.sft.run_manifest import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]


class ResumeManifestSelectionTest(unittest.TestCase):
    def test_pending_canonical_is_finalized_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = {"execution": {"exit_code": None}}
            self.assertEqual(
                _resume_execution_manifest_path(root, manifest),
                root / "run_manifest.json",
            )

    def test_finalized_execution_gets_next_immutable_resume_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "run_manifest.resume.1.json").write_text("{}", encoding="utf-8")
            manifest = {"execution": {"exit_code": 0}}
            self.assertEqual(
                _resume_execution_manifest_path(root, manifest),
                root / "run_manifest.resume.2.json",
            )


class _SemanticTokenizer:
    """Small loader contract fake mirroring the fields used by the verifier."""

    def __init__(self, path):
        import hashlib

        path = Path(path)
        material = b"".join(
            (path / name).read_bytes()
            for name in ("tokenizer_config.json", "vocab.json", "merges.txt")
            if (path / name).is_file()
        )
        digest = hashlib.sha256(material).hexdigest()
        self._vocab = {"<pad>": 0, "token": int(digest[:8], 16) % 1000 + 1}
        self._added = {"<added>": 1001}
        self.vocab_size = 1002
        self.all_special_tokens = ["<pad>", "<eos>"]
        self.all_special_ids = [0, 2]
        self.special_tokens_map = {"pad_token": "<pad>", "eos_token": "<eos>"}
        self.chat_template = "{{ messages }}"
        self.model_input_names = ["input_ids", "attention_mask"]

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added

    def __call__(self, *args, **kwargs):
        return {"input_ids": [[1]], "attention_mask": [[1]]}


def _fake_loaders(*, forward_ok=True, trainable=(2048, 1_000_000), adapter_keys=None, tokenizer_tamper=None):
    def load_tokenizer(path, revision=None):
        tokenizer = _SemanticTokenizer(path)
        if tokenizer_tamper and Path(path).name in {"adapter-run", "merged"}:
            if tokenizer_tamper == "special":
                tokenizer.all_special_ids = [0, 99]
            elif tokenizer_tamper == "chat":
                tokenizer.chat_template = "{{ changed_messages }}"
        return tokenizer

    return {
        "load_model": lambda path, dtype="bf16": {"path": str(path), "dtype": dtype},
        "load_tokenizer": load_tokenizer,
        "load_peft_adapter": lambda base, adapter: {"base": base, "adapter": str(adapter)},
        "adapter_weight_keys": lambda path: adapter_keys
        or [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        ],
        "trainable_stats": lambda model: trainable,
        "short_forward": lambda model, tokenizer, device: ForwardResult(
            ok=forward_ok, device=device, logits_shape=[1, 8, 128], finite=forward_ok
        ),
    }


def _make_base_model(tmpdir: Path, *, name="Qwen3.5-2B") -> Path:
    """带假权重分片与 tokenizer 文件的本地基座目录；hash 由真实计算得出。"""
    from shopping_grpo.training.sft.run_manifest import hash_weight_files

    base = tmpdir / "models" / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.safetensors").write_bytes(b"base-weights")
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer_config.json").write_text('{"model": "qwen"}', encoding="utf-8")
    (base / "vocab.json").write_bytes(b"base-vocab")
    (base / "merges.txt").write_bytes(b"base-merges")
    hash_weight_files(base)  # 确认可计算
    return base


def _make_adapter_run(
    tmpdir: Path,
    *,
    train_loss=0.5,
    exit_code=0,
    base: Path | None = None,
    weights_sha256: str | None = None,
    revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
):
    from shopping_grpo.training.sft.run_manifest import hash_weight_files

    base = base or _make_base_model(tmpdir)
    run_dir = tmpdir / "adapter-run"
    run_dir.mkdir()
    (run_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(base),
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
    # tokenizer 文件从基座复制（与真实 trainer 行为一致）
    (run_dir / "tokenizer_config.json").write_bytes(
        (base / "tokenizer_config.json").read_bytes()
    )
    (run_dir / "vocab.json").write_bytes((base / "vocab.json").read_bytes())
    (run_dir / "merges.txt").write_bytes((base / "merges.txt").read_bytes())
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "train_loss": train_loss,
                "metrics": {},
                "lora": {
                    "trainable_parameters": 2048,
                    "total_parameters": 1_000_000,
                    "trainable_ratio": 2048 / 1_000_000,
                    "target_modules": ["q_proj"],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "shopping-sft-run-manifest-v1",
                "run_id": "0" * 64,
                "stage": "smoke",
                "model": {
                    "path_or_repo": str(base),
                    "revision": revision,
                    "weights_sha256": weights_sha256
                    or hash_weight_files(base)["weights_sha256"],
                },
                "data": {},
                "recipe": {},
                "runtime": {"code_hashes": {}, "dependency_versions": {}, "gpu": None},
                "execution": {"command": [], "resume_from_checkpoint": None, "checkpoint_paths": [], "exit_code": exit_code, "error": None},
                "result": {"train_loss": None, "eval_loss": None, "peak_gpu_memory_gib": None, "adapter_reload": {"passed": None}},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class VerifySftAdapterTest(unittest.TestCase):
    def test_interrupted_canonical_uses_successful_resume_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            canonical_path = run_dir / "run_manifest.json"
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            canonical["execution"]["exit_code"] = None
            canonical_path.write_text(json.dumps(canonical), encoding="utf-8")

            resumed = json.loads(json.dumps(canonical))
            resumed["execution"]["exit_code"] = 0
            resume_path = run_dir / "run_manifest.resume.1.json"
            resume_path.write_text(json.dumps(resumed), encoding="utf-8")

            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())

            self.assertTrue(passed)
            canonical_after = json.loads(canonical_path.read_text(encoding="utf-8"))
            resumed_after = json.loads(resume_path.read_text(encoding="utf-8"))
            self.assertIsNone(canonical_after["execution"]["exit_code"])
            self.assertTrue(resumed_after["result"]["adapter_reload"]["passed"])

    def test_default_peft_reload_enables_trainable_parameters(self):
        import sys
        import types

        calls = []

        class _PeftModel:
            @classmethod
            def from_pretrained(cls, base, path, **kwargs):
                calls.append(kwargs)
                return cls()

        with patch.dict(sys.modules, {"peft": types.SimpleNamespace(PeftModel=_PeftModel)}):
            loaders = default_reload_loaders()
            loaders["load_peft_adapter"]("base", "adapter")
        self.assertEqual(calls, [{"is_trainable": True}])

    def test_fake_loader_missing_semantic_contract_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_tokenizer_identity(
                Path(tmpdir), Path(tmpdir), output_tokenizer={}, base_tokenizer={}
            )
        self.assertFalse(result["passed"])
        self.assertIn("缺少 tokenizer 语义字段", result["detail"]["problems"][0])

    def test_tokenizer_identity_rejects_special_and_chat_semantic_tamper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            for tamper in ("special", "chat"):
                _, passed = verify_run_dir(
                    run_dir, loaders=_fake_loaders(tokenizer_tamper=tamper)
                )
                self.assertFalse(passed)

    def test_happy_path_passes_and_patches_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertTrue(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["result"]["adapter_reload"]["passed"])
            names = {check["name"] for check in manifest["result"]["adapter_reload"]["checks"]}
            self.assertIn("short_forward_finite", names)
            self.assertIn("trainable_params_match_summary", names)

    def test_nonfinite_loss_and_bad_exit_code_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir), train_loss=float("nan"), exit_code=1)
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_trainable_param_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            _, passed = verify_run_dir(
                run_dir, loaders=_fake_loaders(trainable=(999, 1_000_000))
            )
            self.assertFalse(passed)

    def test_base_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            config = json.loads((run_dir / "adapter_config.json").read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = "/models/SomeOtherModel"
            (run_dir / "adapter_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_same_basename_different_path_fails(self):
        """审计 P0-A.1：basename 相同但路径不同必须失败，不能用 basename 比较。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = _make_base_model(root)
            run_dir = _make_adapter_run(root, base=base)
            impostor = root / "elsewhere" / base.name
            impostor.mkdir(parents=True)
            (impostor / "model.safetensors").write_bytes(b"tampered-weights")
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["model"]["path_or_repo"] = str(impostor)
            manifest["model"]["weights_sha256"] = __import__(
                "shopping_grpo.training.sft.run_manifest",
                fromlist=["hash_weight_files"],
            ).hash_weight_files(impostor)["weights_sha256"]
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_weights_hash_tamper_fails(self):
        """审计 P0-A.1：基座权重被改动必须被权重 hash 校验拦下。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            (base / "model.safetensors").write_bytes(b"tampered-base-weights")
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_revision_consistency(self):
        """审计 P0-A.1：revision 冲突失败；两侧皆缺也失败（不允许静默跳过）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            _, passed = verify_run_dir(
                run_dir, revision="another-revision", loaders=_fake_loaders()
            )
            self.assertFalse(passed)

            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["model"]["revision"] = None
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_tokenizer_identity_semantic_tamper_fails(self):
        """语义 vocab/config 被替换必须被解析后指纹拦下。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            (run_dir / "tokenizer_config.json").write_text(
                '{"model": "something-else"}', encoding="utf-8"
            )
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            checks = {c["name"]: c for c in manifest["result"]["adapter_reload"]["checks"]}
            self.assertIn("tokenizer_identity_matches_base", checks)
            self.assertFalse(checks["tokenizer_identity_matches_base"]["passed"])

    def test_tokenizer_identity_holds_on_intact_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertTrue(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            checks = {c["name"]: c for c in manifest["result"]["adapter_reload"]["checks"]}
            self.assertTrue(checks["tokenizer_identity_matches_base"]["passed"])

    def test_tokenizer_identity_ignores_redundant_processor_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            (base / "processor_config.json").write_text("{}", encoding="utf-8")
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertTrue(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            check = next(c for c in manifest["result"]["adapter_reload"]["checks"] if c["name"] == "tokenizer_identity_matches_base")
            self.assertTrue(check["passed"])

    def test_tokenizer_identity_ignores_cache_and_redundant_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            (run_dir / "tokenizer_extra.json").write_text("{}", encoding="utf-8")
            (run_dir / ".cache").mkdir()
            (run_dir / ".cache" / "tokenizer_extra.json").write_text("{}", encoding="utf-8")
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertTrue(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            check = next(c for c in manifest["result"]["adapter_reload"]["checks"] if c["name"] == "tokenizer_identity_matches_base")
            self.assertTrue(check["passed"])

    def test_tokenizer_identity_covers_vocab_and_merges_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _make_base_model(Path(tmpdir))
            run_dir = _make_adapter_run(Path(tmpdir), base=base)
            (run_dir / "vocab.json").write_bytes(b"changed-vocab")
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders())
            self.assertFalse(passed)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            check = next(c for c in manifest["result"]["adapter_reload"]["checks"] if c["name"] == "tokenizer_identity_matches_base")
            self.assertFalse(check["passed"])

    def test_forward_failure_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            _, passed = verify_run_dir(run_dir, loaders=_fake_loaders(forward_ok=False))
            self.assertFalse(passed)

    def test_skip_forward_is_never_a_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _make_adapter_run(Path(tmpdir))
            _, passed = verify_run_dir(
                run_dir, loaders=_fake_loaders(), skip_forward=True
            )
            self.assertFalse(passed)


def _hash_dir(directory: Path, exclude=("merge_manifest.json",)) -> dict:
    from shopping_grpo.training.sft.run_manifest import sha256_file

    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in exclude
    }


def _make_merged_dir(tmpdir: Path, *, dtype="bfloat16", corrupt=False, tamper_base=False, tamper_tokenizer=False):
    from shopping_grpo.training.sft.run_manifest import sha256_file

    merged = tmpdir / "merged"
    merged.mkdir()
    (merged / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "torch_dtype": dtype}), encoding="utf-8"
    )
    (merged / "tokenizer_config.json").write_bytes(b'merged-tokenizer-config')
    (merged / "vocab.json").write_bytes(b"base-vocab")
    (merged / "merges.txt").write_bytes(b"base-merges")
    shard = merged / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"merged-weights")
    if corrupt:
        shard.write_bytes(b"tampered-weights")

    base = tmpdir / "models" / "Qwen3.5-2B"
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"base-weights")
    (base / "tokenizer_config.json").write_bytes(b'merged-tokenizer-config')
    (base / "vocab.json").write_bytes(b"base-vocab")
    (base / "merges.txt").write_bytes(b"base-merges")
    adapter = tmpdir / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
    if tamper_tokenizer:
        # merge 之后 tokenizer 被替换（模拟 revision 漂移的最后防线）
        (merged / "tokenizer_config.json").write_bytes(b'drifted-tokenizer-config')

    manifest = build_merge_manifest(
        str(base),
        adapter,
        merged,
        "qwen3_5",
        merge={
            "dtype": dtype,
            "max_shard_size": "5GB",
            "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "source_run_id": "0" * 64,
        },
        inputs_sha256={"base": _hash_dir(base), "adapter": _hash_dir(adapter)},
        output_files_sha256={
            "config.json": sha256_file(merged / "config.json"),
            "tokenizer_config.json": sha256_file(merged / "tokenizer_config.json"),
            "vocab.json": sha256_file(merged / "vocab.json"),
            "merges.txt": sha256_file(merged / "merges.txt"),
            "model-00001-of-00001.safetensors": sha256_bytes(b"merged-weights"),
        },
    )
    (merged / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged


class VerifyMergedCheckpointTest(unittest.TestCase):
    def test_happy_path_passes_and_writes_verification_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir))
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            # The manifest output hash audit may flag a post-merge file, but
            # tokenizer identity itself must not treat redundant artifacts as
            # semantic drift.
            self.assertTrue(passed)
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["verification"]["passed"])
            self.assertNotIn("adapter_config.json", manifest["output_files_sha256"])

    def test_tampered_shard_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir), corrupt=True)
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            changed = [
                check
                for check in manifest["verification"]["checks"]
                if check["name"] == "merge_manifest_output_hashes" and not check["passed"]
            ]
            self.assertTrue(changed)

    def test_missing_source_run_id_fails_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir))
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            manifest["merge"]["source_run_id"] = None
            (merged / "merge_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_base_input_tamper_fails_input_hash_check(self):
        """审计 P0-A.2：merge 之后基座文件被改动必须被输入 hash 校验拦下。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir))
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            base = Path(manifest["source"]["base_model"])
            (base / "model.safetensors").write_bytes(b"tampered-after-merge")
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            names = {c["name"] for c in manifest["verification"]["checks"]}
            self.assertIn("input_hashes_verified", names)

    def test_missing_base_revision_fails_contract(self):
        """审计 P0-A.2：merge manifest 必须记录基座 revision。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir))
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            manifest["merge"]["base_revision"] = None
            (merged / "merge_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)

    def test_merged_tokenizer_drift_fails_identity_check(self):
        """merge 之后 tokenizer 语义被替换必须被解析后指纹拦下。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir), tamper_tokenizer=True)
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            checks = {c["name"]: c for c in manifest["verification"]["checks"]}
            self.assertIn("tokenizer_identity_matches_base", checks)
            self.assertFalse(checks["tokenizer_identity_matches_base"]["passed"])

    def test_merged_tokenizer_identity_ignores_redundant_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = _make_merged_dir(Path(tmpdir))
            (merged / "processor_config.json").write_text("{}", encoding="utf-8")
            _, passed = verify_merged_dir(merged, loaders=_fake_loaders())
            self.assertFalse(passed)  # output hash audit catches post-merge additions
            manifest = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
            check = next(c for c in manifest["verification"]["checks"] if c["name"] == "tokenizer_identity_matches_base")
            self.assertTrue(check["passed"])


class SftResumeDriftTest(unittest.TestCase):
    """审计 P0-A.3：resume 前的合同漂移比较。"""

    def test_drift_detects_recipe_change_and_ignores_runtime_counts(self):
        from scripts.train_lora_sft import _manifest_drift

        base = {
            "model": {"path_or_repo": "/m", "revision": "r", "weights_sha256": "h"},
            "data": {
                "train_path": "t.jsonl",
                "train_sha256": "x",
                "validation_path": None,
                "validation_sha256": None,
                "sft_ready_manifest_sha256": None,
                "preflight_report_sha256": None,
                "train_examples": 8,
                "validation_examples": 100,
            },
            "recipe": {"lora_r": 16, "seed": 42},
            "runtime": {"code_hashes": {"a": "1"}, "dependency_versions": {}, "gpu": None},
        }
        import copy

        same = copy.deepcopy(base)
        same["data"]["train_examples"] = 999  # finalize 回填值不构成漂移
        same["execution"] = {"exit_code": 1}
        self.assertEqual(_manifest_drift(base, same), [])

        drifted = copy.deepcopy(base)
        drifted["recipe"]["seed"] = 43
        drift = _manifest_drift(base, drifted)
        self.assertEqual(drift, ["recipe.seed: 42 -> 43"])

        model_drifted = copy.deepcopy(base)
        model_drifted["model"]["weights_sha256"] = "different"
        self.assertTrue(_manifest_drift(base, model_drifted))


class MergeManifestBackwardCompatTest(unittest.TestCase):
    def test_legacy_four_argument_call_keeps_original_fields(self):
        manifest = build_merge_manifest(
            base_model="Qwen/Qwen3.5-2B",
            adapter_path="checkpoints/sft",
            output_path="checkpoints/sft_merged",
            model_type="qwen3_5",
        )
        self.assertEqual(manifest["operation"], "peft_merge_and_unload")
        self.assertEqual(manifest["source"]["adapter"], "checkpoints/sft")
        self.assertEqual(manifest["output"], "checkpoints/sft_merged")
        self.assertNotIn("merge", manifest)

    def test_extended_call_records_audit_fields(self):
        manifest = build_merge_manifest(
            "/models/Qwen3.5-2B",
            "outputs/adapter",
            "outputs/merged",
            "qwen3_5",
            merge={"dtype": "bfloat16", "max_shard_size": "5GB", "source_run_id": "r-1"},
            inputs_sha256={"adapter": {"adapter_config.json": "x"}},
            output_files_sha256={"config.json": "y"},
        )
        self.assertEqual(manifest["merge"]["dtype"], "bfloat16")
        self.assertEqual(manifest["merge"]["source_run_id"], "r-1")
        self.assertEqual(manifest["source"]["inputs_sha256"]["adapter"]["adapter_config.json"], "x")
        self.assertEqual(manifest["output_files_sha256"]["config.json"], "y")

    def test_merge_rejects_non_empty_output_directory(self):
        import scripts.merge_lora_adapter as merge_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "not-empty"
            output.mkdir()
            (output / "something.txt").write_text("keep out", encoding="utf-8")
            argv = [
                "merge_lora_adapter.py",
                "--base-model",
                "/models/Qwen3.5-2B",
                "--adapter",
                "outputs/adapter",
                "--output",
                str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(SystemExit):
                    merge_module.main()


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


class ChooseModelClassRegressionTest(unittest.TestCase):
    def test_qwen35_uses_multimodal_model_class(self):
        self.assertEqual(choose_model_class(_Config("qwen3_5"), "causal", "multimodal"), "multimodal")
        self.assertEqual(choose_model_class(_Config("qwen3"), "causal", "multimodal"), "causal")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
