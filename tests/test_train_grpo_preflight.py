"""train_grpo.py 启动前校验、smoke/resume、run manifest 与 patch helper 的纯逻辑测试。

不依赖 torch / transformers / verl / pyarrow：
- 模型、parquet、metadata、merge manifest、runtime contract、checkpoint 全部用
  临时目录里的 fake 文件；
- ``subprocess.call`` / ``dump_resolved_config`` 被替换，绝不会触碰真实 veRL 或
  ShopSimulator；
- dynamic sampling 诊断直接调用 ``shopping_grpo.training.grpo.dynamic_sampling``
  （该模块只依赖标准库，import 链干净）；
- runtime gate（check_grpo_runtime）用注入的 fake patched SHA 验证 marker+SHA256
  双重校验；
- patch helper 用 fake 常量 + fake patch 程序验证 原始 hash → apply → patched
  hash → restore 的完整校验链。
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import apply_verl_dynamic_sampling_patch as patcher
from scripts import check_grpo_runtime as runtime_gate
from scripts import train_grpo
from scripts.build_grpo_data import (
    FROZEN_TOKENIZER_REVISION,
    METADATA_SCHEMA,
    SFT_SOURCE_IDENTITIES,
    canonical_task_ids_sha256,
)
from scripts.train_grpo import (
    build_command,
    build_grpo_run_manifest,
    hydra_overrides,
    parse_args,
    preflight,
)
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    build_rollout_diagnostics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SHA256 = "b" * 64
REWARD_VERSION = "shopsimulator-reward-v3"
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
TOOL_SCHEMA_SHA256 = "9eccfc80db97f7fb9b34eea2f2f65f0b8bcb9bae76d8c2ee031db99c435ee1ad"
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fake_merged_model(
    directory: Path,
    *,
    verification_passed=True,
    dtype="bfloat16",
    base_model="Qwen/Qwen3.5-2B",
    base_revision=train_grpo.EXPECTED_MODEL_REVISION,
    include_revision=True,
    output_override=None,
) -> Path:
    model = directory / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake-weights")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    merge_block = {"dtype": dtype, "max_shard_size": "5GB", "source_run_id": "run-1"}
    if include_revision:
        merge_block["base_revision"] = base_revision
    merge_manifest = {
        "operation": "peft_merge_and_unload",
        "source": {
            "base_model": base_model,
            "adapter": "outputs/models/process-sft",
            "model_type": "qwen3",
        },
        "output": str(model) if output_override is None else output_override,
        "merge": merge_block,
        "verification": {"passed": verification_passed, "checks": []},
    }
    (model / "merge_manifest.json").write_text(
        json.dumps(merge_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model


def _write_fake_data(
    directory: Path, *, example=False, wrong_hash=False, with_metadata=True, versions=None
) -> tuple[Path, Path, Path]:
    data = directory / "data"
    data.mkdir(parents=True, exist_ok=True)
    train = data / "train.parquet"
    validation = data / "validation.parquet"
    train.write_bytes(b"fake-train-parquet")
    validation.write_bytes(b"fake-val-parquet")
    if not with_metadata:
        return train, validation, data / "metadata.json"
    version_fields = versions or {}
    # Keep the normal preflight fixture on the same strict provenance contract
    # as real builder output.  Otherwise every unrelated preflight test stops
    # at the SFT gate before exercising the behavior it intends to cover.
    sft_dir = data / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    sft_sources = {}
    for index, identity in enumerate(SFT_SOURCE_IDENTITIES, start=1):
        task_ids = [1000 + index * 10, 1001 + index * 10]
        source_path = sft_dir / f"{identity}.jsonl"
        source_path.write_text(
            "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
            encoding="utf-8",
        )
        sft_sources[identity] = {
            "identity": identity,
            "path": f"sft/{source_path.name}",
            "sha256": _sha(source_path.read_bytes()),
            "task_count": len(task_ids),
            "task_ids_sha256": canonical_task_ids_sha256(task_ids),
        }
    train_hash = "f" * 64 if wrong_hash else _sha(train.read_bytes())
    validation_hash = _sha(validation.read_bytes())
    metadata = {
        "schema_version": train_grpo.DATA_METADATA_SCHEMA,
        "example": example,
        "source": {
            "runtime_contract_sha256": CONTRACT_SHA256,
            "reward_version": version_fields.get("reward_version", REWARD_VERSION),
            "environment_version": version_fields.get(
                "environment_version", ENVIRONMENT_VERSION
            ),
            "tool_schema_sha256": version_fields.get("tool_schema_sha256", TOOL_SCHEMA_SHA256),
            "system_prompt_sha256": version_fields.get(
                "system_prompt_sha256", SYSTEM_PROMPT_SHA256
            ),
        },
        "build_parameters": {
            "tokenizer": "fake-tokenizer",
            "tokenizer_revision": version_fields.get(
                "tokenizer_revision", FROZEN_TOKENIZER_REVISION
            ),
        },
        "files": {
            "grpo_train": {
                "path": "train.parquet",
                "rows": 2,
                "unique_task_ids": 2,
                "file_sha256": train_hash,
            },
            "grpo_validation": {
                "path": "validation.parquet",
                "rows": 2,
                "unique_task_ids": 2,
                "file_sha256": validation_hash,
            },
        },
        "leakage": {
            "all_checks_passed": True,
            "sft_task_id_check": {
                "checked": True,
                "source_count": 4,
                "sources": sft_sources,
                "overlaps": {f"sft/{identity}.jsonl": [] for identity in SFT_SOURCE_IDENTITIES},
            },
        },
    }
    metadata_path = data / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return train, validation, metadata_path


def _write_fake_contract(
    directory: Path,
    *,
    contract_sha256=CONTRACT_SHA256,
    reward_version=REWARD_VERSION,
    environment_version=ENVIRONMENT_VERSION,
    tool_schema_hash=TOOL_SCHEMA_SHA256,
    system_prompt_hash=SYSTEM_PROMPT_SHA256,
    name="runtime_contract.json",
) -> Path:
    path = Path(directory) / name
    path.write_text(
        json.dumps(
            {
                "schema_version": "commerce-runtime-contract-v1",
                "contract_sha256": contract_sha256,
                "reward_version": reward_version,
                "environment_version": environment_version,
                "tool_schema_hash": tool_schema_hash,
                "system_prompt_hash": system_prompt_hash,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_checkpoint(output: Path, *, step=3, complete=True, tracker="3") -> Path:
    """veRL 0.8 checkpoint 布局：global_step_N/actor/{model,optim,extra_state}_*.pt
    加 output 根目录的 latest_checkpointed_iteration.txt tracker。"""
    checkpoint = output / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True, exist_ok=True)
    if complete:
        (actor / "model_world_size_1_rank_0.pt").write_bytes(b"actor-weights")
        (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"optimizer-state")
        (actor / "extra_state_world_size_1_rank_0.pt").write_bytes(b"scheduler-and-rng-state")
    if tracker is not None:
        (output / "latest_checkpointed_iteration.txt").write_text(tracker, encoding="utf-8")
    return checkpoint


def _make_args(tmp: Path, **overrides) -> list[str]:
    """默认一组可以通过 preflight 的 CLI 参数；测试按需覆盖。"""
    if "model" not in overrides:
        overrides["model"] = _write_fake_merged_model(tmp)
    train, validation, metadata = _write_fake_data(
        tmp, with_metadata=overrides.pop("with_metadata", True),
        example=overrides.pop("example", False), wrong_hash=overrides.pop("wrong_hash", False),
        versions=overrides.pop("versions", None),
    )
    argv = [
        "train_grpo.py",
        "--model", str(overrides.pop("model")),
        "--train-data", str(train),
        "--val-data", str(validation),
        "--data-metadata", str(overrides.pop("data_metadata", metadata)),
        "--output", str(overrides.pop("output", tmp / "out")),
        "--config", str(overrides.pop("config", ROOT / "configs/grpo.yaml")),
        "--env-url", "http://127.0.0.1:5700",
    ]
    runtime_contract = overrides.pop("runtime_contract", None)
    if runtime_contract is None:
        runtime_contract = _write_fake_contract(tmp)
    argv.extend(["--runtime-contract", str(runtime_contract)])
    extra_overrides = overrides.pop("hydra_overrides", None)
    for key, value in overrides.items():
        flag = key.replace("_", "-")
        if value is True:
            argv.append(f"--{flag}")
        elif value is not False and value is not None:
            argv.extend([f"--{flag}", str(value)])
    if extra_overrides:
        # argparse REMAINDER 会吞掉首个非选项 token 之后的所有内容，必须放最后。
        argv.append("--")
        argv.extend(extra_overrides)
    return argv


def _parse(argv: list[str]):
    with patch.object(sys, "argv", argv):
        return parse_args()


def _fake_dump(command, environment, destination):
    """注入的 resolved-config dump：写一个可识别的文件并返回记录块。"""
    destination = Path(destination)
    destination.write_bytes(b"# fake resolved config\ntrainer: {}\n")
    return {
        "status": "dumped",
        "path": str(destination),
        "sha256": _sha(destination.read_bytes()),
        "command": list(command),
    }


class GrpoPreflightTest(unittest.TestCase):
    def test_smoke_overrides_resolved_command_and_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), smoke=True, smoke_steps=1,
                              hydra_overrides=["--", "trainer.total_training_steps=999"])
            args = _parse(argv)
            command, environment = build_command(args)
            overrides = hydra_overrides(args)
            self.assertIn("verl.trainer.main_ppo", command)
            self.assertIn("trainer.logger=[console]", overrides)
            # 用户 override 在前，smoke 预算最后追加并因此生效。
            self.assertLess(
                overrides.index("trainer.total_training_steps=999"),
                overrides.index("trainer.total_training_steps=1"),
            )
            self.assertIn("trainer.save_freq=1", overrides)

            preflight_result = preflight(args)
            manifest = build_grpo_run_manifest(args, command, environment, preflight_result)
            self.assertEqual(manifest["schema_version"], "shopping-grpo-run-manifest-v1")
            self.assertEqual(manifest["stage"], "smoke")
            self.assertTrue(manifest["recipe"]["smoke"])
            self.assertEqual(manifest["contract"]["environment_version"], ENVIRONMENT_VERSION)
            self.assertTrue(manifest["model"]["verification_passed"])
            self.assertEqual(manifest["data"]["train_rows"], 2)
            # 版本合同快照必须进入 manifest（audit item 7）。
            self.assertEqual(manifest["versions"]["reward_version"], REWARD_VERSION)
            self.assertEqual(manifest["versions"]["tool_schema_sha256"], TOOL_SCHEMA_SHA256)
            self.assertEqual(manifest["versions"]["system_prompt_sha256"], SYSTEM_PROMPT_SHA256)
            self.assertTrue(manifest["versions"]["runtime_contract_path"])

    def test_v5_entropy_off_override_is_recorded_before_smoke_budget(self):
        """V5 changes only actor entropy at runtime; the canonical config stays diagnostic-on."""
        self.assertIn(
            "calculate_entropy: true",
            (ROOT / "configs/grpo.yaml").read_text(encoding="utf-8"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                smoke=True,
                smoke_steps=1,
                hydra_overrides=[
                    "--",
                    "actor_rollout_ref.actor.calculate_entropy=false",
                ],
            )
            args = _parse(argv)
            command, environment = build_command(args)
            overrides = hydra_overrides(args)
            entropy_override = "actor_rollout_ref.actor.calculate_entropy=false"
            self.assertIn(entropy_override, overrides)
            self.assertLess(
                overrides.index(entropy_override),
                overrides.index("trainer.total_training_steps=1"),
            )

            manifest = build_grpo_run_manifest(
                args, command, environment, preflight(args)
            )
            self.assertIn(entropy_override, manifest["hydra_overrides"])
            self.assertTrue(manifest["recipe"]["resolved_config_dump"])

    def test_stop_manifest_records_synchronous_actor_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _parse(_make_args(Path(tmp), stop_after_step=100))
            command, environment = build_command(args)
            manifest = build_grpo_run_manifest(args, command, environment, preflight(args))
            evidence = manifest["recipe"]["actor_checkpoint_async_save"]
            self.assertFalse(evidence["value"])
            self.assertEqual(environment["SHOPPING_GRPO_ACTOR_CHECKPOINT_ASYNC_SAVE"], "false")

    def test_metadata_schema_constant_is_shared_with_builder(self):
        """回归（audit item 8）：launcher 与构建器共用同一 metadata schema 常量。"""
        self.assertIs(train_grpo.DATA_METADATA_SCHEMA, METADATA_SCHEMA)
        self.assertEqual(METADATA_SCHEMA, "shopping-grpo-data-metadata-v2")

    def test_launcher_does_not_trust_all_checks_passed_summary(self):
        """A forged summary bit without the four source records is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            train, validation, metadata_path = _write_fake_data(Path(tmp))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["leakage"] = {"all_checks_passed": True}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "four-source SFT"):
                train_grpo.validate_data_metadata(train, validation, metadata_path)

    def test_resume_from_uses_registered_verl_resume_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), resume_from=Path(tmp) / "checkpoint")
            args = _parse(argv)
            self.assertIn("trainer.resume_mode=resume_path", hydra_overrides(args))
            self.assertNotIn("trainer.resume_mode=resumable_path", hydra_overrides(args))

    def test_ref_model_defaults_to_actor_and_is_recorded(self):
        """回归（audit item 4）：ref 缺省解析到 actor 同一路径并写进 manifest。"""
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            result = preflight(args)
            model_path = str(Path(args.model).expanduser().resolve())
            self.assertEqual(result["model_paths"]["actor_model_path"], model_path)
            self.assertEqual(result["model_paths"]["ref_model_path"], model_path)
            self.assertTrue(result["model_paths"]["ref_matches_actor"])
            command, environment = build_command(args)
            manifest = build_grpo_run_manifest(args, command, environment, result)
            self.assertEqual(manifest["model"]["ref_model_path"], model_path)
            self.assertTrue(manifest["model"]["ref_matches_actor"])

    def test_ref_model_must_resolve_to_actor_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "other-model"
            other.mkdir()
            argv = _make_args(Path(tmp), ref_model=other)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("same M2 merged checkpoint", str(ctx.exception))
            self.assertIn("ref=", str(ctx.exception))

    def test_base_model_mismatch_is_rejected(self):
        """Base 变体（如 Qwen3.5-2B-Base）不能冒充 M2 的基座。"""
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                model=_write_fake_merged_model(Path(tmp), base_model="Qwen/Qwen3.5-2B-Base"),
            )
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("base model", str(ctx.exception))
            self.assertIn("Qwen/Qwen3.5-2B-Base", str(ctx.exception))

    def test_local_base_model_path_with_same_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                model=_write_fake_merged_model(
                    Path(tmp), base_model="/root/autodl-tmp/models/Qwen3.5-2B"
                ),
            )
            args = _parse(argv)
            result = preflight(args)
            self.assertEqual(result["model"]["base_model"], "/root/autodl-tmp/models/Qwen3.5-2B")

    def test_base_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                model=_write_fake_merged_model(Path(tmp), base_revision="deadbeef" * 8),
            )
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("base revision mismatch", str(ctx.exception))

    def test_missing_base_revision_requires_explicit_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                model=_write_fake_merged_model(Path(tmp), include_revision=False),
            )
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("merge.base_revision", str(ctx.exception))
            self.assertIn("--allow-unverified-revision", str(ctx.exception))

            acknowledged = _parse(
                _make_args(
                    Path(tmp),
                    model=_write_fake_merged_model(Path(tmp), include_revision=False),
                    allow_unverified_revision=True,
                )
            )
            result = preflight(acknowledged)
            self.assertFalse(result["model"]["revision_verified"])
            self.assertIn("unverified", result["model"]["revision_note"])

    def test_merge_manifest_output_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(
                Path(tmp),
                model=_write_fake_merged_model(Path(tmp), output_override="/elsewhere/model"),
            )
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("records output", str(ctx.exception))

    def test_relative_merge_manifest_output_resolves_from_project_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            model = _write_fake_merged_model(
                project / "outputs" / "models",
                output_override="outputs/models/model",
            )
            args = _parse(_make_args(Path(tmp), model=model))
            result = preflight(args)
            self.assertEqual(result["model"]["verification_passed"], True)

    def test_relative_merge_manifest_output_different_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _write_fake_merged_model(Path(tmp), output_override="other-model")
            args = _parse(_make_args(Path(tmp), model=model))
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("records output", str(ctx.exception))

    def test_relative_merge_manifest_output_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _write_fake_merged_model(Path(tmp), output_override="../model")
            args = _parse(_make_args(Path(tmp), model=model))
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("records output", str(ctx.exception))

    def test_relative_merge_manifest_output_ambiguous_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _write_fake_merged_model(
                Path(tmp), output_override="outputs/models/model"
            )
            args = _parse(_make_args(Path(tmp), model=model))
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("records output", str(ctx.exception))

    def test_dry_run_prints_resolved_env_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), dry_run=True)
            output = io.StringIO()
            with patch.object(sys, "argv", argv):
                with patch.dict("os.environ", {"SWANLAB_API_KEY": "super-secret-value"}):
                    with contextlib.redirect_stdout(output):
                        train_grpo.main()
            audit = json.loads(output.getvalue())
            self.assertIn("verl.trainer.main_ppo", audit["command"])
            self.assertIn("GRPO_MODEL_PATH", audit["environment"])
            serialized = json.dumps(audit)
            self.assertNotIn("super-secret-value", serialized)
            self.assertNotIn("SWANLAB_API_KEY", serialized)
            # dry-run 不创建输出目录。
            self.assertFalse(Path(audit["output"]).exists())

    def test_dry_run_fails_without_merge_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), with_metadata=True)
            model = Path(argv[argv.index("--model") + 1])
            (model / "merge_manifest.json").unlink()
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("merge_manifest.json", str(ctx.exception))

    def test_dry_run_fails_without_verified_merge_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), model=_write_fake_merged_model(Path(tmp), verification_passed=False))
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("no passed verification block", str(ctx.exception))

    def test_dry_run_fails_without_train_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            Path(argv[argv.index("--train-data") + 1]).unlink()
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                build_command(args)
            self.assertIn("train parquet does not exist", str(ctx.exception))

    def test_dry_run_fails_without_data_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), with_metadata=False)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("GRPO data metadata is missing", str(ctx.exception))

    def test_dry_run_fails_on_parquet_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), wrong_hash=True)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("hash mismatch", str(ctx.exception))

    def test_example_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), example=True)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("example metadata", str(ctx.exception))

    def test_runtime_contract_cross_check_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = _write_fake_contract(Path(tmp), contract_sha256="9" * 64)
            argv = _make_args(Path(tmp), runtime_contract=contract)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("runtime contract does not match", str(ctx.exception))

    def test_runtime_contract_file_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing" / "runtime_contract.json"
            argv = _make_args(Path(tmp), runtime_contract=missing)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("runtime contract not found", str(ctx.exception))

    def test_metadata_version_mismatch_with_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = _write_fake_contract(Path(tmp), reward_version="shopsimulator-reward-v2")
            argv = _make_args(Path(tmp), runtime_contract=contract)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("disagrees with GRPO data metadata", str(ctx.exception))
            self.assertIn("reward_version", str(ctx.exception))

    def test_metadata_missing_version_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), versions={"reward_version": None})
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("missing version fields", str(ctx.exception))

    def test_metadata_tokenizer_revision_must_be_frozen(self):
        for revision in (None, "deadbeef"):
            with self.subTest(revision=revision):
                with tempfile.TemporaryDirectory() as tmp:
                    argv = _make_args(
                        Path(tmp), versions={"tokenizer_revision": revision}
                    )
                    args = _parse(argv)
                    with self.assertRaisesRegex(SystemExit, "tokenizer_revision"):
                        preflight(args)

    def test_metadata_system_prompt_drift_is_rejected(self):
        """parquet metadata 的 system prompt 与代码内 SYSTEM_PROMPT 不一致必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), versions={"system_prompt_sha256": "0" * 64})
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("SYSTEM_PROMPT", str(ctx.exception))
            self.assertIn("rebuild the data", str(ctx.exception))


class GrpoResumeTest(unittest.TestCase):
    def _write_prior_run_manifest(self, tmp: Path, args) -> dict:
        command, environment = build_command(args)
        manifest = build_grpo_run_manifest(args, command, environment, preflight(args))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # veRL 0.8 resume 合同：完整 checkpoint 产物 + 可读 tracker。
        _write_fake_checkpoint(output)
        return manifest

    def test_resume_passes_when_contract_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            resume_argv = fresh_argv + ["--resume"]
            args = _parse(resume_argv)
            result = preflight(args)
            self.assertIsNotNone(result["resume"])
            self.assertEqual(result["resume"]["prior_stage"], "training")
            self.assertEqual(result["resume"]["checkpoint"]["global_step"], 3)
            self.assertEqual(result["resume"]["tracker_step"], 3)
            self.assertEqual(len(result["resume"]["checkpoint"]["matched_artifacts"]), 3)

    def test_resume_from_records_resume_true_and_explicit_checkpoint_path(self):
        """Regression: --resume-from must not be serialized as a fresh run."""
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            output = Path(_parse(fresh_argv).output)
            checkpoint = output / "global_step_3"
            args = _parse(fresh_argv + ["--resume-from", str(checkpoint)])

            result = preflight(args)
            command, environment = build_command(args)
            manifest = build_grpo_run_manifest(args, command, environment, result)

            self.assertEqual(manifest["stage"], "resume")
            self.assertTrue(manifest["recipe"]["resume"])
            self.assertEqual(manifest["recipe"]["resume_from"], str(checkpoint))
            self.assertEqual(
                manifest["resume"]["checkpoint"]["path"], str(checkpoint)
            )

    def test_resume_rejects_contract_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            manifest = self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            drifted = dict(manifest["contract"])
            drifted["train_data_sha256"] = "d" * 64
            manifest["contract"] = drifted
            output = Path(_parse(fresh_argv).output)
            (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            args = _parse(fresh_argv + ["--resume"])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("refusing to resume: run contract drifted", str(ctx.exception))
            self.assertIn("create a new run", str(ctx.exception))

    def test_resume_without_prior_run_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv + ["--resume"])
            Path(args.output).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("requires a prior run_manifest.json", str(ctx.exception))

    def test_resume_rejects_incomplete_checkpoint_artifacts(self):
        """回归（audit item 3）：缺 optimizer/scheduler/RNG 产物必须拒绝续训。"""
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            output = Path(_parse(fresh_argv).output)
            # complete=False：只有 actor 目录，缺 optimizer/scheduler/RNG 产物。
            _write_fake_checkpoint(output, step=4, complete=False, tracker="4")

            args = _parse(fresh_argv + ["--resume"])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            message = str(ctx.exception)
            self.assertIn("missing required veRL 0.8 artifacts", message)
            self.assertIn("optim_world_size", message)
            self.assertIn("start a new run", message)

    def test_resume_rejects_unreadable_global_step(self):
        """回归（audit item 3）：global step 文件不可读时拒绝续训。"""
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            output = Path(_parse(fresh_argv).output)
            _write_fake_checkpoint(output, step=3, complete=True, tracker="not-a-number")

            args = _parse(fresh_argv + ["--resume"])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("readable latest_checkpointed_iteration.txt", str(ctx.exception))

    def test_resume_rejects_tracker_checkpoint_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            self._write_prior_run_manifest(Path(tmp), _parse(fresh_argv))
            output = Path(_parse(fresh_argv).output)
            _write_fake_checkpoint(output, step=3, complete=True, tracker="5")

            args = _parse(fresh_argv + ["--resume"])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("tracker file records step 5", str(ctx.exception))

    def test_resume_without_checkpoint_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            args = _parse(fresh_argv)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            self._write_prior_run_manifest(Path(tmp), args)
            # 删除 _write_prior_run_manifest 创建的 checkpoint，只留 manifest。
            for child in sorted(output.glob("global_step_*")):
                shutil.rmtree(child)
            (output / "latest_checkpointed_iteration.txt").unlink()
            args = _parse(fresh_argv + ["--resume"])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("global_step_N checkpoint directory", str(ctx.exception))

    def test_resume_from_requires_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            args = _parse(fresh_argv)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            self._write_prior_run_manifest(Path(tmp), args)
            # complete=False：只有 actor 权重，缺 optimizer/scheduler/RNG 产物。
            checkpoint = _write_fake_checkpoint(output, step=7, complete=False, tracker=None)

            args = _parse(fresh_argv + ["--resume-from", str(checkpoint)])
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("missing required veRL 0.8 artifacts", str(ctx.exception))

    def test_resume_from_accepts_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            args = _parse(fresh_argv)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            self._write_prior_run_manifest(Path(tmp), args)
            checkpoint = _write_fake_checkpoint(output, step=7, complete=True, tracker=None)

            args = _parse(fresh_argv + ["--resume-from", str(checkpoint)])
            result = preflight(args)
            self.assertEqual(result["resume"]["checkpoint"]["global_step"], 7)
            self.assertEqual(
                result["resume"]["checkpoint"]["global_step_source"],
                "checkpoint directory name",
            )

    def test_smoke_and_resume_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), smoke=True, resume=True)
            args = _parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                preflight(args)
            self.assertIn("not allowed", str(ctx.exception))

    def test_resume_run_manifest_is_appended_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_argv = _make_args(Path(tmp))
            fresh_args = _parse(fresh_argv)
            self._write_prior_run_manifest(Path(tmp), fresh_args)
            output = Path(fresh_args.output)
            prior_bytes = (output / "run_manifest.json").read_bytes()

            calls = []

            def fake_call(cmd, **kwargs):
                calls.append(cmd)
                return 0

            with patch.object(sys, "argv", fresh_argv + ["--resume"]):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", _fake_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 0)
            # 首次启动的 run manifest 不被覆盖；续训记录追加成独立文件。
            self.assertEqual((output / "run_manifest.json").read_bytes(), prior_bytes)
            self.assertTrue((output / "run_manifest.resume.1.json").is_file())
            resumed = json.loads((output / "run_manifest.resume.1.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed["stage"], "resume")
            self.assertEqual(resumed["resume"]["checkpoint"]["global_step"], 3)

    def test_full_launch_writes_run_manifest_and_calls_preflight_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            calls = []

            def fake_call(cmd, **kwargs):
                calls.append(cmd)
                return 0

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", _fake_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 0)
            self.assertEqual(len(calls), 2)  # check_grpo_runtime + verl.trainer.main_ppo
            self.assertIn("check_grpo_runtime.py", calls[0][1])
            self.assertIn("verl.trainer.main_ppo", " ".join(calls[1]))
            manifest_path = Path(args.output) / "run_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "training")
            self.assertIn("command", manifest)
            self.assertNotIn("SWANLAB_API_KEY", json.dumps(manifest))


class GrpoLaunchOrderTest(unittest.TestCase):
    """回归（audit item 3/6）：runtime preflight 与 resolved config dump 全部通过
    之前不得写 run manifest；训练结束后必须写 checkpoint manifest。"""

    def test_runtime_preflight_failure_leaves_no_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            output = Path(args.output)
            calls = []

            def fake_call(cmd, **kwargs):
                calls.append(cmd)
                return 1 if "check_grpo_runtime.py" in " ".join(cmd) else 0

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", _fake_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertEqual(len(calls), 1)  # veRL 从未被调用
            self.assertFalse((output / "run_manifest.json").is_file())

    def test_resolved_config_dump_is_written_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            output = Path(args.output)
            dumped_with = []

            def recording_dump(command, environment, destination):
                dumped_with.append(destination)
                return _fake_dump(command, environment, destination)

            def fake_call(cmd, **kwargs):
                return 0

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", recording_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 0)
            self.assertEqual(
                dumped_with[0], output / train_grpo.RESOLVED_CONFIG_FILE
            )
            self.assertTrue((output / train_grpo.RESOLVED_CONFIG_FILE).is_file())
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["resolved_config"]["status"], "dumped")
            self.assertEqual(
                manifest["resolved_config"]["path"], str(output / train_grpo.RESOLVED_CONFIG_FILE)
            )
            self.assertTrue(manifest["resolved_config"]["sha256"])

    def test_dump_failure_leaves_no_run_manifest_and_skips_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            output = Path(args.output)
            calls = []

            def fake_call(cmd, **kwargs):
                calls.append(cmd)
                return 0

            def failing_dump(command, environment, destination):
                raise SystemExit("failed to dump the resolved GRPO config")

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", failing_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertIn("failed to dump", str(ctx.exception))
            self.assertEqual(len(calls), 1)  # 只有 runtime preflight；veRL 未启动
            self.assertFalse((output / "run_manifest.json").is_file())

    def test_no_dump_flag_is_recorded_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp), no_dump_resolved_config=True)
            args = _parse(argv)
            output = Path(args.output)

            def unexpected_dump(command, environment, destination):  # pragma: no cover
                raise AssertionError("dump must not run with --no-dump-resolved-config")

            def fake_call(cmd, **kwargs):
                return 0

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", unexpected_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 0)
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["resolved_config"]["status"], "skipped_explicitly")
            self.assertFalse(manifest["recipe"]["resolved_config_dump"])

    def test_checkpoint_manifest_written_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            output = Path(args.output)
            calls = []

            def fake_call(cmd, **kwargs):
                calls.append(cmd)
                if "verl.trainer.main_ppo" in " ".join(cmd):
                    # 模拟真实 veRL 运行在 output 下留下的 checkpoint 产物。
                    _write_fake_checkpoint(output, step=1, complete=True, tracker="1")
                    return 0
                return 0

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", _fake_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 0)
            manifest_path = output / train_grpo.CHECKPOINT_MANIFEST_FILE
            self.assertTrue(manifest_path.is_file())
            checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_manifest["schema_version"], "shopping-grpo-checkpoint-manifest-v1")
            self.assertEqual(checkpoint_manifest["exit_code"], 0)
            self.assertEqual(checkpoint_manifest["checkpoint_count"], 1)
            self.assertEqual(checkpoint_manifest["latest_global_step"], 1)
            entry = checkpoint_manifest["checkpoints"][0]
            self.assertEqual(entry["global_step"], 1)
            self.assertEqual(entry["file_count"], 3)
            for name in (
                "actor/model_world_size_1_rank_0.pt",
                "actor/optim_world_size_1_rank_0.pt",
                "actor/extra_state_world_size_1_rank_0.pt",
            ):
                self.assertIn(name, entry["files"])
                self.assertEqual(entry["files"][name], _sha((output / "global_step_1" / name).read_bytes()))

    def test_checkpoint_manifest_records_failed_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = _make_args(Path(tmp))
            args = _parse(argv)
            output = Path(args.output)

            def fake_call(cmd, **kwargs):
                # runtime preflight 通过，训练进程以非零码退出。
                return 0 if "check_grpo_runtime.py" in " ".join(cmd) else 3

            with patch.object(sys, "argv", argv):
                with patch.object(train_grpo.subprocess, "call", side_effect=fake_call):
                    with patch.object(train_grpo, "dump_resolved_config", _fake_dump):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(SystemExit) as ctx:
                                train_grpo.main()
            self.assertEqual(ctx.exception.code, 3)
            checkpoint_manifest = json.loads(
                (output / train_grpo.CHECKPOINT_MANIFEST_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint_manifest["exit_code"], 3)
            self.assertEqual(checkpoint_manifest["checkpoint_count"], 0)


class _AttrDict(dict):
    """dict + 属性访问：模拟 OmegaConf composed config 的双风格访问。"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _attr_dict(obj):
    if isinstance(obj, dict):
        return _AttrDict({key: _attr_dict(value) for key, value in obj.items()})
    return obj


class RuntimePatchShaGateTest(unittest.TestCase):
    """回归（audit item 5）：runtime gate 必须 marker + patched SHA256 双重校验。"""

    @staticmethod
    def _fake_dynamic_sampling_config():
        return _attr_dict(
            {
                "shopping_dynamic_sampling": {
                    "enable": True,
                    "metric": "seq_reward",
                    "max_num_gen_batches": 3,
                    "max_consecutive_skipped_updates": 10,
                    "reward_tolerance": 1.0e-8,
                },
                "algorithm": {"rollout_correction": {"bypass_mode": True}},
                "actor_rollout_ref": {"rollout": {"calculate_log_probs": True}},
            }
        )

    def _verl_tree(self, directory: Path, ray_trainer_text: str):
        verl_source = Path(directory) / "verl" / "__init__.py"
        ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
        ray_trainer.parent.mkdir(parents=True, exist_ok=True)
        verl_source.write_text("", encoding="utf-8")
        ray_trainer.write_text(ray_trainer_text, encoding="utf-8")
        return verl_source, ray_trainer

    def test_marker_without_matching_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            verl_source, ray_trainer = self._verl_tree(
                Path(tmp), f"# {runtime_gate.PATCH_MARKER}\ndef fit(self):\n    return 0\n"
            )
            with patch.object(runtime_gate, "EXPECTED_PATCHED_SHA256", "f" * 64):
                with self.assertRaises(SystemExit) as ctx:
                    runtime_gate.validate_dynamic_sampling(
                        self._fake_dynamic_sampling_config(), verl_source, {"verl": "0.8.0"}
                    )
            self.assertIn("SHA256 mismatch", str(ctx.exception))
            self.assertIn("apply_verl_dynamic_sampling_patch.py", str(ctx.exception))

    def test_markerless_file_is_rejected_before_sha_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            verl_source, _ = self._verl_tree(Path(tmp), "# unpatched\n")
            with self.assertRaises(SystemExit) as ctx:
                runtime_gate.validate_dynamic_sampling(
                    self._fake_dynamic_sampling_config(), verl_source, {"verl": "0.8.0"}
                )
            self.assertIn("patch marker is missing", str(ctx.exception))

    def test_matching_marker_and_sha_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            verl_source, ray_trainer = self._verl_tree(
                Path(tmp), f"# {runtime_gate.PATCH_MARKER}\ndef fit(self):\n    return 0\n"
            )
            actual = _sha(ray_trainer.read_bytes())
            with patch.object(runtime_gate, "EXPECTED_PATCHED_SHA256", actual):
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime_gate.validate_dynamic_sampling(
                        self._fake_dynamic_sampling_config(), verl_source, {"verl": "0.8.0"}
                    )

    def test_patcher_constants_match_runtime_gate(self):
        """runtime gate 与 patch 脚本必须使用同一份 patched SHA 口径。"""
        self.assertEqual(
            runtime_gate.EXPECTED_PATCHED_SHA256, patcher.EXPECTED_PATCHED_SHA256
        )


class ActorRefContractGateTest(unittest.TestCase):
    """check_grpo_runtime 层面的 ref/actor 同源校验（composed config）。"""

    def test_ref_without_model_override_passes(self):
        config = _attr_dict(
            {"actor_rollout_ref": {"model": {"path": "/models/m2"}, "ref": {"fsdp_config": {}}}}
        )
        with patch.dict("os.environ", {"GRPO_MODEL_PATH": "/models/m2"}):
            with contextlib.redirect_stdout(io.StringIO()):
                runtime_gate.validate_actor_ref_model_contract(config)

    def test_ref_model_override_is_rejected(self):
        config = _attr_dict(
            {
                "actor_rollout_ref": {
                    "model": {"path": "/models/m2"},
                    "ref": {"model": {"path": "/models/base"}},
                }
            }
        )
        with self.assertRaises(SystemExit) as ctx:
            runtime_gate.validate_actor_ref_model_contract(config)
        self.assertIn("must not override the actor model path", str(ctx.exception))

    def test_composed_path_must_match_env(self):
        config = _attr_dict(
            {"actor_rollout_ref": {"model": {"path": "/models/base"}, "ref": {}}}
        )
        with patch.dict("os.environ", {"GRPO_MODEL_PATH": "/models/m2"}):
            with self.assertRaises(SystemExit) as ctx:
                runtime_gate.validate_actor_ref_model_contract(config)
        self.assertIn("does not match GRPO_MODEL_PATH", str(ctx.exception))


class VerlPatchHelperTest(unittest.TestCase):
    """用注入的 fake 常量/patch 程序验证原始→apply→patched→restore 校验链。"""

    ORIGINAL_BODY = (
        "def fit(self):\n"
        "    batch = generate()\n"
        "    return update(batch)\n"
    )

    def setUp(self):
        self._patches = []

    def _install_fake_hashes(self, original: Path, patched: Path):
        original_hash = _sha(original.read_bytes())
        patched_hash = _sha(patched.read_bytes())
        fake_patch_file = patched.parent / "fake-verl.patch"
        fake_patch_file.write_bytes(b"fake patch payload\n")
        self._patches = [
            patch.object(patcher, "EXPECTED_ORIGINAL_SHA256", original_hash),
            patch.object(patcher, "EXPECTED_PATCHED_SHA256", patched_hash),
            # 本地工作树按 skip-worktree 不落盘真实 patch 文件；测试注入 fake。
            patch.object(patcher, "PATCH_FILE", fake_patch_file),
        ]
        for item in self._patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in self._patches])

    @staticmethod
    def _fake_patch_program(target: Path, marker: str) -> None:
        # fake 'patch' 可执行文件：把 marker 以合法 Python 注释追加到目标文件。
        text = target.read_text(encoding="utf-8")
        target.write_text(
            text
            + f"# {marker}\n"
            + f"# {patcher.STEP_BARRIER_MARKER}\n"
            + "def _barrier_probe(self):\n"
            + "    if should_controlled_stop():\n"
            + "        return\n",
            encoding="utf-8",
        )

    def test_apply_restore_chain_with_unknown_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "ray_trainer.py"
            original.write_text(self.ORIGINAL_BODY, encoding="utf-8")
            patched = Path(tmp) / "expected_patched.py"
            patched.write_text(
                self.ORIGINAL_BODY
                + f"# {patcher.PATCH_MARKER}\n"
                + f"# {patcher.STEP_BARRIER_MARKER}\n"
                + "def _barrier_probe(self):\n"
                + "    if should_controlled_stop():\n"
                + "        return\n",
                encoding="utf-8",
            )
            self._install_fake_hashes(original, patched)

            with patch.object(
                patcher.shutil, "which", return_value="fake-patch"
            ), patch.object(
                patcher.subprocess,
                "run",
                side_effect=lambda cmd, **kw: self._fake_patch_program(
                    Path(cmd[4]), patcher.PATCH_MARKER
                ),
            ):
                # apply/restore 自带 stdout 打印；测试里静音，保持 unittest 输出干净。
                with contextlib.redirect_stdout(io.StringIO()):
                    patcher.apply_patch(original)
                self.assertEqual(
                    _sha(original.read_bytes()), patcher.EXPECTED_PATCHED_SHA256
                )
                self.assertIn(patcher.PATCH_MARKER, original.read_text(encoding="utf-8"))

                # 已打补丁时重复 apply 必须幂等。
                with contextlib.redirect_stdout(io.StringIO()):
                    patcher.apply_patch(original)

                # backup 记录了原始 hash，restore 必须还原到字节一致。
                backup = Path(str(original) + patcher.BACKUP_SUFFIX)
                self.assertEqual(_sha(backup.read_bytes()), patcher.EXPECTED_ORIGINAL_SHA256)
                with contextlib.redirect_stdout(io.StringIO()):
                    patcher.restore_patch(original)
                self.assertEqual(_sha(original.read_bytes()), patcher.EXPECTED_ORIGINAL_SHA256)

            # 未知 hash 的文件必须被拒绝且不被修改、不留 backup。
            unknown = Path(tmp) / "unknown_ray_trainer.py"
            unknown.write_text(self.ORIGINAL_BODY + "\n# drifted\n", encoding="utf-8")
            before = unknown.read_bytes()
            with self.assertRaises(RuntimeError) as ctx:
                patcher.apply_patch(unknown)
            self.assertIn("refusing to patch unknown ray_trainer.py", str(ctx.exception))
            self.assertEqual(unknown.read_bytes(), before)
            self.assertFalse(Path(str(unknown) + patcher.BACKUP_SUFFIX).exists())

    def test_apply_rolls_back_when_patched_hash_does_not_match(self):
        """fake patch 只写 marker 不产出正确 hash：apply 必须回滚到原始字节。"""
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "ray_trainer.py"
            original.write_text(self.ORIGINAL_BODY, encoding="utf-8")
            patched = Path(tmp) / "expected_patched.py"
            # 期望的 patched 内容与 fake patch 程序的实际产出不同（多了额外行）。
            patched.write_text(
                self.ORIGINAL_BODY + f"# {patcher.PATCH_MARKER}\n# extra drift\n",
                encoding="utf-8",
            )
            self._install_fake_hashes(original, patched)

            with patch.object(
                patcher.shutil, "which", return_value="fake-patch"
            ), patch.object(
                patcher.subprocess,
                "run",
                side_effect=lambda cmd, **kw: self._fake_patch_program(
                    Path(cmd[4]), patcher.PATCH_MARKER
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError) as ctx:
                        patcher.apply_patch(original)
            self.assertIn("hash mismatch", str(ctx.exception))
            # 回滚后必须回到原始 hash，且 backup 仍在。
            self.assertEqual(_sha(original.read_bytes()), patcher.EXPECTED_ORIGINAL_SHA256)
            backup = Path(str(original) + patcher.BACKUP_SUFFIX)
            self.assertTrue(backup.is_file())

    def test_verify_patched_rejects_markerless_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "ray_trainer.py"
            original.write_text(self.ORIGINAL_BODY, encoding="utf-8")
            # fake patched 文件 hash 正确但缺少 marker；同样必须拒绝。
            patched = Path(tmp) / "expected_patched.py"
            patched.write_text(self.ORIGINAL_BODY + "# unrelated\n", encoding="utf-8")
            self._install_fake_hashes(original, patched)
            with self.assertRaises(RuntimeError) as ctx:
                patcher.verify_patched(original)
            self.assertIn("hash mismatch", str(ctx.exception))
            with self.assertRaises(RuntimeError) as ctx:
                patcher.verify_patched(patched)
            self.assertIn("missing marker", str(ctx.exception))


def _shopping_info(utility: float, *, success: bool, infrastructure_invalid=False, **extra) -> dict:
    reward = {
        "full": float(utility),
        "strict": float(utility),
        "native": float(utility),
        "semantic": float(utility),
        "total": float(utility),
        "efficiency": 0.0,
        "penalty_overlong": 0.0,
        "penalty_unfinished": 0.0,
        "penalty_repeat": 0.0,
        "repeat_action_rate": 0.0,
        "r_type": 0.0,
        "r_att": 0.0,
        "r_option": 0.0,
        "r_price": 0.0,
        "terminal_utility": float(utility),
        "purchase_success": success,
        "sampling_invalid": False,
    }
    return {
        "infrastructure_invalid": infrastructure_invalid,
        "reward": reward,
        "steps": 3,
        "done": True,
        "termination_reason": "gold_purchase" if success else "finish_without_purchase",
        **extra,
    }


class DynamicSamplingDiagnosticsTest(unittest.TestCase):
    """fake shopping_infos：mixed / all-equal / invalid group 的诊断字段。"""

    def test_mixed_group_kept_and_equal_or_invalid_groups_dropped(self):
        infos = (
            [_shopping_info(0.0, success=False),
             _shopping_info(1.0, success=True),
             _shopping_info(0.0, success=False),
             _shopping_info(0.25, success=False)]
            + [_shopping_info(0.0, success=False)] * 4
            + [_shopping_info(1.0, success=True, infrastructure_invalid=True)] * 4
        )
        uids = ["mixed"] * 4 + ["all_equal"] * 4 + ["invalid"] * 4
        seq_rewards = [float(info["reward"]["total"]) for info in infos]
        utilities, successes, invalids, reasons = extract_shopping_group_signals(infos)
        indices, stats = select_reward_varying_groups(
            uids,
            seq_rewards,
            terminal_utilities=utilities,
            purchase_success=successes,
            sampling_invalid=invalids,
            sampling_invalid_reasons=reasons,
        )
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["num_groups"], 3)
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["dropped_group_count"], 2)
        self.assertEqual(stats["all_equal_group_count"], 2)
        self.assertEqual(stats["sampling_invalid_group_count"], 1)
        self.assertEqual(stats["kept_uids"], ("mixed",))
        self.assertEqual(stats["dropped_uids"], ("all_equal", "invalid"))
        self.assertEqual(
            stats["sampling_invalid_reason_counts"], {"infrastructure_invalid": 1}
        )
        drop_reasons = {group["uid"]: group["drop_reason"] for group in stats["groups"]}
        self.assertEqual(drop_reasons["all_equal"], "constant_reward")
        self.assertEqual(drop_reasons["invalid"], "sampling_invalid")
        self.assertIsNone(drop_reasons["mixed"])

        metrics = aggregate_shopping_metrics(infos)
        self.assertAlmostEqual(metrics["reward/purchase_success_rate"], 5 / 12)
        self.assertAlmostEqual(metrics["trajectory/infrastructure_invalid_rate"], 4 / 12)
        self.assertAlmostEqual(metrics["reward/terminal_utility_mean"], 5.25 / 12)

    def test_build_rollout_diagnostics_assigns_rollout_index_per_uid(self):
        infos = [_shopping_info(0.0, success=False)] * 6
        records = build_rollout_diagnostics(["g1", "g1", "g1", "g2", "g2", "g2"], infos)
        self.assertEqual(
            [record["rollout_index"] for record in records], [0, 1, 2, 0, 1, 2]
        )
        self.assertEqual([record["uid"] for record in records], ["g1"] * 3 + ["g2"] * 3)
        with self.assertRaises(ValueError):
            build_rollout_diagnostics(["g1"], infos)

    def test_extract_signals_rejects_missing_or_non_finite_fields(self):
        with self.assertRaises(ValueError):
            extract_shopping_group_signals([{"reward": {}}])
        broken = _shopping_info(0.0, success=False)
        broken["reward"]["terminal_utility"] = float("nan")
        with self.assertRaises(ValueError):
            extract_shopping_group_signals([broken])
        overlong = _shopping_info(1.0, success=True, overlong=True)
        _, _, invalid, reasons = extract_shopping_group_signals([overlong])
        self.assertTrue(invalid[0])
        self.assertIn("overlong", reasons[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
