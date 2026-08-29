#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe.

启动闭环（WP2）：
- 启动前校验 model contract（必须是带 ``merge_manifest.json`` 验收记录的 merged
  checkpoint：base=Qwen/Qwen3.5-2B、``merge.base_revision`` 与
  ``--expected-model-revision`` 一致、manifest ``output`` 与目录一致）与 data
  contract（parquet 必须带 metadata.json 且 hash 一致）；
- actor 与 reference 必须解析到同一个 merged checkpoint 路径（``--ref-model`` 只允许
  显式给出同一路径，用于证明冻结 ref 没有被改指 Base 或其他 checkpoint）；
- reward/environment/tool-schema/system-prompt 版本在 data metadata、runtime
  contract 与代码内 ``SYSTEM_PROMPT`` 之间三方交叉校验，不一致拒绝启动；
- resolved command、contract 快照与 resolved config 写入 run manifest（不含 secret）；
- ``--smoke`` 显式 1–2 optimizer step 模式；
- ``--resume`` / ``--resume-from`` 在任何 contract hash 漂移、或 checkpoint 缺少
  actor/optimizer/scheduler/RNG/global-step 产物时拒绝续训并要求新 run；
- run manifest 在 runtime preflight（check_grpo_runtime.py）与 resolved config dump
  全部通过之后才落盘，preflight 失败不留半成品 manifest；
- 训练结束后扫描 output 目录写 checkpoint_manifest.json；
- ``--dry-run`` 只做校验并打印不含 secret 的 resolved env/command。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许 `python scripts/train_grpo.py` 直接运行：metadata schema 常量必须与数据
    # 构建器保持同一来源（见下），因此把仓库根目录放进 sys.path。
    sys.path.insert(0, str(ROOT))

from scripts.build_grpo_data import (  # noqa: E402
    METADATA_SCHEMA as DATA_METADATA_SCHEMA,  # 单一常量来源：构建器与 launcher 完全一致
)

DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/validation.parquet"
DEFAULT_RUNTIME_CONTRACT = ROOT / "data/manifests/runtime_contract.json"

RUN_MANIFEST_FILE = "run_manifest.json"
RESOLVED_CONFIG_FILE = "resolved_config.yaml"
CHECKPOINT_MANIFEST_FILE = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA = "shopping-grpo-checkpoint-manifest-v1"
MERGE_MANIFEST_FILE = "merge_manifest.json"
GRPO_RUN_MANIFEST_SCHEMA = "shopping-grpo-run-manifest-v1"
ADAPTER_ONLY_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
EXPECTED_MERGE_DTYPE = "bfloat16"
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
# GRPO 的 actor/reference 只能来自 Qwen/Qwen3.5-2B 的 M2 merged checkpoint。
EXPECTED_BASE_MODEL = "Qwen/Qwen3.5-2B"
# Qwen/Qwen3.5-2B refs/main；merge manifest 必须用同一 revision 记录（scripts/
# merge_lora_adapter.py --revision）。与 SFT/merge/verify 合同保持一致。
EXPECTED_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# veRL 0.8.0 resumable checkpoint 布局（verl/utils/checkpoint/fsdp_checkpoint_manager.py、
# verl/trainer/ppo/ray_trainer.py::_save_checkpoint）：
#   {output}/global_step_{N}/actor/model_world_size_{w}_rank_{r}.pt        # actor 权重
#   {output}/global_step_{N}/actor/optim_world_size_{w}_rank_{r}.pt        # optimizer
#   {output}/global_step_{N}/actor/extra_state_world_size_{w}_rank_{r}.pt  # scheduler + RNG
#   {output}/latest_checkpointed_iteration.txt                             # global step tracker
# 远端若文件名有出入，用 --resume-expected-file 覆盖 glob 列表。
DEFAULT_CHECKPOINT_ARTIFACT_GLOBS = (
    "actor/model_world_size_*_rank_*.pt",
    "actor/optim_world_size_*_rank_*.pt",
    "actor/extra_state_world_size_*_rank_*.pt",
)
DEFAULT_GLOBAL_STEP_FILES = (
    "latest_checkpointed_iteration.txt",
    "latest_checkpoint_tokened_step.json",
    "trainer_state.json",
)
_STEP_DIR_RE = re.compile(r"(?:global_step_|step_)?(\d+)")
# 记录进 run manifest 的环境变量：只记录本项目显式设置的非 secret 变量。
RUNTIME_ENV_KEYS = (
    "PYTHONPATH",
    "SHOPPING_GRPO_ROOT",
    "SHOPPING_ENVIRONMENT_VERSION",
    "SHOPPING_ENV_MANIFEST",
    "GRPO_MODEL_PATH",
    "GRPO_TRAIN_FILE",
    "GRPO_VAL_FILE",
    "GRPO_OUTPUT_DIR",
    "SHOPPING_GRPO_DIAGNOSTICS_PATH",
    "SHOPSIM_BASE_URL",
    "SHOPPING_AGENT_LOOP_CONFIG",
    "SHOPPING_TOOL_CONFIG",
    "GRPO_CONFIG_NAME",
)


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--ref-model",
        type=Path,
        help="reference 模型目录；必须与 --model 解析到同一路径（冻结 ref = 同一 M2 merged）",
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="显式 1–2 optimizer step 冒烟模式（覆盖 total_training_steps/save_freq/test_freq）",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        choices=(1, 2),
        default=2,
        help="--smoke 模式下的 optimizer step 数（默认 2）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --output 下最新 checkpoint 续训；任何 contract hash 漂移或 checkpoint "
        "产物缺失都会拒绝",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="从指定 checkpoint 目录续训（等价 --resume 但显式给出路径）",
    )
    parser.add_argument(
        "--resume-expected-file",
        action="append",
        help="覆盖 resume checkpoint 必须包含的文件 glob（可多次传入；缺省使用 veRL 0.8 布局）",
    )
    parser.add_argument(
        "--expected-base-model",
        default=EXPECTED_BASE_MODEL,
        help=f"merge manifest 记录的基座必须指向该模型（默认 {EXPECTED_BASE_MODEL}）",
    )
    parser.add_argument(
        "--expected-model-revision",
        default=EXPECTED_MODEL_REVISION,
        help="merge manifest 记录的 merge.base_revision 必须等于该值（Qwen/Qwen3.5-2B refs/main）",
    )
    parser.add_argument(
        "--allow-unverified-revision",
        action="store_true",
        help="merge manifest 缺少 merge.base_revision 时显式放行（记录进 run manifest，不会静默通过）",
    )
    parser.add_argument(
        "--data-metadata",
        type=Path,
        help="GRPO data metadata.json；缺省取 train parquet 同目录的 metadata.json",
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=DEFAULT_RUNTIME_CONTRACT,
        help="runtime_contract.json；与 data metadata 的版本字段交叉校验"
        f"（缺省 {DEFAULT_RUNTIME_CONTRACT}，即冻结证据链仓库的 data/manifests/）",
    )
    parser.add_argument(
        "--dump-resolved-config",
        type=Path,
        help="resolved config 落盘路径；缺省写 <output>/resolved_config.yaml"
        "（通过 `hydra --cfg job --resolve` 获取真正 resolved 的 veRL 配置）",
    )
    parser.add_argument(
        "--no-dump-resolved-config",
        action="store_true",
        help="显式跳过 resolved config dump（会记录进 run manifest，不会静默）",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _is_resuming(args: argparse.Namespace) -> bool:
    return bool(args.resume or args.resume_from)


def hydra_overrides(args: argparse.Namespace) -> list[str]:
    """构造传给 veRL / runtime preflight 的 Hydra override 列表。

    顺序即优先级：logger 与 experiment_name 最先，用户 override 其次，
    --smoke 固定步数最后追加，保证 smoke 的 1–2 step 预算不被覆盖。
    """
    overrides = [
        (
            "trainer.logger=[console,swanlab]"
            if args.logger == "swanlab"
            else "trainer.logger=[console]"
        ),
        f"trainer.experiment_name={args.experiment_name}",
    ]
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    overrides.extend(extra)
    if args.smoke:
        overrides.extend(
            [
                f"trainer.total_training_steps={int(args.smoke_steps)}",
                "trainer.save_freq=1",
                "trainer.test_freq=1",
                "trainer.val_before_train=true",
            ]
        )
    if args.resume_from is not None:
        overrides.extend(
            [
                # veRL 0.8.0's registered enum is ``resume_path``.  The old
                # ``resumable_path`` spelling reaches Hydra but is rejected by
                # the trainer at startup.
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={Path(args.resume_from).expanduser().resolve()}",
            ]
        )
    elif args.resume:
        overrides.append("trainer.resume_mode=auto")
    return overrides


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        # 续训必须复用已有 run 目录；全新 run 仍然要求目录为空。
        if any(output.iterdir()) and not _is_resuming(args):
            raise SystemExit(f"output directory must be new or empty: {output}")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": ENVIRONMENT_VERSION,
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *hydra_overrides(args),
    ]
    return command, environment


def _load_json_file(path: Path, description: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {description}: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{description} is not valid JSON: {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    """流式 SHA256；避免在启动路径上重复实现 hash 逻辑。"""
    from shopping_grpo.training.sft.run_manifest import sha256_file

    return sha256_file(path)


def _ensure_no_secrets(manifest: dict) -> None:
    from shopping_grpo.training.sft.run_manifest import ensure_no_secrets

    ensure_no_secrets(manifest)


def base_model_identity_matches(actual: object, expected: str) -> bool:
    """merge.source.base_model 可能是 HF repo 名或本地权重目录；两者都指向同一基座。

    例：``Qwen/Qwen3.5-2B`` 与 ``/root/autodl-tmp/models/Qwen3.5-2B`` 视为同一基座；
    ``Qwen/Qwen3.5-2B-Base``（Base 变体）不匹配。
    """
    if not isinstance(actual, str) or not actual:
        return False
    if actual == expected:
        return True
    return Path(actual).name == Path(expected).name


def validate_actor_ref_paths(args: argparse.Namespace) -> dict:
    """冻结 reference 与 actor 同源：resolved 后必须逐字节同一路径。"""
    actor_path = Path(args.model).expanduser().resolve()
    if args.ref_model is None:
        ref_path = actor_path
        ref_source = "default (same as --model)"
    else:
        ref_path = Path(args.ref_model).expanduser().resolve()
        ref_source = "--ref-model"
        if ref_path != actor_path:
            raise SystemExit(
                "ref model must resolve to the same M2 merged checkpoint as the actor: "
                f"actor={actor_path}, ref={ref_path}; GRPO forbids pointing the frozen "
                "reference at Base or another checkpoint"
            )
    return {
        "actor_model_path": str(actor_path),
        "ref_model_path": str(ref_path),
        "ref_source": ref_source,
        "ref_matches_actor": True,
    }


def validate_merged_model(
    model: Path,
    *,
    expected_base_model: str = EXPECTED_BASE_MODEL,
    expected_revision: str = EXPECTED_MODEL_REVISION,
    allow_unverified_revision: bool = False,
) -> dict:
    """GRPO 只能从已验收的 merged checkpoint（M2）初始化，拒绝 adapter 或裸基座。"""
    manifest_path = model / MERGE_MANIFEST_FILE
    if not manifest_path.is_file():
        raise SystemExit(
            "GRPO requires a merged checkpoint with "
            f"{MERGE_MANIFEST_FILE}; not found in {model}. "
            "Run scripts/merge_lora_adapter.py + scripts/verify_merged_checkpoint.py first."
        )
    merge_manifest = _load_json_file(manifest_path, "merge manifest")
    verification = merge_manifest.get("verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise SystemExit(
            f"{manifest_path} has no passed verification block; "
            "run scripts/verify_merged_checkpoint.py --merged-dir <dir> first"
        )
    adapter_leftovers = [name for name in ADAPTER_ONLY_FILES if (model / name).is_file()]
    if adapter_leftovers:
        raise SystemExit(
            "model directory still contains LoRA adapter-only files; "
            f"GRPO needs the merged checkpoint: {adapter_leftovers}"
        )
    merge_block = merge_manifest.get("merge") or {}
    source_block = merge_manifest.get("source") or {}
    dtype = merge_block.get("dtype")
    if dtype != EXPECTED_MERGE_DTYPE:
        raise SystemExit(
            f"merged checkpoint dtype must be {EXPECTED_MERGE_DTYPE}, got {dtype!r}"
        )
    recorded_output = merge_manifest.get("output")
    if not recorded_output:
        raise SystemExit(
            f"{manifest_path} does not record its output directory; "
            "re-merge with scripts/merge_lora_adapter.py"
        )
    if Path(str(recorded_output)).expanduser().resolve() != model:
        raise SystemExit(
            f"{manifest_path} records output {recorded_output!r} which is not {model}; "
            "refusing to trust a manifest that describes a different checkpoint"
        )
    base_model = source_block.get("base_model")
    if not base_model_identity_matches(base_model, expected_base_model):
        raise SystemExit(
            "GRPO requires the merged checkpoint to be built from base model "
            f"{expected_base_model!r}, but {manifest_path} records {base_model!r}"
        )
    base_revision = merge_block.get("base_revision")
    revision_verified = False
    revision_note = None
    if base_revision is None:
        if not allow_unverified_revision:
            raise SystemExit(
                f"{manifest_path} does not record merge.base_revision; re-merge with "
                f"`scripts/merge_lora_adapter.py --revision {expected_revision}` or pass "
                "--allow-unverified-revision explicitly (recorded in the run manifest)"
            )
        revision_note = (
            "unverified: merge manifest lacks merge.base_revision; "
            "accepted via --allow-unverified-revision"
        )
    elif str(base_revision) != expected_revision:
        raise SystemExit(
            "merged checkpoint base revision mismatch: merge manifest records "
            f"{base_revision!r}, expected {expected_revision!r} ({expected_base_model} refs/main)"
        )
    else:
        revision_verified = True
    return {
        "merge_manifest_sha256": _sha256_file(manifest_path),
        "merge_dtype": dtype,
        "base_model": str(base_model),
        "base_revision": str(base_revision) if base_revision is not None else None,
        "revision_verified": revision_verified,
        "revision_note": revision_note,
        "adapter_source": (
            str(source_block["adapter"]) if source_block.get("adapter") else None
        ),
        "verification_passed": True,
    }


def _metadata_entry_for(metadata: dict, key: str, parquet_path: Path) -> dict:
    files = metadata.get("files")
    entry = (files or {}).get(key)
    if not isinstance(entry, dict):
        raise SystemExit(f"GRPO data metadata is missing the files.{key} entry")
    recorded_name = Path(str(entry.get("path", ""))).name
    if recorded_name != parquet_path.name:
        raise SystemExit(
            f"metadata files.{key}.path ({recorded_name!r}) does not describe {parquet_path.name}"
        )
    return entry


def validate_sft_source_contract(leakage: dict) -> dict:
    """Independently validate the four-source SFT provenance contract.

    ``all_checks_passed`` is only a summary and is therefore never trusted on
    its own.  This check intentionally validates the shape, identities, and
    non-empty/hash-bearing records before any training process is started.
    """
    from scripts.build_grpo_data import SFT_SOURCE_IDENTITIES

    audit = leakage.get("sft_task_id_check")
    if not isinstance(audit, dict) or audit.get("checked") is not True:
        raise SystemExit(
            "GRPO data metadata must record a checked four-source SFT leakage audit"
        )
    if audit.get("source_count") != len(SFT_SOURCE_IDENTITIES):
        raise SystemExit(
            "GRPO data metadata SFT source_count must be exactly 4, got "
            f"{audit.get('source_count')!r}"
        )
    sources = audit.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SFT_SOURCE_IDENTITIES):
        raise SystemExit(
            "GRPO data metadata must identify exactly the four SFT sources: "
            + ", ".join(SFT_SOURCE_IDENTITIES)
        )
    for identity in SFT_SOURCE_IDENTITIES:
        record = sources[identity]
        if not isinstance(record, dict) or record.get("identity") != identity:
            raise SystemExit(f"SFT source {identity!r} has invalid identity metadata")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise SystemExit(f"SFT source {identity!r} is missing its source path")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            raise SystemExit(f"SFT source {identity!r} has invalid source sha256")
        if not isinstance(record.get("task_count"), int) or record["task_count"] <= 0:
            raise SystemExit(f"SFT source {identity!r} must be non-empty")
        if not isinstance(record.get("task_ids_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["task_ids_sha256"]
        ):
            raise SystemExit(f"SFT source {identity!r} has invalid task id hash")
    overlaps = audit.get("overlaps")
    if not isinstance(overlaps, dict) or any(value for value in overlaps.values()):
        raise SystemExit("GRPO data metadata SFT leakage audit contains overlaps")
    return {"source_count": 4, "source_identities": list(SFT_SOURCE_IDENTITIES)}


def validate_data_metadata(train_data: Path, val_data: Path, metadata_path: Path) -> dict:
    """parquet 必须带 build_grpo_data.py 生成的 metadata 且文件 hash 一致。"""
    if not metadata_path.is_file():
        raise SystemExit(
            f"GRPO data metadata is missing: {metadata_path}; "
            "run scripts/build_grpo_data.py first"
        )
    metadata = _load_json_file(metadata_path, "GRPO data metadata")
    if metadata.get("schema_version") != DATA_METADATA_SCHEMA:
        raise SystemExit(
            f"unexpected GRPO data metadata schema_version: {metadata.get('schema_version')!r} "
            f"(expected {DATA_METADATA_SCHEMA})"
        )
    if metadata.get("example") is True:
        raise SystemExit(
            f"{metadata_path} is an example metadata file; build the real parquet first"
        )
    leakage = metadata.get("leakage") or {}
    if leakage.get("all_checks_passed") is not True:
        raise SystemExit("GRPO data metadata leakage checks did not pass; refusing to launch")
    # Validate each provenance field independently before consulting the summary
    # bit, so forged ``all_checks_passed=true`` metadata cannot launch training.
    sft_info = validate_sft_source_contract(leakage)

    versions = {
        key: (metadata.get("source") or {}).get(key)
        for key in ("reward_version", "environment_version", "tool_schema_sha256", "system_prompt_sha256")
    }
    missing_versions = sorted(key for key, value in versions.items() if not value)
    if missing_versions:
        raise SystemExit(
            f"GRPO data metadata is missing version fields {missing_versions}; "
            "rebuild with scripts/build_grpo_data.py so the launcher can cross-check "
            "reward/environment/tool/system-prompt versions"
        )
    # The builder records the exact tokenizer revision used for prompt-length
    # auditing.  Treat this as an independent launcher contract: a forged or
    # legacy metadata file must not silently pair data with another tokenizer.
    from scripts.build_grpo_data import FROZEN_TOKENIZER_REVISION

    tokenizer_revision = (metadata.get("build_parameters") or {}).get("tokenizer_revision")
    if tokenizer_revision != FROZEN_TOKENIZER_REVISION:
        raise SystemExit(
            "GRPO data metadata tokenizer_revision must be the frozen Qwen revision "
            f"{FROZEN_TOKENIZER_REVISION!r}; got {tokenizer_revision!r}"
        )
    try:
        from shopping_grpo.environment.tools import validate_runtime_tool_schema

        canonical_tool_hash = validate_runtime_tool_schema(DEFAULT_TOOL_CONFIG)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if versions["tool_schema_sha256"] != canonical_tool_hash:
        raise SystemExit(
            "GRPO data metadata tool_schema_sha256 does not match the canonical "
            f"configs/tools.json/Python schema hash {canonical_tool_hash}: "
            f"got {versions['tool_schema_sha256']!r}"
        )

    validated = {}
    for key, parquet in (("grpo_train", train_data), ("grpo_validation", val_data)):
        entry = _metadata_entry_for(metadata, key, parquet)
        recorded_hash = entry.get("file_sha256")
        actual_hash = _sha256_file(parquet)
        if recorded_hash != actual_hash:
            raise SystemExit(
                f"{parquet} hash mismatch: metadata records {recorded_hash!r}, file is {actual_hash!r}; "
                "rebuild with scripts/build_grpo_data.py or fix the data directory"
            )
        rows = int(entry.get("rows", 0))
        unique_tasks = int(entry.get("unique_task_ids", 0))
        if rows <= 0 or rows != unique_tasks:
            raise SystemExit(
                f"metadata files.{key} rows/unique_task_ids inconsistent: rows={rows}, unique={unique_tasks}"
            )
        validated[key] = {"rows": rows, "file_sha256": actual_hash}
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "runtime_contract_sha256": (metadata.get("source") or {}).get("runtime_contract_sha256"),
        "tokenizer_revision": tokenizer_revision,
        "train_rows": validated["grpo_train"]["rows"],
        "validation_rows": validated["grpo_validation"]["rows"],
        "versions": versions,
        "sft": sft_info,
    }


def compute_local_system_prompt_sha256() -> str:
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def validate_runtime_contract(data_info: dict, contract: dict) -> dict:
    """runtime contract ↔ data metadata ↔ 代码内 SYSTEM_PROMPT 三方交叉校验。

    reward/environment/tool-schema/system-prompt 任一不一致都拒绝启动（audit item 7）。
    """
    declared = contract.get("contract_sha256")
    if declared != data_info.get("runtime_contract_sha256"):
        raise SystemExit(
            "runtime contract does not match the one recorded in GRPO data metadata: "
            f"metadata={data_info.get('runtime_contract_sha256')!r}, contract={declared!r}"
        )
    from scripts.build_grpo_data import METADATA_CONTRACT_VERSION_FIELDS
    from shopping_grpo.environment.tools import validate_runtime_tool_schema

    versions = data_info["versions"]
    try:
        canonical_tool_hash = validate_runtime_tool_schema(DEFAULT_TOOL_CONFIG)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if versions["tool_schema_sha256"] != canonical_tool_hash:
        raise SystemExit(
            "GRPO data metadata tool_schema_sha256 does not match the canonical "
            f"configs/tools.json/Python schema hash {canonical_tool_hash}: "
            f"got {versions['tool_schema_sha256']!r}"
        )
    local_prompt_sha256 = compute_local_system_prompt_sha256()
    if versions["system_prompt_sha256"] != local_prompt_sha256:
        raise SystemExit(
            "GRPO data metadata system_prompt_sha256 does not match the installed "
            "shopping_grpo.evaluation.rollout.SYSTEM_PROMPT: "
            f"metadata={versions['system_prompt_sha256']!r}, local={local_prompt_sha256!r}; "
            "the parquet was built against a different system prompt, rebuild the data"
        )
    mismatches = {}
    for metadata_key, contract_key in METADATA_CONTRACT_VERSION_FIELDS:
        declared_value = contract.get(contract_key)
        if declared_value != versions[metadata_key]:
            mismatches[metadata_key] = {
                "metadata": versions[metadata_key],
                "runtime_contract": declared_value,
            }
    if mismatches:
        raise SystemExit(
            "runtime contract disagrees with GRPO data metadata: "
            + json.dumps(mismatches, sort_keys=True)
            + "; refusing to launch on reward/environment/tool/system-prompt drift"
        )
    if str(contract.get("environment_version")) != ENVIRONMENT_VERSION:
        raise SystemExit(
            f"launcher pins environment {ENVIRONMENT_VERSION!r} but the runtime contract "
            f"declares {contract.get('environment_version')!r}"
        )
    return {**versions, "system_prompt_sha256": local_prompt_sha256}


def build_contract_snapshot(
    args: argparse.Namespace,
    model_info: dict,
    data_info: dict,
    model_paths: dict,
    contract: dict,
) -> dict:
    """resume 比对用的 contract 快照：任何成员变化都意味着不能续训。"""
    from shopping_grpo.training.sft.run_manifest import hash_weight_files

    weights = hash_weight_files(args.model)
    versions = data_info["versions"]
    snapshot = {
        "model_weights_sha256": weights.get("weights_sha256"),
        "model_merge_manifest_sha256": model_info["merge_manifest_sha256"],
        "actor_model_path": model_paths["actor_model_path"],
        "ref_model_path": model_paths["ref_model_path"],
        "expected_model_revision": str(args.expected_model_revision),
        "train_data_sha256": _sha256_file(args.train_data.expanduser().resolve()),
        "val_data_sha256": _sha256_file(args.val_data.expanduser().resolve()),
        "data_metadata_sha256": data_info["metadata_sha256"],
        "config_sha256": _sha256_file(args.config.expanduser().resolve()),
        "environment_version": ENVIRONMENT_VERSION,
        "runtime_contract_sha256": contract.get("contract_sha256"),
        "reward_version": versions["reward_version"],
        "tool_schema_sha256": versions["tool_schema_sha256"],
        "system_prompt_sha256": versions["system_prompt_sha256"],
    }
    return snapshot


def load_runtime_contract(args: argparse.Namespace) -> tuple[Path, dict]:
    contract_path = Path(args.runtime_contract).expanduser().resolve()
    if not contract_path.is_file():
        raise SystemExit(
            f"runtime contract not found: {contract_path}; pass --runtime-contract pointing "
            "to the frozen data/manifests/runtime_contract.json so reward/environment/tool/"
            "system-prompt versions can be cross-checked"
        )
    return contract_path, _load_json_file(contract_path, "runtime contract")


def validate_resume(output: Path, snapshot: dict) -> dict:
    """resume 前比对 contract 快照；任何漂移都拒绝并要求新 run。"""
    run_manifest_path = output / RUN_MANIFEST_FILE
    if not run_manifest_path.is_file():
        raise SystemExit(
            f"--resume requires a prior {RUN_MANIFEST_FILE} in {output}; "
            "start a new run without --resume/--resume-from"
        )
    prior = _load_json_file(run_manifest_path, "prior GRPO run manifest")
    prior_snapshot = prior.get("contract")
    if not isinstance(prior_snapshot, dict):
        raise SystemExit(
            f"prior {RUN_MANIFEST_FILE} in {output} has no contract snapshot; cannot verify resume"
        )
    drift = {
        key: {"prior": prior_snapshot.get(key), "current": value}
        for key, value in snapshot.items()
        if prior_snapshot.get(key) != value
    }
    if drift:
        raise SystemExit(
            "refusing to resume: run contract drifted: "
            + json.dumps(drift, sort_keys=True)
            + f"; create a new run in a fresh output directory instead of {output}"
        )
    return {"run_manifest_path": str(run_manifest_path), "prior_stage": prior.get("stage")}


def parse_step_dir_name(name: str) -> int | None:
    match = _STEP_DIR_RE.fullmatch(name)
    return int(match.group(1)) if match else None


def find_latest_checkpoint_dir(output: Path) -> Path:
    """veRL 0.8 布局：resume checkpoint 是 ``{output}/global_step_{N}``。"""
    candidates = []
    for child in sorted(output.iterdir()):
        if child.is_dir():
            step = parse_step_dir_name(child.name)
            if step is not None:
                candidates.append((step, child))
    if not candidates:
        raise SystemExit(
            f"--resume could not locate a global_step_N checkpoint directory under {output}; "
            "start a new run or pass --resume-from <checkpoint dir>"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _read_global_step_file(path: Path) -> int | None:
    """读取 global step 文件：纯文本整数或含单个整数值的 JSON。"""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        values = {
            value
            for value in data.values()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if len(values) == 1:
            return values.pop()
        return None
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    try:
        return int(text)
    except ValueError:
        return None


def validate_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    artifact_globs: list[str] | tuple[str, ...] | None = None,
    global_step_files: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """resume checkpoint 必须包含 actor/optimizer/scheduler/RNG 产物且 global step 可读。

    缺任一项都拒绝续训并要求新 run；文件名以可注入的 expected glob 列表校验
    （CLI: --resume-expected-file，Python: artifact_globs 参数）。
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise SystemExit(
            f"resume checkpoint does not exist or is not a directory: {checkpoint_dir}"
        )
    globs = tuple(artifact_globs) if artifact_globs else DEFAULT_CHECKPOINT_ARTIFACT_GLOBS
    step_files = tuple(global_step_files) if global_step_files else DEFAULT_GLOBAL_STEP_FILES

    matched: dict[str, list[str]] = {}
    missing: list[str] = []
    for pattern in globs:
        hits = sorted(item.name for item in checkpoint_dir.glob(pattern))
        if hits:
            matched[pattern] = hits
        else:
            missing.append(pattern)
    if missing:
        raise SystemExit(
            f"refusing to resume: checkpoint {checkpoint_dir} is missing required veRL 0.8 "
            f"artifacts {missing} (actor weights / optimizer / scheduler+RNG state); "
            "start a new run instead"
        )

    step = None
    step_source = None
    for name in step_files:
        candidate = checkpoint_dir / name
        if candidate.is_file():
            step = _read_global_step_file(candidate)
            if step is not None:
                step_source = f"checkpoint file {name}"
                break
    if step is None:
        dir_step = parse_step_dir_name(checkpoint_dir.name)
        if dir_step is not None:
            step, step_source = dir_step, "checkpoint directory name"
    if step is None:
        for name in step_files:
            candidate = checkpoint_dir.parent / name
            if candidate.is_file():
                step = _read_global_step_file(candidate)
                if step is not None:
                    step_source = f"tracker file {name} next to the checkpoint"
                    break
    if step is None:
        raise SystemExit(
            f"refusing to resume: cannot read the global step from checkpoint {checkpoint_dir} "
            f"(looked for {list(step_files)} and a global_step_N directory name); "
            "start a new run instead"
        )
    return {
        "path": str(checkpoint_dir),
        "global_step": int(step),
        "global_step_source": step_source,
        "matched_artifacts": matched,
    }


def preflight(args: argparse.Namespace) -> dict:
    """启动前合同校验；--dry-run 也会完整执行。返回 contract 快照与校验记录。"""
    model = args.model.expanduser().resolve()
    train_data = args.train_data.expanduser().resolve()
    val_data = args.val_data.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if args.smoke and _is_resuming(args):
        raise SystemExit("--smoke must be a fresh 1-2 step run; --resume/--resume-from is not allowed")
    if args.resume and args.resume_from is not None:
        raise SystemExit("--resume and --resume-from are mutually exclusive")
    if args.resume_from is not None and not Path(args.resume_from).exists():
        raise SystemExit(f"resume checkpoint does not exist: {args.resume_from}")
    if args.no_dump_resolved_config and args.dump_resolved_config is not None:
        raise SystemExit("--no-dump-resolved-config and --dump-resolved-config are mutually exclusive")

    model_paths = validate_actor_ref_paths(args)
    model_info = validate_merged_model(
        model,
        expected_base_model=args.expected_base_model,
        expected_revision=args.expected_model_revision,
        allow_unverified_revision=args.allow_unverified_revision,
    )
    metadata_path = (
        args.data_metadata.expanduser().resolve()
        if args.data_metadata
        else train_data.parent / "metadata.json"
    )
    data_info = validate_data_metadata(train_data, val_data, metadata_path)
    contract_path, contract = load_runtime_contract(args)
    version_info = validate_runtime_contract(data_info, contract)
    snapshot = build_contract_snapshot(args, model_info, data_info, model_paths, contract)

    resume_info = None
    if _is_resuming(args):
        resume_info = validate_resume(output, snapshot)
        tracker_step = None
        if args.resume_from is not None:
            checkpoint_dir = Path(args.resume_from).expanduser().resolve()
        else:
            checkpoint_dir = find_latest_checkpoint_dir(output)
            tracker_step = _read_global_step_file(output / "latest_checkpointed_iteration.txt")
            if tracker_step is None:
                raise SystemExit(
                    f"--resume requires a readable latest_checkpointed_iteration.txt tracker "
                    f"file in {output}; start a new run instead"
                )
            resume_info["tracker_step"] = tracker_step
        checkpoint_info = validate_resume_checkpoint(
            checkpoint_dir, artifact_globs=args.resume_expected_file
        )
        if tracker_step is not None and tracker_step != checkpoint_info["global_step"]:
            raise SystemExit(
                f"refusing to resume: tracker file records step {tracker_step} but the latest "
                f"checkpoint directory is step {checkpoint_info['global_step']}; "
                "start a new run instead"
            )
        resume_info["checkpoint"] = checkpoint_info
    return {
        "model": model_info,
        "model_paths": model_paths,
        "data": data_info,
        "versions": version_info,
        "runtime_contract_path": str(contract_path),
        "contract": snapshot,
        "resume": resume_info,
    }


def resolved_config_dump_command(command: list[str]) -> list[str]:
    """把 veRL 启动命令改写为 ``--cfg job --resolve``：打印 resolved config 后立即退出。"""
    if len(command) < 4 or command[1:3] != ["-m", "verl.trainer.main_ppo"]:
        raise SystemExit(f"cannot derive a resolved-config dump command from {command}")
    head = list(command[:4])
    config_flags = [
        flag for flag in command[4:] if flag.startswith("--config-path=") or flag.startswith("--config-name=")
    ]
    overrides = [flag for flag in command[4:] if flag not in config_flags]
    return [*head, *config_flags, "--cfg", "job", "--resolve", *overrides]


def dump_resolved_config(command: list[str], environment: dict[str, str], destination: Path) -> dict:
    """运行 dump 命令并把 resolved config 原样落盘；失败即拒绝启动。"""
    dump_cmd = resolved_config_dump_command(command)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(dump_cmd, cwd=ROOT, env=environment, capture_output=True)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
        raise SystemExit(
            f"failed to dump the resolved GRPO config (exit {proc.returncode}); "
            "refusing to launch without it: " + " | ".join(stderr_tail)
        )
    payload = proc.stdout
    if not payload.strip():
        raise SystemExit("resolved config dump is empty; refusing to launch without it")
    destination.write_bytes(payload)
    return {
        "status": "dumped",
        "path": str(destination),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "command": dump_cmd,
    }


def maybe_dump_resolved_config(
    args: argparse.Namespace, command: list[str], environment: dict[str, str], output: Path
) -> dict:
    if args.no_dump_resolved_config:
        return {
            "status": "skipped_explicitly",
            "note": "--no-dump-resolved-config; the resolved config is NOT recorded for this run",
        }
    destination = (
        args.dump_resolved_config.expanduser().resolve()
        if args.dump_resolved_config
        else output / RESOLVED_CONFIG_FILE
    )
    return dump_resolved_config(command, environment, destination)


def build_checkpoint_manifest(output: Path, *, exit_code: int) -> dict:
    """post-run 扫描 output 下的 veRL checkpoint（global_step_N）并记录关键文件 hash。"""
    checkpoints = []
    if output.is_dir():
        step_dirs = []
        for child in sorted(output.iterdir()):
            if child.is_dir():
                step = parse_step_dir_name(child.name)
                if step is not None:
                    step_dirs.append((step, child))
        step_dirs.sort(key=lambda item: item[0])
        for step, step_dir in step_dirs:
            files = {}
            for path in sorted(step_dir.rglob("*")):
                if path.is_file():
                    files[path.relative_to(step_dir).as_posix()] = _sha256_file(path)
            checkpoints.append(
                {
                    "global_step": step,
                    "path": str(step_dir),
                    "file_count": len(files),
                    "files": files,
                }
            )
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "output_dir": str(output),
        "exit_code": int(exit_code),
        "checkpoint_count": len(checkpoints),
        "latest_global_step": checkpoints[-1]["global_step"] if checkpoints else None,
        "checkpoints": checkpoints,
        "note": (
            "files sha256 covers actor/optim/extra_state shards and data.pt under each "
            "global_step_N directory"
            if checkpoints
            else "no global_step_N checkpoint directory found under output"
        ),
    }


def build_grpo_run_manifest(
    args: argparse.Namespace,
    command: list[str],
    environment: dict[str, str],
    preflight_result: dict,
) -> dict:
    """resolved command + contract 快照落盘；禁止 secret 字段（复用 SFT 守卫）。"""
    stage = "smoke" if args.smoke else ("resume" if _is_resuming(args) else "training")
    runtime_env = {key: environment[key] for key in RUNTIME_ENV_KEYS if key in environment}
    output = Path(args.output).expanduser().resolve()
    planned_dump_path = (
        args.dump_resolved_config.expanduser().resolve()
        if args.dump_resolved_config
        else output / RESOLVED_CONFIG_FILE
    )
    manifest = {
        "schema_version": GRPO_RUN_MANIFEST_SCHEMA,
        "stage": stage,
        "command": list(command),
        "hydra_overrides": hydra_overrides(args),
        "model": {
            "path": str(Path(args.model).expanduser().resolve()),
            "ref_model_path": preflight_result["model_paths"]["ref_model_path"],
            "ref_source": preflight_result["model_paths"]["ref_source"],
            "ref_matches_actor": preflight_result["model_paths"]["ref_matches_actor"],
            **preflight_result["model"],
        },
        "data": {
            "train_path": str(Path(args.train_data).expanduser().resolve()),
            "val_path": str(Path(args.val_data).expanduser().resolve()),
            "train_rows": preflight_result["data"]["train_rows"],
            "validation_rows": preflight_result["data"]["validation_rows"],
            "metadata_path": preflight_result["data"]["metadata_path"],
            "metadata_sha256": preflight_result["data"]["metadata_sha256"],
        },
        "versions": {
            **preflight_result["versions"],
            "runtime_contract_path": preflight_result["runtime_contract_path"],
            "runtime_contract_sha256": preflight_result["contract"]["runtime_contract_sha256"],
        },
        "recipe": {
            "config": str(args.config.expanduser().resolve()),
            "logger": args.logger,
            "experiment_name": args.experiment_name,
            "smoke": bool(args.smoke),
            "smoke_steps": int(args.smoke_steps) if args.smoke else None,
            "resume": bool(args.resume),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "expected_base_model": str(args.expected_base_model),
            "expected_model_revision": str(args.expected_model_revision),
            "allow_unverified_revision": bool(args.allow_unverified_revision),
            "resume_expected_files": list(args.resume_expected_file or []),
            "resolved_config_dump": not bool(args.no_dump_resolved_config),
        },
        "resolved_config": {
            "status": "pending",
            "planned_path": str(planned_dump_path),
        },
        "runtime": {
            "environment_version": ENVIRONMENT_VERSION,
            "env_url": str(args.env_url),
            "python_version": sys.version.split()[0],
            "swanlab_key_configured": bool(os.environ.get("SWANLAB_API_KEY")),
            "environment_variables": runtime_env,
        },
        "contract": preflight_result["contract"],
        "execution": {
            "launched_at_epoch_s": int(time.time()),
            "dry_run": bool(args.dry_run),
        },
    }
    if preflight_result.get("resume"):
        # 续训记录：checkpoint 位置、可读 global step 与已验证的产物清单。
        manifest["resume"] = preflight_result["resume"]
    _ensure_no_secrets(manifest)
    return manifest


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    # 合同校验放在打印之前：缺模型/缺 metadata/hash 漂移必须在 dry-run 就失败。
    preflight_result = preflight(args)
    run_manifest = build_grpo_run_manifest(args, command, environment, preflight_result)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "config": str(args.config.resolve()),
        "stage": run_manifest["stage"],
        "environment": run_manifest["runtime"]["environment_variables"],
        "run_manifest": run_manifest,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output = Path(environment["GRPO_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)

    # runtime preflight 必须先通过：失败时不能留下半成品 run manifest（audit item 3）。
    overrides = run_manifest["hydra_overrides"]
    runtime_preflight_cmd = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *overrides,
    ]
    preflight_status = subprocess.call(runtime_preflight_cmd, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)

    # resolved config 落盘：同样在 manifest 写盘之前完成，失败不留半成品。
    run_manifest["resolved_config"] = maybe_dump_resolved_config(args, command, environment, output)
    _ensure_no_secrets(run_manifest)

    if run_manifest["stage"] == "resume":
        # 续训不覆盖首次启动的 run manifest；追加一份续训记录。
        existing = sorted(output.glob("run_manifest.resume.*.json"))
        run_manifest_path = output / f"run_manifest.resume.{len(existing) + 1}.json"
    else:
        run_manifest_path = output / RUN_MANIFEST_FILE
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"GRPO run manifest written: {run_manifest_path}")

    status = subprocess.call(command, cwd=ROOT, env=environment)
    # post-run：扫描 output 下的 checkpoint 产物并写 checkpoint_manifest.json。
    checkpoint_manifest = build_checkpoint_manifest(output, exit_code=status)
    checkpoint_manifest_path = output / CHECKPOINT_MANIFEST_FILE
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"GRPO checkpoint manifest written: {checkpoint_manifest_path}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
