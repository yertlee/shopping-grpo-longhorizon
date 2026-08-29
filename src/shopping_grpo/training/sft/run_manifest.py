"""SFT run manifest 构建与校验。

一次 LoRA SFT run 的输入、配方、代码、执行和结果都被记录进
``run_manifest.json``；``run_id`` 是对 run 配置（模型、数据、配方、代码 hash）
的规范 JSON 做内容寻址，输入不变则 run_id 不变。执行与结果字段允许事后
finalize / 由 verifier 回填，但不参与 run_id 计算。

禁止把 secret（API key、token、密码、env 值）写进 manifest；
环境变量只允许记录名字。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = "shopping-sft-run-manifest-v1"
CHECKPOINT_OWNER_SCHEMA_VERSION = "shopping-sft-checkpoint-owner-v1"
CHECKPOINT_OWNER_FILE = "checkpoint_owner.json"
# The project deliberately pins the base model identity.  Keeping this in the
# shared manifest module prevents each CLI/verifier from silently accepting a
# different (or merely self-consistent) revision string.
FROZEN_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"

# manifest 里禁止出现的键名片段（大小写不敏感）；环境变量只允许记录名字。
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "bearer",
    "authorization",
    "credential",
)

# manifest 顶层允许出现的键；防止把任意负载塞进 manifest。
_ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "stage",
        "model",
        "data",
        "recipe",
        "runtime",
        "execution",
        "result",
    }
)


class ManifestSecretError(ValueError):
    """manifest 中出现了禁止的 secret 键。"""


def canonical_json_bytes(value) -> bytes:
    """规范 JSON 序列化：键排序、紧凑分隔符；作为所有 hash 的输入。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _checkpoint_file_inventory(checkpoint_dir: str | Path) -> list[dict]:
    """Return the canonical recursive file set for a checkpoint.

    The ownership sidecar is deliberately excluded from its own inventory.
    Every other regular file is bound by relative POSIX path, byte size and
    SHA256, so copying, deleting, adding, or editing checkpoint contents is
    detected before resume.
    """
    root = Path(checkpoint_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint is not a directory: {checkpoint_dir}")
    files = []
    owner_path = root / CHECKPOINT_OWNER_FILE
    for path in root.rglob("*"):
        if not path.is_file() or path == owner_path:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(files, key=lambda item: item["path"])


def _checkpoint_owner_payload(
    *, run_id: str, checkpoint_path: str, global_step: int, files: list[dict]
) -> dict:
    """Build the unsigned portion of a checkpoint ownership sidecar.

    The sidecar is intentionally self-contained: its hash detects accidental
    edits/copies, while ownership is established by matching the immutable
    run_id in the signed-by-content run manifest and the resolved path below.
    No secret signing key is involved in this contract.
    """
    return {
        "schema_version": CHECKPOINT_OWNER_SCHEMA_VERSION,
        "run_id": str(run_id),
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "global_step": int(global_step),
        "files": files,
        "files_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def build_checkpoint_owner(
    *,
    run_id: str,
    checkpoint_path: str,
    global_step: int,
    files: list[dict] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Return a checkpoint owner sidecar with a canonical content hash."""
    if files is None:
        if checkpoint_dir is None:
            raise ValueError("checkpoint file inventory is required to build owner sidecar")
        files = _checkpoint_file_inventory(checkpoint_dir)
    payload = _checkpoint_owner_payload(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        files=files,
    )
    payload["self_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_checkpoint_owner(
    owner: dict,
    *,
    run_id: str,
    checkpoint_path: str,
    global_step: int,
    checkpoint_dir: str | Path,
) -> None:
    """Validate sidecar identity and exact recursive checkpoint file inventory."""
    if not isinstance(owner, dict):
        raise ValueError("checkpoint owner sidecar must be a JSON object")
    actual_files = _checkpoint_file_inventory(checkpoint_dir)
    expected = _checkpoint_owner_payload(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        global_step=global_step,
        files=actual_files,
    )
    if set(owner) != set(expected) | {"self_sha256"}:
        raise ValueError("checkpoint owner sidecar contains unexpected fields")
    if owner.get("schema_version") != CHECKPOINT_OWNER_SCHEMA_VERSION:
        raise ValueError("checkpoint owner sidecar schema mismatch")
    if owner.get("self_sha256") != sha256_bytes(canonical_json_bytes(expected)):
        raise ValueError("checkpoint owner sidecar self-hash mismatch")
    for key, value in expected.items():
        if owner.get(key) != value:
            raise ValueError(
                f"checkpoint owner sidecar {key} mismatch: "
                f"recorded={owner.get(key)!r}, expected={value!r}"
            )


def write_checkpoint_owner(
    checkpoint_dir: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    global_step: int,
) -> Path:
    """Atomically persist the ownership sidecar after Trainer saves a checkpoint."""
    checkpoint = Path(checkpoint_dir).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    relative = checkpoint.relative_to(output).as_posix()
    if not relative or relative == ".":
        raise ValueError("checkpoint must be a child of the training output directory")
    files = _checkpoint_file_inventory(checkpoint)
    owner = build_checkpoint_owner(
        run_id=run_id,
        checkpoint_path=relative,
        global_step=global_step,
        files=files,
    )
    target = checkpoint / CHECKPOINT_OWNER_FILE
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(checkpoint), delete=False, suffix=".tmp"
    )
    try:
        with handle:
            json.dump(owner, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def sha256_file(path: str | Path) -> str:
    """流式计算大文件（多 GB 权重、JSONL）的 SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_weight_files(model_path: str | Path) -> dict:
    """对本地模型目录中的权重分片做 SHA256；远程 repo 返回 None + 原因。"""
    root = Path(model_path)
    if not root.is_dir():
        return {
            "weights_sha256": None,
            "weight_files": None,
            "weights_sha256_note": "model path is not a local directory; see model.revision",
        }
    weight_files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt"}
    )
    if not weight_files:
        return {
            "weights_sha256": None,
            "weight_files": [],
            "weights_sha256_note": "no local weight shard found in model directory",
        }
    files = {path.name: sha256_file(path) for path in weight_files}
    return {
        "weights_sha256": sha256_bytes(canonical_json_bytes(files)),
        "weight_files": files,
        "weights_sha256_note": "sha256 of canonical {filename: sha256} mapping",
    }


def ensure_no_secrets(value, _key_path: str = "") -> None:
    """递归拒绝 secret 键；值不会被读取或记录。"""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in _SECRET_KEY_FRAGMENTS):
                raise ManifestSecretError(
                    f"manifest 禁止包含 secret 键：{_key_path}/{key}"
                )
            ensure_no_secrets(item, f"{_key_path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            ensure_no_secrets(item, f"{_key_path}[{index}]")


def compute_run_id(manifest: dict) -> str:
    """run_id 只由 run 配置决定；execution / result 不参与。

    ``data.train_examples`` / ``data.validation_examples`` 是运行时的产物
    （finalize 阶段回填），必须从身份计算中剔除，否则 finalize 会改变 run_id。
    """
    import copy

    body = {
        key: copy.deepcopy(manifest[key])
        for key in ("schema_version", "stage", "model", "data", "recipe", "runtime")
        if key in manifest
    }
    data = body.get("data")
    if isinstance(data, dict):
        data.pop("train_examples", None)
        data.pop("validation_examples", None)
    return sha256_bytes(canonical_json_bytes(body))


def validate_run_id(manifest: dict, *, allow_legacy_incomplete: bool = True) -> dict:
    """Recompute and validate the content-addressed run identity.

    Older hand-built fixtures did not contain the data/recipe identity block;
    those are retained for read-only compatibility, but every manifest emitted
    by the current trainer is complete and is checked strictly.
    """
    expected = compute_run_id(manifest)
    actual = manifest.get("run_id")
    incomplete = not (manifest.get("data") or {}).get("train_sha256") or not manifest.get("recipe")
    passed = actual == expected
    if incomplete and allow_legacy_incomplete:
        return {
            "passed": True,
            "skipped": True,
            "reason": "legacy incomplete manifest; run_id cannot be independently recomputed",
            "actual": actual,
            "expected": expected,
        }
    return {"passed": passed, "skipped": False, "actual": actual, "expected": expected}


def validate_model_revision(revision: str | None, *, expected: str = FROZEN_MODEL_REVISION) -> str:
    """Require the project's frozen model revision, never just pairwise equality."""
    if revision != expected:
        raise ValueError(
            f"model revision must equal frozen revision {expected}, got {revision!r}"
        )
    return expected


def build_run_manifest(
    *,
    stage: str,
    model_path: str,
    model_revision: str | None,
    weights: dict,
    train_path: str | Path,
    validation_path: str | Path | None,
    sft_ready_manifest_path: str | Path | None,
    preflight_report_path: str | Path | None,
    recipe: dict,
    code_hashes: dict,
    dependency_versions: dict,
    gpu: dict | None,
    command: list[str],
    resume_from_checkpoint: str | None,
) -> dict:
    """构建 run 配置部分；execution / result 由调用方在运行后回填。"""

    validate_model_revision(model_revision)

    def file_hash(path: str | Path | None) -> str | None:
        if path is None:
            return None
        return sha256_file(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": None,
        "stage": stage,
        "model": {
            "path_or_repo": str(model_path),
            "revision": model_revision,
            "weights_sha256": weights.get("weights_sha256"),
            "weight_files": weights.get("weight_files"),
            "weights_sha256_note": weights.get("weights_sha256_note"),
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": file_hash(train_path),
            "validation_path": str(validation_path) if validation_path else None,
            "validation_sha256": file_hash(validation_path),
            "sft_ready_manifest_sha256": file_hash(sft_ready_manifest_path),
            "preflight_report_sha256": file_hash(preflight_report_path),
            "train_examples": None,
            "validation_examples": None,
        },
        "recipe": dict(recipe),
        "runtime": {
            "code_hashes": dict(code_hashes),
            "dependency_versions": dict(dependency_versions),
            "gpu": gpu,
        },
        "execution": {
            "command": list(command),
            "resume_from_checkpoint": resume_from_checkpoint,
            "checkpoint_paths": None,
            "exit_code": None,
            "error": None,
        },
        "result": {
            "train_loss": None,
            "eval_loss": None,
            "peak_gpu_memory_gib": None,
            "adapter_reload": {
                "passed": None,
                "note": "run scripts/verify_sft_adapter.py --run-dir <output> to fill this field",
            },
        },
    }
    ensure_no_secrets(manifest)
    manifest["run_id"] = compute_run_id(manifest)
    return manifest


def finalize_run_manifest(
    manifest: dict,
    *,
    train_examples: int,
    validation_examples: int,
    train_loss: float | None,
    eval_loss: float | None,
    peak_gpu_memory_gib: float | None,
    checkpoint_paths: list[str],
    exit_code: int,
    error: str | None = None,
) -> dict:
    """用实际运行结果回填 manifest；未知值必须是显式 None，不允许估计值。"""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema_version')!r}")
    if manifest["execution"]["exit_code"] is not None:
        raise ValueError("run manifest 已 finalize，拒绝重复覆盖")
    manifest["data"]["train_examples"] = train_examples
    manifest["data"]["validation_examples"] = validation_examples
    manifest["execution"]["checkpoint_paths"] = list(checkpoint_paths)
    manifest["execution"]["exit_code"] = int(exit_code)
    manifest["execution"]["error"] = error
    manifest["result"]["train_loss"] = train_loss
    manifest["result"]["eval_loss"] = eval_loss
    manifest["result"]["peak_gpu_memory_gib"] = peak_gpu_memory_gib
    ensure_no_secrets(manifest)
    return manifest


def validate_top_level_keys(manifest: dict) -> None:
    unknown = set(manifest) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"manifest 出现未知顶层键：{sorted(unknown)}")


def write_run_manifest(path: str | Path, manifest: dict) -> None:
    """原子写入；先做 secret 与 schema 校验，失败时不产生半截文件。"""
    ensure_no_secrets(manifest)
    validate_top_level_keys(manifest)
    target = Path(path)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(target.parent), delete=False, suffix=".tmp"
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def load_run_manifest(path: str | Path, *, verify_run_id: bool = True) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_top_level_keys(manifest)
    ensure_no_secrets(manifest)
    if verify_run_id:
        identity = validate_run_id(manifest)
        if not identity["passed"]:
            raise ValueError(
                "run manifest run_id mismatch: "
                f"recorded={identity['actual']!r}, recomputed={identity['expected']!r}"
            )
    return manifest
