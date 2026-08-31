#!/usr/bin/env python3
"""验收 merged checkpoint：文件、hash、BF16 reload、短前向、无 adapter 残留。

用法：
    PYTHONPATH=src python scripts/verify_merged_checkpoint.py \
      --merged-dir outputs/models/process-sft-merged [--device auto]

验收结果写回 ``merge_manifest.json`` 的 ``verification`` 块；
全部通过返回 0，任何一项失败返回 1。
"""

from __future__ import annotations

import argparse
import json
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.sft.reload_check import (  # noqa: E402
    check_tokenizer_identity,
    default_reload_loaders,
    resolve_device,
    tokenizer_artifact_hashes,
)
from shopping_grpo.training.sft.run_manifest import (  # noqa: E402
    FROZEN_MODEL_REVISION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

MANIFEST_FILE = "merge_manifest.json"
ADAPTER_ONLY_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", type=Path, required=True, help="merge 输出目录")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help=argparse.SUPPRESS,  # 仅供离线环境诊断；正式验收不允许跳过前向。
    )
    return parser.parse_args()


def check_required_files(merged_dir: Path) -> dict:
    required = ["config.json", MANIFEST_FILE, "tokenizer_config.json"]
    has_shards = (merged_dir / "model.safetensors.index.json").is_file() or any(
        path.suffix in WEIGHT_SUFFIXES for path in merged_dir.glob("model*.safetensors")
    )
    if not has_shards:
        required.append("model shards (index 或分片)")
    missing = [name for name in required if not (merged_dir / name).is_file()]
    if not (merged_dir / "model.safetensors.index.json").is_file():
        has_single = any(path.suffix in WEIGHT_SUFFIXES for path in merged_dir.glob("model*.safetensors"))
        if not has_single:
            pass  # 已在 has_shards 记录
    return {
        "name": "required_files_present",
        "passed": not missing,
        "detail": {"missing": missing, "has_weight_shards": has_shards},
    }


def check_no_adapter_files(merged_dir: Path) -> dict:
    leftovers = [name for name in ADAPTER_ONLY_FILES if (merged_dir / name).is_file()]
    return {
        "name": "no_adapter_only_files",
        "passed": not leftovers,
        "detail": {"adapter_only_files_found": leftovers},
    }


def recompute_output_hashes(merged_dir: Path) -> dict:
    """对 merge 产物（除 manifest 自身）重新计算 canonical hash。"""
    files = {}
    for path in sorted(merged_dir.iterdir()):
        if not path.is_file() or path.name == MANIFEST_FILE:
            continue
        files[path.name] = sha256_file(path)
    return files


def check_manifest_hashes(merged_dir: Path, manifest: dict) -> dict:
    recorded = manifest.get("output_files_sha256")
    if not recorded:
        return {
            "name": "merge_manifest_output_hashes",
            "passed": False,
            "detail": {"reason": "merge_manifest.json 缺少 output_files_sha256；请用新版 merge 脚本重新合并"},
        }
    live = recompute_output_hashes(merged_dir)
    missing = sorted(set(recorded) - set(live))
    extra = sorted(set(live) - set(recorded))
    changed = sorted(name for name in set(recorded) & set(live) if recorded[name] != live[name])
    passed = not missing and not extra and not changed
    return {
        "name": "merge_manifest_output_hashes",
        "passed": passed,
        "detail": {"missing": missing, "unexpected": extra, "changed": changed},
    }


def check_manifest_contract(manifest: dict) -> dict:
    problems = []
    source = manifest.get("source") or {}
    merge = manifest.get("merge") or {}
    if not source.get("base_model"):
        problems.append("缺少 source.base_model")
    if not source.get("adapter"):
        problems.append("缺少 source.adapter")
    if not str(manifest.get("output", "")):
        problems.append("缺少 output 路径")
    if merge.get("dtype") is None:
        problems.append("缺少 merge.dtype")
    if merge.get("max_shard_size") is None:
        problems.append("缺少 merge.max_shard_size")
    if merge.get("base_revision") != FROZEN_MODEL_REVISION:
        problems.append(
            f"merge.base_revision 必须是冻结值 {FROZEN_MODEL_REVISION!r}"
        )
    resolved_revision = merge.get("resolved_revision")
    if resolved_revision and resolved_revision != FROZEN_MODEL_REVISION:
        problems.append(
            f"merge.resolved_revision {resolved_revision!r} 与冻结 revision 不一致"
        )
    if merge.get("source_run_id") is None:
        problems.append("缺少 merge.source_run_id（adapter 未携带 run manifest 或未记录）")
    if not source.get("inputs_sha256", {}).get("adapter"):
        problems.append("缺少 source.inputs_sha256.adapter（adapter 输入文件 hash）")
    return {
        "name": "merge_manifest_contract_fields",
        "passed": not problems,
        "detail": {"problems": problems},
    }


def _hash_dir_matching(directory: str, recorded: dict, *, allowed_unhashed=()) -> dict:
    """对 recorded 中列出的文件重算 SHA256；目录缺失或文件变化都会体现在结果里。"""
    root = Path(directory)
    if not root.is_dir():
        return {"verified": False, "reason": f"输入目录不存在：{directory}"}
    missing = sorted(name for name in recorded if not (root / name).is_file())
    # The recorded mapping is a closed file set.  Adapter run manifests are
    # intentionally excluded from the merge input hash, so that one file is an
    # explicit, auditable whitelist rather than an unnoticed escape hatch.
    allowed_unhashed = set(allowed_unhashed)
    live_names = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(live_names - set(recorded) - allowed_unhashed)
    changed = sorted(
        name
        for name in recorded
        if (root / name).is_file() and sha256_file(root / name) != recorded[name]
    )
    return {
        "verified": not missing and not changed and not unexpected,
        "missing": missing,
        "unexpected": unexpected,
        "allowed_unhashed": sorted(allowed_unhashed & live_names),
        "changed": changed,
    }


def check_input_hashes(manifest: dict) -> dict:
    """强制核对 base / adapter 输入 hash：merge 后输入被改动即失败。"""
    inputs = (manifest.get("source") or {}).get("inputs_sha256") or {}
    base_model = (manifest.get("source") or {}).get("base_model")
    adapter = (manifest.get("source") or {}).get("adapter")
    problems = []
    detail = {}
    for label, directory, recorded in (
        ("base", base_model, inputs.get("base")),
        ("adapter", adapter, inputs.get("adapter")),
    ):
        if not recorded:
            problems.append(f"缺少 inputs_sha256.{label}")
            continue
        result = _hash_dir_matching(
            directory,
            recorded,
            allowed_unhashed=("run_manifest.json",) if label == "adapter" else (),
        )
        detail[label] = result
        if not result.get("verified"):
            problems.append(f"{label} 输入文件缺失或已被修改：{result}")
    return {
        "name": "input_hashes_verified",
        "passed": not problems,
        "detail": {"problems": problems, **detail},
    }


def _call_loader(loader, *args, revision=None):
    if revision is None:
        return loader(*args)
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(*args, revision=revision)
    accepts = "revision" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return loader(*args, revision=revision) if accepts else loader(*args)


def check_dtype_config(merged_dir: Path, manifest: dict) -> dict:
    config = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    recorded_dtype = manifest.get("merge", {}).get("dtype")
    # Transformers 5 serializes this as ``dtype``; Transformers 4 used
    # ``torch_dtype``.  They represent the same checkpoint contract.
    config_dtype = config.get("dtype", config.get("torch_dtype"))
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "torch.float16": "float16",
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
    }
    expected = aliases.get(str(recorded_dtype), "")
    resolved = aliases.get(str(config_dtype), "")
    passed = bool(expected) and resolved == expected
    return {
        "name": "config_dtype_matches_manifest",
        "passed": passed,
        "detail": {
            "config_dtype": config_dtype,
            "config_field": "dtype" if "dtype" in config else "torch_dtype",
            "manifest_dtype": recorded_dtype,
            "normalized_dtype": resolved,
        },
    }


def verify_merged_dir(
    merged_dir: Path,
    *,
    device_request: str = "auto",
    skip_forward: bool = False,
    loaders=None,
) -> tuple[dict, bool]:
    merged_dir = Path(merged_dir)
    checks: list[dict] = []
    if loaders is None:
        loaders = default_reload_loaders()

    manifest = None
    if (merged_dir / MANIFEST_FILE).is_file():
        manifest = json.loads((merged_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest is None:
        checks.append(
            {
                "name": "required_files_present",
                "passed": False,
                "detail": {"missing": [MANIFEST_FILE]},
            }
        )
    else:
        checks.append(check_required_files(merged_dir))
        checks.append(check_no_adapter_files(merged_dir))
        checks.append(check_manifest_contract(manifest))
        checks.append(check_dtype_config(merged_dir, manifest))
        checks.append(check_manifest_hashes(merged_dir, manifest))
        checks.append(check_input_hashes(manifest))
        try:
            model = _call_loader(
                loaders["load_model"],
                merged_dir,
                manifest["merge"]["dtype"],
                revision=manifest["merge"].get("base_revision"),
            )
            tokenizer = _call_loader(
                loaders["load_tokenizer"],
                merged_dir,
                revision=manifest["merge"].get("base_revision") or FROZEN_MODEL_REVISION,
            )
            base_tokenizer = _call_loader(
                loaders["load_tokenizer"],
                manifest["source"]["base_model"],
                revision=manifest["merge"].get("base_revision") or FROZEN_MODEL_REVISION,
            )
            # Compare resolved semantics.  File hashes remain audit evidence;
            # save_pretrained normalization and Hub cache files are not identity.
            checks.append(
                check_tokenizer_identity(
                    merged_dir,
                    manifest.get("source", {}).get("base_model", ""),
                    output_tokenizer=tokenizer,
                    base_tokenizer=base_tokenizer,
                )
            )
            checks.append({"name": "bf16_reload", "passed": True, "detail": {"dtype": manifest["merge"]["dtype"]}})
            if skip_forward:
                checks.append(
                    {
                        "name": "short_forward_finite",
                        "passed": False,
                        "detail": {"reason": "--skip-forward 仅限离线诊断，正式验收不得使用"},
                    }
                )
            else:
                device = resolve_device(device_request)
                forward = loaders["short_forward"](model, tokenizer, device)
                checks.append(
                    {
                        "name": "short_forward_finite",
                        "passed": bool(forward.ok and forward.finite),
                        "detail": {
                            "device": forward.device,
                            "logits_shape": forward.logits_shape,
                            "finite": forward.finite,
                            "error": forward.error,
                        },
                    }
                )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "name": "bf16_reload",
                    "passed": False,
                    "detail": {"error": f"{exc.__class__.__name__}: {exc}"},
                }
            )
            checks.append(
                {
                    "name": "tokenizer_identity_matches_base",
                    "passed": False,
                    "detail": {
                        "reason": f"tokenizer/base reload 失败：{exc.__class__.__name__}: {exc}",
                        "output_artifact_sha256": tokenizer_artifact_hashes(merged_dir),
                    },
                }
            )

    all_passed = all(check["passed"] for check in checks)
    if manifest is not None:
        manifest["verification"] = {
            "passed": all_passed,
            "device": resolve_device(device_request),
            "checks": checks,
            "preprocessing_output_files_sha256": tokenizer_artifact_hashes(merged_dir),
            "verified_at_epoch_s": int(time.time()),
            "verifier_code_sha256": sha256_file(Path(__file__).resolve()),
            "manifest_output_digest": sha256_bytes(canonical_json_bytes(manifest.get("output", {}))),
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (merged_dir / MANIFEST_FILE).write_text(payload, encoding="utf-8")
    return manifest, all_passed


def main() -> int:
    args = parse_args()
    if not args.merged_dir.is_dir():
        print(f"ERROR: merged 目录不存在：{args.merged_dir}")
        return 2
    _, passed = verify_merged_dir(
        args.merged_dir,
        device_request=args.device,
        skip_forward=args.skip_forward,
    )
    status = "PASSED" if passed else "FAILED"
    print(f"merged checkpoint verification {status}: {args.merged_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
