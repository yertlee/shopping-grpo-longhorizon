#!/usr/bin/env python3
"""验收一个已完成的 LoRA SFT run：文件、target keys、reload、短前向、summary。

用法：
    PYTHONPATH=src python scripts/verify_sft_adapter.py \
      --run-dir outputs/models/process-sft-smoke [--device auto]

验收结果写回 ``run_manifest.json`` 的 ``result.adapter_reload``；
全部通过返回 0，任何一项失败返回 1。校验本身不产生环境副作用
（不做推理生成、不写模型目录以外的东西）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import inspect
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
    hash_weight_files,
    load_run_manifest,
    sha256_file,
    sha256_text,
    write_run_manifest,
    validate_model_revision,
    validate_run_id,
)

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_SUFFIXES = ("adapter_model.safetensors", "adapter_model.bin")
SUMMARY_FILE = "train_summary.json"
MANIFEST_FILE = "run_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="SFT 输出目录（含 adapter 与 manifest）")
    parser.add_argument(
        "--base-model",
        default=None,
        help="覆盖 manifest 记录的基座；默认使用 run_manifest.json 的 model.path_or_repo",
    )
    parser.add_argument("--revision", default=None, help="覆盖期望的模型 revision")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="短前向设备；auto 在无 CUDA 时退回 CPU",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help=argparse.SUPPRESS,  # 仅供离线环境诊断；正式验收不允许跳过前向。
    )
    return parser.parse_args()


def _load_adapter_config(run_dir: Path) -> dict:
    return json.loads((run_dir / ADAPTER_CONFIG).read_text(encoding="utf-8"))


def check_files(run_dir: Path) -> dict:
    required = [ADAPTER_CONFIG, SUMMARY_FILE, MANIFEST_FILE, "tokenizer_config.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    weights = [name for name in ADAPTER_WEIGHT_SUFFIXES if (run_dir / name).is_file()]
    if not weights:
        missing.append("adapter_model.safetensors|.bin")
    return {
        "name": "required_files_present",
        "passed": not missing,
        "detail": {"missing": missing, "adapter_weights": weights[0] if weights else None},
    }


def _normalize_model_path(value) -> str:
    """路径归一化：统一分隔符并去掉尾部斜杠；跨平台比较用。"""
    import os

    return os.path.normpath(str(value)).replace("\\", "/").rstrip("/")


def check_base_identity(adapter_config: dict, manifest: dict, base_override: str | None) -> dict:
    """严格身份：adapter 记录的基座路径必须与期望路径**完全一致**（归一化后）。

    basename 相同但路径不同视为失败——同机可能存在多个同名模型目录。
    """
    expected = base_override or (manifest or {}).get("model", {}).get("path_or_repo")
    recorded = adapter_config.get("base_model_name_or_path")
    if not expected or not recorded:
        return {
            "name": "base_model_identity_matches_manifest",
            "passed": False,
            "detail": {"adapter_records": recorded, "expected": expected, "reason": "缺少路径记录"},
        }
    expected_norm = _normalize_model_path(expected)
    recorded_norm = _normalize_model_path(recorded)
    passed = expected_norm == recorded_norm
    detail = {"adapter_records": recorded, "expected": expected, "normalized": [recorded_norm, expected_norm]}
    if not passed and Path(recorded_norm).name == Path(expected_norm).name:
        detail["reason"] = "basename 相同但路径不同；同机可能存在多个同名目录，请用确切路径重跑"
    return {
        "name": "base_model_identity_matches_manifest",
        "passed": passed,
        "detail": detail,
    }


def check_weights_hash(manifest: dict, base_override: str | None) -> dict:
    """实测基座权重目录 hash，与 manifest 记录比对；这是基座身份的最终证明。"""
    model_block = (manifest or {}).get("model") or {}
    expected_hash = model_block.get("weights_sha256")
    base_path = base_override or model_block.get("path_or_repo")
    if not expected_hash:
        return {
            "name": "weights_hash_matches_manifest",
            "passed": False,
            "detail": {"reason": "manifest 缺少 weights_sha256（旧版 trainer 产物），无法验证基座身份"},
        }
    if not base_path:
        return {
            "name": "weights_hash_matches_manifest",
            "passed": False,
            "detail": {"reason": "缺少基座路径，无法实测权重 hash"},
        }
    live = hash_weight_files(base_path)
    live_hash = live.get("weights_sha256")
    if live_hash is None:
        return {
            "name": "weights_hash_matches_manifest",
            "passed": False,
            "detail": {"reason": live.get("weights_sha256_note"), "note": live.get("weights_sha256_note")},
        }
    return {
        "name": "weights_hash_matches_manifest",
        "passed": live_hash == expected_hash,
        "detail": {"manifest": expected_hash, "live": live_hash},
    }


def check_revision(manifest: dict, revision_arg: str | None) -> dict:
    """Require the frozen revision; pairwise metadata consistency is not enough."""
    recorded = (manifest or {}).get("model", {}).get("revision")
    requested = revision_arg or recorded
    try:
        validate_model_revision(requested)
    except ValueError as exc:
        return {
            "name": "revision_consistency",
            "passed": False,
            "detail": {"manifest": recorded, "cli": revision_arg, "expected": FROZEN_MODEL_REVISION, "reason": str(exc)},
        }
    if recorded != requested:
        return {
            "name": "revision_consistency",
            "passed": False,
            "detail": {"manifest": recorded, "cli": revision_arg, "reason": "manifest and CLI revision differ"},
        }
    return {
        "name": "revision_consistency",
        "passed": True,
        "detail": {"manifest": recorded, "cli": revision_arg, "expected": FROZEN_MODEL_REVISION},
    }


def _call_loader(loader, *args, revision=None):
    """Pass revision to capable injected loaders while retaining old test fakes."""
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


def check_target_keys(adapter_config: dict, weight_keys: list[str]) -> dict:
    targets = list(adapter_config.get("target_modules") or [])
    lora_keys = [key for key in weight_keys if ".lora_A." in key or ".lora_B." in key]
    missing_targets = [
        target
        for target in targets
        if not any(target in key for key in lora_keys)
    ]
    return {
        "name": "lora_target_keys_non_empty",
        "passed": bool(lora_keys) and not missing_targets,
        "detail": {
            "target_modules": targets,
            "lora_key_count": len(lora_keys),
            "missing_targets": missing_targets,
        },
    }


def check_trainable_params(summary: dict, trainable_stats) -> dict:
    lora_block = (summary or {}).get("lora") or {}
    recorded = lora_block.get("trainable_parameters")
    detail = {"summary_records": lora_block}
    if recorded is None:
        return {
            "name": "trainable_params_match_summary",
            "passed": False,
            "detail": {**detail, "reason": "train_summary.json 缺少 lora 块；请用新版 trainer 重跑"},
        }
    live_trainable, live_total = trainable_stats()
    ratio_recorded = lora_block.get("trainable_ratio")
    ratio_live = (live_trainable / live_total) if live_total else None
    passed = live_trainable == recorded and (
        ratio_recorded is None or abs(ratio_live - ratio_recorded) < 1e-6
    )
    return {
        "name": "trainable_params_match_summary",
        "passed": passed,
        "detail": {
            **detail,
            "live": {"trainable": live_trainable, "total": live_total, "ratio": ratio_live},
        },
    }


def check_summary_sanity(summary: dict, manifest: dict) -> dict:
    problems = []
    train_loss = (summary or {}).get("train_loss")
    if not isinstance(train_loss, (int, float)) or train_loss != train_loss or train_loss in (float("inf"), float("-inf")):
        problems.append(f"train_loss 非有限数值: {train_loss!r}")
    metrics = (summary or {}).get("metrics") or {}
    for name in ("eval_loss", "eval_accuracy"):
        value = metrics.get(name)
        if isinstance(value, float) and value != value:
            problems.append(f"{name} 是 NaN")
    exit_code = (manifest or {}).get("execution", {}).get("exit_code")
    if exit_code != 0:
        problems.append(f"run_manifest execution.exit_code != 0: {exit_code!r}")
    return {
        "name": "summary_values_finite_and_successful",
        "passed": not problems,
        "detail": {"problems": problems},
    }


def verify_run_dir(
    run_dir: Path,
    *,
    base_override: str | None = None,
    revision: str | None = None,
    device_request: str = "auto",
    skip_forward: bool = False,
    loaders=None,
) -> tuple[dict, bool]:
    """执行全部检查并返回 (patched_manifest, all_passed)。"""
    run_dir = Path(run_dir)
    checks: list[dict] = []
    if loaders is None:
        loaders = default_reload_loaders()

    files = check_files(run_dir)
    checks.append(files)

    manifest = None
    if (run_dir / MANIFEST_FILE).is_file():
        manifest = load_run_manifest(run_dir / MANIFEST_FILE, verify_run_id=False)
    summary = None
    if (run_dir / SUMMARY_FILE).is_file():
        summary = json.loads((run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))

    adapter_config: dict = {}
    if (run_dir / ADAPTER_CONFIG).is_file():
        adapter_config = _load_adapter_config(run_dir)

    if manifest is not None:
        identity = validate_run_id(manifest)
        checks.append({
            "name": "run_id_matches_manifest",
            "passed": identity["passed"],
            "detail": identity,
        })

    if adapter_config and manifest is not None:
        checks.append(check_base_identity(adapter_config, manifest, base_override))
        checks.append(check_weights_hash(manifest, base_override))
        checks.append(check_revision(manifest, revision))

    if adapter_config and files["passed"]:
        try:
            weight_keys = loaders["adapter_weight_keys"](run_dir)
            checks.append(check_target_keys(adapter_config, weight_keys))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "name": "lora_target_keys_non_empty",
                    "passed": False,
                    "detail": {"error": f"{exc.__class__.__name__}: {exc}"},
                }
            )

    model = None
    if files["passed"]:
        base_model_path = base_override or (manifest or {}).get("model", {}).get("path_or_repo")
        try:
            model = _call_loader(
                loaders["load_model"], base_model_path, revision=revision or FROZEN_MODEL_REVISION
            )
            model = loaders["load_peft_adapter"](model, run_dir)
            checks.append(
                check_trainable_params(summary, lambda: loaders["trainable_stats"](model))
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "name": "adapter_reload",
                    "passed": False,
                    "detail": {"error": f"{exc.__class__.__name__}: {exc}"},
                }
            )
            checks.append(
                {
                    "name": "trainable_params_match_summary",
                    "passed": False,
                    "detail": {"reason": "adapter reload 失败，无法统计 trainable 参数"},
                }
            )

    tokenizer = None
    base_tokenizer = None
    if files["passed"]:
        try:
            # Keep the tokenizer/processor on the exact same frozen snapshot
            # as the model.  Omitting revision here silently resolves a moving
            # default branch and invalidates the reload evidence.
            tokenizer = _call_loader(
                loaders["load_tokenizer"],
                run_dir,
                revision=revision or FROZEN_MODEL_REVISION,
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "name": "tokenizer_or_processor_loadable",
                    "passed": False,
                    "detail": {"error": f"{exc.__class__.__name__}: {exc}"},
                }
            )
        else:
            checks.append({"name": "tokenizer_or_processor_loadable", "passed": True, "detail": {}})

    # Compare the semantic objects resolved by both loaders.  Serialized files
    # are retained as audit hashes, but save_pretrained normalization and Hub
    # cache metadata must not become an identity contract.
    if files["passed"] and manifest is not None:
        base_model_path = base_override or (manifest.get("model") or {}).get("path_or_repo")
        if base_model_path:
            try:
                base_tokenizer = _call_loader(
                    loaders["load_tokenizer"],
                    base_model_path,
                    revision=revision or FROZEN_MODEL_REVISION,
                )
            except Exception as exc:  # noqa: BLE001
                checks.append(
                    {
                        "name": "base_tokenizer_or_processor_loadable",
                        "passed": False,
                        "detail": {"error": f"{exc.__class__.__name__}: {exc}"},
                    }
                )
            checks.append(
                check_tokenizer_identity(
                    run_dir,
                    base_model_path,
                    output_tokenizer=tokenizer,
                    base_tokenizer=base_tokenizer,
                )
            )

    if summary is not None:
        checks.append(check_summary_sanity(summary, manifest))

    if skip_forward:
        checks.append(
            {
                "name": "short_forward_finite",
                "passed": False,
                "detail": {"reason": "--skip-forward 仅限离线诊断，正式验收不得使用"},
            }
        )
    elif model is not None and tokenizer is not None:
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

    all_passed = all(check["passed"] for check in checks)

    if manifest is not None:
        manifest["result"]["adapter_reload"] = {
            "passed": all_passed,
            "device": resolve_device(device_request),
            "revision": revision,
            "checks": checks,
            "preprocessing_output_files_sha256": tokenizer_artifact_hashes(run_dir),
            "verified_at_epoch_s": int(time.time()),
            "verifier_code_sha256": sha256_file(Path(__file__).resolve()),
        }
        write_run_manifest(run_dir / MANIFEST_FILE, manifest)
    return manifest, all_passed


def main() -> int:
    args = parse_args()
    if not args.run_dir.is_dir():
        print(f"ERROR: run 目录不存在：{args.run_dir}")
        return 2
    _, passed = verify_run_dir(
        args.run_dir,
        base_override=args.base_model,
        revision=args.revision,
        device_request=args.device,
        skip_forward=args.skip_forward,
    )
    status = "PASSED" if passed else "FAILED"
    print(f"adapter verification {status}: {args.run_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
