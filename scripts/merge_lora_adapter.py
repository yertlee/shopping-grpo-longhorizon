#!/usr/bin/env python3
"""把完成 SFT 的 LoRA adapter 合并为 GRPO 的独立起点。"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.sft.run_manifest import (  # noqa: E402
    FROZEN_MODEL_REVISION,
    load_run_manifest,
    sha256_file,
    validate_model_revision,
    validate_run_id,
)


def choose_model_class(config, causal_model_class, multimodal_model_class):
    """Qwen3.5 走官方多模态类；其他 CausalLM 保持普通路径。"""
    return multimodal_model_class if str(getattr(config, "model_type", "")).startswith("qwen3_5") else causal_model_class


def _hash_dir_files(directory: Path, exclude: set[str] | None = None) -> dict | None:
    """对目录内文件做 {filename: sha256}；目录不存在返回 None。"""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    exclude = exclude or set()
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name not in exclude
    )
    return {path.name: sha256_file(path) for path in files}


def _read_source_run_id(adapter_path: Path) -> str | None:
    manifest_path = Path(adapter_path) / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = load_run_manifest(manifest_path)
        identity = validate_run_id(manifest)
        if not identity["passed"]:
            return None
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    run_id = manifest.get("run_id")
    return str(run_id) if run_id else None


def _dependency_versions() -> dict:
    result = {}
    for package in ("torch", "transformers", "peft", "safetensors"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _resolved_revision(config):
    for name in ("_commit_hash", "commit_hash", "revision"):
        value = getattr(config, name, None)
        if value:
            return str(value)
    return None


def build_merge_manifest(
    base_model,
    adapter_path,
    output_path,
    model_type,
    *,
    merge=None,
    inputs_sha256=None,
    output_files_sha256=None,
):
    """输出可审计清单；GRPO 必须新挂 adapter，不能覆盖这份 checkpoint。

    旧调用（4 个位置参数）保持兼容；扩展字段通过关键字参数传入。
    """
    manifest = {
        "operation": "peft_merge_and_unload",
        "source": {"base_model": str(base_model), "adapter": str(adapter_path), "model_type": str(model_type)},
        "output": str(output_path),
        "next_step": "load this standalone checkpoint as GRPO base and attach a new LoRA adapter",
    }
    if merge is not None:
        manifest["merge"] = dict(merge)
    if inputs_sha256 is not None:
        manifest["source"]["inputs_sha256"] = inputs_sha256
    if output_files_sha256 is not None:
        manifest["output_files_sha256"] = dict(output_files_sha256)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="合并 LoRA SFT adapter，为 GRPO 创建独立 BF16 起点")
    parser.add_argument("--base-model", required=True, help="与 SFT 完全一致的原始模型路径或 Hugging Face 名称")
    parser.add_argument("--adapter", type=Path, required=True, help="SFT LoRA adapter 目录")
    parser.add_argument("--output", type=Path, required=True, help="新的 merged checkpoint 目录，必须为空")
    parser.add_argument("--bf16", action="store_true", help="以 bf16 合并；4090/RTX PRO 6000 建议开启")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--revision",
        default=None,
        help="基座模型 revision（如 HF refs/main）；写入 merge manifest 供 GRPO preflight 校验",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="合并后在进程内立即执行 reload + 短前向验收（等价于随后运行 verify_merged_checkpoint.py）",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="验收前向设备")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        validate_model_revision(args.revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{args.output}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("缺少合并依赖；请执行：uv sync --extra sft") from exc

    dtype_name = "bfloat16" if args.bf16 else "float32"
    load_kwargs = {"trust_remote_code": True, "revision": args.revision}
    config = AutoConfig.from_pretrained(args.base_model, **load_kwargs)
    resolved_revision = _resolved_revision(config)
    if resolved_revision and resolved_revision != args.revision:
        raise SystemExit(
            f"resolved model revision {resolved_revision!r} does not match frozen revision {args.revision!r}"
        )
    model_class = choose_model_class(config, AutoModelForCausalLM, AutoModelForMultimodalLM)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"加载 base={args.base_model} model_type={config.model_type} dtype={dtype}")
    base = model_class.from_pretrained(args.base_model, torch_dtype=dtype, **load_kwargs)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True, max_shard_size=args.max_shard_size)
    processor = AutoProcessor.from_pretrained(args.base_model, **load_kwargs)
    processor_revision = _resolved_revision(getattr(processor, "config", processor))
    if processor_revision and processor_revision != args.revision:
        raise SystemExit(
            f"resolved processor revision {processor_revision!r} does not match frozen revision {args.revision!r}"
        )
    processor.save_pretrained(str(args.output))
    del merged, base

    manifest = build_merge_manifest(
        args.base_model,
        args.adapter,
        args.output,
        config.model_type,
        merge={
            "dtype": dtype_name,
            "max_shard_size": args.max_shard_size,
            "base_revision": args.revision,
            "resolved_revision": resolved_revision,
            "source_run_id": _read_source_run_id(args.adapter),
            "dependency_versions": _dependency_versions(),
        },
        inputs_sha256={
            "base": _hash_dir_files(args.base_model),
            "adapter": _hash_dir_files(args.adapter, exclude={"run_manifest.json"}),
        },
        output_files_sha256=_hash_dir_files(args.output, exclude={"merge_manifest.json"}),
    )
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))

    if args.verify:
        from shopping_grpo.training.sft.reload_check import default_reload_loaders

        try:
            from scripts.verify_merged_checkpoint import verify_merged_dir
        except ImportError:  # 直接以文件方式执行本脚本时
            from verify_merged_checkpoint import verify_merged_dir  # type: ignore[no-redef]

        _, passed = verify_merged_dir(
            args.output,
            device_request=args.device,
            loaders=default_reload_loaders(),
        )
        status = "PASSED" if passed else "FAILED"
        print(f"merged checkpoint verification {status}: {args.output}")
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
