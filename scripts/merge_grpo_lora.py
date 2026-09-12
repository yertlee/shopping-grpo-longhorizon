#!/usr/bin/env python3
"""把 veRL/FSDP merger 导出的 GRPO LoRA checkpoint 合并成 standalone HF 模型。

veRL 的 ``model_merger`` 只把 LoRA 拆到 ``lora_adapter/``，base 权重仍是冻结的
actor 初始化（本项目的 M2）。因此必须显式做 tensor merge：

    W_merged = dtype_round(W_base + (B @ A) * lora_alpha / r)

并写出 ``merge_manifest.json``，使 ``verify_merged_checkpoint.py`` 能验收来源、
输入 hash、输出文件 hash、dtype 与冻结 revision。

用法：
    PYTHONPATH=src python scripts/merge_grpo_lora.py \
      --base-dir outputs/models/m3-step50-merged \
      --output outputs/models/m3-step50-final \
      --source-run-id grpo-pilot-100/global_step_50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.training.sft.run_manifest import (  # noqa: E402
    FROZEN_MODEL_REVISION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

MANIFEST_FILE = "merge_manifest.json"
BASE_WEIGHT_FILE = "model.safetensors"
ADAPTER_WEIGHT_FILE = "adapter_model.safetensors"
COPY_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def lora_pairs(adapter_sd: dict[str, torch.Tensor]) -> dict[str, dict[str, str]]:
    """把 adapter state dict 里成对的 lora_A / lora_B 聚合成 {stem: {A, B}}。"""
    pairs: dict[str, dict[str, str]] = {}
    for name in adapter_sd:
        if name.endswith(".lora_A.weight"):
            pairs.setdefault(name[: -len(".lora_A.weight")], {})["A"] = name
        elif name.endswith(".lora_B.weight"):
            pairs.setdefault(name[: -len(".lora_B.weight")], {})["B"] = name
    return {stem: keys for stem, keys in pairs.items() if "A" in keys and "B" in keys}


def hf_weight_key(stem: str) -> str:
    """adapter stem -> 输出 HF 权重 key（去掉 PEFT 前缀，补回 .weight）。"""
    return stem.removeprefix("base_model.model.") + ".weight"


def merge_state_dicts(
    base: dict[str, torch.Tensor],
    adapter: dict[str, torch.Tensor],
    *,
    scale: float,
    dtype: torch.dtype,
) -> dict[str, object]:
    """原地把 LoRA 增量合进 base，返回统计。缺失配对或非有限 delta 直接失败。"""
    pairs = lora_pairs(adapter)
    if not pairs:
        raise SystemExit("adapter 中没有成对的 lora_A/lora_B 权重")
    applied = 0
    missing: list[str] = []
    vision_max_abs = 0.0
    for stem, keys in pairs.items():
        target = hf_weight_key(stem)
        if target not in base:
            missing.append(target)
            continue
        delta = (adapter[keys["B"]].float() @ adapter[keys["A"]].float()) * scale
        if not torch.isfinite(delta).all():
            raise SystemExit(f"non-finite LoRA delta: {target}")
        if ".visual." in target:
            vision_max_abs = max(vision_max_abs, float(delta.abs().max().item()))
        weight = base[target]
        base[target] = (weight.float() + delta).to(dtype)
        applied += 1
    if missing:
        raise SystemExit(f"adapter 目标权重在 base 中缺失（{len(missing)} 个）：{missing[:5]}")
    return {
        "lora_pairs": len(pairs),
        "applied_pairs": applied,
        "missing_pairs": len(missing),
        "vision_delta_max_abs": vision_max_abs,
        "vision_delta_all_zero": vision_max_abs == 0.0,
    }


def _sha256_map(directory: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in exclude
    }


def _normalize_config_dtype(config_path: Path, dtype_name: str) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("torch_dtype", None)
    config["dtype"] = dtype_name
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="veRL/FSDP merger 的原始输出目录（含 base 权重与 lora_adapter/）")
    parser.add_argument("--adapter-dir", type=Path,
                        help="LoRA adapter 目录；缺省为 <base-dir>/lora_adapter")
    parser.add_argument("--output", type=Path, required=True, help="standalone HF 输出目录")
    parser.add_argument("--source-run-id", required=True,
                        help="产出该 adapter 的训练 run 身份（写入 merge_manifest）")
    parser.add_argument("--base-revision", default=FROZEN_MODEL_REVISION)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_MAP))
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.base_dir.expanduser().resolve()
    adapter_dir = (args.adapter_dir or (base_dir / "lora_adapter")).expanduser().resolve()
    output = args.output.expanduser().resolve()
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not (base_dir / BASE_WEIGHT_FILE).is_file():
        raise SystemExit(f"base 权重缺失：{base_dir / BASE_WEIGHT_FILE}")
    if not (adapter_dir / ADAPTER_WEIGHT_FILE).is_file() or not adapter_config_path.is_file():
        raise SystemExit(f"adapter 权重/config 缺失：{adapter_dir}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{output}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    rank = int(adapter_config["r"])
    alpha = float(adapter_config.get("lora_alpha") or 0.0)
    if rank <= 0 or alpha <= 0:
        raise SystemExit(f"非法 LoRA r/alpha：r={rank}, alpha={alpha}")
    scale = alpha / rank
    dtype = DTYPE_MAP[args.dtype]

    print(f"loading base={base_dir / BASE_WEIGHT_FILE}", flush=True)
    base = load_file(str(base_dir / BASE_WEIGHT_FILE))
    adapter = load_file(str(adapter_dir / ADAPTER_WEIGHT_FILE))
    stats = merge_state_dicts(base, adapter, scale=scale, dtype=dtype)
    print(f"merged {stats}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    save_file(base, str(output / BASE_WEIGHT_FILE), metadata={"format": "pt"})
    for name in COPY_FILES:
        source = base_dir / name
        if source.is_file():
            shutil.copy2(source, output / name)
    _normalize_config_dtype(output / "config.json", args.dtype)

    manifest = {
        "schema_version": "shopping-grpo-lora-merge-manifest-v1",
        "output": str(output),
        "source": {
            "base_model": str(base_dir),
            "adapter": str(adapter_dir),
            "inputs_sha256": {
                "base": _sha256_map(base_dir),
                "adapter": _sha256_map(adapter_dir),
            },
        },
        "merge": {
            "dtype": args.dtype,
            "max_shard_size": args.max_shard_size,
            "base_revision": args.base_revision,
            "resolved_revision": None,
            "source_run_id": args.source_run_id,
        },
        "lora": {
            "r": rank,
            "lora_alpha": alpha,
            "scale": scale,
            **stats,
        },
        "output_files_sha256": _sha256_map(output, exclude=(MANIFEST_FILE,)),
    }
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest["lora"]))
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"merge_manifest written: {output / MANIFEST_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
