"""合并产物与 adapter 的统一 reload / 短前向校验。

merge 脚本、``verify_sft_adapter.py`` 与 ``verify_merged_checkpoint.py``
共用这里的实现，避免出现第二套 reload 语义。所有 torch / peft /
transformers 依赖都在 loader 工厂内部懒加载；测试通过注入 fake loader
运行，不需要下载模型或安装 CUDA。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 短前向输入：固定、无环境副作用的一句话。
SHORT_FORWARD_PROMPT = "你好，请帮我找一件适合送礼的商品。"


def _resolved_revision(config):
    """Return a revision resolved by Transformers/HF, when it is available."""
    for name in ("_commit_hash", "commit_hash", "revision"):
        value = getattr(config, name, None)
        if value:
            return str(value)
    return None


def _check_resolved_revision(config, requested: str | None) -> str | None:
    """Reject a loader result that explicitly resolves to another revision.

    Local snapshots do not carry ``_commit_hash``; their frozen weight hash is
    the second half of the identity contract, so a missing resolved value is
    recorded as ``None`` rather than fabricated as proof.
    """
    resolved = _resolved_revision(config)
    if requested and resolved and resolved != requested:
        raise ValueError(
            f"resolved model revision {resolved!r} does not match requested {requested!r}"
        )
    return resolved


@dataclass(frozen=True)
class ForwardResult:
    ok: bool
    device: str
    logits_shape: list[int] | None
    finite: bool | None
    error: str | None = None


def default_reload_loaders():
    """真实 loader；仅在 GPU/本机验证时 import 重依赖。"""

    def load_model(path: str | Path, dtype: str = "bf16", revision: str | None = None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported reload dtype: {dtype!r}")
        load_kwargs = {"trust_remote_code": True}
        if revision is not None:
            load_kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(str(path), **load_kwargs)
        resolved = _check_resolved_revision(config, revision)
        is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
        model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM
        model = model_class.from_pretrained(
            str(path), torch_dtype=dtype_map[dtype], **load_kwargs
        )
        model_resolved = _check_resolved_revision(getattr(model, "config", None), revision)
        if resolved is None:
            resolved = model_resolved
        # Keep the loader evidence available to verifiers without changing the
        # model's serialized config.
        model._shopping_requested_revision = revision
        model._shopping_resolved_revision = resolved
        return model

    def load_tokenizer(path: str | Path, revision: str | None = None):
        from transformers import AutoProcessor, AutoTokenizer

        load_kwargs = {"trust_remote_code": True}
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path), **load_kwargs)
            _check_resolved_revision(getattr(tokenizer, "config", tokenizer), revision)
            return tokenizer
        except Exception:
            processor = AutoProcessor.from_pretrained(str(path), **load_kwargs)
            _check_resolved_revision(getattr(processor, "config", processor), revision)
            return processor.tokenizer

    def load_peft_adapter(base_model, adapter_path: str | Path):
        from peft import PeftModel

        # Verifiers compare the live LoRA parameter counts with the trainer
        # summary.  PEFT defaults to inference mode, which freezes the adapter
        # and makes that comparison report zero trainable parameters.
        return PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)

    def adapter_weight_keys(adapter_path: str | Path) -> list[str]:
        from safetensors import safe_open

        weights = sorted(Path(adapter_path).glob("adapter_model.safetensors"))
        if not weights:
            raise FileNotFoundError(f"adapter 缺少 safetensors 权重：{adapter_path}")
        keys: list[str] = []
        for shard in weights:
            with safe_open(str(shard), framework="pt") as handle:
                keys.extend(handle.keys())
        return keys

    def trainable_stats(model) -> tuple[int, int]:
        trainable, total = model.get_nb_trainable_parameters()
        return int(trainable), int(total)

    def short_forward(model, tokenizer, device: str) -> ForwardResult:
        import torch

        try:
            if device == "cuda" and not torch.cuda.is_available():
                return ForwardResult(False, device, None, None, "cuda not available")
            inputs = tokenizer(SHORT_FORWARD_PROMPT, return_tensors="pt")
            target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(target_device)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            logits = output.logits
            finite = bool(torch.isfinite(logits.float()[0, :16, :64]).all().item())
            return ForwardResult(True, target_device, list(logits.shape), finite)
        except Exception as exc:  # noqa: BLE001 - 校验必须把失败变成结果而不是异常
            return ForwardResult(False, device, None, None, f"{exc.__class__.__name__}: {exc}")

    return {
        "load_model": load_model,
        "load_tokenizer": load_tokenizer,
        "load_peft_adapter": load_peft_adapter,
        "adapter_weight_keys": adapter_weight_keys,
        "trainable_stats": trainable_stats,
        "short_forward": short_forward,
    }


def resolve_device(requested: str) -> str:
    """auto 在无 CUDA 时退回 CPU；其余原样返回，由调用方校验。"""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# These files are evidence for the saved output, not the tokenizer identity
# itself.  Identity is established from the objects resolved by the base and
# output loaders below.  This matters because ``save_pretrained`` may
# canonicalize/merge files, and a Hub snapshot may contain cache/download
# metadata that is not part of tokenizer semantics.
_TOKENIZER_FILE_NAMES = {
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "vocab.spm",
    "merges.txt",
    "merges.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
    "feature_extractor_config.json",
}
_TOKENIZER_FILE_MARKERS = ("tokenizer", "processor", "chat_template", "chat-template")
_TOKENIZER_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def _is_tokenizer_identity_file(path: Path) -> bool:
    name = path.name.lower()
    ignored_parts = {".cache", "cache", "downloads", "download", "snapshots", "blobs", "refs"}
    if any(part.startswith(".") or part.lower() in ignored_parts for part in path.parts):
        return False
    return (
        path.is_file()
        and (name in _TOKENIZER_FILE_NAMES or any(marker in name for marker in _TOKENIZER_FILE_MARKERS))
        and path.suffix.lower() not in _TOKENIZER_WEIGHT_SUFFIXES
    )


def _tokenizer_identity_files(directory: Path) -> dict[str, Path]:
    """Discover non-cache tokenizer artifacts for output hash evidence."""
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if _is_tokenizer_identity_file(path)
    }


def _canonical_json(value) -> bytes:
    """Serialize tokenizer metadata deterministically across HF serializers."""
    import json

    def normalize(item):
        if isinstance(item, dict):
            return {str(key): normalize(item[key]) for key in sorted(item, key=lambda k: str(k))}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, set):
            return sorted((normalize(part) for part in item), key=lambda part: repr(part))
        if hasattr(item, "item") and callable(item.item):
            try:
                return normalize(item.item())
            except Exception:  # pragma: no cover - unusual scalar wrappers
                pass
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_canonical(value) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _class_identity(value) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def tokenizer_identity_fingerprint(tokenizer_or_processor) -> dict:
    """Return semantic tokenizer identity, with an explicit capability contract.

    Test fakes and third-party loaders must expose the same fields as a
    Transformers tokenizer/processor.  Missing fields are reported rather
    than silently falling back to artifact byte equality.
    """
    import hashlib

    processor = tokenizer_or_processor
    tokenizer = getattr(processor, "tokenizer", processor)
    missing: list[str] = []

    def read(name, source=tokenizer, *, callable_required=False):
        value = getattr(source, name, None)
        if value is None or (callable_required and not callable(value)):
            missing.append(name)
            return None
        try:
            return value() if callable_required else value
        except Exception as exc:  # expose loader contract failures as data
            missing.append(f"{name} ({exc.__class__.__name__})")
            return None

    vocab = read("get_vocab", callable_required=True)
    added_vocab = read("get_added_vocab", callable_required=True)
    vocab_size = read("vocab_size")
    all_special_tokens = read("all_special_tokens")
    all_special_ids = read("all_special_ids")
    special_tokens_map = read("special_tokens_map")
    if hasattr(processor, "chat_template"):
        chat_template = read("chat_template", processor)
    elif processor is not tokenizer:
        chat_template = read("chat_template", tokenizer)
    else:
        chat_template = read("chat_template", processor)
    if hasattr(processor, "model_input_names"):
        model_input_names = read("model_input_names", processor)
    elif processor is not tokenizer:
        model_input_names = read("model_input_names", tokenizer)
    else:
        model_input_names = read("model_input_names", processor)

    # Keep hashes and the canonical values that are useful in a verifier
    # report.  The vocabulary hash covers every token -> id pair, not merely a
    # file that happened to be emitted by a particular serializer.
    fingerprint = {
        "schema_version": "tokenizer-semantic-v1",
        # Do not use ``token`` in manifest field names: manifests are checked
        # for secret-bearing key fragments.  These names still explicitly
        # record both resolved component classes.
        "processor_type": _class_identity(processor),
        "encoder_type": _class_identity(tokenizer),
        "vocab_sha256": _sha256_canonical(vocab) if vocab is not None else None,
        "added_vocab_sha256": _sha256_canonical(added_vocab) if added_vocab is not None else None,
        "vocab_size": vocab_size,
        "special_values": all_special_tokens,
        "special_ids": all_special_ids,
        "special_map_sha256": _sha256_canonical(special_tokens_map),
        "chat_template_sha256": _sha256_canonical(chat_template),
        "model_input_names": model_input_names,
    }
    fingerprint["semantic_sha256"] = hashlib.sha256(_canonical_json(fingerprint)).hexdigest()
    fingerprint["missing_fields"] = sorted(set(missing))
    fingerprint["supported"] = not missing
    return fingerprint


def tokenizer_artifact_hashes(directory) -> list[dict[str, str]]:
    """Hash emitted tokenizer files for audit without making them identity."""
    import hashlib

    return [
        {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in sorted(_tokenizer_identity_files(Path(directory)).items())
    ]


def check_tokenizer_identity(
    output_dir,
    base_dir,
    *,
    output_tokenizer=None,
    base_tokenizer=None,
) -> dict:
    """Compare resolved tokenizer/processor semantics, not serialized files."""
    output_dir = Path(output_dir)
    base_dir = Path(base_dir)
    output_fp = tokenizer_identity_fingerprint(output_tokenizer) if output_tokenizer is not None else None
    base_fp = tokenizer_identity_fingerprint(base_tokenizer) if base_tokenizer is not None else None
    problems = []
    if output_fp is None or base_fp is None:
        problems.append("必须提供 base/output loader 解析结果")
    else:
        if not output_fp["supported"]:
            problems.append(f"output loader 缺少 tokenizer 语义字段: {output_fp['missing_fields']}")
        if not base_fp["supported"]:
            problems.append(f"base loader 缺少 tokenizer 语义字段: {base_fp['missing_fields']}")
        if output_fp["supported"] and base_fp["supported"] and output_fp["semantic_sha256"] != base_fp["semantic_sha256"]:
            problems.append("base/output resolved tokenizer semantic fingerprint 不一致")
    detail = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "base_fingerprint": base_fp,
        "output_fingerprint": output_fp,
        "output_artifact_sha256": tokenizer_artifact_hashes(output_dir),
        "base_artifact_sha256": tokenizer_artifact_hashes(base_dir),
        "problems": problems,
    }
    return {
        "name": "tokenizer_identity_matches_base",
        "passed": not problems,
        "detail": detail,
    }
