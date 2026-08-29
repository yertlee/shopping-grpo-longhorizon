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

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
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

        return PeftModel.from_pretrained(base_model, str(adapter_path))

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


# Only files whose names identify tokenizer/processor/template artifacts are
# part of this identity.  In particular, do not use a broad extension glob:
# model shards and other checkpoint payloads must never become tokenizer
# identity evidence.  The predicate is intentionally shared by adapter and
# merged verifiers so both enforce the same exact-set contract.
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
    return (
        path.is_file()
        and (name in _TOKENIZER_FILE_NAMES or any(marker in name for marker in _TOKENIZER_FILE_MARKERS))
        and path.suffix.lower() not in _TOKENIZER_WEIGHT_SUFFIXES
    )


def _tokenizer_identity_files(directory: Path) -> dict[str, Path]:
    """Discover relevant tokenizer artifacts recursively by relative path.

    Hugging Face tokenizers/processors are normally saved at the directory
    root, but recursive discovery also covers processor subdirectories.  An
    unrelated config/weight file is not included merely because it is in the
    model directory.
    """
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if _is_tokenizer_identity_file(path)
    }


def check_tokenizer_identity(output_dir, base_dir) -> dict:
    """Require an exact tokenizer/processor artifact set and byte identity."""
    import hashlib

    output_dir = Path(output_dir)
    base_dir = Path(base_dir)

    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if not base_dir.is_dir() or not output_dir.is_dir():
        return {
            "name": "tokenizer_identity_matches_base",
            "passed": False,
            "detail": {
                "reason": f"base/output 目录不存在，无法比对 tokenizer：{base_dir} / {output_dir}",
                "missing_in_output": [],
                "unexpected_in_output": [],
                "hash_mismatch": [],
            },
        }
    base_files = _tokenizer_identity_files(base_dir)
    output_files = _tokenizer_identity_files(output_dir)
    base_names = set(base_files)
    output_names = set(output_files)
    missing_in_output = sorted(base_names - output_names)
    unexpected_in_output = sorted(output_names - base_names)
    hash_mismatch = sorted(
        name
        for name in base_names & output_names
        if _hash(base_files[name]) != _hash(output_files[name])
    )
    passed = bool(base_names) and not (
        missing_in_output or unexpected_in_output or hash_mismatch
    )
    detail = {
        "base_files": sorted(base_names),
        "output_files": sorted(output_names),
        "missing_in_output": missing_in_output,
        "unexpected_in_output": unexpected_in_output,
        "hash_mismatch": hash_mismatch,
        "base_dir": str(base_dir),
    }
    if not base_names:
        detail["reason"] = "基座目录没有可识别的 tokenizer/processor 文件"
    return {
        "name": "tokenizer_identity_matches_base",
        "passed": passed,
        "detail": detail,
    }
