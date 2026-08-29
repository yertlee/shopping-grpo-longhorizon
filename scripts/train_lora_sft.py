#!/usr/bin/env python3
"""对验收后的 Shopping tool-calling 数据进行最小 LoRA SFT。"""

import argparse
import json
import sys
import time as _time
from functools import partial
from pathlib import Path

from shopping_grpo.evaluation.artifacts import ArtifactError
from shopping_grpo.evaluation.blind_guard import guard_blind_final
from shopping_grpo.training.sft.dataset import (
    load_supervised_examples,
    select_training_examples,
)
from shopping_grpo.training.sft.run_manifest import (
    FROZEN_MODEL_REVISION,
    build_run_manifest,
    finalize_run_manifest,
    hash_weight_files,
    load_run_manifest,
    CHECKPOINT_OWNER_FILE,
    sha256_file,
    validate_checkpoint_owner,
    write_checkpoint_owner,
    write_run_manifest,
    validate_model_revision,
    validate_run_id,
)

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 的大多数文本层是 Gated DeltaNet，不能遗漏其线性注意力投影。
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)

# ---- Run manifest 状态：main() 写入；失败时由 __main__ 钩子落盘。 ----
_RUN_STATE = {
    "manifest_path": None,
    "train_examples": None,
    "validation_examples": None,
}


def _dependency_versions():
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors", "liger-kernel", "bitsandbytes"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _gpu_info(torch):
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "device_count": 0, "cuda_runtime": torch.version.cuda}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "cuda_runtime": torch.version.cuda,
    }


def _code_hashes():
    import shopping_grpo.training.sft.dataset as dataset_module
    import shopping_grpo.training.sft.run_manifest as run_manifest_module

    return {
        "scripts/train_lora_sft.py": sha256_file(Path(__file__).resolve()),
        "src/shopping_grpo/training/sft/dataset.py": sha256_file(Path(dataset_module.__file__).resolve()),
        "src/shopping_grpo/training/sft/run_manifest.py": sha256_file(
            Path(run_manifest_module.__file__).resolve()
        ),
    }


def _manifest_drift(existing: dict, candidate: dict) -> list[str]:
    """比较两个 manifest 的身份部分；execution/result 与运行时产物计数不参与。"""
    drift = []
    for section in ("model", "data", "recipe", "runtime"):
        old = existing.get(section) or {}
        new = candidate.get(section) or {}
        for key in sorted(set(old) | set(new)):
            if section == "data" and key in {"train_examples", "validation_examples"}:
                continue  # finalize 阶段回填的运行时产物，不是合同
            if old.get(key) != new.get(key):
                drift.append(f"{section}.{key}: {old.get(key)!r} -> {new.get(key)!r}")
    return drift


def _write_failure_manifest_if_pending():
    """main() 未走到成功 finalize 时，把失败写进 run manifest；不吞异常。"""
    path = _RUN_STATE["manifest_path"]
    if path is None:
        return
    manifest = load_run_manifest(path)
    if manifest["execution"]["exit_code"] is not None:
        return
    exc = sys.exc_info()[1]
    finalize_run_manifest(
        manifest,
        train_examples=_RUN_STATE["train_examples"] or 0,
        validation_examples=_RUN_STATE["validation_examples"] or 0,
        train_loss=None,
        eval_loss=None,
        peak_gpu_memory_gib=None,
        checkpoint_paths=sorted(
            str(path)
            for path in _RUN_STATE["manifest_path"].parent.glob("checkpoint-*")
            if path.is_dir()
        ),
        exit_code=1,
        error=f"{exc.__class__.__name__}: {exc}" if exc is not None else "interrupted",
    )
    write_run_manifest(path, manifest)
    print(f"[run-manifest] 已记录失败状态到 {path}", flush=True)


def _validate_resume_checkpoint(path: str, output: Path, manifest: dict) -> Path:
    """Ensure a resume checkpoint is present and cryptographically self-consistent.

    ``execution.checkpoint_paths`` is a post-run inventory and is deliberately
    not an ownership proof: an attacker can edit that mutable field.  The
    checkpoint must carry an owner sidecar created by the Trainer callback,
    bound to this manifest's immutable run_id and its actual relative path.
    """
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_dir():
        raise SystemExit(f"resume checkpoint 不存在或不是目录：{checkpoint}")
    try:
        checkpoint_resolved = checkpoint.resolve(strict=True)
        output_resolved = output.resolve(strict=True)
        checkpoint_resolved.relative_to(output_resolved)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            "resume checkpoint 必须位于当前 --output 目录内；拒绝 foreign checkpoint："
            f" {checkpoint}"
        ) from exc
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SystemExit(
            "run_manifest 缺少 run_id；拒绝无法证明归属的 checkpoint"
        )
    state_path = checkpoint_resolved / "trainer_state.json"
    if not state_path.is_file():
        raise SystemExit(
            f"resume checkpoint 缺少 trainer_state.json，无法核对 global_step：{checkpoint}"
        )
    try:
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        global_step = int(trainer_state["global_step"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"resume checkpoint trainer_state.json 缺少有效 global_step：{checkpoint}"
        ) from exc
    owner_path = checkpoint_resolved / CHECKPOINT_OWNER_FILE
    if not owner_path.is_file():
        raise SystemExit(
            f"resume checkpoint 缺少 ownership sidecar {CHECKPOINT_OWNER_FILE}：{checkpoint}"
        )
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        relative = checkpoint_resolved.relative_to(output_resolved).as_posix()
        validate_checkpoint_owner(
            owner,
            run_id=run_id,
            checkpoint_path=relative,
            global_step=global_step,
            checkpoint_dir=checkpoint_resolved,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"resume checkpoint ownership sidecar 校验失败，拒绝续训：{checkpoint} ({exc})"
        ) from exc
    return checkpoint_resolved


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Transformers + PEFT 执行 Shopping LoRA SFT")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--train", type=Path, required=True, help="训练 SFT JSONL")
    parser.add_argument("--validation", type=Path, default=None, help="可选验证 SFT JSONL")
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        default=None,
        help="可选课程清单；与 --curriculum-stage 一起按 task_id 选择数据。",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=("a", "b", "c"),
        default=None,
        help="课程阶段；训练/验证都从同一 manifest 的累计 bucket 中读取。",
    )
    parser.add_argument("--output", type=Path, required=True, help="LoRA adapter 输出目录")
    # 24k 可保留当前真实轨迹的约 93%，48G 显存配合 batch=1 与梯度检查点可稳定训练。
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="模型与训练精度；auto 在 CUDA 上优先 bf16，其次 fp16，CPU 使用 fp32。",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="兼容旧命令；等价于 --dtype bf16，不能与其他 --dtype 同时使用。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选模型 revision；本地路径通常不需要。",
    )
    parser.add_argument("--liger-kernel", action="store_true", help="启用 Liger 融合 loss，避免全序列 logits 常驻")
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "sdpa"),
        default="auto",
        help="注意力后端；sdpa 使用 PyTorch 原生内存高效实现，不要求编译 FlashAttention 2。",
    )
    parser.add_argument("--qlora", action="store_true", help="以 NF4 4-bit 加载基座，并按 PEFT 标准预处理")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--stage",
        choices=("smoke", "process", "outcome"),
        default=None,
        help="写入 run manifest 的 stage 标记；缺省记为 unspecified。",
    )
    parser.add_argument(
        "--sft-ready-manifest",
        type=Path,
        default=None,
        help="teacher-sft-ready manifest 路径；其 SHA256 会被记录进 run manifest。",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=None,
        help="最终 preflight report 路径；其 SHA256 会被记录进 run manifest。",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="最大训练步数（-1=完整 epoch）；用于冒烟测试")
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 训练监控")
    parser.add_argument("--swanlab-project", default="shopping-grpo", help="SwanLab project 名")
    parser.add_argument("--swanlab-run-name", default=None, help="SwanLab run 名；默认自动生成")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local"),
        default="online",
        help="SwanLab 在线同步或只保存在本地；仅 --swanlab 时生效。",
    )
    return parser.parse_args()


def _curriculum_task_ids(path, stage, split):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        bucket_names = manifest["stages"][stage]["buckets"]
        key = f"{split}_task_ids"
        return {
            int(task_id)
            for bucket_name in bucket_names
            for task_id in manifest["buckets"][bucket_name][key]
        }
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"课程清单缺少阶段 {stage!r} 的 {split} task IDs") from exc


def _training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv sync --extra sft") from exc
    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )


def _model_load_kwargs(args, dtype, bits_and_bytes_config):
    """构造可审计的模型加载参数；加速功能必须显式开启。"""
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.attention_implementation != "auto":
        kwargs["attn_implementation"] = args.attention_implementation
    if args.qlora:
        kwargs["quantization_config"] = bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _prepare_model_for_training(model, args, prepare_model_for_kbit_training):
    """按 PEFT 推荐顺序准备量化模型与梯度检查点。"""
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if not args.qlora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _validate_optional_training_dependencies(args):
    """仅在所选实验需要时检查可选加速包，保持基础 LoRA 环境轻量。"""
    if args.qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--qlora 需要 bitsandbytes；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc
    if args.liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--liger-kernel 需要 liger-kernel；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc


def _resolve_dtype(args, torch):
    """Resolve one explicit dtype for model loading and TrainingArguments."""

    requested = args.dtype
    if args.bf16:
        if requested not in {"auto", "bf16"}:
            raise SystemExit("--bf16 cannot be combined with a non-bf16 --dtype")
        requested = "bf16"
    if requested == "auto":
        if torch.cuda.is_available():
            requested = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            requested = "fp32"
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return requested, mapping[requested]


def _swanlab_config(args):
    """准备官方 Transformers 集成所需的最小 SwanLab 配置。"""
    if not args.swanlab:
        return "none", None
    try:
        import swanlab  # noqa: F401 - 仅验证可选依赖存在。
    except ImportError as exc:
        raise SystemExit("缺少 SwanLab。请执行：uv sync --extra sft") from exc

    run_name = args.swanlab_run_name or (
        f"lora-r{args.lora_r}-bs{args.per_device_train_batch_size}"
        f"x{args.gradient_accumulation_steps}-lr{args.learning_rate}"
    )
    return "swanlab", run_name


def _loss_only_eval_trainer_class(trainer_base, enable_skip_logits):
    """构造只在 loss-only 验证时显式跳过完整词表 logits 的 Trainer。

    Qwen3.5 的 Liger forward 默认只在 ``model.training`` 时启用融合
    LM-head + cross-entropy；Trainer 验证会先调用 ``model.eval()``，即使最终
    只需要 eval_loss，也会物化 ``[batch, sequence, vocab]`` logits。20K 上下文
    和 248K 词表会因此产生约 20 GiB 的瞬时 FP32 张量。

    ``skip_logits`` 是 Liger Qwen3.5 forward 的公开参数。这里只在 Trainer 已经
    明确 ``prediction_loss_only=True`` 且输入含 labels 时传入，不改变训练前向，
    也不影响需要 predictions/metrics 的评估。
    """

    class LossOnlyEvalTrainer(trainer_base):
        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
        ):
            if enable_skip_logits and prediction_loss_only and inputs.get("labels") is not None:
                inputs = dict(inputs)
                inputs["skip_logits"] = True
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

    return LossOnlyEvalTrainer


def _load_preprocessing_components(
    model_name,
    auto_config,
    auto_tokenizer,
    auto_processor,
    revision=None,
):
    """按模型配置选择 chat template 的持有者。

    Qwen3.5 是带视觉编码器的条件生成模型，官方模板由 processor 提供；本项目
    当前数据仅含文本和工具调用，因此 labels 仍用 processor.tokenizer 的 token id。
    其他纯文本因果模型保持原来的 tokenizer 路径。
    """
    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    config = auto_config.from_pretrained(model_name, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = auto_processor.from_pretrained(model_name, **load_kwargs)
        return processor.tokenizer, processor, True
    tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
    return tokenizer, tokenizer, False


def _torch_dataset(examples, torch):
    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }

    return TokenizedDataset()


def _collate(batch, pad_token_id, torch):
    """右侧 padding，labels 的 padding 永远不参与 loss。"""
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    _start_time = _time.time()
    args = parse_args()
    try:
        validate_model_revision(args.revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_length < 1 or args.epochs <= 0:
        raise SystemExit("--max-length 与 --epochs 必须为正数")
    if bool(args.curriculum_manifest) != bool(args.curriculum_stage):
        raise SystemExit("--curriculum-manifest 与 --curriculum-stage 必须一起提供")
    if args.curriculum_manifest and not args.validation:
        raise SystemExit("课程训练必须提供 --validation（通常与 --train 指向同一 Pure V4 文件）")
    train_task_ids = validation_task_ids = None
    if args.curriculum_manifest:
        train_task_ids = _curriculum_task_ids(
            args.curriculum_manifest, args.curriculum_stage, "train"
        )
        validation_task_ids = _curriculum_task_ids(
            args.curriculum_manifest, args.curriculum_stage, "validation"
        )
    try:
        guard_blind_final(
            [args.train, *([args.validation] if args.validation else [])],
            allowed=False,
        )
    except ArtifactError as exc:
        raise SystemExit(str(exc)) from exc
    _validate_optional_training_dependencies(args)
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = _training_dependencies()
    dtype_name, dtype = _resolve_dtype(args, torch)

    # ---- Run manifest：先固化输入 / 配方 / 代码 hash；失败也要留下可审计记录。 ----
    manifest_path = args.output / "run_manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = build_run_manifest(
        stage=args.stage or "unspecified",
        model_path=args.model,
        model_revision=args.revision,
        weights=hash_weight_files(args.model),
        train_path=args.train,
        validation_path=args.validation,
        sft_ready_manifest_path=args.sft_ready_manifest,
        preflight_report_path=args.preflight_report,
        recipe={
            "max_length": args.max_length,
            "epochs": args.epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
            "dtype": dtype_name,
            "attention": args.attention_implementation,
            "liger_kernel": args.liger_kernel,
            "qlora": args.qlora,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "subset_seed": args.subset_seed,
            "train_count": args.train_count,
            "train_ratio": args.train_ratio,
            "max_steps": args.max_steps,
            "curriculum_stage": args.curriculum_stage,
        },
        code_hashes=_code_hashes(),
        dependency_versions=_dependency_versions(),
        gpu=_gpu_info(torch),
        command=list(sys.argv),
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    existing_manifest = None
    if manifest_path.is_file():
        existing_manifest = load_run_manifest(manifest_path)
        if not args.resume_from_checkpoint:
            raise SystemExit(
                f"输出目录已有 run manifest，拒绝覆盖：{manifest_path}；"
                "续训请使用 --resume-from-checkpoint，或为本次运行指定新的 --output 目录"
            )
        identity = validate_run_id(existing_manifest)
        if not identity["passed"]:
            raise SystemExit(
                "resume 前原始 run_manifest 的 run_id 校验失败："
                f" recorded={identity['actual']!r} recomputed={identity['expected']!r}"
            )
        _validate_resume_checkpoint(args.resume_from_checkpoint, args.output, existing_manifest)
        drift = _manifest_drift(existing_manifest, run_manifest)
        if existing_manifest["run_id"] != run_manifest["run_id"] or drift:
            detail = "\n  ".join(drift) if drift else (
                f"run_id {existing_manifest['run_id']} != {run_manifest['run_id']}"
            )
            raise SystemExit(
                "resume 前合同漂移检查失败：当前模型/数据/配方/代码与该 checkpoint "
                f"的原始 run 不一致，拒绝续训（--output={args.output}）。\n漂移项：\n  {detail}"
            )
        # Preserve the finalized source manifest.  A resume is a new execution
        # record with the same identity, so finalize/failure handling cannot
        # accidentally report or overwrite the original run.
        resume_index = 1
        while (args.output / f"run_manifest.resume.{resume_index}.json").exists():
            resume_index += 1
        manifest_path = args.output / f"run_manifest.resume.{resume_index}.json"
    write_run_manifest(manifest_path, run_manifest)
    _RUN_STATE["manifest_path"] = manifest_path

    # --- Progress callback: 只补充 Trainer 默认没有的耗时和显存指标。 ---
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.step_start = None
            self.epoch_start = None

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = _time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return control
            elapsed = _time.time() - self.step_start if self.step_start else 0.0
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            logs["step_time_s"] = round(elapsed, 3)
            logs["gpu_peak_memory_gib"] = round(gpu_mem, 3)
            eta_seconds = (state.max_steps - state.global_step) * elapsed if state.global_step else 0
            eta_str = f"{eta_seconds/60:.0f}min" if eta_seconds > 0 else "?"
            print(
                f"[step {state.global_step}/{state.max_steps}] "
                f"loss={float(logs['loss']):.4f} step_t={elapsed:.1f}s "
                f"GPU={gpu_mem:.1f}GiB ETA={eta_str}"
            )
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = _time.time()
            print(f"\n{'='*60}\n  EPOCH {int(state.epoch)} 开始  steps={state.max_steps}\n{'='*60}")
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_time = _time.time() - self.epoch_start if self.epoch_start else 0
            print(f"  EPOCH {int(state.epoch)} 完成  耗时={epoch_time/60:.1f}min")

    class CheckpointOwnershipCallback(TrainerCallback):
        """Bind every Transformers checkpoint to this run after it is saved."""

        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            write_checkpoint_owner(
                checkpoint,
                output_dir=args.output_dir,
                run_id=run_manifest["run_id"],
                global_step=int(state.global_step),
            )
            return control

    tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
        args.model,
        auto_config=AutoConfig,
        auto_tokenizer=AutoTokenizer,
        auto_processor=AutoProcessor,
        revision=args.revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Phase 1: 加载训练数据 ----
    print(f"\n{'='*60}")
    print(f"  Phase 1/3: 加载 & Tokenize 训练数据 (max_length={args.max_length})")
    print(f"{'='*60}")
    train_examples, train_stats = load_supervised_examples(
        args.train,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=args.max_length,
        task_ids=train_task_ids,
    )
    if train_task_ids is not None and train_stats["matched"] != len(train_task_ids):
        raise SystemExit("课程清单中的训练 task_id 未全部出现在 --train 数据中")
    try:
        train_examples = select_training_examples(
            train_examples,
            count=args.train_count,
            ratio=args.train_ratio,
            seed=args.subset_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    train_stats["selected"] = len(train_examples)
    _RUN_STATE["train_examples"] = len(train_examples)
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本；请检查 data/sft/ 中的 JSONL 格式")
    validation_examples = []
    if args.validation:
        validation_examples, validation_stats = load_supervised_examples(
            args.validation,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=args.max_length,
            task_ids=validation_task_ids,
        )
        if validation_task_ids is not None and validation_stats["matched"] != len(
            validation_task_ids
        ):
            raise SystemExit("课程清单中的验证 task_id 未全部出现在 --validation 数据中")
        print("validation_data=", validation_stats)
        if not validation_examples:
            raise SystemExit("验证集没有可用样本；请调整划分或 --max-length")
        _RUN_STATE["validation_examples"] = len(validation_examples)

    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM

    # ---- Phase 2: 加载模型 + LoRA ----
    print(f"\n{'='*60}")
    print("  Phase 2/3: 加载模型与 LoRA")
    print(f"{'='*60}")
    print(f"  model={args.model}")
    print(f"  revision={args.revision or 'default/local'}")
    print(f"  dtype={dtype_name}")
    print(f"  attention_implementation={args.attention_implementation}")
    print(f"  qlora={args.qlora}")
    print(f"  lora_r={args.lora_r} lora_alpha={args.lora_alpha}")
    print(f"  lora_targets={','.join(args.target_modules)}")
    model = model_class.from_pretrained(
        args.model,
        **_model_load_kwargs(
            args,
            dtype=dtype,
            bits_and_bytes_config=BitsAndBytesConfig,
        ),
    )
    model = _prepare_model_for_training(
        model,
        args,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.target_modules),
        ),
    )
    model.print_trainable_parameters()
    trainable_params, total_params = model.get_nb_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    report_to, run_name = _swanlab_config(args)
    if report_to == "swanlab":
        import swanlab
        swanlab.init(
            project=args.swanlab_project,
            name=run_name,
            mode=args.swanlab_mode,
            logdir=str(args.output / "swanlab"),
        )
        print(f"[SwanLab] project={args.swanlab_project} run={run_name}")
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.liger_kernel,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if validation_examples else "no",
        report_to=report_to,
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer_class = _loss_only_eval_trainer_class(
        Trainer,
        enable_skip_logits=args.liger_kernel and is_multimodal,
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=_torch_dataset(train_examples, torch),
        eval_dataset=_torch_dataset(validation_examples, torch) if validation_examples else None,
        data_collator=partial(_collate, pad_token_id=tokenizer.pad_token_id, torch=torch),
        callbacks=[ProgressCallback(), CheckpointOwnershipCallback()],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    chat_template.save_pretrained(str(args.output))

    # --- 训练完成摘要 ---
    total_time = _time.time() - _start_time
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    train_summary = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_loss": result.training_loss,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "peak_gpu_memory_gib": round(gpu_peak, 2),
        "total_time_minutes": round(total_time / 60, 1) if total_time else None,
        "monitoring": {
            "backend": report_to,
            "project": args.swanlab_project if args.swanlab else None,
            "run_name": run_name,
            "mode": args.swanlab_mode if args.swanlab else None,
        },
        "acceleration": {
            "dtype": dtype_name,
            "liger_kernel": args.liger_kernel,
            "attention_implementation": args.attention_implementation,
            "qlora": args.qlora,
        },
        "lora": {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_ratio": (trainable_params / total_params) if total_params else None,
            "target_modules": list(args.target_modules),
            "is_multimodal": is_multimodal,
        },
        "arguments": vars(args),
    }

    print(f"\n{'='*60}")
    print("  训练完成")
    print(f"  train_loss={result.training_loss:.4f}")
    print(f"  eval_loss={result.metrics.get('eval_loss', 'N/A')}")
    print(f"  peak_gpu={gpu_peak:.1f} GiB")
    print(f"  adapter → {args.output}")
    print(f"{'='*60}\n")

    (args.output / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"LoRA adapter 已保存到 {args.output}")

    # ---- 用实际运行结果 finalize run manifest。 ----
    train_loss = float(result.training_loss) if result.training_loss is not None else None
    eval_loss_raw = result.metrics.get("eval_loss")
    eval_loss = float(eval_loss_raw) if eval_loss_raw is not None else None
    run_manifest = finalize_run_manifest(
        run_manifest,
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        train_loss=train_loss,
        eval_loss=eval_loss,
        peak_gpu_memory_gib=round(gpu_peak, 2),
        checkpoint_paths=sorted(str(path) for path in args.output.glob("checkpoint-*")),
        exit_code=0,
    )
    write_run_manifest(manifest_path, run_manifest)
    print(f"run manifest 已写入 {manifest_path} (run_id={run_manifest['run_id']})")


if __name__ == "__main__":
    try:
        main()
    finally:
        _write_failure_manifest_if_pending()
