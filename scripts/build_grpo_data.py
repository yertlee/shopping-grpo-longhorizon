#!/usr/bin/env python3
"""从冻结 split manifest / task facts 构建 GRPO train/validation 数据与 metadata。

输入全部来自阶段 A 冻结证据链（commerce-agent-posttrain 侧 manifest），本脚本只
消费、不重建：split manifest 提供 grpo_train / grpo_validation / final /
teacher_pool 池，task facts 提供每题的公开 instruction（query），reachability
manifest 提供不可达任务的冻结排除原因。

硬性保护（HANDOFF_TRAINING_IMPLEMENTATION_GAPS §6.3）：
- 每个 split 的 task_id 唯一，且 Train/Validation/Final/teacher_pool/SFT 无交集；
  SFT 泄漏检查是强制的：正式构建必须显式提供四个 SFT JSONL（Process/Outcome 两臂
  × Train/Dev）；不提供时构建直接失败，显式
  ``--i-know-leak-check-is-required`` 跳过时会在 metadata 里写醒目审计字段并让
  ``leakage.all_checks_passed=false``（launcher 会拒绝启动该数据集）；
- prompt 只含 system prompt + 用户 query，绝不写入 target_product_ids /
  target_title / target_options / product_family 等隐藏目标字段；
- 行数、唯一 task 数、task_id hash、文件 SHA256 全部写入 metadata.json；
- metadata 同时携带 reward_version / environment_version / tool_schema_sha256 /
  system_prompt_sha256，供 launcher preflight 与 runtime contract 交叉校验；
- 不可达任务按 reachability manifest 的冻结原因留审计记录，绝不静默删行；
- 同一输入两次构建输出字节一致（确定性；metadata 不含时间戳与绝对路径）。

本地开发环境可能没有 pyarrow：parquet 写入层通过 ``writer_factory`` 注入，
默认工厂在真正写入时才懒加载 pyarrow；单测注入 JSONL fake writer。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT  # noqa: E402
from shopping_grpo.environment.tools import validate_runtime_tool_schema  # noqa: E402
from shopping_grpo.training.sft.run_manifest import sha256_file  # noqa: E402

SPLIT_MANIFEST_SCHEMA = "commerce-split-manifest-v1"
CONTRACT_SCHEMA = "commerce-runtime-contract-v1"
REACHABILITY_SCHEMA = "commerce-task-reachability-manifest-v1"
TASK_FACTS_SCHEMA = "commerce-task-facts-v1"
METADATA_SCHEMA = "shopping-grpo-data-metadata-v2"

# grpo_train/grpo_validation 之外的池：只用于泄漏断言，绝不从中取数构建样本。
FORBIDDEN_SPLITS = ("final", "teacher_pool")
# task facts 中禁止进入训练样本的隐藏字段（目标商品 / 选项 / 品类族）。
FORBIDDEN_FACT_FIELDS = ("target_title", "target_options", "target_product_ids", "product_family")
# extra_info 白名单：任何其他键都会让构建失败，防止把隐藏数据夹带进样本。
EXTRA_INFO_KEYS = ("task_id", "split", "contract_sha256")

# SFT 泄漏检查（audit P0：checked=false 不允许静默通过）。
SFT_LEAK_CHECK_OVERRIDE_FLAG = "--i-know-leak-check-is-required"
SFT_TASK_ID_SOURCES_EXPECTED = 4  # Process/Outcome 两臂 × Train/Dev（或一份合并 manifest）
SFT_SOURCE_IDENTITIES = (
    "process_train",
    "process_dev",
    "outcome_train",
    "outcome_dev",
)
# metadata 携带、launcher 交叉校验的版本字段 ↔ runtime contract 字段映射。
METADATA_CONTRACT_VERSION_FIELDS = (
    ("reward_version", "reward_version"),
    ("environment_version", "environment_version"),
    ("tool_schema_sha256", "tool_schema_hash"),
    ("system_prompt_sha256", "system_prompt_hash"),
)

TRAIN_SPLIT = "grpo_train"
VALIDATION_SPLIT = "grpo_validation"
SPLIT_TO_FILENAME = {TRAIN_SPLIT: "train.parquet", VALIDATION_SPLIT: "validation.parquet"}

DEFAULT_MAX_PROMPT_TOKENS = 4096  # 与 configs/grpo.yaml data.max_prompt_length 一致
# Prompt 长度审计必须与 SFT/merge/GRPO 使用的同一冻结基座匹配。这个值既是
# CLI 默认值，也是 metadata/launcher 的正式合同；不能退回 refs/main 或 None。
FROZEN_TOKENIZER_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# Descriptive alias for callers that use the launcher naming convention.
EXPECTED_TOKENIZER_REVISION = FROZEN_TOKENIZER_REVISION
FORBIDDEN_SCAN_MIN_CHARS = 4  # 过短的值容易在自然语言里误命中，扫描时跳过

# token 计数器注入签名：输入完整 prompt messages，返回预估 token 数。
TokenCounter = Callable[[Sequence[Mapping[str, str]]], int]
# parquet 写入工厂注入签名：输入目标路径，返回带 write(row)/close() 的 writer。
WriterFactory = Callable[[Path], Any]


def canonical_task_ids_sha256(task_ids: Sequence[int]) -> str:
    """split manifest 的 task_ids_sha256 口径：紧凑 JSON 数组后取 SHA256。"""
    payload = json.dumps([int(task_id) for task_id in task_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read frozen input {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"frozen input is not valid JSON: {path}: {exc}") from exc


def load_task_facts(path: Path, expected_rows: int) -> dict[int, dict]:
    """读取 task facts JSONL 并建立 task_id 索引；行数必须与 split manifest 一致。"""
    facts: dict[int, dict] = {}
    rows = 0
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read task facts {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"task facts {path} line {line_number} is not valid JSON: {exc}") from exc
            task_id = row.get("task_id")
            if not isinstance(task_id, int):
                raise SystemExit(f"task facts {path} line {line_number} is missing integer task_id")
            if row.get("schema_version") != TASK_FACTS_SCHEMA:
                raise SystemExit(
                    f"task facts {path} line {line_number} has unexpected schema_version: "
                    f"{row.get('schema_version')!r} (expected {TASK_FACTS_SCHEMA})"
                )
            if task_id in facts:
                raise SystemExit(f"task facts {path} contains duplicate task_id {task_id}")
            facts[task_id] = row
    if rows != expected_rows:
        raise SystemExit(
            f"task facts {path} has {rows} rows, but the frozen split manifest declares {expected_rows}"
        )
    return facts


def validate_split_manifest(manifest: Mapping) -> None:
    """校验 split manifest 自身结构，以及与 runtime contract 的绑定关系。"""
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise SystemExit(
            f"split manifest schema_version must be {SPLIT_MANIFEST_SCHEMA}, got {manifest.get('schema_version')!r}"
        )
    for name in (TRAIN_SPLIT, VALIDATION_SPLIT, *FORBIDDEN_SPLITS):
        split = manifest.get("splits", {}).get(name)
        if not isinstance(split, dict):
            raise SystemExit(f"split manifest is missing splits.{name}")
        task_ids = split.get("task_ids")
        if not isinstance(task_ids, list) or not all(isinstance(item, int) for item in task_ids):
            raise SystemExit(f"splits.{name}.task_ids must be a list of integers")
        if split.get("count") != len(task_ids):
            raise SystemExit(
                f"splits.{name}.count={split.get('count')} does not match {len(task_ids)} task_ids"
            )
        declared = split.get("task_ids_sha256")
        computed = canonical_task_ids_sha256(task_ids)
        if declared != computed:
            raise SystemExit(
                f"splits.{name}.task_ids_sha256 mismatch: declared {declared!r}, computed {computed}"
            )
        if len(set(task_ids)) != len(task_ids):
            raise SystemExit(f"splits.{name}.task_ids contains duplicates")


def validate_contract_binding(manifest: Mapping, contract: Mapping) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise SystemExit(
            f"runtime contract schema_version must be {CONTRACT_SCHEMA}, got {contract.get('schema_version')!r}"
        )
    declared = manifest.get("runtime_contract_sha256")
    if declared != contract.get("contract_sha256"):
        raise SystemExit(
            "split manifest and runtime contract disagree: "
            f"manifest declares {declared!r}, contract declares {contract.get('contract_sha256')!r}"
        )


def validate_reachability(reachability: Mapping, split_manifest_path: Path, task_facts_sha256: str) -> None:
    """reachability manifest 必须绑定当前 split manifest 文件与 task facts。"""
    if reachability.get("schema_version") != REACHABILITY_SCHEMA:
        raise SystemExit(
            "reachability manifest schema_version must be "
            f"{REACHABILITY_SCHEMA}, got {reachability.get('schema_version')!r}"
        )
    inputs = reachability.get("inputs", {})
    manifest_file_sha256 = sha256_file(split_manifest_path)
    if inputs.get("split_manifest_sha256") != manifest_file_sha256:
        raise SystemExit(
            "reachability manifest was computed over a different split manifest file: "
            f"expected {manifest_file_sha256}, got {inputs.get('split_manifest_sha256')!r}"
        )
    if inputs.get("task_facts_sha256") != task_facts_sha256:
        raise SystemExit(
            "reachability manifest was computed over different task facts: "
            f"expected {task_facts_sha256}, got {inputs.get('task_facts_sha256')!r}"
        )


def validate_system_prompt(contract: Mapping) -> str:
    """system prompt 必须逐字节等于冻结 contract 记录的版本。"""
    computed = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if contract.get("system_prompt_hash") != computed:
        raise SystemExit(
            "SYSTEM_PROMPT does not match the frozen runtime contract: "
            f"expected {contract.get('system_prompt_hash')!r}, computed {computed}"
        )
    return computed


def validate_contract_versions(contract: Mapping) -> None:
    """runtime contract 必须携带可交叉校验的版本字段（reward/env/tool/system-prompt）。"""
    missing = [
        contract_key
        for _, contract_key in METADATA_CONTRACT_VERSION_FIELDS
        if not contract.get(contract_key)
    ]
    if missing:
        raise SystemExit(
            f"runtime contract is missing required version fields: {missing}; "
            "refusing to build GRPO data against an incomplete contract"
        )
    try:
        runtime_tool_hash = validate_runtime_tool_schema()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if contract.get("tool_schema_hash") != runtime_tool_hash:
        raise SystemExit(
            "runtime contract tool_schema_hash does not match the canonical "
            f"configs/tools.json/Python schema hash: expected {runtime_tool_hash}, "
            f"got {contract.get('tool_schema_hash')!r}"
        )


def infer_sft_source_identity(path: Path) -> str:
    """Resolve one of the four required SFT source roles from its explicit name.

    A bare/ambiguous file is rejected: source identity must survive into the
    metadata rather than being guessed from argument order.
    """
    path = Path(path)
    text = (
        "/".join(path.parts)
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    matches = [role for role in SFT_SOURCE_IDENTITIES if role in text]
    if len(matches) != 1:
        raise SystemExit(
            f"SFT source {path} must have exactly one explicit identity from "
            f"{SFT_SOURCE_IDENTITIES}; found {matches or 'none'}"
        )
    return matches[0]


def validate_split_disjointness(
    manifest: Mapping,
    sft_task_id_files: Sequence[Path],
    *,
    example: bool = False,
    allow_unchecked_sft_leakage: bool = False,
) -> dict:
    """断言 GRPO 池与 Validation/Final/teacher_pool/SFT 互斥；返回泄漏审计块。

    SFT 检查是强制的：正式构建（非 example）必须显式提供 SFT task id 文件；
    唯一例外是显式 ``--i-know-leak-check-is-required``，此时 metadata 会记录醒目
    审计字段并且 ``all_checks_passed=false``，launcher（train_grpo.py）会拒绝该数据集。
    """
    splits = manifest["splits"]
    train = set(splits[TRAIN_SPLIT]["task_ids"])
    validation = set(splits[VALIDATION_SPLIT]["task_ids"])
    pools = {name: set(splits[name]["task_ids"]) for name in FORBIDDEN_SPLITS}

    overlaps = {
        "train_validation": sorted(train & validation),
    }
    for name, pool in pools.items():
        overlaps[f"train_{name}"] = sorted(train & pool)
        overlaps[f"validation_{name}"] = sorted(validation & pool)
    failed = {name: ids for name, ids in overlaps.items() if ids}
    if failed:
        raise SystemExit(
            "GRPO split leakage detected (refusing to build): " + json.dumps(failed, sort_keys=True)
        )

    if sft_task_id_files:
        sft_audit: dict[str, Any] = {
            "checked": True,
            "source_count": len(sft_task_id_files),
            "overlaps": {},
            "sources": {},
        }
    elif example:
        sft_audit = {
            "checked": False,
            "scope": "example",
            "overlaps": {},
        }
    elif allow_unchecked_sft_leakage:
        sft_audit = {
            "checked": False,
            "skipped_via_explicit_override": True,
            "override_flag": SFT_LEAK_CHECK_OVERRIDE_FLAG,
            "overlaps": {},
        }
    else:
        raise SystemExit(
            "SFT leakage check is mandatory for real GRPO builds: pass the four frozen SFT "
            "task id files (process/outcome × train/dev, "
            f"{SFT_TASK_ID_SOURCES_EXPECTED} files) or one manifest containing their task id "
            "sets via --sft-task-ids; refusing to build a dataset whose SFT overlap is "
            f"unchecked. To skip the check anyway, pass {SFT_LEAK_CHECK_OVERRIDE_FLAG} "
            "(recorded prominently in metadata; the GRPO launcher will refuse this dataset)."
        )

    seen_paths: set[Path] = set()
    seen_identities: set[str] = set()
    for path in sft_task_id_files:
        path = Path(path).expanduser().resolve()
        if path in seen_paths:
            raise SystemExit(f"SFT source files must be four different files; duplicate: {path}")
        seen_paths.add(path)
        sft_ids = load_task_id_file(path)
        if not sft_ids:
            raise SystemExit(f"SFT source {path} must contain at least one task id")
        overlap = sorted(train & sft_ids) + sorted(validation & sft_ids)
        # Preserve the strongest diagnostic for a leaked task even when a
        # caller also supplied an ambiguously named source file.
        if overlap:
            raise SystemExit(
                f"GRPO tasks overlap SFT task ids from {path}: {overlap[:10]}"
                f" (+{max(len(overlap) - 10, 0)} more); refusing to build"
            )
        identity = infer_sft_source_identity(path)
        if identity in seen_identities:
            raise SystemExit(
                f"SFT source identity {identity!r} was provided more than once; "
                f"required identities are {SFT_SOURCE_IDENTITIES}"
            )
        seen_identities.add(identity)
        # 保留父目录/文件名键供旧审计报告阅读；规范 identity 另存于 sources。
        source_key = f"{path.parent.name}/{path.name}"
        sft_audit["overlaps"][source_key] = overlap
        sft_hash = sha256_file(path)
        sft_audit.setdefault("sources_sha256", {})[source_key] = sft_hash
        sft_audit["sources"][identity] = {
            "identity": identity,
            # Store a portable provenance label, not a machine-specific absolute path.
            "path": source_key,
            "sha256": sft_hash,
            "task_count": len(sft_ids),
            "task_ids_sha256": canonical_task_ids_sha256(sorted(sft_ids)),
        }
    if sft_task_id_files and (
        len(sft_task_id_files) != SFT_TASK_ID_SOURCES_EXPECTED
        or seen_identities != set(SFT_SOURCE_IDENTITIES)
    ):
        missing = sorted(set(SFT_SOURCE_IDENTITIES) - seen_identities)
        raise SystemExit(
            "formal GRPO builds require exactly four distinct, non-empty SFT sources "
            f"({', '.join(SFT_SOURCE_IDENTITIES)}); got {len(sft_task_id_files)}, "
            f"missing {missing or 'none'}"
        )
    return {
        "train_validation_overlap": overlaps["train_validation"],
        **{f"{name}_overlap": ids for name, ids in overlaps.items() if name != "train_validation"},
        "sft_task_id_check": sft_audit,
        # SFT 检查被显式跳过时整体置 False：launcher 依据它拒绝启动该数据集。
        "all_checks_passed": (
            bool(sft_audit["checked"])
            and len(sft_task_id_files) == SFT_TASK_ID_SOURCES_EXPECTED
            and seen_identities == set(SFT_SOURCE_IDENTITIES)
        ) or example,
    }


def load_task_id_file(path: Path) -> set[int]:
    """读取 SFT task id 文件：JSON 数组或每行一个 task_id 字段的 JSONL。"""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise SystemExit(f"SFT task id file is empty: {path}")
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"SFT task id file {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, list) or not all(isinstance(item, int) for item in data):
            raise SystemExit(f"SFT task id file {path} must be a JSON array of integers")
        if len(set(data)) != len(data):
            raise SystemExit(f"SFT task id file {path} contains duplicate task ids")
        return set(data)
    ids: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"SFT task id file {path} line {line_number} is not valid JSON: {exc}") from exc
        task_id = row.get("task_id") if isinstance(row, dict) else row
        if not isinstance(task_id, int):
            raise SystemExit(f"SFT task id file {path} line {line_number} is missing integer task_id")
        if task_id in ids:
            raise SystemExit(f"SFT task id file {path} contains duplicate task_id {task_id}")
        ids.add(task_id)
    return ids


def build_prompt_messages(query: str) -> list[dict[str, str]]:
    """GRPO 初始对话：冻结 system prompt + 用户 query，无任何隐藏字段。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def assert_no_forbidden_content(row: Mapping, fact_row: Mapping, task_id: int) -> None:
    """隐藏目标字段的值不得进入样本中 user query 之外的任何部分。

    用户 query 本身是可见需求，合法任务里 query 可能提到目标品类词（真实任务
    788 的 query 就包含目标 option），因此扫描范围排除 user 消息；但 system
    prompt、ability、data_source、reward_model、extra_info 必须完全无泄漏。
    """
    system_message = json.dumps(row["prompt"][0], ensure_ascii=False)
    non_prompt = json.dumps(
        {key: value for key, value in row.items() if key != "prompt"},
        ensure_ascii=False,
    )
    haystack = system_message + "\n" + non_prompt
    for field in FORBIDDEN_FACT_FIELDS:
        value = fact_row.get(field)
        candidates: list[str] = []
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [item for item in value if isinstance(item, str)]
        for candidate in candidates:
            if len(candidate) >= FORBIDDEN_SCAN_MIN_CHARS and candidate in haystack:
                raise SystemExit(
                    f"task {task_id}: forbidden fact field {field} leaked into the training sample"
                )


def build_row(
    *,
    task_id: int,
    split_name: str,
    fact_row: Mapping,
    contract_sha256: str,
    data_source: str,
    reward_model: Mapping[str, str],
) -> dict:
    """构建一行 veRL 0.8 兼容样本；字段顺序即 parquet schema 顺序。"""
    query = fact_row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise SystemExit(f"task {task_id}: task facts row is missing a non-empty query")
    row = {
        "task_id": int(task_id),
        "prompt": build_prompt_messages(query),
        "ability": str(fact_row.get("difficulty", "unknown")),
        "data_source": data_source,
        "reward_model": {
            "style": "rule",
            "reward_version": str(reward_model["reward_version"]),
        },
        "extra_info": {
            "task_id": int(task_id),
            "split": str(split_name),
            "contract_sha256": str(contract_sha256),
        },
    }
    if set(row["extra_info"]) != set(EXTRA_INFO_KEYS):
        raise SystemExit(f"task {task_id}: extra_info keys diverged from the frozen whitelist")
    assert_no_forbidden_content(row, fact_row, task_id)
    return row


def default_writer_factory(path: Path):
    """默认 parquet 写入层：只在真正写入时懒加载 pyarrow（本地可能没有）。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            f"pyarrow is unavailable in this environment ({exc}); "
            "install pyarrow or inject writer_factory for tests"
        ) from exc

    schema = pa.schema(
        [
            pa.field("task_id", pa.int64()),
            pa.field(
                "prompt",
                pa.list_(
                    pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
                ),
            ),
            pa.field("ability", pa.string()),
            pa.field("data_source", pa.string()),
            pa.field(
                "reward_model",
                pa.struct(
                    [
                        pa.field("style", pa.string()),
                        pa.field("reward_version", pa.string()),
                    ]
                ),
            ),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("task_id", pa.int64()),
                        pa.field("split", pa.string()),
                        pa.field("contract_sha256", pa.string()),
                    ]
                ),
            ),
        ]
    )
    writer = pq.ParquetWriter(str(path), schema)

    class _BufferedWriter:
        """缓冲全部行后一次性写出，保证同输入的字节确定性。"""

        def __init__(self):
            self.rows: list[dict] = []

        def write(self, row: dict) -> None:
            self.rows.append(row)

        def close(self) -> None:
            table = pa.Table.from_pylist(self.rows, schema=schema)
            writer.write_table(table)
            writer.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self.close()
            return False

    return _BufferedWriter()


def _validate_tokenizer_revision(revision: str | None) -> str:
    if not isinstance(revision, str) or not revision.strip():
        raise SystemExit(
            "tokenizer revision is required for the prompt length audit; pass the frozen "
            f"revision {FROZEN_TOKENIZER_REVISION} explicitly"
        )
    if revision != FROZEN_TOKENIZER_REVISION:
        raise SystemExit(
            "tokenizer revision is not the frozen Qwen revision: "
            f"got {revision!r}, expected {FROZEN_TOKENIZER_REVISION!r}"
        )
    return revision


def default_token_counter(
    tokenizer_name: str, revision: str | None = FROZEN_TOKENIZER_REVISION
) -> TokenCounter:
    """用冻结 tokenizer 估算 prompt 长度；只在远端构建时才懒加载 transformers。"""
    revision = _validate_tokenizer_revision(revision)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"transformers is unavailable ({exc}); pass --tokenizer-name on the GPU host, "
            "or use --skip-length-audit explicitly (recorded in metadata)"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, revision=revision
    )

    def counter(messages: Sequence[Mapping[str, str]]) -> int:
        return sum(
            len(tokenizer.encode(message["content"], add_special_tokens=False))
            for message in messages
        )

    return counter


def resolve_reachability(reachability: Mapping, split_name: str, requested_ids: Sequence[int]) -> tuple[set[int], dict[int, list[str]]]:
    """返回 (可构建 task 集合, 不可达 task 的冻结排除原因)。"""
    split = reachability.get("splits", {}).get(split_name)
    if not isinstance(split, dict):
        raise SystemExit(f"reachability manifest is missing splits.{split_name}")
    requested = set(requested_ids)
    eligible = set(split.get("eligible_task_ids", []))
    unknown = sorted(eligible - requested)
    if unknown:
        raise SystemExit(
            f"reachability manifest lists eligible tasks outside splits.{split_name}: {unknown[:10]}"
        )
    reasons: dict[int, list[str]] = {}
    for entry in split.get("unreachable", []):
        task_id = entry.get("task_id")
        if task_id not in requested:
            raise SystemExit(
                f"reachability manifest lists unreachable task {task_id!r} outside splits.{split_name}"
            )
        reasons[task_id] = [str(reason) for reason in entry.get("reason_codes", ["unreachable"])]
    if set(requested) - eligible != set(reasons):
        raise SystemExit(
            f"reachability manifest for splits.{split_name} is inconsistent with its eligible list"
        )
    return eligible, reasons


def build_split_rows(
    *,
    split_name: str,
    manifest: Mapping,
    reachability: Mapping,
    facts: Mapping[int, Mapping],
    contract: Mapping,
    token_counter: TokenCounter | None,
    max_prompt_tokens: int,
    writer,
) -> dict:
    """构建单个 split 的全部样本行；返回行数/任务统计与审计记录。"""
    requested_ids = list(manifest["splits"][split_name]["task_ids"])
    eligible, unreachable_reasons = resolve_reachability(reachability, split_name, requested_ids)

    contract_sha256 = str(contract["contract_sha256"])
    data_source = str(contract.get("environment_version", "shopsimulator-environment-v2.1"))
    reward_model = {"style": "rule", "reward_version": str(contract.get("reward_version", ""))}
    if not reward_model["reward_version"]:
        raise SystemExit("runtime contract is missing reward_version")

    audit_excluded = []
    written_ids: list[int] = []
    for task_id in requested_ids:
        if task_id not in eligible:
            audit_excluded.append(
                {
                    "split": split_name,
                    "task_id": task_id,
                    "status": "excluded_unreachable",
                    "reasons": unreachable_reasons.get(task_id, ["unreachable"]),
                }
            )
            continue
        fact_row = facts.get(task_id)
        if fact_row is None:
            # task facts 缺行属于证据链断裂，按合同直接失败而不是静默删行。
            raise SystemExit(
                f"task {task_id} is in splits.{split_name} but missing from the frozen task facts"
            )
        row = build_row(
            task_id=task_id,
            split_name=split_name,
            fact_row=fact_row,
            contract_sha256=contract_sha256,
            data_source=data_source,
            reward_model=reward_model,
        )
        if token_counter is not None:
            estimated = token_counter(row["prompt"])
            if estimated > max_prompt_tokens:
                raise SystemExit(
                    f"task {task_id} prompt is {estimated} tokens, above the contract budget {max_prompt_tokens}"
                )
        writer.write(row)
        written_ids.append(int(task_id))

    return {
        "rows": len(written_ids),
        "unique_task_ids": len(set(written_ids)),
        "requested_task_ids_sha256": canonical_task_ids_sha256(requested_ids),
        "included_task_ids_sha256": canonical_task_ids_sha256(written_ids),
        "excluded": audit_excluded,
    }


def build_metadata(
    *,
    manifest: Mapping,
    manifest_file_sha256: str,
    contract: Mapping,
    reachability: Mapping,
    reachability_file_sha256: str,
    task_facts_sha256: str,
    system_prompt_sha256: str,
    split_stats: Mapping[str, Mapping],
    build_parameters: Mapping,
    leakage: Mapping,
    example: bool = False,
) -> dict:
    """构建 metadata.json 内容；不含时间戳与绝对路径，保证重建字节一致。"""
    metadata = {
        "schema_version": METADATA_SCHEMA,
        "example": bool(example),
        "source": {
            "split_manifest_file_sha256": manifest_file_sha256,
            "split_manifest_declared_sha256": manifest.get("manifest_sha256"),
            "runtime_contract_sha256": contract.get("contract_sha256"),
            "task_facts_sha256": task_facts_sha256,
            "task_reachability_manifest_sha256": reachability_file_sha256,
            "task_reachability_declared_sha256": reachability.get("manifest_sha256"),
            "system_prompt_sha256": system_prompt_sha256,
            "tool_version": contract.get("tool_version"),
            "tool_schema_sha256": contract.get("tool_schema_hash"),
            "reward_version": contract.get("reward_version"),
            "environment_version": contract.get("environment_version"),
            "upstream_commit": contract.get("upstream_commit"),
            "tokenizer_revision": build_parameters.get("tokenizer_revision"),
        },
        "build_parameters": dict(build_parameters),
        "files": {
            key: {
                "path": SPLIT_TO_FILENAME[key],
                "rows": split_stats[key]["rows"],
                "unique_task_ids": split_stats[key]["unique_task_ids"],
                "requested_task_ids_sha256": split_stats[key]["requested_task_ids_sha256"],
                "included_task_ids_sha256": split_stats[key]["included_task_ids_sha256"],
            }
            for key in (TRAIN_SPLIT, VALIDATION_SPLIT)
        },
        "leakage": leakage,
        "audit": {
            "excluded_tasks": [
                entry
                for key in (TRAIN_SPLIT, VALIDATION_SPLIT)
                for entry in split_stats[key]["excluded"]
            ],
            "excluded_task_count": sum(
                len(split_stats[key]["excluded"]) for key in (TRAIN_SPLIT, VALIDATION_SPLIT)
            ),
            "excluded_reason_counts": _count_reasons(
                [entry for key in (TRAIN_SPLIT, VALIDATION_SPLIT) for entry in split_stats[key]["excluded"]]
            ),
        },
    }
    if example:
        metadata["_example_note"] = (
            "EXAMPLE ONLY：本文件由 scripts/build_grpo_data.py 在 3–5 个 fake task 的 "
            "注入式测试夹具上生成，仅用于展示 metadata schema；真实 data/grpo/metadata.json "
            "必须在持有冻结 split manifest / task facts 的环境里构建，且真实 parquet 写入"
            "需要安装 pyarrow（见 pyproject.toml 的 grpo extra），并显式提供四个 SFT "
            "task id 文件做泄漏检查。"
        )
    return metadata


def _count_reasons(entries: Sequence[Mapping]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for reason in entry.get("reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def finalize_file_hashes(metadata: dict, output_dir: Path) -> dict:
    """写完 parquet 后把文件 SHA256 回填进 metadata。"""
    for key, entry in metadata["files"].items():
        entry["file_sha256"] = sha256_file(output_dir / entry["path"])
    return metadata


def dump_metadata_bytes(metadata: Mapping) -> bytes:
    """确定性序列化：键排序、固定缩进、结尾换行；重建时字节一致。"""
    return (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_grpo_data(
    *,
    split_manifest_path: Path,
    task_facts_path: Path,
    contract_path: Path | None,
    reachability_path: Path | None,
    output_dir: Path,
    sft_task_id_files: Sequence[Path] = (),
    token_counter: TokenCounter | None = None,
    tokenizer_descriptor: str = "injected",
    tokenizer_revision: str | None = FROZEN_TOKENIZER_REVISION,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    writer_factory: WriterFactory | None = None,
    force: bool = False,
    example: bool = False,
    allow_unchecked_sft_leakage: bool = False,
) -> dict:
    """构建两个 parquet + metadata.json；返回 metadata dict。"""
    split_manifest_path = Path(split_manifest_path)
    task_facts_path = Path(task_facts_path)
    contract_path = Path(contract_path) if contract_path else split_manifest_path.parent / "runtime_contract.json"
    reachability_path = (
        Path(reachability_path) if reachability_path else split_manifest_path.parent / "task_reachability_manifest.json"
    )
    output_dir = Path(output_dir)
    tokenizer_revision = _validate_tokenizer_revision(tokenizer_revision)

    manifest = load_json(split_manifest_path)
    contract = load_json(contract_path)
    reachability = load_json(reachability_path)
    validate_split_manifest(manifest)
    validate_contract_binding(manifest, contract)
    validate_contract_versions(contract)
    validate_system_prompt(contract)
    task_facts_sha256 = sha256_file(task_facts_path)
    if manifest.get("task_source", {}).get("sha256") != task_facts_sha256:
        raise SystemExit(
            "task facts file does not match the frozen split manifest: "
            f"expected {manifest.get('task_source', {}).get('sha256')}, got {task_facts_sha256}"
        )
    validate_reachability(reachability, split_manifest_path, task_facts_sha256)
    leakage = validate_split_disjointness(
        manifest,
        list(sft_task_id_files),
        example=example,
        allow_unchecked_sft_leakage=allow_unchecked_sft_leakage,
    )

    if not force:
        existing = [
            name
            for name in (*SPLIT_TO_FILENAME.values(), "metadata.json")
            if (output_dir / name).exists()
        ]
        if existing:
            raise SystemExit(
                f"refusing to overwrite existing files in {output_dir}: {sorted(existing)}; pass --force to rebuild"
            )

    facts = load_task_facts(task_facts_path, int(manifest.get("task_source", {}).get("rows", -1)))
    manifest_file_sha256 = sha256_file(split_manifest_path)
    reachability_file_sha256 = sha256_file(reachability_path)
    system_prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    writer_factory = writer_factory or default_writer_factory
    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split_name, filename in SPLIT_TO_FILENAME.items():
        with writer_factory(output_dir / filename) as writer:
            split_stats[split_name] = build_split_rows(
                split_name=split_name,
                manifest=manifest,
                reachability=reachability,
                facts=facts,
                contract=contract,
                token_counter=token_counter,
                max_prompt_tokens=max_prompt_tokens,
                writer=writer,
            )

    metadata = build_metadata(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        contract=contract,
        reachability=reachability,
        reachability_file_sha256=reachability_file_sha256,
        task_facts_sha256=task_facts_sha256,
        system_prompt_sha256=system_prompt_sha256,
        split_stats=split_stats,
        build_parameters={
            "splits": [TRAIN_SPLIT, VALIDATION_SPLIT],
            "max_prompt_tokens": int(max_prompt_tokens),
            "prompt_length_audit": "enforced" if token_counter is not None else "skipped",
            "tokenizer": tokenizer_descriptor,
            "tokenizer_revision": tokenizer_revision,
            "writer": getattr(writer_factory, "__name__", "injected"),
            "sft_task_id_sources": [
                f"{Path(path).parent.name}/{Path(path).name}" for path in sft_task_id_files
            ],
            "sft_leak_check": (
                "enforced"
                if sft_task_id_files
                else ("example" if example else "skipped_explicitly")
            ),
        },
        leakage=leakage,
        example=example,
    )
    metadata = finalize_file_hashes(metadata, output_dir)
    (output_dir / "metadata.json").write_bytes(dump_metadata_bytes(metadata))
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True, help="冻结的 split_manifest.json")
    parser.add_argument("--task-facts", type=Path, required=True, help="冻结的 task_facts.jsonl（私有，只读）")
    parser.add_argument(
        "--contract",
        type=Path,
        help="runtime_contract.json；缺省取 split manifest 同目录的同名文件",
    )
    parser.add_argument(
        "--reachability",
        type=Path,
        help="task_reachability_manifest.json；缺省取 split manifest 同目录的同名文件",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/grpo"), help="parquet 与 metadata 输出目录")
    parser.add_argument(
        "--sft-task-ids",
        type=Path,
        nargs="*",
        default=(),
        help="SFT task id 文件（JSONL 或 JSON 数组）；正式构建必须提供四个"
        f"（{SFT_TASK_ID_SOURCES_EXPECTED} 个：Process/Outcome × Train/Dev），"
        "提供时断言与 GRPO 池无交集",
    )
    parser.add_argument(
        "--i-know-leak-check-is-required",
        action="store_true",
        help="显式跳过 SFT 泄漏检查（审计字段写入 metadata 且 all_checks_passed=false，"
        "launcher 会拒绝该数据集；正式构建禁止使用）",
    )
    parser.add_argument(
        "--tokenizer-name",
        help="用于 prompt 长度审计的 tokenizer（GPU 主机传入；本地测试用注入的计数器）",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=FROZEN_TOKENIZER_REVISION,
        help=(
            "tokenizer 的冻结 revision；正式合同固定为 "
            f"{FROZEN_TOKENIZER_REVISION}"
        ),
    )
    parser.add_argument(
        "--skip-length-audit",
        action="store_true",
        help="显式跳过 prompt 长度审计（会记录进 metadata；正式构建不建议）",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"prompt token 预算（默认 {DEFAULT_MAX_PROMPT_TOKENS}，对齐 configs/grpo.yaml data.max_prompt_length）",
    )
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    token_counter = None
    tokenizer_descriptor = "injected"
    if args.tokenizer_name:
        token_counter = default_token_counter(args.tokenizer_name, args.tokenizer_revision)
        tokenizer_descriptor = args.tokenizer_name
    elif not args.skip_length_audit:
        raise SystemExit(
            "prompt length audit requires --tokenizer-name (GPU host) or explicit --skip-length-audit"
        )
    metadata = build_grpo_data(
        split_manifest_path=args.split_manifest,
        task_facts_path=args.task_facts,
        contract_path=args.contract,
        reachability_path=args.reachability,
        output_dir=args.output_dir,
        sft_task_id_files=args.sft_task_ids,
        token_counter=token_counter,
        tokenizer_descriptor=tokenizer_descriptor,
        tokenizer_revision=args.tokenizer_revision,
        max_prompt_tokens=args.max_prompt_tokens,
        force=args.force,
        allow_unchecked_sft_leakage=args.i_know_leak_check_is_required,
    )
    summary = {
        key: {"rows": entry["rows"], "file_sha256": entry["file_sha256"]}
        for key, entry in metadata["files"].items()
    }
    print(
        "GRPO data build complete: "
        + json.dumps(
            {
                "output_dir": str(args.output_dir),
                "metadata_schema": metadata["schema_version"],
                "files": summary,
                "excluded_task_count": metadata["audit"]["excluded_task_count"],
                "sft_leak_check": metadata["build_parameters"]["sft_leak_check"],
                "leakage_all_checks_passed": metadata["leakage"]["all_checks_passed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
