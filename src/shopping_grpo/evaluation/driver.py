"""统一 Student Evaluator driver：一条可恢复、可审计、固定分母的 per-task 流水线。

这里是 WP1 的正式评测主链，把既有 evaluation 包的零件按固定顺序串起来：

    读取并 hash task split
    → Final 模式先 guard blind asset
    → 读取或生成共享 TaskFacts / rubric bundle（按 task 缓存）
    → 加载一个已冻结 actor（M0/M1/M2/M3）
    → reset ShopSimulator 执行统一 tool loop（每题 finally release）
    → 基础设施失败 append-only 记录（judge_status=not_judged，但留在分母）
    → normalize trajectory → deterministic metrics
    → 对合法轨迹调用 Judge；失败置 not_judged
    → assemble per-task evaluation，fsync append 到 evaluations.jsonl
    → 支持从已有 cache resume（schema/hash 不匹配拒绝）
    → expected task IDs 全量断言 + 固定分母 summary

本模块不实现第二套指标、工具协议或 Reward；所有重依赖（环境 client、actor、
judge/curator client、文件路径）都通过构造参数注入，便于用 fake 做无网络测试。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    load_json,
    write_json_atomic,
)
from shopping_grpo.evaluation.blind_guard import (
    guard_blind_final,
    validate_canonical_blind_asset,
)
from shopping_grpo.evaluation.contracts import (
    CONTRACT_VERSION,
    ContractValidationError,
    JUDGE_SCHEMA_VERSION,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.manifest import build_run_manifest, sha256_file
from shopping_grpo.evaluation.metrics import (
    DETERMINISTIC_METRICS_VERSION,
    INFRASTRUCTURE_ERROR_TYPES,
    compute_deterministic_metrics,
    extend_infrastructure_error_types,
)
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_rubric_curator_messages,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    EVALUATION_RESULT_VERSION,
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.rollout import collect_for_task, load_tasks
from shopping_grpo.evaluation.rubric import (
    RUBRIC_EXTRACTOR_VERSION,
    TASK_FACTS_VERSION,
    build_task_facts,
    extract_rubric_candidates,
    materialize_rubric_bundle,
    stable_hash,
)
from shopping_grpo.evaluation.trajectory import (
    NORMALIZED_TRAJECTORY_VERSION,
    normalize_trajectory,
)


DRIVER_VERSION = "shopping-evaluation-driver-v1"
DEV_SPLIT = "dev"
FINAL_SPLIT = "final-200"
SUPPORTED_SPLITS = (DEV_SPLIT, FINAL_SPLIT)
DEFAULT_ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
DEFAULT_BASE_URL = "http://127.0.0.1:5700"
DEFAULT_RUBRIC_VERSION = "wp1-student-eval-rubric-v1"

# not_judged 的 errors.primary 必须落在冻结 taxonomy 内。
INFRASTRUCTURE_INVALID_REASON = "infrastructure_invalid"
JUDGE_FAILURE_REASON = "other"
# cache 记录绑定的 schema：每条 trajectory/normalized/metrics/judge 记录都携带
# task_id + trajectory_id + schema_version + 输入内容 hash，resume 时逐条验证。
CACHE_BINDING_VERSION = "shopping-evaluation-cache-binding-v1"


def _is_infrastructure_exception(exc: BaseException, type_names) -> bool:
    """按统一 taxonomy（metrics.py 唯一定义）分类环境/传输层异常。

    沿 MRO 匹配类名，因此子类（如自定义 OSError 传输错误）同样命中。
    """

    return any(cls.__name__ in type_names for cls in type(exc).__mro__)


class DriverError(RuntimeError):
    """Raised for invalid run configuration or unexpected driver failures."""


class ResumeContractError(DriverError):
    """Raised when resuming a run whose cached artifacts break the contract."""


class BlindFinalGuardError(DriverError):
    """Raised when a split fails the blind-final asset guard."""


class _StageError(Exception):
    """把 per-task 失败与其流水线阶段一起带回 run() 记录到 errors.jsonl。"""

    def __init__(self, stage: str, exc: BaseException):
        super().__init__(str(exc))
        self.stage = stage
        self.exc = exc


def _written_view(record: Mapping) -> dict:
    """JSON 往返后的记录视图；保证写盘时与 resume 读取时 hash 一致。"""

    return json.loads(json.dumps(dict(record), ensure_ascii=False, default=str))


def _record_sha256(record: Mapping) -> str:
    return stable_hash(_written_view(record))


def _judge_input_sha256(normalized: Mapping, metrics: Mapping, rubric: Mapping) -> str:
    """Judge cache 绑定的输入内容 hash：normalized + metrics + rubric。"""

    return stable_hash(
        {
            "deterministic_metrics_sha256": _record_sha256(metrics),
            "normalized_sha256": _record_sha256(normalized),
            "rubric_sha256": _record_sha256(rubric),
        }
    )


def _actor_identity_sha256(actor: Mapping) -> str:
    """Canonical identity used to bind an evaluation to the served actor."""

    fields = (
        "label",
        "model_path",
        "model_revision",
        "served_model_name",
        "weights_sha256",
        "weights_sha256_note",
        "served_model_identity",
    )
    return stable_hash({field: actor.get(field) for field in fields})


def _evaluation_input_sha256(
    *,
    normalized: Mapping,
    metrics: Mapping,
    rubric: Mapping,
    judge: Mapping,
    facts_sha256: str,
    actor_identity_sha256: str,
) -> tuple[str, dict]:
    """Return a combined hash and its individually auditable dependencies."""

    upstream = {
        "normalized_sha256": _record_sha256(normalized),
        "metrics_sha256": _record_sha256(metrics),
        "judge_sha256": _record_sha256(judge),
        "rubric_sha256": _record_sha256(rubric),
        "facts_sha256": str(facts_sha256),
        "actor_identity_sha256": str(actor_identity_sha256),
    }
    return stable_hash(upstream), upstream


def _cache_binding(
    *,
    task_id,
    trajectory_id,
    schema_version,
    input_sha256,
    **extra,
) -> dict:
    binding = {
        "binding_schema": CACHE_BINDING_VERSION,
        "task_id": int(task_id),
        "trajectory_id": str(trajectory_id),
        "schema_version": str(schema_version),
        "input_sha256": str(input_sha256),
    }
    binding.update(extra)
    return binding


def _verify_cache_binding(
    record: Mapping,
    *,
    expected_task_id=None,
    expected_trajectory_id=None,
    expected_schema_version=None,
    expected_input_sha256=None,
    expected_fields=None,
    source,
) -> dict:
    """逐条验证 cache 记录绑定；任何缺失/不匹配都拒绝 resume。"""

    where = f"{source}: task {record.get('task_id')!r}"
    binding = record.get("record_binding")
    if not isinstance(binding, Mapping):
        raise ResumeContractError(f"{where}: cache record has no record_binding")
    if binding.get("binding_schema") != CACHE_BINDING_VERSION:
        raise ResumeContractError(
            f"{where}: unsupported cache binding schema "
            f"{binding.get('binding_schema')!r}"
        )
    expected = {
        "task_id": expected_task_id,
        "trajectory_id": expected_trajectory_id,
        "schema_version": expected_schema_version,
        "input_sha256": expected_input_sha256,
    }
    for field, value in expected.items():
        if value is None:
            continue
        actual = binding.get(field)
        if field in {"task_id"}:
            try:
                matches = actual is not None and int(actual) == int(value)
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = str(actual) == str(value)
        if not matches:
            raise ResumeContractError(
                f"{where}: cache binding {field} mismatch "
                f"(cached={actual!r} expected={value!r})"
            )
    for field, value in (expected_fields or {}).items():
        actual = binding.get(field)
        if isinstance(value, Mapping) and isinstance(actual, Mapping):
            matches = dict(actual) == dict(value)
        else:
            matches = actual == value
        if not matches:
            raise ResumeContractError(
                f"{where}: cache binding {field} mismatch "
                f"(cached={actual!r} expected={value!r})"
            )
    return dict(binding)


def _hash_model_weights(model_path: str | None) -> dict:
    """复用训练侧 hash_weight_files（只 import 不修改）计算权重目录 hash。"""

    if not model_path:
        return {
            "weights_sha256": None,
            "weight_files": None,
            "weights_sha256_note": "no model path provided",
        }
    from shopping_grpo.training.sft.run_manifest import hash_weight_files

    try:
        return hash_weight_files(model_path)
    except OSError as exc:
        raise DriverError(
            f"cannot hash model weights at {model_path!r}: {exc}"
        ) from exc


class LazyEnvironmentTaskFactsSource:
    """默认 task facts 来源：运行期懒加载环境侧冻结 task facts 文档。

    Final/IDs-only 分割不得把 goal/target_product 写进 split 行；facts 在运行期
    从环境侧数据文档取（``SHOP_TASK_FACTS_PATH`` 或显式路径），文档支持两种形态：

    - ``{"schema_version": ..., "facts": [task facts 记录...]}``；
    - ``{"goals": [...], "product_item_dict": {...}}``（经
      ``task_facts_from_environment`` 构建）。

    TODO(远端验收): 远端接入后把该路径指向冻结 ShopSimulator 数据导出。
    """

    SOURCE_SCHEMA = "shopping-task-facts-source-v1"

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else self._resolve_default_path()
        self._facts_by_id: dict[int, dict] | None = None

    @staticmethod
    def _resolve_default_path():
        configured = os.environ.get("SHOP_TASK_FACTS_PATH")
        return Path(configured) if configured else None

    def _load(self) -> dict[int, dict]:
        from shopping_grpo.evaluation.task_facts import task_facts_from_environment

        if self.data_path is None:
            raise DriverError(
                "IDs-only split 需要 task facts，但未配置环境侧 facts 数据："
                "请通过 task_facts_source 注入、--task-facts-path 指定，或设置 "
                "SHOP_TASK_FACTS_PATH 环境变量"
            )
        try:
            document = json.loads(self.data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DriverError(
                f"cannot load environment task facts from {self.data_path}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise DriverError(
                f"environment task facts document must be an object: {self.data_path}"
            )
        if isinstance(document.get("facts"), list):
            rows = {}
            for row in document["facts"]:
                if not isinstance(row, Mapping):
                    raise DriverError("task facts rows must be objects")
                if row.get("schema_version") != TASK_FACTS_VERSION:
                    raise DriverError(
                        "task facts row has unsupported schema_version "
                        f"{row.get('schema_version')!r}"
                    )
                rows[int(row["task_id"])] = dict(row)
            return rows
        if isinstance(document.get("goals"), list) and isinstance(
            document.get("product_item_dict"), Mapping
        ):
            rows = task_facts_from_environment(
                task_ids=range(len(document["goals"])),
                goals=document["goals"],
                product_item_dict=document["product_item_dict"],
            )
            return {int(row["task_id"]): row for row in rows}
        raise DriverError(
            "environment task facts document must contain 'facts' rows or "
            "'goals' + 'product_item_dict'"
        )

    def __call__(self, task_id: int) -> dict:
        if self._facts_by_id is None:
            self._facts_by_id = self._load()
        facts = self._facts_by_id.get(int(task_id))
        if facts is None:
            raise DriverError(
                f"environment task facts source has no facts for task {task_id}"
            )
        return facts


def default_facts_builder(task: Mapping, *, task_facts_source=None) -> dict:
    """默认 TaskFacts 构建：优先读 task 行内嵌字段，IDs-only 行走 facts 来源。

    Dev 合成分割在每一行内嵌 ``goal``（含 instruction_text）与
    ``target_product``；正式 Final 分割是 IDs-only blind input——本函数不会从
    这样的行里寻找（也不允许存在）goal/product，而是通过注入的
    ``task_facts_source``（默认懒加载环境侧 task facts）在运行期取 facts，
    facts 内容不会写进任何 actor 可见输出。
    """

    goal = task.get("goal")
    product = task.get("target_product")
    if not isinstance(product, Mapping):
        product = task.get("product")
    if isinstance(goal, Mapping) and isinstance(product, Mapping):
        query = str(
            goal.get("instruction_text")
            or task.get("instruction")
            or task.get("query")
            or ""
        ).strip()
        if not query:
            raise DriverError(
                f"task {task.get('task_id')!r} 内嵌 goal 缺少 instruction_text，"
                "无法构建 TaskFacts"
            )
        instruction_record = {
            "instruction": query,
            "attributes": goal.get("attributes") or [],
            "instruction_options": goal.get("goal_options") or [],
        }
        return build_task_facts(
            task_id=int(task["task_id"]),
            query=query,
            target_product=product,
            instruction_record=instruction_record,
            reward_goal=goal,
        )
    # IDs-only blind 行：绝不从输入行读取 goal/product。
    source = task_facts_source
    if source is None:
        source = LazyEnvironmentTaskFactsSource()
    facts = source(int(task["task_id"]))
    if not isinstance(facts, Mapping) or facts.get("schema_version") != TASK_FACTS_VERSION:
        raise DriverError(
            f"task_facts_source returned an unsupported TaskFacts record for "
            f"task {task.get('task_id')!r}"
        )
    if int(facts.get("task_id", -1)) != int(task["task_id"]):
        raise DriverError(
            f"task_facts_source returned task_id {facts.get('task_id')!r} for "
            f"task {task.get('task_id')!r}"
        )
    return dict(facts)


def default_env_factory(base_url: str):
    """懒加载真实 ShopSimulator 环境 client，避免评测入口引入训练依赖。"""

    from shopping_grpo.environment.client import ShopAgentEnv

    return ShopAgentEnv(base_url=base_url)


class EvaluationDriver:
    """按固定顺序执行 per-task 评测流水线；全部重依赖均可注入。"""

    def __init__(
        self,
        *,
        run_dir,
        task_split_path,
        split,
        actor_label,
        actor_client,
        judge_client,
        curator_client,
        env_factory=None,
        facts_builder=None,
        task_facts_source=None,
        extra_infrastructure_error_types=None,
        blind_guard=None,
        blind_asset_validator=None,
        base_url=DEFAULT_BASE_URL,
        temperature=0.0,
        top_p=1.0,
        max_steps=35,
        environment_version=DEFAULT_ENVIRONMENT_VERSION,
        model_path=None,
        model_revision=None,
        served_model_name=None,
        served_model_identity=None,
        judge_model="unknown-judge",
        curator_model="unknown-curator",
        rubric_version=DEFAULT_RUBRIC_VERSION,
        actor_protocol=None,
        shared_rubric_cache=None,
        tool_schemas=None,
        resume=False,
        allow_blind_final=False,
        run_id=None,
        created_at=None,
    ):
        if split not in SUPPORTED_SPLITS:
            raise DriverError(f"unsupported split {split!r}; expected {SUPPORTED_SPLITS}")
        if int(max_steps) < 1:
            raise DriverError("max_steps must be positive")
        self.run_dir = Path(run_dir)
        self.task_split_path = Path(task_split_path)
        self.split = split
        self.actor_label = str(actor_label)
        self.actor_client = actor_client
        self.judge_client = judge_client
        self.curator_client = curator_client
        self.env_factory = env_factory or default_env_factory
        # facts_builder 允许为 None：此时走 default_facts_builder + facts source。
        self.facts_builder = facts_builder
        self.task_facts_source = task_facts_source
        # 基础 taxonomy 只在 metrics.py 定义一处；这里只做可注入扩展。
        self.infrastructure_error_types = extend_infrastructure_error_types(
            extra_infrastructure_error_types
        )
        self.blind_guard = blind_guard or guard_blind_final
        self.blind_asset_validator = blind_asset_validator or validate_canonical_blind_asset
        self.base_url = str(base_url)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_steps = int(max_steps)
        self.environment_version = str(environment_version)
        self.model_path = str(model_path) if model_path else None
        self.model_revision = model_revision
        self.served_model_name = str(served_model_name or actor_label)
        if served_model_identity is not None:
            self.served_model_identity = dict(served_model_identity)
        else:
            # served endpoint 身份无法在本地证明时必须显式记录，不许静默省略。
            self.served_model_identity = {
                "proven": False,
                "reason": (
                    "served endpoint identity cannot be proven from the local "
                    "machine; the manifest only records the requested --model-path "
                    "and its weight-file hashes"
                ),
            }
        self.judge_model = str(judge_model)
        self.curator_model = str(curator_model)
        self.rubric_version = str(rubric_version)
        self.actor_protocol = dict(actor_protocol or {})
        self.shared_rubric_cache = Path(shared_rubric_cache) if shared_rubric_cache else None
        self.tool_schemas = list(tool_schemas) if tool_schemas else None
        self.resume = bool(resume)
        self.allow_blind_final = bool(allow_blind_final)
        self.created_at = created_at
        self.run_id = run_id
        # 权重目录 hash 只依赖本地文件；远程 repo / 不存在路径按 hash_weight_files
        # 的语义返回 None + note，manifest 仍显式记录该事实。
        self.weights_identity = _hash_model_weights(self.model_path)
        # 供审计与测试检查最近一次 Judge 输入是否包含模型身份（盲化检查点）。
        self.last_judge_messages = None
        self.blind_asset_report = None

        self.paths = {
            "trajectories": self.run_dir / "trajectories.jsonl",
            "normalized": self.run_dir / "normalized.jsonl",
            "metrics": self.run_dir / "metrics.jsonl",
            "rubrics": self.run_dir / "rubrics.jsonl",
            "judge": self.run_dir / "judge.jsonl",
            "evaluations": self.run_dir / "evaluations.jsonl",
            "errors": self.run_dir / "errors.jsonl",
            "summary": self.run_dir / "summary.json",
            "manifest": self.run_dir / "run_manifest.json",
        }
        self.protocol_hash = self._compute_protocol_hash()

    # ------------------------------------------------------------------
    # 启动阶段：split 读取、hash、blind guard、run 目录准备、cache 加载
    # ------------------------------------------------------------------

    def _load_and_hash_split(self):
        """固定顺序第一步：读取 task split 并计算文件 SHA256。"""

        tasks = load_tasks(self.task_split_path)
        if not tasks:
            raise DriverError(f"task split is empty: {self.task_split_path}")
        expected_ids = [int(task["task_id"]) for task in tasks]
        if len(set(expected_ids)) != len(expected_ids):
            raise DriverError("task split contains duplicate task_ids")
        return tasks, sha256_file(self.task_split_path)

    def _guard_split(self, expected_ids):
        """固定顺序第二步：Final 先过 blind guard；Dev 拒绝触碰 blind asset。

        Final（allowed=True）模式下 guard 绝不提前返回：输入文件会被重算
        SHA256、校验 IDs-only 结构并做隐藏字段扫描；report 记入 run manifest。
        """

        try:
            if self.split == FINAL_SPLIT:
                if not self.allow_blind_final:
                    raise BlindFinalGuardError(
                        "Final-200 需要显式 --allow-blind-final-after-freeze 才能运行"
                    )
                _, frozen_ids = self.blind_asset_validator()
                if set(expected_ids) != set(frozen_ids):
                    raise BlindFinalGuardError(
                        "Final-200 task IDs 与冻结 blind asset 不一致："
                        f"split={len(expected_ids)} frozen={len(frozen_ids)} "
                        f"missing={sorted(set(frozen_ids) - set(expected_ids))[:10]} "
                        f"extra={sorted(set(expected_ids) - set(frozen_ids))[:10]}"
                    )
                reports = self.blind_guard([self.task_split_path], allowed=True)
                if not isinstance(reports, Mapping):
                    raise BlindFinalGuardError(
                        "blind guard did not return per-file reports"
                    )
                report = reports.get(str(self.task_split_path))
                if not isinstance(report, Mapping):
                    raise BlindFinalGuardError(
                        "blind guard returned no report for the task split"
                    )
                # These are security predicates, not informational metadata;
                # injected/custom guards must prove both checks explicitly.
                if report.get("content_matches_declared_asset") is not True:
                    raise BlindFinalGuardError(
                        "blind asset bytes do not match the frozen declaration"
                    )
                if report.get("hidden_field_scan") != "passed":
                    raise BlindFinalGuardError(
                        "blind asset hidden-field scan did not pass"
                    )
                self.blind_asset_report = {
                    str(path): dict(value) if isinstance(value, Mapping) else value
                    for path, value in reports.items()
                }
            else:
                self.blind_guard([self.task_split_path], allowed=False)
        except BlindFinalGuardError:
            raise
        except Exception as exc:
            raise BlindFinalGuardError(f"blind guard 拒绝该分割：{exc}") from exc

    def _compute_protocol_hash(self) -> str:
        """统一协议 hash：四模型必须共享同一 task 顺序、prompt、tool schema 与上限。"""

        schemas = self.tool_schemas
        if schemas is None:
            from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

            schemas = SHOP_TOOL_SCHEMAS
        return stable_hash(
            {
                "driver_version": DRIVER_VERSION,
                "evaluation_contract": CONTRACT_VERSION,
                "environment_version": self.environment_version,
                "tool_schema_hash": stable_hash(schemas),
                "max_steps": self.max_steps,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "rubric_version": self.rubric_version,
                "curator_model": self.curator_model,
                "judge_model": self.judge_model,
                "curator_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
                "judge_prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
                "extractor_version": RUBRIC_EXTRACTOR_VERSION,
                "actor_protocol": self.actor_protocol,
            }
        )

    def _prepare_run_dir(self, task_split_sha: str) -> None:
        """拒绝覆盖现有输出；resume 时校验 manifest 中的合同 hash。"""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.paths.values() if path.exists()]
        if existing:
            if not self.resume:
                raise DriverError(
                    "refusing to overwrite existing outputs; pass resume=True "
                    f"to continue: {[str(path) for path in existing]}"
                )
            if not self.paths["manifest"].exists():
                raise ResumeContractError(
                    "cache files exist without run_manifest.json; refusing to resume"
                )
            manifest = load_json(self.paths["manifest"])
            if manifest.get("task_manifest", {}).get("sha256") != task_split_sha:
                raise ResumeContractError(
                    "task split sha256 does not match the original run manifest"
                )
            if manifest.get("protocol", {}).get("protocol_hash") != self.protocol_hash:
                raise ResumeContractError(
                    "protocol hash does not match the original run manifest"
                )
            # 模型身份块：label / path / revision / weights hash 必须与请求一致。
            actor_block = manifest.get("actor") or {}
            identity_expectations = {
                "label": self.actor_label,
                "model_path": self.model_path,
                "model_revision": self.model_revision,
                "weights_sha256": self.weights_identity.get("weights_sha256"),
                "served_model_name": self.served_model_name,
            }
            for field, expected in identity_expectations.items():
                actual = actor_block.get(field)
                if str(actual) != str(expected):
                    raise ResumeContractError(
                        f"actor {field} does not match the original run manifest "
                        f"(cached={actual!r} requested={expected!r})"
                    )
            if _actor_identity_sha256(actor_block) != _actor_identity_sha256(
                self.actor_metadata
            ):
                raise ResumeContractError(
                    "served actor identity does not match the original run "
                    "manifest"
                )
        elif self.resume:
            # 空 run 目录允许 --resume：等价于全新 run，但不写任何冲突输出。
            pass

    def _facts_source_sha256(self) -> str:
        """Hash the configured facts source identity, including file bytes.

        A rubric is shareable only with the same facts source.  For injected
        callables the stable module/qualname identifies the source; file-backed
        sources additionally include the resolved path and current file hash.
        """

        source = self.task_facts_source
        if self.facts_builder is not None:
            builder = self.facts_builder
            descriptor = {
                "kind": "facts_builder",
                "module": getattr(builder, "__module__", type(builder).__module__),
                "qualname": getattr(
                    builder, "__qualname__", type(builder).__qualname__
                ),
            }
            return stable_hash(descriptor)
        if source is None:
            configured = os.environ.get("SHOP_TASK_FACTS_PATH")
            if configured:
                source = LazyEnvironmentTaskFactsSource(configured)
            else:
                return stable_hash({"kind": "default-task-facts-source"})
        path = getattr(source, "data_path", None)
        if path is not None:
            path = Path(path)
            descriptor = {"kind": "file", "path": str(path.resolve())}
            try:
                descriptor["sha256"] = sha256_file(path)
            except OSError as exc:
                raise DriverError(
                    f"cannot hash task facts source at {path}: {exc}"
                ) from exc
            return stable_hash(descriptor)
        descriptor = {
            "kind": "callable",
            "module": getattr(source, "__module__", type(source).__module__),
            "qualname": getattr(source, "__qualname__", type(source).__qualname__),
        }
        return stable_hash(descriptor)
    def _blind_asset_manifest_block(self) -> dict | None:
        """Final 模式的 blind asset 校验报告；Dev 返回 None。"""

        if self.split != FINAL_SPLIT:
            return None
        report = (self.blind_asset_report or {}).get(str(self.task_split_path))
        if report is None:
            return {
                "input_sha256": None,
                "declared_task_sha256": None,
                "content_matches_declared_asset": False,
                "hidden_field_scan": "not_reported",
                "note": "injected blind guard returned no report",
            }
        return dict(report)

    def _write_initial_manifest(self, expected_ids, task_split_sha) -> None:
        """在第一个 task 开始前固化 run manifest（包含全部输入 hash）。"""

        task_manifest = {
            "path": str(self.task_split_path),
            "sha256": task_split_sha,
            "split": self.split,
            "task_count": len(expected_ids),
        }
        if self.split == DEV_SPLIT:
            # Final 模式只落盘 count + hash，不把 blind task ID 列表写进日志。
            task_manifest["task_ids"] = list(expected_ids)
        else:
            blind_asset = self._blind_asset_manifest_block()
            if blind_asset is not None:
                task_manifest["blind_asset"] = blind_asset
        if self.run_id is None:
            self.run_id = "eval-" + stable_hash(
                {
                    "task_split_sha256": task_split_sha,
                    "protocol_hash": self.protocol_hash,
                    "actor": self.actor_metadata,
                }
            )[:16]
        manifest = build_run_manifest(
            run_id=self.run_id,
            actor=self._manifest_actor_block(),
            task_manifest=task_manifest,
            environment={
                "environment_version": self.environment_version,
                "env_base_url": self.base_url,
            },
            protocol={
                "protocol_hash": self.protocol_hash,
                "max_steps": self.max_steps,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "rubric_version": self.rubric_version,
                "curator_model": self.curator_model,
                "judge_model": self.judge_model,
                **self.actor_protocol,
            },
            code={
                "driver_version": DRIVER_VERSION,
                "driver_module_sha256": _driver_code_hash(),
            },
            judge={
                "model": self.judge_model,
                "curator_model": self.curator_model,
                "rubric_version": self.rubric_version,
            },
            created_at=self.created_at,
        )
        write_json_atomic(self.paths["manifest"], manifest, force=True)

    # ------------------------------------------------------------------
    # cache 加载与 per-task 生成
    # ------------------------------------------------------------------

    def _load_caches(self, tasks) -> dict:
        """加载各 append-only cache；duplicate / unknown task / 绑定不匹配直接拒绝。

        每个 trajectory/normalized/metrics/judge 记录都携带 ``record_binding``
        （task_id + trajectory_id + schema_version + 输入内容 hash）；resume 时
        用当前输入逐条重算并比对，任何缺失或不匹配都拒绝。
        """

        caches = {
            "evaluations": {},
            "trajectories": {},
            "normalized": {},
            "metrics": {},
            "rubrics": {},
            "judges": {},
            "facts": {},
        }
        expected_ids = [int(task["task_id"]) for task in tasks]
        expected_set = set(expected_ids)
        rows_by_id = {int(task["task_id"]): task for task in tasks}
        if self.paths["evaluations"].exists():
            caches["evaluations"] = self._indexed(
                self.paths["evaluations"], key="task_id", allowed_keys=expected_set
            )
            for task_id, record in caches["evaluations"].items():
                if record.get("schema_version") != EVALUATION_RESULT_VERSION:
                    raise ResumeContractError(
                        f"{self.paths['evaluations']}: task {task_id} has unsupported "
                        f"schema {record.get('schema_version')!r}"
                    )
        if self.paths["trajectories"].exists():
            caches["trajectories"] = self._indexed(
                self.paths["trajectories"], key="task_id", allowed_keys=expected_set
            )
            for task_id, record in caches["trajectories"].items():
                _verify_cache_binding(
                    record,
                    expected_task_id=task_id,
                    expected_trajectory_id=record.get("trajectory_id"),
                    expected_schema_version=DRIVER_VERSION,
                    expected_input_sha256=_record_sha256(rows_by_id[int(task_id)]),
                    source=self.paths["trajectories"],
                )
        if self.paths["normalized"].exists():
            caches["normalized"] = self._indexed(
                self.paths["normalized"], key="trajectory_id"
            )
            for trajectory_id, record in caches["normalized"].items():
                source_trajectory = caches["trajectories"].get(
                    int(record.get("task_id", -1))
                )
                if source_trajectory is None:
                    raise ResumeContractError(
                        f"{self.paths['normalized']}: trajectory {trajectory_id} has "
                        "no cached source trajectory to verify its binding against"
                    )
                _verify_cache_binding(
                    record,
                    expected_task_id=record.get("task_id"),
                    expected_trajectory_id=trajectory_id,
                    expected_schema_version=NORMALIZED_TRAJECTORY_VERSION,
                    expected_input_sha256=_record_sha256(source_trajectory),
                    source=self.paths["normalized"],
                )
        if self.paths["metrics"].exists():
            caches["metrics"] = self._indexed(self.paths["metrics"], key="trajectory_id")
            for trajectory_id, record in caches["metrics"].items():
                source_normalized = caches["normalized"].get(trajectory_id)
                if source_normalized is None:
                    raise ResumeContractError(
                        f"{self.paths['metrics']}: trajectory {trajectory_id} has "
                        "no cached normalized trajectory to verify its binding against"
                    )
                _verify_cache_binding(
                    record,
                    expected_task_id=record.get("task_id"),
                    expected_trajectory_id=trajectory_id,
                    expected_schema_version=DETERMINISTIC_METRICS_VERSION,
                    expected_input_sha256=_record_sha256(source_normalized),
                    source=self.paths["metrics"],
                )
        rubric_sources = []
        if self.shared_rubric_cache is not None and self.shared_rubric_cache.exists():
            rubric_sources.append(self.shared_rubric_cache)
        if self.paths["rubrics"].exists():
            rubric_sources.append(self.paths["rubrics"])
        for source in rubric_sources:
            bundles = self._indexed(source, key="task_id", allowed_keys=expected_set)
            for task_id, bundle in bundles.items():
                try:
                    validate_rubric_bundle(bundle, expected_task_id=task_id)
                except ContractValidationError as exc:
                    raise ResumeContractError(
                        f"{source}: cached rubric for task {task_id} is invalid: {exc}"
                    ) from exc
                # Rebuild facts from the current source before accepting a
                # rubric cache.  This catches source-file and builder drift,
                # even when the rubric's query happens to remain unchanged.
                facts = caches["facts"].get(task_id)
                if facts is None:
                    facts = self._facts_for_task(rows_by_id[task_id])
                    caches["facts"][task_id] = facts
                facts_sha = _record_sha256(facts)
                source_sha = self._facts_source_sha256()
                _verify_cache_binding(
                    bundle,
                    expected_task_id=task_id,
                    expected_trajectory_id="",
                    expected_schema_version=bundle.get("schema_version"),
                    expected_input_sha256=facts_sha,
                    expected_fields={
                        "facts_sha256": facts_sha,
                        "facts_source_sha256": source_sha,
                    },
                    source=source,
                )
            caches["rubrics"].update(bundles)
        if self.paths["judge"].exists():
            # Judge cache 以 trajectory_id 为唯一键：同一 trajectory 的第二条
            # 记录（含非法 fallback 追加）在这里直接被拒绝。
            caches["judges"] = self._indexed(self.paths["judge"], key="trajectory_id")
            for trajectory_id, record in caches["judges"].items():
                task_id = int(record.get("task_id", -1))
                source_normalized = caches["normalized"].get(trajectory_id)
                source_metrics = caches["metrics"].get(trajectory_id)
                rubric = caches["rubrics"].get(task_id)
                if source_normalized is None or source_metrics is None or rubric is None:
                    raise ResumeContractError(
                        f"{self.paths['judge']}: trajectory {trajectory_id} is missing "
                        "its normalized/metrics/rubric inputs; cannot verify binding"
                    )
                _verify_cache_binding(
                    record,
                    expected_task_id=task_id,
                    expected_trajectory_id=trajectory_id,
                    expected_schema_version=JUDGE_SCHEMA_VERSION,
                    expected_input_sha256=_judge_input_sha256(
                        source_normalized, source_metrics, rubric
                    ),
                    source=self.paths["judge"],
                )
        # Evaluations are the cache-hit surface of run().  Validate them only
        # after every upstream cache has been loaded so their complete binding
        # can be recomputed from normalized/metrics/judge/rubric/facts and the
        # requested actor identity.
        for task_id, record in caches["evaluations"].items():
            trajectory_id = str(record.get("trajectory_id") or "")
            normalized = caches["normalized"].get(trajectory_id)
            metrics = caches["metrics"].get(trajectory_id)
            judge = caches["judges"].get(trajectory_id)
            rubric = caches["rubrics"].get(int(task_id))
            facts = caches["facts"].get(int(task_id))
            if any(item is None for item in (normalized, metrics, judge, rubric, facts)):
                raise ResumeContractError(
                    f"{self.paths['evaluations']}: task {task_id} is missing "
                    "an upstream cache required for binding verification"
                )
            facts_sha = _record_sha256(facts)
            actor_sha = _actor_identity_sha256(self.actor_metadata)
            input_sha, upstream = _evaluation_input_sha256(
                normalized=normalized,
                metrics=metrics,
                rubric=rubric,
                judge=judge,
                facts_sha256=facts_sha,
                actor_identity_sha256=actor_sha,
            )
            if int(normalized.get("task_id", -1)) != int(task_id):
                raise ResumeContractError(
                    f"{self.paths['evaluations']}: task {task_id} trajectory task mismatch"
                )
            _verify_cache_binding(
                record,
                expected_task_id=task_id,
                expected_trajectory_id=trajectory_id,
                expected_schema_version=EVALUATION_RESULT_VERSION,
                expected_input_sha256=input_sha,
                expected_fields={"upstream": upstream, **upstream},
                source=self.paths["evaluations"],
            )
            cached_actor = record.get("actor")
            if not isinstance(cached_actor, Mapping) or _actor_identity_sha256(
                cached_actor
            ) != actor_sha:
                raise ResumeContractError(
                    f"{self.paths['evaluations']}: task {task_id} actor/model "
                    "identity does not match the requested actor"
                )
            try:
                expected_evaluation = assemble_task_evaluation(
                    actor=self.actor_metadata,
                    normalized_trajectory=normalized,
                    deterministic_metrics=metrics,
                    rubric_bundle=rubric,
                    judge_result=judge,
                )
            except Exception as exc:
                raise ResumeContractError(
                    f"{self.paths['evaluations']}: task {task_id} upstream "
                    f"artifacts cannot reconstruct evaluation: {exc}"
                ) from exc
            cached_without_binding = dict(record)
            cached_without_binding.pop("record_binding", None)
            if _written_view(cached_without_binding) != _written_view(
                expected_evaluation
            ):
                raise ResumeContractError(
                    f"{self.paths['evaluations']}: task {task_id} evaluation "
                    "payload does not match its bound upstream artifacts"
                )
        return caches

    @staticmethod
    def _indexed(path: Path, *, key: str, allowed_keys=None) -> dict:
        try:
            return index_jsonl(path, key=key, allowed_keys=allowed_keys)
        except Exception as exc:  # ArtifactError / JSON / duplicate / unexpected
            raise ResumeContractError(f"{path}: {exc}") from exc

    def _facts_for_task(self, task: Mapping) -> dict:
        """解析 TaskFacts：注入 builder 优先；否则默认 builder + facts source。

        facts 内容只在 rubric/curator 侧使用，不进入任何 actor 可见输出。
        """

        if self.facts_builder is not None:
            return self.facts_builder(task)
        return default_facts_builder(task, task_facts_source=self.task_facts_source)

    def _rubric_for_task(self, task: Mapping, caches: dict) -> dict:
        """读取或生成共享 rubric bundle；按 task 缓存（含可选共享 cache 文件）。"""

        task_id = int(task["task_id"])
        facts = caches["facts"].get(task_id)
        if facts is None:
            facts = self._facts_for_task(task)
            caches["facts"][task_id] = facts
        facts_sha = _record_sha256(facts)
        source_sha = self._facts_source_sha256()
        cached = caches["rubrics"].get(task_id)
        if cached is not None:
            validate_rubric_bundle(cached, expected_task_id=task_id)
            _verify_cache_binding(
                cached,
                expected_task_id=task_id,
                expected_trajectory_id="",
                expected_schema_version=cached.get("schema_version"),
                expected_input_sha256=facts_sha,
                expected_fields={
                    "facts_sha256": facts_sha,
                    "facts_source_sha256": source_sha,
                },
                source=self.paths["rubrics"],
            )
            return cached
        candidates = extract_rubric_candidates(facts)
        messages = build_rubric_curator_messages(
            task_id=task_id,
            query=candidates["query"],
            candidates=candidates["candidates"],
        )
        response = self.curator_client.complete_json(messages)["result"]
        bundle = materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=response,
            curator_model=self.curator_model,
            curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
            rubric_version=self.rubric_version,
        )
        bundle["record_binding"] = _cache_binding(
            task_id=task_id,
            trajectory_id="",
            schema_version=bundle["schema_version"],
            input_sha256=facts_sha,
            facts_sha256=facts_sha,
            facts_source_sha256=source_sha,
        )
        append_jsonl_fsync(self.paths["rubrics"], bundle)
        caches["rubrics"][task_id] = bundle
        return bundle

    def _trajectory_for_task(self, task: Mapping, caches: dict):
        """加载 actor 并执行统一 tool loop；cache 命中时直接复用原始轨迹。"""

        task_id = int(task["task_id"])
        cached = caches["trajectories"].get(task_id)
        if cached is not None:
            return cached, True
        trajectory = collect_for_task(
            task,
            client=self.actor_client,
            env_factory=self.env_factory,
            base_url=self.base_url,
            max_steps=self.max_steps,
            tools=self.tool_schemas,
            attempt_index=0,
        )
        # 绑定 task 输入内容 hash：resume 时可发现 split 行内容被篡改的缓存。
        trajectory["record_binding"] = _cache_binding(
            task_id=task_id,
            trajectory_id=str(trajectory.get("trajectory_id") or ""),
            schema_version=DRIVER_VERSION,
            input_sha256=_record_sha256(task),
        )
        append_jsonl_fsync(self.paths["trajectories"], trajectory)
        caches["trajectories"][task_id] = trajectory
        return trajectory, False

    def _validated_judge(self, judge: Mapping, *, rubric, normalized: Mapping):
        """校验一条（新鲜或缓存的）Judge 结果；非法 schema 返回 None。"""

        event_ids = [
            event["event_id"]
            for event in normalized.get("events") or []
            if isinstance(event, Mapping) and event.get("event_id")
        ]
        try:
            return validate_judge_result(
                judge,
                rubric_ids=[item["rubric_id"] for item in rubric["rubrics"]],
                expected_task_id=int(normalized["task_id"]),
                expected_trajectory_id=str(normalized["trajectory_id"]),
                allowed_event_ids=event_ids,
            )
        except (ContractValidationError, KeyError, TypeError, ValueError):
            return None

    def _judge_for_task(self, normalized: Mapping, metrics: Mapping, rubric, caches: dict):
        """对合法轨迹调用 Judge；失败/非法 schema 一律置 judge_status=not_judged。"""

        task_id = int(normalized["task_id"])
        trajectory_id = str(normalized["trajectory_id"])
        judge_input_sha = _judge_input_sha256(normalized, metrics, rubric)

        def _bind(record: dict) -> dict:
            record["record_binding"] = _cache_binding(
                task_id=task_id,
                trajectory_id=trajectory_id,
                schema_version=JUDGE_SCHEMA_VERSION,
                input_sha256=judge_input_sha,
            )
            return record

        eligible = bool((metrics.get("validity") or {}).get("judge_eligible"))
        if not eligible:
            judge = build_not_judged_result(
                task_id=task_id,
                trajectory_id=trajectory_id,
                reason=INFRASTRUCTURE_INVALID_REASON,
            )
            self._record_error(
                task_id=task_id,
                trajectory_id=trajectory_id,
                stage="trajectory_validity",
                message="轨迹基础设施无效，Judge 未调用；任务保留在固定分母",
            )
            if trajectory_id not in caches["judges"]:
                append_jsonl_fsync(self.paths["judge"], _bind(judge))
                caches["judges"][trajectory_id] = judge
            return judge
        cached = caches["judges"].get(trajectory_id)
        if cached is not None:
            validated = self._validated_judge(cached, rubric=rubric, normalized=normalized)
            if validated is not None:
                return validated
            self._record_error(
                task_id=task_id,
                trajectory_id=trajectory_id,
                stage="judge_cache",
                message="缓存的 Judge 结果不符合合同；按 not_judged 处理",
            )
            # 非法 Judge cache 的 fallback 只在内存中置 not_judged；append-only
            # 的 judge.jsonl 以 trajectory_id 为唯一键，同一 trajectory 绝不追加
            # 第二条记录。
            judge = build_not_judged_result(
                task_id=task_id,
                trajectory_id=trajectory_id,
                reason=JUDGE_FAILURE_REASON,
            )
            caches["judges"][trajectory_id] = _bind(judge)
            return judge
        messages = build_trajectory_judge_messages(
            normalized=normalized,
            rubric_bundle=rubric,
            deterministic_metrics=metrics,
        )
        # 盲化检查点：Judge 输入只含 Actor 可见轨迹与 Rubric，不含模型身份。
        self.last_judge_messages = messages
        try:
            result = self.judge_client.complete_json(messages)["result"]
            judge = self._validated_judge(result, rubric=rubric, normalized=normalized)
            if judge is None:
                raise ContractValidationError(
                    "judge result violates the frozen schema"
                )
        except Exception as exc:
            self._record_error(
                task_id=task_id,
                trajectory_id=trajectory_id,
                stage="judge",
                message=f"Judge 调用或校验失败：{type(exc).__name__}: {exc}",
            )
            judge = build_not_judged_result(
                task_id=task_id,
                trajectory_id=trajectory_id,
                reason=JUDGE_FAILURE_REASON,
            )
        append_jsonl_fsync(self.paths["judge"], _bind(judge))
        caches["judges"][trajectory_id] = judge
        return judge

    def _evaluate_task(self, task: Mapping, caches: dict) -> dict:
        """固定顺序处理单个 task，返回唯一一条最终 per-task evaluation。"""

        task_id = int(task["task_id"])
        try:
            rubric = self._rubric_for_task(task, caches)
        except DriverError:
            raise
        except Exception as exc:
            raise _StageError("rubric", exc) from exc
        trajectory, _from_cache = self._trajectory_for_task(task, caches)
        trajectory_id = str(trajectory.get("trajectory_id") or "")
        normalized = normalize_trajectory(trajectory)
        if normalized["trajectory_id"] not in caches["normalized"]:
            normalized["record_binding"] = _cache_binding(
                task_id=task_id,
                trajectory_id=trajectory_id,
                schema_version=NORMALIZED_TRAJECTORY_VERSION,
                input_sha256=_record_sha256(trajectory),
            )
            append_jsonl_fsync(self.paths["normalized"], normalized)
            caches["normalized"][normalized["trajectory_id"]] = normalized
        metrics = compute_deterministic_metrics(
            normalized,
            infrastructure_error_types=self.infrastructure_error_types,
        )
        if normalized["trajectory_id"] not in caches["metrics"]:
            metrics["record_binding"] = _cache_binding(
                task_id=task_id,
                trajectory_id=trajectory_id,
                schema_version=DETERMINISTIC_METRICS_VERSION,
                input_sha256=_record_sha256(normalized),
            )
            append_jsonl_fsync(self.paths["metrics"], metrics)
            caches["metrics"][normalized["trajectory_id"]] = metrics
        judge = self._judge_for_task(normalized, metrics, rubric, caches)
        evaluation = assemble_task_evaluation(
            actor=self.actor_metadata,
            normalized_trajectory=normalized,
            deterministic_metrics=metrics,
            rubric_bundle=rubric,
            judge_result=judge,
        )
        facts = caches["facts"].get(task_id)
        if facts is None:
            # _rubric_for_task normally populated this; retain an explicit
            # invariant so an injected implementation cannot produce an
            # unbound evaluation.
            raise DriverError(f"task {task_id}: missing TaskFacts for evaluation binding")
        facts_sha = _record_sha256(facts)
        actor_sha = _actor_identity_sha256(self.actor_metadata)
        evaluation_input_sha, upstream = _evaluation_input_sha256(
            normalized=normalized,
            metrics=metrics,
            rubric=rubric,
            judge=judge,
            facts_sha256=facts_sha,
            actor_identity_sha256=actor_sha,
        )
        evaluation["record_binding"] = _cache_binding(
            task_id=task_id,
            trajectory_id=trajectory_id,
            schema_version=EVALUATION_RESULT_VERSION,
            input_sha256=evaluation_input_sha,
            upstream=upstream,
            **upstream,
        )
        append_jsonl_fsync(self.paths["evaluations"], evaluation)
        caches["evaluations"][task_id] = evaluation
        return evaluation

    def _record_error(self, *, task_id, trajectory_id, stage, message) -> None:
        """基础设施/合同失败的 append-only 审计记录，不影响固定分母。"""

        append_jsonl_fsync(
            self.paths["errors"],
            {
                "task_id": int(task_id),
                "trajectory_id": trajectory_id,
                "stage": stage,
                "error_type": "evaluation_error",
                "message": str(message),
                "created_at": _now(),
            },
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """执行完整 run；返回 summary 与 run 级审计信息。"""

        tasks, task_split_sha = self._load_and_hash_split()
        expected_ids = [int(task["task_id"]) for task in tasks]
        self._guard_split(expected_ids)
        self._prepare_run_dir(task_split_sha)
        self._write_initial_manifest(expected_ids, task_split_sha)
        # errors.jsonl 是 append-only 审计文件，从头保证存在（可为空）。
        self.paths["errors"].touch(exist_ok=True)
        caches = self._load_caches(tasks)

        evaluations = []
        cached_tasks = 0
        recorded_errors = 0
        for task in tasks:
            task_id = int(task["task_id"])
            cached = caches["evaluations"].get(task_id)
            if cached is not None:
                evaluations.append(cached)
                cached_tasks += 1
                continue
            try:
                evaluations.append(self._evaluate_task(task, caches))
            except DriverError:
                # env_factory 等运行环境不可用：立即失败，避免整批任务被污染。
                raise
            except _StageError as exc:
                if _is_infrastructure_exception(
                    exc.exc, self.infrastructure_error_types
                ):
                    raise DriverError(
                        f"environment infrastructure failure at task {task_id}: "
                        f"{type(exc.exc).__name__}: {exc.exc}"
                    ) from exc.exc
                self._record_error(
                    task_id=task_id,
                    trajectory_id=None,
                    stage=exc.stage,
                    message=f"任务处理失败：{type(exc.exc).__name__}: {exc.exc}",
                )
                recorded_errors += 1
            except Exception as exc:
                if _is_infrastructure_exception(exc, self.infrastructure_error_types):
                    raise DriverError(
                        f"environment infrastructure failure at task {task_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                self._record_error(
                    task_id=task_id,
                    trajectory_id=None,
                    stage="task",
                    message=f"任务处理失败：{type(exc).__name__}: {exc}",
                )
                recorded_errors += 1

        summary = self._build_summary(evaluations, expected_ids, task_split_sha,
                                      cached_tasks, recorded_errors)
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "summary": summary,
            "cached_tasks": cached_tasks,
            "recorded_errors": recorded_errors,
        }

    def _assert_expected_coverage(self, evaluations, expected_ids):
        """expected task IDs 全量断言：不重不漏，缺题留在分母。"""

        seen = [int(record["task_id"]) for record in evaluations]
        if len(seen) != len(set(seen)):
            raise DriverError("internal error: duplicate evaluations for a task")
        unexpected = sorted(set(seen) - set(expected_ids))
        if unexpected:
            raise DriverError(f"unexpected task_ids in evaluations: {unexpected}")
        return sorted(set(expected_ids) - set(seen))

    def _build_summary(self, evaluations, expected_ids, task_split_sha,
                       cached_tasks, recorded_errors) -> dict:
        missing = self._assert_expected_coverage(evaluations, expected_ids)
        summary = summarize_evaluations(
            expected_task_ids=expected_ids,
            evaluations=evaluations,
        )
        # GAPS §5.4 要求的字段别名与 run 级审计信息；不引入任何 composite score。
        summary["completed_tasks"] = summary["completed_evaluations"]
        summary["not_judged_tasks"] = summary["trajectory_quality"][
            "judge_status_counts"
        ].get("not_judged", 0)
        summary["run"] = {
            "run_id": self.run_id,
            "driver_version": DRIVER_VERSION,
            "split": self.split,
            "actor_label": self.actor_label,
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "weights_sha256": self.weights_identity.get("weights_sha256"),
            "served_model_identity": dict(self.served_model_identity),
            "environment_version": self.environment_version,
            "task_split_path": str(self.task_split_path),
            "task_split_sha256": task_split_sha,
            "protocol_hash": self.protocol_hash,
            "cached_tasks": cached_tasks,
            "recorded_errors": recorded_errors,
        }
        write_json_atomic(self.paths["summary"], summary, force=True)
        outputs = {
            name: _count_rows(self.paths[name])
            for name in (
                "trajectories",
                "normalized",
                "metrics",
                "rubrics",
                "judge",
                "evaluations",
                "errors",
            )
        }
        outputs["summary"] = 1
        final_task_manifest = {
            "path": str(self.task_split_path),
            "sha256": task_split_sha,
            "split": self.split,
            "task_count": len(expected_ids),
            **(
                {"task_ids": list(expected_ids)}
                if self.split == DEV_SPLIT
                else {}
            ),
        }
        if self.split == FINAL_SPLIT:
            blind_asset = self._blind_asset_manifest_block()
            if blind_asset is not None:
                final_task_manifest["blind_asset"] = blind_asset
        manifest = build_run_manifest(
            run_id=self.run_id,
            actor=self._manifest_actor_block(),
            task_manifest=final_task_manifest,
            environment={
                "environment_version": self.environment_version,
                "env_base_url": self.base_url,
            },
            protocol={
                "protocol_hash": self.protocol_hash,
                "max_steps": self.max_steps,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "rubric_version": self.rubric_version,
                "curator_model": self.curator_model,
                "judge_model": self.judge_model,
                **self.actor_protocol,
            },
            code={
                "driver_version": DRIVER_VERSION,
                "driver_module_sha256": _driver_code_hash(),
            },
            judge={
                "model": self.judge_model,
                "curator_model": self.curator_model,
                "rubric_version": self.rubric_version,
            },
            outputs=outputs,
            created_at=self.created_at,
        )
        write_json_atomic(self.paths["manifest"], manifest, force=True)
        return summary

    @property
    def actor_metadata(self) -> dict:
        """Actor 外层元数据；只进 evaluation/manifest，不进 Judge 输入。"""

        return {
            "label": self.actor_label,
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "served_model_name": self.served_model_name,
            "weights_sha256": self.weights_identity.get("weights_sha256"),
            "weights_sha256_note": self.weights_identity.get("weights_sha256_note"),
            "served_model_identity": dict(self.served_model_identity),
        }

    def _manifest_actor_block(self) -> dict:
        """manifest 专用 actor 块：metadata + 完整权重文件 hash 清单。"""

        block = self.actor_metadata
        block["weights"] = dict(self.weights_identity)
        return block


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _driver_code_hash() -> str | None:
    try:
        return sha256_file(Path(__file__))
    except OSError:
        return None
