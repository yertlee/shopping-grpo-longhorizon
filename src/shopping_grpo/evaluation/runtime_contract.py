"""冻结 runtime contract 的集中式加载与验证（协议身份的权威来源）。

本模块只依赖 stdlib。合同文件、canonical 口径与冻结值由项目侧生成器
（commerce-agent-posttrain/src/commerce_posttrain/contracts/runtime.py）
定义；这里复现其算法但不引入对项目仓库的运行时依赖。

三种 hash 口径（指令书 §2）：
- ``declared_sha256``  ：JSON 里的 ``contract_sha256`` 字段；
- ``canonical_sha256``：删除该字段后按 canonical JSON（ensure_ascii=False、
  sort_keys=True、separators=(",", ":")）算出的 SHA256；
- ``file_sha256``    ：合同文件原始物理字节 SHA256（绝对路径不参与语义身份，
  物理身份由它单独记录）。

加载 fail closed：任何异常（缺文件、坏 JSON、字段缺失、类型错误、hash 格式
错误、自校验失败、偏离冻结值）都立即抛出明确异常，绝不返回部分结果。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

# 冻结合同身份（指令书 §2）：declared == canonical == 该值。
FROZEN_CONTRACT_SHA256 = "73855a719bc303fba3821cf215d8503b0cdef7bfc960a5d004d61ac320f11855"
FROZEN_SCHEMA_VERSION = "commerce-runtime-contract-v1"

# 必需字段：缺失即失败（指令书 §3）。
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "environment_version",
        "reward_version",
        "observation_version",
        "tool_version",
        "tool_schema_hash",
        "system_prompt_hash",
        "observation_projection_hash",
        "observation_projection_contract",
        "attempts_per_task",
        "max_steps",
        "context_window",
        "context_safety_margin",
        "max_generated_tokens_per_turn",
        "observation_search_tokens",
        "observation_detail_tokens",
        "observation_generic_tokens",
        "observation_search_top_k",
        "contract_sha256",
    }
)

# 冻结字段值（指令书 §3）：与冻结合同文件一致，偏离即失败。
FROZEN_FIELD_VALUES = {
    "schema_version": "commerce-runtime-contract-v1",
    "environment_version": "shopsimulator-environment-v2.1",
    "reward_version": "shopsimulator-reward-v3",
    "observation_version": "shopping-observation-v2",
    "observation_projection_contract": "shopping-observation-v2",
    "tool_version": "shopping-tools-v2",
    "attempts_per_task": 3,
    "max_steps": 35,
    "context_window": 24576,
    "context_safety_margin": 512,
    "max_generated_tokens_per_turn": 512,
    "observation_search_tokens": 1536,
    "observation_detail_tokens": 4096,
    "observation_generic_tokens": 768,
    "observation_search_top_k": 20,
}

_HEX64 = frozenset("0123456789abcdef")


class RuntimeContractError(ValueError):
    """合同加载/验证失败；信息必须可读、可定位。"""


def _freeze_runtime_value(value: Any) -> Any:
    """Recursively freeze a JSON-shaped value.

    ``dataclass(frozen=True)`` only protects the outer object.  Contract
    payloads are nested, so dictionaries/lists/sets must be replaced by
    immutable equivalents at the validation boundary.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_runtime_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_runtime_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_runtime_value(item) for item in value)
    return value


def thaw_runtime_value(value: Any) -> Any:
    """Return a stable, ordinary JSON-shaped copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: thaw_runtime_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_runtime_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [thaw_runtime_value(item) for item in value]
        return sorted(values, key=lambda item: canonical_json_bytes(item))
    return value


@dataclass(frozen=True)
class ValidatedRuntimeContract:
    """已验证的冻结合同；EvaluationDriver 只接受这个对象，不接受任意 dict。

    ``__post_init__`` 强制自校验（declared == canonical == 冻结值、payload
    重算一致、必需字段与冻结值齐全），并把手写构造的路径也堵死——手工构造
    一个不自洽的对象会在构造时直接抛错。内部 dict 以只读视图保存，验证后
    任何嵌套字段都不可修改。
    """

    contract: Mapping[str, Any]
    canonical_payload: Mapping[str, Any]
    declared_sha256: str
    canonical_sha256: str
    file_sha256: str
    _source_path: Path | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, Mapping):
            raise RuntimeContractError(
                "ValidatedRuntimeContract.contract 必须是 Mapping，"
                f"got {type(self.contract).__name__}"
            )
        contract = thaw_runtime_value(self.contract)
        if not isinstance(self.canonical_payload, Mapping):
            raise RuntimeContractError(
                "ValidatedRuntimeContract.canonical_payload 必须是 Mapping"
            )
        missing = sorted(REQUIRED_FIELDS - set(contract))
        if missing:
            raise RuntimeContractError(
                f"ValidatedRuntimeContract 缺少必需字段：{missing}"
            )
        _validate_declared_hash(contract.get("contract_sha256"))
        _validate_field_types(contract)
        if self.declared_sha256 != contract.get("contract_sha256"):
            raise RuntimeContractError(
                "ValidatedRuntimeContract.declared_sha256 与 contract 字段不一致"
            )
        recomputed = compute_runtime_contract_canonical_sha256(contract)
        if recomputed != self.declared_sha256:
            raise RuntimeContractError(
                "ValidatedRuntimeContract canonical 重算与 declared 不一致："
                f"declared={self.declared_sha256} recomputed={recomputed}"
            )
        if self.declared_sha256 != FROZEN_CONTRACT_SHA256:
            raise RuntimeContractError(
                f"ValidatedRuntimeContract 偏离冻结身份：{self.declared_sha256}"
            )
        if self.canonical_sha256 != self.declared_sha256:
            raise RuntimeContractError(
                "ValidatedRuntimeContract canonical_sha256 与 declared 不一致"
            )
        expected_payload = canonical_contract_payload(contract)
        supplied_payload = thaw_runtime_value(self.canonical_payload)
        if supplied_payload != expected_payload:
            raise RuntimeContractError(
                "ValidatedRuntimeContract.canonical_payload 与 contract 不一致"
            )
        for name in ("declared_sha256", "canonical_sha256", "file_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in _HEX64 for char in value)
            ):
                raise RuntimeContractError(
                    f"ValidatedRuntimeContract.{name} 必须是 64 位小写十六进制"
                )
        # 验证后不可变：递归冻结 dict/list/set，任何后续修改都会失败。
        frozen_contract = _freeze_runtime_value(contract)
        frozen_payload = _freeze_runtime_value(canonical_contract_payload(contract))
        object.__setattr__(self, "contract", frozen_contract)
        object.__setattr__(
            self,
            "canonical_payload",
            frozen_payload,
        )


def verify_runtime_code_against_contract(
    contract: ValidatedRuntimeContract,
    *,
    tool_config_path: str | Path | None = None,
) -> dict:
    """指令书 §4：核对实际运行代码与冻结合同一致；任何漂移立即抛错。

    校验项：SYSTEM_PROMPT 原文、Python tool schema、configs/tools.json 与
    Python schema 一致性、projection contract 版本、projection 原始字节、
    reward 版本。返回各检查项的实测 hash（供审计）。
    """
    import hashlib

    from shopping_grpo.environment import projection as projection_module
    from shopping_grpo.environment.projection import PROJECTION_CONTRACT_VERSION
    from shopping_grpo.environment.tools import (
        SHOP_TOOL_SCHEMAS,
        validate_runtime_tool_schema,
    )
    from shopping_grpo.evaluation.metrics import REWARD_V3
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT

    c = contract.contract
    checks: dict = {}

    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    checks["system_prompt_sha256"] = prompt_hash
    if prompt_hash != c["system_prompt_hash"]:
        raise RuntimeContractError(
            "SYSTEM_PROMPT 原文与冻结合同不一致："
            f"code={prompt_hash} contract={c['system_prompt_hash']}"
        )

    schema_hash = hashlib.sha256(
        canonical_json_bytes(SHOP_TOOL_SCHEMAS)
    ).hexdigest()
    checks["tool_schema_sha256"] = schema_hash
    if schema_hash != c["tool_schema_hash"]:
        raise RuntimeContractError(
            "Python SHOP_TOOL_SCHEMAS 与冻结合同不一致："
            f"code={schema_hash} contract={c['tool_schema_hash']}"
        )

    try:
        config_hash = validate_runtime_tool_schema(tool_config_path)
    except ValueError as exc:
        raise RuntimeContractError(
            f"configs/tools.json 与 Python tool schema 漂移：{exc}"
        ) from exc
    checks["tool_config_sha256"] = config_hash
    if config_hash != c["tool_schema_hash"]:
        raise RuntimeContractError(
            "configs/tools.json 与冻结合同不一致："
            f"config={config_hash} contract={c['tool_schema_hash']}"
        )

    if PROJECTION_CONTRACT_VERSION != c["observation_projection_contract"]:
        raise RuntimeContractError(
            "PROJECTION_CONTRACT_VERSION 与冻结合同不一致："
            f"code={PROJECTION_CONTRACT_VERSION!r} "
            f"contract={c['observation_projection_contract']!r}"
        )
    checks["observation_projection_contract"] = PROJECTION_CONTRACT_VERSION

    try:
        projection_bytes = Path(projection_module.__file__).resolve().read_bytes()
    except Exception as exc:  # noqa: BLE001 - 缺失/不可读必须失败
        raise RuntimeContractError(
            f"observation projection 模块不可读，无法核对冻结字节：{exc}"
        ) from exc
    projection_hash = hashlib.sha256(projection_bytes).hexdigest()
    checks["observation_projection_code_sha256"] = projection_hash
    if projection_hash != c["observation_projection_hash"]:
        raise RuntimeContractError(
            "projection 原始字节与冻结合同不一致（部署包必须使用 LF 冻结字节）："
            f"code={projection_hash} contract={c['observation_projection_hash']}"
        )

    if REWARD_V3 != c["reward_version"]:
        raise RuntimeContractError(
            f"metrics.REWARD_V3 与冻结合同不一致："
            f"code={REWARD_V3!r} contract={c['reward_version']!r}"
        )
    checks["reward_version"] = REWARD_V3
    return checks


def canonical_json_bytes(value: Any) -> bytes:
    """规范 JSON 序列化；非 JSON 值直接失败，不使用 default=str。

    ``allow_nan=False``：NaN/Infinity 不是合法 JSON，必须拒绝而不是静默输出。
    """
    return json.dumps(
        thaw_runtime_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_contract_payload(contract: Mapping[str, Any]) -> dict:
    """返回删除 ``contract_sha256`` 后的深拷贝；不修改输入。"""
    if not isinstance(contract, Mapping):
        raise RuntimeContractError(
            f"runtime contract 必须是 JSON object，got {type(contract).__name__}"
        )
    return {key: value for key, value in contract.items() if key != "contract_sha256"}


def compute_runtime_contract_canonical_sha256(contract: Mapping[str, Any]) -> str:
    payload = canonical_contract_payload(contract)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_runtime_contract_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_declared_hash(declared: Any) -> str:
    if not isinstance(declared, str) or len(declared) != 64:
        raise RuntimeContractError(
            f"contract_sha256 必须是 64 位小写十六进制，got {declared!r}"
        )
    if any(char not in _HEX64 for char in declared):
        raise RuntimeContractError(
            f"contract_sha256 含非十六进制字符：{declared!r}"
        )
    return declared


def _validate_field_types(contract: Mapping[str, Any]) -> None:
    for field in FROZEN_FIELD_VALUES:
        value = contract.get(field)
        expected = FROZEN_FIELD_VALUES[field]
        if isinstance(expected, int):
            if not isinstance(value, int) or isinstance(value, bool) or value != expected:
                raise RuntimeContractError(
                    f"runtime contract {field} 必须等于冻结值 {expected}，got {value!r}"
                )
        else:
            if value != expected:
                raise RuntimeContractError(
                    f"runtime contract {field} 必须等于冻结值 {expected!r}，got {value!r}"
                )
    for field in (
        "tool_schema_hash",
        "system_prompt_hash",
        "observation_projection_hash",
    ):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeContractError(
                f"runtime contract {field} 必须是 64 位十六进制，got {value!r}"
            )


def load_and_validate_runtime_contract(path: str | Path) -> ValidatedRuntimeContract:
    """加载并完整校验冻结合同；任何异常 fail closed。"""
    path = Path(path)
    if not path.is_file():
        raise RuntimeContractError(f"runtime contract 文件不存在：{path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeContractError(f"runtime contract 不是有效 UTF-8：{path}") from exc
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeContractError(
            f"runtime contract JSON 解析失败：{path}（{exc}）"
        ) from exc
    if not isinstance(contract, dict):
        raise RuntimeContractError(
            f"runtime contract 根必须是 JSON object，got {type(contract).__name__}：{path}"
        )
    missing = sorted(REQUIRED_FIELDS - set(contract))
    if missing:
        raise RuntimeContractError(
            f"runtime contract 缺少必需字段：{missing}（{path}）"
        )
    declared = _validate_declared_hash(contract.get("contract_sha256"))
    _validate_field_types(contract)
    canonical = compute_runtime_contract_canonical_sha256(contract)
    if declared != canonical:
        raise RuntimeContractError(
            f"runtime contract 自校验失败：declared={declared} canonical={canonical}（{path}）"
        )
    if declared != FROZEN_CONTRACT_SHA256:
        raise RuntimeContractError(
            f"runtime contract 偏离冻结合同身份：declared={declared} "
            f"frozen={FROZEN_CONTRACT_SHA256}（{path}）"
        )
    validated = ValidatedRuntimeContract(
        contract=dict(contract),
        canonical_payload=canonical_contract_payload(contract),
        declared_sha256=declared,
        canonical_sha256=canonical,
        file_sha256=compute_runtime_contract_file_sha256(path),
    )
    object.__setattr__(validated, "_source_path", path.resolve())
    return validated


def revalidate_runtime_contract_source(contract: ValidatedRuntimeContract) -> None:
    """Re-read and revalidate the exact source file used to create ``contract``."""
    source_path = getattr(contract, "_source_path", None)
    if not isinstance(source_path, Path):
        raise RuntimeContractError(
            "ValidatedRuntimeContract 缺少受信任 source_path；拒绝手工伪造合同"
        )
    try:
        live_file_sha = compute_runtime_contract_file_sha256(source_path)
    except (OSError, ValueError) as exc:
        raise RuntimeContractError(
            f"runtime contract source 不可读：{source_path}: {exc}"
        ) from exc
    if live_file_sha != contract.file_sha256:
        raise RuntimeContractError(
            "runtime contract 源文件在加载后发生变化："
            f"stored={contract.file_sha256} live={live_file_sha} path={source_path}"
        )
    live = load_and_validate_runtime_contract(source_path)
    if thaw_runtime_value(live.contract) != thaw_runtime_value(contract.contract):
        raise RuntimeContractError(
            "runtime contract source payload 与已验证对象不一致"
        )
    if thaw_runtime_value(live.canonical_payload) != thaw_runtime_value(
        contract.canonical_payload
    ):
        raise RuntimeContractError(
            "runtime contract source canonical payload 与已验证对象不一致"
        )
    for name in ("declared_sha256", "canonical_sha256", "file_sha256"):
        if getattr(live, name) != getattr(contract, name):
            raise RuntimeContractError(
                "runtime contract source identity 与已验证对象不一致："
                f"{name} stored={getattr(contract, name)} live={getattr(live, name)}"
            )
