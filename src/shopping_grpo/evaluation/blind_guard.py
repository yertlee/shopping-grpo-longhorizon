"""Content- and task-ID protection for the frozen blind final test."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path

from shopping_grpo.evaluation.artifacts import ArtifactError

BLIND_GUARD_SCHEMA = "shopping-blind-asset-guard-v1"
BLIND_TASK_IDS_SCHEMA = "shopping-blind-task-ids-v1"
_RESOURCE_PACKAGE = "shopping_grpo.resources"
_GUARD_RESOURCE = "blind_guard.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# 进入 Final 流水线的输入必须是 IDs-only；以下字段名（递归扫描，大小写不敏感）
# 一旦出现即视为隐藏 goal/product/rubric/answer 内容并拒绝。
HIDDEN_FIELD_NAMES = frozenset(
    {
        "answer",
        "answers",
        "asin",
        "attribute",
        "attributes",
        "brand",
        "category",
        "customization_options",
        "expected_brand",
        "expected_core_functions",
        "expected_model",
        "goal",
        "goal_options",
        "goals",
        "instruction",
        "instruction_options",
        "instruction_text",
        "price",
        "price_upper",
        "pricing",
        "product",
        "product_item",
        "product_item_dict",
        "products",
        "prompt",
        "query",
        "required_options_by_key",
        "reward",
        "reward_detail",
        "reward_features",
        "rubric",
        "rubrics",
        "shop_name",
        "target_product",
        "title",
    }
)
# IDs-only blind asset 允许的行结构；extra_info 只允许携带 task_id。
BLIND_ASSET_ALLOWED_ROW_KEYS = frozenset({"task_id", "extra_info"})
BLIND_ASSET_ALLOWED_EXTRA_KEYS = frozenset({"task_id"})
_EXPECTED_METADATA = {
    "asset": "shop_benchmark_reward_v3_final_200_clean",
    "contract": "environment-v2.1/reward-v3/curated-final200-clean-v1",
    "environment_version": "shopsimulator-environment-v2.1",
    "reward_version": "shopsimulator-reward-v3",
    "evaluated": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resource_object(name: str) -> dict:
    try:
        resource = files(_RESOURCE_PACKAGE).joinpath(name)
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (ModuleNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read packaged blind resource: {name}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"packaged blind resource must be an object: {name}")
    return value


def _row_task_id(row: Mapping) -> int | None:
    value = row.get("task_id")
    if value is None:
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("task_id")
    if value is None:
        normalized = row.get("normalized_trajectory")
        if isinstance(normalized, Mapping):
            value = normalized.get("task_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid task_id in {row!r}") from exc


def _jsonl_task_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    opener = gzip.open if path.name.endswith(".gz") else open
    task_ids = set()
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is not None:
                    task_ids.add(task_id)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return task_ids


def _declared_digest(guard: Mapping, field: str) -> str:
    """声明 hash 必须是 64 位小写十六进制，而不是任意 64 字符字符串。"""

    digest = guard.get(field)
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ArtifactError(f"invalid packaged blind {field}")
    return digest


def _find_hidden_fields(value: object, path: str) -> list[str]:
    """递归收集隐藏 goal/product/rubric/answer 类字段名。"""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            location = f"{path}.{key}"
            if key_text in HIDDEN_FIELD_NAMES:
                found.append(location)
            found.extend(_find_hidden_fields(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_hidden_fields(child, f"{path}[{index}]"))
    return found


def _load_blind_asset_rows(path: Path) -> list[tuple[int, Mapping, int]]:
    """解析 blind asset 行；返回 (line_number, row, task_id) 列表。"""

    opener = gzip.open if path.name.endswith(".gz") else open
    rows: list[tuple[int, Mapping, int]] = []
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: blind asset row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is None:
                    raise ArtifactError(
                        f"{path}:{line_number}: blind asset row has no task_id"
                    )
                rows.append((line_number, value, task_id))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return rows


def validate_blind_asset_file(
    path: str | Path,
    *,
    declared_task_sha256: str,
    expected_task_ids: set[int],
) -> dict:
    """读入 blind asset 输入文件并完整校验：hash、结构、隐藏字段。

    - 重算文件实际 SHA256 并与冻结 asset 声明值比对；比对结果显式写进返回
      report，绝不静默省略。
    - 行结构必须是 IDs-only（task_id / extra_info.task_id）。
    - 递归扫描隐藏 goal/product/rubric/answer 类字段，出现即拒绝——即使
      task ID 集合完全正确也不放行。
    """

    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"blind asset input is not a file: {path}")
    actual_sha256 = _sha256_file(path)
    # The Final asset is frozen by bytes, not merely by its task-ID set.  A
    # caller must not be able to continue with a same-IDs/different-bytes
    # document and treat the report's boolean as advisory metadata.
    if actual_sha256 != declared_task_sha256:
        raise ArtifactError(
            f"blind asset content SHA256 does not match the frozen declaration: "
            f"path={path} actual={actual_sha256} declared={declared_task_sha256}"
        )
    rows = _load_blind_asset_rows(path)
    if not rows:
        raise ArtifactError(f"blind asset input is empty: {path}")

    seen: set[int] = set()
    for line_number, row, task_id in rows:
        unexpected_keys = sorted(set(map(str, row)) - BLIND_ASSET_ALLOWED_ROW_KEYS)
        if unexpected_keys:
            raise ArtifactError(
                f"{path}:{line_number}: blind asset rows must be IDs-only; "
                f"unexpected keys {unexpected_keys}"
            )
        extra = row.get("extra_info")
        if extra is not None:
            extra_keys = set(map(str, extra)) if isinstance(extra, Mapping) else None
            if extra_keys is None or extra_keys - BLIND_ASSET_ALLOWED_EXTRA_KEYS:
                raise ArtifactError(
                    f"{path}:{line_number}: blind asset extra_info may only "
                    "carry task_id"
                )
        hidden = _find_hidden_fields(row, f"row(task_id={task_id})")
        if hidden:
            raise ArtifactError(
                f"{path}:{line_number}: blind asset input contains hidden "
                f"goal/product/rubric fields {hidden[:10]}; Final 流水线只接受 "
                "IDs-only 输入"
            )
        if task_id in seen:
            raise ArtifactError(
                f"{path}:{line_number}: duplicate task_id {task_id} in blind asset"
            )
        seen.add(task_id)

    missing = sorted(expected_task_ids - seen)
    extra_ids = sorted(seen - expected_task_ids)
    if missing or extra_ids:
        raise ArtifactError(
            "blind asset task IDs do not match the frozen set: "
            f"missing={missing[:10]} extra={extra_ids[:10]}"
        )
    return {
        "input_sha256": actual_sha256,
        "declared_task_sha256": declared_task_sha256,
        "content_matches_declared_asset": True,
        "task_count": len(seen),
        "hidden_field_scan": "passed",
        "row_schema": "ids_only",
    }


def validate_canonical_blind_asset() -> tuple[dict, set[int]]:
    """Validate the wheel-packaged guard contract and frozen task-ID set."""

    guard = _load_resource_object(_GUARD_RESOURCE)
    if guard.get("schema_version") != BLIND_GUARD_SCHEMA:
        raise ArtifactError("unsupported blind guard schema")
    if guard.get("manifest_version") != 1:
        raise ArtifactError("unsupported blind guard manifest version")
    if guard.get("split_role") != "blind_final_test":
        raise ArtifactError("blind guard split_role must be blind_final_test")
    required = guard.get("required_metadata")
    if not isinstance(required, Mapping):
        raise ArtifactError("blind guard required_metadata must be an object")
    mismatches = {
        key: {"required": expected, "actual": required.get(key)}
        for key, expected in _EXPECTED_METADATA.items()
        if required.get(key) != expected
    }
    if mismatches:
        raise ArtifactError(
            "packaged blind metadata contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    for field in ("task_sha256", "metadata_sha256"):
        _declared_digest(guard, field)

    ids_resource = guard.get("task_ids_resource")
    if not isinstance(ids_resource, str) or not ids_resource:
        raise ArtifactError("blind guard task_ids_resource is missing")
    ids_document = _load_resource_object(ids_resource)
    if ids_document.get("schema_version") != BLIND_TASK_IDS_SCHEMA:
        raise ArtifactError("unsupported blind task-ID schema")
    raw_task_ids = ids_document.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in raw_task_ids
    ):
        raise ArtifactError("packaged blind task_ids must be integers")
    task_ids = set(raw_task_ids)
    if len(task_ids) != len(raw_task_ids):
        raise ArtifactError("packaged blind task_ids contain duplicates")
    if len(task_ids) != int(guard.get("task_count", -1)):
        raise ArtifactError("packaged blind task count mismatch")
    return guard, task_ids


def guard_blind_final(
    paths: Iterable[Path],
    *,
    allowed: bool,
) -> dict[str, dict] | None:
    """Reject any artifact containing final-test task IDs, independent of name.

    ``allowed=False``（Dev）：任何与冻结 blind asset 同内容或 task ID 重叠的输入
    直接拒绝。

    ``allowed=True``（Final）：绝不提前返回。每个输入文件都要重算实际 SHA256 与
    声明值比对、校验 IDs-only 行结构、递归扫描隐藏 goal/product/rubric/answer
    字段，并断言 task ID 集合与冻结集合一致。返回 ``{path: report}`` 供 run
    manifest 显式记录；任何校验失败抛 :class:`ArtifactError`。
    """

    guard, final_task_ids = validate_canonical_blind_asset()
    declared_sha = str(guard["task_sha256"])
    if not allowed:
        blocked = {}
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                continue
            same_content = _sha256_file(path) == declared_sha
            overlap = sorted(_jsonl_task_ids(path) & final_task_ids)
            if same_content or overlap:
                blocked[str(path)] = {
                    "same_content": same_content,
                    "overlap_count": len(overlap),
                    "sample_task_ids": overlap[:10],
                }
        if blocked:
            raise ArtifactError(
                "refusing to consume frozen blind-final tasks without "
                "--allow-blind-final: "
                + json.dumps(blocked, ensure_ascii=False, sort_keys=True)
            )
        return None
    reports: dict[str, dict] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        reports[str(path)] = validate_blind_asset_file(
            path,
            declared_task_sha256=declared_sha,
            expected_task_ids=final_task_ids,
        )
    return reports
