"""Build task/product/query-cluster disjoint split manifests.

The builder consumes private normalized TaskFacts.  Only task IDs, counts and
content hashes are emitted to the public manifest; target products and queries
remain in the private source file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPLIT_MANIFEST_VERSION = "commerce-split-manifest-v1"
TASK_FACTS_VERSION = "commerce-task-facts-v1"
LEAKAGE_CONTRACT_VERSION = "commerce-leakage-contract-v1"


class SplitLeakageError(ValueError):
    """One or more task, product or semantic-cluster boundaries overlap."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return "".join(character for character in text if character.isalnum())


def query_template(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "<num>", text)
    text = re.sub(r"[^\w\u4e00-\u9fff<>]+", "", text)
    return text


def _string_list(value: Any, *, field: str, task_id: int, required: bool = False) -> list[str]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, (str, int)):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        raise ValueError(f"task {task_id} field {field} must be a string or list")
    normalized = sorted({normalize_text(item) for item in values if normalize_text(item)})
    if required and not normalized:
        raise ValueError(f"task {task_id} requires at least one {field}")
    return normalized


def normalize_task_facts(row: Mapping[str, Any]) -> dict:
    if not isinstance(row, Mapping):
        raise ValueError("each TaskFacts row must be an object")
    try:
        task_id = int(row["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TaskFacts row requires integer task_id") from exc
    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    query = str(row.get("query") or row.get("instruction") or "").strip()
    if not query:
        raise ValueError(f"task {task_id} requires query/instruction")
    product_ids = _string_list(
        row.get("target_product_ids", row.get("target_product_id", row.get("target_asin"))),
        field="target_product_ids",
        task_id=task_id,
        required=True,
    )
    explicit_query_cluster = normalize_text(row.get("query_cluster"))
    exact_query = normalize_text(query)
    if row.get("template_id") is not None:
        template = normalize_text(row["template_id"])
    elif row.get("query_template") is not None:
        # A normalized TaskFacts row already contains the derived fingerprint.
        # Run it through the template normalizer only to make round-trips stable.
        template = query_template(row["query_template"])
    else:
        template = query_template(query)
    if not exact_query or not template:
        raise ValueError(f"task {task_id} query cannot produce leakage fingerprints")
    return {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": task_id,
        "query": query,
        "target_product_ids": product_ids,
        "query_cluster": explicit_query_cluster or exact_query,
        "query_template": template,
        "model_tokens": _string_list(
            row.get("model_tokens"), field="model_tokens", task_id=task_id
        ),
        "product_family": normalize_text(row.get("product_family")),
        "difficulty": normalize_text(row.get("difficulty")) or "unknown",
    }


def load_task_facts_jsonl(path: str | Path) -> list[dict]:
    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = normalize_task_facts(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid TaskFacts at line {line_number}: {exc}") from exc
            if row["task_id"] in seen:
                raise ValueError(f"duplicate task_id in TaskFacts: {row['task_id']}")
            seen.add(row["task_id"])
            rows.append(row)
    if not rows:
        raise ValueError("TaskFacts source is empty")
    return rows


def load_task_ids(path: str | Path) -> list[int]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("task_ids")
        if not isinstance(value, list):
            raise ValueError("JSON task-id file must be an array or {task_ids: [...]} object")
        ids = [int(item) for item in value]
    else:
        ids = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                ids.append(int(value["task_id"] if isinstance(value, dict) else value))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid task ID at line {line_number}: {exc}") from exc
    if len(ids) != len(set(ids)):
        raise ValueError("task-id file contains duplicates")
    return ids


def leakage_keys(task: Mapping[str, Any]) -> set[str]:
    keys = {f"product:{value}" for value in task["target_product_ids"]}
    keys.add(f"query:{task['query_cluster']}")
    keys.add(f"template:{task['query_template']}")
    keys.update(f"model:{value}" for value in task["model_tokens"])
    if task.get("product_family"):
        keys.add(f"family:{task['product_family']}")
    return keys


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def leakage_components(tasks: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    union = _UnionFind(task["task_id"] for task in tasks)
    key_owner: dict[str, int] = {}
    for task in sorted(tasks, key=lambda item: item["task_id"]):
        task_id = task["task_id"]
        for key in sorted(leakage_keys(task)):
            owner = key_owner.setdefault(key, task_id)
            union.union(owner, task_id)
    components: dict[int, list[int]] = defaultdict(list)
    for task in tasks:
        components[union.find(task["task_id"])].append(task["task_id"])
    return {root: sorted(ids) for root, ids in sorted(components.items())}


def _stable_score(seed: int, split_name: str, value: Any) -> str:
    return sha256_bytes(f"{seed}:{split_name}:{value}".encode("utf-8"))


def _interleaved_task_order(
    task_ids: Iterable[int],
    *,
    tasks_by_id: Mapping[int, Mapping[str, Any]],
    seed: int,
    split_name: str,
) -> list[int]:
    strata: dict[str, list[int]] = defaultdict(list)
    for task_id in task_ids:
        strata[tasks_by_id[task_id]["difficulty"]].append(task_id)
    queues = {}
    for difficulty, ids in strata.items():
        queues[difficulty] = deque(
            sorted(ids, key=lambda value: _stable_score(seed, f"{split_name}:{difficulty}", value))
        )
    order = []
    difficulty_order = sorted(queues)
    while any(queues.values()):
        for difficulty in difficulty_order:
            if queues[difficulty]:
                order.append(queues[difficulty].popleft())
    return order


def _allocate_split(
    *,
    split_name: str,
    count: int,
    available_roots: set[int],
    components: Mapping[int, list[int]],
    root_by_task: Mapping[int, int],
    tasks_by_id: Mapping[int, Mapping[str, Any]],
    seed: int,
) -> tuple[list[int], set[int], list[int]]:
    candidate_ids = [
        task_id
        for root in available_roots
        for task_id in components[root]
    ]
    ordered = _interleaved_task_order(
        candidate_ids,
        tasks_by_id=tasks_by_id,
        seed=seed,
        split_name=split_name,
    )
    selected = []
    claimed_roots = set()
    excluded_remainder = []
    for task_id in ordered:
        if len(selected) >= count:
            break
        root = root_by_task[task_id]
        if root not in available_roots or root in claimed_roots:
            continue
        claimed_roots.add(root)
        component_order = sorted(
            components[root],
            key=lambda value: _stable_score(seed, f"{split_name}:component", value),
        )
        remaining = count - len(selected)
        selected.extend(component_order[:remaining])
        excluded_remainder.extend(component_order[remaining:])
    if len(selected) != count:
        raise ValueError(
            f"cannot allocate {count} tasks to {split_name}; only {len(selected)} "
            "remain after leakage isolation"
        )
    return sorted(selected), claimed_roots, sorted(excluded_remainder)


def _difficulty_counts(
    task_ids: Iterable[int], tasks_by_id: Mapping[int, Mapping[str, Any]]
) -> dict:
    return dict(sorted(Counter(tasks_by_id[task_id]["difficulty"] for task_id in task_ids).items()))


def _split_record(task_ids: Sequence[int], tasks_by_id: Mapping[int, Mapping[str, Any]]) -> dict:
    ids = sorted(int(task_id) for task_id in task_ids)
    return {
        "count": len(ids),
        "task_ids": ids,
        "task_ids_sha256": sha256_bytes(canonical_json_bytes(ids)),
        "difficulty_counts": _difficulty_counts(ids, tasks_by_id),
    }


def audit_split_leakage(
    splits: Mapping[str, Sequence[int]],
    tasks_by_id: Mapping[int, Mapping[str, Any]],
) -> dict:
    owners_by_task: dict[int, str] = {}
    owners_by_key: dict[str, tuple[str, int]] = {}
    violations = []
    for split_name in sorted(splits):
        for task_id in splits[split_name]:
            if task_id in owners_by_task:
                violations.append(
                    {
                        "kind": "task_id",
                        "key": str(task_id),
                        "left_split": owners_by_task[task_id],
                        "right_split": split_name,
                    }
                )
            owners_by_task[task_id] = split_name
            for key in leakage_keys(tasks_by_id[task_id]):
                previous = owners_by_key.get(key)
                if previous and previous[0] != split_name:
                    violations.append(
                        {
                            "kind": key.split(":", 1)[0],
                            "key_hash": sha256_bytes(key.encode("utf-8")),
                            "left_split": previous[0],
                            "left_task_id": previous[1],
                            "right_split": split_name,
                            "right_task_id": task_id,
                        }
                    )
                else:
                    owners_by_key[key] = (split_name, task_id)
    if violations:
        preview = json.dumps(violations[:5], ensure_ascii=False, sort_keys=True)
        raise SplitLeakageError(f"cross-split leakage detected: {preview}")
    return {
        "contract_version": LEAKAGE_CONTRACT_VERSION,
        "task_id_overlap_count": 0,
        "product_overlap_count": 0,
        "query_cluster_overlap_count": 0,
        "template_overlap_count": 0,
        "model_overlap_count": 0,
        "family_overlap_count": 0,
    }


@dataclass(frozen=True)
class SplitSpec:
    seed: int
    counts: dict[str, int]
    allocation_order: tuple[str, ...]


def load_split_spec(path: str | Path) -> SplitSpec:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "commerce-split-spec-v1":
        raise ValueError("unsupported split-spec schema")
    if value.get("leakage_contract") != LEAKAGE_CONTRACT_VERSION:
        raise ValueError("split spec does not select the frozen leakage contract")
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("split spec requires counts")
    normalized_counts = {}
    for name in ("teacher_pool", "grpo_train", "grpo_validation", "final"):
        count = counts.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"split count {name} must be a positive integer")
        normalized_counts[name] = count
    order = tuple(value.get("allocation_order") or ())
    if set(order) != {"teacher_pool", "grpo_train", "grpo_validation"}:
        raise ValueError("allocation_order must contain each train split exactly once")
    return SplitSpec(seed=int(value["seed"]), counts=normalized_counts, allocation_order=order)


def build_split_manifest(
    *,
    tasks: Sequence[Mapping[str, Any]],
    final_task_ids: Sequence[int],
    spec: SplitSpec,
    task_source_hash: str,
    runtime_contract_hash: str,
    task_source_label: str,
) -> dict:
    normalized = [normalize_task_facts(task) for task in tasks]
    tasks_by_id = {task["task_id"]: task for task in normalized}
    if len(tasks_by_id) != len(normalized):
        raise ValueError("TaskFacts contains duplicate task IDs")
    final_ids = [int(task_id) for task_id in final_task_ids]
    if len(final_ids) != spec.counts["final"] or len(set(final_ids)) != len(final_ids):
        raise ValueError(f"Final split must contain exactly {spec.counts['final']} unique task IDs")
    missing = sorted(set(final_ids) - set(tasks_by_id))
    if missing:
        raise ValueError(f"TaskFacts is missing Final task IDs: {missing[:10]}")

    components = leakage_components(normalized)
    root_by_task = {
        task_id: root
        for root, task_ids in components.items()
        for task_id in task_ids
    }
    final_roots = {root_by_task[task_id] for task_id in final_ids}
    final_conflict_excluded = sorted(
        task_id
        for root in final_roots
        for task_id in components[root]
        if task_id not in set(final_ids)
    )
    available_roots = set(components) - final_roots
    split_ids: dict[str, list[int]] = {"final": sorted(final_ids)}
    component_remainders = []
    for split_name in spec.allocation_order:
        selected, claimed_roots, remainders = _allocate_split(
            split_name=split_name,
            count=spec.counts[split_name],
            available_roots=available_roots,
            components=components,
            root_by_task=root_by_task,
            tasks_by_id=tasks_by_id,
            seed=spec.seed,
        )
        split_ids[split_name] = selected
        available_roots -= claimed_roots
        component_remainders.extend(remainders)

    leakage_audit = audit_split_leakage(split_ids, tasks_by_id)
    manifest = {
        "schema_version": SPLIT_MANIFEST_VERSION,
        "leakage_contract_version": LEAKAGE_CONTRACT_VERSION,
        "selection_algorithm": "deterministic-component-aware-stratified-v1",
        "seed": spec.seed,
        "task_source": {
            "label": task_source_label,
            "sha256": task_source_hash,
            "rows": len(normalized),
        },
        "runtime_contract_sha256": runtime_contract_hash,
        "splits": {
            name: _split_record(split_ids[name], tasks_by_id)
            for name in ("teacher_pool", "grpo_train", "grpo_validation", "final")
        },
        "leakage_audit": {
            **leakage_audit,
            "component_count": len(components),
            "final_component_excluded_count": len(final_conflict_excluded),
            "partial_component_excluded_count": len(set(component_remainders)),
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
) -> dict:
    observed = dict(manifest)
    if observed.get("schema_version") != SPLIT_MANIFEST_VERSION:
        raise ValueError("unsupported split-manifest schema")
    recorded_hash = observed.pop("manifest_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json_bytes(observed)):
        raise ValueError("split manifest content hash mismatch")
    normalized = [normalize_task_facts(task) for task in tasks]
    tasks_by_id = {task["task_id"]: task for task in normalized}
    if len(tasks_by_id) != len(normalized):
        raise ValueError("TaskFacts contains duplicate task IDs")
    expected_names = {"teacher_pool", "grpo_train", "grpo_validation", "final"}
    if set(observed.get("splits", {})) != expected_names:
        raise ValueError("split manifest must contain the four frozen split names")
    split_ids = {
        name: split["task_ids"]
        for name, split in observed["splits"].items()
    }
    for name, task_ids in split_ids.items():
        if task_ids != sorted(task_ids):
            raise ValueError(f"split {name} task IDs are not canonical-sorted")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"split {name} contains duplicate task IDs")
        if observed["splits"][name].get("count") != len(task_ids):
            raise ValueError(f"split {name} count does not match task IDs")
        if observed["splits"][name]["task_ids_sha256"] != sha256_bytes(
            canonical_json_bytes(sorted(task_ids))
        ):
            raise ValueError(f"split {name} task-ID hash mismatch")
        missing = set(task_ids) - set(tasks_by_id)
        if missing:
            raise ValueError(f"split {name} references missing TaskFacts")
        expected_difficulty = _difficulty_counts(task_ids, tasks_by_id)
        if observed["splits"][name].get("difficulty_counts") != expected_difficulty:
            raise ValueError(f"split {name} difficulty counts do not match TaskFacts")
    audit_split_leakage(split_ids, tasks_by_id)
    observed["manifest_sha256"] = recorded_hash
    return observed


def write_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, output)
