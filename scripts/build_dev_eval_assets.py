#!/usr/bin/env python3
"""Build the deterministic, private Dev-100 evaluation inputs.

The builder deliberately separates the public IDs-only split from the private
facts document.  Goal construction is injected in tests, while the production
path uses the ShopSimulator loader and goal generator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "src"))

PROCESS_SHA256 = "3ad667d0d3697ccfe7ad18693b0de65a14c50ad547b65562fe370bd9ad5bc67c"
OUTCOME_SHA256 = "8a1220b6344942f9c89470bfcd543e3d83d51134c496e1df16ac9294ffd67c92"
PRODUCT_SHA256 = "f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125"
PRODUCT_COUNT = 23_421
FACTS_SCHEMA = "shopping-task-facts-source-v1"
BUILDER_VERSION = "dev-eval-assets-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_ids(path: Path) -> tuple[list[int], bytes]:
    raw = path.read_bytes()
    ids: list[int] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), int):
            raise ValueError(f"{path}:{line_number}: task_id must be an integer")
        ids.append(int(row["task_id"]))
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(f"{path}: expected 100 unique rows, got {len(ids)}")
    return ids, raw


def ordered_id_hash(task_ids: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in task_ids], separators=(",", ":"))
    return sha256_bytes(payload.encode())


def canonical_id_hash(task_ids: Sequence[int]) -> str:
    payload = json.dumps(sorted({int(value) for value in task_ids}), separators=(",", ":"))
    return sha256_bytes(payload.encode())


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite different output: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_products(product_gzip: Path, shopsim_root: Path) -> tuple[list[Mapping], Mapping, Mapping]:
    """Use the official ShopSimulator loader, not a reimplemented parser."""
    import sys

    shop_env = shopsim_root / "shop_env"
    if str(shop_env) not in sys.path:
        sys.path.insert(0, str(shop_env))
    from web_agent_site.engine.engine import load_products

    with gzip.open(product_gzip, "rb") as source:
        raw_json = source.read()
    # The official loader accepts plain JSON, while the frozen source is gzip.
    # Keep the decompressed bridge private and ephemeral.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw_json)
    try:
        products, product_item_dict, product_prices, _ = load_products(
            filepath=str(temporary), human_goals=True
        )
    finally:
        temporary.unlink(missing_ok=True)
    return products, product_item_dict, product_prices


def _official_goals(products: list[Mapping], prices: Mapping) -> list[Mapping]:
    from web_agent_site.engine.goal import get_goals

    return get_goals(products, prices)


def _default_facts(task_ids: Sequence[int], goals: list[Mapping], product_item_dict: Mapping) -> list[dict]:
    from shopping_grpo.evaluation.task_facts import task_facts_from_environment

    return task_facts_from_environment(
        task_ids=task_ids, goals=goals, product_item_dict=product_item_dict
    )


def _default_code_hashes(shopsim_root: Path) -> dict[str, str]:
    paths = {
        "builder": Path(__file__).resolve(),
        "shopsim_engine": shopsim_root / "shop_env/web_agent_site/engine/engine.py",
        "shopsim_goals": shopsim_root / "shop_env/web_agent_site/engine/goal.py",
        "evaluator_task_facts": ROOT / "src/shopping_grpo/evaluation/task_facts.py",
        "evaluator_rubric": ROOT / "src/shopping_grpo/evaluation/rubric.py",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("cannot hash required contract code: " + ", ".join(missing))
    return {label: sha256_file(path) for label, path in paths.items()}


def build_dev_eval_assets(
    *,
    process_path: Path,
    outcome_path: Path,
    product_gzip: Path,
    shopsim_root: Path,
    ids_output: Path,
    facts_output: Path,
    manifest_output: Path,
    goal_builder: Callable[[list[Mapping], list[Mapping]], list[Mapping]] | None = None,
    facts_builder: Callable[[Sequence[int], list[Mapping], Mapping], list[dict]] | None = None,
    product_loader: Callable[[Path, Path], tuple[list[Mapping], Mapping, list[Mapping]]] | None = None,
    final_ids_provider: Callable[[], set[int]] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    code_hashes: Mapping[str, str] | None = None,
    expected_product_count: int = PRODUCT_COUNT,
) -> dict[str, Any]:
    """Build and atomically publish the three deterministic Dev artifacts."""
    expected = {"process": PROCESS_SHA256, "outcome": OUTCOME_SHA256, "products": PRODUCT_SHA256}
    if expected_hashes:
        expected.update(expected_hashes)
    process_ids, process_bytes = _read_jsonl_ids(process_path)
    outcome_ids, outcome_bytes = _read_jsonl_ids(outcome_path)
    source_hashes = {
        "process": sha256_bytes(process_bytes),
        "outcome": sha256_bytes(outcome_bytes),
        "products": sha256_file(product_gzip),
    }
    for name, actual in source_hashes.items():
        if expected.get(name) and actual != expected[name]:
            raise ValueError(f"{name} source SHA256 mismatch: {actual}")
    if set(process_ids) != set(outcome_ids):
        raise ValueError("process/outcome task_id sets differ")
    task_ids = process_ids
    if final_ids_provider is None:
        from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset

        final_ids_provider = lambda: validate_canonical_blind_asset()[1]
    final_ids = final_ids_provider()
    overlap = set(task_ids) & set(final_ids)
    if overlap:
        raise ValueError(f"Dev task IDs overlap frozen Final asset ({len(overlap)} IDs)")

    products, product_item_dict, prices = (product_loader or _load_products)(product_gzip, shopsim_root)
    if len(products) != expected_product_count:
        raise ValueError(f"official product loader returned {len(products)} products")
    goals = (goal_builder or _official_goals)(products, prices)
    if len(goals) <= max(task_ids):
        raise ValueError("official goal list does not cover all Dev task IDs")
    facts = (facts_builder or _default_facts)(task_ids, goals, product_item_dict)
    if len(facts) != len(task_ids):
        raise ValueError("facts builder returned the wrong number of rows")
    if any(row.get("task_id") != task_id for row, task_id in zip(facts, task_ids)):
        raise ValueError("facts are not in process-dev task order")
    if any(row.get("schema_version") != "shopping-task-facts-v1" for row in facts):
        raise ValueError("facts builder returned an unsupported schema_version")

    ids_bytes = b"".join(
        json.dumps({"task_id": task_id}, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        for task_id in task_ids
    )
    facts_bytes = json.dumps(
        {"schema_version": FACTS_SCHEMA, "facts": facts},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode() + b"\n"
    manifest = {
        "builder_version": BUILDER_VERSION,
        "facts_schema": FACTS_SCHEMA,
        "sources": {
            "process_dev": {"path_label": process_path.name, "sha256": source_hashes["process"]},
            "outcome_dev": {"path_label": outcome_path.name, "sha256": source_hashes["outcome"]},
            "products": {"path_label": product_gzip.name, "sha256": source_hashes["products"], "count": len(products)},
        },
        "task_count": len(task_ids),
        "task_ids_sha256": canonical_id_hash(task_ids),
        "task_order_sha256": ordered_id_hash(task_ids),
        "source_sequences_equal": process_ids == outcome_ids,
        "final_overlap_count": 0,
        "outputs": {
            "ids_only_sha256": sha256_bytes(ids_bytes),
            "facts_sha256": sha256_bytes(facts_bytes),
        },
        "code_sha256": dict(sorted((code_hashes or _default_code_hashes(shopsim_root)).items())),
    }
    manifest["manifest_content_sha256"] = sha256_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    # Validate every existing destination before publishing any missing one.
    for path, data in ((ids_output, ids_bytes), (facts_output, facts_bytes), (manifest_output, manifest_bytes)):
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite different output: {path}")
    for path, data in ((ids_output, ids_bytes), (facts_output, facts_bytes), (manifest_output, manifest_bytes)):
        _atomic_bytes(path, data)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (("process", "process-dev JSONL"), ("outcome", "outcome-dev JSONL"), ("product-gzip", "ShopSimulator product gzip"), ("shopsim-root", "ShopSimulator root"), ("ids-output", "IDs-only output"), ("facts-output", "private facts JSON output"), ("manifest-output", "builder manifest output")):
        parser.add_argument(f"--{name}", required=True, type=Path, help=help_text)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_dev_eval_assets(
        process_path=args.process, outcome_path=args.outcome, product_gzip=args.product_gzip,
        shopsim_root=args.shopsim_root, ids_output=args.ids_output, facts_output=args.facts_output,
        manifest_output=args.manifest_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
