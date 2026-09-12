#!/usr/bin/env python3
"""Build the private Final-200 evaluation facts document.

The evaluator's ``LazyEnvironmentTaskFactsSource`` needs a private
``shopping-task-facts-source-v1`` document for any IDs-only split.  Dev-100 gets
one from ``build_dev_eval_assets.py``; Final-200 previously had none, so this
script produces the Final counterpart with the *same* official ShopSimulator
loader (not a reimplemented parser) and the same ``shopping-task-facts-v1`` row
schema.

Blindness is enforced here:
- the public input is the IDs-only ``final_task_ids.jsonl``;
- the produced task_id set must equal the canonical frozen blind asset;
- the produced task_id set must not overlap the Dev-100 split.

Usage:
    PYTHONPATH=src python scripts/build_final_eval_facts.py \
      --task-ids data/splits/final_task_ids.jsonl \
      --product-gzip <ShopSimulator product gzip> \
      --shopsim-root <ShopSimulator root> \
      --facts-output data/private/final-200-evaluation-facts.json \
      --manifest-output data/manifests/final-200-evaluation-facts.manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the exact Dev loader/helpers so Dev and Final facts share one semantics.
from scripts.build_dev_eval_assets import (  # noqa: E402
    BUILDER_VERSION,
    FACTS_SCHEMA,
    PRODUCT_COUNT,
    PRODUCT_SHA256,
    _atomic_bytes,
    _default_code_hashes,
    _default_facts,
    _load_products,
    _official_goals,
    canonical_id_hash,
    ordered_id_hash,
    sha256_file,
)

FINAL_FACTS_BUILDER_VERSION = "final-eval-facts-v1"
EXPECTED_TASK_COUNT = 200


def _read_ids(path: Path, expected: int) -> list[int]:
    ids: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), int):
            raise ValueError(f"{path}:{line_number}: task_id must be an integer")
        ids.append(int(row["task_id"]))
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique rows, got {len(ids)}")
    return ids


def build_final_eval_facts(
    *,
    task_ids_path: Path,
    product_gzip: Path,
    shopsim_root: Path,
    facts_output: Path,
    manifest_output: Path,
    expected_task_count: int = EXPECTED_TASK_COUNT,
    expected_product_count: int = PRODUCT_COUNT,
    expected_product_sha256: str = PRODUCT_SHA256,
    product_loader: Callable[[Path, Path], tuple[list[Mapping], Mapping, Mapping]] | None = None,
    goal_builder: Callable[[list[Mapping], Mapping], list[Mapping]] | None = None,
    facts_builder: Callable[[Sequence[int], list[Mapping], Mapping], list[dict]] | None = None,
    canonical_ids_provider: Callable[[], set[int]] | None = None,
    dev_ids_provider: Callable[[], set[int]] | None = None,
    code_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    task_ids = _read_ids(task_ids_path, expected_task_count)

    product_sha = sha256_file(product_gzip)
    if expected_product_sha256 and product_sha != expected_product_sha256:
        raise ValueError(f"product gzip SHA256 mismatch: {product_sha}")

    if canonical_ids_provider is None:
        from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset

        canonical_ids_provider = lambda: set(validate_canonical_blind_asset()[1])
    canonical_ids = set(canonical_ids_provider())
    if set(task_ids) != canonical_ids:
        raise ValueError(
            "final task_ids differ from the canonical frozen blind asset "
            f"(input={len(set(task_ids))}, canonical={len(canonical_ids)})"
        )
    if dev_ids_provider is None:
        dev_path = ROOT / "data/evaluation/dev-100/task_ids.jsonl"
        dev_ids_provider = lambda: set(_read_ids(dev_path, 100))
    overlap = set(task_ids) & set(dev_ids_provider())
    if overlap:
        raise ValueError(f"Final task IDs overlap Dev-100 split ({len(overlap)} IDs)")

    products, product_item_dict, prices = (product_loader or _load_products)(
        product_gzip, shopsim_root
    )
    if len(products) != expected_product_count:
        raise ValueError(
            f"official product loader returned {len(products)} products, "
            f"expected {expected_product_count}"
        )
    goals = (goal_builder or _official_goals)(products, prices)
    if len(goals) <= max(task_ids):
        raise ValueError("official goal list does not cover all Final task IDs")
    facts = (facts_builder or _default_facts)(task_ids, goals, product_item_dict)
    if len(facts) != len(task_ids):
        raise ValueError("facts builder returned the wrong number of rows")
    if any(row.get("task_id") != task_id for row, task_id in zip(facts, task_ids)):
        raise ValueError("facts are not in Final task id order")
    if any(row.get("schema_version") != "shopping-task-facts-v1" for row in facts):
        raise ValueError("facts builder returned an unsupported schema_version")

    facts_bytes = json.dumps(
        {"schema_version": FACTS_SCHEMA, "facts": facts},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode() + b"\n"
    _atomic_bytes(facts_output, facts_bytes)

    manifest = {
        "builder_version": FINAL_FACTS_BUILDER_VERSION,
        "facts_schema": FACTS_SCHEMA,
        "sources": {
            "final_task_ids": {"path_label": task_ids_path.name, "sha256": sha256_file(task_ids_path)},
            "products": {"path_label": product_gzip.name, "sha256": product_sha, "count": len(products)},
        },
        "task_count": len(task_ids),
        "task_ids_sha256": canonical_id_hash(task_ids),
        "task_order_sha256": ordered_id_hash(task_ids),
        "canonical_blind_asset_match": True,
        "dev_overlap_count": 0,
        "outputs": {
            "facts": {
                "path_label": facts_output.name,
                "sha256": sha256_file(facts_output),
                "row_count": len(facts),
            }
        },
        "code_hashes": dict(code_hashes or _default_code_hashes(shopsim_root)),
        "dev_builder_version": BUILDER_VERSION,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode() + b"\n"
    _atomic_bytes(manifest_output, manifest_bytes)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--product-gzip", type=Path, required=True)
    parser.add_argument("--shopsim-root", type=Path, required=True)
    parser.add_argument("--facts-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--dev-task-ids",
        type=Path,
        default=None,
        help="Dev-100 split（用于零重叠校验）；缺省尝试 <repo>/data/evaluation/dev-100/task_ids.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dev_ids_provider = None
    if args.dev_task_ids is not None:
        dev_path = args.dev_task_ids

        def dev_ids_provider() -> set[int]:  # noqa: F811 - local override
            return set(_read_ids(dev_path, 100))

    manifest = build_final_eval_facts(
        task_ids_path=args.task_ids,
        product_gzip=args.product_gzip,
        shopsim_root=args.shopsim_root,
        facts_output=args.facts_output,
        manifest_output=args.manifest_output,
        dev_ids_provider=dev_ids_provider,
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
