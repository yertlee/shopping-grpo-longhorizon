"""Build and validate a private-TaskFacts split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.splits.manifest import (  # noqa: E402
    build_split_manifest,
    load_split_spec,
    load_task_facts_jsonl,
    load_task_ids,
    sha256_file,
    validate_split_manifest,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="private normalized TaskFacts JSONL",
    )
    parser.add_argument(
        "--final-task-ids",
        type=Path,
        default=ROOT / "data" / "splits" / "final_task_ids.jsonl",
    )
    parser.add_argument("--spec", type=Path, default=ROOT / "configs" / "splits" / "core.json")
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "data" / "manifests" / "runtime_contract.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "split_manifest.json",
    )
    parser.add_argument("--source-label", default="data/private/task_facts.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_task_facts_jsonl(args.tasks)
    final_ids = load_task_ids(args.final_task_ids)
    spec = load_split_spec(args.spec)
    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    manifest = build_split_manifest(
        tasks=tasks,
        final_task_ids=final_ids,
        spec=spec,
        task_source_hash=sha256_file(args.tasks),
        runtime_contract_hash=runtime["contract_sha256"],
        task_source_label=args.source_label,
    )
    validate_split_manifest(manifest, tasks=tasks)
    write_split_manifest(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
