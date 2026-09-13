"""Verify single-worker or four-worker deterministic ShopSimulator replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.data_quality.reachability import (  # noqa: E402
    validate_reachability_manifest,
)
from commerce_posttrain.environment.replay import (  # noqa: E402
    canonical_json_bytes,
    load_private_tasks,
    run_replay_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--workers", type=int, choices=(1, 4), required=True)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data/private/task_facts.jsonl"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data/manifests/split_manifest.json"
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "data/manifests/runtime_contract.json",
    )
    parser.add_argument(
        "--reachability",
        type=Path,
        default=ROOT / "data/manifests/task_reachability_manifest.json",
    )
    parser.add_argument("--split-name", default="teacher_pool")
    parser.add_argument("--task-count", type=int, default=4)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_rows = load_private_tasks(args.tasks)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    reachability = validate_reachability_manifest(
        json.loads(args.reachability.read_text(encoding="utf-8")),
        expected_split_manifest_sha256=hashlib.sha256(
            args.split.read_bytes()
        ).hexdigest(),
    )
    if args.split_name not in split["splits"]:
        raise ValueError(f"unknown split: {args.split_name}")
    quality_split = reachability["splits"].get(args.split_name)
    if quality_split is None:
        raise ValueError(f"reachability manifest lacks split: {args.split_name}")
    split_task_ids = [int(value) for value in split["splits"][args.split_name]["task_ids"]]
    classified = {
        int(value) for value in quality_split["eligible_task_ids"]
    } | {int(row["task_id"]) for row in quality_split["unreachable"]}
    if classified != set(split_task_ids):
        raise ValueError("reachability classifications do not match selected split")
    eligible = set(int(value) for value in quality_split["eligible_task_ids"])
    task_ids = [task_id for task_id in split_task_ids if task_id in eligible][
        : args.task_count
    ]
    if len(task_ids) != args.task_count:
        raise ValueError(
            f"split {args.split_name} has only {len(task_ids)} eligible tasks"
        )
    tasks = [task_rows[int(task_id)] for task_id in task_ids]
    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    reference = None
    if args.reference:
        previous = json.loads(args.reference.read_text(encoding="utf-8"))
        if previous["runtime_contract_sha256"] != runtime["contract_sha256"]:
            raise ValueError("reference uses a different runtime contract")
        if previous.get("reachability_manifest_sha256") != reachability[
            "manifest_sha256"
        ]:
            raise ValueError("reference uses a different reachability manifest")
        reference = {int(key): value for key, value in previous["signatures"].items()}
    # A concurrent replay is an infrastructure stress gate, so its default must
    # exercise each task more than once instead of accepting one lucky pass.
    repeats = args.repeats if args.repeats is not None else (2 if args.workers == 1 else 3)
    report = run_replay_suite(
        tasks,
        base_url=args.base_url,
        workers=args.workers,
        repeats=repeats,
        runtime_contract_hash=runtime["contract_sha256"],
        reference=reference,
    )
    report.pop("report_sha256", None)
    report["split_name"] = args.split_name
    report["reachability_manifest_sha256"] = reachability["manifest_sha256"]
    report["excluded_unreachable_count"] = quality_split["unreachable_count"]
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
