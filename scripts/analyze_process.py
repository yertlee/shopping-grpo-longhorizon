"""Materialize Actor-visible process features from raw Teacher trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.curation.process_analyzer import (  # noqa: E402
    PROCESS_ANALYZER_VERSION,
    analyze_trajectory,
    attach_first_divergence,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data/private/task_facts.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = {int(row["task_id"]): row for row in load_jsonl(args.tasks)}
    grouped = defaultdict(list)
    for trajectory in load_jsonl(args.raw):
        task_id = int(trajectory["task_id"])
        grouped[task_id].append(
            (trajectory, analyze_trajectory(trajectory, query=tasks[task_id]["query"]))
        )
    records = []
    for task_id in sorted(grouped):
        ordered = sorted(grouped[task_id], key=lambda pair: pair[0]["attempt_index"])
        records.extend(
            attach_first_divergence(
                [pair[1] for pair in ordered],
                [pair[0] for pair in ordered],
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "commerce-process-features-manifest-v1",
        "analyzer_version": PROCESS_ANALYZER_VERSION,
        "analyzer_sha256": sha256_file(
            ROOT / "src/commerce_posttrain/curation/process_analyzer.py"
        ),
        "raw_path": str(args.raw),
        "raw_sha256": sha256_file(args.raw),
        "task_facts_sha256": sha256_file(args.tasks),
        "rows": len(records),
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
