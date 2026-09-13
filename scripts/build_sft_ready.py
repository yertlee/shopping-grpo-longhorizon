#!/usr/bin/env python3
"""Apply a shared length gate and deterministic Reserve replacement to SFT arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.curation.pipeline import (  # noqa: E402
    count_jsonl_rows,
    manifest_hash,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-train-count", type=int, default=400)
    parser.add_argument("--target-dev-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def read_jsonl_map(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in rows:
                raise ValueError(f"duplicate task {task_id} in {path}")
            rows[task_id] = row
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(seed: int, namespace: str, task_id: int) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{task_id}".encode()).hexdigest()


def main() -> int:
    args = parse_args()
    source_manifest = json.loads(
        (args.curated_dir / "curation_manifest.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if int(preflight["max_length"]) <= 0:
        raise ValueError("preflight report has invalid max_length")
    selections = read_jsonl_map(args.curated_dir / "selections.jsonl")
    source_paths = {
        (arm, split): args.curated_dir / arm / f"{split}.jsonl"
        for arm in ("outcome", "process")
        for split in ("train", "dev", "reserve")
    }
    source_rows = {
        key: read_jsonl_map(path) for key, path in source_paths.items()
    }

    invalid_by_file = {}
    for name, report in preflight["files"].items():
        invalid_by_file[name] = {
            int(row["task_id"]): row for row in report.get("failures", [])
        }
    invalid_by_split = {}
    for split in ("train", "dev", "reserve"):
        invalid_by_split[split] = set(
            invalid_by_file.get(f"outcome_{split}", {})
        ) | set(invalid_by_file.get(f"process_{split}", {}))

    kept = {
        split: sorted(
            set(source_rows[("outcome", split)]) - invalid_by_split[split]
        )
        for split in ("train", "dev")
    }
    reserve_candidates = sorted(
        set(source_rows[("outcome", "reserve")]) - invalid_by_split["reserve"],
        key=lambda task_id: stable_key(args.seed, "reserve", task_id),
    )
    if set(source_rows[("outcome", "reserve")]) != set(
        source_rows[("process", "reserve")]
    ):
        raise ValueError("Outcome/Process Reserve task sets differ")

    targets = {"train": args.target_train_count, "dev": args.target_dev_count}
    replacements = []
    # Preserve the full Development size first; it is the checkpoint-selection set.
    for split in ("dev", "train"):
        missing_slots = [
            selections[task_id]["difficulty"]
            for task_id in sorted(invalid_by_split[split])
        ]
        while len(kept[split]) < targets[split] and reserve_candidates:
            desired = missing_slots.pop(0) if missing_slots else None
            matching = [
                task_id
                for task_id in reserve_candidates
                if selections[task_id]["difficulty"] == desired
            ]
            task_id = matching[0] if matching else reserve_candidates[0]
            reserve_candidates.remove(task_id)
            kept[split].append(task_id)
            replacements.append(
                {
                    "task_id": task_id,
                    "from": "reserve",
                    "to": split,
                    "difficulty": selections[task_id]["difficulty"],
                    "requested_difficulty": desired,
                }
            )
        kept[split].sort()

    if set(kept["train"]) & set(kept["dev"]):
        raise ValueError("final train/dev overlap")
    output_paths = {
        (arm, split): args.output_dir / arm / f"{split}.jsonl"
        for arm in ("outcome", "process")
        for split in ("train", "dev")
    }
    for arm in ("outcome", "process"):
        for split in ("train", "dev"):
            rows = []
            for task_id in kept[split]:
                source_split = selections[task_id]["split"]
                rows.append(source_rows[(arm, source_split)][task_id])
            write_jsonl(output_paths[(arm, split)], rows)

    final_task_sets = {
        (arm, split): set(read_jsonl_map(path))
        for (arm, split), path in output_paths.items()
    }
    for split in ("train", "dev"):
        if final_task_sets[("outcome", split)] != final_task_sets[("process", split)]:
            raise ValueError(f"Outcome/Process final {split} task sets differ")

    difficulty_counts = {}
    for split in ("train", "dev"):
        difficulty_counts[split] = dict(
            sorted(Counter(selections[task_id]["difficulty"] for task_id in kept[split]).items())
        )
    excluded = {}
    for split in ("train", "dev", "reserve"):
        excluded[split] = []
        for task_id in sorted(invalid_by_split[split]):
            excluded[split].append(
                {
                    "task_id": task_id,
                    "difficulty": selections[task_id]["difficulty"],
                    "outcome_failure": invalid_by_file.get(
                        f"outcome_{split}", {}
                    ).get(task_id),
                    "process_failure": invalid_by_file.get(
                        f"process_{split}", {}
                    ).get(task_id),
                }
            )

    manifest = {
        "schema_version": "commerce-sft-ready-manifest-v1",
        "selection_policy": "shared-union-length-gate-then-difficulty-matched-reserve-v1",
        "seed": args.seed,
        "max_length": preflight["max_length"],
        "inputs": {
            "curation_manifest_sha256": source_manifest["manifest_sha256"],
            "preflight_report_sha256": preflight["report_sha256"],
            "preflight_report_file_sha256": sha256_file(args.preflight_report),
        },
        "excluded": excluded,
        "reserve_replacements": replacements,
        "unused_valid_reserve_task_ids": reserve_candidates,
        "final": {
            "train_tasks": len(kept["train"]),
            "dev_tasks": len(kept["dev"]),
            "train_dev_overlap": 0,
            "outcome_process_task_sets_equal": True,
            "difficulty_counts": difficulty_counts,
        },
        "outputs": {},
    }
    for (arm, split), path in output_paths.items():
        manifest["outputs"][f"{arm}_{split}"] = {
            "path": str(path),
            "rows": count_jsonl_rows(path),
            "sha256": sha256_file(path),
        }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sft_ready_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
