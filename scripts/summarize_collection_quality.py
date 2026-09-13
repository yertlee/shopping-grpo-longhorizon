"""Report raw and dataset-valid Teacher collection denominators side by side."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.collection.schema import validate_raw_trajectory  # noqa: E402
from commerce_posttrain.data_quality.reachability import (  # noqa: E402
    validate_reachability_manifest,
)
from commerce_posttrain.data_quality.summary import (  # noqa: E402
    summarize_collection_quality,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--reachability",
        type=Path,
        default=ROOT / "data/manifests/task_reachability_manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reachability = validate_reachability_manifest(
        json.loads(args.reachability.read_text(encoding="utf-8"))
    )
    trajectories = []
    with args.raw.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                trajectories.append(validate_raw_trajectory(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid raw row {line_number}: {exc}") from exc
    summary = summarize_collection_quality(
        trajectories,
        reachability_manifest=reachability,
        raw_sha256=sha256_file(args.raw),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
