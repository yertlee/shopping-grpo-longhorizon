"""Build or verify the frozen runtime contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.contracts.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    validate_runtime_contract,
    write_runtime_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        contract = validate_runtime_contract(args.output, args.config, project_root=ROOT)
    else:
        contract = write_runtime_contract(args.output, args.config, project_root=ROOT)
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
