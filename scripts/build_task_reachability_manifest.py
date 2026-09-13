"""Build a frozen option-reachability manifest from the locked upstream."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.data_quality.reachability import (  # noqa: E402
    build_reachability_manifest,
    validate_reachability_manifest,
)

LOCKED_COMMIT = "4ed73020e1d7d07eb93e7375a4606b0901d3cded"
UPSTREAM_URL = "https://github.com/YYHDBL/shopping-grpo-longhorizon.git"
PRODUCT_SOURCE = (
    "environments/ShopSimulator/shop_env/data/"
    "fine_items_eval_train_all.json.gz"
)
REWARD_FEATURES_SOURCE = (
    "environments/ShopSimulator/shop_env/web_agent_site/engine/"
    "reward_features.py"
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:{path}"]
    )


def load_reward_normalizer(repository: Path, commit: str):
    resolved = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved != commit:
        raise RuntimeError(f"upstream HEAD is {resolved}, expected locked {commit}")
    blob = git_blob(repository, commit, REWARD_FEATURES_SOURCE)
    differs = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--quiet",
            commit,
            "--",
            REWARD_FEATURES_SOURCE,
        ],
        check=False,
    ).returncode
    if differs != 0:
        raise RuntimeError("reward_features.py worktree differs from locked commit")
    shop_env = repository / "environments/ShopSimulator/shop_env"
    sys.path.insert(0, str(shop_env))
    module = importlib.import_module("web_agent_site.engine.reward_features")
    return module.normalize_option_text, module.REWARD_FEATURE_VERSION, sha256_bytes(blob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", type=Path, default=ROOT,
    )
    parser.add_argument("--commit", default=LOCKED_COMMIT)
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data/manifests/split_manifest.json"
    )
    parser.add_argument(
        "--task-facts", type=Path, default=ROOT / "data/private/task_facts.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/manifests/task_reachability_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.upstream.resolve()
    normalize, reward_version, reward_hash = load_reward_normalizer(
        repository, args.commit
    )
    product_blob = git_blob(repository, args.commit, PRODUCT_SOURCE)
    products = json.loads(gzip.decompress(product_blob))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    manifest = build_reachability_manifest(
        products=products,
        split_manifest=split,
        normalize_option_text=normalize,
        upstream_repository=UPSTREAM_URL,
        upstream_commit=args.commit,
        product_source_path=PRODUCT_SOURCE,
        product_source_compressed_sha256=sha256_bytes(product_blob),
        product_source_decompressed_sha256=sha256_bytes(gzip.decompress(product_blob)),
        reward_features_path=REWARD_FEATURES_SOURCE,
        reward_features_sha256=reward_hash,
        reward_feature_version=reward_version,
        verifier_sha256=sha256_file(
            ROOT / "src/commerce_posttrain/data_quality/reachability.py"
        ),
        task_facts_sha256=sha256_file(args.task_facts),
        split_manifest_sha256=sha256_file(args.split),
    )
    validate_reachability_manifest(
        manifest, expected_split_manifest_sha256=sha256_file(args.split)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": manifest["manifest_sha256"],
                "selected_task_count": manifest["selected_task_count"],
                "eligible_task_count": manifest["eligible_task_count"],
                "unreachable_task_count": manifest["unreachable_task_count"],
                "split_counts": {
                    name: {
                        "eligible": value["eligible_count"],
                        "unreachable": value["unreachable_count"],
                    }
                    for name, value in manifest["splits"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
