"""Export actor-private split facts from the locked ShopSimulator product blob."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_COMMIT = "4ed73020e1d7d07eb93e7375a4606b0901d3cded"
UPSTREAM_URL = "https://github.com/YYHDBL/shopping-grpo-longhorizon.git"
PRODUCT_SOURCE = (
    "environments/ShopSimulator/shop_env/data/"
    "fine_items_eval_train_all.json.gz"
)
PRODUCT_DATA_SHA256 = (
    "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
)
EXTRACTOR_VERSION = "commerce-task-facts-exporter-v1"
MODEL_TOKEN = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+-]{2,24}(?![a-z0-9]))"
    r"(?=[a-z0-9._+-]*\d)[a-z0-9._+-]+",
    flags=re.IGNORECASE,
)
BUDGET = re.compile(r"预算|价格|\d+(?:\.\d+)?\s*(?:元|块|万|千|[kK])")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def numeric_template(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "<num>", text)
    return re.sub(r"[^\w\u4e00-\u9fff<>]+", "", text)


def product_family(item: dict) -> str:
    category = canonical_text(item.get("category"))
    title = numeric_template(item.get("title"))
    if not category or not title:
        raise ValueError(f"product {item.get('asin')!r} lacks category/title family keys")
    return sha256_bytes(f"{category}|{title}".encode("utf-8"))


def explicit_model_tokens(query: str, item: dict) -> list[str]:
    query_tokens = {token.casefold() for token in MODEL_TOKEN.findall(query)}
    target_text = " ".join(
        str(value)
        for value in (item.get("title"), item.get("full_description"))
        if value
    )
    target_tokens = {token.casefold() for token in MODEL_TOKEN.findall(target_text)}
    return sorted(query_tokens.intersection(target_tokens))


def difficulty_bucket(instruction: dict, model_tokens: list[str]) -> str:
    query = str(instruction.get("instruction") or "")
    count = len(instruction.get("attributes") or [])
    count += len(instruction.get("instruction_options") or [])
    count += len(model_tokens)
    count += int(bool(BUDGET.search(query)))
    if count <= 3:
        return "easy"
    if count <= 6:
        return "medium"
    return "hard"


def load_products(repository: Path, commit: str) -> tuple[list[dict], str]:
    try:
        compressed = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{PRODUCT_SOURCE}"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot read locked product data: {detail}") from exc
    decompressed = gzip.decompress(compressed)
    digest = sha256_bytes(decompressed)
    if digest != PRODUCT_DATA_SHA256:
        raise RuntimeError(
            f"product data hash mismatch: expected {PRODUCT_DATA_SHA256}, got {digest}"
        )
    products = json.loads(decompressed)
    if not isinstance(products, list) or not products:
        raise ValueError("ShopSimulator product data must be a non-empty array")
    return products, sha256_bytes(compressed)


def build_rows(products: list[dict]) -> list[dict]:
    rows = []
    for item_index, item in enumerate(products):
        instructions = item.get("instructions") or []
        if len(instructions) != 1 or not instructions[0].get("attributes"):
            raise ValueError(
                f"product index {item_index} breaks the frozen one-goal-per-product mapping"
            )
        instruction = instructions[0]
        query = str(instruction.get("instruction") or "").strip()
        asin = str(item.get("asin") or "").strip()
        if not query or not asin:
            raise ValueError(f"product index {item_index} lacks query or ASIN")
        models = explicit_model_tokens(query, item)
        rows.append(
            {
                "schema_version": "commerce-task-facts-v1",
                "task_id": item_index,
                "query": query,
                "target_product_ids": [asin],
                "target_title": str(item.get("title") or "").strip(),
                "target_options": sorted(
                    {
                        str(value).strip()
                        for value in instruction.get("instruction_options") or []
                        if str(value).strip()
                    }
                ),
                "model_tokens": models,
                "product_family": product_family(item),
                "difficulty": difficulty_bucket(instruction, models),
            }
        )
    return rows


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream",
        type=Path,
        default=ROOT,
    )
    parser.add_argument("--commit", default=LOCKED_COMMIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "private" / "task_facts.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "task_facts_source.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.upstream.resolve()
    resolved = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", args.commit], text=True
    ).strip()
    if resolved != args.commit:
        raise RuntimeError(f"requested commit resolved to {resolved}, expected {args.commit}")
    products, compressed_hash = load_products(repository, args.commit)
    rows = build_rows(products)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write(args.output, content)
    manifest = {
        "schema_version": "commerce-task-facts-source-v1",
        "extractor_version": EXTRACTOR_VERSION,
        "extractor_sha256": sha256_file(Path(__file__)),
        "upstream_repository": UPSTREAM_URL,
        "upstream_commit": args.commit,
        "source_path": PRODUCT_SOURCE,
        "source_compressed_sha256": compressed_hash,
        "source_decompressed_sha256": PRODUCT_DATA_SHA256,
        "row_count": len(rows),
        "task_facts_sha256": sha256_file(args.output),
        "private_output": "data/private/task_facts.jsonl",
    }
    atomic_write(
        args.manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
