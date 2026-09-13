"""Build and validate the cross-stage ShopSimulator runtime contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from commerce_posttrain.curation.process_contract import (
    ACTOR_VISIBLE_FEATURES,
    EVIDENCE_SOURCE,
    PROCESS_CONTRACT_VERSION,
    PROCESS_SELECTION_ORDER,
)
from shopping_grpo.environment.manifest import validate_manifest as validate_environment_manifest
from shopping_grpo.environment.projection import PROJECTION_CONTRACT_VERSION
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

CONTRACT_SCHEMA_VERSION = "commerce-runtime-contract-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runtime" / "runtime.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "runtime_contract.json"
# Provenance identity is part of the checked-in contract, but the integrated
# repository does not require a sibling checkout or a source-map file at run
# time.  Keep the public source identity stable while validating the canonical
# environment manifest in this repository.
UPSTREAM_REPOSITORY = "https://github.com/YYHDBL/shopping-grpo-longhorizon.git"
UPSTREAM_COMMIT = "4ed73020e1d7d07eb93e7375a4606b0901d3cded"
UPSTREAM_SOURCE_MANIFEST_HASH = "9b8117408cd7230bd0b6dd5faa7c1867f1d5d51f4d3bbe6850b698470f78af28"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_crlf_text(path: str | Path) -> str:
    """Hash text with CRLF newlines to preserve the frozen v1 contract ID."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(content.replace(b"\n", b"\r\n"))


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _project_path(project_root: Path, value: str, *, field: str) -> Path:
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} resolves outside project root: {value}") from exc
    if not path.is_file():
        raise ValueError(f"{field} file does not exist: {path}")
    return path


def validate_source_map(source_map: Mapping[str, Any], *, project_root: Path) -> None:
    if source_map.get("schema_version") != "commerce-upstream-source-map-v1":
        raise ValueError("unsupported upstream source-map schema")
    commit = source_map.get("upstream_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("upstream source map has no full Git commit")
    imports = source_map.get("imports")
    if not isinstance(imports, list) or not imports:
        raise ValueError("upstream source map has no imported files")
    seen = set()
    for entry in imports:
        if not isinstance(entry, dict):
            raise ValueError("upstream source-map entries must be objects")
        destination_path = entry.get("destination_path")
        if not isinstance(destination_path, str) or destination_path in seen:
            raise ValueError("upstream source map has duplicate or invalid destination")
        seen.add(destination_path)
        destination = _project_path(project_root, destination_path, field="destination_path")
        actual = sha256_file(destination)
        if actual != entry.get("destination_sha256"):
            raise ValueError(f"imported file hash mismatch: {destination_path}")
        if entry.get("transform") == "exact_bytes" and actual != entry.get("source_sha256"):
            raise ValueError(f"exact upstream import changed bytes: {destination_path}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "commerce-runtime-config-v1":
        raise ValueError("unsupported runtime-config schema")
    positive_ints = (
        "attempts_per_task",
        "context_window",
        "context_safety_margin",
        "max_generated_tokens_per_turn",
        "max_steps",
        "observation_search_tokens",
        "observation_detail_tokens",
        "observation_generic_tokens",
        "observation_search_top_k",
    )
    for field in positive_ints:
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"runtime config {field} must be a positive integer")
    if config["attempts_per_task"] != 3:
        raise ValueError("core runtime requires exactly three attempts per task")
    if config["max_steps"] != 35:
        raise ValueError("ShopSimulator runtime requires max_steps=35")
    if config["observation_search_top_k"] != 20:
        raise ValueError("observation search top-k must equal the frozen page size 20")
    reserved_tokens = (
        config["context_safety_margin"] + config["max_generated_tokens_per_turn"]
    )
    if reserved_tokens >= config["context_window"]:
        raise ValueError("context reserve leaves no positive input budget")
    teacher = config.get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("model"):
        raise ValueError("runtime config requires a Teacher model")


def build_runtime_contract(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    _validate_config(config)

    environment_path = _project_path(
        project_root,
        str(config["environment_manifest"]),
        field="environment_manifest",
    )
    prompt_path = _project_path(
        project_root,
        str(config["system_prompt"]),
        field="system_prompt",
    )
    environment = validate_environment_manifest(_load_json(environment_path))

    action_guard_path = project_root / "src" / "shopping_grpo" / "environment" / "actions.py"
    projection_path = project_root / "src" / "shopping_grpo" / "environment" / "projection.py"
    process_contract_path = (
        project_root / "src" / "commerce_posttrain" / "curation" / "process_contract.py"
    )
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "environment_version": environment.get(
            "environment_version", "shopsimulator-environment-v2.1"
        ),
        "reward_version": environment["reward"]["version"],
        "observation_version": environment["observation_version"],
        "tool_version": environment["tool_version"],
        "shopsimulator_commit": environment["shopsimulator_commit"],
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        # The v1 contract was first frozen from a Windows checkout. Canonical
        # CRLF hashing preserves that identity while making validation portable.
        "upstream_source_manifest_hash": UPSTREAM_SOURCE_MANIFEST_HASH,
        "tool_schema_hash": sha256_bytes(canonical_json_bytes(SHOP_TOOL_SCHEMAS)),
        "action_guard_hash": sha256_file(action_guard_path),
        "system_prompt_hash": sha256_file(prompt_path),
        "observation_projection_hash": sha256_file(projection_path),
        "observation_projection_contract": PROJECTION_CONTRACT_VERSION,
        "process_analyzer_hash": sha256_file(process_contract_path),
        "process_contract": {
            "version": PROCESS_CONTRACT_VERSION,
            "evidence_source": EVIDENCE_SOURCE,
            "features_hash": sha256_bytes(canonical_json_bytes(ACTOR_VISIBLE_FEATURES)),
            "selection_order": list(PROCESS_SELECTION_ORDER),
        },
        "product_data_hash": environment["product_data_sha256"],
        "search_config_hash": sha256_bytes(canonical_json_bytes(environment["search"])),
        "search_version": environment["search"]["version"],
        "attempts_per_task": config["attempts_per_task"],
        "max_steps": config["max_steps"],
        "context_window": config["context_window"],
        "context_safety_margin": config["context_safety_margin"],
        "max_generated_tokens_per_turn": config["max_generated_tokens_per_turn"],
        "observation_search_tokens": config["observation_search_tokens"],
        "observation_detail_tokens": config["observation_detail_tokens"],
        "observation_generic_tokens": config["observation_generic_tokens"],
        "observation_search_top_k": config["observation_search_top_k"],
        "teacher": config["teacher"],
    }
    payload["contract_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_runtime_contract(
    contract: Mapping[str, Any] | str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    observed = _load_json(Path(contract)) if isinstance(contract, (str, Path)) else dict(contract)
    expected = build_runtime_contract(config_path, project_root=project_root)
    if observed != expected:
        keys = sorted(set(observed) | set(expected))
        changed = [key for key in keys if observed.get(key) != expected.get(key)]
        raise ValueError("runtime contract mismatch in: " + ", ".join(changed))
    return observed


def write_runtime_contract(
    output_path: str | Path = DEFAULT_OUTPUT,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    contract = build_runtime_contract(config_path, project_root=project_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    validate_runtime_contract(output, config_path, project_root=project_root)
    return contract
