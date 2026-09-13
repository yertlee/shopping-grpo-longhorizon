"""Append-only JSONL store with a content-addressed completion index."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from commerce_posttrain.collection.schema import (
    canonical_json_bytes,
    validate_raw_trajectory,
)


class RawTrajectoryStore:
    def __init__(self, raw_path: str | Path, index_path: str | Path):
        self.raw_path = Path(raw_path)
        self.index_path = Path(index_path)
        self._lock = Lock()
        self._index = self.rebuild_index()

    def rebuild_index(self) -> dict[str, dict]:
        index: dict[str, dict] = {}
        if self.raw_path.exists():
            with self.raw_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = validate_raw_trajectory(json.loads(line))
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid raw trajectory at line {line_number}: {exc}"
                        ) from exc
                    request_id = row["request_id"]
                    previous = index.get(request_id)
                    if previous and previous["status"] != "infrastructure_failure":
                        raise ValueError(f"duplicate completed request_id: {request_id}")
                    index[request_id] = {
                        "line_number": line_number,
                        "status": row["status"],
                        "trajectory_sha256": row["trajectory_sha256"],
                    }
        self._write_index(index)
        return index

    def completed_request_ids(self) -> set[str]:
        return {
            request_id
            for request_id, record in self._index.items()
            if record["status"] != "infrastructure_failure"
        }

    def read_all(self) -> list[dict]:
        """Return every validated raw row in append order."""
        rows: list[dict] = []
        if not self.raw_path.exists():
            return rows
        with self.raw_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(validate_raw_trajectory(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid raw trajectory at line {line_number}: {exc}"
                    ) from exc
        return rows

    def append(self, trajectory: Mapping[str, Any]) -> None:
        row = validate_raw_trajectory(trajectory)
        encoded = canonical_json_bytes(row) + b"\n"
        with self._lock:
            request_id = row["request_id"]
            previous = self._index.get(request_id)
            if previous and previous["status"] != "infrastructure_failure":
                raise ValueError(f"request already completed: {request_id}")
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.raw_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            line_number = 1 + max(
                (record["line_number"] for record in self._index.values()), default=0
            )
            self._index[request_id] = {
                "line_number": line_number,
                "status": row["status"],
                "trajectory_sha256": row["trajectory_sha256"],
            }
            self._write_index(self._index)

    def _write_index(self, index: Mapping[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "commerce-attempt-completion-index-v1",
            "raw_path": str(self.raw_path),
            "request_count": len(index),
            "completed_count": sum(
                record["status"] != "infrastructure_failure"
                for record in index.values()
            ),
            "requests": dict(sorted(index.items())),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.index_path.parent,
            prefix=self.index_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, self.index_path)
