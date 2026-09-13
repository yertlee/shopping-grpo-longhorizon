"""Shared Teacher collection driver (Smoke / Pilot / Full).

Runs teacher-pool tasks x attempts against the running ShopSimulator using the
DeepSeek Teacher, writing content-addressed raw trajectories plus a completion
index to ``outputs/``. Resume is automatic: completed ``request_id`` entries are
skipped on re-run, and infrastructure failures are retried.

Credentials are read from the environment, optionally loaded from a git-ignored
``.env`` file at the repo root::

  DEEPSEEK_BASE_URL=https://api.deepseek.com
  DEEPSEEK_API_KEY=sk-...
  MODEL_PATH=/path/to/authorized/model

The Teacher model, temperature, top-p, thinking switch and token budgets are
frozen in ``data/manifests/runtime_contract.json`` and are not overridable here.

Examples::

  python scripts/collect_teacher.py --task-count 20 --attempts 3   # Smoke-20x3
  python scripts/collect_teacher.py --task-count 100 --attempts 3  # Pilot-100x3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commerce_posttrain.collection.client import (  # noqa: E402
    HuggingFaceTokenCounter,
    OpenAICompatibleTeacher,
)
from commerce_posttrain.collection.collector import collect_attempt  # noqa: E402
from commerce_posttrain.collection.schema import (  # noqa: E402
    request_identity,
    strict_gold_success,
)
from commerce_posttrain.collection.store import RawTrajectoryStore  # noqa: E402
from commerce_posttrain.data_quality.reachability import (  # noqa: E402
    validate_reachability_manifest,
)
from commerce_posttrain.environment.replay import load_private_tasks  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from ``path`` without overriding real env vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent attempts (ShopSimulator exposes ~4 leases).",
    )
    parser.add_argument("--env-base-url", default="http://127.0.0.1:5700")
    parser.add_argument(
        "--teacher-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--teacher-api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data/private/task_facts.jsonl"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data/manifests/split_manifest.json"
    )
    parser.add_argument("--split-name", default="teacher_pool")
    parser.add_argument(
        "--reachability",
        type=Path,
        default=ROOT / "data/manifests/task_reachability_manifest.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the eligible-task and resume plan without creating an API client.",
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=ROOT / "data/manifests/runtime_contract.json",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=ROOT / "configs/runtime/system_prompt.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/teacher")
    return parser.parse_args(argv)


def build_teacher(args: argparse.Namespace, runtime_contract: dict) -> OpenAICompatibleTeacher:
    if not args.teacher_api_key:
        raise SystemExit("missing Teacher API key: set DEEPSEEK_API_KEY or --teacher-api-key")
    if not args.model_path:
        raise SystemExit("missing Qwen tokenizer path: set MODEL_PATH or --model-path")
    teacher = dict(runtime_contract["teacher"])
    token_counter = HuggingFaceTokenCounter(model_path=args.model_path)
    return OpenAICompatibleTeacher(
        model=teacher["model"],
        base_url=args.teacher_base_url,
        api_key=args.teacher_api_key,
        temperature=teacher.get("temperature", 0.7),
        top_p=teacher.get("top_p", 0.9),
        max_tokens=int(runtime_contract["max_generated_tokens_per_turn"]),
        timeout=args.timeout,
        token_counter=token_counter,
    )


def summarize(trajectories: list[dict]) -> dict:
    status: dict[str, int] = {}
    reward_types: dict[str, int] = {}
    gold = 0
    infra = 0
    released = 0
    for traj in trajectories:
        status[traj["status"]] = status.get(traj["status"], 0) + 1
        if strict_gold_success(traj):
            gold += 1
        if traj.get("released") is True:
            released += 1
        if traj["status"] == "infrastructure_failure":
            infra += 1
        reward = (traj.get("terminal_environment_result") or {}).get("reward_detail") or {}
        reward_types[reward.get("reward_type", "none")] = (
            reward_types.get(reward.get("reward_type", "none"), 0) + 1
        )
    total = len(trajectories)
    return {
        "total": total,
        "gold_purchase": gold,
        "gold_purchase_rate": round(gold / total, 4) if total else 0.0,
        "infrastructure_failures": infra,
        "infrastructure_failure_rate": round(infra / total, 4) if total else 0.0,
        "released": released,
        "status": status,
        "reward_types": reward_types,
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    reachability = validate_reachability_manifest(
        json.loads(args.reachability.read_text(encoding="utf-8")),
        expected_split_manifest_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
    )
    if args.workers != 1:
        raise SystemExit("Teacher collection is frozen to --workers 1")
    if args.attempts != runtime["attempts_per_task"]:
        raise SystemExit(
            "attempt count must match the frozen runtime contract: "
            f"{runtime['attempts_per_task']}"
        )
    if args.split_name not in split["splits"]:
        raise SystemExit(f"unknown split: {args.split_name}")
    quality_split = reachability["splits"].get(args.split_name)
    if quality_split is None:
        raise SystemExit(f"reachability manifest lacks split: {args.split_name}")

    task_rows = load_private_tasks(args.tasks)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    split_task_ids = [int(value) for value in split["splits"][args.split_name]["task_ids"]]
    classified = {
        int(value) for value in quality_split["eligible_task_ids"]
    } | {int(row["task_id"]) for row in quality_split["unreachable"]}
    if classified != set(split_task_ids):
        raise SystemExit(
            "reachability manifest classification does not match selected split"
        )
    eligible = set(int(value) for value in quality_split["eligible_task_ids"])
    task_ids = [task_id for task_id in split_task_ids if task_id in eligible][: args.task_count]
    if len(task_ids) != args.task_count:
        raise SystemExit(
            f"split {args.split_name} contains only {len(task_ids)} eligible tasks"
        )
    tasks = [task_rows[int(task_id)] for task_id in task_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = RawTrajectoryStore(
        args.output_dir / "raw.jsonl",
        args.output_dir / "completion_index.json",
    )

    jobs = [
        (task, attempt)
        for task in tasks
        for attempt in range(args.attempts)
    ]
    teacher = dict(runtime["teacher"])
    completed = store.completed_request_ids()
    pending_jobs = []
    for task, attempt in jobs:
        request_id = request_identity(
            task_id=int(task["task_id"]),
            attempt_index=attempt,
            runtime_contract_sha256=runtime["contract_sha256"],
            teacher=teacher,
        )
        if request_id not in completed:
            pending_jobs.append((task, attempt))

    plan = {
        "mode": "dry_run" if args.dry_run else "collect",
        "split_name": args.split_name,
        "reachability_manifest_sha256": reachability["manifest_sha256"],
        "requested_eligible_tasks": args.task_count,
        "selected_task_ids": task_ids,
        "excluded_unreachable_count": quality_split["unreachable_count"],
        "attempts_per_task": args.attempts,
        "total_jobs": len(jobs),
        "already_completed_jobs": len(jobs) - len(pending_jobs),
        "pending_api_jobs": len(pending_jobs),
        "workers": args.workers,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    client = build_teacher(args, runtime)

    def run(job: tuple[dict, int]) -> dict:
        task, attempt = job
        request_id = request_identity(
            task_id=int(task["task_id"]),
            attempt_index=attempt,
            runtime_contract_sha256=runtime["contract_sha256"],
            teacher=teacher,
        )
        if request_id in completed:
            return {"skipped": True, "request_id": request_id}
        traj = collect_attempt(
            task,
            attempt_index=attempt,
            runtime_contract=runtime,
            client=client,
            base_url=args.env_base_url,
            system_prompt=system_prompt,
        )
        store.append(traj)
        return {"trajectory": traj}

    results: list[dict] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(run, jobs))
    else:
        for index, job in enumerate(jobs, start=1):
            result = run(job)
            results.append(result)
            if "trajectory" in result:
                traj = result["trajectory"]
                print(
                    f"[{index}/{len(jobs)}] task={traj['task_id']} "
                    f"attempt={traj['attempt_index']} status={traj['status']} "
                    f"gold={strict_gold_success(traj)}"
                )
            else:
                print(f"[{index}/{len(jobs)}] skipped (resume)")

    # Always summarize the complete append-only artifact. On a resumed run,
    # ``results`` contains only newly collected rows plus skip markers.
    summary = summarize(store.read_all())
    summary["runtime_contract_sha256"] = runtime["contract_sha256"]
    summary["teacher_model"] = teacher["model"]
    summary["requested_tasks"] = args.task_count
    summary["attempts_per_task"] = args.attempts
    summary["split_name"] = args.split_name
    summary["reachability_manifest_sha256"] = reachability["manifest_sha256"]
    summary["excluded_unreachable_count"] = quality_split["unreachable_count"]

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
