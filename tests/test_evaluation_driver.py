"""统一 Student Evaluator driver 的无网络测试（WP1）。

全部使用 fake actor / fake 环境 / fake curator / fake judge；不调用真实 API，
不依赖 torch/transformers。覆盖 GAPS §5.5 要求的场景：固定分母、缺题、
duplicate/unknown task、hash 不匹配 resume、Final blind guard、
infra-invalid not_judged、缓存命中/中断恢复、release 异常、Judge 非法 schema、
模型 label 泄漏检查。
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_student import build_driver, parse_args
from shopping_grpo.evaluation.blind_guard import (
    validate_blind_asset_file,
    validate_canonical_blind_asset,
)
from shopping_grpo.evaluation.artifacts import ArtifactError
from shopping_grpo.evaluation.driver import (
    CACHE_BINDING_VERSION,
    DRIVER_VERSION,
    BlindFinalGuardError,
    DriverError,
    EvaluationDriver,
    ResumeContractError,
    default_facts_builder,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.metrics import (
    INFRASTRUCTURE_ERROR_TYPES,
    extend_infrastructure_error_types,
)
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION
from shopping_grpo.evaluation.rubric import build_task_facts
from shopping_grpo.environment.client import ShopEnvironmentError
from shopping_grpo.training.sft.run_manifest import hash_weight_files


QUERY = "请推荐一台海信65W快充移动电源，预算不超过300元"

OBSERVATION = (
    "搜索功能是否可用: True\n"
    "1|B0TEST00001|海信65W快充移动电源\n"
    '可点击的按钮: ["Buy Now"]'
)

GOLD_RESULT = {
    "observation": "购买成功",
    "reward": 1.0,
    "done": True,
    "over": True,
    "reward_detail": {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "reward_valid": True,
        "purchase_success": True,
        "termination_reason": "gold_purchase",
        "terminal_utility": 1.0,
        "weighted_score": 1.0,
    },
    "purchase": {"asin": "B0TEST00001"},
}

GRACEFUL_RESULT = {
    "observation": "任务结束",
    "reward": -0.15,
    "done": True,
    "over": True,
    "reward_detail": {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "graceful_stop",
        "reward_valid": True,
        "purchase_success": False,
        "termination_reason": "graceful_stop",
        "terminal_utility": -0.15,
        "weighted_score": 0.0,
    },
}


def _tool_call(index, name, **arguments):
    return {
        "id": f"call-{index:03d}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _script(name):
    if name == "abstain":
        return [
            _tool_call(0, "search_products", query="移动电源"),
            _tool_call(
                1, "finish_without_purchase", reason="no_suitable_product"
            ),
        ]
    return [
        _tool_call(0, "search_products", query="移动电源"),
        _tool_call(1, "buy_now"),
    ]


class FakeActor:
    """按 instruction 分发的 fake actor；每个 task 使用唯一 instruction。"""

    def __init__(self):
        self.scripts = {}
        self.calls = []
        self._cursor = {}

    def complete(self, messages, tools):
        self.calls.append([dict(message) for message in messages])
        instruction = next(
            message["content"] for message in messages if message["role"] == "user"
        )
        script = self.scripts[instruction]
        index = self._cursor.get(instruction, 0)
        self._cursor[instruction] = index + 1
        if index >= len(script):
            index = len(script) - 1
        return {"role": "assistant", "content": None, "tool_calls": [script[index]]}


class PoisonActor(FakeActor):
    def complete(self, messages, tools):
        raise AssertionError("cache hit 时不应调用 actor")


class _FakeEnv:
    def __init__(self, factory):
        self._factory = factory
        self._task_id = None

    def reset(self, task_id):
        self._task_id = int(task_id)
        return {
            "env_idx": self._task_id,
            "instruction": self._factory.instructions[self._task_id],
            "observation_state": None,
            "environment_version": "shopsimulator-environment-v2.1",
            "reward_version": "shopsimulator-reward-v3",
        }

    def step(self, action):
        if self._task_id in self._factory.fail_step_for:
            raise ShopEnvironmentError(
                "Unable to get available environment resource"
            )
        if self._task_id in self._factory.fail_step_for_custom:
            raise self._factory.fail_step_for_custom[self._task_id]()
        if action == "click[Buy Now]":
            return dict(GOLD_RESULT)
        if action.startswith("finish["):
            return dict(GRACEFUL_RESULT)
        return {"observation": OBSERVATION, "reward": 0.0, "done": False}

    def release(self):
        if self._task_id in self._factory.fail_release_for:
            raise RuntimeError("release failed")
        self._factory.released.append(self._task_id)


class FakeEnvFactory:
    def __init__(
        self,
        instructions,
        fail_release_for=frozenset(),
        fail_step_for=frozenset(),
        fail_step_for_custom=None,
    ):
        self.instructions = instructions
        self.fail_release_for = set(fail_release_for)
        self.fail_step_for = set(fail_step_for)
        self.fail_step_for_custom = dict(fail_step_for_custom or {})
        self.released = []

    def __call__(self, base_url):
        return _FakeEnv(self)


class PoisonEnvFactory:
    def __call__(self, base_url):
        raise AssertionError("cache hit 时不应创建环境")


class FakeCuratorClient:
    def __init__(self, fail_for_task_ids=frozenset()):
        self.calls = []
        self.fail_for_task_ids = set(fail_for_task_ids)

    def complete_json(self, messages):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        if payload["task_id"] in self.fail_for_task_ids:
            raise RuntimeError("curator service unavailable")
        selected = []
        for candidate in payload["candidates"]:
            spans = candidate.get("query_spans") or []
            selected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "description": candidate.get("description_hint")
                    or candidate["candidate_id"],
                    "hardness": candidate.get("hardness_hint", "hard"),
                    "query_quote": spans[0]["text"] if spans else None,
                    "selection_reason": "query 原文直接支持该约束",
                }
            )
        return {"result": {"selected_constraints": selected}, "metadata": {}}


class PoisonCuratorClient(FakeCuratorClient):
    def complete_json(self, messages):
        raise AssertionError("cache hit 时不应调用 curator")


class FakeJudgeClient:
    def __init__(self, *, invalid_for_task_ids=frozenset()):
        self.calls = []
        self.invalid_for_task_ids = set(invalid_for_task_ids)

    def complete_json(self, messages):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        task_id = payload["task_id"]
        if task_id in self.invalid_for_task_ids:
            # 合法形状但缺少全部 rubric assessments / dimension scores。
            return {
                "result": {
                    "schema_version": payload["required_output"]["schema_version"],
                    "task_id": task_id,
                    "trajectory_id": payload["trajectory_id"],
                    "judge_status": "valid",
                    "rubric_assessments": [],
                    "dimension_scores": {},
                    "errors": {"primary": None, "secondary": [], "evidence_event_ids": []},
                    "overall_diagnosis": "坏数据",
                },
                "metadata": {},
            }
        events = [
            event["event_id"]
            for event in payload["actor_visible_trajectory"]["events"]
        ]
        evidence = [events[0]] if events else []
        return {
            "result": {
                "schema_version": payload["required_output"]["schema_version"],
                "task_id": task_id,
                "trajectory_id": payload["trajectory_id"],
                "judge_status": "valid",
                "rubric_assessments": [
                    {
                        "rubric_id": rubric["rubric_id"],
                        "status": "satisfied",
                        "reason": "ok",
                        "evidence_event_ids": evidence,
                    }
                    for rubric in payload["rubric"]
                ],
                "dimension_scores": {
                    name: {"score": 1, "reason": "ok", "evidence_event_ids": evidence}
                    for name in payload["dimension_spec"]
                },
                "errors": {
                    "primary": None,
                    "secondary": [],
                    "evidence_event_ids": [],
                },
                "overall_diagnosis": "ok",
            },
            "metadata": {},
        }


class PoisonJudgeClient(FakeJudgeClient):
    def complete_json(self, messages):
        raise AssertionError("cache hit 时不应调用 judge")


def _task_row(task_id):
    return {
        "task_id": task_id,
        "goal": {
            "instruction_text": f"{QUERY}#{task_id}",
            "asin": "B0TEST00001",
            "attributes": ["65W快充"],
            "goal_options": [],
            "expected_brand": ["海信"],
            "expected_core_functions": ["65W快充"],
            "price_upper": 300,
        },
        "target_product": {
            "asin": "B0TEST00001",
            "category": "电子配件›移动电源",
            "title": "海信65W快充移动电源",
            "brand": "海信",
            "shop_name": "海信自营",
            "pricing": {"price": 259.0, "currency": "CNY"},
            "attribute": {"容量": "20000mAh"},
            "customization_options": {},
        },
    }


def _synthetic_facts(task):
    """Final-200 guard 测试：只读冻结 task ID 列表，不为 blind 任务伪造目标商品。"""

    task_id = int(task["task_id"])
    instruction_record = {
        "instruction": QUERY,
        "attributes": ["65W快充"],
        "instruction_options": [],
    }
    return build_task_facts(
        task_id=task_id,
        query=QUERY,
        target_product={
            "asin": "B0SYNTH000",
            "category": "电子配件›移动电源",
            "brand": "合成品牌",
            "pricing": {"price": 259.0, "currency": "CNY"},
        },
        instruction_record=instruction_record,
        reward_goal={
            "instruction_text": QUERY,
            "expected_brand": ["海信"],
            "expected_core_functions": ["65W快充"],
        },
    )


_UNSET_RUNTIME_CONTRACT = object()


class Harness:
    """合成分割 + 全套 fake 依赖的 driver 装配器。"""

    def __init__(
        self,
        tmp,
        task_ids=(101, 102, 103),
        *,
        behavior=None,
        fail_release_for=frozenset(),
    ):
        self.task_ids = list(task_ids)
        self.behavior = dict(behavior or {})
        self.instructions = {
            task_id: f"{QUERY}#{task_id}" for task_id in self.task_ids
        }
        self.actor = FakeActor()
        for task_id in self.task_ids:
            self.actor.scripts[self.instructions[task_id]] = _script(
                self.behavior.get(task_id, "gold")
            )
        self.curator = FakeCuratorClient()
        self.judge = FakeJudgeClient()
        self.env_factory = FakeEnvFactory(
            self.instructions, fail_release_for=fail_release_for
        )

    def write_split(self, name="tasks.jsonl"):
        path = Path(self.tmp) / name
        rows = [_task_row(task_id) for task_id in self.task_ids]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def build_driver(self, run_dir, *, split="dev", resume=False,
                     allow_blind_final=False, curator=None, judge=None,
                     actor=None, env_factory=None, max_steps=35,
                     task_split_path=None, facts_builder=None,
                     runtime_contract=_UNSET_RUNTIME_CONTRACT, tool_schemas=None,
                     actor_protocol=None):
        if runtime_contract is _UNSET_RUNTIME_CONTRACT:
            from shopping_grpo.evaluation.runtime_contract import load_and_validate_runtime_contract
            runtime_contract = load_and_validate_runtime_contract(
                Path(__file__).resolve().parent / "fixtures" / "runtime_contract.json"
            )
        if actor_protocol is None and hasattr(runtime_contract, "contract"):
            contract = runtime_contract.contract
            actor_protocol = {
                "actor_max_tokens": contract["max_generated_tokens_per_turn"],
                "actor_context_window": contract["context_window"],
                "actor_context_safety_margin": contract["context_safety_margin"],
                "actor_observation_token_budget": contract["observation_search_tokens"],
                "actor_observation_detail_token_budget": contract["observation_detail_tokens"],
                "actor_observation_generic_token_budget": contract["observation_generic_tokens"],
                "actor_observation_search_top_k": contract["observation_search_top_k"],
            }
        return EvaluationDriver(
            run_dir=run_dir,
            task_split_path=task_split_path or self.split_path,
            split=split,
            actor_label="m2-check",
            actor_client=actor or self.actor,
            judge_client=judge or self.judge,
            curator_client=curator or self.curator,
            env_factory=env_factory or self.env_factory,
            facts_builder=facts_builder,
            base_url="http://127.0.0.1:5700",
            temperature=0.0,
            top_p=1.0,
            max_steps=max_steps,
            model_path="/models/m2-merged",
            judge_model="fake-judge",
            curator_model="fake-curator",
            runtime_contract=runtime_contract,
            tool_schemas=tool_schemas,
            actor_protocol=actor_protocol,
            resume=resume,
            allow_blind_final=allow_blind_final,
        )


class EvaluationDriverTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self._tmp = tmp
        self.tmp = Path(tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _harness(self, **kwargs):
        harness = Harness(self.tmp, **kwargs)
        harness.tmp = self.tmp
        harness.split_path = harness.write_split()
        return harness

    @staticmethod
    def _row_count(path):
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    # ------------------------------------------------------------------
    # 完整固定分母 run
    # ------------------------------------------------------------------

    def test_full_run_fixed_denominator_and_artifacts(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        result = harness.build_driver(run_dir).run()
        summary = result["summary"]

        self.assertEqual(summary["schema_version"], "shopping-evaluation-summary-v1")
        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 3)
        self.assertEqual(summary["missing_task_ids"], [])
        self.assertEqual(summary["reward_and_terminal"]["strict_gold_successes"], 3)
        self.assertAlmostEqual(summary["reward_and_terminal"]["gold_purchase_rate"], 1.0)
        self.assertEqual(summary["trajectory_quality"]["judge_status_counts"], {"valid": 3})
        self.assertEqual(summary["not_judged_tasks"], 0)
        self.assertEqual(summary["deterministic"]["infrastructure_invalid_tasks"], 0)
        self.assertEqual(summary["run"]["driver_version"], DRIVER_VERSION)

        for name in (
            "run_manifest.json",
            "trajectories.jsonl",
            "normalized.jsonl",
            "metrics.jsonl",
            "rubrics.jsonl",
            "judge.jsonl",
            "evaluations.jsonl",
            "summary.json",
            "errors.jsonl",
        ):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertEqual(self._row_count(run_dir / "evaluations.jsonl"), 3)
        evaluation = json.loads(
            (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(evaluation["schema_version"], EVALUATION_RESULT_VERSION)
        self.assertIn("task_id", evaluation)
        self.assertIn("trajectory_id", evaluation)

        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "shopping-evaluation-run-manifest-v1")
        self.assertEqual(
            manifest["task_manifest"]["sha256"], sha256_file(harness.split_path)
        )
        self.assertEqual(manifest["actor"]["label"], "m2-check")
        self.assertTrue(manifest["protocol"]["protocol_hash"])

    def test_run_refuses_to_overwrite_existing_outputs(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()
        with self.assertRaises(DriverError):
            harness.build_driver(run_dir, resume=False).run()

    def test_duplicate_task_ids_in_split_rejected(self):
        harness = Harness(self.tmp, task_ids=(101, 101))
        harness.tmp = self.tmp
        path = self.tmp / "dup.jsonl"
        rows = [_task_row(101), _task_row(101)]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        harness.split_path = path
        with self.assertRaises(DriverError):
            harness.build_driver(self.tmp / "run").run()

    def test_unsupported_split_rejected(self):
        harness = self._harness()
        with self.assertRaises(DriverError):
            harness.build_driver(self.tmp / "run", split="final-183").run()

    # ------------------------------------------------------------------
    # 缺题仍在分母 + 中断恢复
    # ------------------------------------------------------------------

    def test_missing_task_stays_in_fixed_denominator(self):
        harness = self._harness()
        harness.curator = FakeCuratorClient(fail_for_task_ids={102})
        run_dir = self.tmp / "run"
        result = harness.build_driver(run_dir).run()
        summary = result["summary"]

        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["missing_task_ids"], [102])
        # 分母仍是 3：2 个 strict gold / 3。
        self.assertAlmostEqual(summary["reward_and_terminal"]["gold_purchase_rate"], 2 / 3)
        errors = (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(errors), 1)
        self.assertEqual(json.loads(errors[0])["stage"], "rubric")

    def test_resume_completes_interrupted_run_from_caches(self):
        harness = self._harness()
        harness.curator = FakeCuratorClient(fail_for_task_ids={102})
        run_dir = self.tmp / "run"
        first = harness.build_driver(run_dir).run()
        self.assertEqual(first["summary"]["missing_task_ids"], [102])

        harness.curator = FakeCuratorClient()
        second = harness.build_driver(run_dir, resume=True).run()
        summary = second["summary"]
        self.assertEqual(second["cached_tasks"], 2)
        self.assertEqual(summary["completed_tasks"], 3)
        self.assertEqual(summary["missing_task_ids"], [])
        self.assertAlmostEqual(summary["reward_and_terminal"]["gold_purchase_rate"], 1.0)
        for name in ("trajectories", "normalized", "metrics", "rubrics", "judge", "evaluations"):
            self.assertEqual(
                self._row_count(run_dir / f"{name}.jsonl"),
                3,
                name,
            )

    def test_resume_reuses_trajectory_after_evaluation_truncation(self):
        harness = self._harness(behavior={103: "abstain"})
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()

        # 模拟崩溃：丢掉最后一条 evaluation，但保留 trajectory/judge 等缓存。
        evaluations_path = run_dir / "evaluations.jsonl"
        lines = evaluations_path.read_text(encoding="utf-8").splitlines()
        evaluations_path.write_text(
            "\n".join(lines[:-1]) + "\n", encoding="utf-8"
        )

        harness.actor = PoisonActor()
        harness.curator = PoisonCuratorClient()
        harness.env_factory = PoisonEnvFactory()
        result = harness.build_driver(run_dir, resume=True).run()
        self.assertEqual(result["cached_tasks"], 2)
        self.assertEqual(result["summary"]["completed_tasks"], 3)
        self.assertEqual(self._row_count(evaluations_path), 3)
        self.assertEqual(
            self._row_count(run_dir / "normalized.jsonl"), 3, "resume 不应重复追加"
        )

    def test_resume_cache_hit_makes_no_new_calls(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        first = harness.build_driver(run_dir).run()
        counts_before = {
            name: self._row_count(run_dir / f"{name}.jsonl")
            for name in (
                "trajectories",
                "normalized",
                "metrics",
                "rubrics",
                "judge",
                "evaluations",
            )
        }

        harness.actor = PoisonActor()
        harness.curator = PoisonCuratorClient()
        harness.judge = PoisonJudgeClient()
        harness.env_factory = PoisonEnvFactory()
        second = harness.build_driver(run_dir, resume=True).run()

        self.assertEqual(second["cached_tasks"], 3)
        self.assertEqual(
            second["summary"]["reward_and_terminal"]["strict_gold_successes"],
            first["summary"]["reward_and_terminal"]["strict_gold_successes"],
        )
        for name, count in counts_before.items():
            self.assertEqual(self._row_count(run_dir / f"{name}.jsonl"), count, name)

    def test_resume_rejects_task_split_hash_change(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()

        modified = self.tmp / "modified.jsonl"
        rows = [_task_row(task_id) for task_id in harness.task_ids]
        rows.append(_task_row(104))
        modified.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaises(ResumeContractError):
            harness.build_driver(
                run_dir, resume=True, task_split_path=modified
            ).run()

    def test_resume_rejects_protocol_change(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()
        with self.assertRaises((ResumeContractError, DriverError)):
            harness.build_driver(run_dir, resume=True, max_steps=30).run()

    def test_shared_rubric_cache_is_reused_across_models(self):
        harness = self._harness()
        shared = self.tmp / "shared-rubrics.jsonl"
        first_dir = self.tmp / "run-a"
        harness.build_driver(first_dir, allow_blind_final=False).run()
        # 共享 cache 由 run-a 的 rubrics.jsonl 复制而来（跨模型复用的形态）。
        shared.write_text(
            (first_dir / "rubrics.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        second_dir = self.tmp / "run-b"
        harness.actor = PoisonActor()
        harness.curator = PoisonCuratorClient()
        harness.env_factory = PoisonEnvFactory()
        # run-b 只能复用共享 rubric；trajectory/judge/evaluation 必须重新生成。
        harness.judge = FakeJudgeClient()
        fresh_env = FakeEnvFactory(harness.instructions)
        harness.env_factory = fresh_env
        fresh_actor = FakeActor()
        fresh_actor.scripts.update(harness.actor.scripts)
        harness.actor = fresh_actor
        harness.curator = PoisonCuratorClient()

        driver = harness.build_driver(second_dir)
        driver.shared_rubric_cache = shared
        result = driver.run()
        self.assertEqual(result["summary"]["completed_tasks"], 3)
        self.assertEqual(harness.curator.calls, [])
        self.assertEqual(self._row_count(second_dir / "rubrics.jsonl"), 0)

    def test_resume_rejects_unknown_and_duplicate_cache_rows(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()
        evaluations_path = run_dir / "evaluations.jsonl"

        unknown_dir = self.tmp / "run-unknown"
        unknown_dir.mkdir()
        for name in run_dir.iterdir():
            unknown_dir.joinpath(name.name).write_text(
                name.read_text(encoding="utf-8"), encoding="utf-8"
            )
        with unknown_dir.joinpath("evaluations.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "schema_version": EVALUATION_RESULT_VERSION,
                        "task_id": 999,
                        "trajectory_id": "ghost",
                    }
                )
                + "\n"
            )
        with self.assertRaises(ResumeContractError):
            harness.build_driver(unknown_dir, resume=True).run()

        with evaluations_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "schema_version": EVALUATION_RESULT_VERSION,
                        "task_id": 101,
                        "trajectory_id": "duplicate",
                    }
                )
                + "\n"
            )
        with self.assertRaises(ResumeContractError):
            harness.build_driver(run_dir, resume=True).run()

    # ------------------------------------------------------------------
    # 基础设施失败 → not_judged，但留在分母
    # ------------------------------------------------------------------

    def test_release_failure_becomes_not_judged_but_stays_in_denominator(self):
        harness = self._harness(fail_release_for={102})
        run_dir = self.tmp / "run"
        result = harness.build_driver(run_dir).run()
        summary = result["summary"]

        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 3)
        self.assertEqual(summary["deterministic"]["infrastructure_invalid_task_ids"], [102])
        self.assertEqual(summary["not_judged_tasks"], 1)
        # infra-invalid 任务不计入 strict gold，但分母仍是 3。
        self.assertAlmostEqual(summary["reward_and_terminal"]["gold_purchase_rate"], 2 / 3)
        # Judge 不应被调用于 infra-invalid 轨迹。
        self.assertEqual(len(harness.judge.calls), 2)
        evaluations = [
            json.loads(line)
            for line in (run_dir / "evaluations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        by_task = {row["task_id"]: row for row in evaluations}
        self.assertEqual(
            by_task[102]["trajectory_quality"]["judge_status"], "not_judged"
        )
        self.assertEqual(
            by_task[101]["trajectory_quality"]["judge_status"], "valid"
        )

    def test_env_factory_failure_fails_fast(self):
        harness = self._harness()

        def broken_env_factory(base_url):
            raise ShopEnvironmentError(
                "Unable to get available environment resource"
            )

        with self.assertRaises(DriverError):
            harness.build_driver(
                self.tmp / "run", env_factory=broken_env_factory
            ).run()

    def test_judge_invalid_schema_becomes_not_judged(self):
        harness = self._harness()
        harness.judge = FakeJudgeClient(invalid_for_task_ids={102})
        run_dir = self.tmp / "run"
        result = harness.build_driver(run_dir).run()
        summary = result["summary"]

        self.assertEqual(summary["not_judged_tasks"], 1)
        self.assertEqual(
            summary["trajectory_quality"]["judge_status_counts"],
            {"not_judged": 1, "valid": 2},
        )
        # Reward 面板不受 Judge 失败影响，分母仍为 3。
        self.assertEqual(summary["reward_and_terminal"]["strict_gold_successes"], 3)
        errors = (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
        stages = {json.loads(line)["stage"] for line in errors}
        self.assertIn("judge", stages)

    # ------------------------------------------------------------------
    # Final blind guard
    # ------------------------------------------------------------------

    def test_final_mode_requires_explicit_flag(self):
        harness = self._harness()
        with self.assertRaises(BlindFinalGuardError):
            harness.build_driver(
                self.tmp / "run", split="final-200", allow_blind_final=False
            ).run()
        self.assertEqual(harness.actor.calls, [])

    def test_final_mode_rejects_mismatched_task_ids(self):
        harness = self._harness()
        with self.assertRaises(BlindFinalGuardError):
            harness.build_driver(
                self.tmp / "run", split="final-200", allow_blind_final=True
            ).run()
        self.assertEqual(harness.actor.calls, [])

    def test_dev_split_rejects_blind_final_task_ids(self):
        _, frozen_ids = validate_canonical_blind_asset()
        frozen = sorted(frozen_ids)
        harness = self._harness(task_ids=(frozen[0], 101, 102))
        with self.assertRaises(BlindFinalGuardError):
            harness.build_driver(self.tmp / "run", split="dev").run()

    def test_final_mode_rejects_same_ids_with_non_frozen_asset_bytes(self):
        _, frozen_ids = validate_canonical_blind_asset()
        frozen = sorted(frozen_ids)
        path = self.tmp / "final.jsonl"
        path.write_text(
            "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in frozen),
            encoding="utf-8",
        )
        instructions = {task_id: f"{QUERY}#{task_id}" for task_id in frozen}
        actor = FakeActor()
        for instruction in instructions.values():
            actor.scripts[instruction] = _script("gold")
        harness = Harness.__new__(Harness)
        harness.task_ids = frozen
        harness.instructions = instructions
        harness.actor = actor
        harness.curator = FakeCuratorClient()
        harness.judge = FakeJudgeClient()
        harness.env_factory = FakeEnvFactory(instructions)
        harness.tmp = self.tmp
        harness.split_path = path

        with self.assertRaises(BlindFinalGuardError):
            harness.build_driver(
                self.tmp / "run",
                split="final-200",
                allow_blind_final=True,
                facts_builder=_synthetic_facts,
            ).run()
        self.assertEqual(actor.calls, [])

    def test_blind_allow_mode_scans_hidden_fields(self):
        _, frozen_ids = validate_canonical_blind_asset()
        task_id = min(frozen_ids)
        path = self.tmp / "hidden.jsonl"
        path.write_text(
            json.dumps(
                {"task_id": task_id, "extra_info": {"task_id": task_id, "query": "secret"}}
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ArtifactError):
            validate_blind_asset_file(
                path,
                declared_task_sha256=sha256_file(path),
                expected_task_ids={task_id},
            )

    def test_resume_rejects_tampered_evaluation_actor_identity(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()
        path = run_dir / "evaluations.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["actor"]["label"] = "forged-model"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaises(ResumeContractError):
            harness.build_driver(run_dir, resume=True).run()

    def test_resume_rejects_task_facts_source_drift_in_rubric_cache(self):
        harness = self._harness()
        state = {"query": QUERY}

        def facts_builder(task):
            return build_task_facts(
                task_id=int(task["task_id"]),
                query=state["query"],
                target_product={"asin": "B0SYNTH000", "category": "电子配件"},
                instruction_record={
                    "instruction": state["query"],
                    "attributes": [],
                    "instruction_options": [],
                },
                reward_goal={"instruction_text": state["query"]},
            )

        run_dir = self.tmp / "run"
        harness.build_driver(run_dir, facts_builder=facts_builder).run()
        state["query"] = "事实来源已漂移"
        with self.assertRaises(ResumeContractError):
            harness.build_driver(
                run_dir, resume=True, facts_builder=facts_builder
            ).run()

    def test_resume_rejects_served_identity_manifest_drift(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["actor"]["served_model_identity"] = {
            "proven": True,
            "served_model_name": "forged-endpoint",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            harness.build_driver(run_dir, resume=True).run()

    # ------------------------------------------------------------------
    # 盲化与 facts builder
    # ------------------------------------------------------------------

    def test_model_label_and_path_never_enter_judge_input(self):
        harness = self._harness()
        run_dir = self.tmp / "run"
        harness.build_driver(run_dir).run()

        self.assertTrue(harness.judge.calls)
        for payload in harness.judge.calls:
            rendered = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("m2-check", rendered)
            self.assertNotIn("/models/m2-merged", rendered)
        for payload in harness.curator.calls:
            rendered = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("m2-check", rendered)
            self.assertNotIn("/models/m2-merged", rendered)
        # 外层 evaluation 保留 actor 元数据，用于四模型对比。
        evaluations = (
            (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        )
        first = json.loads(evaluations[0])
        self.assertEqual(first["actor"]["label"], "m2-check")

    def test_default_facts_builder_requires_embedded_goal_and_product(self):
        with self.assertRaises(DriverError):
            default_facts_builder({"task_id": 1})
        facts = default_facts_builder(_task_row(101))
        self.assertEqual(facts["task_id"], 101)
        self.assertTrue(facts["query"].startswith("请推荐一台海信"))


class EvaluateStudentCliTest(unittest.TestCase):
    def test_parse_args_matches_gaps_dev_example(self):
        args = parse_args(
            [
                "--model-label", "m2",
                "--model-path", "outputs/models/process-sft-merged",
                "--split", "dev",
                "--task-ids", "generated-dev-task-ids.jsonl",
                "--output", "outputs/evaluation/dev/m2",
                "--environment-version", "shopsimulator-environment-v2.1",
                "--temperature", "0",
                "--top-p", "1",
                "--max-steps", "35",
                "--resume",
            ]
        )
        self.assertEqual(args.model_label, "m2")
        self.assertEqual(args.split, "dev")
        self.assertEqual(args.environment_version, "shopsimulator-environment-v2.1")
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.max_steps, 35)  # 显式传冻结值合法
        self.assertIsNone(args.actor_max_tokens)  # 合同填充
        self.assertTrue(args.resume)
        self.assertFalse(args.allow_blind_final_after_freeze)

    def test_final_requires_explicit_blind_flag(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--model-label", "m2",
                        "--model-path", "outputs/models/process-sft-merged",
                        "--split", "final-200",
                        "--task-ids", "data/splits/final_task_ids.jsonl",
                        "--output", "outputs/evaluation/final-200/m2",
                    ]
                )
        self.assertIn("allow-blind-final-after-freeze", stderr.getvalue())

    def test_final_with_flag_passes_parse(self):
        args = parse_args(
            [
                "--model-label", "m2",
                "--model-path", "outputs/models/process-sft-merged",
                "--split", "final-200",
                "--task-ids", "data/splits/final_task_ids.jsonl",
                "--output", "outputs/evaluation/final-200/m2",
                "--allow-blind-final-after-freeze",
            ]
        )
        self.assertTrue(args.allow_blind_final_after_freeze)

    def test_build_driver_wires_injected_fakes(self):
        harness_dir = Path(tempfile.mkdtemp())
        try:
            split = harness_dir / "tasks.jsonl"
            split.write_text(
                json.dumps({"task_id": 1}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            import shutil

            from tests.test_protocol_hash_binding import FIXTURE as CONTRACT_FIXTURE

            contract_path = harness_dir / "runtime_contract.json"
            shutil.copyfile(CONTRACT_FIXTURE, contract_path)
            args = parse_args(
                [
                    "--model-label", "m2",
                    "--model-path", "outputs/models/process-sft-merged",
                    "--split", "dev",
                    "--task-ids", str(split),
                    "--output", str(harness_dir / "run"),
                    "--env-base-url", "http://127.0.0.1:5700",
                    "--runtime-contract", str(contract_path),
                ]
            )
            driver = build_driver(
                args,
                actor_client=FakeActor(),
                judge_client=FakeJudgeClient(),
                curator_client=FakeCuratorClient(),
                env_factory=FakeEnvFactory({1: QUERY}),
            )
            self.assertIsInstance(driver, EvaluationDriver)
            self.assertEqual(driver.actor_label, "m2")
            self.assertEqual(driver.split, "dev")
            self.assertEqual(driver.max_steps, 35)
            self.assertEqual(driver.base_url, "http://127.0.0.1:5700")
            self.assertFalse(driver.allow_blind_final)
            self.assertEqual(driver.actor_metadata["model_path"],
                             "outputs/models/process-sft-merged")
        finally:
            import shutil

            shutil.rmtree(harness_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
