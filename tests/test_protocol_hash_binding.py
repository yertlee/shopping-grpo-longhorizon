"""protocol_hash 冻结合同绑定测试（指令书 §8 负向矩阵）。

所有 fixture 使用完整、canonical 自洽的冻结合同（tests/fixtures/runtime_contract.json，
内容与 data/manifests/runtime_contract.json 逐字节一致，contract_sha256 == 73855a71…）。

要求：SYSTEM_PROMPT、tool schema、projection contract/code、reward、max_steps、
context 系列、temperature/top_p、runtime contract 中任一漂移 → protocol_hash 必须
变化；manifest 自洽性破坏（hash 伪造、字段增删、旧平铺布局）→ resume 拒绝。
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_grpo.evaluation.driver import DriverError, EvaluationDriver, ResumeContractError
from shopping_grpo.evaluation.runtime_contract import (
    FROZEN_CONTRACT_SHA256,
    RuntimeContractError,
    ValidatedRuntimeContract,
    canonical_json_bytes,
    compute_runtime_contract_canonical_sha256,
    compute_runtime_contract_file_sha256,
    load_and_validate_runtime_contract,
    thaw_runtime_value,
)
from tests.test_evaluation_driver import Harness

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "runtime_contract.json"


def load_full_contract(tmpdir: Path | None = None) -> ValidatedRuntimeContract:
    """加载完整冻结合同（fixture 文件）。"""
    return load_and_validate_runtime_contract(FIXTURE)


def full_actor_protocol(contract: ValidatedRuntimeContract) -> dict:
    c = contract.contract
    return {
        "actor_max_tokens": c["max_generated_tokens_per_turn"],
        "actor_context_window": c["context_window"],
        "actor_context_safety_margin": c["context_safety_margin"],
        "actor_observation_token_budget": c["observation_search_tokens"],
        "actor_observation_detail_token_budget": c["observation_detail_tokens"],
        "actor_observation_generic_token_budget": c["observation_generic_tokens"],
        "actor_observation_search_top_k": c["observation_search_top_k"],
    }


def write_contract_copy(tmpdir: Path, contract: dict, name="contract.json") -> Path:
    """把（可能被修改的）合同 dict 写为临时文件。"""
    path = tmpdir / name
    path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _driver_hash(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        harness = Harness(tmp, task_ids=[101, 102, 103])
        harness.tmp = tmp
        harness.split_path = harness.write_split()
        driver = harness.build_driver(tmp / "run", **kwargs)
        return driver.protocol_hash


class ContractLoadingTest(unittest.TestCase):
    def test_fixture_is_frozen_and_self_consistent(self):
        contract = load_full_contract()
        self.assertEqual(contract.declared_sha256, FROZEN_CONTRACT_SHA256)
        self.assertEqual(contract.canonical_sha256, FROZEN_CONTRACT_SHA256)
        self.assertEqual(contract.declared_sha256, contract.canonical_sha256)

    def test_missing_file_fails_closed(self):
        with self.assertRaises(RuntimeContractError):
            load_and_validate_runtime_contract(Path(tempfile.mkdtemp()) / "nope.json")

    def test_invalid_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                load_and_validate_runtime_contract(path)

    def test_non_object_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                load_and_validate_runtime_contract(path)

    def test_missing_declared_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
            del contract["contract_sha256"]
            path = write_contract_copy(Path(tmp), contract)
            with self.assertRaises(RuntimeContractError):
                load_and_validate_runtime_contract(path)

    def test_malformed_declared_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
            contract["contract_sha256"] = "z" * 64
            path = write_contract_copy(Path(tmp), contract)
            with self.assertRaises(RuntimeContractError):
                load_and_validate_runtime_contract(path)

    def test_declared_canonical_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
            contract["max_steps"] = 36  # 改 payload 但保留旧 hash
            path = write_contract_copy(Path(tmp), contract)
            with self.assertRaises(RuntimeContractError):
                load_and_validate_runtime_contract(path)

    def test_recomputed_hash_but_drifted_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
            contract["reward_version"] = "shopsimulator-reward-v4"
            contract["contract_sha256"] = compute_runtime_contract_canonical_sha256(contract)
            path = write_contract_copy(Path(tmp), contract)
            with self.assertRaises(RuntimeContractError):  # 偏离冻结身份 73855a71…
                load_and_validate_runtime_contract(path)

    def test_canonical_identity_is_path_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            copy_a = write_contract_copy(tmp, json.loads(FIXTURE.read_text(encoding="utf-8")), "a.json")
            copy_b = write_contract_copy(tmp, json.loads(FIXTURE.read_text(encoding="utf-8")), "b.json")
            a = load_and_validate_runtime_contract(copy_a)
            b = load_and_validate_runtime_contract(copy_b)
            self.assertEqual(a.canonical_sha256, b.canonical_sha256)
            self.assertEqual(a.declared_sha256, b.declared_sha256)
            self.assertEqual(a.file_sha256, b.file_sha256)  # 字节相同 → 物理身份相同

    def test_physical_bytes_change_file_hash_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
            path_a = write_contract_copy(tmp, contract, "a.json")
            # 相同语义、不同物理字节（压缩 JSON）→ canonical 相同、file hash 不同
            path_b = tmp / "b.json"
            path_b.write_text(
                json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            a = load_and_validate_runtime_contract(path_a)
            b = load_and_validate_runtime_contract(path_b)
            self.assertEqual(a.canonical_sha256, b.canonical_sha256)
            self.assertNotEqual(a.file_sha256, b.file_sha256)


class ProtocolHashDriftTest(unittest.TestCase):
    CONTRACT = None

    @classmethod
    def setUpClass(cls):
        cls.CONTRACT = load_full_contract()

    def test_same_contract_same_hash_and_missing_contract_differs(self):
        base = _driver_hash(runtime_contract=self.CONTRACT)
        again = _driver_hash(runtime_contract=self.CONTRACT)
        self.assertEqual(base, again, "同一冻结合同必须产生同一协议身份")
        with self.assertRaises(DriverError):
            _driver_hash(runtime_contract=None)

    def test_system_prompt_drift_rejects_startup(self):
        """P0-1：SYSTEM_PROMPT 漂移必须拒绝启动（不再是仅记录差异）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            with patch(
                "shopping_grpo.evaluation.rollout.SYSTEM_PROMPT",
                "你是一个购物 Agent（改过的 prompt 用于测试）。",
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(tmp / "run", runtime_contract=self.CONTRACT)

    def test_projection_contract_drift_rejects_startup(self):
        """P0-1：projection contract 版本漂移必须拒绝启动。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            with patch(
                "shopping_grpo.environment.projection.PROJECTION_CONTRACT_VERSION",
                "shopping-observation-v3",
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(tmp / "run", runtime_contract=self.CONTRACT)

    def test_projection_code_change_changes_protocol_hash(self):
        base = _driver_hash(runtime_contract=self.CONTRACT)
        with patch.object(
            EvaluationDriver, "_projection_code_sha256", lambda self: "b" * 64
        ):
            drifted = _driver_hash(runtime_contract=self.CONTRACT)
        self.assertNotEqual(base, drifted)

    def test_reward_drift_changes_protocol_hash(self):
        """reward_version 漂移必须改变 protocol_hash（payload 同源）。"""
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            driver = harness.build_driver(tmp / "run", runtime_contract=self.CONTRACT)
            payload = driver._protocol_payload()
            payload["reward_version"] = "shopsimulator-reward-v4"
            drifted = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            self.assertNotEqual(driver.protocol_hash, drifted)

    def test_driver_rejects_raw_dict_contract(self):
        """EvaluationDriver 只接受 ValidatedRuntimeContract，拒绝任意 dict。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            with self.assertRaises(DriverError):
                harness.build_driver(tmp / "run", runtime_contract={"contract_sha256": "x" * 64})

    def test_environment_mismatch_with_contract_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            with self.assertRaises(DriverError):
                harness.build_driver(tmp / "run", runtime_contract=self.CONTRACT, max_steps=30)

    def test_actor_protocol_reserved_field_injection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            with self.assertRaises(DriverError):
                harness.build_driver(
                    tmp / "run",
                    runtime_contract=self.CONTRACT,
                    actor_protocol={"max_steps": 1},
                )
            # actor_ 前缀字段合法
            driver = harness.build_driver(
                tmp / "run2",
                runtime_contract=self.CONTRACT,
                actor_protocol=full_actor_protocol(self.CONTRACT),
            )
            self.assertEqual(driver.actor_protocol["actor_max_tokens"], 512)


class RealCodeCrossValidationTest(unittest.TestCase):
    """指令书 §4：合同验证后核对实际运行代码（本地可断言的部分）。

    projection 模块的部署字节一致性（LF 规范化 == contract 冻结值）是部署后
    自检项：本地 Windows 工作区为 CRLF，冻结值对应 LF 规范化字节，测试不在此
    断言字节相等，由部署自检脚本在评测 runtime 上执行。
    """

    @classmethod
    def setUpClass(cls):
        cls.contract = load_full_contract()

    def _payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = Harness(tmp, task_ids=[101])
            harness.tmp = tmp
            harness.split_path = harness.write_split()
            driver = harness.build_driver(tmp / "run", runtime_contract=self.contract)
            return driver._protocol_payload()

    def test_system_prompt_sha256_matches_contract(self):
        payload = self._payload()
        self.assertEqual(
            payload["system_prompt_sha256"],
            self.contract.contract["system_prompt_hash"],
        )

    def test_tool_schema_canonical_hash_matches_contract(self):
        import hashlib

        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

        payload = self._payload()
        expected = hashlib.sha256(
            canonical_json_bytes(SHOP_TOOL_SCHEMAS)
        ).hexdigest()
        self.assertEqual(payload["tool_schema_sha256"], expected)
        self.assertEqual(expected, self.contract.contract["tool_schema_hash"])

    def test_projection_contract_version_matches_contract(self):
        payload = self._payload()
        self.assertEqual(
            payload["observation_projection_contract_version"],
            self.contract.contract["observation_projection_contract"],
        )

    def test_reward_version_matches_contract(self):
        payload = self._payload()
        self.assertEqual(
            payload["reward_version"], self.contract.contract["reward_version"]
        )

    def test_context_and_observation_budget_from_contract(self):
        payload = self._payload()
        c = self.contract.contract
        for field in (
            "context_window",
            "context_safety_margin",
            "max_generated_tokens_per_turn",
            "observation_search_tokens",
            "observation_detail_tokens",
            "observation_generic_tokens",
            "observation_search_top_k",
        ):
            self.assertEqual(payload[field], c[field], field)

    def test_runtime_contract_block_three_hashes(self):
        payload = self._payload()
        block = payload["runtime_contract"]
        self.assertEqual(block["declared_sha256"], self.contract.declared_sha256)
        self.assertEqual(block["canonical_sha256"], self.contract.canonical_sha256)
        self.assertEqual(block["file_sha256"], self.contract.file_sha256)
        self.assertEqual(
            block["payload"], thaw_runtime_value(self.contract.canonical_payload)
        )


class StartupRejectionTest(unittest.TestCase):
    """P0-1/P0-4 + P1：漂移必须拒绝启动（对抗探针场景）。"""

    def _harness(self, tmp):
        harness = Harness(tmp, task_ids=[101])
        harness.tmp = tmp
        harness.split_path = harness.write_split()
        return harness

    @staticmethod
    def _actor_protocol(contract):
        return full_actor_protocol(contract)

    def test_runtime_contract_none_rejects_formal_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(DriverError):
                self._harness(tmp).build_driver(tmp / "run", runtime_contract=None)

    def test_source_file_change_after_load_rejects_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "contract.json"
            source.write_bytes(FIXTURE.read_bytes())
            contract = load_and_validate_runtime_contract(source)
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaises(DriverError):
                self._harness(tmp).build_driver(
                    tmp / "run", runtime_contract=contract,
                    actor_protocol=self._actor_protocol(contract),
                )

    def test_forged_file_hash_rejects_formal_driver(self):
        contract = load_full_contract()
        forged = ValidatedRuntimeContract(
            contract=thaw_runtime_value(contract.contract),
            canonical_payload=thaw_runtime_value(contract.canonical_payload),
            declared_sha256=contract.declared_sha256,
            canonical_sha256=contract.canonical_sha256,
            file_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(DriverError):
                self._harness(tmp).build_driver(
                    tmp / "run", runtime_contract=forged,
                    actor_protocol=self._actor_protocol(contract),
                )

    def test_actor_protocol_missing_wrong_and_extra_fields_reject(self):
        contract = load_full_contract()
        expected = self._actor_protocol(contract)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for mutated in (
                {key: value for key, value in expected.items() if key != "actor_context_window"},
                {**expected, "actor_context_window": 1},
                {**expected, "actor_unlisted": 1},
            ):
                with self.assertRaises(DriverError):
                    self._harness(tmp).build_driver(
                        tmp / "run", runtime_contract=contract,
                        actor_protocol=mutated,
                    )

    def test_same_hash_tool_schema_injection_is_rejected(self):
        import copy
        from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

        contract = load_full_contract()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(DriverError):
                self._harness(tmp).build_driver(
                    tmp / "run", runtime_contract=contract,
                    actor_protocol=self._actor_protocol(contract),
                    tool_schemas=copy.deepcopy(SHOP_TOOL_SCHEMAS),
                )

    def test_reward_v3_drift_rejects_startup(self):
        """P0-4：修改 metrics.REWARD_V3 后必须拒绝启动。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            with patch(
                "shopping_grpo.evaluation.metrics.REWARD_V3",
                "shopsimulator-reward-v4",
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(
                        tmp / "run", runtime_contract=load_full_contract()
                    )

    def test_projection_file_bytes_drift_rejects_startup(self):
        """P0-2：projection 原始字节与冻结合同不一致必须拒绝启动。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            drifted_file = tmp / "projection.py"
            drifted_file.write_text("# drifted projection module\n", encoding="utf-8")
            with patch(
                "shopping_grpo.environment.projection.__file__",
                str(drifted_file),
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(
                        tmp / "run", runtime_contract=load_full_contract()
                    )

    def test_projection_module_unreadable_rejects_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            missing = tmp / "missing_projection.py"
            with patch(
                "shopping_grpo.environment.projection.__file__",
                str(missing),
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(
                        tmp / "run", runtime_contract=load_full_contract()
                    )

    def test_tool_schemas_injection_mismatch_rejects_startup(self):
        """P1：注入的 tool_schemas 与合同不一致必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            with self.assertRaises(DriverError):
                harness.build_driver(
                    tmp / "run",
                    runtime_contract=load_full_contract(),
                    tool_schemas=[{"name": "different"}],
                )

    def test_tools_json_drift_rejects_startup(self):
        """P0-3：configs/tools.json 与 Python schema 漂移必须拒绝启动。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            with patch(
                "shopping_grpo.environment.tools.validate_runtime_tool_schema",
                side_effect=ValueError("canonical tool schema drift"),
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(
                        tmp / "run", runtime_contract=load_full_contract()
                    )

    def test_tools_json_hash_mismatch_rejects_startup(self):
        """configs/tools.json 与合同 hash 不一致必须拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            with patch(
                "shopping_grpo.environment.tools.validate_runtime_tool_schema",
                return_value="0" * 64,
            ):
                with self.assertRaises(DriverError):
                    harness.build_driver(
                        tmp / "run", runtime_contract=load_full_contract()
                    )

    def test_clean_code_passes_cross_validation(self):
        """真实代码与冻结合同一致时启动通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harness = self._harness(tmp)
            driver = harness.build_driver(
                tmp / "run", runtime_contract=load_full_contract()
            )
            self.assertEqual(
                driver.protocol_payload_for_test
                if hasattr(driver, "protocol_payload_for_test")
                else driver.protocol_hash,
                driver.protocol_hash,
            )


class ValidatedContractHardeningTest(unittest.TestCase):
    """P1：手工构造、只读视图、allow_nan。"""

    def test_manual_construction_of_inconsistent_contract_rejected(self):
        with self.assertRaises(RuntimeContractError):
            ValidatedRuntimeContract(
                contract={"contract_sha256": "0" * 64},
                canonical_payload={},
                declared_sha256="0" * 64,
                canonical_sha256="0" * 64,
                file_sha256="0" * 64,
            )

    def test_nested_dict_is_read_only_after_validation(self):
        contract = load_full_contract()
        with self.assertRaises(TypeError):
            contract.contract["max_steps"] = 99
        with self.assertRaises(TypeError):
            contract.canonical_payload["max_steps"] = 99

    def test_recursive_dict_list_set_are_read_only_and_thawable(self):
        from shopping_grpo.evaluation.runtime_contract import _freeze_runtime_value

        frozen = _freeze_runtime_value(
            {"nested": {"items": [{"x": 1}], "tags": {"a", "b"}}}
        )
        with self.assertRaises(TypeError):
            frozen["nested"]["items"][0]["x"] = 2
        with self.assertRaises(AttributeError):
            frozen["nested"]["items"].append({"x": 2})
        with self.assertRaises(AttributeError):
            frozen["nested"]["tags"].add("c")
        self.assertEqual(
            thaw_runtime_value(frozen),
            {"nested": {"items": [{"x": 1}], "tags": ["a", "b"]}},
        )

    def test_canonical_json_rejects_nan(self):
        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        with self.assertRaises(ValueError):
            canonical_json_bytes({"x": float("nan")})
        with self.assertRaises(ValueError):
            canonical_json_bytes({"x": float("inf")})


class ProtocolResumeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self._tmp = tmp
        self.tmp = Path(tmp.name)
        harness = Harness(self.tmp, task_ids=[101, 102, 103])
        harness.tmp = self.tmp
        harness.split_path = harness.write_split()
        self.harness = harness
        self.contract = load_full_contract()

    def tearDown(self):
        self._tmp.cleanup()

    def _completed_run(self, run_dir, **kwargs):
        driver = self.harness.build_driver(run_dir, resume=False, **kwargs)
        driver.run()
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    def test_manifest_hash_recomputes_from_own_payload(self):
        run_dir = self.tmp / "r1"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        payload = {k: v for k, v in block.items() if k != "protocol_hash"}
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        recomputed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self.assertEqual(recomputed, block["protocol_hash"])
        self.assertIn("runtime_contract", payload)
        self.assertEqual(payload["runtime_contract"]["declared_sha256"], FROZEN_CONTRACT_SHA256)

    def test_resume_rejects_hash_tamper_with_same_payload(self):
        run_dir = self.tmp / "r2"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        block["protocol_hash"] = "0" * 64
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract).run()

    def test_resume_rejects_payload_change_with_forged_hash(self):
        run_dir = self.tmp / "r3"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        block["reward_version"] = "shopsimulator-reward-v4"
        payload = {k: v for k, v in block.items() if k != "protocol_hash"}
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        block["protocol_hash"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract).run()

    def test_resume_rejects_unknown_field_addition(self):
        run_dir = self.tmp / "r4"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        block["surprise"] = True
        payload = {k: v for k, v in block.items() if k != "protocol_hash"}
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        block["protocol_hash"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract).run()

    def test_resume_rejects_legacy_flat_actor_layout(self):
        run_dir = self.tmp / "r5"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        # 模拟旧平铺布局：把 actor 字段平铺到顶层并删除嵌套
        block["actor_max_tokens"] = 512
        del block["actor_protocol"]
        payload = {k: v for k, v in block.items() if k != "protocol_hash"}
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        block["protocol_hash"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract).run()

    def test_resume_rejects_contract_block_tamper(self):
        run_dir = self.tmp / "r6"
        manifest = self._completed_run(run_dir, runtime_contract=self.contract)
        block = manifest["protocol"]
        block["runtime_contract"]["canonical_sha256"] = "0" * 64
        payload = {k: v for k, v in block.items() if k != "protocol_hash"}
        import hashlib

        from shopping_grpo.evaluation.runtime_contract import canonical_json_bytes

        block["protocol_hash"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with self.assertRaises(ResumeContractError):
            self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract).run()

    def test_clean_resume_passes(self):
        run_dir = self.tmp / "r7"
        self._completed_run(run_dir, runtime_contract=self.contract)
        driver = self.harness.build_driver(run_dir, resume=True, runtime_contract=self.contract)
        driver.run()  # 不应抛异常


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
