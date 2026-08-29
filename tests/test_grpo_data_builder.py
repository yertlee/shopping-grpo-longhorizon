"""build_grpo_data.py 的纯逻辑测试：schema、唯一性、泄漏断言、hash 稳定性与确定性。

不依赖 pandas / pyarrow / torch：parquet 写入层注入 JSONL fake writer，
token 计数器注入固定字符预算函数。所有冻结输入都用 3–5 个 fake task 重建。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import build_grpo_data as builder
from scripts import train_grpo

SYSTEM_PROMPT_SHA256 = hashlib.sha256(builder.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
TOOL_SCHEMA_SHA256 = "9eccfc80db97f7fb9b34eea2f2f65f0b8bcb9bae76d8c2ee031db99c435ee1ad"

# 5 个 fake task：10/20/30 进 grpo_train（20 不可达），40/50 进 grpo_validation。
FACT_ROWS = (
    {
        "schema_version": "commerce-task-facts-v1",
        "task_id": 10,
        "query": "我想找一款适合跑步的运动水壶，容量五百毫升左右，最好有提绳。",
        "difficulty": "easy",
        "target_title": "HIDDEN-TITLE-10-跑步水壶",
        "target_options": ["HIDDEN-OPTION-10-蓝色"],
        "target_product_ids": ["HIDDEN-ASIN-10"],
        "product_family": "family-10",
    },
    {
        "schema_version": "commerce-task-facts-v1",
        "task_id": 20,
        "query": "帮我挑一台适合小户型的高性价比洗烘一体机，预算三千元以内。",
        "difficulty": "hard",
        "target_title": "HIDDEN-TITLE-20-洗烘一体机",
        "target_options": ["HIDDEN-OPTION-20-白色"],
        "target_product_ids": ["HIDDEN-ASIN-20"],
        "product_family": "family-20",
    },
    {
        "schema_version": "commerce-task-facts-v1",
        "task_id": 30,
        "query": "想要一个能放在书桌上的机械键盘，青轴，带背光，价格三百元上下。",
        "difficulty": "medium",
        "target_title": "HIDDEN-TITLE-30-机械键盘",
        "target_options": ["HIDDEN-OPTION-30-青轴"],
        "target_product_ids": ["HIDDEN-ASIN-30"],
        "product_family": "family-30",
    },
    {
        "schema_version": "commerce-task-facts-v1",
        "task_id": 40,
        "query": "想给孩子买一双防滑的室内拖鞋，秋冬穿，尺码三十码左右。",
        "difficulty": "easy",
        "target_title": "HIDDEN-TITLE-40-儿童拖鞋",
        "target_options": ["HIDDEN-OPTION-40-粉色"],
        "target_product_ids": ["HIDDEN-ASIN-40"],
        "product_family": "family-40",
    },
    {
        "schema_version": "commerce-task-facts-v1",
        "task_id": 50,
        "query": "需要一台静音的空气循环扇，可以定时，遥控操作最好。",
        "difficulty": "medium",
        "target_title": "HIDDEN-TITLE-50-循环扇",
        "target_options": ["HIDDEN-OPTION-50-黑色"],
        "target_product_ids": ["HIDDEN-ASIN-50"],
        "product_family": "family-50",
    },
)


class JsonlWriter:
    """fake parquet writer：按行写 JSONL，键排序保证字节可比较。"""

    def __init__(self, path: Path):
        self.path = path
        self.rows = []

    def write(self, row: dict) -> None:
        self.rows.append(row)

    def close(self) -> None:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in self.rows
        )
        self.path.write_text(payload, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        return False


def jsonl_writer_factory(path: Path) -> JsonlWriter:
    return JsonlWriter(path)


def fake_token_counter(messages) -> int:
    """确定性计数：字符数四舍五入到 token；足够小以通过默认预算。"""
    return sum(len(message["content"]) // 4 + 1 for message in messages)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_fixture(
    directory: Path,
    *,
    fact_rows=FACT_ROWS,
    final_task_ids=(900, 901),
    teacher_pool_task_ids=(800,),
    grpo_train_task_ids=(10, 20, 30),
    grpo_validation_task_ids=(40, 50),
    reachable_train=(10, 30),
    sft_task_id_sets=(
        (100, 101),  # process/train
        (102, 103),  # process/dev
        (104, 105),  # outcome/train
        (106, 107),  # outcome/dev
    ),
) -> dict:
    """构建一套完整冻结输入；返回各文件路径、SFT task id 文件与 task facts hash。"""
    directory.mkdir(parents=True, exist_ok=True)
    facts_path = directory / "task_facts.jsonl"
    _write_jsonl(facts_path, fact_rows)
    task_facts_sha256 = hashlib.sha256(facts_path.read_bytes()).hexdigest()

    sft_dir = directory / "sft"
    sft_task_id_files = []
    for name, ids in zip(
        ("process_train", "process_dev", "outcome_train", "outcome_dev"), sft_task_id_sets
    ):
        path = sft_dir / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in ids),
            encoding="utf-8",
        )
        sft_task_id_files.append(path)

    def split_entry(task_ids):
        return {
            "count": len(task_ids),
            "task_ids": list(task_ids),
            "task_ids_sha256": builder.canonical_task_ids_sha256(task_ids),
        }

    contract = {
        "schema_version": "commerce-runtime-contract-v1",
        "contract_sha256": "b" * 64,
        "environment_version": "shopsimulator-environment-v2.1",
        "reward_version": "shopsimulator-reward-v3",
        "tool_version": "shopping-tools-v2",
        "tool_schema_hash": TOOL_SCHEMA_SHA256,
        "system_prompt_hash": SYSTEM_PROMPT_SHA256,
        "upstream_commit": "4ed73020e1d7d07eb93e7375a4606b0901d3cded",
    }
    contract_path = directory / "runtime_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    split_manifest = {
        "schema_version": "commerce-split-manifest-v1",
        "manifest_sha256": "a" * 64,
        "runtime_contract_sha256": contract["contract_sha256"],
        "task_source": {"label": "fake", "rows": len(fact_rows), "sha256": task_facts_sha256},
        "splits": {
            "grpo_train": split_entry(grpo_train_task_ids),
            "grpo_validation": split_entry(grpo_validation_task_ids),
            "final": split_entry(final_task_ids),
            "teacher_pool": split_entry(teacher_pool_task_ids),
        },
    }
    split_manifest_path = directory / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def reach_entry(eligible, requested):
        unreachable = [
            {
                "task_id": task_id,
                "status": "unreachable",
                "reason_codes": ["required_option_not_exposed"],
            }
            for task_id in requested
            if task_id not in eligible
        ]
        return {
            "input_count": len(requested),
            "eligible_count": len(eligible),
            "eligible_task_ids": list(eligible),
            "unreachable": unreachable,
            "unreachable_count": len(unreachable),
        }

    reachability = {
        "schema_version": "commerce-task-reachability-manifest-v1",
        "manifest_sha256": "c" * 64,
        "inputs": {
            "split_manifest_sha256": hashlib.sha256(split_manifest_path.read_bytes()).hexdigest(),
            "task_facts_sha256": task_facts_sha256,
        },
        "splits": {
            "grpo_train": reach_entry(reachable_train, grpo_train_task_ids),
            "grpo_validation": reach_entry(
                grpo_validation_task_ids, grpo_validation_task_ids
            ),
        },
    }
    reachability_path = directory / "task_reachability_manifest.json"
    reachability_path.write_text(
        json.dumps(reachability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "split_manifest": split_manifest_path,
        "task_facts": facts_path,
        "contract": contract_path,
        "reachability": reachability_path,
        "task_facts_sha256": task_facts_sha256,
        "sft_task_id_files": sft_task_id_files,
    }


def _build(directory: Path, fixture: dict, *, output_dir_name="out", **kwargs) -> dict:
    # 默认带上四个 SFT task id 文件：正式构建的泄漏检查是强制的。
    if "sft_task_id_files" not in kwargs:
        kwargs["sft_task_id_files"] = fixture["sft_task_id_files"]
    return builder.build_grpo_data(
        split_manifest_path=fixture["split_manifest"],
        task_facts_path=fixture["task_facts"],
        contract_path=fixture["contract"],
        reachability_path=fixture["reachability"],
        output_dir=directory / output_dir_name,
        token_counter=fake_token_counter,
        tokenizer_descriptor="fake-counter",
        max_prompt_tokens=kwargs.pop("max_prompt_tokens", 512),
        writer_factory=kwargs.pop("writer_factory", jsonl_writer_factory),
        **kwargs,
    )


class GrpoDataBuilderTest(unittest.TestCase):
    def test_schema_rows_and_metadata_hashes_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            metadata = _build(Path(tmp), fixture)

            train_rows = [
                json.loads(line)
                for line in (Path(tmp) / "out/train.parquet").read_text(encoding="utf-8").splitlines()
            ]
            validation_rows = [
                json.loads(line)
                for line in (Path(tmp) / "out/validation.parquet")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["task_id"] for row in train_rows], [10, 30])
            self.assertEqual([row["task_id"] for row in validation_rows], [40, 50])

            for row in train_rows + validation_rows:
                self.assertEqual(
                    [message["role"] for message in row["prompt"]], ["system", "user"]
                )
                self.assertEqual(row["prompt"][0]["content"], builder.SYSTEM_PROMPT)
                self.assertEqual(
                    set(row["extra_info"]), set(builder.EXTRA_INFO_KEYS)
                )
                self.assertEqual(row["extra_info"]["contract_sha256"], "b" * 64)
                self.assertEqual(row["data_source"], "shopsimulator-environment-v2.1")
                self.assertEqual(
                    row["reward_model"],
                    {"style": "rule", "reward_version": "shopsimulator-reward-v3"},
                )
            self.assertEqual(
                {row["extra_info"]["split"] for row in train_rows}, {"grpo_train"}
            )
            self.assertEqual(
                {row["extra_info"]["split"] for row in validation_rows},
                {"grpo_validation"},
            )

            files = metadata["files"]
            self.assertEqual(files["grpo_train"]["rows"], 2)
            self.assertEqual(files["grpo_train"]["unique_task_ids"], 2)
            self.assertEqual(files["grpo_validation"]["rows"], 2)
            train_hash = builder.sha256_file(Path(tmp) / "out/train.parquet")
            validation_hash = builder.sha256_file(Path(tmp) / "out/validation.parquet")
            self.assertEqual(files["grpo_train"]["file_sha256"], train_hash)
            self.assertEqual(files["grpo_validation"]["file_sha256"], validation_hash)
            # 请求的 task 集合 hash 必须等于 manifest 声明口径；包含集合因 20 不可达而缩小。
            self.assertEqual(
                files["grpo_train"]["requested_task_ids_sha256"],
                builder.canonical_task_ids_sha256([10, 20, 30]),
            )
            self.assertEqual(
                files["grpo_train"]["included_task_ids_sha256"],
                builder.canonical_task_ids_sha256([10, 30]),
            )
            self.assertTrue(metadata["leakage"]["all_checks_passed"])
            # 版本合同字段（audit item 7）：metadata 必须携带四项，launcher 交叉校验。
            self.assertEqual(metadata["source"]["tool_schema_sha256"], TOOL_SCHEMA_SHA256)
            self.assertEqual(metadata["source"]["reward_version"], "shopsimulator-reward-v3")
            self.assertEqual(
                metadata["source"]["environment_version"], "shopsimulator-environment-v2.1"
            )
            self.assertEqual(metadata["source"]["system_prompt_sha256"], SYSTEM_PROMPT_SHA256)
            sft_check = metadata["leakage"]["sft_task_id_check"]
            self.assertTrue(sft_check["checked"])
            self.assertEqual(sft_check["source_count"], 4)
            self.assertEqual(
                set(sft_check["overlaps"]),
                {"sft/process_train.jsonl", "sft/process_dev.jsonl", "sft/outcome_train.jsonl", "sft/outcome_dev.jsonl"},
            )
            self.assertEqual(metadata["build_parameters"]["sft_leak_check"], "enforced")
            self.assertEqual(
                metadata["build_parameters"]["tokenizer_revision"],
                builder.FROZEN_TOKENIZER_REVISION,
            )

    def test_tokenizer_revision_is_required_and_frozen(self):
        for revision in (None, "deadbeef"):
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(SystemExit, "tokenizer revision"):
                    builder.default_token_counter("Qwen/Qwen3.5-2B", revision)

    def test_tokenizer_loader_receives_frozen_revision(self):
        calls = []

        class FakeTokenizer:
            def encode(self, content, add_special_tokens=False):
                return list(content)

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append((name, kwargs))
                return FakeTokenizer()

        with patch.dict(
            "sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)}
        ):
            counter = builder.default_token_counter(
                "Qwen/Qwen3.5-2B", builder.FROZEN_TOKENIZER_REVISION
            )
            self.assertEqual(counter([{"role": "user", "content": "hello"}]), 5)

        self.assertEqual(calls, [(
            "Qwen/Qwen3.5-2B",
            {
                "trust_remote_code": True,
                "revision": builder.FROZEN_TOKENIZER_REVISION,
            },
        )])

    def test_real_build_requires_sft_task_id_files(self):
        """回归（audit item 2）：不传 SFT 文件的正式构建必须直接失败且信息可读。"""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture, sft_task_id_files=[])
            message = str(ctx.exception)
            self.assertIn("SFT leakage check is mandatory", message)
            self.assertIn("--sft-task-ids", message)
            self.assertIn("process/outcome", message)

    def test_formal_build_requires_four_named_nonempty_unique_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            one = fixture["sft_task_id_files"][0]
            with self.assertRaisesRegex(SystemExit, "exactly four"):
                _build(Path(tmp), fixture, sft_task_id_files=[one])

            empty = Path(tmp) / "sft" / "outcome_dev.jsonl"
            empty.write_text("[]", encoding="utf-8")
            files = [*fixture["sft_task_id_files"][:3], empty]
            with self.assertRaisesRegex(SystemExit, "must contain at least one"):
                _build(Path(tmp), fixture, sft_task_id_files=files, output_dir_name="empty")

            duplicate_role = Path(tmp) / "sft" / "process_train_copy.jsonl"
            duplicate_role.write_text("{\"task_id\": 108}\n", encoding="utf-8")
            files = [*fixture["sft_task_id_files"][:3], duplicate_role]
            with self.assertRaisesRegex(SystemExit, "provided more than once"):
                _build(Path(tmp), fixture, sft_task_id_files=files, output_dir_name="duplicate")

            ambiguous = Path(tmp) / "sft" / "teacher.jsonl"
            ambiguous.write_text("{\"task_id\": 999}\n", encoding="utf-8")
            files = [*fixture["sft_task_id_files"][:3], ambiguous]
            with self.assertRaisesRegex(SystemExit, "explicit identity"):
                _build(Path(tmp), fixture, sft_task_id_files=files, output_dir_name="ambiguous")

    def test_explicit_leak_check_override_is_recorded_and_blocks_launch(self):
        """--i-know-leak-check-is-required：醒目审计字段 + all_checks_passed=false，
        且 launcher（train_grpo.validate_data_metadata）拒绝该数据集。"""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            output_dir = Path(tmp) / "out"
            metadata = _build(
                Path(tmp), fixture, sft_task_id_files=[], allow_unchecked_sft_leakage=True
            )
            sft_check = metadata["leakage"]["sft_task_id_check"]
            self.assertFalse(sft_check["checked"])
            self.assertTrue(sft_check["skipped_via_explicit_override"])
            self.assertEqual(sft_check["override_flag"], "--i-know-leak-check-is-required")
            self.assertFalse(metadata["leakage"]["all_checks_passed"])
            self.assertEqual(
                metadata["build_parameters"]["sft_leak_check"], "skipped_explicitly"
            )

            train = output_dir / "train.parquet"
            validation = output_dir / "validation.parquet"
            with self.assertRaises(SystemExit) as ctx:
                train_grpo.validate_data_metadata(train, validation, output_dir / "metadata.json")
            self.assertIn("leakage checks did not pass", str(ctx.exception))

    def test_example_build_may_skip_sft_check_and_notes_pyarrow(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            metadata = _build(Path(tmp), fixture, sft_task_id_files=[], example=True)
            self.assertTrue(metadata["example"])
            self.assertFalse(metadata["leakage"]["sft_task_id_check"]["checked"])
            self.assertTrue(metadata["leakage"]["all_checks_passed"])
            self.assertIn("pyarrow", metadata["_example_note"])

    def test_unreachable_task_is_audited_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            metadata = _build(Path(tmp), fixture)
            audit = metadata["audit"]
            self.assertEqual(audit["excluded_task_count"], 1)
            entry = audit["excluded_tasks"][0]
            self.assertEqual(entry["task_id"], 20)
            self.assertEqual(entry["split"], "grpo_train")
            self.assertEqual(entry["reasons"], ["required_option_not_exposed"])
            self.assertEqual(
                audit["excluded_reason_counts"], {"required_option_not_exposed": 1}
            )

    def test_hidden_target_fields_never_reach_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            _build(Path(tmp), fixture)
            for name in ("train.parquet", "validation.parquet"):
                content = (Path(tmp) / "out" / name).read_text(encoding="utf-8")
                self.assertNotIn("HIDDEN", content)
                self.assertNotIn("family-", content)

    def test_hidden_value_scan_exempts_user_query_but_not_other_fields(self):
        # 真实任务里用户 query 可能提到目标品类词（如任务 788），合同只要求
        # 隐藏字段不进入 query 之外的部分。
        fact_row = {
            "query": "想买一双防滑拖鞋，冬天穿，室内用。",
            "difficulty": "easy",
            "target_title": "冬季室内防滑棉拖鞋标题",
            "target_options": ["防滑拖鞋"],
            "target_product_ids": ["B0HIDDEN123"],
            "product_family": "family-x",
        }
        row = builder.build_row(
            task_id=7,
            split_name="grpo_train",
            fact_row=fact_row,
            contract_sha256="b" * 64,
            data_source="shopsimulator-environment-v2.1",
            reward_model={"reward_version": "shopsimulator-reward-v3"},
        )
        # 把隐藏值人为塞进 extra_info 必须被扫描拦截。
        row["extra_info"]["contract_sha256"] = "B0HIDDEN123"
        with self.assertRaises(SystemExit) as ctx:
            builder.assert_no_forbidden_content(row, fact_row, 7)
        self.assertIn("forbidden fact field target_product_ids", str(ctx.exception))

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            first = _build(Path(tmp), fixture, output_dir_name="out-1")
            second = _build(Path(tmp), fixture, output_dir_name="out-2")
            for name in ("train.parquet", "validation.parquet", "metadata.json"):
                left = (Path(tmp) / "out-1" / name).read_bytes()
                right = (Path(tmp) / "out-2" / name).read_bytes()
                self.assertEqual(left, right, f"{name} is not deterministic")
            self.assertEqual(
                first["files"]["grpo_train"]["file_sha256"],
                second["files"]["grpo_train"]["file_sha256"],
            )

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp), final_task_ids=(10, 901))
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture)
            self.assertIn("leakage", str(ctx.exception))

    def test_sft_task_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            sft_ids = Path(tmp) / "sft_train.jsonl"
            sft_ids.write_text('{"task_id": 40}\n{"task_id": 60}\n', encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture, sft_task_id_files=[sft_ids])
            self.assertIn("overlap SFT task ids", str(ctx.exception))

    def test_missing_task_fact_row_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            remaining = [row for row in FACT_ROWS if row["task_id"] != 30]
            fixture = _write_fixture(Path(tmp), fact_rows=remaining)
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture)
            self.assertIn("task 30", str(ctx.exception))
            self.assertIn("missing from the frozen task facts", str(ctx.exception))

    def test_prompt_token_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture, max_prompt_tokens=16)
            self.assertIn("above the contract budget", str(ctx.exception))

    def test_existing_outputs_are_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            _build(Path(tmp), fixture)
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture)
            self.assertIn("refusing to overwrite", str(ctx.exception))
            forced = _build(Path(tmp), fixture, force=True)
            self.assertEqual(forced["files"]["grpo_train"]["rows"], 2)

    def test_task_facts_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            facts = fixture["task_facts"]
            original = json.loads(facts.read_text(encoding="utf-8").splitlines()[0])
            original["query"] = "被篡改过的需求"
            _write_jsonl(facts, [original] + list(FACT_ROWS[1:]))
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture)
            self.assertIn("task facts file does not match", str(ctx.exception))

    def test_contract_missing_version_fields_is_rejected(self):
        """runtime contract 必须携带 reward/env/tool/system-prompt 版本字段。"""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp))
            contract = fixture["contract"]
            data = json.loads(contract.read_text(encoding="utf-8"))
            del data["tool_schema_hash"]
            contract.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                _build(Path(tmp), fixture)
            self.assertIn("missing required version fields", str(ctx.exception))
            self.assertIn("tool_schema_hash", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
