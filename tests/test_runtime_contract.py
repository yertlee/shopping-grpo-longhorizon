import json
import tempfile
import unittest
from pathlib import Path

from commerce_posttrain.contracts.runtime import (
    DEFAULT_OUTPUT,
    build_runtime_contract,
    validate_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTest(unittest.TestCase):
    def test_checked_in_runtime_contract_matches_current_sources(self):
        contract = validate_runtime_contract(DEFAULT_OUTPUT)
        self.assertEqual(contract["attempts_per_task"], 3)
        self.assertEqual(
            contract["environment_version"], "shopsimulator-environment-v2.1"
        )
        self.assertEqual(contract["reward_version"], "shopsimulator-reward-v3")
        self.assertEqual(
            contract["process_contract"]["evidence_source"],
            "actor_visible_observation_only",
        )

    def test_contract_is_deterministic(self):
        self.assertEqual(build_runtime_contract(), build_runtime_contract())

    def test_contract_tampering_fails_closed(self):
        contract = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        contract["observation_search_top_k"] = 10
        with self.assertRaisesRegex(ValueError, "observation_search_top_k"):
            validate_runtime_contract(contract)

    def test_runtime_config_rejects_wrong_attempt_count(self):
        config = json.loads(
            (ROOT / "configs/runtime/runtime.json").read_text(encoding="utf-8")
        )
        config["attempts_per_task"] = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly three attempts"):
                build_runtime_contract(path)


if __name__ == "__main__":
    unittest.main()
