import json
import tempfile
import unittest
from pathlib import Path

from commerce_posttrain.collection.client import HuggingFaceTokenCounter
from commerce_posttrain.collection.collector import collect_attempt
from commerce_posttrain.collection.schema import (
    strict_gold_success,
    validate_raw_trajectory,
)
from commerce_posttrain.collection.store import RawTrajectoryStore
from scripts.collect_teacher import summarize

ASIN = "12345678"


def tool_call(name, arguments, index):
    return {
        "id": f"call-{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeClient:
    def __init__(self):
        self.calls = [
            tool_call("search_products", {"query": "保温杯"}, 0),
            tool_call("open_product", {"asin": ASIN}, 1),
            tool_call("buy_now", {}, 2),
        ]
        self.last_metadata = {}

    def count_text_tokens(self, text):
        return len(text)

    def count_chat_tokens(self, messages, tools):
        return len(json.dumps([messages, tools], ensure_ascii=False))

    def complete(self, messages, tools):
        call = self.calls.pop(0)
        self.last_metadata = {"provider_request_id": call["id"], "usage": {}}
        return {"role": "assistant", "content": None, "tool_calls": [call]}


class FakeEnv:
    def __init__(self, base_url):
        self.base_url = base_url
        self.released = False

    def reset(self, task_id):
        return {
            "env_idx": 0,
            "instruction": "买一个保温杯",
            "observation_state": {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_home",
                "search_available": True,
                "actions": [],
            },
        }

    def step(self, action):
        if action.startswith("search["):
            return {
                "reward": 0.0,
                "done": False,
                "observation_state": {
                    "observation_version": "shopping-observation-v2",
                    "page_type": "search_results",
                    "search_available": True,
                    "query": "保温杯",
                    "normalized_query": "保温杯",
                    "page": 1,
                    "total_pages": 1,
                    "total_results": 1,
                    "rank_start": 1,
                    "rank_end": 1,
                    "products": [
                        {
                            "rank": 1,
                            "asin": ASIN,
                            "price": "39.0",
                            "brand": "测试",
                            "category": "杯",
                            "key_attributes": ["保温"],
                            "title": "测试保温杯",
                        }
                    ],
                    "actions": [ASIN],
                },
            }
        if action == f"click[{ASIN}]":
            return {
                "reward": 0.0,
                "done": False,
                "observation_state": {
                    "observation_version": "shopping-observation-v2",
                    "page_type": "product_detail",
                    "search_available": False,
                    "product": {
                        "asin": ASIN,
                        "title": "测试保温杯",
                        "brand": "测试",
                        "category": "杯",
                        "price": "39.0",
                        "key_attributes": ["保温"],
                    },
                    "selected_price": "39.0",
                    "selected_options": {},
                    "available_options": {},
                    "actions": ["Buy Now", "Back to Search"],
                },
            }
        return {
            "reward": 1.0,
            "done": True,
            "over": True,
            "observation_state": {
                "observation_version": "shopping-observation-v2",
                "page_type": "terminal",
                "search_available": False,
                "actions": [],
            },
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
            "purchase": {"asin": ASIN, "price": 39.0, "options": {}},
            "goal": {"asin": "PRIVATE"},
        }

    def release(self):
        self.released = True


def runtime_contract():
    return {
        "contract_sha256": "a" * 64,
        "teacher": {
            "model": "deepseek-v4-flash",
            "temperature": 0.7,
            "top_p": 0.9,
            "thinking": False,
        },
        "max_steps": 35,
        "context_window": 24576,
        "context_safety_margin": 512,
        "max_generated_tokens_per_turn": 512,
        "observation_search_tokens": 1536,
        "observation_detail_tokens": 4096,
        "observation_generic_tokens": 768,
        "observation_search_top_k": 20,
    }


def collected():
    return collect_attempt(
        {"task_id": 7, "query": "买一个保温杯"},
        attempt_index=0,
        runtime_contract=runtime_contract(),
        client=FakeClient(),
        base_url="http://fake",
        system_prompt="shopping system",
        env_factory=FakeEnv,
    )


class CollectionRuntimeTest(unittest.TestCase):
    def test_token_counter_wraps_malformed_tool_arguments_for_qwen_template(self):
        class FakeTokenizer:
            def __init__(self):
                self.messages = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
                self.assert_mapping = isinstance(arguments, dict)
                return [1, 2, 3]

        counter = object.__new__(HuggingFaceTokenCounter)
        counter.tokenizer = FakeTokenizer()
        original = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "arguments": "{not-valid-json",
                    },
                }
            ],
        }

        self.assertEqual(counter.chat([original], []), 3)
        self.assertTrue(counter.tokenizer.assert_mapping)
        self.assertEqual(
            counter.tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"],
            {"__raw_arguments__": "{not-valid-json"},
        )
        self.assertEqual(
            original["tool_calls"][0]["function"]["arguments"], "{not-valid-json"
        )

    def test_token_counter_reads_input_ids_from_batch_encoding_mapping(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {
                    "input_ids": [11, 12, 13, 14, 15],
                    "attention_mask": [1, 1, 1, 1, 1],
                }

        counter = object.__new__(HuggingFaceTokenCounter)
        counter.tokenizer = FakeTokenizer()
        self.assertEqual(counter.chat([], []), 5)

    def test_fake_actor_loop_produces_valid_strict_raw_record(self):
        trajectory = collected()
        self.assertEqual(validate_raw_trajectory(trajectory), trajectory)
        self.assertTrue(strict_gold_success(trajectory))
        self.assertTrue(trajectory["released"])
        self.assertEqual(
            [event["type"] for event in trajectory["events"]],
            [
                "environment_reset",
                "assistant",
                "environment_step",
                "assistant",
                "environment_step",
                "assistant",
                "environment_step",
            ],
        )

    def test_store_rebuilds_resume_index_and_rejects_duplicate_completion(self):
        trajectory = collected()
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir) / "raw.jsonl"
            index = Path(temp_dir) / "completion_index.json"
            store = RawTrajectoryStore(raw, index)
            store.append(trajectory)
            rebuilt = RawTrajectoryStore(raw, index)
            self.assertEqual(rebuilt.completed_request_ids(), {trajectory["request_id"]})
            self.assertEqual(rebuilt.read_all(), [trajectory])
            self.assertEqual(summarize(rebuilt.read_all())["total"], 1)
            with self.assertRaisesRegex(ValueError, "already completed"):
                rebuilt.append(trajectory)


if __name__ == "__main__":
    unittest.main()
