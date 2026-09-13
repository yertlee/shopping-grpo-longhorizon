"""Minimal OpenAI-compatible Teacher client with local tokenizer accounting."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from threading import local
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HuggingFaceTokenCounter:
    def __init__(self, model_path: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for tokenizer accounting") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )

    def text(self, value: str) -> int:
        return len(self.tokenizer.encode(str(value), add_special_tokens=False))

    def chat(self, messages: list[dict], tools: list[dict]) -> int:
        # The Qwen chat template expects tool_call.function.arguments to be a
        # mapping (it iterates `.items()`), but the Teacher API returns it as a
        # JSON string. Normalize on a copy so the raw messages stay untouched.
        normalized = []
        for message in messages:
            message = dict(message)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                parsed_calls = []
                for tool_call in tool_calls:
                    tool_call = dict(tool_call)
                    function = tool_call.get("function")
                    if isinstance(function, dict):
                        function = dict(function)
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except (json.JSONDecodeError, ValueError):
                                arguments = {"__raw_arguments__": arguments}
                        if not isinstance(arguments, Mapping):
                            arguments = {"__raw_arguments__": arguments}
                        function["arguments"] = dict(arguments)
                        tool_call["function"] = function
                    parsed_calls.append(tool_call)
                message["tool_calls"] = parsed_calls
            normalized.append(message)
        rendered = self.tokenizer.apply_chat_template(
            normalized,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered.get("input_ids")
        if rendered is None:
            raise ValueError("chat template returned no input_ids")
        if hasattr(rendered, "shape"):
            shape = tuple(rendered.shape)
            return int(shape[-1]) if shape else 0
        if rendered and isinstance(rendered[0], (list, tuple)):
            return len(rendered[0])
        return len(rendered)


class OpenAICompatibleTeacher:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
        token_counter: HuggingFaceTokenCounter,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.token_counter = token_counter
        self._thread_state = local()

    @property
    def last_metadata(self) -> dict:
        return getattr(self._thread_state, "last_metadata", {})

    @last_metadata.setter
    def last_metadata(self, value: dict) -> None:
        self._thread_state.last_metadata = value

    def count_text_tokens(self, value: str) -> int:
        return self.token_counter.text(value)

    def count_chat_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self.token_counter.chat(messages, tools)

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.model.casefold().startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "commerce-agent-posttrain/0.1",
            },
            method="POST",
        )
        for retry in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                    self.last_metadata = {
                        "provider_request_id": response.headers.get("x-request-id"),
                        "usage": value.get("usage") or {},
                        "finish_reason": (value.get("choices") or [{}])[0].get(
                            "finish_reason"
                        ),
                    }
                    return dict(value["choices"][0]["message"])
            except (HTTPError, URLError, TimeoutError, OSError):
                if retry == 2:
                    raise
                time.sleep(retry + 1)
        raise AssertionError("unreachable")
