#!/usr/bin/env python3
"""统一 Student Evaluator 正式入口（WP1）：Dev 与 Final-200 的 M0/M1/M2/M3 评测。

这是本实验唯一正式的评测 CLI；scripts/evaluate_shop_benchmark.py 只是
legacy/raw-trajectory 入口，不得作为 M0–M3 的正式结果来源。

Dev 示例（GAPS §5.5）：

    PYTHONPATH=src python scripts/evaluate_student.py \
      --model-label m2 --model-path outputs/models/process-sft-merged \
      --split dev --task-ids <generated-dev-task-ids.jsonl> \
      --output outputs/evaluation/dev/m2 \
      --environment-version shopsimulator-environment-v2.1 \
      --temperature 0 --top-p 1 --max-steps 35 --resume

Final 示例只在代码/模型/数据/环境/统计方案冻结后运行一次：

    PYTHONPATH=src python scripts/evaluate_student.py \
      --model-label m2 --model-path outputs/models/process-sft-merged \
      --split final-200 --task-ids data/splits/final_task_ids.jsonl \
      --output outputs/evaluation/final-200/m2 --temperature 0 --top-p 1 \
      --allow-blind-final-after-freeze

本入口不 import torch/transformers：actor 通过 OpenAI-compatible 推理服务访问
（本地 vLLM），环境 client 与 Judge/curator client 全部按需构建并注入 driver。
建议配合 ``set -o pipefail`` 用 ``2>&1 | tee <run>/stdout.log`` 保存日志。
API key 只从 OPENAI_API_KEY 环境变量读取，禁止写入命令行或 manifest。
"""

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.evaluation.driver import (
    DEFAULT_BASE_URL,
    DEFAULT_ENVIRONMENT_VERSION,
    FINAL_SPLIT,
    SUPPORTED_SPLITS,
    EvaluationDriver,
    LazyEnvironmentTaskFactsSource,
)
from shopping_grpo.evaluation.model_client import (
    DEFAULT_FLASH_MODEL,
    DEFAULT_PRO_MODEL,
    OpenAIJSONClient,
)
from shopping_grpo.evaluation.rollout import OpenAIChatClient


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="统一 Student Evaluator（WP1 正式评测入口）")
    parser.add_argument("--model-label", required=True, help="模型标签，如 m0/m1/m2/m3")
    parser.add_argument("--model-path", required=True, help="冻结模型/merged checkpoint 路径（仅记录审计 hash 身份）")
    parser.add_argument("--model-revision", default=None, help="模型 revision（可缺省）")
    parser.add_argument("--split", required=True, choices=list(SUPPORTED_SPLITS))
    parser.add_argument("--task-ids", required=True, type=Path,
                        help="task split JSONL；每行至少含 task_id（Final 必须是 IDs-only，"
                             "合成分割可内嵌 goal/target_product）")
    parser.add_argument("--task-facts-path", default=None,
                        help="环境侧冻结 task facts 文档（IDs-only split 运行期取 facts 用）；"
                             "缺省读 SHOP_TASK_FACTS_PATH 环境变量")
    parser.add_argument("--output", required=True, type=Path, help="run 目录（不覆盖已有输出）")
    parser.add_argument("--environment-version", default=DEFAULT_ENVIRONMENT_VERSION)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--resume", action="store_true",
                        help="从已有 cache 恢复；schema/hash 不匹配会被拒绝")
    parser.add_argument("--allow-blind-final-after-freeze", action="store_true",
                        help="显式确认冻结后允许运行 Final-200（blind guard 仍校验 task ID 集合）")
    parser.add_argument("--env-base-url", default=DEFAULT_BASE_URL,
                        help="ShopSimulator 环境服务地址")
    parser.add_argument("--actor-base-url", default=None,
                        help="actor 推理服务地址；缺省读 OPENAI_BASE_URL")
    parser.add_argument("--actor-model-name", default=None,
                        help="推理服务上的模型名；缺省使用 --model-label")
    parser.add_argument("--actor-max-tokens", type=int, default=512,
                        help="单次 actor 生成上限，防止未调用工具时耗尽上下文")
    parser.add_argument("--actor-timeout", type=int, default=180)
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--judge-base-url", default=None,
                        help="Judge/curator 服务地址；缺省读 OPENAI_BASE_URL")
    parser.add_argument("--judge-model", default=DEFAULT_PRO_MODEL)
    parser.add_argument("--curator-model", default=DEFAULT_FLASH_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--rubric-version", default=None, help="Rubric 版本标签")
    parser.add_argument("--shared-rubric-cache", type=Path, default=None,
                        help="跨模型共享的 rubric cache JSONL（可选）")
    parser.add_argument("--run-id", default=None, help="显式 run id（缺省内容寻址）")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps 必须为正数")
    if args.actor_max_tokens < 1:
        parser.error("--actor-max-tokens 必须为正数")
    if args.split == FINAL_SPLIT and not args.allow_blind_final_after_freeze:
        parser.error(
            "Final-200 只允许在代码/模型/数据/环境/统计方案冻结后运行一次；"
            "必须显式传入 --allow-blind-final-after-freeze"
        )
    return args


def _resolve_llm_endpoint(explicit, *, what):
    base_url = explicit or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise SystemExit(f"{what} 需要 --参数或 OPENAI_BASE_URL 环境变量")
    api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    return str(base_url), api_key


def build_driver(
    args,
    *,
    actor_client=None,
    judge_client=None,
    curator_client=None,
    env_factory=None,
):
    """把 CLI 参数装配成 EvaluationDriver；client 均可注入以便测试。"""

    if actor_client is None:
        actor_base_url, actor_api_key = _resolve_llm_endpoint(
            args.actor_base_url, what="actor"
        )
        actor_client = OpenAIChatClient(
            model=args.actor_model_name or args.model_label,
            base_url=actor_base_url,
            api_key=actor_api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.actor_timeout,
            max_tokens=args.actor_max_tokens,
            context_window=args.context_window or None,
            observation_token_budget=args.observation_token_budget or None,
        )
    if judge_client is None or curator_client is None:
        llm_base_url, llm_api_key = _resolve_llm_endpoint(
            args.judge_base_url, what="judge/curator"
        )
        judge_client = judge_client or OpenAIJSONClient(
            model=args.judge_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            max_tokens=args.judge_max_tokens,
            timeout=args.llm_timeout,
            response_format_json=True,
        )
        curator_client = curator_client or OpenAIJSONClient(
            model=args.curator_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            max_tokens=args.judge_max_tokens,
            timeout=args.llm_timeout,
            response_format_json=True,
        )
    return EvaluationDriver(
        run_dir=args.output,
        task_split_path=args.task_ids,
        split=args.split,
        actor_label=args.model_label,
        actor_client=actor_client,
        judge_client=judge_client,
        curator_client=curator_client,
        env_factory=env_factory,
        task_facts_source=LazyEnvironmentTaskFactsSource(args.task_facts_path),
        base_url=args.env_base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.max_steps,
        environment_version=args.environment_version,
        model_path=str(args.model_path),
        model_revision=args.model_revision,
        served_model_name=args.actor_model_name or args.model_label,
        judge_model=args.judge_model,
        curator_model=args.curator_model,
        rubric_version=args.rubric_version or "wp1-student-eval-rubric-v1",
        actor_protocol={
            "actor_max_tokens": args.actor_max_tokens,
            "actor_context_window": args.context_window,
            "actor_observation_token_budget": args.observation_token_budget,
        },
        shared_rubric_cache=args.shared_rubric_cache,
        resume=args.resume,
        allow_blind_final=args.split == FINAL_SPLIT
        and args.allow_blind_final_after_freeze,
        run_id=args.run_id,
    )


def main(argv=None):
    args = parse_args(argv)
    if not args.task_ids.is_file():
        raise SystemExit(f"task split 文件不存在: {args.task_ids}")
    if not Path(args.model_path).exists():
        raise SystemExit(f"model path 不存在: {args.model_path}")
    driver = build_driver(args)
    result = driver.run()
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
