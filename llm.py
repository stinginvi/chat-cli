"""LLM 调用封装：非流式、流式、重试、成本累计。

这一层的目标是「可测」：除了 self.client 发请求那一刻，其余逻辑都是纯计算。
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

import config

logger = logging.getLogger(__name__)

# 元 / 百万 token。价格以 platform.deepseek.com 为准，变动只改这里。
PRICE = {
    "deepseek-flash": {"cache_hit": 0.2, "cache_miss": 1.0, "output": 2.0},
    "deepseek-v4-pro": {"cache_hit": 1.0, "cache_miss": 12.0, "output": 24.0},
}

# 这些状态码值得重试：限流、请求超时、服务端故障
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class Usage:
    """一次调用的 token 账本，也可以累计多轮。"""

    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, model: str) -> float:
        """算出成本，单位：元。

        TODO: 按这个公式实现（价格单位是「元 / 百万 token」，记得除 1e6）
            成本 = 命中 × 命中价 + 未命中 × 未命中价 + 输出 × 输出价
        模型不在 PRICE 里时直接 KeyError，不要静默返回 0。
        """
        return (
            self.cache_hit_tokens * PRICE[model]["cache_hit"]
            + self.cache_miss_tokens * PRICE[model]["cache_miss"]
            + self.completion_tokens * PRICE[model]["output"]
        ) / 1e6

    def add(self, other: "Usage") -> None:
        """把另一次调用的账加进来（累计用）。"""
        self.prompt_tokens += other.prompt_tokens
        self.cache_hit_tokens += other.cache_hit_tokens
        self.cache_miss_tokens += other.cache_miss_tokens
        self.completion_tokens += other.completion_tokens

    @classmethod
    def from_sdk(cls, usage) -> "Usage":
        """把 OpenAI SDK 返回的 usage 对象转成本类的账本。

        注意：DeepSeek 专有字段名是 prompt_cache_hit_tokens / prompt_cache_miss_tokens，
        关闭思考时 completion_tokens_details 会整个是 None，所以这里一律用 getattr 容错。
        """
        return cls(
            prompt_tokens=usage.prompt_tokens or 0,
            cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
            cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
            completion_tokens=usage.completion_tokens or 0,
        )


def with_retry(max_attempts: int = 4):
    """指数退避重试装饰器，只包住「发请求」这一步。

    TODO:
    1. 用 for attempt in range(max_attempts) 循环调用 fn
    2. 捕获 (APIConnectionError, APITimeoutError) → 可重试
    3. 捕获 APIStatusError 时，只有 e.status_code in RETRYABLE_STATUS 才重试，
       其余的（400/401/402/422）立刻重新抛出——重试也没用
    4. 最后一次还失败就直接 raise
    5. 退避时间：`2 ** attempt + random.uniform(0, 0.5)` 秒，用 time.sleep
       动手前先想清楚：为什么要加随机抖动？
    """
    assert max_attempts >= 1, "重试次数必须 >= 1"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                last = attempt == max_attempts - 1
                try:
                    return fn(*args, **kwargs)
                except APIStatusError as e:
                    if e.status_code not in RETRYABLE_STATUS or last:
                        raise
                    logger.warning("HTTP %s，第 %d 次重试", e.status_code, attempt + 1)
                except (APIConnectionError, APITimeoutError) as e:
                    if last:
                        raise
                    logger.warning("连接/超时错误，第 %d 次重试：%s", attempt + 1, e)
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
            raise AssertionError("重试循环结束后不应执行到这里")

        return wrapper

    return decorator




class ChatClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.get_model()
        self.client = OpenAI(
            api_key=config.get_api_key(),
            base_url=config.get_base_url(),
            timeout=60,
        )
        self.total_usage = Usage()

    def _extra_body(self) -> dict:
        """V4 默认开思考，这里默认关掉，减少无效输出和费用。"""
        if config.thinking_enabled():
            return {}
        return {"thinking": {"type": "disabled"}}

    @with_retry()
    def chat(self, messages: list[dict], **params) -> str:
        """非流式调用，返回回答文本。"""
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            extra_body=self._extra_body(), **params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        u = Usage.from_sdk(resp.usage)
        self.total_usage.add(u)
        finish = resp.choices[0].finish_reason

        logger.info(
            "model=%s in=%d(hit=%d/miss=%d) out=%d cost=¥%.6f %dms finish=%s",
            self.model, u.prompt_tokens, u.cache_hit_tokens,
            u.cache_miss_tokens, u.completion_tokens,
            u.cost(self.model), elapsed_ms, finish,
        )
        if finish == "length":
            logger.warning("回答被 max_tokens 截断，已返回不完整内容")

        return resp.choices[0].message.content or ""

    def stream_chat(self, messages, **params) -> str:
        start = time.perf_counter()
        parts = []
        ttft_ms = None  # 首字延迟，第一次拿到文字时记录
        usage = Usage()  # 本次账本，最后从最后一个 chunk 拿
        finish = None  # 只有最后那个带 choices 的块才有 finish_reason

        stream = self.client.chat.completions.create(
            model=self.model, messages=messages,
            extra_body=self._extra_body(), **params,
            stream=True, stream_options={"include_usage": True},
        )

        for chunk in stream:
            # ① 先看 usage：最后一个 chunk 才会带，拿到就更新账本
            if chunk.usage is not None:
                usage = Usage.from_sdk(chunk.usage)

            # ② 再跳过没有 choices 的块
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason

            delta = choice.delta
            text = delta.content  # 注意这里不是 .get

            # ③ 空文本跳过（None 和 "" 都算空）
            if not text:
                continue

            # ④ 第一次拿到文字时记 TTFT
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000

            # ⑤ 收集 + 立即打印
            parts.append(text)
            print(text, end="", flush=True)

        # ⑥ 收尾：换行、记账、日志、返回
        print()  # 流式没换行，这里补一个
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.total_usage.add(usage)

        logger.info(
            "model=%s in=%d(hit=%d/miss=%d) out=%d cost=¥%.6f %dms ttft=%.0fms finish=%s",
            self.model, usage.prompt_tokens, usage.cache_hit_tokens,
            usage.cache_miss_tokens, usage.completion_tokens,
            usage.cost(self.model), elapsed_ms, ttft_ms or 0.0, finish,
        )
        if finish == "length":
            logger.warning("回答被 max_tokens 截断，已返回不完整内容")

        return "".join(parts)

    def total_cost(self) -> float:
        return self.total_usage.cost(self.model)
