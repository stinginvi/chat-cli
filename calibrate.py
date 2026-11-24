"""校准 token 估算：对比 count_tokens 的估算值和 API 返回的 usage。

用法：.venv/Scripts/python.exe calibrate.py
会真实调用 API，6 轮大约 0.001 元。
"""

import logging
import sys

from context import ContextManager
from llm import ChatClient

QUESTIONS = [
    "1+1 等于几？",
    "用一句话说明什么是 HTTP。",
    "GET 和 POST 的核心区别是什么？",
    "什么是幂等性？举一个例子。",
    "为什么说 PUT 是幂等的，POST 不是？",
    "用 Python 写一行代码求 1 到 100 的和。",
]

SYSTEM = "你是一个简洁的中文助手，回答尽量短。"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，不设会报错
    logging.disable(logging.INFO)

    client = ChatClient()
    ctx = ContextManager(SYSTEM, max_tokens=4000)

    print(f"{'轮次':<4}{'估算':>6}{'实际':>6}{'误差':>8}")
    for i, question in enumerate(QUESTIONS, 1):
        ctx.add_user(question)

        sent = ctx.count()
        before = client.total_usage.prompt_tokens
        answer = client.chat(ctx.messages, max_tokens=150)
        actual = client.total_usage.prompt_tokens - before

        error = (sent - actual) / actual * 100
        print(f"{i:<4}{sent:>6}{actual:>6}{error:>7.0f}%")

        ctx.add_assistant(answer)

    print(f"\n总花费 {client.total_cost():.6f} 元")


if __name__ == "__main__":
    main()
