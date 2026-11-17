"""Day 1 验证脚本：写完后跑这个，对照输出。

跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe demo_day1.py
"""

import logging

from llm import ChatClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

client = ChatClient()

answer = client.chat(
    [{"role": "user", "content": "用一句话解释什么是 token"}],
    max_tokens=200,
)

print("\n--- 回答 ---")
print(answer)

u = client.total_usage
print("\n--- 账本 ---")
print(f"输入 {u.prompt_tokens}（命中 {u.cache_hit_tokens} / 未命中 {u.cache_miss_tokens}）")
print(f"输出 {u.completion_tokens}，合计 {u.total_tokens} tokens")
print(f"本次成本 ¥{client.total_cost():.6f}")
