"""Day 2 验证：非流式 vs 流式的体感差异。

跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe demo_day2.py
"""

import logging
import time

from llm import ChatClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

client = ChatClient()
messages = [{"role": "user", "content": "用 150 字左右介绍 Python 的历史"}]

print("=== 非流式：等全部生成完才显示 ===")
t0 = time.perf_counter()
answer = client.chat(messages, max_tokens=400)
elapsed = time.perf_counter() - t0
print(answer)
print(f"-- 总耗时 {elapsed:.2f}s，{len(answer)} 字，累计成本 ¥{client.total_cost():.6f}\n")

print("=== 流式：第一个字应该马上出现 ===")
t0 = time.perf_counter()
answer = client.stream_chat(messages, max_tokens=400)
elapsed = time.perf_counter() - t0
print(f"-- 总耗时 {elapsed:.2f}s，{len(answer)} 字，累计成本 ¥{client.total_cost():.6f}")
