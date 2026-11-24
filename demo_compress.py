"""对比两种上下文策略：滑动窗口（trim）vs 摘要压缩（compress）。

同一段 30 轮对话，两种策略各跑一遍，看最后能不能答对早期信息。
对话内容写死在下面，不调 API；真正花钱的只有「压缩」和「最后一次提问」。

用法：.venv/Scripts/python.exe demo_compress.py
"""

import logging
import sys

from context import ContextManager
from llm import ChatClient

SYSTEM = "你是一个简洁的中文助手，回答尽量短。"
MAX_TOKENS = 300        # 故意设小，让 30 轮对话一定会超预算
KEEP_PAIRS = 2          # 最近 2 轮保留原文

# 30 轮对话：早期埋了身份、偏好、项目名，后半段是无关寒暄
DIALOG = [
    ("我叫雨霖铃，之前做后端开发，现在转行学 Agent 开发。", "记住了。"),
    ("我的技术栈是 Python、FastAPI、PostgreSQL。", "基础不错。"),
    ("我不喜欢用 ORM，更习惯直接写 SQL。", "明白，会避开 ORM。"),
    ("我在做第一个项目，叫 chat-cli。", "命令行聊天工具，合适。"),
    ("它用的是 DeepSeek 的 API。", "成本较低的选择。"),
    ("我的目标是三个月内找到 Agent 开发的工作。", "时间紧凑。"),
    ("我每天能学 6 个小时。", "强度足够。"),
    ("我比较怕数学，但不怕写代码。", "工程路线更适合你。"),
    ("我做过一个用户资源 API，有 JWT 鉴权。", "已经是完整项目。"),
    ("那个项目用了 Redis 做缓存。", "缓存是常见考点。"),
    ("我打算下一步学 RAG。", "顺序合理。"),
    ("但我不知道 RAG 和 Agent 的关系。", "RAG 是 Agent 的一种能力。"),
    ("我的预算不多，每月 API 费用想控制在 100 元内。", "够用。"),
    ("我用的编辑器是 PyCharm。", "好的。"),
    ("我用 Windows 11 和 VMware。", "记住了。"),
    ("所以我没开 WSL，Docker 只能写不能跑。", "可以理解。"),
    ("我的 PostgreSQL 版本是 18。", "比较新。"),
    ("我算法题只做了 16 道。", "面试前要补。"),
    ("但我不想刷太多题，想把时间放在项目上。", "项目优先合理。"),
    ("我的 GitHub 上有六个公开仓库。", "简历有东西写了。"),
    ("每个仓库都打了 v0.1 标签。", "习惯不错。"),
    ("我还没写过集成测试。", "可以后面补。"),
    ("我英语阅读还行，听说不行。", "看文档够用。"),
    ("我打算下个月开始投简历。", "提前准备。"),
    ("我担心项目深度不够。", "把评测做扎实。"),
    ("我每天记录学习日志。", "好习惯。"),
    ("每周会写一次复盘。", "方便回顾。"),
    ("我想做一个能展示给面试官的作品。", "评测报告最有说服力。"),
    ("你觉得先补算法还是先做 RAG？", "建议先做 RAG。"),
    ("综合我最开始说的背景和偏好：下一步该重点学什么？我对 ORM 是什么态度？", None),
]

CHECK_WORDS = ("雨霖铃", "不喜欢", "SQL", "RAG")


def run(strategy: str, client: ChatClient) -> tuple[int, int, float, str, str]:
    """跑一遍策略，返回 (估算 token, 实际 token, 花费, 回答, 摘要)。"""
    ctx = ContextManager(SYSTEM, max_tokens=MAX_TOKENS, keep_pairs=KEEP_PAIRS)

    for user, assistant in DIALOG[:-1]:
        ctx.add_user(user)
        ctx.add_assistant(assistant)

    ctx.add_user(DIALOG[-1][0])

    if strategy == "trim":
        ctx.trim()
    else:
        ctx.compress(client)
        ctx.trim()          # 压缩后还超预算，再兜底裁剪

    sent = ctx.count()
    before_tokens = client.total_usage.prompt_tokens
    before_cost = client.total_cost()

    answer = client.chat(ctx.build_messages(), max_tokens=300)

    actual = client.total_usage.prompt_tokens - before_tokens
    cost = client.total_cost() - before_cost
    return sent, actual, cost, answer, ctx.summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.disable(logging.INFO)

    client = ChatClient()
    print(f"同一段 {len(DIALOG)} 轮对话，预算 {MAX_TOKENS} tokens，保留最近 {KEEP_PAIRS} 轮\n")

    for name, strategy in [("滑动窗口", "trim"), ("摘要压缩", "compress")]:
        sent, actual, cost, answer, summary = run(strategy, client)
        hits = "、".join(w for w in CHECK_WORDS if w in answer) or "无"

        print(f"【{name}】")
        print(f"  估算 {sent} / 实际 {actual} tokens，花费 {cost:.6f} 元")
        print(f"  答对关键词：{hits}")
        if summary:
            print(f"  摘要：{summary}")
        print(f"  回答：{answer.strip()}\n")


if __name__ == "__main__":
    main()
