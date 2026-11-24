"""命令行聊天入口：REPL + 斜杠命令。"""

import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError

from context import ContextManager
from llm import ChatClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "你是一个简洁的中文助手，回答尽量短。"

# 按状态码告诉用户该做什么，别把英文原文甩给用户
STATUS_HINTS = {
    400: "请求格式有问题，换个方式提问试试",
    401: "API key 无效，请检查 DEEPSEEK_API_KEY",
    402: "账户余额不足，请充值",
    422: "参数不合法，请检查输入",
    429: "请求太频繁，等几秒再试",
}

# 窗口虽然是 1M，但这里故意设小，方便看到压缩真的发生
MAX_CONTEXT_TOKENS = 4000


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    client = ChatClient()
    ctx = ContextManager(SYSTEM_PROMPT, max_tokens=MAX_CONTEXT_TOKENS)

    print(
        "输入 /exit 退出，/clear 清空上下文，/cost 看花费，"
        "/ctx 看上下文占用，/compress 手动压缩\n"
    )

    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text == "/exit":
            break

        if text == "/clear":
            # 只保留 system，历史才算真的清干净
            ctx.reset()
            print("已清空上下文")
            continue

        if text == "/cost":
            usage = client.total_usage
            print(
                f"累计 输入 {usage.prompt_tokens}（命中 {usage.cache_hit_tokens}） "
                f"输出 {usage.completion_tokens} tokens，"
                f"约 ¥{client.total_cost():.6f}"
            )
            continue

        if text == "/ctx":
            s = ctx.stats()
            tip = f"，上次估算误差 {s['error']:+.0f}%" if ctx.last_actual else ""
            summary = f"，摘要 {s['summary_chars']} 字" if s["summary_chars"] else ""
            print(
                f"上下文 {s['tokens']} / {s['budget']} tokens，"
                f"{s['pairs']} 轮{summary}{tip}"
            )
            continue

        if text == "/compress":
            compressed = ctx.compress(client)
            print(f"已压缩 {compressed} 条消息" if compressed else "当前不需要压缩")
            continue

        ctx.add_user(text)

        # 两条防线：先压缩（保住信息），压不动再裁剪（保住预算）
        try:
            compressed = ctx.compress(client)
        except Exception:
            logger.exception("摘要压缩失败，退回裁剪")
            compressed = 0
        if compressed:
            print(f"[上下文] 已把最老 {compressed} 条消息压成摘要")

        dropped = ctx.trim()
        if dropped:
            print(f"[上下文] 仍超预算，丢弃最老 {dropped} 条消息")
        if ctx.over_budget():
            print("[上下文] 最近几轮就超预算了，建议 /clear 后重新开始")

        sent = ctx.count()
        before = client.total_usage.prompt_tokens

        # 注意：APITimeoutError 是 APIConnectionError 的子类，必须写在前面的分支
        try:
            answer = client.stream_chat(ctx.build_messages())
        except APITimeoutError:
            print("\n[超时] 请求太久没有响应，稍后再试")
            logger.exception("请求超时")
            continue
        except APIConnectionError:
            print("\n[连接失败] 请检查网络后重试")
            logger.exception("网络连接失败")
            continue
        except APIStatusError as e:
            hint = STATUS_HINTS.get(e.status_code, "服务端返回错误，稍后再试")
            print(f"\n[{e.status_code}] {hint}")
            logger.exception("API 返回错误状态码 %s", e.status_code)
            continue
        except Exception:
            print("\n[未知错误] 详情见日志")
            logger.exception("未预期的异常")
            continue

        # 记下这一轮的估算值和真实值，/ctx 会显示误差
        ctx.last_estimate = sent
        ctx.last_actual = client.total_usage.prompt_tokens - before
        logger.info("上下文估算 %d，实际 %d tokens", sent, ctx.last_actual)

        if answer:
            ctx.add_assistant(answer)
        else:
            print("[空回答] 模型没有返回内容")


if __name__ == "__main__":
    main()
