"""命令行聊天入口：REPL + 斜杠命令。"""

import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    client = ChatClient()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("输入 /exit 退出，/clear 清空上下文，/cost 查看累计花费\n")

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
            messages = messages[:1]
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

        messages.append({"role": "user", "content": text})

        # 注意：APITimeoutError 是 APIConnectionError 的子类，必须写在前面的分支
        try:
            answer = client.stream_chat(messages)
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

        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            print("[空回答] 模型没有返回内容")


if __name__ == "__main__":
    main()
