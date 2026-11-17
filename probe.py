"""探测脚本：打印真实响应结构，用于对照知识笔记。

跑法：.venv/Scripts/python.exe probe.py
"""

import json

from openai import OpenAI

from config import get_api_key, get_base_url, get_model


def main() -> None:
    client = OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=60,
    )

    print("=== 可用模型 ===")
    for m in client.models.list().data:
        print(m.id)

    model = get_model()
    print(f"\n=== 非流式调用：{model} ===")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用一句话回答：1+1 等于几？"}],
        max_tokens=100,
    )
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))

    print(f"\n=== 流式调用：{model} ===")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "数到三，每个数字之间用逗号"}],
        max_tokens=100,
        stream=True,
        stream_options={"include_usage": True},
    )
    for i, chunk in enumerate(stream):
        print(f"[chunk {i}] {json.dumps(chunk.model_dump(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
