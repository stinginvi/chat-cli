"""流式拼接的单元测试：假 stream，不连网、不花钱。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm import ChatClient, Usage


def chunk(content=None, finish=None, usage=None, choices=True):
    """造一个假的 chunk，形状和 SDK 返回的一致。"""
    delta = SimpleNamespace(content=content, reasoning_content=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice] if choices else [], usage=usage)


def fake_usage(prompt=10, completion=2):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_cache_hit_tokens=0,
        prompt_cache_miss_tokens=prompt,
        completion_tokens_details=None,
    )


@pytest.fixture
def chat_client():
    """绕开 __init__，避免单测依赖真实 API key。"""
    c = ChatClient.__new__(ChatClient)
    c.model = "deepseek-flash"
    c.total_usage = Usage()
    c.client = MagicMock()
    return c


def test_joins_stream_text(chat_client, capsys):
    chat_client.client.chat.completions.create.return_value = iter([
        chunk("你"),
        chunk(None),                       # 空文本块，要跳过
        chunk("好", finish="stop"),
        chunk(choices=False, usage=fake_usage()),  # 最后一个 usage-only 块
    ])

    answer = chat_client.stream_chat([{"role": "user", "content": "hi"}])

    assert answer == "你好"
    assert chat_client.total_usage.prompt_tokens == 10
    assert chat_client.total_usage.completion_tokens == 2


def test_empty_stream_returns_empty_string(chat_client):
    chat_client.client.chat.completions.create.return_value = iter([
        chunk(choices=False, usage=fake_usage()),
    ])

    assert chat_client.stream_chat([{"role": "user", "content": "hi"}]) == ""


def test_usage_is_accumulated_across_calls(chat_client):
    def make_stream():
        return iter([chunk("好"), chunk(choices=False, usage=fake_usage(5, 1))])

    chat_client.client.chat.completions.create.side_effect = lambda **kw: make_stream()

    chat_client.stream_chat([{"role": "user", "content": "1"}])
    chat_client.stream_chat([{"role": "user", "content": "2"}])

    assert chat_client.total_usage.prompt_tokens == 10
    assert chat_client.total_usage.completion_tokens == 2
