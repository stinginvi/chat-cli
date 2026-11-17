"""成本计算的单元测试：不连网、不花钱。"""

import pytest

from llm import Usage


def test_zero_cost():
    assert Usage().cost("deepseek-flash") == 0.0


def test_output_only():
    u = Usage(completion_tokens=1_000_000)
    assert u.cost("deepseek-flash") == pytest.approx(2.0)


def test_cache_hit_is_cheaper_than_miss():
    hit = Usage(cache_hit_tokens=1_000_000).cost("deepseek-flash")
    miss = Usage(cache_miss_tokens=1_000_000).cost("deepseek-flash")
    assert hit < miss
    assert hit == pytest.approx(0.2)
    assert miss == pytest.approx(1.0)


def test_mixed():
    u = Usage(
        cache_hit_tokens=1_000_000,
        cache_miss_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert u.cost("deepseek-flash") == pytest.approx(0.2 + 1.0 + 2.0)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        Usage(completion_tokens=1).cost("no-such-model")


def test_add_accumulates():
    a = Usage(prompt_tokens=10, completion_tokens=5)
    b = Usage(prompt_tokens=20, cache_hit_tokens=3, completion_tokens=1)
    a.add(b)
    assert a.prompt_tokens == 30
    assert a.completion_tokens == 6
    assert a.cache_hit_tokens == 3
    assert a.total_tokens == 36
