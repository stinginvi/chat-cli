"""重试逻辑的单元测试：用假异常，不连网。"""

import httpx2 as httpx
import pytest
from openai import APIConnectionError, APIStatusError

from llm import with_retry


def make_status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(code, request=request)
    return APIStatusError("boom", response=response, body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("llm.time.sleep", lambda _: None)


def test_retries_connection_error_then_succeeds():
    calls = {"n": 0}

    @with_retry(max_attempts=3)
    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise make_status_error(503)
        return "ok"

    assert fn() == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    @with_retry(max_attempts=3)
    def fn():
        calls["n"] += 1
        raise make_status_error(429)

    with pytest.raises(APIStatusError):
        fn()
    assert calls["n"] == 3


def test_does_not_retry_401():
    calls = {"n": 0}

    @with_retry(max_attempts=4)
    def fn():
        calls["n"] += 1
        raise make_status_error(401)

    with pytest.raises(APIStatusError):
        fn()
    assert calls["n"] == 1


def test_does_not_retry_400():
    calls = {"n": 0}

    @with_retry(max_attempts=4)
    def fn():
        calls["n"] += 1
        raise make_status_error(400)

    with pytest.raises(APIStatusError):
        fn()
    assert calls["n"] == 1


def test_retries_timeout():
    calls = {"n": 0}

    @with_retry(max_attempts=2)
    def fn():
        calls["n"] += 1
        raise APIConnectionError(request=httpx.Request("POST", "https://x"))

    with pytest.raises(APIConnectionError):
        fn()
    assert calls["n"] == 2
