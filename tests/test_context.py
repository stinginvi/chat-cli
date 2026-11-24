"""上下文管理的单元测试。不连网、不花钱。"""

from context import ContextManager, count_tokens, estimate_tokens


class FakeClient:
    """假客户端：不连网，固定返回一段摘要，顺便记下收到的提示词。"""

    def __init__(self, reply: str = "用户叫雨霖铃，不喜欢 ORM。"):
        self.reply = reply
        self.last_messages = None

    def chat(self, messages, **params) -> str:
        self.last_messages = messages
        return self.reply


def make_manager(**kwargs) -> ContextManager:
    """所有测试共用的小工厂，方便统一改预算。"""
    return ContextManager("你是助手", **kwargs)


def long_chat(m: ContextManager, rounds: int) -> None:
    """灌一段足够长的对话，用来触发压缩。"""
    for i in range(rounds):
        m.add_user(f"第{i}个问题，内容写得长一点")
        m.add_assistant(f"第{i}个回答，也写得长一点")


# ---------- estimate_tokens ----------

def test_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_chinese_close_to_char_count():
    # 「你好世界」4 个汉字，按校准后的 0.7 系数约 2.8 token
    assert 2 <= estimate_tokens("你好世界") <= 5


def test_english_cheaper_than_chinese():
    # 同样长度的英文比中文省 token
    zh = estimate_tokens("你好你好你好你好")
    en = estimate_tokens("aaaaaaaaaaaaaaaa")
    assert en < zh


def test_mixed_text():
    text = "用 FastAPI 写一个接口"
    value = estimate_tokens(text)
    # 混合文本：比「全按英文算」高，比「全按汉字算」低
    assert len(text) * 0.3 < value < len(text)


# ---------- count_tokens ----------

def test_count_includes_overhead():
    one = count_tokens([{"role": "user", "content": ""}])
    assert one >= 4  # 空内容也要算 role 和分隔符的开销


def test_count_grows_with_messages():
    short = [{"role": "user", "content": "你好"}]
    long = short + [{"role": "assistant", "content": "你好，有什么可以帮你"}]
    assert count_tokens(long) > count_tokens(short)


# ---------- ContextManager ----------

def test_starts_with_system_only():
    m = make_manager()
    assert len(m.messages) == 1
    assert m.messages[0]["role"] == "system"


def test_add_user_and_assistant():
    m = make_manager()
    m.add_user("你好")
    m.add_assistant("你好呀")
    assert [x["role"] for x in m.messages] == ["system", "user", "assistant"]


def test_empty_answer_is_not_added():
    m = make_manager()
    m.add_user("你好")
    m.add_assistant("")
    assert [x["role"] for x in m.messages] == ["system", "user"]


def test_short_chat_is_within_budget():
    m = make_manager(max_tokens=10_000)
    m.add_user("你好")
    m.add_assistant("你好呀")
    assert not m.over_budget()
    assert m.trim() == 0


def test_trim_keeps_system_and_recent_turns():
    m = make_manager(max_tokens=100, keep_pairs=1)
    for i in range(10):
        m.add_user(f"第{i}个问题，内容写得长一点让它占 token")
        m.add_assistant(f"第{i}个回答，也写得长一点")

    dropped = m.trim()

    assert dropped > 0
    assert m.count() <= m.max_tokens
    assert m.messages[0]["role"] == "system"          # system 永远在
    assert m.messages[-1]["content"].startswith("第9个")  # 最近一轮原文保留
    assert m.messages[1]["role"] == "user"            # 裁完不能以 assistant 开头


def test_trim_keeps_min_pairs_even_if_still_over_budget():
    # 预算小到不可能满足，也必须保留最近 keep_pairs 轮
    m = make_manager(max_tokens=1, keep_pairs=2)
    for i in range(5):
        m.add_user(f"问题{i}")
        m.add_assistant(f"回答{i}")

    m.trim()

    assert len(m.messages) >= 1 + 2 * 2
    assert m.stats()["pairs"] >= 2


def test_reset_clears_history():
    m = make_manager()
    m.add_user("你好")
    m.add_assistant("你好呀")
    m.summary = "用户叫雨霖铃"
    m.reset()
    assert [x["role"] for x in m.messages] == ["system"]
    assert m.summary == ""
    assert m.stats()["tokens"] < m.stats()["budget"]


def test_stats_shape():
    m = make_manager(max_tokens=500)
    m.add_user("你好")
    stats = m.stats()
    assert set(stats) == {
        "tokens", "budget", "messages", "pairs", "summary_chars", "error"
    }
    assert stats["budget"] == 500


# ---------- build_messages / compress ----------

def test_build_messages_without_summary_is_plain_history():
    m = make_manager()
    m.add_user("你好")
    assert [x["role"] for x in m.build_messages()] == ["system", "user"]


def test_build_messages_puts_summary_after_system():
    m = make_manager()
    m.add_user("你好")
    m.summary = "用户叫雨霖铃。"
    built = m.build_messages()
    assert [x["role"] for x in built] == ["system", "system", "user"]
    assert "早期对话摘要" in built[1]["content"]
    assert "雨霖铃" in built[1]["content"]


def test_compress_does_nothing_when_within_budget():
    m = make_manager(max_tokens=10_000)
    m.add_user("你好")
    assert m.compress(FakeClient()) == 0


def test_compress_turns_old_messages_into_summary():
    m = make_manager(max_tokens=60, keep_pairs=1)
    long_chat(m, 10)
    client = FakeClient()

    compressed = m.compress(client)

    assert compressed > 0
    assert m.summary == client.reply              # 摘要来自模型
    assert m.messages[-1]["content"].startswith("第9个")   # 最近一轮保留原文
    assert m.messages[1]["role"] == "user"        # 摘要段后从 user 开始


def test_compress_carries_old_summary_forward():
    m = make_manager(max_tokens=60, keep_pairs=1)
    m.summary = "用户叫雨霖铃。"
    long_chat(m, 5)
    client = FakeClient()

    m.compress(client)

    prompt = client.last_messages[0]["content"]
    assert "用户叫雨霖铃" in prompt                # 旧摘要必须一起喂进去


def test_compress_keeps_summary_in_count():
    m = make_manager(max_tokens=60, keep_pairs=1)
    long_chat(m, 10)
    m.compress(FakeClient("这是一段很长很长的摘要" * 10))
    assert m.stats()["summary_chars"] > 0
    assert m.count() == count_tokens(m.build_messages())  # 摘要算进预算
