import pytest

from host.context import budget_messages_for_replay, estimate_tokens, total_chars, truncate_middle, check_overflow
from host.errors import OpenAIError


def test_truncate_middle_noop_when_short():
    assert truncate_middle("short", 100) == "short"


def test_truncate_middle_keeps_head_and_tail():
    text = "x" * 1000
    out = truncate_middle(text, 100)
    assert len(out) < len(text)
    assert out.startswith("x" * 10)
    assert "[truncated]" in out


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_budget_messages_for_replay_keeps_system_and_recent():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old " * 5000},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest question"},
    ]
    out = budget_messages_for_replay(messages, max_chars=200)
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "latest question"


def test_total_chars():
    assert total_chars([{"role": "user", "content": "abcd"}]) == 4


class FakeModels:
    def context_length(self, model_id):
        return 100


def test_check_overflow_raises_when_too_large():
    messages = [{"role": "user", "content": "x" * 10000}]
    with pytest.raises(OpenAIError) as exc:
        check_overflow(messages, "any-model", FakeModels())
    assert exc.value.code == "context_length_exceeded"


def test_check_overflow_ok_when_small():
    messages = [{"role": "user", "content": "hi"}]
    check_overflow(messages, "any-model", FakeModels())
