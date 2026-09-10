import pytest

from host.session import RunSession, SessionRegistry, is_new_conversation, trailing_tool_messages, transcript_hash


def test_is_new_conversation_true_for_single_user_message():
    assert is_new_conversation([{"role": "user", "content": "hi"}])


def test_is_new_conversation_false_with_assistant():
    assert not is_new_conversation([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])


def test_is_new_conversation_false_with_tool_message():
    assert not is_new_conversation([{"role": "tool", "content": "result", "tool_call_id": "1"}])


def test_trailing_tool_messages():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "1"},
        {"role": "tool", "content": "r2", "tool_call_id": "2"},
    ]
    out = trailing_tool_messages(msgs)
    assert [m["tool_call_id"] for m in out] == ["1", "2"]


def test_trailing_tool_messages_none_when_last_is_user():
    msgs = [{"role": "tool", "content": "r", "tool_call_id": "1"}, {"role": "user", "content": "next"}]
    assert trailing_tool_messages(msgs) == []


def test_transcript_hash_stable_and_order_sensitive():
    a = [{"role": "user", "content": "hi"}]
    b = [{"role": "user", "content": "hi"}]
    c = [{"role": "user", "content": "bye"}]
    assert transcript_hash(a) == transcript_hash(b)
    assert transcript_hash(a) != transcript_hash(c)


@pytest.mark.asyncio
async def test_run_session_pending_lifecycle():
    session = RunSession("conv-1", agent=None, cwd="/tmp/x", tools_signature="none", remote_context="remote")
    assert session.remote_context == "remote"
    fut = session.register_pending("call_1", "read_file")
    assert not fut.done()
    assert session.resolve_pending("call_1", "content")
    assert fut.result() == "content"
    assert not session.resolve_pending("call_1", "again")


@pytest.mark.asyncio
async def test_run_session_cancel_all_pending():
    session = RunSession("conv-1", agent=None, cwd="/tmp/x", tools_signature="none")
    fut = session.register_pending("call_1", "read_file")
    session.cancel_all_pending("closed")
    assert fut.done()
    assert isinstance(fut.exception(), TimeoutError)


def test_completed_tool_call_lookup():
    registry = SessionRegistry(None)
    session = RunSession("conv-1", None, "/tmp/x", "sig")
    registry._by_conv[session.conversation_id] = session
    registry.mark_completed_call(session, "call-1")
    assert registry.find_by_completed_tool_call("call-1") is session
