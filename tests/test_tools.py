import asyncio

import pytest

from host.tools import build_custom_tools, classify_mode, tools_signature


def test_classify_mode_no_tools():
    assert classify_mode(None) == "ask"
    assert classify_mode([]) == "ask"


def test_classify_mode_read_only():
    tools = [{"type": "function", "function": {"name": "read_file"}}, {"type": "function", "function": {"name": "list_dir"}}]
    assert classify_mode(tools) == "plan"


def test_classify_mode_write():
    tools = [{"type": "function", "function": {"name": "read_file"}}, {"type": "function", "function": {"name": "edit_file"}}]
    assert classify_mode(tools) == "agent"


def test_classify_mode_unknown_defaults_agent():
    tools = [{"type": "function", "function": {"name": "do_something_weird"}}]
    assert classify_mode(tools) == "agent"


def test_tools_signature_stable():
    tools = [{"type": "function", "function": {"name": "a"}}]
    assert tools_signature(tools) == tools_signature(tools)
    assert tools_signature(None) == "none"
    assert tools_signature([]) == "none"


def test_tools_signature_changes_with_content():
    a = [{"type": "function", "function": {"name": "a"}}]
    b = [{"type": "function", "function": {"name": "b"}}]
    assert tools_signature(a) != tools_signature(b)


class FakeSession:
    def __init__(self):
        self.pending = {}
        self.emitted = []

    def register_pending(self, call_id, name):
        fut = asyncio.get_event_loop().create_future()
        self.pending[call_id] = fut
        return fut

    async def emit(self, item):
        self.emitted.append(item)


class FakeRegistry:
    def __init__(self):
        self.registered = []

    def register_pending_call(self, session, call_id):
        self.registered.append(call_id)

    def unregister_pending_call(self, call_id):
        pass


@pytest.mark.asyncio
async def test_build_custom_tools_suspends_and_resumes():
    openai_tools = [{"type": "function", "function": {"name": "read_file", "description": "reads", "parameters": {"type": "object"}}}]
    session = FakeSession()
    registry = FakeRegistry()
    session_ref = {"session": session}
    custom_tools = build_custom_tools(openai_tools, session_ref, registry)
    assert "read_file" in custom_tools

    class Ctx:
        tool_call_id = "call_1"

    task = asyncio.create_task(custom_tools["read_file"].execute({"path": "x.py"}, Ctx()))
    await asyncio.sleep(0.01)
    assert "call_1" in session.pending
    assert registry.registered == ["call_1"]
    assert session.emitted[0]["type"] == "tool_call"

    session.pending["call_1"].set_result("file contents")
    result = await task
    assert result == "file contents"


@pytest.mark.asyncio
async def test_build_custom_tools_timeout_unregisters(monkeypatch):
    import host.tools as tools_mod

    class FakeConfig:
        suspend_ttl_s = 0.02

    monkeypatch.setattr(tools_mod, "CONFIG", FakeConfig())
    openai_tools = [{"type": "function", "function": {"name": "slow_tool"}}]
    session = FakeSession()
    registry = FakeRegistry()
    session_ref = {"session": session}
    custom_tools = build_custom_tools(openai_tools, session_ref, registry)

    class Ctx:
        tool_call_id = "call_timeout"

    with pytest.raises(asyncio.TimeoutError):
        await custom_tools["slow_tool"].execute({}, Ctx())
