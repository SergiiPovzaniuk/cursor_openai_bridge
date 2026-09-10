import asyncio
import json

import pytest

from host.tools import build_custom_tools, classify_mode, ensure_remote_tool_args, normalize_tool_args, tools_signature


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


def test_continue_tool_names_and_schemas_are_preserved():
    schemas = {
        "file_glob_search": {"type": "object", "required": ["pattern"], "properties": {"pattern": {"type": "string"}}},
        "ls": {"type": "object", "properties": {"dirPath": {"type": "string"}, "recursive": {"type": "boolean"}}},
        "multi_edit": {"type": "object", "required": ["filepath", "edits"], "properties": {"filepath": {"type": "string"}, "edits": {"type": "array"}}},
        "mcp_dynamic_tool": {"type": "object", "required": ["value"], "properties": {"value": {"type": "string"}}},
    }
    tools = [{"type": "function", "function": {"name": name, "description": name, "parameters": schema}} for name, schema in schemas.items()]
    custom = build_custom_tools(tools, {}, None)
    assert set(custom) == set(schemas) | {"bridge_reply"}
    for name, schema in schemas.items():
        assert custom[name].input_schema == schema


def test_agent_mode_exposes_all_edit_tools():
    tools = [
        {"type": "function", "function": {"name": "edit_existing_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "single_find_and_replace", "parameters": {"type": "object"}}},
    ]
    custom = build_custom_tools(tools, {}, None)
    assert "edit_existing_file" in custom
    assert "single_find_and_replace" in custom


def test_plan_mode_excludes_mutating_tools():
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "edit_existing_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "run_terminal_command", "parameters": {"type": "object"}}},
    ]
    custom = build_custom_tools(tools, {}, None, read_only=True)
    assert set(custom) == {"read_file", "bridge_reply"}


def test_no_tool_request_gets_inert_sdk_capability():
    assert set(build_custom_tools(None, {}, None)) == {"bridge_reply"}


def test_continue_argument_aliases_are_normalized():
    read_schema = {"required": ["filepath"], "properties": {"filepath": {"type": "string"}}}
    edit_schema = {"required": ["filepath", "changes"], "properties": {"filepath": {"type": "string"}, "changes": {"type": "string"}}}
    assert normalize_tool_args("read_file", {"path": "a.py"}, read_schema) == {"filepath": "a.py"}
    assert normalize_tool_args("edit_existing_file", {"path": "a.py", "instructions": "replace"}, edit_schema) == {
        "filepath": "a.py",
        "changes": "replace",
    }


def test_dynamic_tool_arguments_are_untouched():
    args = {"path": "a.py", "extra": True}
    assert normalize_tool_args("mcp_dynamic_tool", args, {"properties": {"filepath": {"type": "string"}}}) is args


def test_missing_continue_arguments_are_rejected():
    schema = {"required": ["filepath"], "properties": {"filepath": {"type": "string"}}}
    with pytest.raises(ValueError, match="filepath"):
        normalize_tool_args("read_file", {}, schema)


def test_host_sandbox_paths_are_rejected():
    class Session:
        cwd = "C:\\host\\sandboxes\\s123"

    with pytest.raises(ValueError, match="host sandbox"):
        ensure_remote_tool_args({"filepath": "C:\\host\\sandboxes\\s123\\file.txt"}, Session())
    ensure_remote_tool_args({"filepath": "src/file.txt"}, Session())


class FakeSession:
    def __init__(self):
        self.pending = {}
        self.emitted = []
        self.cwd = "C:\\host\\sandboxes\\s123"
        self.run_task = None

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

    args = {"filepath": "x.py"}
    task = asyncio.create_task(custom_tools["read_file"].execute(args, Ctx()))
    await asyncio.sleep(0.01)
    assert "call_1" in session.pending
    assert registry.registered == ["call_1"]
    assert session.emitted[0]["type"] == "tool_call"
    assert session.emitted[0]["call"] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"filepath": "x.py"}'},
    }

    session.pending["call_1"].set_result("file contents")
    result = await task
    assert result == "file contents"


@pytest.mark.asyncio
async def test_search_web_uses_remote_fetch_tool():
    openai_tools = [
        {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "fetch_url_content", "parameters": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}}}},
    ]
    session = FakeSession()
    custom_tools = build_custom_tools(openai_tools, {"session": session}, FakeRegistry())

    class Ctx:
        tool_call_id = "call_search"

    task = asyncio.create_task(custom_tools["search_web"].execute({"query": "Spring Boot 3.3"}, Ctx()))
    await asyncio.sleep(0.01)
    call = session.emitted[0]["call"]["function"]
    assert call["name"] == "fetch_url_content"
    assert json.loads(call["arguments"]) == {"url": "https://www.bing.com/search?q=Spring+Boot+3.3"}
    session.pending["call_search"].set_result("results")
    assert await task == "results"


@pytest.mark.asyncio
async def test_edit_existing_file_uses_deterministic_remote_tools():
    openai_tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "required": ["filepath"], "properties": {"filepath": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "edit_existing_file", "parameters": {"type": "object", "required": ["filepath", "changes"], "properties": {"filepath": {"type": "string"}, "changes": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "single_find_and_replace", "parameters": {"type": "object", "required": ["filepath", "old_string", "new_string"], "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}}}},
    ]
    session = FakeSession()
    custom_tools = build_custom_tools(openai_tools, {"session": session}, FakeRegistry())

    class Ctx:
        tool_call_id = "call_edit"

    task = asyncio.create_task(custom_tools["edit_existing_file"].execute(
        {"filepath": "probe.txt", "changes": "new content\n"}, Ctx()
    ))
    await asyncio.sleep(0.01)
    read_call = session.emitted[0]["call"]
    assert read_call["function"]["name"] == "read_file"
    session.pending[read_call["id"]].set_result("old content\n")
    await asyncio.sleep(0.01)
    edit_call = session.emitted[1]["call"]
    assert edit_call["function"]["name"] == "single_find_and_replace"
    assert json.loads(edit_call["function"]["arguments"]) == {
        "filepath": "probe.txt",
        "old_string": "old content\n",
        "new_string": "new content\n",
        "replace_all": False,
    }
    session.pending[edit_call["id"]].set_result("edited")
    assert await task == "edited"


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
    assert session.pending == {}
