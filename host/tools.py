from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any

from cursor_sdk import CustomTool, CustomToolContext

from .config import CONFIG
from .session import RunSession
from .logging_setup import get_logger

log = get_logger(__name__)

READ_ONLY_HINTS = ("read", "list", "search", "grep", "glob", "view", "get", "fetch", "lookup", "ls", "cat", "find", "status", "diagnostics")
WRITE_HINTS = ("write", "edit", "create", "delete", "remove", "apply", "patch", "run", "exec", "terminal", "shell", "command", "mkdir", "move", "rename", "install")


def tools_signature(openai_tools: list[dict] | None) -> str:
    if not openai_tools:
        return "none"
    canon = json.dumps(openai_tools, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def classify_mode(openai_tools: list[dict] | None) -> str:
    if not openai_tools:
        return "ask"
    for tool in openai_tools:
        fn = tool.get("function", tool)
        name = (fn.get("name") or "").lower()
        if any(h in name for h in WRITE_HINTS):
            return "agent"
        if not any(h in name for h in READ_ONLY_HINTS):
            return "agent"
    return "plan"


def build_custom_tools(openai_tools: list[dict] | None, session_ref: dict[str, RunSession], registry: Any) -> dict[str, CustomTool]:
    if not openai_tools:
        return {}
    out: dict[str, CustomTool] = {}
    for tool in openai_tools:
        fn = tool.get("function", tool)
        name = fn.get("name")
        if not name:
            continue
        out[name] = CustomTool(
            description=fn.get("description") or "",
            input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            execute=_make_execute(name, session_ref, registry),
        )
    return out


def _make_execute(name: str, session_ref: dict[str, RunSession], registry: Any):
    async def execute(args: dict, context: CustomToolContext) -> Any:
        session = session_ref["session"]
        call_id = context.tool_call_id or f"call_{uuid.uuid4().hex}"
        future = session.register_pending(call_id, name)
        registry.register_pending_call(session, call_id)
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }
        await session.emit({"type": "tool_call", "call": tool_call})
        try:
            result = await asyncio.wait_for(future, timeout=CONFIG.suspend_ttl_s)
        except asyncio.TimeoutError:
            registry.unregister_pending_call(call_id)
            raise
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    return execute
