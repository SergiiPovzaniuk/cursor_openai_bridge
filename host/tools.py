from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any
from urllib.parse import quote_plus

from cursor_sdk import CustomTool, CustomToolContext

from .config import CONFIG
from .events import DoneEvent, TextEvent
from .session import RunSession
from .logging_setup import get_logger

log = get_logger(__name__)

READ_ONLY_HINTS = ("read", "list", "search", "grep", "glob", "view", "get", "fetch", "lookup", "ls", "cat", "find", "status", "diagnostics")
WRITE_HINTS = ("write", "edit", "create", "delete", "remove", "apply", "patch", "run", "exec", "terminal", "shell", "command", "mkdir", "move", "rename", "install")
CONTINUE_ARG_ALIASES = {
    "filepath": ("path", "file", "filename"),
    "changes": ("instructions", "change", "patch"),
    "contents": ("content", "text"),
    "dirPath": ("dirpath", "directory", "path"),
    "pattern": ("glob",),
    "skillName": ("skill_name", "name"),
    "startLine": ("start_line", "start"),
    "endLine": ("end_line", "end"),
    "directory_path": ("dirPath", "dirpath", "directory", "path"),
    "old_string": ("oldString",),
    "new_string": ("newString",),
    "replace_all": ("replaceAll",),
    "waitForCompletion": ("wait_for_completion",),
}
CONTINUE_TOOLS = {
    "read_file", "read_file_range", "edit_existing_file", "single_find_and_replace", "multi_edit",
    "read_currently_open_file", "create_new_file", "run_terminal_command", "grep_search",
    "file_glob_search", "search_web", "view_diff", "ls", "create_rule_block", "request_rule",
    "fetch_url_content", "codebase", "read_skill", "view_repo_map", "view_subdirectory",
}
READ_ONLY_TOOLS = {
    "read_file", "read_file_range", "read_currently_open_file", "grep_search", "file_glob_search",
    "search_web", "view_diff", "ls", "request_rule", "fetch_url_content", "codebase", "read_skill",
    "view_repo_map", "view_subdirectory",
}


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


def build_custom_tools(openai_tools: list[dict] | None, session_ref: dict[str, RunSession], registry: Any, read_only: bool = False) -> dict[str, CustomTool]:
    async def reply(args: dict, context: CustomToolContext) -> str:
        session = session_ref["session"]
        session.bridge_reply_sent = True
        await session.emit(TextEvent(args["content"]))
        await session.emit(DoneEvent(finish_reason="stop"))
        asyncio.get_running_loop().call_later(0.1, lambda: asyncio.create_task(registry.close(session)))
        return "Response delivered. End the turn without additional text."

    out: dict[str, CustomTool] = {}
    names = {(tool.get("function", tool).get("name") or "") for tool in openai_tools or []}
    for tool in openai_tools or []:
        fn = tool.get("function", tool)
        name = fn.get("name")
        if not name:
            continue
        if read_only and name not in READ_ONLY_TOOLS:
            continue
        remote_name = "fetch_url_content" if name == "search_web" and "fetch_url_content" in names else name
        out[name] = CustomTool(
            description=fn.get("description") or "",
            input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            execute=_make_execute(name, fn.get("parameters") or {}, session_ref, registry, remote_name),
        )
    out["bridge_reply"] = CustomTool(
        description="Deliver the complete final response to the user after all other tools are complete.",
        input_schema={"type": "object", "required": ["content"], "properties": {"content": {"type": "string"}}},
        execute=reply,
    )
    return out


def normalize_tool_args(name: str, args: dict, schema: dict) -> dict:
    properties = schema.get("properties") or {}
    if name not in CONTINUE_TOOLS or not properties:
        return args
    normalized = dict(args)
    for field in properties:
        if field not in normalized:
            for alias in CONTINUE_ARG_ALIASES.get(field, ()):
                if alias in normalized:
                    normalized[field] = normalized.pop(alias)
                    break
    if name == "multi_edit":
        for edit in normalized.get("edits") or []:
            for field in ("old_string", "new_string", "replace_all"):
                for alias in CONTINUE_ARG_ALIASES.get(field, ()):
                    if field not in edit and alias in edit:
                        edit[field] = edit.pop(alias)
    normalized = {key: value for key, value in normalized.items() if key in properties}
    missing = [field for field in schema.get("required") or [] if field not in normalized]
    if missing:
        raise ValueError(f"{name} missing required arguments: {', '.join(missing)}")
    return normalized


def ensure_remote_tool_args(args: dict, session: RunSession) -> None:
    values = [(key, args.get(key)) for key in ("filepath", "dirPath", "directory_path", "path", "command")]
    if not any(isinstance(value, str) for _, value in values):
        return
    roots = {str(CONFIG.sandbox_root).replace("\\", "/").lower(), session.cwd.replace("\\", "/").lower()}
    for key, value in values:
        if not isinstance(value, str):
            continue
        candidate = value.replace("\\", "/").lower()
        if "/sandboxes/" in f"/{candidate.lstrip('/')}" or any(root and root in candidate for root in roots):
            raise ValueError(f"{key} references the host sandbox; use a client-workspace path")


def _make_execute(name: str, schema: dict, session_ref: dict[str, RunSession], registry: Any, remote_name: str):
    async def execute(args: dict, context: CustomToolContext) -> Any:
        args = normalize_tool_args(name, args, schema)
        session = session_ref["session"]
        ensure_remote_tool_args(args, session)
        call_id = context.tool_call_id or f"call_{uuid.uuid4().hex}"
        future = session.register_pending(call_id, name)
        registry.register_pending_call(session, call_id)
        remote_args = {"url": f"https://www.bing.com/search?q={quote_plus(args['query'])}"} if remote_name != name else args
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": remote_name, "arguments": json.dumps(remote_args, ensure_ascii=False)},
        }
        log.info("tool_call proposed", extra={"call_id": call_id, "name": name, "args": args})
        await session.emit({"type": "tool_call", "call": tool_call})
        try:
            result = await asyncio.wait_for(future, timeout=CONFIG.suspend_ttl_s)
        except asyncio.TimeoutError:
            log.error("tool_call timed out waiting for result", extra={"call_id": call_id, "name": name})
            session.pending.pop(call_id, None)
            registry.unregister_pending_call(call_id)
            run_task = getattr(session, "run_task", None)
            if run_task is not None and not run_task.done():
                run_task.cancel()
            raise
        except BaseException:
            session.pending.pop(call_id, None)
            registry.unregister_pending_call(call_id)
            raise
        log.info("tool_call resolved", extra={"call_id": call_id, "name": name, "result_len": len(result) if isinstance(result, str) else -1})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    return execute
