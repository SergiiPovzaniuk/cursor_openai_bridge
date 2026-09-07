from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator

from cursor_sdk import CursorAgentError, RateLimitError, SendOptions, TextDeltaUpdate, ThinkingDeltaUpdate

from .config import CONFIG
from .context import budget_messages_for_replay, check_overflow, usage_to_openai
from .errors import OpenAIError, conversation_busy
from .events import DoneEvent, ErrorEvent, Event, ReasoningEvent, TextEvent, ToolCallsEvent, UsageEvent
from .logging_setup import get_logger
from .models import ModelsCache
from .runtime import CursorRuntime
from .session import RunSession, SessionRegistry, is_new_conversation, trailing_tool_messages
from .tools import build_custom_tools, classify_mode, tools_signature

log = get_logger(__name__)

TOOL_CALL_COALESCE_WINDOW_S = 0.015


def to_sdk_mode(business_mode: str) -> str:
    return "agent" if business_mode == "agent" else "plan"


def _content_text(content: object) -> str:
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return str(content or "")


def _last_message_text(messages: list[dict]) -> str:
    return _content_text(messages[-1].get("content")) if messages else ""


def _system_prompt(messages: list[dict]) -> str:
    parts = [_content_text(m.get("content")) for m in messages if m.get("role") == "system"]
    return "\n".join(p for p in parts if p)


def _render_prior_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        text = _content_text(m.get("content"))
        if not text:
            continue
        lines.append(f"{role}: {text}")
    if not lines:
        return ""
    return "[Prior conversation, for context only]\n" + "\n".join(lines) + "\n[End of prior context]\n\n"


class ChatContext:
    def __init__(self, runtime: CursorRuntime, registry: SessionRegistry, models: ModelsCache):
        self.runtime = runtime
        self.registry = registry
        self.models = models


async def run_turn(payload: dict, headers: dict, ctx: ChatContext) -> AsyncIterator[Event]:
    messages = payload.get("messages") or []
    if not messages:
        raise OpenAIError("messages is required")

    model_id = payload.get("model") or CONFIG.default_model
    await ctx.models.ensure_fresh()
    base_model, mode_suffix = ctx.models.base_model_id(model_id)
    check_overflow(messages, model_id, ctx.models)

    tool_msgs = trailing_tool_messages(messages)
    resolved_session: RunSession | None = None
    if tool_msgs:
        for m in tool_msgs:
            call_id = m.get("tool_call_id")
            if not call_id:
                continue
            session = ctx.registry.find_by_tool_call(call_id)
            if session is None:
                continue
            resolved_session = session
            session.resolve_pending(call_id, _content_text(m.get("content")))
            ctx.registry.unregister_pending_call(call_id)

    if resolved_session is not None:
        async for ev in _attach_and_stream(resolved_session, ctx.registry, messages, already_running=True):
            yield ev
        return

    openai_tools = payload.get("tools")
    sdk_mode = mode_suffix or classify_mode(openai_tools)
    header_mode = headers.get("x-cursor-mode")
    if header_mode in ("ask", "plan", "agent"):
        sdk_mode = header_mode

    conv_header = headers.get("x-conversation-id")

    system = _system_prompt(messages)
    system_prefix = f"[System instructions]\n{system}\n[End of system instructions]\n\n" if system else ""

    # A conv_header identifies one Continue chat 1:1; if a session already exists under it,
    # this is a continuation (or a retry/resubmit of an earlier turn) -- never fall through to
    # creating a second session for the same header, which would orphan the live agent run
    # (still mid read/edit-tool-loop in the background) and restart the task from scratch,
    # causing the same read+edit to repeat forever.
    existing = ctx.registry.get(conv_header) if conv_header else None
    if existing is not None and existing.tools_signature == tools_signature(openai_tools):
        text = (system_prefix + _last_message_text(messages)) if is_new_conversation(messages) else _last_message_text(messages)
        async for ev in _send_and_stream(existing, ctx.registry, text, messages, sdk_mode):
            yield ev
        return

    if is_new_conversation(messages):
        session = await _create_session(ctx, base_model, sdk_mode, openai_tools, conv_header)
        text = system_prefix + _last_message_text(messages)
        async for ev in _send_and_stream(session, ctx.registry, text, messages, sdk_mode):
            yield ev
        return

    prior = messages[:-1]
    session = ctx.registry.find_by_prior_transcript(prior)
    if session is not None and session.tools_signature == tools_signature(openai_tools):
        text = _last_message_text(messages)
        async for ev in _send_and_stream(session, ctx.registry, text, messages, sdk_mode):
            yield ev
        return

    session = await _create_session(ctx, base_model, sdk_mode, openai_tools, conv_header)
    seed = _render_prior_transcript(budget_messages_for_replay(prior, CONFIG.replay_max_chars))
    text = system_prefix + seed + _last_message_text(messages)
    async for ev in _send_and_stream(session, ctx.registry, text, messages, sdk_mode):
        yield ev


async def _create_session(ctx: ChatContext, model: str, sdk_mode: str, openai_tools: list[dict] | None, conversation_id: str | None) -> RunSession:
    session_ref: dict[str, RunSession] = {}
    custom_tools = build_custom_tools(openai_tools, session_ref, ctx.registry)
    session = await ctx.registry.create(
        model=model,
        sdk_mode=to_sdk_mode(sdk_mode),
        custom_tools=custom_tools,
        conversation_id=conversation_id,
        tools_signature=tools_signature(openai_tools),
    )
    session_ref["session"] = session
    return session


async def _send_and_stream(session: RunSession, registry: SessionRegistry, text: str, full_messages: list[dict], sdk_mode: str) -> AsyncIterator[Event]:
    if session.turn_lock.locked():
        raise conversation_busy()
    async with session.turn_lock:
        session.touch()
        session.run_active = True
        from .session import OutputSink

        session.sink = OutputSink()

        async def on_delta(update) -> None:
            if isinstance(update, TextDeltaUpdate):
                await session.emit(TextEvent(update.text))
            elif isinstance(update, ThinkingDeltaUpdate):
                await session.emit(ReasoningEvent(update.text))

        try:
            run = await session.agent.send(text, SendOptions(mode=to_sdk_mode(sdk_mode), on_delta=on_delta))
        except CursorAgentError as e:
            session.run_active = False
            yield ErrorEvent(str(e), status=e.status or 500, code=e.code)
            yield DoneEvent(finish_reason="error", error=str(e))
            return

        async def waiter() -> None:
            try:
                result = await run.wait()
                finish = _map_finish_reason(result.status)
                if finish == "error":
                    log.error("run finished with error status", extra={"status": result.status, "detail": result.result})
                    await session.emit(DoneEvent(finish_reason="error", error=result.result or f"run ended with status {result.status}"))
                else:
                    await session.emit(UsageEvent(usage_to_openai(result.usage)))
                    await session.emit(DoneEvent(finish_reason=finish))
            except Exception as e:  # noqa: BLE001
                log.error("run failed", exc_info=True)
                await session.emit(DoneEvent(finish_reason="error", error=f"{type(e).__name__}: {e}"))
            finally:
                session.run_active = False

        task = asyncio.create_task(waiter())
        session.run_task = task
        try:
            async for ev in _consume(session, registry, full_messages):
                yield ev
        except BaseException:
            # Only abort the underlying run on a real disconnect/error. A normal
            # early return here means _consume hit a tool_calls pause (or a
            # terminal Done) and the run must keep going in the background so a
            # later resolve_pending() can unblock the suspended tool execute().
            if not task.done():
                task.cancel()
            raise


async def _attach_and_stream(session: RunSession, registry: SessionRegistry, full_messages: list[dict], already_running: bool) -> AsyncIterator[Event]:
    if session.turn_lock.locked():
        raise conversation_busy()
    async with session.turn_lock:
        session.touch()
        task = session.run_task
        try:
            async for ev in _consume(session, registry, full_messages):
                yield ev
        except BaseException:
            if task is not None and not task.done():
                task.cancel()
            raise


async def _consume(session: RunSession, registry: SessionRegistry, full_messages: list[dict]) -> AsyncIterator[Event]:
    sink = session.sink
    if sink is None:
        yield ErrorEvent("no active run for this conversation")
        return
    text_acc: list[str] = []
    pending_calls: list[dict] = []
    while True:
        item = await sink.get()
        if item is None:
            return
        if isinstance(item, TextEvent):
            text_acc.append(item.text)
            yield item
        elif isinstance(item, ReasoningEvent):
            yield item
        elif isinstance(item, dict) and item.get("type") == "tool_call":
            pending_calls.append(item["call"])
            deadline = time.monotonic() + TOOL_CALL_COALESCE_WINDOW_S
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(sink.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(nxt, dict) and nxt.get("type") == "tool_call":
                    pending_calls.append(nxt["call"])
                    continue
                if nxt is not None:
                    await sink.put(nxt)
                break
            yield ToolCallsEvent(list(pending_calls))
            yield DoneEvent(finish_reason="tool_calls")
            return
        elif isinstance(item, UsageEvent):
            yield item
        elif isinstance(item, DoneEvent):
            yield item
            if item.finish_reason in ("stop", "length"):
                assistant_msg = {"role": "assistant", "content": "".join(text_acc)}
                registry.index_completed_turn(session, full_messages + [assistant_msg])
            return
        elif isinstance(item, ErrorEvent):
            yield item


def _map_finish_reason(status: str) -> str:
    return {"finished": "stop", "cancelled": "cancel", "error": "error", "expired": "error"}.get(status, "stop")


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


async def to_stream_chunks(events: AsyncIterator[Event], completion_id: str, created: int, model: str, include_usage: bool) -> AsyncIterator[dict]:
    role_sent = False
    async for ev in events:
        if isinstance(ev, TextEvent):
            delta = {"content": ev.text}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield _chunk(completion_id, created, model, delta, None)
        elif isinstance(ev, ReasoningEvent):
            delta = {"reasoning_content": ev.text}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield _chunk(completion_id, created, model, delta, None)
        elif isinstance(ev, ToolCallsEvent):
            delta = {"tool_calls": [{"index": i, **c} for i, c in enumerate(ev.calls)]}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield _chunk(completion_id, created, model, delta, None)
        elif isinstance(ev, UsageEvent):
            if include_usage and ev.usage:
                yield _chunk(completion_id, created, model, {}, None, usage=ev.usage)
        elif isinstance(ev, DoneEvent):
            if ev.finish_reason == "error":
                raise OpenAIError(ev.error or "internal error", status=502, type_="server_error")
            yield _chunk(completion_id, created, model, {}, ev.finish_reason)
        elif isinstance(ev, ErrorEvent):
            raise OpenAIError(ev.message, status=ev.status, code=ev.code)


def _chunk(cid: str, created: int, model: str, delta: dict, finish_reason: str | None, usage: dict | None = None) -> dict:
    out = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        out["usage"] = usage
    return out


async def to_completion(events: AsyncIterator[Event], completion_id: str, created: int, model: str) -> dict:
    text_acc: list[str] = []
    calls: list[dict] = []
    usage: dict | None = None
    finish_reason = "stop"
    async for ev in events:
        if isinstance(ev, TextEvent):
            text_acc.append(ev.text)
        elif isinstance(ev, ToolCallsEvent):
            calls.extend(ev.calls)
        elif isinstance(ev, UsageEvent):
            usage = ev.usage
        elif isinstance(ev, DoneEvent):
            if ev.finish_reason == "error":
                raise OpenAIError(ev.error or "internal error", status=502, type_="server_error")
            finish_reason = ev.finish_reason
        elif isinstance(ev, ErrorEvent):
            raise OpenAIError(ev.message, status=ev.status, code=ev.code)

    message: dict = {"role": "assistant", "content": "".join(text_acc) or None}
    if calls:
        message["tool_calls"] = calls
        message["content"] = None
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
