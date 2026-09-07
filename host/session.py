from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from cursor_sdk import AgentOptions, CursorAgentError, CustomTool, LocalAgentOptions, RateLimitError

from .config import CONFIG
from .logging_setup import get_logger
from .runtime import CursorRuntime

log = get_logger(__name__)


def transcript_hash(messages: list[dict]) -> str:
    canon = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def is_new_conversation(messages: list[dict]) -> bool:
    return not any(m.get("role") in ("assistant", "tool") for m in messages)


def trailing_tool_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in reversed(messages):
        if m.get("role") == "tool":
            out.append(m)
        else:
            break
    return list(reversed(out))


@dataclass
class PendingCall:
    future: asyncio.Future
    name: str
    created_at: float = field(default_factory=time.monotonic)


class OutputSink:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def put(self, item: Any) -> None:
        if not self.closed:
            await self.queue.put(item)

    async def get(self) -> Any:
        return await self.queue.get()

    def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(None)


class RunSession:
    def __init__(self, conversation_id: str, agent: Any, cwd: str, tools_signature: str) -> None:
        self.conversation_id = conversation_id
        self.agent = agent
        self.cwd = cwd
        self.tools_signature = tools_signature
        self.pending: dict[str, PendingCall] = {}
        self.sink: OutputSink | None = None
        self.turn_lock = asyncio.Lock()
        self.created_at = time.monotonic()
        self.last_active = time.monotonic()
        self.run_active = False
        self.run_task: asyncio.Task | None = None
        self.custom_tools: dict[str, CustomTool] = {}

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def register_pending(self, call_id: str, name: str) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self.pending[call_id] = PendingCall(future=fut, name=name)
        return fut

    def resolve_pending(self, call_id: str, result: Any) -> bool:
        pc = self.pending.pop(call_id, None)
        if pc is None or pc.future.done():
            return False
        pc.future.set_result(result)
        return True

    def cancel_all_pending(self, reason: str) -> None:
        for call_id, pc in list(self.pending.items()):
            if not pc.future.done():
                pc.future.set_exception(TimeoutError(reason))
        self.pending.clear()

    async def emit(self, item: Any) -> None:
        sink = self.sink
        if sink is not None:
            await sink.put(item)


class SessionRegistry:
    def __init__(self, runtime: CursorRuntime) -> None:
        self._runtime = runtime
        self._by_conv: dict[str, RunSession] = {}
        self._by_hash: dict[str, str] = {}
        self._by_pending_call: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._reap_task: asyncio.Task | None = None

    def get(self, conversation_id: str) -> RunSession | None:
        return self._by_conv.get(conversation_id)

    def find_by_tool_call(self, call_id: str) -> RunSession | None:
        conv = self._by_pending_call.get(call_id)
        return self._by_conv.get(conv) if conv else None

    def find_by_prior_transcript(self, messages_prefix: list[dict]) -> RunSession | None:
        h = transcript_hash(messages_prefix)
        conv = self._by_hash.get(h)
        return self._by_conv.get(conv) if conv else None

    def index_completed_turn(self, session: RunSession, full_transcript: list[dict]) -> None:
        self._by_hash[transcript_hash(full_transcript)] = session.conversation_id

    def register_pending_call(self, session: RunSession, call_id: str) -> None:
        self._by_pending_call[call_id] = session.conversation_id

    def unregister_pending_call(self, call_id: str) -> None:
        self._by_pending_call.pop(call_id, None)

    async def create(self, *, model: str, sdk_mode: str, custom_tools: dict[str, CustomTool], conversation_id: str | None = None, tools_signature: str = "") -> RunSession:
        async with self._lock:
            if len(self._by_conv) >= CONFIG.agent_max_count:
                await self._evict_oldest_locked()
            conv_id = conversation_id or f"conv-{uuid.uuid4().hex}"
            # NOTE: the Cursor bridge silently stalls (~150s) then fails when the agent's
            # local cwd folder basename is longer than ~20 chars for a brand-new folder.
            # Keep the on-disk folder name short regardless of the (longer) conversation id.
            while True:
                folder_name = f"s{uuid.uuid4().hex[:12]}"
                cwd = CONFIG.sandbox_root / folder_name
                if not cwd.exists():
                    break
            cwd.mkdir(parents=True, exist_ok=True)
            # `tools` is an allowlist of built-in Cursor tools. Suppressing built-ins with
            # tools=[] also disables the "mcp" gateway that custom_tools are served through,
            # so keep "mcp" allowed whenever there are custom tools to expose.
            options = AgentOptions(
                model=model,
                api_key=CONFIG.cursor_api_key or None,
                mode=sdk_mode,
                tools=["mcp"] if custom_tools else [],
                local=LocalAgentOptions(cwd=str(cwd), setting_sources=[], custom_tools=custom_tools or None),
            )
            agent = await self._create_agent_with_retry(options)
            session = RunSession(conv_id, agent, str(cwd), tools_signature)
            session.custom_tools = custom_tools
            self._by_conv[conv_id] = session
            log.info("session created", extra={"conversation_id": conv_id})
            return session

    async def _create_agent_with_retry(self, options: AgentOptions, attempts: int = 4) -> Any:
        # Cursor's cloud model-list check behind create_agent has its own low-frequency
        # rate limit (independent of ours); back off and retry rather than surfacing
        # a hard failure to the client for what is normally a transient burst.
        delay = 1.0
        for attempt in range(attempts):
            try:
                return await self._runtime.client.create_agent(options)
            except CursorAgentError as e:
                if not (e.is_retryable or isinstance(e, RateLimitError)) or attempt == attempts - 1:
                    raise
                wait_s = float(e.retry_after) if e.retry_after else delay
                await asyncio.sleep(wait_s)
                delay = min(delay * 2, 10.0)

    async def close(self, session: RunSession) -> None:
        session.cancel_all_pending("session closed")
        self._by_conv.pop(session.conversation_id, None)
        for call_id in [k for k, v in self._by_pending_call.items() if v == session.conversation_id]:
            self._by_pending_call.pop(call_id, None)
        for h in [k for k, v in self._by_hash.items() if v == session.conversation_id]:
            self._by_hash.pop(h, None)
        try:
            await session.agent.close()
        except Exception:
            log.warning("error closing agent", exc_info=True)

    async def _evict_oldest_locked(self) -> None:
        if not self._by_conv:
            return
        oldest = min(self._by_conv.values(), key=lambda s: s.last_active)
        await self.close(oldest)

    async def reap_idle(self) -> None:
        cutoff = time.monotonic() - CONFIG.agent_idle_ttl_s
        idle = [s for s in self._by_conv.values() if s.last_active < cutoff and not s.run_active]
        for s in idle:
            log.info("reaping idle session", extra={"conversation_id": s.conversation_id})
            await self.close(s)

    async def start_reaper(self) -> None:
        async def loop() -> None:
            while True:
                await asyncio.sleep(60)
                try:
                    await self.reap_idle()
                except Exception:
                    log.warning("reaper iteration failed", exc_info=True)

        self._reap_task = asyncio.create_task(loop())

    async def stop_reaper(self) -> None:
        if self._reap_task is not None:
            self._reap_task.cancel()

    def count(self) -> int:
        return len(self._by_conv)

    def all_ids(self) -> list[str]:
        return list(self._by_conv.keys())
