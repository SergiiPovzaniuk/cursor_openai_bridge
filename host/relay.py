from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse

from .chat import ChatContext, new_completion_id, run_turn, to_stream_chunks
from .config import CONFIG
from .errors import OpenAIError
from .logging_setup import get_logger
from .metrics import METRICS
from .security import check_allowlist, check_rate_limit, check_relay_token

log = get_logger(__name__)

POLL_TIMEOUT_S = 25
BATCH_WINDOW_S = 0.04
REPLAY_BUFFER_SIZE = 500
OUTBOUND_QUEUE_SIZE = 2048
CONNECTION_TTL_S = 120


class StreamState:
    def __init__(self) -> None:
        self.seq = 0
        self.buffer: deque[dict] = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.task: asyncio.Task | None = None


class RelayConnection:
    def __init__(self, ctx: ChatContext) -> None:
        self.ctx = ctx
        self.streams: dict[str, StreamState] = {}
        self.outq: asyncio.Queue[dict] = asyncio.Queue(maxsize=OUTBOUND_QUEUE_SIZE)
        self.last_active = time.monotonic()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    async def close(self) -> None:
        for state in self.streams.values():
            if state.task and not state.task.done():
                state.task.cancel()
        await asyncio.gather(
            *(state.task for state in self.streams.values() if state.task),
            return_exceptions=True,
        )
        self.streams.clear()

    async def _send(self, frame: dict) -> None:
        await self.outq.put(frame)

    async def _emit(self, sid: str, frame: dict) -> None:
        state = self.streams.setdefault(sid, StreamState())
        state.seq += 1
        frame = {**frame, "sid": sid, "q": state.seq}
        state.buffer.append(frame)
        await self._send(frame)

    async def handle_send(self, sid: str, payload: dict, headers: dict) -> None:
        state = self.streams.setdefault(sid, StreamState())
        if state.task and not state.task.done():
            state.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.task
        METRICS.active_streams += 1
        METRICS.requests_total += 1
        started = time.monotonic()
        first_token = False

        async def run() -> None:
            nonlocal first_token
            try:
                await self._emit(sid, {"k": "open"})
                cid = new_completion_id()
                created = int(time.time())
                model = payload.get("model") or CONFIG.default_model
                events = run_turn(payload, headers, self.ctx)
                async for chunk in to_stream_chunks(events, cid, created, model, include_usage=True):
                    if not first_token:
                        first_token = True
                        METRICS.record_ttft(time.monotonic() - started)
                    await self._emit(sid, {"k": "d", "data": chunk})
                await self._emit(sid, {"k": "end", "reason": "stop"})
            except asyncio.CancelledError:
                await self._emit(sid, {"k": "end", "reason": "cancel"})
            except OpenAIError as e:
                METRICS.errors_total += 1
                await self._emit(sid, {"k": "end", "reason": "error", "err": {**e.body()["error"], "status": e.status}})
            except Exception as e:  # noqa: BLE001
                METRICS.errors_total += 1
                log.error("relay stream failed", exc_info=True)
                await self._emit(sid, {"k": "end", "reason": "error", "err": {"message": str(e)}})
            finally:
                METRICS.active_streams -= 1

        state.task = asyncio.create_task(run())

    async def handle_cancel(self, sid: str) -> None:
        state = self.streams.get(sid)
        if state and state.task and not state.task.done():
            state.task.cancel()

    async def handle_models(self, sid: str) -> None:
        try:
            data = await self.ctx.models.list()
            await self._send({"k": "models", "sid": sid, "data": data})
        except Exception as e:  # noqa: BLE001
            log.error("relay models failed", exc_info=True)
            await self._send({"k": "models", "sid": sid, "err": {"message": str(e)}})

    async def handle_resend(self, sid: str, from_seq: int) -> None:
        state = self.streams.get(sid)
        if not state:
            return
        for frame in list(state.buffer):
            if frame["q"] >= from_seq:
                await self._send(frame)

    async def handle_frame(self, frame: dict) -> None:
        kind = frame.get("k")
        sid = frame.get("sid")
        if kind == "send" and sid:
            headers = {k.lower(): v for k, v in (frame.get("headers") or {}).items()}
            asyncio.create_task(self.handle_send(sid, frame.get("payload") or {}, headers))
        elif kind == "cancel" and sid:
            await self.handle_cancel(sid)
        elif kind == "resend" and sid:
            await self.handle_resend(sid, int(frame.get("fromSeq", 0)))
        elif kind == "models":
            asyncio.create_task(self.handle_models(sid))


# Keyed by "token:conn" so a stale browser tab (e.g. left over from a crashed/killed
# relay process) can never steal frames meant for the current live tab -- each page
# load gets a fresh random conn id from Java, isolating its queue entirely.
_connections: dict[str, RelayConnection] = {}


def _conn_key(request: Request) -> str:
    token = request.headers.get("x-relay-token") or request.query_params.get("token") or ""
    conn = request.query_params.get("conn") or "default"
    return f"{token}:{conn}"


def _conn_for(ctx: ChatContext, key: str) -> RelayConnection:
    conn = _connections.get(key)
    if conn is None:
        conn = RelayConnection(ctx)
        _connections[key] = conn
        conn.outq.put_nowait({"k": "hello"})
    return conn


async def _prune_connections(current_key: str) -> None:
    cutoff = time.monotonic() - CONNECTION_TTL_S
    stale = [(key, conn) for key, conn in _connections.items() if key != current_key and conn.last_active < cutoff]
    for key, conn in stale:
        _connections.pop(key, None)
        await conn.close()


async def relay_send(request: Request, ctx: ChatContext) -> JSONResponse:
    token = request.headers.get("x-relay-token") or request.query_params.get("token")
    if not check_relay_token(token):
        return JSONResponse(status_code=401, content={"error": "bad token"})
    check_allowlist(request)
    check_rate_limit(request)
    try:
        frame = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "bad json"})
    key = _conn_key(request)
    await _prune_connections(key)
    conn = _conn_for(ctx, key)
    conn.touch()
    await conn.handle_frame(frame)
    return JSONResponse({"ok": True})


async def relay_poll(request: Request, ctx: ChatContext) -> JSONResponse:
    token = request.headers.get("x-relay-token") or request.query_params.get("token")
    if not check_relay_token(token):
        return JSONResponse(status_code=401, content={"error": "bad token"})
    check_allowlist(request)
    check_rate_limit(request)
    key = _conn_key(request)
    await _prune_connections(key)
    conn = _conn_for(ctx, key)
    conn.touch()
    frames: list[dict] = []
    try:
        frames.append(await asyncio.wait_for(conn.outq.get(), timeout=POLL_TIMEOUT_S))
    except asyncio.TimeoutError:
        return JSONResponse({"frames": frames})
    # Briefly keep draining so a burst of fast-arriving stream chunks rides one
    # HTTP round trip instead of one round trip per chunk (this is what made
    # generation feel slow over a real network vs. loopback).
    deadline = time.monotonic() + BATCH_WINDOW_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            frames.append(await asyncio.wait_for(conn.outq.get(), timeout=remaining))
        except asyncio.TimeoutError:
            break
    return JSONResponse({"frames": frames})
