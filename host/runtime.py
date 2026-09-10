from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from cursor_sdk import AsyncClient

from .config import CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)


class CursorRuntime:
    def __init__(self) -> None:
        self._client: AsyncClient | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._restart_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        self._closing = False
        self._before_restart: Callable[[], Awaitable[None]] | None = None

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("Cursor runtime not started")
        return self._client

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def set_before_restart(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._before_restart = callback

    async def start(self) -> None:
        CONFIG.sandbox_root.mkdir(parents=True, exist_ok=True)
        await self._launch()
        self._supervisor_task = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        while not self._closing:
            await asyncio.sleep(20)
            if self._closing:
                return
            if not await self.ping():
                log.warning("bridge unresponsive, restarting")
                try:
                    await self.restart()
                except Exception:
                    log.error("bridge restart failed", exc_info=True)

    async def _launch(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            log.info("launching cursor sdk bridge")
            self._client = await AsyncClient.launch_bridge(workspace=str(CONFIG.sandbox_root))
            self._ready.set()
            log.info("cursor sdk bridge ready")

    async def wait_ready(self, timeout: float = 30.0) -> None:
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)

    async def restart(self) -> None:
        if self._before_restart is not None:
            await self._before_restart()
        async with self._lock:
            old = self._client
            self._client = None
            self._ready.clear()
        if old is not None:
            try:
                await old.aclose()
            except Exception:
                log.warning("error closing previous bridge client", exc_info=True)
        await self._launch()

    async def stop(self) -> None:
        self._closing = True
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
        client = self._client
        self._client = None
        self._ready.clear()
        if client is not None:
            await client.aclose()

    async def ping(self) -> bool:
        client = self._client
        if client is None:
            return False
        try:
            await client.ping()
            return True
        except Exception:
            return False


RUNTIME = CursorRuntime()
