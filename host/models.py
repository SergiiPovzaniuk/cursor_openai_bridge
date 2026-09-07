from __future__ import annotations

import asyncio
import time

from cursor_sdk import SDKModel

from .config import CONFIG
from .errors import OpenAIError
from .logging_setup import get_logger
from .runtime import CursorRuntime

log = get_logger(__name__)

MODE_SUFFIXES = ("ask", "plan", "agent")


def _to_openai_model(model: SDKModel, created: int, context_length: int) -> dict:
    return {
        "id": model.id,
        "object": "model",
        "created": created,
        "owned_by": "cursor",
        "display_name": model.display_name,
        "context_length": context_length,
    }


def _expand_with_modes(base: list[dict], created: int) -> list[dict]:
    out = list(base)
    seen = {m["id"] for m in base}
    for m in base:
        for suffix in MODE_SUFFIXES:
            mid = f"{m['id']}:{suffix}"
            if mid not in seen:
                seen.add(mid)
                out.append({**m, "id": mid})
    return out


class ModelsCache:
    def __init__(self, runtime: CursorRuntime) -> None:
        self._runtime = runtime
        self._models: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._raw_by_base_id: dict[str, SDKModel] = {}
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self._context_limits = CONFIG.load_context_limits()

    def base_model_id(self, model_id: str) -> tuple[str, str | None]:
        if ":" in model_id:
            base, _, suffix = model_id.rpartition(":")
            if suffix in MODE_SUFFIXES:
                return base, suffix
        return model_id, None

    def context_length(self, model_id: str) -> int:
        base, _ = self.base_model_id(model_id)
        return self._context_limits.get(base, self._context_limits.get("default", 200000))

    async def refresh(self) -> None:
        async with self._lock:
            raw = await self._runtime.client.list_models(api_key=CONFIG.cursor_api_key or None)
            created = int(time.time())
            base = [_to_openai_model(m, created, self.context_length(m.id)) for m in raw]
            self._models = _expand_with_modes(base, created)
            self._by_id = {m["id"]: m for m in self._models}
            self._raw_by_base_id = {m.id: m for m in raw}
            self._expires_at = time.monotonic() + CONFIG.models_cache_ttl_s
            log.info("models refreshed", extra={"count": len(self._models)})

    async def ensure_fresh(self) -> None:
        if time.monotonic() >= self._expires_at or not self._models:
            try:
                await self.refresh()
            except Exception:
                log.warning("model refresh failed, serving stale/empty cache", exc_info=True)
                if not self._models:
                    self._models = _expand_with_modes(
                        [{"id": CONFIG.default_model, "object": "model", "created": int(time.time()), "owned_by": "cursor", "display_name": CONFIG.default_model}],
                        int(time.time()),
                    )
                    self._by_id = {m["id"]: m for m in self._models}

    async def list(self) -> list[dict]:
        await self.ensure_fresh()
        return self._models

    async def get(self, model_id: str) -> dict:
        await self.ensure_fresh()
        model = self._by_id.get(model_id)
        if model is None:
            base, _ = self.base_model_id(model_id)
            model = self._by_id.get(base)
        if model is None:
            raise OpenAIError(f"model '{model_id}' not found", status=404, type_="invalid_request_error", code="model_not_found")
        return model

    def known_parameters(self, base_model_id: str):
        model = self._raw_by_base_id.get(base_model_id)
        return model.parameters if model else ()
