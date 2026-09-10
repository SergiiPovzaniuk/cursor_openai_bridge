from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _list(name: str) -> list[str]:
    v = os.environ.get(name, "")
    return [p.strip() for p in v.split(",") if p.strip()]


@dataclass(frozen=True)
class Config:
    dev_mode: bool = field(default_factory=lambda: _bool("DEV_MODE", False))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("PORT", 8787))
    cursor_api_key: str = field(default_factory=lambda: os.environ.get("CURSOR_API_KEY", ""))
    bearer_token: str = field(default_factory=lambda: os.environ.get("BEARER_TOKEN", ""))
    relay_token: str = field(default_factory=lambda: os.environ.get("RELAY_TOKEN", ""))
    bridge_mode: str = field(default_factory=lambda: os.environ.get("BRIDGE_MODE", "suspend"))
    suspend_ttl_s: float = field(default_factory=lambda: float(_int("SUSPEND_TTL_S", 900)))
    agent_idle_ttl_s: float = field(default_factory=lambda: float(_int("AGENT_IDLE_TTL_S", 1800)))
    agent_max_count: int = field(default_factory=lambda: _int("AGENT_MAX_COUNT", 64))
    sandbox_root: Path = field(default_factory=lambda: Path(os.environ.get("SANDBOX_ROOT", "./sandboxes")).resolve())
    models_cache_ttl_s: float = field(default_factory=lambda: float(_int("MODELS_CACHE_TTL_S", 300)))
    context_limits_path: Path = field(
        default_factory=lambda: Path(os.environ.get("CONTEXT_LIMITS_PATH", "./context_limits.json")).resolve()
    )
    tls_cert_path: str = field(default_factory=lambda: os.environ.get("TLS_CERT_PATH", ""))
    tls_key_path: str = field(default_factory=lambda: os.environ.get("TLS_KEY_PATH", ""))
    allowlist_ips: tuple[str, ...] = field(default_factory=lambda: tuple(_list("ALLOWLIST_IPS")))
    rate_limit_per_minute: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MINUTE", 120))
    max_body_bytes: int = field(default_factory=lambda: _int("MAX_BODY_BYTES", 4 * 1024 * 1024))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: _bool("LOG_JSON", True))
    default_model: str = field(default_factory=lambda: os.environ.get("DEFAULT_MODEL", "composer-2.5"))
    replay_max_chars: int = field(default_factory=lambda: _int("REPLAY_MAX_CHARS", 32000))

    def validate(self) -> None:
        if self.bridge_mode != "suspend":
            raise RuntimeError("BRIDGE_MODE must be suspend")
        if self.dev_mode:
            return
        missing = [name for name, value in (
            ("CURSOR_API_KEY", self.cursor_api_key),
            ("BEARER_TOKEN", self.bearer_token),
            ("RELAY_TOKEN", self.relay_token),
        ) if not value]
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")

    def load_context_limits(self) -> dict[str, int]:
        try:
            return json.loads(self.context_limits_path.read_text())
        except Exception:
            return {"default": 200000}


CONFIG = Config()
