from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, Request

from .config import CONFIG
from .errors import rate_limited, unauthorized

_buckets: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_allowlist(request: Request) -> None:
    if not CONFIG.allowlist_ips:
        return
    ip = _client_ip(request)
    if ip not in CONFIG.allowlist_ips:
        raise unauthorized(f"IP {ip} not allowed")


def check_rate_limit(request: Request) -> None:
    if CONFIG.rate_limit_per_minute <= 0:
        return
    key = _client_ip(request)
    now = time.monotonic()
    bucket = _buckets[key]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= CONFIG.rate_limit_per_minute:
        raise rate_limited()
    bucket.append(now)


def require_bearer(authorization: str | None = Header(default=None)) -> None:
    if not CONFIG.bearer_token:
        return
    expected = f"Bearer {CONFIG.bearer_token}"
    if authorization != expected:
        raise unauthorized()


def check_request(request: Request) -> None:
    check_allowlist(request)
    check_rate_limit(request)
    auth = request.headers.get("authorization")
    require_bearer(auth)


def check_relay_token(token: str | None) -> bool:
    if not CONFIG.relay_token:
        return True
    return token == CONFIG.relay_token
