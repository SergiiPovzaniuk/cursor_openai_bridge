# cursor-openai-bridge

OpenAI-compatible API on top of the Cursor SDK. Runs on the machine that has
`CURSOR_API_KEY`. A separate Java relay (`open_ai_api`, see its README) runs
on the remote PC next to VS Code Continue, drives a headless Chromium tab
that loads this host's `/static/robot.html`, and gets all model output by
that browser tab pushing WebSocket frames to Java — no polling, no DOM reads.

```
Continue (remote PC) -> Java relay :18080 -> Chromium -> wss /relay -> this host :8787 -> cursor-sdk -> Cursor cloud
```

## Run locally

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # set CURSOR_API_KEY, BEARER_TOKEN, RELAY_TOKEN
python run.py
```

Server listens on `http://<HOST>:<PORT>` (default `0.0.0.0:8787`).

## Endpoints

- `GET /healthz`, `GET /readyz`, `GET /metrics`
- `GET /v1/models`, `GET /v1/models/{id}`
- `POST /v1/chat/completions` (stream + non-stream, tool calls, modes via `model:ask|plan|agent` suffix or `X-Cursor-Mode` header)
- `POST /v1/cursor/sessions/reset`, `DELETE /v1/cursor/sessions/{id}`
- `WS /relay?token=...` — multiplexed relay for the Chromium robot page (`/static/robot.html`)

All `/v1/*` routes require `Authorization: Bearer <BEARER_TOKEN>`.

## Remote setup (VS Code Continue + Java relay + Chromium)

This host only needs to be reachable from the remote PC on `PORT` (default
`8787`) for `/relay` (WebSocket) and `/v1/*`. Everything else — installing
Chromium, running the Java relay, and the Continue `config.yaml` snippet —
is documented in [`open_ai_api/README.md`](../open_ai_api/README.md), since
those steps run on the *remote* machine, not here.

Quick summary: set the same `RELAY_TOKEN` in both `.env` files, give the
relay this host's `BRIDGE_URL=http://<this-machine-ip>:8787/`, and put TLS in
front of this port once it leaves localhost. `ALLOWLIST_IPS` and
`RATE_LIMIT_PER_MINUTE` in `.env` restrict who can reach `/v1/*` and `/relay`.

## Decommissioning the old stack

`open_ai_cursor_api` (Node/Hono ACP-over-stdio service) and the old
`data-testid`-based robot page are fully superseded by this host + the
rewritten `open_ai_api` Java relay. Once the new chain is verified, stop
`open_ai_cursor_api` and remove any process manager entry for it — nothing
here depends on it.

Autocomplete and embeddings still need a separate Continue provider; the
Cursor SDK is an agent SDK and cannot serve `/v1/completions` (FIM) or
`/v1/embeddings` (both return `501` here).

## Tests

```
pytest
python scripts/simulate_continue.py           # tool loop, concurrent chats, cancel, overflow (no browser)
python scripts/bench.py                        # TTFT/tokens-per-s, direct vs through the Java relay
python scripts/soak.py --duration 120          # sustained concurrent load
```
