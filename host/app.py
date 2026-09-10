from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chat import ChatContext, new_completion_id, run_turn, to_completion, to_stream_chunks
from .config import CONFIG
from .errors import OpenAIError, not_implemented
from .logging_setup import get_logger, setup_logging
from .metrics import METRICS
from .models import ModelsCache
from .relay import relay_poll, relay_send
from .runtime import RUNTIME
from .security import check_request
from .session import SessionRegistry

setup_logging()
log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

registry: SessionRegistry | None = None
models_cache: ModelsCache | None = None
chat_ctx: ChatContext | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, models_cache, chat_ctx
    CONFIG.validate()
    await RUNTIME.start()
    registry = SessionRegistry(RUNTIME)
    RUNTIME.set_before_restart(registry.close_all)
    models_cache = ModelsCache(RUNTIME)
    chat_ctx = ChatContext(RUNTIME, registry, models_cache)
    await models_cache.ensure_fresh()
    await registry.start_reaper()
    log.info("bridge ready")
    yield
    await registry.stop_reaper()
    await registry.close_all()
    await RUNTIME.stop()


app = FastAPI(title="cursor-openai-bridge", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > CONFIG.max_body_bytes:
                return JSONResponse(status_code=413, content={"error": {"message": "request body too large", "type": "invalid_request_error"}})
        except ValueError:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid content-length", "type": "invalid_request_error"}})
    return await call_next(request)


@app.exception_handler(OpenAIError)
async def openai_error_handler(request: Request, exc: OpenAIError):
    return exc.response()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    ok = RUNTIME.is_ready() and await RUNTIME.ping()
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok})


@app.get("/metrics")
async def metrics():
    snap = METRICS.snapshot()
    snap["active_sessions"] = registry.count() if registry else 0
    return snap


@app.get("/v1/models")
async def list_models(_: None = Depends(check_request)):
    return {"object": "list", "data": await models_cache.list()}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, _: None = Depends(check_request)):
    return await models_cache.get(model_id)


@app.post("/v1/embeddings")
async def embeddings(_: None = Depends(check_request)):
    raise not_implemented("Cursor SDK is an agent SDK and cannot serve embeddings. Configure a separate embeddings provider in Continue.")


@app.post("/v1/completions")
async def completions(_: None = Depends(check_request)):
    raise not_implemented("Cursor SDK cannot serve raw FIM/completions. Configure a separate autocomplete provider in Continue.")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: None = Depends(check_request)):
    payload = await request.json()
    headers = {k.lower(): v for k, v in request.headers.items()}
    model = payload.get("model") or CONFIG.default_model
    stream = bool(payload.get("stream"))
    include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))
    cid = new_completion_id()
    created = int(time.time())
    METRICS.requests_total += 1

    if not stream:
        events = run_turn(payload, headers, chat_ctx)
        try:
            result = await to_completion(events, cid, created, model)
        except Exception:
            METRICS.errors_total += 1
            raise
        return JSONResponse(content=result)

    async def sse():
        t0 = time.monotonic()
        first = True
        n_chunks = 0
        METRICS.active_streams += 1
        events = run_turn(payload, headers, chat_ctx)
        try:
            async for chunk in to_stream_chunks(events, cid, created, model, include_usage):
                if first:
                    METRICS.record_ttft(time.monotonic() - t0)
                    first = False
                n_chunks += 1
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except OpenAIError as e:
            METRICS.errors_total += 1
            yield f"data: {json.dumps(e.body(), ensure_ascii=False)}\n\n"
        finally:
            METRICS.active_streams -= 1
            elapsed = time.monotonic() - t0
            if elapsed > 0 and n_chunks:
                METRICS.record_tps(n_chunks / elapsed)
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/v1/cursor/sessions/reset")
async def reset_sessions(request: Request, _: None = Depends(check_request)):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    conv_id = body.get("conversation_id") if isinstance(body, dict) else None
    if conv_id:
        session = registry.get(conv_id)
        if session:
            await registry.close(session)
        return {"closed": [conv_id]}
    ids = registry.all_ids()
    for i in ids:
        s = registry.get(i)
        if s:
            await registry.close(s)
    return {"closed": ids}


@app.delete("/v1/cursor/sessions/{conversation_id}")
async def delete_session(conversation_id: str, _: None = Depends(check_request)):
    session = registry.get(conversation_id)
    if session:
        await registry.close(session)
    return {"closed": bool(session)}


@app.post("/relay/send")
async def relay_send_route(request: Request):
    return await relay_send(request, chat_ctx)


@app.get("/relay/poll")
async def relay_poll_route(request: Request):
    return await relay_poll(request, chat_ctx)


if STATIC_DIR.exists():
    @app.get("/")
    async def index():
        index_file = STATIC_DIR / "robot.html"
        if index_file.exists():
            return StreamingResponse(iter([index_file.read_text(encoding="utf-8")]), media_type="text/html")
        return JSONResponse({"status": "cursor-openai-bridge running"})
