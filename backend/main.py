"""
FastAPI app for the live emotion-projection backend.

Endpoints:
  GET  /health         — liveness probe + warmup status
  POST /chat           — SSE stream of {token, thinking} events for one prompt

Run locally:
  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from backend.inference import DEFAULT_LAYER, EmotionEngine, TARGET_LAYERS


_engine: Optional[EmotionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = EmotionEngine()
    yield
    _engine = None


limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(title="understanding-backend", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# CORS — default to local dev (Next.js on :3000). In production set the
# ALLOWED_ORIGINS env var to a comma-separated list of accepted Vercel
# origins (e.g. "https://understanding.vercel.app,https://*.vercel.app").
# A bare "*" disables the allowlist entirely (only useful for debugging).
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_env_origins = os.environ.get("ALLOWED_ORIGINS", _default_origins)
_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]

# `allow_origin_regex` covers preview deployments (*.vercel.app) — opt in
# via env if needed. Default off.
_origin_regex = os.environ.get("ALLOWED_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Slow down."},
    )


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[list[ChatHistoryItem]] = None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "warmed_up": _engine is not None,
        "model": _engine.model.config._name_or_path if _engine else None,
        "device": str(_engine.device) if _engine else None,
        "target_layers": TARGET_LAYERS,
        "default_layer": DEFAULT_LAYER,
        "calibrated": {
            scope: bool(_engine and _engine.calibration.get(scope) is not None)
            for scope in ("reply", "thought")
        } if _engine else None,
    }


@app.get("/layers")
async def layers():
    """Layer list for the frontend selector. No semantic labels —
    interpretive claims about what each layer does are not supported by
    the evidence we have, so we just expose the raw indices."""
    return [{"layer": L} for L in TARGET_LAYERS]


@app.post("/chat")
@limiter.limit("12/minute")
async def chat(request: Request, body: ChatRequest):
    if _engine is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"error": "Model not warmed up yet."},
        )

    history = (
        [{"role": h.role, "content": h.content} for h in body.history]
        if body.history else None
    )

    async def event_publisher():
        try:
            async for event in _engine.generate_stream(
                message=body.message,
                history=history,
            ):
                yield {"data": json.dumps(event)}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "error": str(e)})}

    return EventSourceResponse(event_publisher())
