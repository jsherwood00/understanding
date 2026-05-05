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

# Local-dev CORS: Next.js runs on :3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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
        "calibrated": bool(_engine and _engine.calibration is not None),
    }


@app.get("/layers")
async def layers():
    """Layer metadata for the frontend selector."""
    labels = {
        13: "Sensory — emotional content of recent input",
        17: "Sensory–integrated",
        21: "Integrated — context being processed",
        25: "Action–integrated",
        28: "Action — preparing to express",
        32: "Output — predicting next token",
    }
    return [
        {"layer": L, "label": labels.get(L, f"layer {L}")}
        for L in TARGET_LAYERS
    ]


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
