"""Modal deployment for the understanding backend.

Two services deployed in a single Modal app:

  1. GemmaBackend  — GPU L4. Wraps backend.main:app (the existing
     FastAPI app), so every endpoint already defined there (/health,
     /layers, /chat with SSE streaming) is reachable via Modal.
     Vectors and calibration files are bundled into the image. Model
     weights are cached in a Modal Volume mounted at HF_HOME so they
     survive container scale-down.

  2. DebertaSentiment — CPU. Wraps deberta-v3-large-zeroshot-v2.0 so
     the Vercel /api/sentiment route can proxy here instead of
     loading the model in the Vercel function. CPU is cheap and
     adequate for zero-shot on ≤1500-char inputs.

Both services scale to zero (60s idle window) to minimize cost.

DEPLOY
------

    # one-time: store the HF token as a Modal secret named "hf-token"
    modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxx

    # deploy
    modal deploy modal_app.py

Modal prints the per-endpoint URLs. The Gemma backend's `/health` and
`/chat` live under the GemmaBackend ASGI app; the DeBERTa endpoint
is a single function. See DEPLOY.md for env var wiring on Vercel.
"""
from __future__ import annotations
import os
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Image / volume setup
# ---------------------------------------------------------------------------

# Local repo layout we mirror into the image. Everything under data/ that
# the backend actually needs is small (~5MB total) so we bundle it directly
# rather than using a Volume.
LOCAL_DATA = Path(__file__).parent / "data"
VECTOR_DIRS_TO_BUNDLE = [
    "data/thinking/vectors/reply/cutoff_50",
    "data/thinking/vectors/thought/cutoff_50",
]

# Versions pinned to match the local .venv at write time so behavior
# stays identical between local dev and Modal. If we ever need bleeding-
# edge nightly torch for new CUDA support, swap to the .pip_install_from_pyproject
# pattern.
PY_VERSION = "3.12"

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        add_python=PY_VERSION,
    )
    .apt_install("git")
    # Heavy ML stack first so its layer caches across edits to the app code.
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "sentencepiece>=0.2",
        "nnsight>=0.7",
    )
    # Service stack.
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.32",
        "sse-starlette>=2.1",
        "slowapi>=0.1.9",
        "pydantic>=2",
        "numpy>=2",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .pip_install("hf-transfer>=0.1")
    # App code + vectors. Adds late so source edits don't bust the heavier
    # pip layers above.
    .add_local_dir("backend", "/app/backend")
    .add_local_dir("pipeline", "/app/pipeline")  # for topic_split.json
    .add_local_dir(
        "data/thinking/vectors/reply/cutoff_50",
        "/app/data/thinking/vectors/reply/cutoff_50",
    )
    .add_local_dir(
        "data/thinking/vectors/thought/cutoff_50",
        "/app/data/thinking/vectors/thought/cutoff_50",
    )
    .workdir("/app")
)

cpu_image = (
    modal.Image.debian_slim(python_version=PY_VERSION)
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "sentencepiece>=0.2",
        "fastapi>=0.115",
        "pydantic>=2",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# Persistent volume for HuggingFace weight cache. First container ever
# downloads Gemma weights; all subsequent cold-starts pull from this
# volume (Modal local SSD), much faster than HF Hub.
hf_volume = modal.Volume.from_name(
    "understanding-hf-cache", create_if_missing=True,
)
deberta_volume = modal.Volume.from_name(
    "understanding-deberta-cache", create_if_missing=True,
)

hf_secret = modal.Secret.from_name("hf-token")


app = modal.App("understanding")


# ---------------------------------------------------------------------------
# Gemma backend (GPU)
# ---------------------------------------------------------------------------

@app.cls(
    image=gpu_image,
    gpu="L4",
    timeout=600,
    scaledown_window=60,
    secrets=[hf_secret],
    volumes={"/cache/huggingface": hf_volume},
    min_containers=0,         # scale-to-zero
    max_containers=2,
)
@modal.concurrent(max_inputs=4)
class GemmaBackend:
    """Hosts the existing FastAPI app (backend.main:app). Lifespan runs
    on container start, which is where EmotionEngine loads Gemma + vectors."""

    @modal.asgi_app()
    def fastapi_app(self):
        # Import inside the function so the import happens in the Modal
        # container (which has CUDA / the GPU). The lifespan inside
        # backend.main loads the engine on startup.
        from backend.main import app as fastapi_app
        return fastapi_app


# ---------------------------------------------------------------------------
# DeBERTa zero-shot classifier (CPU)
# ---------------------------------------------------------------------------

CANDIDATE_LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]
LABEL_TO_CAPITAL = {
    "joy": "Joy", "sadness": "Sadness", "anger": "Anger",
    "fear": "Fear", "disgust": "Disgust", "surprise": "Surprise",
}
HYPOTHESIS_TEMPLATE = "This text expresses {}."
MAX_INPUT_CHARS = 2000


@app.cls(
    image=cpu_image,
    cpu=2.0,
    memory=4096,
    timeout=120,
    scaledown_window=60,
    volumes={"/cache/huggingface": deberta_volume},
    min_containers=0,
    max_containers=4,
)
@modal.concurrent(max_inputs=8)
class DebertaSentiment:
    """Zero-shot Ekman-6 classifier. The Vercel /api/sentiment route
    POSTs here with {text: string}. Returns {emotions: {Joy: int, ...}}
    so the frontend's emotion-classifier.ts contract stays identical."""

    @modal.enter()
    def load(self):
        from transformers import pipeline
        self.clf = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            device="cpu",
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def classify(self, body: dict) -> dict:
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return {"emotions": {v: 0 for v in LABEL_TO_CAPITAL.values()}}
        cleaned = text.strip()[:MAX_INPUT_CHARS]
        if len(cleaned) < 3:
            return {"emotions": {v: 0 for v in LABEL_TO_CAPITAL.values()}}

        result = self.clf(
            cleaned,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        out = {
            LABEL_TO_CAPITAL[label]: int(round(scores.get(label, 0.0) * 100))
            for label in CANDIDATE_LABELS
        }
        return {"emotions": out}


# ---------------------------------------------------------------------------
# Sanity check helper — runs entrypoints from CLI:
#   modal run modal_app.py::warm
#   modal run modal_app.py::health
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def warm():
    """Force-start one container of each service to download model
    weights into the Modal volumes. Useful after first deploy or when
    the volume cache is empty."""
    print("Warming GemmaBackend (this triggers Gemma weight download)...")
    # Just calling any method on the class triggers @modal.enter() in a
    # container, which will lifespan-init the FastAPI app and load weights.
    g = GemmaBackend()
    print(f"  GemmaBackend ASGI URL: {g.fastapi_app.get_web_url()}")
    print("Warming DebertaSentiment (downloads deberta-large)...")
    d = DebertaSentiment()
    print(f"  DebertaSentiment URL: {d.classify.get_web_url()}")
    print("Done. Check Modal dashboard for download progress.")
