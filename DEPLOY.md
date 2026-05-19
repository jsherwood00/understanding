# Deployment — Modal + Vercel

This project deploys as two pieces:

- **Backend** (Gemma 4 E4B + DeBERTa zero-shot) → **Modal** (GPU + CPU
  serverless functions, scale-to-zero, cheapest viable tier per the
  user's stated preference)
- **Frontend** (Next.js app) → **Vercel**

Both pieces are scale-to-zero by default. With no traffic the project
costs ~$0/day. Each cold start of the Gemma container takes ~10-15 s.
Each cold start of the DeBERTa container takes ~5-10 s.

---

## Prerequisites

1. **HuggingFace token** with access to `google/gemma-4-E4B-it` (the
   gated Gemma model). Request access at
   https://huggingface.co/google/gemma-4-E4B-it then create a read
   token at https://huggingface.co/settings/tokens
2. **Modal account** with billing. Free tier covers some usage; you
   need credits for sustained GPU time. `pip install modal` and
   `modal token new` to authenticate the CLI.
3. **Vercel account** and the Vercel CLI: `npm i -g vercel`
4. **GitHub repo** that Vercel will track (or use `vercel deploy`
   directly from this repo).

---

## Step 1 — Deploy the backend to Modal

```bash
# one-time: store the HF token as a Modal secret
modal secret create hf-token HF_TOKEN=hf_xxxxxxxxxxxx

# deploy both services (GemmaBackend GPU + DebertaSentiment CPU)
modal deploy modal_app.py
```

Modal prints two URLs:

```
✓ Created web function GemmaBackend.fastapi_app =>
    https://<workspace>--understanding-gemmabackend-fastapi-app.modal.run

✓ Created web function DebertaSentiment.classify =>
    https://<workspace>--understanding-debertasentiment-classify.modal.run
```

Copy these. You'll set them as env vars in Vercel.

**Optional warm-up** (downloads weights to the Modal volumes so the
first user request doesn't pay the HF-download latency):

```bash
modal run modal_app.py::warm
```

This triggers `@modal.enter()` in one container of each service, which
loads Gemma (~5GB) and DeBERTa-large (~1.4GB) into the Modal volumes
(`understanding-hf-cache`, `understanding-deberta-cache`). Subsequent
cold starts read from the volume rather than re-downloading.

### Verifying the backend

```bash
curl https://<your-workspace>--understanding-gemmabackend-fastapi-app.modal.run/health
# {"ok":true,"warmed_up":true,...,"calibrated":{"reply":true,"thought":true}}

curl -X POST \
  https://<your-workspace>--understanding-debertasentiment-classify.modal.run \
  -H "Content-Type: application/json" \
  -d '{"text":"I am so happy right now"}'
# {"emotions":{"Joy":95,"Sadness":2,...}}
```

---

## Step 2 — Deploy the frontend to Vercel

From the repo root:

```bash
vercel link            # connects this directory to a Vercel project
vercel env add NEXT_PUBLIC_BACKEND_URL production
# paste: https://<workspace>--understanding-gemmabackend-fastapi-app.modal.run

vercel env add MODAL_DEBERTA_URL production
# paste: https://<workspace>--understanding-debertasentiment-classify.modal.run

vercel deploy --prod
```

The Vercel build picks up these env vars and routes:

- frontend requests → `${NEXT_PUBLIC_BACKEND_URL}/chat` (SSE) and
  `/health`
- `/api/sentiment` → proxies to `${MODAL_DEBERTA_URL}` (server-side
  only; `MODAL_DEBERTA_URL` is NOT prefixed with `NEXT_PUBLIC_` so it
  stays out of the browser bundle)

If `MODAL_DEBERTA_URL` is unset, `/api/sentiment` falls back to running
DeBERTa locally via `@huggingface/transformers` (the dev-mode path).

---

## Step 3 — Configure backend CORS for Vercel

The Modal-hosted FastAPI app reads `ALLOWED_ORIGINS` from its env. The
default permits only `http://localhost:3000`, which blocks Vercel.

Either:

**(a) Single production origin** — add a Modal secret with your Vercel
URL and reference it in `modal_app.py`:

```bash
modal secret create app-origins ALLOWED_ORIGINS=https://your-app.vercel.app
```

Then in `modal_app.py`, add `Secret.from_name("app-origins")` to the
`GemmaBackend` `@app.cls` decorator's `secrets=[...]` list.

**(b) Preview deployments included** — add a regex secret as well:

```bash
modal secret create app-origins-regex \
  ALLOWED_ORIGIN_REGEX='^https://([a-z0-9-]+\.)?vercel\.app$'
```

Then redeploy: `modal deploy modal_app.py`.

---

## Step 4 — Verify the production wiring

Open your Vercel URL. Send a prompt. Check:

- The chat reply streams via SSE (browser DevTools → Network → look for
  the `chat` request streaming `data:` events)
- Halos and dots animate during streaming (proves the per-token
  projection JSON is reaching the frontend)
- Turn on the classifier toggle. Send a prompt with emotional content.
  Sentiment ticks should appear at end-of-turn (proves
  `/api/sentiment` → Modal DeBERTa is working)

If nothing streams, check Modal's dashboard for the container logs of
the GemmaBackend service. The first request after deployment will
download Gemma weights — first chat response is delayed ~30-90s if you
haven't run `modal run modal_app.py::warm` first.

---

## Cost notes

- **L4 GPU on Modal**: ~$0.70/hr while warm. With scale-to-zero and a
  60-second idle window, a typical demo session of 10 chats costs
  pennies. Idle: $0.
- **CPU container for DeBERTa**: ~$0.05/hr while warm. Same scaling
  rules. Idle: $0.
- **Modal Volumes** (HF cache): a few dollars/month for ~7GB of cached
  model weights.
- **Vercel**: free tier covers the Next.js app indefinitely for low
  traffic.

If costs ever surprise you, check Modal's billing dashboard — they
break down per-function GPU-seconds.

---

## Local development

Modal is only used in production. Local dev still uses
`backend/main.py` directly with uvicorn:

```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
npm run dev
```

Without `NEXT_PUBLIC_BACKEND_URL` set, the frontend defaults to
`http://localhost:8000`. Without `MODAL_DEBERTA_URL` set,
`/api/sentiment` runs DeBERTa via `@huggingface/transformers` in the
Node process.

---

## Updating vectors / calibration

When `pipeline/compute_vectors.py` or `pipeline/percentile_calibrate.py`
is rerun and the calibration JSONs change, you need to rebuild the
Modal image so the new files are bundled in:

```bash
modal deploy modal_app.py
```

Modal will detect the local-mount changes and rebuild the relevant
image layer (the heavy pip layers above are cached).

---

## Tearing down

```bash
modal app stop understanding         # stops all containers
modal volume delete understanding-hf-cache       # ~5GB of Gemma weights
modal volume delete understanding-deberta-cache  # ~1.4GB of DeBERTa weights
```

Vercel: project settings → delete deployment.
