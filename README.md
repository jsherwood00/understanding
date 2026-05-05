# understanding

A live emotion-visualization tool for Gemma 4 E4B. Two-pane interface: a 6-emotion bar chart (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

Each emotion column shows two indicators that float independently:

- **Sharp dot (output)** — lexicon sentiment on the model's reply text. What the words sound like.
- **Soft halo (thinking)** — live projection of Gemma's residual stream onto the per-emotion direction vector at the chosen layer. What the model's internals look like.

When the dot floats away from the halo, the model's internal state isn't being expressed in its output — the **dissonance** signal. When they overlap, surface and internals are aligned.

The halo is computed by dotting Gemma's residual-stream activation at a chosen layer against six pre-computed contrastive emotion vectors, then mapping into [0, 100] using per-(emotion, layer) percentile bounds. The vectors were built from 7200 contrastive-prompted stories following the methodology of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model"), adapted for Gemma 4 E4B.

## Stack

- **Frontend**: Next.js 16 + React 19 (App Router), TypeScript, Tailwind v4
- **Backend**: FastAPI + sse-starlette, serves a live SSE token stream
- **Model**: Gemma 4 E4B in 4-bit nf4 on a local GPU (RTX 5060 Ti tested), via transformers 5.7 + bitsandbytes
- **Sentiment lexicon**: local TS file at `lib/sentiment.ts` for the surface dot
- **Selection sentiment**: a Next.js route (`/api/sentiment`) running j-hartmann distilroberta-base for highlight-to-analyze on transcript text

## Run (local, two processes)

You need an HF token with access to `google/gemma-4-E4B-it`. Activate the venv and start the backend:

```bash
source .venv/bin/activate
HF_TOKEN=hf_... uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Cold-start loads the model in ~5–10s and prints `Application startup complete`.

In another terminal, start the frontend:

```bash
npm install   # only the first time
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend reads `NEXT_PUBLIC_BACKEND_URL` (defaults to `http://localhost:8000`).

## How a turn flows

1. User submits a message in the browser.
2. `Workspace.tsx` POSTs `{message, layer, history}` to `${NEXT_PUBLIC_BACKEND_URL}/chat`.
3. The FastAPI route streams Server-Sent Events. For each generated token:
   - A forward hook captures the residual-stream activation at the chosen layer.
   - The backend dots the activation against each of the 6 emotion vectors at that layer, normalizes via calibration, and emits a `data: {type: "token", text, thinking}` event.
4. The frontend appends the text delta to the streaming reply, sets the **halo** heights from `thinking`, and reruns the local lexicon over the accumulated reply for the **dot** heights.
5. On the closing `done` event the frontend stores the turn (with snapshots) so it can be scrubbed/replayed via the controls below the bars.

## Layout

```
app/
  layout.tsx              Root layout, fonts, metadata
  page.tsx                Server shell (workspace)
  globals.css             Theme tokens, base styles, animations
  api/sentiment/route.ts  POST → distilroberta on selected text → 6-emotion vector
backend/
  main.py                 FastAPI: GET /health, GET /layers, POST /chat (SSE)
  inference.py            EmotionEngine: model load, forward hooks, generation loop
  calibrate.py            One-time script that produces data/vectors/calibration.json
  requirements.txt
components/
  Workspace.tsx           Client wrapper — chat state, SSE parsing, layer selection
  EmotionPanel.tsx        Left pane (6 columns of dot+halo)
  ChatPane.tsx            Right pane (transcript + input)
  LayerSelector.tsx       Pill row of {13, 17, 21, 25, 28, 32}
lib/
  emotions.ts             Types, colors, baseline, backend-key mapping
  sentiment.ts            Lexicon-based 6-emotion analyzer (the dot)
  emotion-classifier.ts   Bridge to /api/sentiment
pipeline/
  generate_stories.py     Builds the 7200-story corpus + activations
  compute_vectors.py      Contrastive subtraction → 36 emotion vectors
  verify.py               Sanity checks on the corpus
data/
  vectors/*.npy           36 contrastive vectors, shape (2560,)
  vectors/calibration.json  per-(emotion, layer) p5/p95 bounds
  vectors/best_layer_per_emotion.json
  topics.json, config.json
```

## Pipeline (already run, don't re-run)

The vectors in `data/vectors/` are inputs. Regenerating them takes hours on a single GPU and overwrites the calibration. Don't run `pipeline/generate_stories.py` or `pipeline/compute_vectors.py` unless you specifically want a fresh corpus. `pipeline/verify.py` is read-only and safe.

## Calibration

`backend/calibrate.py` runs ~50 hand-curated prompts spanning all 6 emotional contexts plus neutral, captures raw projection scores at every target layer for every generated token (~143k samples), and writes p5/p95 bounds per (emotion, layer) to `data/vectors/calibration.json`. Layer magnitudes vary 10–30× across layers, so per-layer bounds are essential — the live backend uses them to map raw scores into [0, 100].

```bash
HF_TOKEN=hf_... .venv/bin/python -m backend.calibrate
```

Re-run after vector changes or to recalibrate to a different prompt distribution. ~2 min on RTX 5060 Ti.

## Tuning knobs

In `backend/inference.py`:
- `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS` — generation defaults
- `DEFAULT_LAYER` — layer used when the request omits `layer` (default 21, paper's main-analysis depth)
- `FALLBACK_RAW_BOUND` — [-5, +5] linear map used only when `calibration.json` is missing

In `components/Workspace.tsx`:
- `SNAPSHOT_EVERY_N_TOKENS` — cadence for recording the bar trajectory (default 5)
- `DEFAULT_LAYER` — initial layer for the selector (default 21)

In `components/EmotionPanel.tsx`:
- Bar transition timing: `transition: bottom 300ms ease-out` on the dot and halo elements.

In `lib/sentiment.ts`:
- `LEXICON` — word → partial emotion-weight map for the dot
- Density curve `100 * (1 - exp(-14 * density))` controls saturation

## Layer cheat-sheet

The 6 target layers map roughly to:

| Layer | Label | What it tracks |
| ---: | --- | --- |
| 13 | Sensory | Emotional content of recent input |
| 17 | Sensory–integrated | |
| 21 | Integrated | Context being processed (paper's main depth, ~2/3 of 42) |
| 25 | Action–integrated | |
| 28 | Action | What's about to be expressed |
| 32 | Output | Predicting the next token |

Switching layers re-routes the halo's source. The dot (lexicon over reply text) doesn't change with the layer.
