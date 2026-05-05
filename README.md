# understanding

A live emotion-visualization tool for Gemma 4 E4B. Two-pane interface: a 6-emotion bar chart (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

Each emotion column has two indicators that float independently:

- **Soft halo (thinking)** — live projection of Gemma's residual stream onto the per-emotion direction vector at the chosen layer. Updates *every token* during generation. What the model's internals are doing as it composes the reply.
- **Sharp dot (output)** — distilroberta classification of the model's *full reply text*, run **once** when the turn ends. Hidden during streaming. What the response, as a whole, sounds like to a strong text-emotion classifier.

When the dot floats away from the halo, the model's internal state isn't being reflected in its words — the **dissonance** signal. When they're close, surface and internals are aligned.

The halo is computed by dotting Gemma's residual-stream activation at a chosen layer against six pre-computed contrastive emotion vectors, then mapping into [0, 100] using per-(emotion, layer) percentile bounds. The vectors were built from 7200 contrastive-prompted stories following the methodology of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model"), adapted for Gemma 4 E4B.

The dot is computed once per turn by sending the full reply text to a Next.js route that runs `SamLowe/roberta-base-go_emotions-onnx` (28-label GoEmotions, mapped to Ekman 6 via the official Demszky 2020 mapping). distilroberta is pre-trained and outputs calibrated probabilities, so this side needs no per-emotion bounds.

## Stack

- **Frontend**: Next.js 16 + React 19 (App Router), TypeScript, Tailwind v4
- **FastAPI backend**: serves a live SSE token stream of Gemma's residual-stream projections (`backend/`)
- **Gemma 4 E4B** in 4-bit nf4 on a local GPU (RTX 5060 Ti tested), via transformers 5.7 + bitsandbytes
- **distilroberta classifier** (`SamLowe/roberta-base-go_emotions-onnx`) for the post-hoc surface dot AND the highlight-to-analyze tooltip, hosted in a Next.js route at `/api/sentiment`
- **Lexicon** (`lib/sentiment.ts`) — fast instant fallback for the highlight-to-analyze feature only; not used during streaming.

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
2. `Workspace.tsx` POSTs `{message, layer, history}` to `${NEXT_PUBLIC_BACKEND_URL}/chat`. Output dot (state) is set to `null` — the dot disappears.
3. The FastAPI route streams Server-Sent Events. For each generated token:
   - A forward hook captures the residual-stream activation at the chosen layer.
   - The backend dots the activation against each of the 6 emotion vectors at that layer, normalizes via calibration, and emits a `{type: "token", text, thinking}` event.
4. The frontend appends the text delta to the streaming reply and sets the **halo** heights from `thinking`. The **dot stays hidden** the whole time.
5. On the closing `done` event the frontend POSTs the full reply text to `/api/sentiment` (distilroberta). When that returns the **dot appears** at the classifier's reading. The turn is then stored (with per-snapshot thinking values) so it can be scrubbed/replayed; during replay the dot is constant at the post-hoc value while the halo varies.

A pre-warm POST to `/api/sentiment` fires on Workspace mount so the first turn doesn't pay the 5–15s ONNX cold-start.

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

There are **two emotion signals on screen**, and they're calibrated very differently. Both procedures are documented here so you can tell what's a knob and what's a fixed model artifact.

### 1. Halo — projection at chosen layer

**What's calibrated**: raw dot products between Gemma's residual-stream activation and the contrastive emotion vectors don't have a natural [0, 100] range. We compute the empirical distribution of those dot products under typical generation and use the 5th/95th percentile as the bar's `[0, 100]` endpoints.

**Where**: `backend/calibrate.py` → `data/vectors/calibration.json`.

**Method (per-emotion, per-layer)**:
- 50 hand-curated prompts, 8 each across 6 emotional contexts + 2 neutral (see `PROMPTS` in `backend/calibrate.py`)
- Up to 80 generated tokens per prompt (sampling at temperature 0.7, top-p 0.95, seed 42)
- For each token, capture the raw projection score (dot product) at every target layer (13/17/21/25/28/32) for every emotion → ~3970 samples per (emotion, layer)
- Compute p5 and p95 per (emotion, layer)
- Total samples: 142,920 (= 6 emotions × 6 layers × ~3970)

**Why per-layer**: layer magnitudes vary 10–30× across the network. Joy at layer 13 has p5/p95 of −159 to −99 (span 60); joy at layer 21 has p5/p95 of +7 to +22 (span 15). A single normalization range is impossible.

**At runtime (`backend/inference.py`)**:
```python
display = max(0, min(100, (raw - lo) / (hi - lo) * 100))
```
where `(lo, hi)` is the (emotion, layer)-specific p5/p95 pair from `calibration.json`. If the file is missing, falls back to `[-5, +5]` linear (which saturates almost everything to 0 or 100, demonstrating why calibration matters).

**Re-run** when vectors change or to retune to a different prompt distribution:
```bash
HF_TOKEN=hf_... .venv/bin/python -m backend.calibrate
```
~2 min on RTX 5060 Ti.

### 2. Dot — distilroberta classification of the full reply

**What's calibrated**: nothing. distilroberta (`SamLowe/roberta-base-go_emotions-onnx`) is pre-trained on GoEmotions and outputs per-label probabilities in [0, 1]. We sum probabilities for the GoEmotions labels that map to each Ekman bucket (Demszky 2020 mapping, encoded in `lib/emotion-classifier.ts`), clamp to [0, 1], scale ×100, round to integer. No empirical bounds are needed — the model's outputs are already the calibrated quantities.

**Where**: classifier loads in the Next.js server runtime via `@huggingface/transformers`, called from `app/api/sentiment/route.ts`, source in `lib/emotion-classifier.ts`.

**When**: only at the **end of each turn** (the dot is hidden during streaming), and on **selection** when the user highlights an excerpt in the chat (250ms debounce).

**Why no per-token classification during streaming**: distilroberta is too slow per token (~50–200ms warm) and the lexicon-density approach we tried first was misleading: the halo's projection sees the entire conversation context (residual stream is cumulative), but a per-token lexicon over the reply alone sees only what's been emitted so far. That asymmetry made the dissonance gap look bigger than it is. Running distilroberta once on the full reply is symmetric (full reply text vs. full reply context) and uses a real model instead of a 600-word lexicon.

**Pre-warm**: a single dummy `/api/sentiment` POST fires on `Workspace.tsx` mount so the first turn doesn't eat the 5–15s ONNX cold-start when distilroberta loads.

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
- `LEXICON` — word → partial emotion-weight map. Used only as the instant fallback for the highlight-to-analyze tooltip while distilroberta is still being fetched. Not used during streaming.

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
