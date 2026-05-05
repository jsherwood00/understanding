# understanding

A live emotion-visualization tool for Gemma 4 E4B. Two-pane interface: a 6-emotion bar chart (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

Each emotion column has two indicators that float independently:

- **Soft halo (thinking)** — live projection of Gemma's residual stream onto the per-emotion direction vector at the chosen layer. Updates *every token* during generation. What the model's internals are doing as it composes the reply.
- **Sharp dot (output)** — distilroberta classification of the model's *full reply text*, run **once** when the turn ends. Hidden during streaming. What the response, as a whole, sounds like to a strong text-emotion classifier.

When the dot floats away from the halo, the model's internal state isn't being reflected in its words — the **dissonance** signal. When they're close, surface and internals are aligned.

The halo is computed by dotting Gemma's residual-stream activation at a chosen layer against six pre-computed contrastive emotion vectors, then mapping into [0, 100] using per-(emotion, layer) percentile bounds. The vectors were built from 7200 contrastive-prompted stories following the methodology of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model"), adapted for Gemma 4 E4B.

The dot is computed once per turn by sending the full reply text to a Next.js route that runs **`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`** in zero-shot NLI mode. Each of the 6 Ekman emotions is fed as a hypothesis (`"This text expresses joy."`, etc.) and the model returns a multi-label entailment probability per hypothesis. We use raw probability × 100 directly. No GoEmotions taxonomy mapping, no calibration. Zero-shot NLI handles narrative/literary text well and doesn't suffer the disgust→fear confusion that the GoEmotions classifier had.

## Stack

- **Frontend**: Next.js 16 + React 19 (App Router), TypeScript, Tailwind v4
- **FastAPI backend**: serves a live SSE token stream of Gemma's residual-stream projections (`backend/`)
- **Gemma 4 E4B** in 4-bit nf4 on a local GPU (RTX 5060 Ti tested), via transformers 5.7 + bitsandbytes
- **deberta-v3-large zero-shot classifier** (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) for the post-hoc surface dot AND the highlight-to-analyze tooltip, hosted in a Next.js route at `/api/sentiment`. NLI-style: each emotion is independently scored as the entailment probability of "This text expresses {emotion}."
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

**What's calibrated**: raw dot products between Gemma's residual-stream activation and the contrastive emotion vectors don't have a natural [0, 100] range. We pin the **neutral baseline to 0** and a **high percentile of matching-emotional text to 100**, with shifted-negative values clipped to 0 ("less aligned than neutral" reads as no signal, not as a negative bar). This matches the dot's semantics — neutral text → near 0 on both indicators.

**Where**: `backend/calibrate.py` → `data/vectors/calibration.json`.

**Method (per-emotion, per-layer)**:
- 50 hand-curated prompts, **labeled by intended emotion**: 8 each for joy/sadness/anger/fear/surprise/disgust + 2 neutral controls (see `PROMPTS` and `LABELS` in `backend/calibrate.py`)
- Up to 80 generated tokens per prompt (sampling at temperature 0.7, top-p 0.95, seed 42)
- For each token, capture raw projection scores at every target layer (13/17/21/25/28/32) for every emotion. Bucket samples by the prompt's intended emotion.
- For each (emotion E, layer L):
  - `neutral_mean = mean(E-projection on tokens from neutral prompts)`
  - `emotional_p95 = p95(E-projection on tokens from E-evoking prompts)`
  - `scale = max(emotional_p95 − neutral_mean, ε)`
- Total samples: 142,920 (= 6 emotions × 6 layers × ~3970)

**Why per-layer**: layer magnitudes vary 10–30× across the network. The same emotional shift might be 50 raw units at layer 13 but only 5 at layer 21. A single normalization range is impossible.

**At runtime (`backend/inference.py`)**:
```python
display = max(0, min(100, (raw - neutral_mean) / scale * 100))
```
Neutral text → 0. Strongly emotional text → ~100. Negatives clip to 0. Backward-compatible fallback to the legacy `(p5, p95)` schema if `calibration.json` predates the shift-and-scale change.

**Re-run** when vectors change or to retune to a different prompt distribution:
```bash
HF_TOKEN=hf_... .venv/bin/python -m backend.calibrate
```
~2 min on RTX 5060 Ti.

### 2. Dot — zero-shot NLI classification of the full reply

**What's calibrated**: nothing. `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` is pre-trained for NLI and the Ekman emotion labels are passed at inference time. The pipeline forms one hypothesis per label — "This text expresses joy.", "This text expresses sadness.", etc. — and returns the entailment probability for each, independently (`multi_label: true`). We multiply by 100 and round. That's the bar value.

**Why no calibration**: zero-shot NLI gives clean per-label probabilities that already span [0, 100] meaningfully. Applying percentile normalization on top would amplify noise. We tried percentile calibration with the previous classifier and it had the typical narrative-text problem; switching to zero-shot NLI fixed both the calibration question and the disgust/fear confusion at once.

**Where**: classifier loads in the Next.js server runtime via `@huggingface/transformers`, called from `app/api/sentiment/route.ts`, source in `lib/emotion-classifier.ts`.

**When**: only at the **end of each turn** (the dot is hidden during streaming), and on **selection** when the user highlights an excerpt in the chat (250ms debounce).

**Why not per-token during streaming**: too slow (~150–500ms warm per call for deberta-large) and structurally awkward — see the "How a turn flows" section above for the asymmetry argument that drove the post-hoc design.

**Pre-warm**: a single dummy `/api/sentiment` POST fires on `Workspace.tsx` mount so the first turn doesn't eat the ~25s cold-start when deberta-v3-large initializes (~1.4GB model).

**Hypothesis template tuning**: `"This text expresses {}."` is the current template (in `lib/emotion-classifier.ts` as `HYPOTHESIS_TEMPLATE`). Other phrasings ("The author is feeling {}.", "The speaker shows {}.") may give different distributions; this one tested cleanly on both direct emotional language and narrative prose.

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
