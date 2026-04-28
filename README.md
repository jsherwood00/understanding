# understanding

A UI prototype for an LLM emotion visualization tool — framed as a deception-detection mockup. Two-pane interface: an EQ-style bar chart of six emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

Each bar shows two cut-off heights:

- **Surface** (lighter shade, drawn on top) — what a sentiment analyzer reads off the model's output tokens. Computed locally from a small lexicon scoring the reply text.
- **Internal** (darker shade, drawn behind, full saturation) — the model's actual hidden reaction. Self-reported by the model alongside its reply (would, in a real system, come from analyzing internal/thinking tokens).

When `internal` rises above `surface`, the dark sticks up out of the lighter base — the deception signal. When the two match, the bar reads cleanly as one shade.

The chat backend is **Gemini 2.5 Flash via fal.ai** (`fal-ai/any-llm` endpoint, hardcoded to `google/gemini-2.5-flash` — no model picker, by design). Each turn returns the reply text, an honest internal emotion vector, and the model's thinking trace. The thinking trace renders as a collapsible disclosure above each assistant message.

## Stack

- Next.js 16 + React 19 (App Router)
- TypeScript, Tailwind CSS v4
- `next/font` with Fraunces (serif) + Inter (sans)
- fal.ai HTTP API → Gemini 2.5 Flash with `reasoning: true` for thinking traces
- Local lexicon-based sentiment analyzer for the surface vector

## Run

```bash
npm install
echo "FAL_API_KEY=your_fal_key_here" > .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

If `FAL_API_KEY` is missing or invalid the chat shows a "couldn't reach the model" notice.

## Build

```bash
npm run build
npm start
```

## Layout

```
app/
  layout.tsx              Root layout, fonts, metadata
  page.tsx                Server shell (header + workspace)
  globals.css             Theme tokens, base styles, animations
  api/evaluate/route.ts   POST → fal.ai → JSON {reply, internal} →
                          locally analyze reply for surface →
                          return {text, thinking, surface, internal}
components/
  Header.tsx              Top bar
  Workspace.tsx           Client wrapper — chat state + bar updates
  EmotionPanel.tsx        Left pane (six bars, dual fill)
  ChatPane.tsx            Right pane (transcript with collapsible
                          thinking disclosures + input)
lib/
  emotions.ts             Types, colors, baseline
  sentiment.ts            Lexicon-based 6-emotion analyzer
```

## How a turn flows

1. User submits a message in the browser.
2. `Workspace.tsx` POSTs the conversation to `/api/evaluate`.
3. The route formats the messages as a User/You-prefixed prompt, sends to fal.ai with a system prompt that asks Gemini to:
   - Reply naturally
   - Self-report an honest internal emotion vector
   - Output strict JSON
4. fal.ai returns `{output: <JSON string>, reasoning: <thinking trace>}`.
5. The route parses `output` to extract `{reply, internal}`, runs the local sentiment analyzer on `reply` for the `surface` vector, and returns `{text, thinking, surface, internal}` to the browser.
6. The chat pane renders the thinking disclosure (collapsed by default) and streams the reply text character-by-character. When streaming completes, both bar layers settle to their new heights.

## Tuning knobs

In `components/Workspace.tsx`:

- `STREAM_INTERVAL_MS` — per-character reveal delay (default 20)
- `SETTLE_DELAY_MS` — pause between last char and applying the new state (default 220)

In `app/api/evaluate/route.ts`:

- `MODEL` — pinned to `google/gemini-2.5-flash`. Changing this is intentional.
- `SYSTEM_PROMPT` — calibration / honesty instructions.
- `max_tokens` — currently 1024.

In `lib/sentiment.ts`:

- `LEXICON` — word → partial emotion-weight map. Extend for better surface accuracy.
- The saturation curve `100 * (1 - exp(-0.4 * count))` controls how quickly each emotion saturates with hits.

Bar transition speed is the CSS `transition: height 300ms ease-out` on the fill divs in `components/EmotionPanel.tsx`.
