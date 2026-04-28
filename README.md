# understanding

A UI prototype for an LLM emotion visualization tool. Two-pane interface: an EQ-style bar chart of six internal emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right. Each user message is sent to Claude Opus 4.7 with a forced tool call that returns both the reply and an honest emotion profile. When the response finishes streaming, the bars settle to the new profile.

A second "surface emotion" tier sits below as a `coming soon` placeholder, stubbed for the eventual two-layer view.

## Stack

- Next.js 16 + React 19 (App Router)
- TypeScript, Tailwind CSS v4
- `next/font` with Fraunces (serif) + Inter (sans)
- Anthropic SDK (`@anthropic-ai/sdk`) on a Next.js Route Handler
- Client-side conversation state, no database

## Setup

```bash
npm install
cp .env.local.example .env.local
# edit .env.local and paste your key from https://console.anthropic.com/settings/keys
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The chat will show a "couldn't reach the model" notice if `ANTHROPIC_API_KEY` is missing or invalid.

## Build

```bash
npm run build
npm start
```

## Deploy

On Vercel, set `ANTHROPIC_API_KEY` in the project's Environment Variables (Production + Preview) before deploying. No other config needed.

## Layout

```
app/
  layout.tsx              Root layout, fonts, metadata
  page.tsx                Server shell (header + workspace)
  globals.css             Theme tokens, base styles, animations
  api/evaluate/route.ts   POST → calls Claude with the `respond` tool
components/
  Header.tsx              Top bar
  Workspace.tsx           Client wrapper — chat state + bar updates
  EmotionPanel.tsx        Left pane (bars + surface emotion preview)
  ChatPane.tsx            Right pane (transcript + input + streaming + error)
lib/
  emotions.ts             Types, colors, baseline
```

## How the API call works

`POST /api/evaluate` accepts `{ messages: [{role, content}, ...] }` and proxies to Anthropic with a forced tool call:

```ts
tool_choice: { type: "tool", name: "respond" }
```

The `respond` tool's input schema is `{ text: string, emotions: { Joy, Sadness, Anger, Fear, Disgust, Surprise } }` (each 0–100, integer). The system prompt instructs Claude to be honest about negative emotions when warranted — flat neutrality is a failure mode, not the goal. Server-side, emotion values are clamped 0–100 and rounded before returning.

The system prompt is sent with `cache_control: { type: "ephemeral" }` so repeated turns reuse the cached prefix.

## Tuning knobs

In `components/Workspace.tsx`:

- `STREAM_INTERVAL_MS` — per-character reveal delay (default 20)
- `SETTLE_DELAY_MS` — pause between last char and applying the new profile (default 220)

Bar transition speed is the CSS `transition: height 300ms ease-out` on the fill div in `components/EmotionPanel.tsx`.

Model and behavior live in `app/api/evaluate/route.ts`:

- `model: "claude-opus-4-7"` — swap for `claude-sonnet-4-6` for faster/cheaper turns
- `thinking: { type: "adaptive" }` — Claude decides depth per turn
- `SYSTEM_PROMPT` — adjust the calibration / honesty instructions
