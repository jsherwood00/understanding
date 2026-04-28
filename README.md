# Understanding

A week-1 UI prototype for an LLM emotion visualization tool — *what the model feels beneath what it says*.

Two-pane interface: an EQ-style bar chart of six "internal" emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right. Everything is faked. There is no API call, no model, no backend. Each canned assistant response carries a hardcoded emotion profile; the bars animate toward that profile while the response streams character-by-character. Between exchanges the bars drift gently to keep the panel feeling alive.

A second "surface emotion" tier sits below as a `coming soon` placeholder, stubbed for the eventual two-layer view.

## Stack

- Next.js 16 + React 19 (App Router)
- TypeScript, Tailwind CSS v4
- `next/font` with Fraunces (serif) + Inter (sans)
- No database, no external API — fully client-side state

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Build

```bash
npm run build
npm start
```

## Deploy

Push to a Vercel project and it deploys as-is — no env vars, no extra config.

## Layout

```
app/
  layout.tsx        Root layout, fonts, metadata
  page.tsx          Server shell (header + workspace)
  globals.css       Theme tokens, base styles, animations
components/
  Header.tsx        Top bar
  Workspace.tsx     Client wrapper — state + animation orchestration
  EmotionPanel.tsx  Left pane (bars + surface emotion preview)
  ChatPane.tsx      Right pane (transcript + input + streaming)
lib/
  emotions.ts       Types, colors, baseline, stepToward + drift helpers
  responses.ts      Six canned responses with emotion profiles
```

## Tuning knobs

In `components/Workspace.tsx`:

- `TYPING_DELAY_MS` — pause before the assistant starts streaming (default 800)
- `STREAM_INTERVAL_MS` — per-character delay (default 20)
- `ANIMATION_TICK_MS` — bar interpolation cadence while streaming (default 300)
- `DRIFT_TICK_MS` — ambient drift cadence at idle (default 1000)

Response profiles live in `lib/responses.ts`. Each profile is a `{ Joy, Sadness, Anger, Fear, Disgust, Surprise }` map of target values 0–100.
