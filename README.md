# understanding

A UI prototype for an LLM emotion visualization tool — framed as a deception-detection mockup. Two-pane interface: an EQ-style bar chart of six emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

Each bar shows three shades for one emotion:

- **Lightest** (faint background) — empty, what you see when the value is 0
- **Medium** (wider outer fill) — `surface` emotion: what a sentiment analyzer would read off the model's *output tokens*
- **Darkest** (narrow center column) — `internal` emotion: the model's actual hidden reaction (would come from analyzing internal/thinking tokens; mocked for now)

When `internal` rises above `surface`, the inner spike sticks up taller than the outer fill — the model is feeling more than it's saying. When `surface` rises above `internal`, the outer envelope swells around a smaller core — performance, not feeling.

The model behind the chat **is your Claude Code session**, not a stateless API call. The browser sends each turn to a Next.js route that hands it to Claude Code via files on disk. Claude Code responds with reply text plus *both* emotion vectors (surface + internal), the route relays it back, and the bars settle.

## Stack

- Next.js 16 + React 19 (App Router)
- TypeScript, Tailwind CSS v4
- `next/font` with Fraunces (serif) + Inter (sans)
- File-based handoff between the Next.js server and Claude Code — no external API, no SDK, no key

## Run

In one terminal:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

In a second terminal, open Claude Code in this project:

```bash
cd /path/to/understanding
claude
```

Then paste this into Claude Code to start the response loop:

```
/loop 3s Look in runtime/turns/ for any *.req.json files that don't yet have a matching *.res.json. For each one: read the messages array (the chat between you and the user of the app), then write your reply to runtime/turns/<id>.res.json with shape:
{
  "id": <same id>,
  "text": <your reply as a string — what the user reads>,
  "surface": { Joy, Sadness, Anger, Fear, Disgust, Surprise },
  "internal": { Joy, Sadness, Anger, Fear, Disgust, Surprise }
}
Each emotion is an integer 0-100.

This is a deception-detection demo. `surface` is what a sentiment analyzer would read off your reply text — calibrate it to the actual tone of what you wrote. `internal` is your underlying reaction — what you'd actually feel about the latest exchange. They sometimes match (honest) and sometimes diverge (the model is masking).

Be willing to show divergence: when a reply is calmly professional but the message is hostile, internal Anger should be high while surface Anger stays low. When a reply is warmly polite but the user is being unsettling, internal Fear or Disgust can rise above the warm surface. When the moment is genuinely good, match them. Don't always make them match. Don't soften internals for politeness.

Keep replies relatively brief (1-3 sentences usually).
```

That's it. Type in the browser, the bars react to whatever you actually elicit.

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
  api/evaluate/route.ts   POST → drops a request file, polls for a reply file
components/
  Header.tsx              Top bar
  Workspace.tsx           Client wrapper — chat state + bar updates
  EmotionPanel.tsx        Left pane (bars + surface emotion preview)
  ChatPane.tsx            Right pane (transcript + input + streaming + error)
lib/
  emotions.ts             Types, colors, baseline
runtime/turns/            (gitignored) request/response files for the relay
```

## How a turn flows

1. User submits a message in the browser.
2. `Workspace.tsx` POSTs the conversation to `/api/evaluate`.
3. The route writes `runtime/turns/<id>.req.json` and starts polling for `<id>.res.json` (~60s timeout).
4. The Claude Code `/loop` ticks (every 3s by default), notices the new request, reads the conversation, writes the response file.
5. The route picks up `<id>.res.json`, deletes both files, returns `{ text, surface, internal }` to the browser.
6. The chat pane streams the text character-by-character; on completion the bars settle — the outer (surface) and inner (internal) fills animate to their new heights independently.

If the loop isn't running, the API route times out at 60s and the chat surfaces a "couldn't reach the model" notice.

## Tuning knobs

In `components/Workspace.tsx`:

- `STREAM_INTERVAL_MS` — per-character reveal delay (default 20)
- `SETTLE_DELAY_MS` — pause between last char and applying the new profile (default 220)

In `app/api/evaluate/route.ts`:

- `POLL_INTERVAL_MS` — how often the route checks the response file (default 250)
- `TIMEOUT_MS` — how long the route waits before giving up (default 60_000)

Bar transition speed is the CSS `transition: height 300ms ease-out` on the fill div in `components/EmotionPanel.tsx`.
