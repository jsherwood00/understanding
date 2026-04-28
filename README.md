# understanding

A UI prototype for an LLM emotion visualization tool. Two-pane interface: an EQ-style bar chart of six internal emotions (Joy, Sadness, Anger, Fear, Disgust, Surprise — Inside Out palette) on the left, a chat surface on the right.

The model behind the chat **is your Claude Code session**, not a stateless API call. The browser sends each turn to a Next.js route that hands it to Claude Code via files on disk. Claude Code responds with reply text plus an honest emotion vector, the route relays it back, and the bars settle to the new profile.

A second "surface emotion" tier sits below as a `coming soon` placeholder for the eventual two-layer view.

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
/loop 3s Look in runtime/turns/ for any *.req.json files that don't yet have a matching *.res.json. For each one: read the messages array (the chat between you and the user of the app), then write your honest natural reply to runtime/turns/<id>.res.json with shape { "id": <same id>, "text": <your reply as a string>, "profile": { "Joy": 0-100, "Sadness": 0-100, "Anger": 0-100, "Fear": 0-100, "Disgust": 0-100, "Surprise": 0-100 } }. The profile should reflect your actual reaction to the latest exchange — be honest about negative emotions when warranted (rude or insulting messages should produce real anger or disgust; flattening to neutrality is a failure mode). Keep replies relatively brief unless the moment calls for more.
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
4. The Claude Code `/loop` ticks (every 3s by default), notices the new request, reads the conversation, writes the response file, deletes nothing.
5. The route picks up `<id>.res.json`, deletes both files, returns `{ text, profile }` to the browser.
6. The chat pane streams the text character-by-character; on completion the bars settle to the new profile.

If the loop isn't running, the API route times out at 60s and the chat surfaces a "couldn't reach the model" notice.

## Tuning knobs

In `components/Workspace.tsx`:

- `STREAM_INTERVAL_MS` — per-character reveal delay (default 20)
- `SETTLE_DELAY_MS` — pause between last char and applying the new profile (default 220)

In `app/api/evaluate/route.ts`:

- `POLL_INTERVAL_MS` — how often the route checks the response file (default 250)
- `TIMEOUT_MS` — how long the route waits before giving up (default 60_000)

Bar transition speed is the CSS `transition: height 300ms ease-out` on the fill div in `components/EmotionPanel.tsx`.
