import { classifyText, isClassifiable, zeros } from "@/lib/emotion-classifier";

export const runtime = "nodejs";
// First call has to download + initialize ~1.4GB of deberta-v3-large
// weights; subsequent calls are warm. Bumped from 30s. In production
// (MODAL_DEBERTA_URL set), we proxy to Modal and never load the model
// here, so a short timeout is fine — but keep the headroom in case
// Modal cold-start adds latency.
export const maxDuration = 120;

const MODAL_DEBERTA_URL = process.env.MODAL_DEBERTA_URL;

export async function POST(request: Request) {
  let body: { text?: unknown };
  try {
    body = await request.json();
  } catch {
    return Response.json({ emotions: zeros() }, { status: 200 });
  }

  const text = typeof body.text === "string" ? body.text : "";

  // Hard pre-filter — never call the model on garbage.
  if (!isClassifiable(text)) {
    return Response.json({ emotions: zeros() }, { status: 200 });
  }

  // Honor client aborts.
  if (request.signal.aborted) {
    return Response.json({ emotions: zeros() }, { status: 200 });
  }

  // Production path: proxy to Modal-hosted deberta. The Modal endpoint
  // returns the same {emotions: {Joy: int, ...}} contract our frontend
  // already expects, so this is a pure URL swap.
  if (MODAL_DEBERTA_URL) {
    try {
      const res = await fetch(MODAL_DEBERTA_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: request.signal,
      });
      if (!res.ok) {
        return Response.json({ emotions: zeros() }, { status: 200 });
      }
      const data = await res.json();
      return Response.json(data, { status: 200 });
    } catch {
      return Response.json({ emotions: zeros() }, { status: 200 });
    }
  }

  // Local-dev path: load deberta via @huggingface/transformers in this
  // Node process. ~1.4GB download on first call.
  try {
    const emotions = await classifyText(text);
    return Response.json({ emotions }, { status: 200 });
  } catch {
    return Response.json({ emotions: zeros() }, { status: 200 });
  }
}
