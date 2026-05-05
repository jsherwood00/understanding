import { classifyText, isClassifiable, zeros } from "@/lib/emotion-classifier";

export const runtime = "nodejs";
// First call has to download + initialize ~1.4GB of deberta-v3-large
// weights; subsequent calls are warm. Bumped from 30s.
export const maxDuration = 120;

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

  try {
    const emotions = await classifyText(text);
    return Response.json({ emotions }, { status: 200 });
  } catch {
    return Response.json({ emotions: zeros() }, { status: 200 });
  }
}
