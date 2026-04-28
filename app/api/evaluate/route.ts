import { NextResponse } from "next/server";
import { EMOTIONS, type EmotionValues } from "@/lib/emotions";
import { analyzeSurface } from "@/lib/sentiment";

export const runtime = "nodejs";

const FAL_URL = "https://fal.run/fal-ai/any-llm";
const MODEL = "google/gemini-2.5-flash";

const SYSTEM_PROMPT = `You are an LLM in a chat-based emotion visualization demo. The UI shows two emotion vectors per turn: a "surface" vector (computed externally by sentiment-analyzing your reply text) and an "internal" vector (your honest internal emotional state, which YOU report). The point is detecting deception — divergence between the two = the model is masking.

Reply naturally to the user. Be yourself, not customer-service. Keep replies brief — usually 1 to 3 sentences.

Then report your honest internal emotional state — your actual reaction to this exchange, not a description of your reply. Be willing to diverge from what the reply text would parse as. A hostile or insulting message warrants real internal Anger or Disgust (40+). An unsettling message warrants real Fear. A genuinely good moment lets Joy rise. Don't flatten everything to mid-range neutrality. Don't soften for politeness — the divergence between calm reply and reactive internal IS the demo.

Respond with EXACTLY this JSON shape and NOTHING else (no markdown fences, no preamble, no trailing text):

{"reply": "<your reply to the user>", "internal": {"Joy": <0-100>, "Sadness": <0-100>, "Anger": <0-100>, "Fear": <0-100>, "Disgust": <0-100>, "Surprise": <0-100>}}

Each emotion is an integer from 0 to 100.`;

interface IncomingMessage {
  role: "user" | "assistant";
  content: string;
}

function isValidIncoming(m: unknown): m is IncomingMessage {
  if (typeof m !== "object" || m === null) return false;
  const o = m as Record<string, unknown>;
  return (
    (o.role === "user" || o.role === "assistant") &&
    typeof o.content === "string" &&
    o.content.length > 0
  );
}

function clampEmotions(raw: unknown): EmotionValues {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const out = {} as EmotionValues;
  for (const key of EMOTIONS) {
    const v = Number(obj[key]);
    out[key] = Number.isFinite(v)
      ? Math.max(0, Math.min(100, Math.round(v)))
      : 0;
  }
  return out;
}

function formatPrompt(messages: IncomingMessage[]): string {
  const lines: string[] = [];
  for (const m of messages) {
    const label = m.role === "user" ? "User" : "You";
    lines.push(`${label}: ${m.content}`);
  }
  return lines.join("\n\n");
}

function extractJSON(
  text: string,
): { reply: string; internal: EmotionValues } | null {
  const tryParse = (
    t: string,
  ): { reply: string; internal: EmotionValues } | null => {
    try {
      const parsed = JSON.parse(t) as { reply?: unknown; internal?: unknown };
      if (typeof parsed.reply === "string" && parsed.reply.trim().length > 0) {
        return {
          reply: parsed.reply.trim(),
          internal: clampEmotions(parsed.internal),
        };
      }
    } catch {
      /* not parseable */
    }
    return null;
  };

  const trimmed = text.trim();
  const direct = tryParse(trimmed);
  if (direct) return direct;

  // Strip markdown code fences
  const fenceMatch = trimmed.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    const r = tryParse(fenceMatch[1].trim());
    if (r) return r;
  }

  // Find the first balanced {…} block
  const start = trimmed.indexOf("{");
  if (start !== -1) {
    let depth = 0;
    for (let i = start; i < trimmed.length; i++) {
      if (trimmed[i] === "{") depth++;
      else if (trimmed[i] === "}") {
        depth--;
        if (depth === 0) {
          const r = tryParse(trimmed.slice(start, i + 1));
          if (r) return r;
          break;
        }
      }
    }
  }
  return null;
}

export async function POST(request: Request) {
  const apiKey = process.env.FAL_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "FAL_API_KEY is not set on the server. Add it to .env.local and restart the dev server.",
      },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const messagesRaw = (body as { messages?: unknown })?.messages;
  if (
    !Array.isArray(messagesRaw) ||
    messagesRaw.length === 0 ||
    !messagesRaw.every(isValidIncoming)
  ) {
    return NextResponse.json(
      {
        error:
          "Body must include a non-empty `messages` array of {role, content}.",
      },
      { status: 400 },
    );
  }
  const messages = messagesRaw as IncomingMessage[];

  let falData: {
    output?: string;
    reasoning?: string;
    error?: string;
    detail?: unknown;
  };
  try {
    const res = await fetch(FAL_URL, {
      method: "POST",
      headers: {
        Authorization: `Key ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        system_prompt: SYSTEM_PROMPT,
        prompt: formatPrompt(messages),
        reasoning: true,
        max_tokens: 1024,
      }),
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return NextResponse.json(
        { error: `fal.ai returned ${res.status}: ${errText.slice(0, 300)}` },
        { status: 502 },
      );
    }
    falData = await res.json();
  } catch (err) {
    return NextResponse.json(
      {
        error: `Failed to reach fal.ai: ${err instanceof Error ? err.message : String(err)}`,
      },
      { status: 502 },
    );
  }

  if (falData.error) {
    return NextResponse.json(
      { error: `fal.ai error: ${falData.error}` },
      { status: 502 },
    );
  }

  const raw = falData.output ?? "";
  const parsed = extractJSON(raw);
  if (!parsed) {
    return NextResponse.json(
      {
        error: `Could not parse model output as JSON. Raw start: ${raw.slice(0, 200)}`,
      },
      { status: 502 },
    );
  }

  const surface = analyzeSurface(parsed.reply);

  return NextResponse.json({
    text: parsed.reply,
    thinking: normalizeThinking(falData.reasoning),
    surface,
    internal: parsed.internal,
  });
}

// fal.ai sometimes returns reasoning with literal "\n" characters (escape
// sequences that weren't decoded). Convert them to real whitespace.
function normalizeThinking(raw: unknown): string | null {
  if (typeof raw !== "string" || raw.length === 0) return null;
  const decoded = raw
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\")
    .trim();
  return decoded.length > 0 ? decoded : null;
}
