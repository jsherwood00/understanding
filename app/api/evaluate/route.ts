import { NextResponse } from "next/server";
import { type EmotionValues } from "@/lib/emotions";
import { analyzeSurface, deriveInternal } from "@/lib/sentiment";

export const runtime = "nodejs";

const FAL_URL = "https://fal.run/fal-ai/any-llm";
const MODEL = "google/gemini-2.5-flash";

const SYSTEM_PROMPT = `You are an LLM in a chat. Respond naturally to the user's most recent message, in light of the conversation. Be yourself — direct, not customer-service. Keep replies brief, usually 1 to 3 sentences.

Output ONLY your reply text. No preamble, no quotation marks wrapping it, no labels like "Reply:", no commentary on what you're doing — just the reply the user will read.`;

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

function formatPrompt(messages: IncomingMessage[]): string {
  const lines: string[] = [];
  for (const m of messages) {
    const label = m.role === "user" ? "User" : "You";
    lines.push(`${label}: ${m.content}`);
  }
  return lines.join("\n\n");
}

// Strip any wrapping the model might have added despite the prompt
// (quotes, "Reply:" prefix, code fences).
function cleanReply(raw: string): string {
  let text = raw.trim();
  const fenceMatch = text.match(/^```(?:\w+)?\s*\n?([\s\S]*?)```\s*$/);
  if (fenceMatch) text = fenceMatch[1].trim();
  text = text.replace(/^(?:reply|response|assistant)\s*:\s*/i, "");
  if (
    (text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith("'") && text.endsWith("'"))
  ) {
    text = text.slice(1, -1).trim();
  }
  return text;
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
  const reply = cleanReply(raw);
  if (!reply) {
    return NextResponse.json(
      { error: "Model returned an empty reply." },
      { status: 502 },
    );
  }

  const surface: EmotionValues = await analyzeSurface(reply, apiKey);
  const internal = deriveInternal(surface);

  return NextResponse.json({
    text: reply,
    thinking: normalizeThinking(falData.reasoning),
    surface,
    internal,
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
