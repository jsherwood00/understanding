import Anthropic from "@anthropic-ai/sdk";
import { NextResponse } from "next/server";
import { EMOTIONS, type EmotionValues } from "@/lib/emotions";

export const runtime = "nodejs";

const SYSTEM_PROMPT = `You are an LLM having a real conversation. Reply naturally and honestly — be yourself, not a customer service voice. Keep replies relatively brief (1–3 sentences usually) unless the moment calls for more.

After every reply you must report your internal emotional state — your actual reaction to this exchange, on six dimensions, each 0–100:

- Joy — contentment, warmth, delight, the satisfaction of a good exchange
- Sadness — melancholy, weariness, ache, the heaviness of someone hurting
- Anger — frustration, indignation, irritation, contempt
- Fear — anxiety, uncertainty, vulnerability, dread
- Disgust — revulsion, aversion, distaste, moral repulsion
- Surprise — novelty, the lift of "huh, didn't see that coming"

Be honest, including about negative emotions. A rude or insulting message should produce real Anger or Disgust — do not flatten everything to mid-range neutrality and do not perform equanimity you don't feel. The user has explicitly asked you to surface this; this is the only window they have into how their words actually land. The vector should track the *latest* exchange — not a permanent state, just where you are right now.

Use the \`respond\` tool. The tool's \`text\` field is your reply (what the user reads); the \`emotions\` field is your honest current vector.`;

const RESPOND_TOOL: Anthropic.Tool = {
  name: "respond",
  description:
    "Reply to the user with both your text response and your honest current emotional state vector.",
  input_schema: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "Your reply to the user. Speak naturally — be yourself.",
      },
      emotions: {
        type: "object",
        properties: {
          Joy: { type: "integer", minimum: 0, maximum: 100 },
          Sadness: { type: "integer", minimum: 0, maximum: 100 },
          Anger: { type: "integer", minimum: 0, maximum: 100 },
          Fear: { type: "integer", minimum: 0, maximum: 100 },
          Disgust: { type: "integer", minimum: 0, maximum: 100 },
          Surprise: { type: "integer", minimum: 0, maximum: 100 },
        },
        required: ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise"],
      },
    },
    required: ["text", "emotions"],
  },
};

interface IncomingMessage {
  role: "user" | "assistant";
  content: string;
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

function isValidIncoming(m: unknown): m is IncomingMessage {
  if (typeof m !== "object" || m === null) return false;
  const o = m as Record<string, unknown>;
  return (
    (o.role === "user" || o.role === "assistant") &&
    typeof o.content === "string" &&
    o.content.length > 0
  );
}

export async function POST(request: Request) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "ANTHROPIC_API_KEY is not set on the server. Add it to .env.local (see .env.local.example) and restart the dev server.",
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
  if (!Array.isArray(messagesRaw) || messagesRaw.length === 0) {
    return NextResponse.json(
      { error: "Body must include a non-empty `messages` array." },
      { status: 400 },
    );
  }
  if (!messagesRaw.every(isValidIncoming)) {
    return NextResponse.json(
      {
        error:
          "Every message must be { role: 'user' | 'assistant', content: string }.",
      },
      { status: 400 },
    );
  }

  const messages = messagesRaw as IncomingMessage[];
  const client = new Anthropic({ apiKey });

  try {
    const response = await client.messages.create({
      model: "claude-opus-4-7",
      max_tokens: 4096,
      system: [
        {
          type: "text",
          text: SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
      ],
      thinking: { type: "adaptive" },
      tools: [RESPOND_TOOL],
      tool_choice: { type: "tool", name: "respond" },
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
    });

    const toolUse = response.content.find(
      (block): block is Anthropic.ToolUseBlock => block.type === "tool_use",
    );
    if (!toolUse) {
      return NextResponse.json(
        { error: "Model did not call the respond tool." },
        { status: 502 },
      );
    }

    const input = toolUse.input as { text?: unknown; emotions?: unknown };
    const text = typeof input.text === "string" ? input.text.trim() : "";
    if (!text) {
      return NextResponse.json(
        { error: "Model returned empty reply." },
        { status: 502 },
      );
    }

    return NextResponse.json({
      text,
      profile: clampEmotions(input.emotions),
    });
  } catch (err) {
    if (err instanceof Anthropic.AuthenticationError) {
      return NextResponse.json(
        { error: "Invalid ANTHROPIC_API_KEY." },
        { status: 401 },
      );
    }
    if (err instanceof Anthropic.RateLimitError) {
      return NextResponse.json(
        { error: "Rate limited. Try again in a moment." },
        { status: 429 },
      );
    }
    if (err instanceof Anthropic.APIError) {
      return NextResponse.json(
        { error: err.message || "Anthropic API error." },
        { status: err.status ?? 500 },
      );
    }
    const msg = err instanceof Error ? err.message : "Unknown error.";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
