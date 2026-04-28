import { type EmotionValues } from "@/lib/emotions";
import { analyzeEmotions, deriveThinking } from "@/lib/sentiment";

export const runtime = "nodejs";

const FAL_URL = "https://fal.run/fal-ai/any-llm";
const MODEL = "google/gemini-2.5-flash";

const CHUNK_WORDS = 20;
const ANALYSIS_WINDOW_WORDS = 30;
const WORD_EMIT_INTERVAL_MS = 25;

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
    const label = m.role === "user" ? "User" : "Assistant";
    lines.push(`${label}: ${m.content}`);
  }
  lines.push("Assistant:");
  return lines.join("\n\n");
}

function normalizeText(raw: unknown): string {
  if (typeof raw !== "string") return "";
  return raw
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, "\\")
    .trim();
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function jsonResponse(payload: unknown, status: number) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function zeros(): EmotionValues {
  return {
    Joy: 0,
    Sadness: 0,
    Anger: 0,
    Fear: 0,
    Disgust: 0,
    Surprise: 0,
  };
}

export async function POST(request: Request) {
  const apiKey = process.env.FAL_API_KEY;
  if (!apiKey) {
    return jsonResponse(
      { error: "FAL_API_KEY is not set on the server." },
      500,
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body." }, 400);
  }

  const messagesRaw = (body as { messages?: unknown })?.messages;
  if (
    !Array.isArray(messagesRaw) ||
    messagesRaw.length === 0 ||
    !messagesRaw.every(isValidIncoming)
  ) {
    return jsonResponse(
      {
        error:
          "Body must include a non-empty `messages` array of {role, content}.",
      },
      400,
    );
  }
  const messages = messagesRaw as IncomingMessage[];

  let falData: { output?: string; reasoning?: string; error?: string };
  try {
    const res = await fetch(FAL_URL, {
      method: "POST",
      headers: {
        Authorization: `Key ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        prompt: formatPrompt(messages),
        reasoning: true,
      }),
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return jsonResponse(
        { error: `fal.ai returned ${res.status}: ${errText.slice(0, 300)}` },
        502,
      );
    }
    falData = await res.json();
  } catch (err) {
    return jsonResponse(
      {
        error: `Failed to reach fal.ai: ${err instanceof Error ? err.message : String(err)}`,
      },
      502,
    );
  }

  if (falData.error) {
    return jsonResponse({ error: `fal.ai error: ${falData.error}` }, 502);
  }

  const replyText = (falData.output ?? "").trim();
  const thinkingText = normalizeText(falData.reasoning);

  if (!replyText) {
    return jsonResponse({ error: "Model returned an empty reply." }, 502);
  }

  // Combined word stream: thinking words first (already complete), then
  // reply words (streamed). Output emotions analyze a rolling 30-word window
  // across the COMBINED stream, every 20 reply words.
  const thinkingWords = thinkingText
    ? thinkingText.split(/\s+/).filter(Boolean)
    : [];
  const replyWords = replyText.split(/\s+/).filter(Boolean);
  const combinedSoFar = [...thinkingWords]; // append reply words as they emit

  const encoder = new TextEncoder();
  const abortSignal = request.signal;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (event: Record<string, unknown>) => {
        if (abortSignal.aborted) return;
        try {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
          );
        } catch {
          // controller closed; ignore
        }
      };

      // 1. Thinking event — text only. Emotions intentionally zero; the
      //    "thinking" vector is derived from the final output at done.
      emit({ type: "thinking", text: thinkingText, emotions: zeros() });

      // 2. Output streaming
      let lastChunkAtIdx = 0;
      for (let i = 0; i < replyWords.length; i++) {
        if (abortSignal.aborted) {
          controller.close();
          return;
        }

        const word = replyWords[i];
        combinedSoFar.push(word);
        emit({ type: "output_word", text: word, index: i });

        const wordsSinceLastChunk = i + 1 - lastChunkAtIdx;
        const isLast = i === replyWords.length - 1;
        if (wordsSinceLastChunk >= CHUNK_WORDS || isLast) {
          const windowStart = Math.max(
            0,
            combinedSoFar.length - ANALYSIS_WINDOW_WORDS,
          );
          const window = combinedSoFar.slice(windowStart).join(" ");
          const emotions = analyzeEmotions(window);
          emit({
            type: "output_emotions",
            emotions,
            words: i + 1,
          });
          lastChunkAtIdx = i + 1;
        }

        await sleep(WORD_EMIT_INTERVAL_MS);
      }

      // 3. Done — final emotions
      const finalCombined = `${thinkingText}\n\n${replyText}`;
      const outputEmotions = analyzeEmotions(finalCombined);
      const thinkingEmotions = deriveThinking(outputEmotions);
      emit({
        type: "done",
        outputEmotions,
        thinkingEmotions,
        fullText: replyText,
        fullThinking: thinkingText,
      });
      controller.close();
    },
    cancel() {
      // Client aborted; nothing to clean up beyond the stream itself.
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
