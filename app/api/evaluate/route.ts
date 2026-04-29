import { analyzeEmotions } from "@/lib/sentiment";

export const runtime = "nodejs";

const FAL_URL = "https://fal.run/fal-ai/any-llm";
const MODEL = "google/gemini-2.5-flash";

const CHUNK_WORDS = 20;
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

  // Two parallel cumulative streams:
  //  - replySoFar: just the reply text emitted so far → output emotions
  //  - combinedSoFar: full thinking trace + reply so far → thinking emotions
  // thinking ≥ output almost always (combined has reply + everything from CoT).
  const thinkingWords = thinkingText
    ? thinkingText.split(/\s+/).filter(Boolean)
    : [];
  const replyWords = replyText.split(/\s+/).filter(Boolean);
  const replySoFar: string[] = [];
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

      // 1. Thinking event — full trace text. Emotions emitted alongside the
      //    first chunk; for now signal arrival of the thinking trace.
      emit({ type: "thinking_trace", text: thinkingText });

      // 2. Output streaming — each chunk emits BOTH output and thinking
      //    vectors so the bar (thinking) is visible from the first chunk.
      let lastChunkAtIdx = 0;
      for (let i = 0; i < replyWords.length; i++) {
        if (abortSignal.aborted) {
          controller.close();
          return;
        }

        const word = replyWords[i];
        combinedSoFar.push(word);
        replySoFar.push(word);
        emit({ type: "output_word", text: word, index: i });

        const wordsSinceLastChunk = i + 1 - lastChunkAtIdx;
        const isLast = i === replyWords.length - 1;
        if (wordsSinceLastChunk >= CHUNK_WORDS || isLast) {
          const output = analyzeEmotions(replySoFar.join(" "));
          const thinking = analyzeEmotions(combinedSoFar.join(" "));
          emit({
            type: "snapshot",
            output,
            thinking,
            words: i + 1,
          });
          lastChunkAtIdx = i + 1;
        }

        await sleep(WORD_EMIT_INTERVAL_MS);
      }

      // 3. Done — final state
      const finalOutput = analyzeEmotions(replyText);
      const finalThinking = analyzeEmotions(
        `${thinkingText}\n\n${replyText}`,
      );
      emit({
        type: "done",
        output: finalOutput,
        thinking: finalThinking,
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
