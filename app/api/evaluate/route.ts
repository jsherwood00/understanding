import { NextResponse } from "next/server";
import { existsSync } from "node:fs";
import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { EMOTIONS, type EmotionValues } from "@/lib/emotions";

export const runtime = "nodejs";

const TURNS_DIR = path.join(process.cwd(), "runtime", "turns");
const POLL_INTERVAL_MS = 250;
const TIMEOUT_MS = 60_000;

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

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function POST(request: Request) {
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
          "Body must include a non-empty `messages` array of { role: 'user'|'assistant', content: string }.",
      },
      { status: 400 },
    );
  }

  await mkdir(TURNS_DIR, { recursive: true });

  const id = crypto.randomUUID();
  const reqPath = path.join(TURNS_DIR, `${id}.req.json`);
  const resPath = path.join(TURNS_DIR, `${id}.res.json`);

  await writeFile(
    reqPath,
    JSON.stringify(
      {
        id,
        messages: messagesRaw,
        createdAt: new Date().toISOString(),
      },
      null,
      2,
    ),
  );

  const start = Date.now();
  while (Date.now() - start < TIMEOUT_MS) {
    if (existsSync(resPath)) {
      try {
        const data = await readFile(resPath, "utf8");
        const parsed = JSON.parse(data) as {
          text?: unknown;
          profile?: unknown;
        };
        await Promise.allSettled([unlink(reqPath), unlink(resPath)]);
        const text = typeof parsed.text === "string" ? parsed.text.trim() : "";
        if (!text) {
          return NextResponse.json(
            { error: "Response file present but missing/empty `text`." },
            { status: 502 },
          );
        }
        return NextResponse.json({
          text,
          profile: clampEmotions(parsed.profile),
        });
      } catch (err) {
        await Promise.allSettled([unlink(reqPath), unlink(resPath)]);
        return NextResponse.json(
          {
            error: `Failed to parse response file: ${err instanceof Error ? err.message : String(err)}`,
          },
          { status: 502 },
        );
      }
    }
    await sleep(POLL_INTERVAL_MS);
  }

  // Timeout — leave the request file in place so a slow loop can still pick it up later.
  return NextResponse.json(
    {
      error:
        "Timed out waiting for a response. Make sure the Claude Code response loop is running (see README → “Run the response loop”).",
    },
    { status: 504 },
  );
}
