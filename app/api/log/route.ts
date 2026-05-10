// Tail the last ~80 lines of one of the whitelisted pipeline logs.
//
// Two correctness points worth flagging:
//
// 1. We never read the whole file. Some `extract.log`s grow into the
//    hundreds of MB over multi-day runs, and pulling that across the
//    process boundary on every 2s poll would torch the dev server.
//    Instead we stat the file, seek to (size - MAX_TAIL_BYTES), and
//    read just that window.
//
// 2. The line count we return is approximate-but-fast. Counting the
//    bytes already read gives us a lower-bound on the true line count;
//    it's only used for the small "(N lines)" badge in the UI, not
//    for anything correctness-sensitive.

import { open, stat } from "node:fs/promises";
import type { NextRequest } from "next/server";
import { resolveLogPath } from "@/lib/monitor-paths";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_TAIL_BYTES = 256 * 1024;
const DEFAULT_LINES = 80;

export async function GET(request: NextRequest): Promise<Response> {
  const fileParam = request.nextUrl.searchParams.get("file");
  const linesParam = request.nextUrl.searchParams.get("lines");
  const requestedLines = linesParam ? Number(linesParam) : DEFAULT_LINES;
  const linesN =
    Number.isFinite(requestedLines) && requestedLines > 0
      ? Math.min(Math.floor(requestedLines), 1000)
      : DEFAULT_LINES;

  if (!fileParam) {
    return Response.json(
      { error: "missing ?file= query param" },
      { status: 400 },
    );
  }

  let absPath: string;
  try {
    absPath = resolveLogPath(fileParam);
  } catch {
    return Response.json(
      { error: "log path not allowed" },
      { status: 400 },
    );
  }

  let st;
  try {
    st = await stat(absPath);
  } catch {
    // File doesn't exist yet (pipeline hasn't run for this corpus).
    // 200 + empty content is friendlier for the UI than a 404 — the
    // log selector keeps showing the dropdown choice.
    return Response.json({
      file: fileParam,
      content: "",
      lineCount: 0,
      sizeBytes: 0,
      mtimeMs: 0,
      truncated: false,
    });
  }

  const size = st.size;
  const start = Math.max(0, size - MAX_TAIL_BYTES);
  const length = size - start;
  const truncated = start > 0;

  let buf: Buffer;
  if (length === 0) {
    buf = Buffer.alloc(0);
  } else {
    const handle = await open(absPath, "r");
    try {
      const dst = Buffer.alloc(length);
      await handle.read(dst, 0, length, start);
      buf = dst;
    } finally {
      await handle.close();
    }
  }

  // Decode as utf-8. If we sliced into the middle of a multi-byte
  // character at the start, we'll lose at most one truncated rune at
  // the very top of the window — fine, since we then drop the first
  // (possibly partial) line below.
  let text = buf.toString("utf8");
  if (truncated) {
    // Drop the first (possibly partial) line.
    const nl = text.indexOf("\n");
    if (nl >= 0) text = text.slice(nl + 1);
  }

  // Take the last `linesN` lines, ignoring a trailing newline if any.
  const trimmed = text.endsWith("\n") ? text.slice(0, -1) : text;
  const allLines = trimmed.length === 0 ? [] : trimmed.split("\n");
  const tail = allLines.slice(-linesN);

  return Response.json({
    file: fileParam,
    content: tail.join("\n"),
    lineCount: allLines.length,
    sizeBytes: size,
    mtimeMs: st.mtimeMs,
    truncated,
  });
}
