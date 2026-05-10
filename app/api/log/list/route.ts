// Enumerate the log files that actually exist on disk right now,
// returning their mtime so the client can default-select whichever
// was most recently written. Files that don't exist yet are simply
// omitted — the dropdown stays in sync with the pipeline.

import { stat } from "node:fs/promises";
import { ALLOWED_LOG_FILES, resolveLogPath } from "@/lib/monitor-paths";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface LogEntry {
  file: string;
  sizeBytes: number;
  mtimeMs: number;
}

export async function GET(): Promise<Response> {
  const entries: LogEntry[] = [];
  for (const rel of ALLOWED_LOG_FILES) {
    try {
      const abs = resolveLogPath(rel);
      const st = await stat(abs);
      if (!st.isFile()) continue;
      entries.push({
        file: rel,
        sizeBytes: st.size,
        mtimeMs: st.mtimeMs,
      });
    } catch {
      // Missing file — skip. We could return it with mtimeMs=0 so the
      // UI can grey it out, but a missing log usually just means that
      // corpus hasn't been started, and showing nothing is cleaner.
    }
  }
  // Most-recently-touched first; the UI picks entries[0] as default.
  entries.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return Response.json({ entries });
}
