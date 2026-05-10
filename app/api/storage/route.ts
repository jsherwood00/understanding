// Sum bytes recursively under each data/<corpus>/activations/ tree.
//
// Walking is done with a hand-rolled iterative stack rather than a
// recursive function so we don't blow the JS stack on very deep
// hierarchies. It's also resilient to per-entry errors (a file
// vanishing mid-walk during an active extract is normal): we swallow
// the error for that node and keep summing.

import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import {
  CORPORA,
  STORAGE_CAP_BYTES,
  activationsDir,
  type Corpus,
} from "@/lib/monitor-paths";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function dirSizeBytes(root: string): Promise<number> {
  let total = 0;
  const stack: string[] = [root];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      // Missing root or transient failure — treat as zero contribution.
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (e.isFile()) {
        try {
          const st = await stat(full);
          total += st.size;
        } catch {
          // File got deleted between readdir and stat (extract.py
          // rotates atomically); ignore.
        }
      }
    }
  }
  return total;
}

export async function GET(): Promise<Response> {
  const perCorpus: Record<Corpus, number> = {
    no_thinking: 0,
    thinking: 0,
    neutral: 0,
  };

  // Walks are independent — run them concurrently to keep the poll
  // snappy on a cold disk.
  const results = await Promise.all(
    CORPORA.map(async (c) => [c, await dirSizeBytes(activationsDir(c))] as const),
  );
  for (const [c, n] of results) perCorpus[c] = n;

  const totalBytes = Object.values(perCorpus).reduce((a, b) => a + b, 0);

  return Response.json({
    perCorpus,
    totalBytes,
    capBytes: STORAGE_CAP_BYTES,
    pct: STORAGE_CAP_BYTES > 0 ? (totalBytes / STORAGE_CAP_BYTES) * 100 : 0,
  });
}
