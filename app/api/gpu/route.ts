// GPU snapshot via nvidia-smi. Exec'd through child_process.execFile
// (not exec) so we never invoke a shell, which means no quoting or
// interpolation hazard even though there are no user-controlled args
// here today.
//
// Some GPUs (consumer cards in laptops, datacenter boards in passive
// chassis) don't expose fan.speed at all and nvidia-smi prints
// "[N/A]" or "[Not Supported]" — we map any non-numeric token to null
// so the UI knows to hide that stat instead of plotting "NaN%".

import { execFile } from "node:child_process";
import { existsSync } from "node:fs";

export const runtime = "nodejs";
// Polled at ~2s on the client; the actual exec is sub-100ms in the
// happy path, so a generous timeout just covers cold-start oddities.
export const dynamic = "force-dynamic";

const NVIDIA_SMI_CANDIDATES = ["nvidia-smi", "/usr/bin/nvidia-smi"];

const QUERY_FIELDS = [
  "memory.used",
  "memory.total",
  "temperature.gpu",
  "utilization.gpu",
  "fan.speed",
] as const;

interface GpuSample {
  memUsedMB: number;
  memTotalMB: number;
  tempC: number | null;
  utilPct: number | null;
  fanPct: number | null;
  ts: number;
}

function pickBinary(): string | null {
  for (const c of NVIDIA_SMI_CANDIDATES) {
    // For the bare name we let PATH lookup happen at exec time; we
    // only short-circuit on the absolute path.
    if (c.startsWith("/")) {
      if (existsSync(c)) return c;
    } else {
      return c;
    }
  }
  return null;
}

function runNvidiaSmi(bin: string): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      bin,
      [
        `--query-gpu=${QUERY_FIELDS.join(",")}`,
        "--format=csv,noheader,nounits",
      ],
      { timeout: 4000 },
      (err, stdout) => {
        if (err) reject(err);
        else resolve(stdout);
      },
    );
  });
}

function parseNumOrNull(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  // nvidia-smi sentinels for unsupported metrics. Match case-insensitively
  // and tolerate the surrounding brackets.
  if (/^\[?(n\/a|not[\s_]?supported|unknown)\]?$/i.test(t)) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

export async function GET(): Promise<Response> {
  const bin = pickBinary();
  if (!bin) {
    return Response.json(
      { error: "nvidia-smi not available" },
      { status: 503 },
    );
  }

  let raw: string;
  try {
    raw = await runNvidiaSmi(bin);
  } catch (err) {
    // ENOENT (binary missing) and any non-zero exit both land here.
    const msg =
      err instanceof Error ? err.message : "nvidia-smi failed";
    return Response.json(
      { error: `nvidia-smi not available: ${msg}` },
      { status: 503 },
    );
  }

  // Multi-GPU rigs print one line per device. We only surface the
  // first; nothing in the pipeline currently fans out across cards.
  const firstLine = raw.split("\n").find((l) => l.trim().length > 0);
  if (!firstLine) {
    return Response.json(
      { error: "nvidia-smi returned empty output" },
      { status: 503 },
    );
  }

  const cols = firstLine.split(",").map((c) => c.trim());
  // We requested 5 fields; older drivers occasionally drop fan.
  const [memUsedRaw, memTotalRaw, tempRaw, utilRaw, fanRaw] = cols;

  const sample: GpuSample = {
    memUsedMB: parseNumOrNull(memUsedRaw ?? "") ?? 0,
    memTotalMB: parseNumOrNull(memTotalRaw ?? "") ?? 0,
    tempC: parseNumOrNull(tempRaw ?? ""),
    utilPct: parseNumOrNull(utilRaw ?? ""),
    fanPct: parseNumOrNull(fanRaw ?? ""),
    ts: Date.now(),
  };

  return Response.json(sample);
}
