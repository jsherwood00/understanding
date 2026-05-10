// User-initiated pause flag — the pipeline polls /tmp/understanding_pause_user
// from its hot loop and idles generation while the file exists. We also
// surface the *thermal* flag (read-only) so the UI can show "GPU is
// auto-paused" without round-tripping a separate endpoint.
//
// POST is a toggle: present -> remove, absent -> create. Body is
// optional; if `{action: "set" | "clear"}` is present we honor it
// instead of toggling, which makes the button idempotent if the
// client and server disagree about the current state.

import { stat, writeFile, unlink } from "node:fs/promises";
import { FLAG_THERMAL_PAUSE, FLAG_USER_PAUSE } from "@/lib/monitor-paths";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function exists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function readState() {
  const [userPaused, thermalPaused] = await Promise.all([
    exists(FLAG_USER_PAUSE),
    exists(FLAG_THERMAL_PAUSE),
  ]);
  return { userPaused, thermalPaused };
}

export async function GET(): Promise<Response> {
  return Response.json(await readState());
}

export async function POST(request: Request): Promise<Response> {
  let action: "set" | "clear" | "toggle" = "toggle";
  try {
    const body = (await request.json()) as { action?: string };
    if (body.action === "set" || body.action === "clear") action = body.action;
  } catch {
    // No / invalid body is fine — fall through to toggle.
  }

  const before = await readState();
  let want: boolean;
  if (action === "set") want = true;
  else if (action === "clear") want = false;
  else want = !before.userPaused;

  try {
    if (want && !before.userPaused) {
      // Empty file is enough — the pipeline only checks existence.
      await writeFile(FLAG_USER_PAUSE, "");
    } else if (!want && before.userPaused) {
      await unlink(FLAG_USER_PAUSE);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : "flag write failed";
    return Response.json({ error: msg }, { status: 500 });
  }

  return Response.json(await readState());
}
