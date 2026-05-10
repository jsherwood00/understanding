"use client";

// Pause control + thermal-pause indicator.
//
// The thermal flag is read-only from the UI's perspective — the
// pipeline writes/removes /tmp/understanding_pause_thermal on its own
// schedule, so we just *display* it. The user-pause flag is the
// button: clicking it POSTs a toggle to /api/pause.

import { Card, SectionLabel } from "@/components/MonitorPrimitives";

export function MonitorPauseCard({
  userPaused,
  thermalPaused,
  onToggle,
}: {
  userPaused: boolean;
  thermalPaused: boolean;
  onToggle: () => void;
}) {
  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <SectionLabel>control</SectionLabel>
        <ThermalBadge thermalPaused={thermalPaused} />
      </div>

      <div className="mt-4 flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-[13px] text-ink">
            {userPaused
              ? "pipeline paused by user"
              : "pipeline running"}
          </span>
          <span className="text-[12px] text-ink-muted">
            click to{" "}
            {userPaused ? "remove" : "create"}{" "}
            <code className="font-mono text-[11px] text-ink-soft">
              /tmp/understanding_pause_user
            </code>
          </span>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className={
            "min-w-[120px] rounded-full border px-4 py-1.5 text-[12px] tracking-wide uppercase transition-colors " +
            (userPaused
              ? "border-ink bg-ink text-canvas hover:bg-ink-soft"
              : "border-ink/30 text-ink-soft hover:border-ink hover:text-ink")
          }
        >
          {userPaused ? "resume" : "pause"}
        </button>
      </div>
    </Card>
  );
}

function ThermalBadge({ thermalPaused }: { thermalPaused: boolean }) {
  if (!thermalPaused) {
    return (
      <span className="text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        thermal: ok
      </span>
    );
  }
  // The thermal pause is something the user *needs* to notice — soft
  // reds are easy to miss in peripheral vision. We emphasize with the
  // same pulse used on the divergence indicator on the chat panel.
  return (
    <span
      className="divergence-gap rounded-full bg-[#e94b3c] px-2 py-0.5 text-[10px] font-semibold tracking-[0.16em] text-canvas uppercase"
      title="Pipeline auto-paused due to GPU temperature"
    >
      thermal pause active
    </span>
  );
}
