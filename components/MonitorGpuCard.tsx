"use client";

// Live GPU stat card. Color escalation matches the user's spec:
//
//   < 80°C   green
//   80–85    yellow
//   85–90    orange  ("watch")
//   > 90     red     ("danger — pipeline should auto-pause")
//
// We show the threshold band as a label next to the temp number so a
// glance at the card answers "is anything wrong?" without parsing.

import { Card, SectionLabel } from "@/components/MonitorPrimitives";

export interface GpuSample {
  memUsedMB: number;
  memTotalMB: number;
  tempC: number | null;
  utilPct: number | null;
  fanPct: number | null;
  ts: number;
}

type ThermalLevel = "cool" | "warm" | "watch" | "danger" | "unknown";

function thermalLevel(tempC: number | null): ThermalLevel {
  if (tempC === null) return "unknown";
  if (tempC < 80) return "cool";
  if (tempC < 85) return "warm";
  if (tempC < 90) return "watch";
  return "danger";
}

const LEVEL_LABEL: Record<ThermalLevel, string> = {
  cool: "ok",
  warm: "warm",
  watch: "watch",
  danger: "danger — auto-pause",
  unknown: "—",
};

// Each cell uses the canvas/ink token system with one color accent
// pulled inline so the level mapping stays in one place.
const LEVEL_COLOR: Record<ThermalLevel, string> = {
  cool: "#7cb342", // disgust-green; reused as a "calm green" accent
  warm: "#d4a017",
  watch: "#e07a1f",
  danger: "#e94b3c", // anger-red
  unknown: "#7a7a76",
};

export function MonitorGpuCard({
  sample,
  error,
  now,
}: {
  sample: GpuSample | null;
  error: string | null;
  /** Wall-clock millis from the parent's 1Hz tick. Passed in so the
   *  "updated Ns ago" label refreshes without us reading Date.now()
   *  during render (forbidden by react-hooks/purity). */
  now: number;
}) {
  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <SectionLabel>gpu</SectionLabel>
        {sample !== null ? (
          <span className="tabular text-[11px] text-ink-faint">
            updated {Math.max(0, Math.round((now - sample.ts) / 1000))}s ago
          </span>
        ) : null}
      </div>

      {error ? (
        <p className="mt-3 text-[13px] text-ink-muted">
          GPU stats unavailable — check nvidia-smi installation.{" "}
          <span className="text-ink-faint">({error})</span>
        </p>
      ) : sample === null ? (
        <p className="mt-3 text-[13px] text-ink-faint">…</p>
      ) : (
        <Body sample={sample} />
      )}
    </Card>
  );
}

function Body({ sample }: { sample: GpuSample }) {
  const level = thermalLevel(sample.tempC);
  const color = LEVEL_COLOR[level];
  const memPct =
    sample.memTotalMB > 0
      ? (sample.memUsedMB / sample.memTotalMB) * 100
      : 0;

  return (
    <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
      <Stat
        label="temp"
        value={
          sample.tempC === null ? "—" : `${Math.round(sample.tempC)}°C`
        }
        sub={LEVEL_LABEL[level]}
        color={color}
      />
      <Stat
        label="util"
        value={
          sample.utilPct === null
            ? "—"
            : `${Math.round(sample.utilPct)}%`
        }
        sub=" "
      />
      <Stat
        label="memory"
        value={`${formatGB(sample.memUsedMB)} / ${formatGB(sample.memTotalMB)}`}
        sub={`${memPct.toFixed(0)}%`}
        bar={memPct}
      />
      <Stat
        label="fan"
        value={
          sample.fanPct === null
            ? "n/a"
            : `${Math.round(sample.fanPct)}%`
        }
        sub=" "
      />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  color,
  bar,
}: {
  label: string;
  value: string;
  sub: string;
  color?: string;
  bar?: number;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="smallcaps text-ink-faint">{label}</span>
      <span
        className="tabular font-serif text-[22px] leading-none"
        style={color ? { color } : undefined}
      >
        {value}
      </span>
      {bar !== undefined ? (
        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-tint">
          <div
            className="h-full bg-ink-soft"
            style={{ width: `${Math.max(0, Math.min(100, bar))}%` }}
          />
        </div>
      ) : null}
      <span className="text-[11px] text-ink-muted">{sub}</span>
    </div>
  );
}

function formatGB(mb: number): string {
  if (!Number.isFinite(mb)) return "—";
  return `${(mb / 1024).toFixed(1)}G`;
}
