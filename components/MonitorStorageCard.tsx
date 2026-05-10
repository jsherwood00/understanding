"use client";

// Storage usage across data/<corpus>/activations/, vs the 200GB hard
// cap. A progress bar handles the "am I close to the wall?" glance,
// and the per-corpus breakdown beneath it makes it obvious which
// run is the heavy one.

import { Card, SectionLabel } from "@/components/MonitorPrimitives";

interface StorageState {
  perCorpus: Record<string, number>;
  totalBytes: number;
  capBytes: number;
  pct: number;
}

export function MonitorStorageCard({
  storage,
  error,
}: {
  storage: StorageState | null;
  error: string | null;
}) {
  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <SectionLabel>storage</SectionLabel>
        {storage !== null ? (
          <span className="tabular text-[11px] text-ink-faint">
            {formatGB(storage.totalBytes)} / {formatGB(storage.capBytes)} ·{" "}
            {storage.pct.toFixed(1)}%
          </span>
        ) : null}
      </div>

      {error ? (
        <p className="mt-3 text-[13px] text-ink-muted">
          could not read storage ({error})
        </p>
      ) : storage === null ? (
        <p className="mt-3 text-[13px] text-ink-faint">…</p>
      ) : (
        <Body storage={storage} />
      )}
    </Card>
  );
}

// Past 80% the bar warms; past 95% it goes red — the user said the
// cap is hard, so the warning needs to feel hard.
function barColor(pct: number): string {
  if (pct >= 95) return "#e94b3c";
  if (pct >= 80) return "#e07a1f";
  return "#4a4a4a";
}

function Body({ storage }: { storage: StorageState }) {
  const pct = Math.max(0, Math.min(100, storage.pct));
  const color = barColor(pct);
  return (
    <div className="mt-4 flex flex-col gap-3">
      <div className="h-2 w-full overflow-hidden rounded-full bg-tint">
        <div
          className="h-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-ink-soft">
        {Object.entries(storage.perCorpus).map(([k, v]) => (
          <span key={k} className="tabular">
            <span className="smallcaps text-ink-faint">{k.replace("_", " ")}</span>{" "}
            {formatGB(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

function formatGB(bytes: number): string {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}
