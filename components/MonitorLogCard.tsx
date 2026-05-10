"use client";

// Log tail card. The dropdown lists every log file currently on disk
// (filtered server-side against the whitelist); the most-recent file
// is selected by default. The body is a fixed-height pre block that
// auto-scrolls to the bottom on every refresh — log tails are only
// useful if you can see the latest line without scrolling.

import { useEffect, useRef } from "react";
import { Card, SectionLabel } from "@/components/MonitorPrimitives";

interface LogEntry {
  file: string;
  sizeBytes: number;
  mtimeMs: number;
}

interface LogState {
  file: string;
  content: string;
  lineCount: number;
  sizeBytes: number;
  mtimeMs: number;
  truncated: boolean;
}

export function MonitorLogCard({
  entries,
  activeFile,
  onSelect,
  log,
  error,
  now,
}: {
  entries: LogEntry[];
  activeFile: string | null;
  onSelect: (file: string) => void;
  log: LogState | null;
  error: string | null;
  /** Wall-clock millis from the parent's 1Hz tick — used to render
   *  the "Xs ago" annotations next to each log entry without
   *  reading Date.now() during render. */
  now: number;
}) {
  const preRef = useRef<HTMLPreElement>(null);

  // Auto-scroll to bottom whenever new content lands. We don't
  // attempt the "stick only when the user is already at the bottom"
  // trick because the pane is short (<= 80 lines) and there's no
  // reason for the user to manually scroll within it.
  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [log?.content]);

  return (
    <Card>
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <SectionLabel>pipeline log</SectionLabel>
        <div className="flex items-center gap-2">
          {log?.truncated ? (
            <span
              className="text-[10px] tracking-[0.16em] text-ink-faint uppercase"
              title="Showing only the tail of a large file"
            >
              tail only
            </span>
          ) : null}
          <select
            value={activeFile ?? ""}
            onChange={(e) => onSelect(e.target.value)}
            disabled={entries.length === 0}
            className="rounded border border-divider bg-canvas px-2 py-1 text-[12px] text-ink-soft hover:border-ink-faint focus:border-ink focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
          >
            {entries.length === 0 ? (
              <option value="">no logs found</option>
            ) : (
              entries.map((e) => (
                <option key={e.file} value={e.file}>
                  {e.file} · {formatAgo(e.mtimeMs, now)}
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {error ? (
        <p className="mt-3 text-[13px] text-ink-muted">
          could not read log ({error})
        </p>
      ) : null}

      <pre
        ref={preRef}
        className="tabular mt-4 h-[420px] overflow-auto rounded border border-divider bg-tint p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-ink-soft"
      >
        {log?.content || (entries.length === 0 ? "—" : "(empty)")}
      </pre>

      {log ? (
        <div className="mt-2 flex justify-between text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          <span>{log.lineCount} lines</span>
          <span>{formatBytes(log.sizeBytes)}</span>
        </div>
      ) : null}
    </Card>
  );
}

function formatAgo(mtimeMs: number, nowMs: number): string {
  if (!mtimeMs) return "—";
  const sec = Math.max(0, Math.round((nowMs - mtimeMs) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}
