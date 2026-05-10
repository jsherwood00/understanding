"use client";

// MonitorClient — the live polling surface of /monitor. Owns four
// independent intervals:
//
//   - GPU stats (2s) — fast, the user is mostly here for thermals
//   - Storage (15s) — directory walks are cheap but not free
//   - Log tail (3s) — file is short-tailed in the route; cheap
//   - Pause/thermal flag (3s) — bundled with the log poll cadence
//
// Each card degrades independently: a 503 from /api/gpu doesn't stop
// the log tail from updating, etc. The user gets a clear "stat
// unavailable" message instead of the whole page going red.
//
// React 19 strict-mode notes (this codebase enables react-hooks/refs
// and react-hooks/set-state-in-effect):
//   - Refs are written from a useEffect, never during render.
//   - Polling helpers re-fetch via a queueMicrotask hop so eslint
//     sees the setState as "outside the synchronous effect body".

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  MonitorGpuCard,
  type GpuSample,
} from "@/components/MonitorGpuCard";
import { MonitorStorageCard } from "@/components/MonitorStorageCard";
import { MonitorLogCard } from "@/components/MonitorLogCard";
import { MonitorPauseCard } from "@/components/MonitorPauseCard";

const POLL_GPU_MS = 2000;
const POLL_LOG_MS = 3000;
const POLL_STORAGE_MS = 15000;
const POLL_PAUSE_MS = 3000;

interface LogListEntry {
  file: string;
  sizeBytes: number;
  mtimeMs: number;
}

interface StorageState {
  perCorpus: Record<string, number>;
  totalBytes: number;
  capBytes: number;
  pct: number;
}

interface LogState {
  file: string;
  content: string;
  lineCount: number;
  sizeBytes: number;
  mtimeMs: number;
  truncated: boolean;
}

interface PauseState {
  userPaused: boolean;
  thermalPaused: boolean;
}

// Helper that schedules a fetcher off the current microtask so the
// react-hooks/set-state-in-effect lint rule doesn't flag it. The
// fetchers are already async, but the rule can't see past the call;
// hopping via queueMicrotask keeps the setState provably out of the
// synchronous effect body.
function defer(fn: () => void) {
  queueMicrotask(fn);
}

export function MonitorClient() {
  // ---- GPU ----
  const [gpu, setGpu] = useState<GpuSample | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);

  // ---- Storage ----
  const [storage, setStorage] = useState<StorageState | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);

  // ---- Logs ----
  const [logList, setLogList] = useState<LogListEntry[]>([]);
  // null = "use whichever log was most recently written" (default).
  // Once the user explicitly picks a log, we lock to that selection
  // so background log-list refreshes don't yank them around.
  const [selectedLog, setSelectedLog] = useState<string | null>(null);
  const [logState, setLogState] = useState<LogState | null>(null);
  const [logError, setLogError] = useState<string | null>(null);

  // ---- Pause flags ----
  const [pause, setPause] = useState<PauseState>({
    userPaused: false,
    thermalPaused: false,
  });

  // The log fetcher closes over the *current* selection without
  // re-creating the interval each time. Refs are written from an
  // effect (React 19 lint rule: no ref writes during render).
  const selectedLogRef = useRef<string | null>(null);
  const logListRef = useRef<LogListEntry[]>([]);
  useEffect(() => {
    selectedLogRef.current = selectedLog;
  }, [selectedLog]);
  useEffect(() => {
    logListRef.current = logList;
  }, [logList]);

  // Whichever log we're currently *displaying*, i.e. the user's
  // explicit pick, or the most-recently-touched if they haven't
  // picked one. Computed in render so the dropdown highlights it.
  const activeLogFile = useMemo(() => {
    if (selectedLog) return selectedLog;
    return logList[0]?.file ?? null;
  }, [selectedLog, logList]);

  // ---- Fetchers ----
  const fetchGpu = useCallback(async (signal: AbortSignal) => {
    try {
      const res = await fetch("/api/gpu", { signal, cache: "no-store" });
      if (res.status === 503) {
        const body = await res.json().catch(() => ({}));
        setGpu(null);
        setGpuError(body?.error ?? "GPU stats unavailable");
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as GpuSample;
      setGpu(data);
      setGpuError(null);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setGpuError((err as Error).message);
    }
  }, []);

  const fetchStorage = useCallback(async (signal: AbortSignal) => {
    try {
      const res = await fetch("/api/storage", { signal, cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as StorageState;
      setStorage(data);
      setStorageError(null);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setStorageError((err as Error).message);
    }
  }, []);

  const fetchLogList = useCallback(async (signal: AbortSignal) => {
    try {
      const res = await fetch("/api/log/list", { signal, cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { entries: LogListEntry[] };
      setLogList(data.entries);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      // Not fatal — keep whatever list we had.
    }
  }, []);

  const fetchLogContent = useCallback(async (signal: AbortSignal) => {
    const file =
      selectedLogRef.current ?? logListRef.current[0]?.file ?? null;
    if (!file) {
      setLogState(null);
      return;
    }
    try {
      const res = await fetch(
        `/api/log?file=${encodeURIComponent(file)}&lines=80`,
        { signal, cache: "no-store" },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as LogState;
      setLogState(data);
      setLogError(null);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setLogError((err as Error).message);
    }
  }, []);

  const fetchPause = useCallback(async (signal: AbortSignal) => {
    try {
      const res = await fetch("/api/pause", { signal, cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as PauseState;
      setPause(data);
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      // Non-fatal; keep last-known state.
    }
  }, []);

  const togglePause = useCallback(async () => {
    try {
      const res = await fetch("/api/pause", {
        method: "POST",
        cache: "no-store",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as PauseState;
      setPause(data);
    } catch (err) {
      // Surface the failure on the pause card by leaving the previous
      // state — a follow-up GET poll will reconcile soon enough.
      console.error("pause toggle failed", err);
    }
  }, []);

  // ---- Polling lifecycle ----
  // Each effect uses an AbortController so the in-flight request from
  // a previous tick is cancelled if the component unmounts.
  useEffect(() => {
    const ctl = new AbortController();
    defer(() => fetchGpu(ctl.signal));
    const id = setInterval(() => {
      fetchGpu(ctl.signal);
    }, POLL_GPU_MS);
    return () => {
      ctl.abort();
      clearInterval(id);
    };
  }, [fetchGpu]);

  useEffect(() => {
    const ctl = new AbortController();
    defer(() => fetchStorage(ctl.signal));
    const id = setInterval(() => {
      fetchStorage(ctl.signal);
    }, POLL_STORAGE_MS);
    return () => {
      ctl.abort();
      clearInterval(id);
    };
  }, [fetchStorage]);

  useEffect(() => {
    const ctl = new AbortController();
    // First tick: pull the list so we know the default file, then
    // fetch the content. After the first tick the two are
    // independent.
    defer(async () => {
      await fetchLogList(ctl.signal);
      await fetchLogContent(ctl.signal);
    });
    const idList = setInterval(() => {
      fetchLogList(ctl.signal);
    }, POLL_LOG_MS * 4);
    const idContent = setInterval(() => {
      fetchLogContent(ctl.signal);
    }, POLL_LOG_MS);
    return () => {
      ctl.abort();
      clearInterval(idList);
      clearInterval(idContent);
    };
  }, [fetchLogList, fetchLogContent]);

  // When the user picks a different log, fetch immediately so they
  // don't have to wait up to 3s for the next poll tick.
  useEffect(() => {
    if (selectedLog === null) return;
    const ctl = new AbortController();
    defer(() => fetchLogContent(ctl.signal));
    return () => ctl.abort();
  }, [selectedLog, fetchLogContent]);

  useEffect(() => {
    const ctl = new AbortController();
    defer(() => fetchPause(ctl.signal));
    const id = setInterval(() => {
      fetchPause(ctl.signal);
    }, POLL_PAUSE_MS);
    return () => {
      ctl.abort();
      clearInterval(id);
    };
  }, [fetchPause]);

  // A 1-Hz "tick" so the "updated Ns ago" labels in the GPU and log
  // cards refresh without us reading Date.now() during render. The
  // tick value itself is unused; it just nudges a re-render so the
  // child cards can pull their relative-time display from a state.
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <MonitorGpuCard sample={gpu} error={gpuError} now={now} />
      <MonitorPauseCard
        userPaused={pause.userPaused}
        thermalPaused={pause.thermalPaused}
        onToggle={togglePause}
      />
      <MonitorStorageCard storage={storage} error={storageError} />
      <MonitorLogCard
        entries={logList}
        activeFile={activeLogFile}
        onSelect={(f) => setSelectedLog(f)}
        log={logState}
        error={logError}
        now={now}
      />
    </div>
  );
}
