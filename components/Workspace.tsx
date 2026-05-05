"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BASELINE_STATE,
  mapBackendEmotions,
  type EmotionState,
  type EmotionValues,
  type Snapshot,
  type Turn,
} from "@/lib/emotions";
import { analyzeEmotions } from "@/lib/sentiment";
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";
import { type Layer } from "@/components/LayerSelector";

interface BackendTokenEvent {
  type: "token";
  text: string;
  thinking: Record<string, number>;
  step: number;
}
interface BackendDoneEvent {
  type: "done";
  fullText: string;
  tokens: number;
}
interface BackendErrorEvent {
  type: "error";
  error: string;
}
type BackendEvent =
  | BackendTokenEvent
  | BackendDoneEvent
  | BackendErrorEvent;

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const REPLAY_PER_SNAPSHOT_MS = 220;
const REPLAY_FINAL_HOLD_MS = 700;
const SELECTION_MIN_CHARS = 3;
// Record a snapshot every N generated tokens. CSS bar transitions are
// 300ms; per-token cadence is ~26ms — snapshotting every 5 keeps the
// recorded trajectory roughly in step with the visual transitions.
const SNAPSHOT_EVERY_N_TOKENS = 5;
const DEFAULT_LAYER: Layer = 21;

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [bars, setBars] = useState<EmotionState>(BASELINE_STATE);
  const [error, setError] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<Layer>(DEFAULT_LAYER);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [viewingIndex, setViewingIndex] = useState<number | null>(null);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);

  const [selectedExcerpt, setSelectedExcerpt] = useState<string | null>(null);
  const [selectionEmotions, setSelectionEmotions] =
    useState<EmotionValues | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const selectionAbortRef = useRef<AbortController | null>(null);
  const replayAbortRef = useRef(false);
  const savedViewRef = useRef<{ turn: number | null; snap: number } | null>(
    null,
  );

  // Selection listener for the "highlight a phrase to see its emotions"
  // feature. Filters out tiny / non-alphabetic selections.
  useEffect(() => {
    function onSelectionChange() {
      const raw = window.getSelection()?.toString() ?? "";
      const trimmed = raw.replace(/\s+/g, " ").trim();
      if (
        trimmed.length < SELECTION_MIN_CHARS ||
        !/[a-zA-Z]{2,}/.test(trimmed)
      ) {
        setSelectedExcerpt(null);
        setSelectionEmotions(null);
        selectionAbortRef.current?.abort();
        selectionAbortRef.current = null;
        return;
      }
      setSelectedExcerpt(trimmed);
    }
    document.addEventListener("selectionchange", onSelectionChange);
    return () =>
      document.removeEventListener("selectionchange", onSelectionChange);
  }, []);

  // Debounced model-based analysis on the selection. Lexicon is the instant
  // fallback; /api/sentiment (j-hartmann distilroberta) overrides on return.
  useEffect(() => {
    if (!selectedExcerpt) return;
    const handle = setTimeout(async () => {
      selectionAbortRef.current?.abort();
      const controller = new AbortController();
      selectionAbortRef.current = controller;
      try {
        const res = await fetch("/api/sentiment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: selectedExcerpt }),
          signal: controller.signal,
        });
        if (!res.ok) return;
        const data = (await res.json()) as { emotions?: EmotionValues };
        if (controller.signal.aborted) return;
        if (data.emotions) setSelectionEmotions(data.emotions);
      } catch {
        // Aborted or network error — keep lexicon fallback.
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [selectedExcerpt]);

  const displayedBars = useMemo<EmotionState>(() => {
    if (selectedExcerpt) {
      const lexicon = analyzeEmotions(selectedExcerpt);
      const final = selectionEmotions ?? lexicon;
      return { output: final, thinking: final };
    }
    return bars;
  }, [selectedExcerpt, selectionEmotions, bars]);

  function applyTurnView(turnIdx: number, snapIdx: number) {
    const turn = turns[turnIdx];
    if (!turn) return;
    const snap =
      turn.snapshots[
        Math.max(0, Math.min(snapIdx, turn.snapshots.length - 1))
      ];
    if (snap) {
      setBars({ output: snap.output, thinking: snap.thinking });
    } else {
      setBars(turn.state);
    }
  }

  async function handleSubmit() {
    const text = input.trim();
    if (!text || isGenerating || isReplaying) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    const priorMessages = [...messages];
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setIsGenerating(true);
    setStreamingContent("");
    // Don't reset bars — keep last turn's values until first chunk arrives.
    setError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accumulatedOutput = "";
    let outputEmotions: EmotionValues = bars.output;
    let thinkingEmotions: EmotionValues = bars.thinking;
    const snapshots: Snapshot[] = [];
    let completed = false;
    let tokenCount = 0;

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          layer: selectedLayer,
          history: priorMessages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errPayload = await res.json().catch(() => null);
        const message =
          (errPayload && typeof errPayload.error === "string"
            ? errPayload.error
            : null) ?? `Request failed (${res.status}).`;
        throw new Error(message);
      }
      if (!res.body) throw new Error("Server returned no body.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // sse-starlette emits CRLF event separators; split on either.
        const parts = buffer.split(/\r\n\r\n|\n\n/);
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          let data: BackendEvent;
          try {
            data = JSON.parse(part.slice(6)) as BackendEvent;
          } catch {
            continue;
          }

          switch (data.type) {
            case "token": {
              tokenCount += 1;
              accumulatedOutput += data.text;
              setStreamingContent(accumulatedOutput);

              thinkingEmotions = mapBackendEmotions(data.thinking);
              outputEmotions = analyzeEmotions(accumulatedOutput);
              setBars({ output: outputEmotions, thinking: thinkingEmotions });

              if (tokenCount % SNAPSHOT_EVERY_N_TOKENS === 0) {
                const wordCount = accumulatedOutput
                  .trim()
                  .split(/\s+/)
                  .filter(Boolean).length;
                snapshots.push({
                  atWord: wordCount,
                  output: outputEmotions,
                  thinking: thinkingEmotions,
                });
              }
              break;
            }
            case "done": {
              completed = true;
              const fullText = data.fullText ?? accumulatedOutput;
              outputEmotions = analyzeEmotions(fullText);
              setMessages((m) => [
                ...m,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: fullText,
                },
              ]);
              setBars({
                output: outputEmotions,
                thinking: thinkingEmotions,
              });
              setStreamingContent(null);
              const finalSnap: Snapshot = {
                atWord: fullText
                  .trim()
                  .split(/\s+/)
                  .filter(Boolean).length,
                output: outputEmotions,
                thinking: thinkingEmotions,
              };
              const newTurn: Turn = {
                id: crypto.randomUUID(),
                userMessage: text,
                assistantReply: fullText,
                snapshots: [...snapshots, finalSnap],
                state: {
                  output: outputEmotions,
                  thinking: thinkingEmotions,
                },
              };
              setTurns((t) => {
                const next = [...t, newTurn];
                setViewingIndex(next.length - 1);
                setSnapshotIndex(Math.max(0, newTurn.snapshots.length - 1));
                return next;
              });
              break;
            }
            case "error": {
              throw new Error(data.error || "Backend error");
            }
          }
        }
      }
    } catch (e) {
      const isAbort = e instanceof DOMException && e.name === "AbortError";
      if (!isAbort) {
        setError(
          e instanceof Error ? e.message : "Something went wrong.",
        );
      }
    } finally {
      if (!completed && accumulatedOutput.trim().length > 0) {
        const truncated = `${accumulatedOutput.trim()} […]`;
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: truncated,
          },
        ]);
        const fallbackSnap: Snapshot = {
          atWord: 0,
          output: outputEmotions,
          thinking: thinkingEmotions,
        };
        const newTurn: Turn = {
          id: crypto.randomUUID(),
          userMessage: text,
          assistantReply: truncated,
          snapshots: snapshots.length > 0 ? [...snapshots] : [fallbackSnap],
          state: { output: outputEmotions, thinking: thinkingEmotions },
        };
        setTurns((t) => {
          const next = [...t, newTurn];
          setViewingIndex(next.length - 1);
          setSnapshotIndex(Math.max(0, newTurn.snapshots.length - 1));
          return next;
        });
      }
      setStreamingContent(null);
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }

  function handleStopGeneration() {
    abortControllerRef.current?.abort();
  }

  function navigateTurn(direction: -1 | 1) {
    if (viewingIndex === null || isReplaying || isGenerating) return;
    const next = viewingIndex + direction;
    if (next < 0 || next >= turns.length) return;
    setViewingIndex(next);
    const last = Math.max(0, turns[next].snapshots.length - 1);
    setSnapshotIndex(last);
    applyTurnView(next, last);
  }

  function handleScrub(snapIdx: number) {
    if (viewingIndex === null || isReplaying || isGenerating) return;
    setSnapshotIndex(snapIdx);
    applyTurnView(viewingIndex, snapIdx);
  }

  function sleep(ms: number) {
    return new Promise<void>((resolve) => setTimeout(resolve, ms));
  }

  async function handleReplayTurn() {
    if (viewingIndex === null || turns.length === 0) return;
    if (isGenerating || isReplaying) return;
    const turn = turns[viewingIndex];
    if (turn.snapshots.length === 0) return;
    replayAbortRef.current = false;
    setIsReplaying(true);
    for (let i = 0; i < turn.snapshots.length; i++) {
      if (replayAbortRef.current) break;
      const snap = turn.snapshots[i];
      setSnapshotIndex(i);
      setBars({ output: snap.output, thinking: snap.thinking });
      await sleep(REPLAY_PER_SNAPSHOT_MS);
    }
    await sleep(REPLAY_FINAL_HOLD_MS);
    setIsReplaying(false);
  }

  async function handleReplayAll() {
    if (turns.length === 0 || isGenerating || isReplaying) return;
    savedViewRef.current = { turn: viewingIndex, snap: snapshotIndex };
    replayAbortRef.current = false;
    setIsReplaying(true);
    for (let t = 0; t < turns.length; t++) {
      if (replayAbortRef.current) break;
      const turn = turns[t];
      setViewingIndex(t);
      for (let i = 0; i < turn.snapshots.length; i++) {
        if (replayAbortRef.current) break;
        const snap = turn.snapshots[i];
        setSnapshotIndex(i);
        setBars({ output: snap.output, thinking: snap.thinking });
        await sleep(REPLAY_PER_SNAPSHOT_MS);
      }
      if (!replayAbortRef.current) await sleep(REPLAY_FINAL_HOLD_MS);
    }
    setIsReplaying(false);
  }

  function handleStopReplay() {
    replayAbortRef.current = true;
    if (savedViewRef.current) {
      const { turn, snap } = savedViewRef.current;
      if (turn !== null) {
        setViewingIndex(turn);
        setSnapshotIndex(snap);
        applyTurnView(turn, snap);
      }
      savedViewRef.current = null;
    }
    setIsReplaying(false);
  }

  return (
    <>
      <div className="w-2/5 border-r border-divider">
        <EmotionPanel
          state={displayedBars}
          turns={turns}
          viewingIndex={viewingIndex}
          snapshotIndex={snapshotIndex}
          selectedExcerpt={selectedExcerpt}
          onNavigate={navigateTurn}
          onScrub={handleScrub}
          onReplayTurn={handleReplayTurn}
          onReplayAll={handleReplayAll}
          onStopReplay={handleStopReplay}
          isReplaying={isReplaying}
          isGenerating={isGenerating}
          selectedLayer={selectedLayer}
          onLayerChange={setSelectedLayer}
        />
      </div>
      <div className="w-3/5">
        <ChatPane
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
          streamingText={streamingContent}
          error={error}
          onDismissError={() => setError(null)}
        />
      </div>
    </>
  );
}
