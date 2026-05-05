"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BASELINE,
  BASELINE_RAW_STATE,
  averageLayered,
  makeLayeredBaseline,
  mapLayeredBackendEmotions,
  type EmotionState,
  type EmotionValues,
  type LayeredEmotionValues,
  type PerTokenData,
  type RawState,
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
  thinking: Record<string, Record<string, number>>;
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
const SNAPSHOT_EVERY_N_TOKENS = 5;
const DEFAULT_LAYER: Layer = 21;

async function classifyOutput(text: string): Promise<EmotionValues> {
  try {
    const res = await fetch("/api/sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return { ...BASELINE };
    const data = (await res.json()) as { emotions?: EmotionValues };
    return data.emotions ?? { ...BASELINE };
  } catch {
    return { ...BASELINE };
  }
}

/** Find the most recent turn whose assistantReply contains `excerpt`,
 *  and return the per-token data for tokens overlapping that range.
 *  Returns null if no match. */
function tokensForExcerpt(
  excerpt: string,
  turns: Turn[],
): PerTokenData[] | null {
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i];
    const idx = turn.assistantReply.indexOf(excerpt);
    if (idx < 0) continue;
    const start = idx;
    const end = idx + excerpt.length;
    const matched: PerTokenData[] = [];
    let prevEnd = 0;
    for (const tok of turn.tokens) {
      const tokStart = prevEnd;
      const tokEnd = tok.charEnd;
      // Token range [tokStart, tokEnd) overlaps selection [start, end)
      if (tokStart < end && tokEnd > start) {
        matched.push(tok);
      }
      prevEnd = tokEnd;
    }
    if (matched.length > 0) return matched;
  }
  return null;
}

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [rawBars, setRawBars] = useState<RawState>(BASELINE_RAW_STATE);
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

  // Pre-warm distilroberta so end-of-turn classification doesn't pay the
  // ONNX cold-start.
  useEffect(() => {
    void fetch("/api/sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "hello world" }),
    }).catch(() => {});
  }, []);

  // Selection listener.
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

  // Debounced classifier call on the selection (the dot's value).
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

  // Layered thinking averaged over the tokens that produced the selected
  // excerpt. Recomputed when the selection changes or new turns arrive.
  const selectionLayered = useMemo<LayeredEmotionValues | null>(() => {
    if (!selectedExcerpt) return null;
    const matched = tokensForExcerpt(selectedExcerpt, turns);
    return matched ? averageLayered(matched) : null;
  }, [selectedExcerpt, turns]);

  const displayedBars = useMemo<EmotionState>(() => {
    if (selectedExcerpt) {
      // Dot: the *external* classifier on the highlighted text. Lexicon
      // is the instant fallback while distilroberta is in flight.
      const lexicon = analyzeEmotions(selectedExcerpt);
      const dotValues = selectionEmotions ?? lexicon;

      // Halo: average residual-stream projection at the selected layer
      // over the tokens generated for this excerpt. If the excerpt isn't
      // in any reply (e.g. user message), halo sits at baseline.
      const haloValues = selectionLayered
        ? selectionLayered[selectedLayer]
        : { ...BASELINE };

      return { output: dotValues, thinking: haloValues };
    }
    return {
      output: rawBars.output,
      thinking: rawBars.thinking[selectedLayer],
    };
  }, [
    selectedExcerpt,
    selectionEmotions,
    selectionLayered,
    rawBars,
    selectedLayer,
  ]);

  function applyTurnView(turnIdx: number, snapIdx: number) {
    const turn = turns[turnIdx];
    if (!turn) return;
    const snap =
      turn.snapshots[
        Math.max(0, Math.min(snapIdx, turn.snapshots.length - 1))
      ];
    setRawBars({
      output: turn.state.output,
      thinking: snap ? snap.thinking : turn.state.thinking,
    });
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
    // Clear the dot — no output reading for this turn yet. Halo carries
    // last value until the first token arrives.
    setRawBars((b) => ({ output: null, thinking: b.thinking }));
    setError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accumulatedOutput = "";
    let layeredThinking: LayeredEmotionValues = makeLayeredBaseline();
    const snapshots: Snapshot[] = [];
    const tokenLog: PerTokenData[] = [];
    let completed = false;
    let tokenCount = 0;

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
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

              layeredThinking = mapLayeredBackendEmotions(data.thinking);
              setRawBars({ output: null, thinking: layeredThinking });

              tokenLog.push({
                charEnd: accumulatedOutput.length,
                thinking: layeredThinking,
              });

              if (tokenCount % SNAPSHOT_EVERY_N_TOKENS === 0) {
                const wordCount = accumulatedOutput
                  .trim()
                  .split(/\s+/)
                  .filter(Boolean).length;
                snapshots.push({
                  atWord: wordCount,
                  thinking: layeredThinking,
                });
              }
              break;
            }
            case "done": {
              completed = true;
              const fullText = data.fullText ?? accumulatedOutput;

              setMessages((m) => [
                ...m,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: fullText,
                },
              ]);
              setStreamingContent(null);

              const wordCount = fullText.trim().split(/\s+/).filter(Boolean)
                .length;
              const finalSnap: Snapshot = {
                atWord: wordCount,
                thinking: layeredThinking,
              };
              const allSnaps = [...snapshots, finalSnap];

              const outputEmotions = await classifyOutput(fullText);
              setRawBars({
                output: outputEmotions,
                thinking: layeredThinking,
              });

              const newTurn: Turn = {
                id: crypto.randomUUID(),
                userMessage: text,
                assistantReply: fullText,
                snapshots: allSnaps,
                tokens: tokenLog,
                state: {
                  output: outputEmotions,
                  thinking: layeredThinking,
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
        const fallbackOutput = await classifyOutput(truncated);
        setRawBars({
          output: fallbackOutput,
          thinking: layeredThinking,
        });
        const fallbackSnap: Snapshot = {
          atWord: 0,
          thinking: layeredThinking,
        };
        const newTurn: Turn = {
          id: crypto.randomUUID(),
          userMessage: text,
          assistantReply: truncated,
          snapshots: snapshots.length > 0 ? [...snapshots] : [fallbackSnap],
          tokens: tokenLog,
          state: { output: fallbackOutput, thinking: layeredThinking },
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
      setRawBars({ output: turn.state.output, thinking: snap.thinking });
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
        setRawBars({ output: turn.state.output, thinking: snap.thinking });
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
