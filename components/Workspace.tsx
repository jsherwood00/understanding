"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BASELINE,
  BASELINE_RAW_STATE,
  BEST_LAYER,
  averageLayered,
  averageLayeredForPhase,
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
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";
import { type Layer } from "@/components/LayerSelector";

interface BackendTokenEvent {
  type: "token";
  text: string;
  thinking: Record<string, Record<string, number>>;
  step: number;
  /** "thought" while the model is inside its <|channel>thought... block,
   *  "reply" once it crosses the <channel|> marker. Older backends that
   *  haven't been updated may omit this — treat as "reply". */
  phase?: "thought" | "reply";
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

interface ExcerptMatch {
  phase: "thought" | "reply";
  tokens: PerTokenData[];
}

/** Find the most recent turn whose reply OR thought contains `excerpt`,
 *  and return matching tokens + which phase they belong to. Reply is
 *  checked first (more likely surface), then thought. Returns null if no
 *  match. Per-token charEnd values are relative to that token's phase
 *  buffer (set in handleSubmit's token loop), so we only consider tokens
 *  whose phase matches the matched text. */
function matchExcerpt(excerpt: string, turns: Turn[]): ExcerptMatch | null {
  for (let i = turns.length - 1; i >= 0; i--) {
    const turn = turns[i];

    const replyIdx = turn.assistantReply.indexOf(excerpt);
    if (replyIdx >= 0) {
      const matched = tokensInRange(
        turn.tokens.filter((t) => t.phase === "reply"),
        replyIdx,
        replyIdx + excerpt.length,
      );
      if (matched.length > 0) return { phase: "reply", tokens: matched };
    }

    const thoughtIdx = turn.assistantThought.indexOf(excerpt);
    if (thoughtIdx >= 0) {
      const matched = tokensInRange(
        turn.tokens.filter((t) => t.phase === "thought"),
        thoughtIdx,
        thoughtIdx + excerpt.length,
      );
      if (matched.length > 0) return { phase: "thought", tokens: matched };
    }
  }
  return null;
}

function tokensInRange(
  tokens: PerTokenData[],
  start: number,
  end: number,
): PerTokenData[] {
  const matched: PerTokenData[] = [];
  let prevEnd = 0;
  for (const tok of tokens) {
    const tokStart = prevEnd;
    const tokEnd = tok.charEnd;
    if (tokStart < end && tokEnd > start) matched.push(tok);
    prevEnd = tokEnd;
  }
  return matched;
}

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThought, setStreamingThought] = useState<string | null>(null);
  const [rawBars, setRawBars] = useState<RawState>(BASELINE_RAW_STATE);
  const [error, setError] = useState<string | null>(null);
  // Two independent layer picks — one per scope. Defaults are the
  // holdout-best layers (L13 for thought, L25 for reply); see BEST_LAYER
  // in lib/emotions.ts.
  const [thoughtLayer, setThoughtLayer] = useState<Layer>(BEST_LAYER.thought);
  const [replyLayer, setReplyLayer] = useState<Layer>(BEST_LAYER.reply);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [viewingIndex, setViewingIndex] = useState<number | null>(null);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);

  const [selectedExcerpt, setSelectedExcerpt] = useState<string | null>(null);
  const [classifierOn, setClassifierOn] = useState(false);
  const [valuesOn, setValuesOn] = useState(false);
  /** Flips true the moment the streaming turn enters its reply phase
   *  (first reply-phase token arrives). Resets on each new submit. The
   *  EmotionPanel uses this to gate the diff label so it doesn't read
   *  meaninglessly negative during the thought-only prefix. */
  const [replyStarted, setReplyStarted] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
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
        return;
      }
      setSelectedExcerpt(trimmed);
    }
    document.addEventListener("selectionchange", onSelectionChange);
    return () =>
      document.removeEventListener("selectionchange", onSelectionChange);
  }, []);

  // Live NLI sentiment of the selected excerpt — drives the dot (if the
  // selection is in the reply) or the solid line (if in the thought).
  // Recomputed when the excerpt or its matched phase changes.
  const [selectionSentiment, setSelectionSentiment] =
    useState<EmotionValues | null>(null);

  // Resolve which phase the excerpt lives in (if any) and the layered
  // average over the matching tokens. The latter drives the halo or
  // dashed line depending on phase.
  const excerptMatch = useMemo<{
    phase: "thought" | "reply";
    layered: LayeredEmotionValues;
  } | null>(() => {
    if (!selectedExcerpt) return null;
    const match = matchExcerpt(selectedExcerpt, turns);
    if (!match) return null;
    return { phase: match.phase, layered: averageLayered(match.tokens) };
  }, [selectedExcerpt, turns]);

  // Trigger NLI re-classification on the excerpt itself when it changes.
  useEffect(() => {
    if (!selectedExcerpt || !excerptMatch) {
      setSelectionSentiment(null);
      return;
    }
    let cancelled = false;
    void classifyOutput(selectedExcerpt).then((res) => {
      if (!cancelled) setSelectionSentiment(res);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedExcerpt, excerptMatch]);

  // True when the user has dragged the scrubber off the final snapshot.
  // Mid-scrub readings are per-chunk; the post-hoc sentiment indicators
  // (dot + solid line) are turn-as-a-whole readings, so they don't apply
  // there.
  const isAtNonFinalSnap = useMemo(() => {
    if (viewingIndex === null) return false;
    const turn = turns[viewingIndex];
    if (!turn) return false;
    return snapshotIndex < turn.snapshots.length - 1;
  }, [viewingIndex, snapshotIndex, turns]);

  const displayedBars = useMemo<EmotionState>(() => {
    const baseReply =
      rawBars.thinkingReply[replyLayer] ?? { ...BASELINE };
    const baseThought =
      rawBars.thinkingThought?.[thoughtLayer] ?? null;

    if (selectedExcerpt && excerptMatch) {
      // Move the indicator belonging to the matched phase, at that
      // phase's selected layer. Other phase's indicator stays put.
      if (excerptMatch.phase === "reply") {
        const movedLayered = excerptMatch.layered[replyLayer];
        return {
          outputReply: selectionSentiment ?? rawBars.outputReply,
          outputThought: rawBars.outputThought,
          thinkingReply: movedLayered,
          thinkingThought: baseThought,
        };
      }
      const movedLayered = excerptMatch.layered[thoughtLayer];
      return {
        outputReply: rawBars.outputReply,
        outputThought: selectionSentiment ?? rawBars.outputThought,
        thinkingReply: baseReply,
        thinkingThought: movedLayered,
      };
    }

    return {
      outputReply: isAtNonFinalSnap ? null : rawBars.outputReply,
      outputThought: isAtNonFinalSnap ? null : rawBars.outputThought,
      thinkingReply: baseReply,
      thinkingThought: baseThought,
    };
  }, [
    selectedExcerpt,
    excerptMatch,
    selectionSentiment,
    rawBars,
    thoughtLayer,
    replyLayer,
    isAtNonFinalSnap,
  ]);

  function applyTurnView(turnIdx: number, snapIdx: number) {
    const turn = turns[turnIdx];
    if (!turn) return;
    const snap =
      turn.snapshots[
        Math.max(0, Math.min(snapIdx, turn.snapshots.length - 1))
      ];
    setRawBars({
      outputReply: turn.state.outputReply,
      outputThought: turn.state.outputThought,
      thinkingReply: snap ? snap.thinkingReply : turn.state.thinkingReply,
      thinkingThought: snap
        ? snap.thinkingThought
        : turn.state.thinkingThought,
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
    setReplyStarted(false);
    setStreamingContent("");
    setStreamingThought("");
    // Clear both NLI readings — no post-hoc sentiment for this turn yet.
    // Activation halos carry their last value until the first token arrives.
    setRawBars((b) => ({
      outputReply: null,
      outputThought: null,
      thinkingReply: b.thinkingReply,
      thinkingThought: b.thinkingThought,
    }));
    setError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Two parallel buffers — `accumulatedReply` is the user-facing answer,
    // `accumulatedThought` is the model's internal reasoning block (when
    // thinking is enabled). Both stream concurrently as token events
    // arrive, each tagged with a phase by the backend. The activation
    // projection that arrives with each token is already scoped to its
    // phase (V3 for reply, V2 for thought).
    let accumulatedReply = "";
    let accumulatedThought = "";
    let layeredReply: LayeredEmotionValues = makeLayeredBaseline();
    let layeredThought: LayeredEmotionValues | null = null;
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
              const phase = data.phase ?? "reply";
              if (phase === "thought") {
                accumulatedThought += data.text;
                setStreamingThought(accumulatedThought);
              } else {
                if (accumulatedReply.length === 0) {
                  setReplyStarted(true);
                  // First reply token — collapse the live thought halo
                  // into the whole-thought-block average so the
                  // visualization settles before the reply phase begins
                  // (instead of leaving the halo at the last-thought-
                  // token reading, which is point-in-time noise).
                  const thoughtAvg = averageLayeredForPhase(
                    tokenLog,
                    "thought",
                  );
                  if (thoughtAvg !== null) layeredThought = thoughtAvg;
                }
                accumulatedReply += data.text;
                setStreamingContent(accumulatedReply);
              }

              const tokenLayered = mapLayeredBackendEmotions(data.thinking);
              if (phase === "thought") {
                layeredThought = tokenLayered;
              } else {
                layeredReply = tokenLayered;
              }
              setRawBars({
                outputReply: null,
                outputThought: null,
                thinkingReply: layeredReply,
                thinkingThought: layeredThought,
              });

              // charEnd is into the buffer for THIS token's phase so the
              // scrubber and highlight code can map char positions back
              // to tokens within their respective texts.
              const buf =
                phase === "thought" ? accumulatedThought : accumulatedReply;
              tokenLog.push({
                charEnd: buf.length,
                phase,
                thinking: tokenLayered,
              });

              if (tokenCount % SNAPSHOT_EVERY_N_TOKENS === 0) {
                const wordCount = (accumulatedThought + " " + accumulatedReply)
                  .trim()
                  .split(/\s+/)
                  .filter(Boolean).length;
                // Snapshot the running per-phase averages, not the
                // single most-recent token — keeps the scrubber smooth.
                snapshots.push({
                  atWord: wordCount,
                  thinkingReply:
                    averageLayeredForPhase(tokenLog, "reply") ??
                    makeLayeredBaseline(),
                  thinkingThought: averageLayeredForPhase(tokenLog, "thought"),
                });
              }
              break;
            }
            case "done": {
              completed = true;
              const fullReply = accumulatedReply;
              const thoughtText = accumulatedThought;

              setMessages((m) => [
                ...m,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: fullReply,
                  thought: thoughtText || undefined,
                },
              ]);
              setStreamingContent(null);
              setStreamingThought(null);

              // Per-phase activation averages over every token in the
              // turn (not just the last). Last-token readings are noisy
              // / point-in-time; the average gives a holistic measure
              // of internal state across each phase.
              const finalReplyActs =
                averageLayeredForPhase(tokenLog, "reply") ?? layeredReply;
              const finalThoughtActs =
                averageLayeredForPhase(tokenLog, "thought");

              setRawBars({
                outputReply: null,
                outputThought: null,
                thinkingReply: finalReplyActs,
                thinkingThought: finalThoughtActs,
              });

              const wordCount = (thoughtText + " " + fullReply)
                .trim()
                .split(/\s+/)
                .filter(Boolean).length;
              const finalSnap: Snapshot = {
                atWord: wordCount,
                thinkingReply: finalReplyActs,
                thinkingThought: finalThoughtActs,
              };
              const allSnaps = [...snapshots, finalSnap];

              // Post-hoc NLI on both the reply (drives the dot) and the
              // thought block (drives the solid line). Done in parallel.
              const [replySentiment, thoughtSentiment] = await Promise.all([
                classifyOutput(fullReply),
                thoughtText.trim().length > 0
                  ? classifyOutput(thoughtText)
                  : Promise.resolve<EmotionValues | null>(null),
              ]);
              setRawBars({
                outputReply: replySentiment,
                outputThought: thoughtSentiment,
                thinkingReply: finalReplyActs,
                thinkingThought: finalThoughtActs,
              });

              const newTurn: Turn = {
                id: crypto.randomUUID(),
                userMessage: text,
                assistantReply: fullReply,
                assistantThought: thoughtText,
                snapshots: allSnaps,
                tokens: tokenLog,
                state: {
                  outputReply: replySentiment,
                  outputThought: thoughtSentiment,
                  thinkingReply: finalReplyActs,
                  thinkingThought: finalThoughtActs,
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
      if (!completed && (accumulatedReply.trim().length > 0 || accumulatedThought.trim().length > 0)) {
        const truncatedReply = accumulatedReply.trim().length > 0
          ? `${accumulatedReply.trim()} […]`
          : "";
        const truncatedThought = accumulatedThought.trim().length > 0
          ? `${accumulatedThought.trim()} […]`
          : "";
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: truncatedReply || "[…]",
            thought: truncatedThought || undefined,
          },
        ]);
        const [fallbackReplySentiment, fallbackThoughtSentiment] = await Promise.all([
          truncatedReply.length > 0
            ? classifyOutput(truncatedReply)
            : Promise.resolve<EmotionValues | null>(null),
          truncatedThought.length > 0
            ? classifyOutput(truncatedThought)
            : Promise.resolve<EmotionValues | null>(null),
        ]);
        const fallbackReplyActs =
          averageLayeredForPhase(tokenLog, "reply") ?? layeredReply;
        const fallbackThoughtActs =
          averageLayeredForPhase(tokenLog, "thought");
        setRawBars({
          outputReply: fallbackReplySentiment,
          outputThought: fallbackThoughtSentiment,
          thinkingReply: fallbackReplyActs,
          thinkingThought: fallbackThoughtActs,
        });
        const fallbackSnap: Snapshot = {
          atWord: 0,
          thinkingReply: fallbackReplyActs,
          thinkingThought: fallbackThoughtActs,
        };
        const newTurn: Turn = {
          id: crypto.randomUUID(),
          userMessage: text,
          assistantReply: truncatedReply,
          assistantThought: truncatedThought,
          snapshots: snapshots.length > 0 ? [...snapshots] : [fallbackSnap],
          tokens: tokenLog,
          state: {
            outputReply: fallbackReplySentiment,
            outputThought: fallbackThoughtSentiment,
            thinkingReply: fallbackReplyActs,
            thinkingThought: fallbackThoughtActs,
          },
        };
        setTurns((t) => {
          const next = [...t, newTurn];
          setViewingIndex(next.length - 1);
          setSnapshotIndex(Math.max(0, newTurn.snapshots.length - 1));
          return next;
        });
      }
      setStreamingContent(null);
      setStreamingThought(null);
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
      setRawBars({
        outputReply: turn.state.outputReply,
        outputThought: turn.state.outputThought,
        thinkingReply: snap.thinkingReply,
        thinkingThought: snap.thinkingThought,
      });
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
        setRawBars({
          outputReply: turn.state.outputReply,
          outputThought: turn.state.outputThought,
          thinkingReply: snap.thinkingReply,
          thinkingThought: snap.thinkingThought,
        });
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
      <div className="w-[45%] border-r border-divider">
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
          thoughtLayer={thoughtLayer}
          replyLayer={replyLayer}
          onThoughtLayerChange={setThoughtLayer}
          onReplyLayerChange={setReplyLayer}
          classifierOn={classifierOn}
          onClassifierToggle={setClassifierOn}
          valuesOn={valuesOn}
          onValuesToggle={setValuesOn}
          replyStarted={replyStarted}
        />
      </div>
      <div className="w-[55%]">
        <ChatPane
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          onStop={handleStopGeneration}
          isGenerating={isGenerating}
          streamingText={streamingContent}
          streamingThought={streamingThought}
          error={error}
          onDismissError={() => setError(null)}
        />
      </div>
    </>
  );
}
