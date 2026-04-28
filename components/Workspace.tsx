"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  BASELINE_STATE,
  EMOTIONS,
  type EmotionState,
  type EmotionValues,
  type OutputSnapshot,
  type Turn,
} from "@/lib/emotions";
import { analyzeEmotions } from "@/lib/sentiment";
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";

interface SSEEvent {
  type: "thinking" | "output_word" | "output_emotions" | "done";
  text?: string;
  emotions?: EmotionValues;
  outputEmotions?: EmotionValues;
  thinkingEmotions?: EmotionValues;
  fullText?: string;
  fullThinking?: string;
  index?: number;
  words?: number;
}

const REPLAY_PER_SNAPSHOT_MS = 220;
const REPLAY_FINAL_HOLD_MS = 700;
const SELECTION_MIN_CHARS = 3;

function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinking, setStreamingThinking] = useState<string | null>(
    null,
  );
  const [bars, setBars] = useState<EmotionState>(BASELINE_STATE);
  const [error, setError] = useState<string | null>(null);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [viewingIndex, setViewingIndex] = useState<number | null>(null);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);

  const [selectedExcerpt, setSelectedExcerpt] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const replayAbortRef = useRef(false);
  const savedViewRef = useRef<{ turn: number | null; snap: number } | null>(
    null,
  );

  // Listen for text selection anywhere — when a non-trivial selection exists
  // override the bars with that text's analyzed emotions.
  useEffect(() => {
    function onSelectionChange() {
      const selection = window.getSelection();
      if (!selection) {
        setSelectedExcerpt(null);
        return;
      }
      const text = selection.toString().trim();
      if (text.length < SELECTION_MIN_CHARS) {
        setSelectedExcerpt((curr) => (curr === null ? curr : null));
        return;
      }
      setSelectedExcerpt(text);
    }
    document.addEventListener("selectionchange", onSelectionChange);
    return () =>
      document.removeEventListener("selectionchange", onSelectionChange);
  }, []);

  // Bars actually rendered — selection overrides everything else.
  const displayedBars = useMemo<EmotionState>(() => {
    if (selectedExcerpt) {
      return { output: analyzeEmotions(selectedExcerpt), thinking: zeros() };
    }
    return bars;
  }, [selectedExcerpt, bars]);

  function applyTurnView(turnIdx: number, snapIdx: number) {
    const turn = turns[turnIdx];
    if (!turn) return;
    const snap =
      turn.outputSnapshots[
        Math.max(0, Math.min(snapIdx, turn.outputSnapshots.length - 1))
      ];
    setBars({
      output: snap?.emotions ?? turn.state.output,
      thinking: turn.thinkingEmotions,
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
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setIsGenerating(true);
    setStreamingContent("");
    setStreamingThinking(null);
    setBars(BASELINE_STATE);
    setError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accumulatedOutput = "";
    let finalThinking: string | null = null;
    let outputEmotions: EmotionValues = BASELINE_STATE.output;
    let thinkingEmotions: EmotionValues = BASELINE_STATE.thinking;
    const outputSnapshots: OutputSnapshot[] = [];
    let completed = false;

    try {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: nextMessages.map((m) => ({
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

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          let data: SSEEvent;
          try {
            data = JSON.parse(part.slice(6)) as SSEEvent;
          } catch {
            continue;
          }

          switch (data.type) {
            case "thinking": {
              const traceText = data.text ?? "";
              finalThinking = traceText.length > 0 ? traceText : null;
              setStreamingThinking(finalThinking);
              // Thinking emotions are intentionally zero during streaming;
              // the real "thinking" vector is derived from final output at done.
              break;
            }
            case "output_word": {
              const word = data.text ?? "";
              accumulatedOutput = accumulatedOutput
                ? `${accumulatedOutput} ${word}`
                : word;
              setStreamingContent(accumulatedOutput);
              break;
            }
            case "output_emotions": {
              if (data.emotions) {
                outputEmotions = data.emotions;
                outputSnapshots.push({
                  atWord: data.words ?? 0,
                  emotions: data.emotions,
                });
                setBars((prev) => ({ ...prev, output: data.emotions! }));
              }
              break;
            }
            case "done": {
              completed = true;
              const fullText = data.fullText ?? accumulatedOutput;
              const fullThinking = data.fullThinking ?? finalThinking;
              outputEmotions = data.outputEmotions ?? outputEmotions;
              thinkingEmotions = data.thinkingEmotions ?? thinkingEmotions;
              setMessages((m) => [
                ...m,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: fullText,
                  thinking: fullThinking || null,
                },
              ]);
              setBars({
                thinking: thinkingEmotions,
                output: outputEmotions,
              });
              setStreamingContent(null);
              setStreamingThinking(null);
              const newTurn: Turn = {
                id: crypto.randomUUID(),
                userMessage: text,
                assistantReply: fullText,
                thinking: fullThinking || null,
                thinkingEmotions,
                outputSnapshots: [...outputSnapshots],
                state: {
                  output: outputEmotions,
                  thinking: thinkingEmotions,
                },
              };
              setTurns((t) => {
                const next = [...t, newTurn];
                setViewingIndex(next.length - 1);
                setSnapshotIndex(
                  Math.max(0, newTurn.outputSnapshots.length - 1),
                );
                return next;
              });
              break;
            }
          }
        }
      }
    } catch (e) {
      const isAbort = e instanceof DOMException && e.name === "AbortError";
      if (!isAbort) {
        setError(e instanceof Error ? e.message : "Something went wrong.");
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
            thinking: finalThinking,
          },
        ]);
        const newTurn: Turn = {
          id: crypto.randomUUID(),
          userMessage: text,
          assistantReply: truncated,
          thinking: finalThinking,
          thinkingEmotions,
          outputSnapshots:
            outputSnapshots.length > 0
              ? [...outputSnapshots]
              : [{ atWord: 0, emotions: outputEmotions }],
          state: { output: outputEmotions, thinking: thinkingEmotions },
        };
        setTurns((t) => {
          const next = [...t, newTurn];
          setViewingIndex(next.length - 1);
          setSnapshotIndex(Math.max(0, newTurn.outputSnapshots.length - 1));
          return next;
        });
      }
      setStreamingContent(null);
      setStreamingThinking(null);
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
    const last = Math.max(0, turns[next].outputSnapshots.length - 1);
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
    if (turn.outputSnapshots.length === 0) return;
    replayAbortRef.current = false;
    setIsReplaying(true);
    setBars({ output: BASELINE_STATE.output, thinking: turn.thinkingEmotions });
    await sleep(120);
    for (let i = 0; i < turn.outputSnapshots.length; i++) {
      if (replayAbortRef.current) break;
      setSnapshotIndex(i);
      setBars({
        output: turn.outputSnapshots[i].emotions,
        thinking: turn.thinkingEmotions,
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
      setBars({
        output: BASELINE_STATE.output,
        thinking: turn.thinkingEmotions,
      });
      await sleep(120);
      for (let i = 0; i < turn.outputSnapshots.length; i++) {
        if (replayAbortRef.current) break;
        setSnapshotIndex(i);
        setBars({
          output: turn.outputSnapshots[i].emotions,
          thinking: turn.thinkingEmotions,
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
          streamingThinking={streamingThinking}
          error={error}
          onDismissError={() => setError(null)}
        />
      </div>
    </>
  );
}
