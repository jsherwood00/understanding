"use client";

import { useRef, useState } from "react";
import {
  BASELINE_STATE,
  type EmotionState,
  type Turn,
} from "@/lib/emotions";
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";

interface SSEEvent {
  type: "thinking" | "output_word" | "output_emotions" | "done";
  text?: string;
  emotions?: EmotionState["thinking"];
  outputEmotions?: EmotionState["output"];
  thinkingEmotions?: EmotionState["thinking"];
  fullText?: string;
  fullThinking?: string;
  index?: number;
  words?: number;
}

const REPLAY_TURN_DURATION_MS = 1400;

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
  const [isReplaying, setIsReplaying] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const replayAbortRef = useRef(false);
  const savedViewingRef = useRef<number | null>(null);

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
    let outputEmotions: EmotionState["output"] = BASELINE_STATE.output;
    let thinkingEmotions: EmotionState["thinking"] = BASELINE_STATE.thinking;
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
              if (data.emotions) {
                thinkingEmotions = data.emotions;
                setBars((prev) => ({ ...prev, thinking: data.emotions! }));
              }
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
                setBars((prev) => ({ ...prev, output: data.emotions! }));
              }
              break;
            }
            case "done": {
              completed = true;
              const fullText = data.fullText ?? accumulatedOutput;
              const fullThinking = data.fullThinking ?? finalThinking;
              outputEmotions =
                data.outputEmotions ?? outputEmotions;
              thinkingEmotions =
                data.thinkingEmotions ?? thinkingEmotions;
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
                state: {
                  output: outputEmotions,
                  thinking: thinkingEmotions,
                },
              };
              setTurns((t) => {
                const next = [...t, newTurn];
                setViewingIndex(next.length - 1);
                return next;
              });
              break;
            }
          }
        }
      }
    } catch (e) {
      const isAbort =
        e instanceof DOMException && e.name === "AbortError";
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
          state: { output: outputEmotions, thinking: thinkingEmotions },
        };
        setTurns((t) => {
          const next = [...t, newTurn];
          setViewingIndex(next.length - 1);
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
    setBars(turns[next].state);
  }

  function sleep(ms: number) {
    return new Promise<void>((resolve) => setTimeout(resolve, ms));
  }

  async function handleReplayTurn() {
    if (viewingIndex === null || turns.length === 0) return;
    if (isGenerating || isReplaying) return;
    const turn = turns[viewingIndex];
    replayAbortRef.current = false;
    setIsReplaying(true);
    setBars(BASELINE_STATE);
    await sleep(80);
    if (!replayAbortRef.current) setBars(turn.state);
    await sleep(REPLAY_TURN_DURATION_MS);
    setIsReplaying(false);
  }

  async function handleReplayAll() {
    if (turns.length === 0 || isGenerating || isReplaying) return;
    savedViewingRef.current = viewingIndex;
    replayAbortRef.current = false;
    setIsReplaying(true);
    for (let i = 0; i < turns.length; i++) {
      if (replayAbortRef.current) break;
      setViewingIndex(i);
      setBars(BASELINE_STATE);
      await sleep(80);
      if (replayAbortRef.current) break;
      setBars(turns[i].state);
      await sleep(REPLAY_TURN_DURATION_MS);
    }
    setIsReplaying(false);
  }

  function handleStopReplay() {
    replayAbortRef.current = true;
    if (savedViewingRef.current !== null) {
      const idx = savedViewingRef.current;
      setViewingIndex(idx);
      const turn = turns[idx];
      if (turn) setBars(turn.state);
      savedViewingRef.current = null;
    }
    setIsReplaying(false);
  }

  return (
    <>
      <div className="w-2/5 border-r border-divider">
        <EmotionPanel
          state={bars}
          turns={turns}
          viewingIndex={viewingIndex}
          onNavigate={navigateTurn}
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
