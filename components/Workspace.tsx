"use client";

import { useRef, useState } from "react";
import { BASELINE_STATE, type EmotionState } from "@/lib/emotions";
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
  const abortControllerRef = useRef<AbortController | null>(null);

  async function handleSubmit() {
    const text = input.trim();
    if (!text || isGenerating) return;

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
                setBars((prev) => ({ ...prev, output: data.emotions! }));
              }
              break;
            }
            case "done": {
              completed = true;
              const fullText = data.fullText ?? accumulatedOutput;
              const fullThinking = data.fullThinking ?? finalThinking;
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
                thinking: data.thinkingEmotions ?? BASELINE_STATE.thinking,
                output: data.outputEmotions ?? BASELINE_STATE.output,
              });
              setStreamingContent(null);
              setStreamingThinking(null);
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
      // If we got partial output but never saw "done", preserve it as a
      // truncated assistant message so the user sees what arrived.
      if (!completed && accumulatedOutput.trim().length > 0) {
        setMessages((m) => [
          ...m,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `${accumulatedOutput.trim()} […]`,
            thinking: finalThinking,
          },
        ]);
      }
      setStreamingContent(null);
      setStreamingThinking(null);
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }

  function handleStop() {
    abortControllerRef.current?.abort();
  }

  return (
    <>
      <div className="w-2/5 border-r border-divider">
        <EmotionPanel state={bars} />
      </div>
      <div className="w-3/5">
        <ChatPane
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          onStop={handleStop}
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
