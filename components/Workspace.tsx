"use client";

import { useEffect, useState } from "react";
import {
  BASELINE,
  type EmotionValues,
  drift,
  snapTo,
  stepToward,
} from "@/lib/emotions";
import { RESPONSES } from "@/lib/responses";
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";

const TYPING_DELAY_MS = 800;
const STREAM_INTERVAL_MS = 20;
const ANIMATION_TICK_MS = 300;
const DRIFT_TICK_MS = 1000;

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [streamingFull, setStreamingFull] = useState<string | null>(null);
  const [streamedChars, setStreamedChars] = useState(0);
  const [responseIndex, setResponseIndex] = useState(0);
  const [target, setTarget] = useState<EmotionValues>(BASELINE);
  const [current, setCurrent] = useState<EmotionValues>(BASELINE);

  const isStreaming = streamingFull !== null;
  const isLocked = isTyping || isStreaming;
  const streamingText = streamingFull
    ? streamingFull.slice(0, streamedChars)
    : null;

  // Bar animation: step toward target while streaming, drift around target when idle.
  useEffect(() => {
    const tickMs = isStreaming ? ANIMATION_TICK_MS : DRIFT_TICK_MS;
    const id = setInterval(() => {
      setCurrent((prev) =>
        isStreaming ? stepToward(prev, target, 0.32) : drift(prev, target, 2),
      );
    }, tickMs);
    return () => clearInterval(id);
  }, [isStreaming, target]);

  // Character streamer: emit one char per STREAM_INTERVAL_MS while a response is being delivered.
  useEffect(() => {
    if (streamingFull === null) return;
    const id = setInterval(() => {
      setStreamedChars((c) => (c < streamingFull.length ? c + 1 : c));
    }, STREAM_INTERVAL_MS);
    return () => clearInterval(id);
  }, [streamingFull]);

  // Completion: when full text has been emitted, finalize the message and snap bars to target.
  useEffect(() => {
    if (streamingFull === null) return;
    if (streamedChars < streamingFull.length) return;
    const id = setTimeout(() => {
      const completed = streamingFull;
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), role: "assistant", content: completed },
      ]);
      setStreamingFull(null);
      setStreamedChars(0);
      setCurrent(snapTo(target));
    }, 220);
    return () => clearTimeout(id);
  }, [streamingFull, streamedChars, target]);

  function handleSubmit() {
    const text = input.trim();
    if (!text || isLocked) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((m) => [...m, userMessage]);
    setInput("");
    setIsTyping(true);

    const next = RESPONSES[responseIndex % RESPONSES.length];
    setResponseIndex((i) => i + 1);

    setTimeout(() => {
      setTarget(next.profile);
      setIsTyping(false);
      setStreamedChars(0);
      setStreamingFull(next.text);
    }, TYPING_DELAY_MS);
  }

  return (
    <>
      <div className="w-2/5 border-r border-divider">
        <EmotionPanel values={current} />
      </div>
      <div className="w-3/5">
        <ChatPane
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          isTyping={isTyping}
          streamingText={streamingText}
          isLocked={isLocked}
        />
      </div>
    </>
  );
}
