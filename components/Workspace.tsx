"use client";

import { useEffect, useState } from "react";
import { BASELINE, type EmotionValues } from "@/lib/emotions";
import { RESPONSES } from "@/lib/responses";
import { ChatPane, type ChatMessage } from "@/components/ChatPane";
import { EmotionPanel } from "@/components/EmotionPanel";

const TYPING_DELAY_MS = 800;
const STREAM_INTERVAL_MS = 20;
const SETTLE_DELAY_MS = 220;

export function Workspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [streamingFull, setStreamingFull] = useState<string | null>(null);
  const [streamedChars, setStreamedChars] = useState(0);
  const [responseIndex, setResponseIndex] = useState(0);
  const [pendingProfile, setPendingProfile] = useState<EmotionValues | null>(
    null,
  );
  const [bars, setBars] = useState<EmotionValues>(BASELINE);

  const isStreaming = streamingFull !== null;
  const isLocked = isTyping || isStreaming;
  const streamingText = streamingFull
    ? streamingFull.slice(0, streamedChars)
    : null;

  // Character streamer: emit one char per STREAM_INTERVAL_MS while a response is being delivered.
  useEffect(() => {
    if (streamingFull === null) return;
    const id = setInterval(() => {
      setStreamedChars((c) => (c < streamingFull.length ? c + 1 : c));
    }, STREAM_INTERVAL_MS);
    return () => clearInterval(id);
  }, [streamingFull]);

  // Completion: when full text has been emitted, finalize the message and apply the new profile.
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
      if (pendingProfile) {
        setBars(pendingProfile);
        setPendingProfile(null);
      }
    }, SETTLE_DELAY_MS);
    return () => clearTimeout(id);
  }, [streamingFull, streamedChars, pendingProfile]);

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
      setIsTyping(false);
      setStreamedChars(0);
      setPendingProfile(next.profile);
      setStreamingFull(next.text);
    }, TYPING_DELAY_MS);
  }

  return (
    <>
      <div className="w-2/5 border-r border-divider">
        <EmotionPanel values={bars} />
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
