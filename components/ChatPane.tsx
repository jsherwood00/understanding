"use client";

import { useEffect, useRef, type FormEvent } from "react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ChatPaneProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (next: string) => void;
  onSubmit: () => void;
  isTyping: boolean;
  streamingText: string | null;
  isLocked: boolean;
}

export function ChatPane({
  messages,
  input,
  onInputChange,
  onSubmit,
  isTyping,
  streamingText,
  isLocked,
}: ChatPaneProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new content
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isTyping, streamingText]);

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (isLocked || !input.trim()) return;
    onSubmit();
  }

  const isEmpty = messages.length === 0 && !isTyping && !streamingText;

  return (
    <section className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-10 py-10">
        {isEmpty ? (
          <EmptyState />
        ) : (
          <div className="mx-auto flex max-w-2xl flex-col gap-6">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
            {streamingText !== null && (
              <MessageBubble
                message={{
                  id: "streaming",
                  role: "assistant",
                  content: streamingText,
                }}
                streaming
              />
            )}
            {isTyping && <TypingIndicator />}
          </div>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-divider px-10 py-5"
      >
        <div className="mx-auto flex max-w-2xl items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            placeholder={
              isLocked ? "the model is responding…" : "say something to the model"
            }
            disabled={isLocked}
            className="flex-1 border-b border-divider bg-transparent py-2 text-base text-ink placeholder:text-ink-faint placeholder:italic focus:border-ink focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
            autoFocus
          />
          <button
            type="submit"
            disabled={isLocked || !input.trim()}
            className="rounded-full border border-ink/15 px-4 py-1.5 text-sm font-medium text-ink transition-colors hover:border-ink hover:bg-ink hover:text-canvas disabled:cursor-not-allowed disabled:border-ink/10 disabled:text-ink-faint disabled:hover:bg-transparent disabled:hover:text-ink-faint"
          >
            Send
          </button>
        </div>
      </form>
    </section>
  );
}

function MessageBubble({
  message,
  streaming = false,
}: {
  message: ChatMessage;
  streaming?: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div
        className={
          isUser
            ? "max-w-[78%] rounded-2xl bg-tint px-4 py-2.5 text-[15px] leading-relaxed text-ink"
            : "max-w-[78%] text-[15px] leading-relaxed text-ink"
        }
      >
        {message.content}
        {streaming && <span className="stream-caret" aria-hidden />}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 py-1 text-ink-muted">
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
        <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto flex h-full max-w-md flex-col items-center justify-center text-center">
      <p className="font-serif text-2xl text-ink-soft italic">Begin.</p>
      <p className="mt-3 text-sm text-ink-muted">
        Whatever you say will move the model&apos;s internal state.
      </p>
    </div>
  );
}
