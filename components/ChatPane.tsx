"use client";

import { useEffect, useRef, type FormEvent } from "react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string | null;
}

interface ChatPaneProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (next: string) => void;
  onSubmit: () => void;
  isTyping: boolean;
  streamingText: string | null;
  streamingThinking: string | null;
  isLocked: boolean;
  error: string | null;
  onDismissError: () => void;
}

export function ChatPane({
  messages,
  input,
  onInputChange,
  onSubmit,
  isTyping,
  streamingText,
  streamingThinking,
  isLocked,
  error,
  onDismissError,
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
                  thinking: streamingThinking,
                }}
                streaming
              />
            )}
            {isTyping && <TypingIndicator />}
            {error && <ErrorNotice message={error} onDismiss={onDismissError} />}
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
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] rounded-2xl bg-tint px-4 py-2.5 text-[15px] leading-relaxed text-ink">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex max-w-[78%] flex-col items-start gap-2">
      {message.thinking && <ThinkingDisclosure text={message.thinking} />}
      <div className="text-[15px] leading-relaxed text-ink">
        {message.content}
        {streaming && <span className="stream-caret" aria-hidden />}
      </div>
    </div>
  );
}

function ThinkingDisclosure({ text }: { text: string }) {
  return (
    <details className="group w-full">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-[10px] tracking-[0.16em] text-ink-faint uppercase select-none hover:text-ink-muted">
        <span className="inline-block transition-transform group-open:rotate-90">
          ▸
        </span>
        thinking
      </summary>
      <div className="mt-2 max-w-prose border-l border-divider py-1 pl-3 text-[13px] leading-relaxed whitespace-pre-wrap text-ink-muted italic">
        {text}
      </div>
    </details>
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

function ErrorNotice({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="rounded-md border border-anger/40 bg-anger/5 px-4 py-3 text-[13px] text-ink-soft">
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="font-medium text-ink">Couldn&apos;t reach the model.</span>{" "}
          {message}
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="text-ink-muted hover:text-ink"
          aria-label="Dismiss"
        >
          ×
        </button>
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
