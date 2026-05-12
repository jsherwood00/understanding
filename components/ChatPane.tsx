"use client";

import { memo, useEffect, useRef, useState, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Internal-reasoning block emitted by the model before the reply,
   *  when thinking mode is enabled. Rendered as a collapsible italic
   *  block above the reply. Undefined for user messages and for replies
   *  generated without thinking. */
  thought?: string;
}

interface ChatPaneProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (next: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isGenerating: boolean;
  streamingText: string | null;
  /** Live thought-block content during streaming. Null when the turn
   *  isn't streaming or when thinking mode is off. */
  streamingThought: string | null;
  error: string | null;
  onDismissError: () => void;
}

export function ChatPane({
  messages,
  input,
  onInputChange,
  onSubmit,
  onStop,
  isGenerating,
  streamingText,
  streamingThought,
  error,
  onDismissError,
}: ChatPaneProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // Tracks whether the user is "near bottom" — when they are, new
  // content auto-scrolls; when they've scrolled up to read, we leave
  // them where they are.
  const stickyBottomRef = useRef(true);

  // Listen for user-driven scroll changes and update the stickiness flag.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handler = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickyBottomRef.current = distFromBottom < 40;
    };
    el.addEventListener("scroll", handler, { passive: true });
    return () => el.removeEventListener("scroll", handler);
  }, []);

  // Auto-scroll on new content — but only when the user was already at
  // (or near) the bottom. If they've scrolled up to read, we don't yank
  // them back down.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickyBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isGenerating, streamingText, streamingThought]);

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (isGenerating || !input.trim()) return;
    onSubmit();
  }

  const hasStreamingContent =
    streamingText !== null && streamingText.length > 0;
  const hasStreamingThought =
    streamingThought !== null && streamingThought.length > 0;
  const showTypingDots =
    isGenerating && !hasStreamingContent && !hasStreamingThought;
  const isEmpty =
    messages.length === 0 &&
    !isGenerating &&
    !streamingText &&
    !streamingThought;

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
            {(hasStreamingContent || hasStreamingThought) && (
              <MessageBubble
                message={{
                  id: "streaming",
                  role: "assistant",
                  content: streamingText ?? "",
                  thought: hasStreamingThought
                    ? streamingThought ?? undefined
                    : undefined,
                }}
                streaming
              />
            )}
            {showTypingDots && <StreamingPlaceholder />}
            {error && (
              <ErrorNotice message={error} onDismiss={onDismissError} />
            )}
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
              isGenerating
                ? "the model is responding…"
                : "say something to the model"
            }
            disabled={isGenerating}
            className="flex-1 border-b border-divider bg-transparent py-2 text-base text-ink placeholder:text-ink-faint placeholder:italic focus:border-ink focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
            autoFocus
          />
          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              className="rounded-full border border-ink/30 px-4 py-1.5 text-sm font-medium text-ink-soft transition-colors hover:border-ink hover:text-ink"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-full border border-ink/15 px-4 py-1.5 text-sm font-medium text-ink transition-colors hover:border-ink hover:bg-ink hover:text-canvas disabled:cursor-not-allowed disabled:border-ink/10 disabled:text-ink-faint disabled:hover:bg-transparent disabled:hover:text-ink-faint"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </section>
  );
}

// Memoized: every parent re-render (e.g. when the user highlights text and
// `selectedExcerpt` changes upstream) would otherwise rebuild the bubble's
// DOM via react-markdown, which invalidates the browser's selection range
// and visually drops the highlight as soon as the mouse releases.
const MessageBubble = memo(function MessageBubble({
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
      {message.thought && (
        <ThoughtDisclosure streaming={streaming}>
          {message.thought}
          {streaming && !message.content && (
            <span className="stream-caret" aria-hidden />
          )}
        </ThoughtDisclosure>
      )}
      <div className="text-[15px] leading-relaxed text-ink">
        <ReactMarkdown
          components={{
            p: ({ children }) => (
              <p className="mb-3 last:mb-0">{children}</p>
            ),
            h1: ({ children }) => (
              <h1 className="mb-2 mt-4 text-lg font-semibold text-ink first:mt-0">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="mb-2 mt-4 text-base font-semibold text-ink first:mt-0">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="mb-2 mt-3 text-[15px] font-semibold text-ink first:mt-0">
                {children}
              </h3>
            ),
            ul: ({ children }) => (
              <ul className="mb-3 ml-5 list-disc space-y-1 last:mb-0">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="mb-3 ml-5 list-decimal space-y-1 last:mb-0">
                {children}
              </ol>
            ),
            li: ({ children }) => <li className="pl-1">{children}</li>,
            strong: ({ children }) => (
              <strong className="font-semibold text-ink">{children}</strong>
            ),
            em: ({ children }) => <em className="italic">{children}</em>,
            code: ({ children }) => (
              <code className="rounded bg-tint px-1 py-0.5 font-mono text-[13px]">
                {children}
              </code>
            ),
            pre: ({ children }) => (
              <pre className="mb-3 overflow-x-auto rounded-md bg-tint p-3 font-mono text-[13px] last:mb-0">
                {children}
              </pre>
            ),
            hr: () => <hr className="my-3 border-divider" />,
            blockquote: ({ children }) => (
              <blockquote className="mb-3 border-l-2 border-divider pl-3 text-ink-soft last:mb-0">
                {children}
              </blockquote>
            ),
          }}
        >
          {message.content}
        </ReactMarkdown>
        {streaming && <span className="stream-caret" aria-hidden />}
      </div>
    </div>
  );
});

/** Collapsible thoughts panel. Opens by default so the user can read the
 *  model's reasoning as it streams; toggle state is owned by component-
 *  local React state so manual clicks stick (the prop-driven `open` would
 *  get re-applied on every re-render and ignore the user's toggle).
 *  Resets to open whenever the bubble is re-mounted (i.e. a new turn). */
function ThoughtDisclosure({
  streaming = false,
  children,
}: {
  streaming?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <details
      className="group self-stretch"
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="thought-summary cursor-pointer list-none text-[12px] italic text-ink-muted hover:text-ink-soft">
        {streaming ? "thinking…" : "thoughts"}
      </summary>
      <div className="mt-1.5 rounded-md border-l-2 border-ink-faint/40 bg-tint/60 px-3 py-2 font-serif text-[13px] leading-relaxed whitespace-pre-wrap text-ink-muted italic">
        {children}
      </div>
    </details>
  );
}

function StreamingPlaceholder() {
  return (
    <div className="flex max-w-[78%] flex-col items-start gap-2">
      <TypingIndicator />
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1 text-ink-muted">
      <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
      <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
      <span className="typing-dot inline-block h-1.5 w-1.5 rounded-full bg-current" />
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
          <span className="font-medium text-ink">
            Couldn&apos;t reach the model.
          </span>{" "}
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
