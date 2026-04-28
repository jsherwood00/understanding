import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
  type Turn,
} from "@/lib/emotions";

interface EmotionPanelProps {
  state: EmotionState;
  turns: Turn[];
  viewingIndex: number | null;
  onNavigate: (direction: -1 | 1) => void;
  onReplayTurn: () => void;
  onReplayAll: () => void;
  onStopReplay: () => void;
  isReplaying: boolean;
  isGenerating: boolean;
}

export function EmotionPanel({
  state,
  turns,
  viewingIndex,
  onNavigate,
  onReplayTurn,
  onReplayAll,
  onStopReplay,
  isReplaying,
  isGenerating,
}: EmotionPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col px-10 py-8">
      <div className="flex justify-end">
        <Legend />
      </div>

      <div className="mt-8 flex min-h-0 flex-1 items-stretch gap-3">
        {EMOTIONS.map((emotion) => (
          <Bar
            key={emotion}
            emotion={emotion}
            thinking={state.thinking[emotion]}
            output={state.output[emotion]}
          />
        ))}
      </div>

      <TurnNavigator
        turns={turns}
        viewingIndex={viewingIndex}
        onNavigate={onNavigate}
        onReplayTurn={onReplayTurn}
        onReplayAll={onReplayAll}
        onStopReplay={onStopReplay}
        isReplaying={isReplaying}
        isGenerating={isGenerating}
      />
    </section>
  );
}

function Bar({
  emotion,
  thinking,
  output,
}: {
  emotion: Emotion;
  thinking: number;
  output: number;
}) {
  const color = EMOTION_COLORS[emotion];
  const thinkingPct = clamp(thinking);
  const outputPct = clamp(output);
  const barColor = `color-mix(in srgb, ${color} 38%, #faf9f6)`;
  const lineColor = `color-mix(in srgb, ${color} 78%, #1a1a1a)`;
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center">
      <span className="tabular text-xs leading-tight">
        <span className="font-medium text-ink-soft">
          {Math.round(thinking)}
        </span>
        <span className="mx-0.5 text-ink-faint">·</span>
        <span className="text-ink-faint">{Math.round(output)}</span>
      </span>
      <div className="relative my-3 w-full max-w-[44px] flex-1 overflow-hidden rounded-sm">
        <div
          className="absolute right-0 bottom-0 left-0 rounded-sm"
          style={{
            height: `${thinkingPct}%`,
            backgroundColor: barColor,
            transition: "height 300ms ease-out",
          }}
        />
        <div
          className="absolute right-0 left-0"
          style={{
            bottom: `calc(${outputPct}% - 1px)`,
            height: "2px",
            backgroundColor: lineColor,
            transition: "bottom 300ms ease-out",
          }}
        />
      </div>
      <span className="smallcaps text-ink-muted">{emotion}</span>
    </div>
  );
}

function Legend() {
  return (
    <div className="text-[10px] tracking-[0.12em] text-ink-faint uppercase">
      bar = thinking · line = output
    </div>
  );
}

function TurnNavigator({
  turns,
  viewingIndex,
  onNavigate,
  onReplayTurn,
  onReplayAll,
  onStopReplay,
  isReplaying,
  isGenerating,
}: Omit<EmotionPanelProps, "state">) {
  if (turns.length === 0) return <div className="mt-8 h-12" />;

  const turn = viewingIndex !== null ? turns[viewingIndex] : null;
  const preview = turn ? previewWords(turn.userMessage, 10) : "";
  const isAtStart = viewingIndex === null || viewingIndex === 0;
  const isAtEnd =
    viewingIndex === null || viewingIndex === turns.length - 1;
  const navDisabled = isReplaying || isGenerating;

  return (
    <div className="mt-8 border-t border-divider pt-4">
      <div className="mb-2 truncate text-[12px] text-ink-muted italic">
        {preview ? `"${preview}"` : "—"}
      </div>
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => onNavigate(-1)}
            disabled={navDisabled || isAtStart}
            className="rounded px-1.5 py-0.5 text-ink-soft hover:bg-tint disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:bg-transparent"
            aria-label="Previous turn"
          >
            ‹
          </button>
          <span className="tabular text-ink-muted">
            turn {viewingIndex !== null ? viewingIndex + 1 : 0} / {turns.length}
          </span>
          <button
            type="button"
            onClick={() => onNavigate(1)}
            disabled={navDisabled || isAtEnd}
            className="rounded px-1.5 py-0.5 text-ink-soft hover:bg-tint disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:bg-transparent"
            aria-label="Next turn"
          >
            ›
          </button>
        </div>
        <div className="flex items-center gap-3 text-[11px] tracking-wide text-ink-muted uppercase">
          {isReplaying ? (
            <button
              type="button"
              onClick={onStopReplay}
              className="rounded border border-ink/30 px-2 py-0.5 text-ink-soft hover:border-ink hover:text-ink"
            >
              stop
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onReplayTurn}
                disabled={isGenerating || viewingIndex === null}
                className="hover:text-ink-soft disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:text-ink-faint"
              >
                replay
              </button>
              <button
                type="button"
                onClick={onReplayAll}
                disabled={isGenerating || turns.length === 0}
                className="hover:text-ink-soft disabled:cursor-not-allowed disabled:text-ink-faint disabled:hover:text-ink-faint"
              >
                replay all
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function previewWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text.trim();
  return words.slice(0, maxWords).join(" ") + "…";
}

function clamp(n: number) {
  return Math.max(0, Math.min(100, n));
}
