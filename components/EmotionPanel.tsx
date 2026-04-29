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
  snapshotIndex: number;
  selectedExcerpt: string | null;
  onNavigate: (direction: -1 | 1) => void;
  onScrub: (snapIdx: number) => void;
  onReplayTurn: () => void;
  onReplayAll: () => void;
  onStopReplay: () => void;
  isReplaying: boolean;
  isGenerating: boolean;
}

export function EmotionPanel(props: EmotionPanelProps) {
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
            thinking={props.state.thinking[emotion]}
            output={props.state.output[emotion]}
          />
        ))}
      </div>

      {props.selectedExcerpt ? (
        <ExcerptIndicator text={props.selectedExcerpt} />
      ) : (
        <TurnNavigator {...props} />
      )}
    </section>
  );
}

function ExcerptIndicator({ text }: { text: string }) {
  return (
    <div className="mt-8 border-t border-divider pt-4">
      <div className="text-[10px] tracking-[0.16em] text-ink-faint uppercase">
        Selected excerpt
      </div>
      <div className="mt-2 line-clamp-3 text-[12px] text-ink-muted italic">
        “{text}”
      </div>
    </div>
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
  const stripColor = `color-mix(in srgb, ${color} 60%, #1a1a1a)`;
  const capColor = `color-mix(in srgb, ${color} 90%, #1a1a1a)`;

  const lo = Math.min(thinkingPct, outputPct);
  const hi = Math.max(thinkingPct, outputPct);
  const span = Math.max(0.5, hi - lo);

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
        {/* Filled strip from min(thinking, output) → max */}
        <div
          className="absolute right-0 left-0 rounded-sm"
          style={{
            bottom: `${lo}%`,
            height: `${span}%`,
            backgroundColor: stripColor,
            opacity: 0.55,
            transition: "bottom 300ms ease-out, height 300ms ease-out",
          }}
        />
        {/* Output cap — solid line */}
        <div
          className="absolute right-0 left-0"
          style={{
            bottom: `calc(${outputPct}% - 1px)`,
            height: "2px",
            backgroundColor: capColor,
            transition: "bottom 300ms ease-out",
          }}
        />
        {/* Thinking cap — dotted line */}
        <div
          className="absolute right-0 left-0"
          style={{
            bottom: `calc(${thinkingPct}% - 1px)`,
            height: "2px",
            backgroundImage: `repeating-linear-gradient(to right, ${capColor} 0 3px, transparent 3px 6px)`,
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
    <div className="flex items-center gap-3 text-[10px] tracking-[0.12em] text-ink-faint uppercase">
      <span className="flex items-center gap-1.5">
        <span
          className="block h-[2px] w-3"
          style={{
            backgroundImage:
              "repeating-linear-gradient(to right, #4a4a4a 0 3px, transparent 3px 6px)",
          }}
        />
        thinking
      </span>
      <span className="flex items-center gap-1.5">
        <span className="block h-[2px] w-3 bg-ink-soft" />
        output
      </span>
    </div>
  );
}

function TurnNavigator(props: Omit<EmotionPanelProps, "state">) {
  const turns = props.turns ?? [];
  const {
    viewingIndex,
    snapshotIndex,
    onNavigate,
    onScrub,
    onReplayTurn,
    onReplayAll,
    onStopReplay,
    isReplaying,
    isGenerating,
  } = props;

  if (turns.length === 0) return <div className="mt-8 h-12" />;

  const turn = viewingIndex !== null ? turns[viewingIndex] : null;
  const preview = turn ? previewWords(turn.userMessage, 10) : "";
  const snapCount = turn?.snapshots.length ?? 0;
  const sliderMax = Math.max(0, snapCount - 1);
  const isAtStart = viewingIndex === null || viewingIndex === 0;
  const isAtEnd =
    viewingIndex === null || viewingIndex === turns.length - 1;
  const navDisabled = isReplaying || isGenerating;
  const scrubDisabled = navDisabled || snapCount <= 1;

  return (
    <div className="mt-8 border-t border-divider pt-4">
      <div className="mb-3 truncate text-[12px] text-ink-muted italic">
        {preview ? `"${preview}"` : "—"}
      </div>

      <div className="flex items-center gap-3 text-xs">
        {/* Snapshot scrubber on the left, takes the bulk of the row */}
        <div className="flex flex-1 items-center gap-2">
          <input
            type="range"
            min={0}
            max={sliderMax}
            value={Math.min(snapshotIndex, sliderMax)}
            onChange={(e) => onScrub(Number(e.target.value))}
            disabled={scrubDisabled}
            className="h-1 flex-1 cursor-pointer accent-ink-soft disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Scrub through chunks of this turn"
          />
          <span className="tabular text-[10px] text-ink-faint">
            {Math.min(snapshotIndex, sliderMax) + 1}/{Math.max(1, snapCount)}
          </span>
        </div>

        {/* Turn navigator on the right */}
        <div className="flex items-center gap-1">
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
            {viewingIndex !== null ? viewingIndex + 1 : 0}/{turns.length}
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
      </div>

      {/* Replay actions */}
      <div className="mt-2 flex justify-end gap-3 text-[11px] tracking-wide text-ink-muted uppercase">
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
