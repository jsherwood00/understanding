import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
  type Turn,
} from "@/lib/emotions";

const RADAR_OUTPUT_COLOR = "#3F6F9A"; // cool deep blue — the visible
const RADAR_THINKING_COLOR = "#B07A2C"; // amber — the underlying

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

      <div className="mt-6 flex min-h-0 flex-1 items-center justify-center">
        <Radar state={props.state} />
      </div>

      {props.selectedExcerpt ? (
        <ExcerptIndicator text={props.selectedExcerpt} />
      ) : (
        <TurnNavigator {...props} />
      )}
    </section>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-3 text-[10px] tracking-[0.12em] text-ink-faint uppercase">
      <span className="flex items-center gap-1.5">
        <span
          className="h-2 w-2.5 rounded-sm"
          style={{ backgroundColor: RADAR_OUTPUT_COLOR }}
        />
        output
      </span>
      <span className="flex items-center gap-1.5">
        <span
          className="h-2 w-2.5 rounded-sm"
          style={{ backgroundColor: RADAR_THINKING_COLOR }}
        />
        thinking
      </span>
    </div>
  );
}

function Radar({ state }: { state: EmotionState }) {
  const size = 320;
  const center = size / 2;
  const maxRadius = center * 0.62;

  function angleFor(i: number) {
    // Top of the ring is Joy (i=0), going clockwise.
    return -Math.PI / 2 + (i * 2 * Math.PI) / EMOTIONS.length;
  }

  function point(i: number, ratio: number) {
    const angle = angleFor(i);
    const r = ratio * maxRadius;
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle),
    };
  }

  function polyPoints(values: EmotionState["output"]): string {
    return EMOTIONS.map((e, i) => {
      const p = point(i, Math.max(0, Math.min(1, values[e] / 100)));
      return `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
    }).join(" ");
  }

  function gridPoints(ratio: number): string {
    return EMOTIONS.map((_, i) => {
      const p = point(i, ratio);
      return `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
    }).join(" ");
  }

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="aspect-square h-full w-full max-w-[520px]"
      role="img"
      aria-label="Emotion radar"
    >
      {/* Concentric grid hexagons */}
      {[0.25, 0.5, 0.75, 1].map((ratio) => (
        <polygon
          key={ratio}
          points={gridPoints(ratio)}
          fill="none"
          stroke="rgba(26, 26, 26, 0.07)"
          strokeWidth="1"
        />
      ))}

      {/* Axis lines + emotion-tinted endpoints */}
      {EMOTIONS.map((emotion, i) => {
        const p = point(i, 1);
        return (
          <line
            key={emotion}
            x1={center}
            y1={center}
            x2={p.x}
            y2={p.y}
            stroke={`color-mix(in srgb, ${EMOTION_COLORS[emotion]} 40%, #faf9f6)`}
            strokeWidth="1"
          />
        );
      })}

      {/* Output polygon (drawn first / behind) */}
      <polygon
        points={polyPoints(state.output)}
        fill={RADAR_OUTPUT_COLOR}
        fillOpacity="0.18"
        stroke={RADAR_OUTPUT_COLOR}
        strokeWidth="1.6"
        strokeLinejoin="round"
        style={{ transition: "all 300ms ease-out" }}
      />

      {/* Thinking polygon (drawn second / on top, dashed) */}
      <polygon
        points={polyPoints(state.thinking)}
        fill={RADAR_THINKING_COLOR}
        fillOpacity="0.18"
        stroke={RADAR_THINKING_COLOR}
        strokeWidth="1.6"
        strokeLinejoin="round"
        strokeDasharray="4 3"
        style={{ transition: "all 300ms ease-out" }}
      />

      {/* Vertex dots — emotion color, sized by output value (subtle anchor) */}
      {EMOTIONS.map((emotion, i) => {
        const p = point(i, 1);
        return (
          <circle
            key={`${emotion}-dot`}
            cx={p.x}
            cy={p.y}
            r="2.5"
            fill={EMOTION_COLORS[emotion]}
          />
        );
      })}

      {/* Emotion labels just outside the ring */}
      {EMOTIONS.map((emotion, i) => {
        const p = point(i, 1.18);
        return (
          <text
            key={`${emotion}-label`}
            x={p.x}
            y={p.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#7a7a76"
            style={{
              fontSize: "10px",
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              fontWeight: 500,
            }}
          >
            {emotion}
          </text>
        );
      })}

      {/* Numeric value at each axis (small, near the dot) */}
      {EMOTIONS.map((emotion, i) => {
        const out = Math.round(state.output[emotion]);
        const think = Math.round(state.thinking[emotion]);
        if (out === 0 && think === 0) return null;
        const labelP = point(i, 0.88);
        return (
          <text
            key={`${emotion}-val`}
            x={labelP.x}
            y={labelP.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#4a4a4a"
            style={{
              fontSize: "9px",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {think}·{out}
          </text>
        );
      })}
    </svg>
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
