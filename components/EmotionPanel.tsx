import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
} from "@/lib/emotions";

interface EmotionPanelProps {
  state: EmotionState;
}

export function EmotionPanel({ state }: EmotionPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col px-10 py-8">
      <div className="flex items-baseline justify-between">
        <span className="smallcaps text-ink-muted">Internal state</span>
        <Legend />
      </div>

      <div className="mt-8 flex min-h-0 flex-1 items-stretch gap-3">
        {EMOTIONS.map((emotion) => (
          <Bar
            key={emotion}
            emotion={emotion}
            internal={state.internal[emotion]}
            surface={state.surface[emotion]}
          />
        ))}
      </div>
    </section>
  );
}

function Bar({
  emotion,
  internal,
  surface,
}: {
  emotion: Emotion;
  internal: number;
  surface: number;
}) {
  const color = EMOTION_COLORS[emotion];
  const internalPct = clamp(internal);
  const surfacePct = clamp(surface);
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center">
      <span className="tabular text-xs leading-tight">
        <span className="font-medium text-ink-soft">
          {Math.round(internal)}
        </span>
        <span className="mx-0.5 text-ink-faint">·</span>
        <span className="text-ink-faint">{Math.round(surface)}</span>
      </span>
      <div
        className="relative my-3 w-full max-w-[44px] flex-1 overflow-hidden rounded-sm"
        style={{ backgroundColor: `${color}1a` }}
      >
        {/* Surface — full width, medium opacity */}
        <div
          className="absolute right-0 bottom-0 left-0 rounded-sm"
          style={{
            height: `${surfacePct}%`,
            backgroundColor: `${color}66`,
            transition: "height 300ms ease-out",
          }}
        />
        {/* Internal — narrower center column, full opacity */}
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2 rounded-sm"
          style={{
            height: `${internalPct}%`,
            width: "40%",
            backgroundColor: color,
            transition: "height 300ms ease-out",
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
        <span className="h-2 w-1 rounded-sm bg-ink-soft" />
        internal
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2.5 rounded-sm bg-ink-soft/40" />
        surface
      </span>
    </div>
  );
}

function clamp(n: number) {
  return Math.max(0, Math.min(100, n));
}
