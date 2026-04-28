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
  // Bar (thinking): mid-saturation tint, gentle.
  // Line (output): darker contrast, visible against bar fill OR canvas.
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
        {/* Thinking — bar fill from 0 to thinking%, lighter shade */}
        <div
          className="absolute right-0 bottom-0 left-0 rounded-sm"
          style={{
            height: `${thinkingPct}%`,
            backgroundColor: barColor,
            transition: "height 300ms ease-out",
          }}
        />
        {/* Output — 2px tick line at output% height, darker shade */}
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

function clamp(n: number) {
  return Math.max(0, Math.min(100, n));
}
