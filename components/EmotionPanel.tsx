import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionValues,
} from "@/lib/emotions";

interface EmotionPanelProps {
  values: EmotionValues;
}

export function EmotionPanel({ values }: EmotionPanelProps) {
  return (
    <section className="flex h-full min-h-0 flex-col px-10 py-8">
      <div className="smallcaps text-ink-muted">Internal state</div>

      <div className="mt-8 flex min-h-0 flex-1 items-stretch gap-3">
        {EMOTIONS.map((emotion) => (
          <Bar key={emotion} emotion={emotion} value={values[emotion]} />
        ))}
      </div>

      <SurfaceEmotionPreview />
    </section>
  );
}

function Bar({ emotion, value }: { emotion: Emotion; value: number }) {
  const color = EMOTION_COLORS[emotion];
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center">
      <span className="tabular text-xs font-medium text-ink-soft">
        {Math.round(value)}
      </span>
      <div
        className="relative my-3 w-full max-w-[44px] flex-1 overflow-hidden rounded-sm"
        style={{ backgroundColor: `${color}1a` }}
      >
        <div
          className="absolute right-0 bottom-0 left-0 rounded-sm"
          style={{
            height: `${Math.max(0, Math.min(100, value))}%`,
            backgroundColor: color,
            transition: "height 300ms ease-out",
          }}
        />
      </div>
      <span className="smallcaps text-ink-muted">{emotion}</span>
    </div>
  );
}

function SurfaceEmotionPreview() {
  return (
    <div className="mt-8 select-none">
      <div className="flex items-center gap-3">
        <span className="smallcaps text-ink-faint">Surface emotion</span>
        <span className="rounded-full border border-ink-faint/60 px-2 py-0.5 text-[10px] tracking-[0.14em] text-ink-faint uppercase">
          coming soon
        </span>
      </div>
      <div className="mt-3 flex h-10 gap-3 opacity-30">
        {EMOTIONS.map((emotion) => (
          <div
            key={emotion}
            className="flex h-full min-w-0 flex-1 flex-col items-center justify-end"
          >
            <div
              className="w-full max-w-[44px] rounded-sm"
              style={{
                backgroundColor: EMOTION_COLORS[emotion],
                height: PLACEHOLDER_HEIGHTS[emotion],
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

// Static placeholder bar heights — just shapes, not meaningful values.
const PLACEHOLDER_HEIGHTS: Record<Emotion, string> = {
  Joy: "55%",
  Sadness: "30%",
  Anger: "20%",
  Fear: "40%",
  Disgust: "25%",
  Surprise: "45%",
};
