"use client";

import {
  EMOTIONS,
  EMOTION_COLORS,
  type Emotion,
  type EmotionState,
} from "@/lib/emotions";
import geometry from "@/lib/emotion_geometry.json";
import { ViewModeToggle, type ViewMode } from "@/components/ViewModeToggle";

interface EmotionGeometryProps {
  state: EmotionState;
  scope: "thought" | "reply";
  viewMode: ViewMode;
  onViewModeChange: (next: ViewMode) => void;
}

const SHORT_LABEL: Record<Emotion, string> = {
  Joy: "JOY",
  Sadness: "SAD",
  Anger: "ANG",
  Fear: "FEAR",
  Disgust: "DISG",
  Surprise: "SURP",
};

// Map TitleCase Emotion to lowercase key used in the JSON.
const EMO_KEY: Record<Emotion, string> = {
  Joy: "joy",
  Sadness: "sadness",
  Anger: "anger",
  Fear: "fear",
  Disgust: "disgust",
  Surprise: "surprise",
};

// Canvas geometry. Drawn into a square SVG; coords from JSON are in
// [-1, 1] and scaled to (PADDING, SIZE-PADDING). PADDING is generous
// so labels offset outside the indicator stay clear of the panel edges.
const SIZE = 520;
const PADDING = 95;
const MIN_RADIUS = 4;
const MAX_RADIUS = 36;

type ScopeKey = "reply" | "thought";

interface ScopeGeom {
  layer: number;
  stress: number;
  coords: Record<string, [number, number]>;
  cosine_matrix: Record<string, Record<string, number>>;
}
const GEOM = geometry as unknown as {
  reply: ScopeGeom;
  thought: ScopeGeom;
  _meta: Record<string, unknown>;
};

function scaleX(x: number): number {
  return PADDING + ((x + 1) / 2) * (SIZE - 2 * PADDING);
}
function scaleY(y: number): number {
  // Flip Y so positive y in MDS coords points UP on screen.
  return PADDING + ((1 - y) / 2) * (SIZE - 2 * PADDING);
}

function radiusFor(pct: number | null | undefined): number {
  if (pct === null || pct === undefined) return 0;
  const clamped = Math.max(0, Math.min(100, pct));
  return MIN_RADIUS + (clamped / 100) * (MAX_RADIUS - MIN_RADIUS);
}

export function EmotionGeometry({
  state,
  scope,
  viewMode,
  onViewModeChange,
}: EmotionGeometryProps) {
  // One embedding at a time. `scope` selects which 6-point layout AND
  // which set of percentiles to render. Thought map = blurry colored
  // clouds at thought MDS coords sized by thinkingThought; output map =
  // hollow rings at reply MDS coords sized by thinkingReply.
  const isThought = scope === "thought";
  const geom = isThought ? GEOM.thought : GEOM.reply;
  const layerNumber = geom.layer;
  const stress = geom.stress;

  return (
    <section className="flex h-full min-h-0 flex-col px-6 py-6">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="text-[10px] tracking-[0.16em] text-ink-faint uppercase">
          {isThought ? "thought map" : "output map"}
          <span className="ml-2 text-ink-faint/70 normal-case tracking-normal">
            ({isThought ? "V2" : "V3"} L{layerNumber}, stress {stress.toFixed(2)})
          </span>
        </div>
        <ViewModeToggle mode={viewMode} onChange={onViewModeChange} />
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="h-full w-full"
          aria-label={`Emotion ${scope}-scope MDS map`}
        >
          {/* Indicator layer */}
          {EMOTIONS.map((emotion) => {
            const key = EMO_KEY[emotion];
            const [px, py] = geom.coords[key];
            const pct = isThought
              ? state.thinkingThought
                ? state.thinkingThought[emotion]
                : null
              : state.thinkingReply[emotion];
            const r = radiusFor(pct);
            if (r <= 0) return null;
            const cx = scaleX(px);
            const cy = scaleY(py);
            const color = EMOTION_COLORS[emotion];
            if (isThought) {
              const haloColor = `color-mix(in srgb, ${color} 70%, #faf9f6)`;
              return (
                <circle
                  key={`indicator-${emotion}`}
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={haloColor}
                  opacity={0.5}
                  style={{
                    filter: "blur(4px)",
                    transition: "r 300ms ease-out",
                  }}
                />
              );
            }
            const ringColor = `color-mix(in srgb, ${color} 80%, #1a1a1a)`;
            return (
              <circle
                key={`indicator-${emotion}`}
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke={ringColor}
                strokeWidth={2}
                style={{ transition: "r 300ms ease-out" }}
              />
            );
          })}

          {/* Labels + numeric percentile, offset away from canvas center
              so they don't overlap the circle at any radius. */}
          {EMOTIONS.map((emotion) => {
            const key = EMO_KEY[emotion];
            const [px, py] = geom.coords[key];
            const cx = scaleX(px);
            const cy = scaleY(py);
            const pct = isThought
              ? state.thinkingThought
                ? state.thinkingThought[emotion]
                : null
              : state.thinkingReply[emotion];
            const dx = cx - SIZE / 2;
            const dy = cy - SIZE / 2;
            const len = Math.hypot(dx, dy) || 1;
            const offset = MAX_RADIUS + 12;
            const lx = cx + (dx / len) * offset;
            const ly = cy + (dy / len) * offset;
            const numStr = pct === null || pct === undefined
              ? "--"
              : String(Math.round(pct));
            return (
              <g key={`label-${emotion}`} style={{ pointerEvents: "none" }}>
                <text
                  x={lx}
                  y={ly - 7}
                  fontSize={12}
                  fontFamily="ui-monospace, monospace"
                  fontWeight="bold"
                  textAnchor="middle"
                  alignmentBaseline="middle"
                  fill="#1a1a1a"
                  style={{ letterSpacing: "0.08em" }}
                >
                  {SHORT_LABEL[emotion]}
                </text>
                <text
                  x={lx}
                  y={ly + 7}
                  fontSize={12}
                  fontFamily="ui-monospace, monospace"
                  textAnchor="middle"
                  alignmentBaseline="middle"
                  fill="#5a564f"
                  style={{ letterSpacing: "0.04em" }}
                >
                  {numStr}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
