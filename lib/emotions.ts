export const EMOTIONS = [
  "Joy",
  "Sadness",
  "Anger",
  "Fear",
  "Disgust",
  "Surprise",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export type EmotionValues = Record<Emotion, number>;

export const EMOTION_COLORS: Record<Emotion, string> = {
  Joy: "#FFD93D",
  Sadness: "#4A90E2",
  Anger: "#E94B3C",
  Fear: "#8E5DD8",
  Disgust: "#7CB342",
  Surprise: "#26C6DA",
};

export const BASELINE: EmotionValues = {
  Joy: 0,
  Sadness: 0,
  Anger: 0,
  Fear: 0,
  Disgust: 0,
  Surprise: 0,
};

export const LAYERS = [13, 17, 21, 25, 28, 32] as const;
export type Layer = (typeof LAYERS)[number];

/** Per-scope "best layer" by 6-way holdout accuracy on the contrastive
 *  corpus (see data/thinking/vectors/<scope>/cutoff_50/
 *  holdout_classification_topic_level.json):
 *    reply   — L25 wins (~54% topic-level accuracy)
 *    thought — L13 wins (~49% topic-level accuracy)
 *  Used as the default selection AND rendered in bold in the selector. */
export const BEST_LAYER: Record<"thought" | "reply", Layer> = {
  thought: 13,
  reply: 25,
};

/** Map of layer → 6-emotion vector. Backend now ships all 6 layers per
 *  token; the frontend stores the full layered shape and picks one to
 *  display so layer-switching is instant. */
export type LayeredEmotionValues = Record<Layer, EmotionValues>;

export function makeLayeredBaseline(): LayeredEmotionValues {
  const out = {} as LayeredEmotionValues;
  for (const L of LAYERS) out[L] = { ...BASELINE };
  return out;
}

/** Four indicators per emotion column:
 *    outputReply     — dot:         NLI sentiment of the reply text
 *    outputThought   — solid line:  NLI sentiment of the thought block
 *    thinkingReply   — halo (cloud): residual-projection avg over reply tokens (V3 vectors)
 *    thinkingThought — dashed line:  residual-projection avg over thought tokens (V2 vectors)
 *  Activation values are display-ready (one selected layer); sentiments are
 *  null while a turn is still streaming or absent (no thought block emitted).
 */
export interface EmotionState {
  outputReply: EmotionValues | null;
  outputThought: EmotionValues | null;
  thinkingReply: EmotionValues;
  thinkingThought: EmotionValues | null;
}

/** Internal state — what the SSE stream populates. The display-ready
 *  EmotionState is derived from this by picking one layer. */
export interface RawState {
  outputReply: EmotionValues | null;
  outputThought: EmotionValues | null;
  thinkingReply: LayeredEmotionValues;
  thinkingThought: LayeredEmotionValues | null;
}

export const BASELINE_STATE: EmotionState = {
  outputReply: null,
  outputThought: null,
  thinkingReply: BASELINE,
  thinkingThought: null,
};

export const BASELINE_RAW_STATE: RawState = {
  outputReply: null,
  outputThought: null,
  thinkingReply: makeLayeredBaseline(),
  thinkingThought: null,
};

export interface Snapshot {
  atWord: number;
  /** Running halo (reply-token average) at this snapshot. */
  thinkingReply: LayeredEmotionValues;
  /** Running dashed line (thought-token average) at this snapshot. Null
   *  before any thought-phase token has been seen. */
  thinkingThought: LayeredEmotionValues | null;
}

/** One generated token: where it ends in its phase-buffer (char offset),
 *  which phase the token belongs to, and the layered projection at that
 *  step (scoped to its phase — V2 for thought, V3 for reply). */
export interface PerTokenData {
  charEnd: number;
  phase: "thought" | "reply";
  thinking: LayeredEmotionValues;
}

export interface Turn {
  id: string;
  userMessage: string;
  assistantReply: string;
  /** The model's <|channel>thought block. Empty string if thinking was
   *  off or the model didn't emit one. */
  assistantThought: string;
  snapshots: Snapshot[];
  /** Per-token data — used to recompute halo / dashed-line when the user
   *  highlights an excerpt of either the reply or the thought. */
  tokens: PerTokenData[];
  state: RawState;
}

const BACKEND_TO_FRONTEND: Record<string, Emotion> = {
  joy: "Joy",
  sadness: "Sadness",
  anger: "Anger",
  fear: "Fear",
  disgust: "Disgust",
  surprise: "Surprise",
};

/** Convert lowercase keys from the FastAPI backend ({joy, sadness, ...})
 *  to the CapitalCase keys the frontend uses everywhere else. Missing
 *  emotions default to 0. */
export function mapBackendEmotions(
  src: Record<string, number> | undefined | null,
): EmotionValues {
  const out = { ...BASELINE };
  if (!src) return out;
  for (const [k, v] of Object.entries(src)) {
    const e = BACKEND_TO_FRONTEND[k];
    if (e) out[e] = Math.max(0, Math.min(100, v));
  }
  return out;
}

/** Backend now ships thinking as {layer: {emotion: value}} per token. */
export function mapLayeredBackendEmotions(
  src: Record<string, Record<string, number>> | undefined | null,
): LayeredEmotionValues {
  const out = makeLayeredBaseline();
  if (!src) return out;
  for (const L of LAYERS) {
    const layerSrc = src[String(L)];
    if (layerSrc) out[L] = mapBackendEmotions(layerSrc);
  }
  return out;
}

/** Average a set of per-token records into one layered thinking vector.
 *  Used when the user highlights an excerpt — averages the projection
 *  over the tokens generated in that span. */
export function averageLayered(
  tokens: PerTokenData[],
): LayeredEmotionValues {
  const out = makeLayeredBaseline();
  if (tokens.length === 0) return out;
  for (const L of LAYERS) {
    for (const e of EMOTIONS) {
      let sum = 0;
      for (const t of tokens) sum += t.thinking[L][e] ?? 0;
      out[L][e] = sum / tokens.length;
    }
  }
  return out;
}

/** Average over tokens of a specific phase. Returns null if no tokens of
 *  that phase were present (e.g. no thought block was emitted). */
export function averageLayeredForPhase(
  tokens: PerTokenData[],
  phase: "thought" | "reply",
): LayeredEmotionValues | null {
  const filtered = tokens.filter((t) => t.phase === phase);
  if (filtered.length === 0) return null;
  return averageLayered(filtered);
}
