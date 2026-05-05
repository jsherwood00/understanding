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

/** Map of layer → 6-emotion vector. Backend now ships all 6 layers per
 *  token; the frontend stores the full layered shape and picks one to
 *  display so layer-switching is instant. */
export type LayeredEmotionValues = Record<Layer, EmotionValues>;

export function makeLayeredBaseline(): LayeredEmotionValues {
  const out = {} as LayeredEmotionValues;
  for (const L of LAYERS) out[L] = { ...BASELINE };
  return out;
}

export interface EmotionState {
  /** External classifier reading of the full reply, computed once at
   *  end-of-turn. Null while a turn is still streaming. */
  output: EmotionValues | null;
  /** Display-ready thinking values for the *currently selected* layer. */
  thinking: EmotionValues;
}

/** Internal-only state — what the SSE stream actually populates. The
 *  display-ready EmotionState is derived by picking one layer. */
export interface RawState {
  output: EmotionValues | null;
  thinking: LayeredEmotionValues;
}

export const BASELINE_STATE: EmotionState = {
  output: null,
  thinking: BASELINE,
};

export const BASELINE_RAW_STATE: RawState = {
  output: null,
  thinking: makeLayeredBaseline(),
};

export interface Snapshot {
  atWord: number;
  thinking: LayeredEmotionValues;
}

/** One generated token: where it ends in the reply (char offset) and the
 *  full layered thinking at that step. Used for highlight-to-analyze
 *  nearest-token mapping. */
export interface PerTokenData {
  charEnd: number;
  thinking: LayeredEmotionValues;
}

export interface Turn {
  id: string;
  userMessage: string;
  assistantReply: string;
  /** Per-chunk snapshots — for the replay/scrub control. */
  snapshots: Snapshot[];
  /** Per-token layered thinking — used to compute the halo when the user
   *  highlights an excerpt of this turn's reply. */
  tokens: PerTokenData[];
  /** The final RawState at end-of-turn (output dot is null if classifier
   *  hasn't returned yet, otherwise classified value). */
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
