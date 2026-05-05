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

export interface EmotionState {
  /** Distilroberta classification of the model's full reply, computed
   *  once at end-of-turn. Null while a turn is still streaming. */
  output: EmotionValues | null;
  /** Live projection of the model's residual stream onto each emotion
   *  vector at the chosen layer (the "internal" reaction). */
  thinking: EmotionValues;
}

export const BASELINE_STATE: EmotionState = {
  output: null,
  thinking: BASELINE,
};

export interface Snapshot {
  atWord: number;
  thinking: EmotionValues;
}

export interface Turn {
  id: string;
  userMessage: string;
  assistantReply: string;
  /** Per-chunk snapshots — both vectors at each emit. */
  snapshots: Snapshot[];
  /** Convenience: the final state at the end of the turn. */
  state: EmotionState;
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
