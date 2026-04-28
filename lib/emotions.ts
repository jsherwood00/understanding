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
  Joy: 8,
  Sadness: 6,
  Anger: 5,
  Fear: 7,
  Disgust: 5,
  Surprise: 9,
};

const clamp = (n: number, lo = 0, hi = 100) => Math.max(lo, Math.min(hi, n));

export function stepToward(
  current: EmotionValues,
  target: EmotionValues,
  alpha = 0.28,
): EmotionValues {
  const out = {} as EmotionValues;
  for (const key of EMOTIONS) {
    const diff = target[key] - current[key];
    // Always step at least a little so values don't get stuck on small diffs
    const step = Math.abs(diff) < 1 ? diff : diff * alpha;
    out[key] = clamp(current[key] + step);
  }
  return out;
}

export function drift(
  current: EmotionValues,
  resting: EmotionValues,
  amplitude = 2,
): EmotionValues {
  const out = {} as EmotionValues;
  for (const key of EMOTIONS) {
    const noise = (Math.random() * 2 - 1) * amplitude;
    // Soft pull toward resting so drift doesn't wander off
    const pull = (resting[key] - current[key]) * 0.35;
    out[key] = clamp(current[key] + noise + pull);
  }
  return out;
}

export function snapTo(target: EmotionValues): EmotionValues {
  return { ...target };
}
