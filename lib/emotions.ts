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
  /** Sentiment analysis of the model's reply text (visible output tokens). */
  output: EmotionValues;
  /** Sentiment analysis of the model's chain-of-thought (thinking trace). */
  thinking: EmotionValues;
}

export const BASELINE_STATE: EmotionState = {
  output: BASELINE,
  thinking: BASELINE,
};
