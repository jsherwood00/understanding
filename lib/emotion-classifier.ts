// Server-side emotion classifier.
// Uses SamLowe/roberta-base-go_emotions-onnx (RoBERTa fine-tuned on GoEmotions,
// 28 labels) and maps to Ekman 6 via Google's official GoEmotions taxonomy:
//   https://github.com/google-research/google-research/tree/master/goemotions
//
// Heavy edge-case guarding: any input that's empty, whitespace-only,
// punctuation-only, just an emoji, a single character, etc. short-circuits
// to zeros without touching the model. Model errors fall through to zeros.

import { EMOTIONS, type Emotion, type EmotionValues } from "@/lib/emotions";

const MODEL_ID = "SamLowe/roberta-base-go_emotions-onnx";

// Official GoEmotions → Ekman 6 mapping (Demszky et al., 2020).
// Each GoEmotions label maps to exactly one Ekman bucket (or "neutral", dropped).
const GOEMOTIONS_TO_EKMAN: Record<string, Emotion | null> = {
  // Anger
  anger: "Anger",
  annoyance: "Anger",
  disapproval: "Anger",
  // Disgust
  disgust: "Disgust",
  // Fear
  fear: "Fear",
  nervousness: "Fear",
  // Joy
  joy: "Joy",
  amusement: "Joy",
  approval: "Joy",
  excitement: "Joy",
  gratitude: "Joy",
  love: "Joy",
  optimism: "Joy",
  relief: "Joy",
  pride: "Joy",
  admiration: "Joy",
  desire: "Joy",
  caring: "Joy",
  // Sadness
  sadness: "Sadness",
  disappointment: "Sadness",
  embarrassment: "Sadness",
  grief: "Sadness",
  remorse: "Sadness",
  // Surprise
  surprise: "Surprise",
  realization: "Surprise",
  confusion: "Surprise",
  curiosity: "Surprise",
  // Neutral — dropped
  neutral: null,
};

const MAX_INPUT_CHARS = 2000;

export function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}

/**
 * Pre-filter — only call the model when there's something genuinely classifiable.
 * Catches single space, single character, quote, emoji-only, punctuation-only,
 * numbers-only, etc.
 */
export function isClassifiable(text: string): boolean {
  if (typeof text !== "string") return false;
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length < 3) return false;
  // Must contain at least one run of ≥2 letters (a real word).
  return /[a-zA-Z]{2,}/.test(cleaned);
}

function preprocess(text: string): string {
  return text
    .replace(/[  ]/g, " ")
    .replace(/[“”„«»]/g, '"')
    .replace(/[‘’‚]/g, "'")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_INPUT_CHARS);
}

type ClassificationItem = { label: string; score: number };
type ClassificationOutput = ClassificationItem[];
type Classifier = (
  text: string,
  opts?: unknown,
) => Promise<ClassificationOutput | ClassificationOutput[]>;

let pipelinePromise: Promise<Classifier | null> | null = null;

async function getPipeline(): Promise<Classifier | null> {
  if (pipelinePromise) return pipelinePromise;
  pipelinePromise = (async () => {
    try {
      const tx = await import("@huggingface/transformers");
      const pipe = await tx.pipeline("text-classification", MODEL_ID);
      return pipe as unknown as Classifier;
    } catch (err) {
      console.error("[emotion-classifier] failed to load pipeline:", err);
      return null;
    }
  })();
  return pipelinePromise;
}

/**
 * Classify text → Ekman 6 emotion intensities (each 0–100).
 * Robust to garbage input: returns zeros without calling the model.
 */
export async function classifyText(text: string): Promise<EmotionValues> {
  if (!isClassifiable(text)) return zeros();
  const cleaned = preprocess(text);
  if (!cleaned) return zeros();

  try {
    const pipe = await getPipeline();
    if (!pipe) return zeros();
    const raw = await pipe(cleaned, { top_k: null });
    const results: ClassificationOutput = Array.isArray(raw)
      ? Array.isArray(raw[0])
        ? (raw[0] as ClassificationOutput)
        : (raw as ClassificationOutput)
      : [];

    // Sum probabilities into Ekman buckets (multi-label model means raw
    // scores aren't a strict softmax across labels; sum is appropriate
    // because we're collapsing related fine-grained labels).
    const acc = zeros();
    for (const item of results) {
      if (
        !item ||
        typeof item.label !== "string" ||
        typeof item.score !== "number" ||
        !Number.isFinite(item.score)
      ) {
        continue;
      }
      const ekman = GOEMOTIONS_TO_EKMAN[item.label.toLowerCase()];
      if (!ekman) continue;
      acc[ekman] = Math.min(1, acc[ekman] + item.score);
    }

    const out = zeros();
    for (const e of EMOTIONS) {
      out[e] = Math.max(0, Math.min(100, Math.round(acc[e] * 100)));
    }
    return out;
  } catch (err) {
    console.error("[emotion-classifier] inference error:", err);
    return zeros();
  }
}
