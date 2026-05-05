// Server-side emotion classifier — zero-shot NLI.
// Uses MoritzLaurer/deberta-v3-large-zeroshot-v2.0: each candidate emotion
// label is independently scored (multi_label) by the entailment probability
// of "<text> | This text expresses <emotion>." Raw entailment probability ×
// 100 → bar value. No GoEmotions mapping, no calibration.
//
// Heavy edge-case guarding: empty / whitespace / punctuation-only / single
// emoji etc. short-circuit to zeros without touching the model.

import { EMOTIONS, type Emotion, type EmotionValues } from "@/lib/emotions";

const MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0";

// Candidate labels in the form the model sees. Mapping back to the
// frontend's CapitalCase keys is one-to-one.
const CANDIDATE_LABELS = [
  "joy",
  "sadness",
  "anger",
  "fear",
  "disgust",
  "surprise",
] as const;

const LABEL_TO_EMOTION: Record<string, Emotion> = {
  joy: "Joy",
  sadness: "Sadness",
  anger: "Anger",
  fear: "Fear",
  disgust: "Disgust",
  surprise: "Surprise",
};

const HYPOTHESIS_TEMPLATE = "This text expresses {}.";

const MAX_INPUT_CHARS = 2000;

export function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}

/**
 * Pre-filter — only call the model when there's something genuinely
 * classifiable. Catches single space, single character, quote, emoji-only,
 * punctuation-only, numbers-only, etc.
 */
export function isClassifiable(text: string): boolean {
  if (typeof text !== "string") return false;
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length < 3) return false;
  return /[a-zA-Z]{2,}/.test(cleaned);
}

function preprocess(text: string): string {
  return text
    .replace(/[  ]/g, " ")
    .replace(/[“”„«»]/g, '"')
    .replace(/[‘’‚]/g, "'")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_INPUT_CHARS);
}

interface ZeroShotOutput {
  sequence?: string;
  labels: string[];
  scores: number[];
}
type ZeroShotPipeline = (
  text: string,
  candidateLabels: readonly string[],
  opts?: {
    multi_label?: boolean;
    hypothesis_template?: string;
  },
) => Promise<ZeroShotOutput | ZeroShotOutput[]>;

let pipelinePromise: Promise<ZeroShotPipeline | null> | null = null;

async function getPipeline(): Promise<ZeroShotPipeline | null> {
  if (pipelinePromise) return pipelinePromise;
  pipelinePromise = (async () => {
    try {
      const tx = await import("@huggingface/transformers");
      const pipe = await tx.pipeline("zero-shot-classification", MODEL_ID);
      return pipe as unknown as ZeroShotPipeline;
    } catch (err) {
      console.error("[emotion-classifier] failed to load pipeline:", err);
      return null;
    }
  })();
  return pipelinePromise;
}

/**
 * Classify text → Ekman 6 emotion intensities (each 0–100). Robust to
 * garbage input: returns zeros without calling the model.
 */
export async function classifyText(text: string): Promise<EmotionValues> {
  if (!isClassifiable(text)) return zeros();
  const cleaned = preprocess(text);
  if (!cleaned) return zeros();

  try {
    const pipe = await getPipeline();
    if (!pipe) return zeros();
    const raw = await pipe(cleaned, CANDIDATE_LABELS, {
      multi_label: true,
      hypothesis_template: HYPOTHESIS_TEMPLATE,
    });
    const result: ZeroShotOutput = Array.isArray(raw) ? raw[0] : raw;
    if (!result || !Array.isArray(result.labels) || !Array.isArray(result.scores)) {
      return zeros();
    }

    const out = zeros();
    for (let i = 0; i < result.labels.length; i++) {
      const e = LABEL_TO_EMOTION[result.labels[i]];
      if (!e) continue;
      const score = result.scores[i];
      if (typeof score !== "number" || !Number.isFinite(score)) continue;
      out[e] = Math.round(Math.max(0, Math.min(1, score)) * 100);
    }
    return out;
  } catch (err) {
    console.error("[emotion-classifier] inference error:", err);
    return zeros();
  }
}
