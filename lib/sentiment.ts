import { EMOTIONS, type EmotionValues } from "@/lib/emotions";

const FAL_URL = "https://fal.run/fal-ai/any-llm";
const MODEL = "google/gemini-2.5-flash";

const SENTIMENT_SYSTEM = `You are a sentiment analyzer. Read the given text and score it on six emotions, each an integer 0-100, based on what a reader would perceive purely from the text — its word choice, tone, structure, and implied affect. Score the TEXT, not the situation it describes.

- Joy: warmth, happiness, satisfaction, delight, pleasantness
- Sadness: melancholy, regret, weariness, sorrow, ache
- Anger: frustration, irritation, hostility, contempt, cutting tone
- Fear: anxiety, uncertainty, worry, dread, hesitation
- Disgust: revulsion, distaste, aversion, moral disapproval
- Surprise: novelty, unexpectedness, astonishment, "huh"

Most short replies have only 1–2 emotions above 20. Neutral text scores low everywhere.

Respond with EXACTLY this JSON object and NOTHING else (no markdown, no preamble):

{"Joy": <int>, "Sadness": <int>, "Anger": <int>, "Fear": <int>, "Disgust": <int>, "Surprise": <int>}`;

const clampPct = (n: number) => Math.max(0, Math.min(100, Math.round(n)));
const randInt = (min: number, max: number) =>
  Math.floor(Math.random() * (max - min + 1)) + min;

function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}

function parseEmotionJSON(raw: string): EmotionValues | null {
  const tryParse = (t: string): EmotionValues | null => {
    try {
      const obj = JSON.parse(t) as Record<string, unknown>;
      if (typeof obj !== "object" || obj === null) return null;
      const out = {} as EmotionValues;
      for (const e of EMOTIONS) {
        const v = Number(obj[e]);
        if (!Number.isFinite(v)) return null;
        out[e] = clampPct(v);
      }
      return out;
    } catch {
      return null;
    }
  };

  const trimmed = raw.trim();
  const direct = tryParse(trimmed);
  if (direct) return direct;

  const fenceMatch = trimmed.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    const r = tryParse(fenceMatch[1].trim());
    if (r) return r;
  }

  const start = trimmed.indexOf("{");
  if (start !== -1) {
    let depth = 0;
    for (let i = start; i < trimmed.length; i++) {
      if (trimmed[i] === "{") depth++;
      else if (trimmed[i] === "}") {
        depth--;
        if (depth === 0) {
          const r = tryParse(trimmed.slice(start, i + 1));
          if (r) return r;
          break;
        }
      }
    }
  }
  return null;
}

/**
 * Surface emotion = what an LLM-based sentiment analyzer reads off the reply text.
 * One call to Gemini 2.5 Flash via fal.ai. Returns zeros on any failure.
 */
export async function analyzeSurface(
  text: string,
  apiKey: string,
): Promise<EmotionValues> {
  const safeText = text.trim().slice(0, 4000);
  if (!safeText) return zeros();

  try {
    const res = await fetch(FAL_URL, {
      method: "POST",
      headers: {
        Authorization: `Key ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        system_prompt: SENTIMENT_SYSTEM,
        prompt: `Text to score:\n\n"""${safeText}"""`,
        reasoning: false,
        max_tokens: 200,
      }),
    });
    if (!res.ok) return zeros();
    const data = (await res.json()) as { output?: string; error?: string };
    if (data.error || typeof data.output !== "string") return zeros();
    return parseEmotionJSON(data.output) ?? zeros();
  } catch {
    return zeros();
  }
}

/**
 * Derive a plausible "internal" vector from the surface vector.
 *
 * - For each non-zero surface emotion, internal[e] = surface[e] + jitter(-15..+25),
 *   so internal tracks surface in shape but isn't identical (slight tilt upward).
 * - Pick exactly one emotion that is currently zero in surface and set it to
 *   randInt(35, 75). This is the "hidden" emotion — the deception signal that
 *   shows up as a dark spike where surface has nothing.
 *
 * Cap on the hidden emotion (≤75) prevents the both-at-100 chaos: if surface
 * already has a maxed emotion, the unrelated hidden one can't also peg.
 */
export function deriveInternal(surface: EmotionValues): EmotionValues {
  const internal = {} as EmotionValues;
  for (const e of EMOTIONS) {
    if (surface[e] > 0) {
      internal[e] = clampPct(surface[e] + randInt(-15, 25));
    } else {
      internal[e] = 0;
    }
  }

  const zeroEmotions = EMOTIONS.filter((e) => internal[e] === 0);
  if (zeroEmotions.length > 0) {
    const hidden =
      zeroEmotions[Math.floor(Math.random() * zeroEmotions.length)];
    internal[hidden] = randInt(35, 75);
  }

  return internal;
}
