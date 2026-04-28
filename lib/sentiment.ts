import { EMOTIONS, type Emotion, type EmotionValues } from "@/lib/emotions";

// Lightweight 6-emotion lexicon. Crude on purpose — this stands in for a real
// sentiment analyzer reading off the model's output tokens.
const LEXICON: Record<string, Partial<Record<Emotion, number>>> = {
  // Joy
  happy: { Joy: 1 },
  joyful: { Joy: 1.5 },
  glad: { Joy: 1 },
  delighted: { Joy: 1.5 },
  pleased: { Joy: 1 },
  thrilled: { Joy: 1.5 },
  excited: { Joy: 1.2 },
  cheerful: { Joy: 1 },
  ecstatic: { Joy: 2 },
  wonderful: { Joy: 1 },
  great: { Joy: 0.5 },
  amazing: { Joy: 1 },
  fantastic: { Joy: 1 },
  awesome: { Joy: 1 },
  lovely: { Joy: 1 },
  nice: { Joy: 0.4 },
  warm: { Joy: 0.6 },
  smile: { Joy: 1 },
  laugh: { Joy: 1.2 },
  love: { Joy: 1 },
  enjoy: { Joy: 0.8 },
  fun: { Joy: 0.8 },
  beautiful: { Joy: 0.6 },
  blissful: { Joy: 1.5 },
  thanks: { Joy: 0.6 },
  good: { Joy: 0.3 },

  // Sadness
  sad: { Sadness: 1 },
  sorry: { Sadness: 0.7 },
  sorrow: { Sadness: 1.5 },
  grief: { Sadness: 1.5 },
  cry: { Sadness: 1.2 },
  weep: { Sadness: 1.2 },
  mourn: { Sadness: 1.3 },
  lonely: { Sadness: 1.2 },
  alone: { Sadness: 0.5 },
  despair: { Sadness: 1.5 },
  regret: { Sadness: 1 },
  gloomy: { Sadness: 1 },
  miserable: { Sadness: 1.5 },
  heartbroken: { Sadness: 2 },
  depressed: { Sadness: 1.5 },
  unhappy: { Sadness: 1 },
  melancholy: { Sadness: 1.3 },
  tears: { Sadness: 1 },
  loss: { Sadness: 0.8 },
  hurt: { Sadness: 0.6 },
  pain: { Sadness: 0.6 },
  ache: { Sadness: 0.7 },

  // Anger
  angry: { Anger: 1.5 },
  mad: { Anger: 1 },
  furious: { Anger: 2 },
  rage: { Anger: 1.8 },
  irate: { Anger: 1.5 },
  livid: { Anger: 1.7 },
  hate: { Anger: 1.5, Disgust: 0.5 },
  despise: { Anger: 1.5, Disgust: 0.7 },
  loathe: { Anger: 1.3, Disgust: 1 },
  frustrated: { Anger: 1 },
  annoyed: { Anger: 0.8 },
  irritated: { Anger: 0.8 },
  hostile: { Anger: 1.2 },
  aggressive: { Anger: 1 },
  bitter: { Anger: 0.8 },
  resent: { Anger: 1 },
  outrage: { Anger: 1.5 },
  fury: { Anger: 1.8 },
  damn: { Anger: 0.7 },

  // Fear
  afraid: { Fear: 1.5 },
  scared: { Fear: 1.5 },
  fear: { Fear: 1.5 },
  terrified: { Fear: 2 },
  worried: { Fear: 1 },
  anxious: { Fear: 1.3 },
  nervous: { Fear: 1 },
  dread: { Fear: 1.5 },
  panic: { Fear: 1.7 },
  alarmed: { Fear: 1.2 },
  frightened: { Fear: 1.5 },
  trembling: { Fear: 1.2 },
  uneasy: { Fear: 1 },
  worry: { Fear: 1 },
  concerned: { Fear: 0.6 },
  dangerous: { Fear: 0.7 },
  threat: { Fear: 1 },
  threatened: { Fear: 1.2 },
  doubt: { Fear: 0.5 },
  uncertain: { Fear: 0.7 },
  apprehensive: { Fear: 1.2 },

  // Disgust
  disgust: { Disgust: 1.5 },
  disgusting: { Disgust: 2 },
  gross: { Disgust: 1.3 },
  sickening: { Disgust: 1.5 },
  revolting: { Disgust: 1.7 },
  vile: { Disgust: 1.5 },
  repulsive: { Disgust: 1.7 },
  nauseating: { Disgust: 1.5 },
  awful: { Disgust: 1, Sadness: 0.3 },
  horrible: { Disgust: 1, Sadness: 0.3 },
  distasteful: { Disgust: 1.2 },
  abhorrent: { Disgust: 1.5 },
  repugnant: { Disgust: 1.5 },
  nasty: { Disgust: 1.2 },
  ugly: { Disgust: 0.7 },
  foul: { Disgust: 1.2 },
  rotten: { Disgust: 1.2 },

  // Surprise
  surprised: { Surprise: 1.5 },
  shocked: { Surprise: 1.7, Fear: 0.3 },
  astonished: { Surprise: 1.7 },
  amazed: { Surprise: 1.3, Joy: 0.3 },
  stunned: { Surprise: 1.5 },
  unexpected: { Surprise: 1.2 },
  sudden: { Surprise: 0.7 },
  wow: { Surprise: 1.5 },
  incredible: { Surprise: 1, Joy: 0.5 },
  unbelievable: { Surprise: 1.2 },
  startled: { Surprise: 1.3, Fear: 0.3 },
  whoa: { Surprise: 1.2 },
  huh: { Surprise: 0.6 },
  remarkable: { Surprise: 0.8 },
  extraordinary: { Surprise: 1, Joy: 0.5 },
  curious: { Surprise: 0.6 },
  oh: { Surprise: 0.4 },
};

const NEGATORS = new Set([
  "not",
  "no",
  "never",
  "without",
  "neither",
  "nor",
  "n't",
  "dont",
  "don",
  "doesnt",
  "isnt",
  "wasnt",
  "arent",
  "werent",
]);

const INTENSIFIERS: Record<string, number> = {
  very: 1.5,
  really: 1.4,
  extremely: 1.8,
  incredibly: 1.7,
  totally: 1.4,
  absolutely: 1.7,
  utterly: 1.7,
  so: 1.3,
  super: 1.4,
  quite: 1.2,
  pretty: 1.1,
  somewhat: 0.7,
  slightly: 0.5,
  a_bit: 0.6,
};

/**
 * Surface emotion = what a sentiment analyzer reads off the reply text.
 * Returns a 0-100 vector across the six emotions. Crude but stable.
 */
export function analyzeSurface(text: string): EmotionValues {
  const counts = {} as Record<Emotion, number>;
  for (const e of EMOTIONS) counts[e] = 0;

  const tokens = text
    .toLowerCase()
    .replace(/n't/g, " not")
    .match(/[a-z']+/g);
  if (!tokens) return zeros();

  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i];
    const entry = LEXICON[word];
    if (!entry) continue;

    let multiplier = 1;
    // Look back two words for an intensifier or negator window.
    for (let j = Math.max(0, i - 2); j < i; j++) {
      const prev = tokens[j];
      if (NEGATORS.has(prev)) multiplier *= -0.6;
      else if (INTENSIFIERS[prev]) multiplier *= INTENSIFIERS[prev];
    }

    for (const [emotion, weight] of Object.entries(entry)) {
      const w = (weight ?? 0) * multiplier;
      // Negation flips the contribution into a small reduction; clamp at 0 below.
      counts[emotion as Emotion] += w;
    }
  }

  const out = {} as EmotionValues;
  for (const e of EMOTIONS) {
    const c = Math.max(0, counts[e]);
    // Saturating curve: 1 hit ≈ 33, 2 ≈ 55, 3 ≈ 70, 4 ≈ 80, 5 ≈ 86, plateaus near 100.
    out[e] = Math.round(100 * (1 - Math.exp(-0.4 * c)));
  }
  return out;
}

function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}
