import { EMOTIONS, type Emotion, type EmotionValues } from "@/lib/emotions";

// 6-emotion lexicon. Words map to partial emotion-weight maps. Negators flip
// contribution; intensifiers scale it. Crude but fast — runs in microseconds
// per chunk so we can analyze every 10 words during streaming.
const LEXICON: Record<string, Partial<Record<Emotion, number>>> = {
  // ── Joy ───────────────────────────────────────────────────────────────
  happy: { Joy: 1 },
  joyful: { Joy: 1.5 },
  joy: { Joy: 1.3 },
  glad: { Joy: 1 },
  delighted: { Joy: 1.5 },
  delight: { Joy: 1.4 },
  pleased: { Joy: 1 },
  thrilled: { Joy: 1.5 },
  excited: { Joy: 1.2 },
  cheerful: { Joy: 1 },
  ecstatic: { Joy: 2 },
  wonderful: { Joy: 1 },
  great: { Joy: 0.6 },
  amazing: { Joy: 1, Surprise: 0.4 },
  fantastic: { Joy: 1.2 },
  awesome: { Joy: 1.2 },
  lovely: { Joy: 1 },
  nice: { Joy: 0.5 },
  warm: { Joy: 0.6 },
  smile: { Joy: 1 },
  smiling: { Joy: 1 },
  laugh: { Joy: 1.2 },
  laughing: { Joy: 1.2 },
  laughter: { Joy: 1.3 },
  love: { Joy: 1 },
  loving: { Joy: 1 },
  enjoy: { Joy: 0.8 },
  enjoying: { Joy: 0.8 },
  fun: { Joy: 0.8 },
  beautiful: { Joy: 0.6 },
  blissful: { Joy: 1.5 },
  thanks: { Joy: 0.6 },
  thank: { Joy: 0.6 },
  good: { Joy: 0.4 },
  congratulations: { Joy: 1.5 },
  congrats: { Joy: 1.4 },
  excellent: { Joy: 1 },
  brilliant: { Joy: 1, Surprise: 0.3 },
  perfect: { Joy: 0.9 },
  yay: { Joy: 1.5 },
  hooray: { Joy: 1.5 },
  proud: { Joy: 1 },
  satisfied: { Joy: 0.9 },
  content: { Joy: 0.8 },
  grateful: { Joy: 1 },
  appreciate: { Joy: 0.7 },
  hello: { Joy: 0.4 },
  hi: { Joy: 0.3 },
  hey: { Joy: 0.3 },
  please: { Joy: 0.3 },
  welcome: { Joy: 0.5 },
  yes: { Joy: 0.3 },
  sure: { Joy: 0.3 },
  cool: { Joy: 0.5 },
  okay: { Joy: 0.2 },
  ok: { Joy: 0.2 },
  helpful: { Joy: 0.5 },
  kind: { Joy: 0.5 },
  bright: { Joy: 0.5 },
  hopeful: { Joy: 0.6 },
  hope: { Joy: 0.4 },

  // ── Sadness ───────────────────────────────────────────────────────────
  sad: { Sadness: 1 },
  saddened: { Sadness: 1.2 },
  sorry: { Sadness: 0.7 },
  sorrow: { Sadness: 1.5 },
  grief: { Sadness: 1.5 },
  grieving: { Sadness: 1.5 },
  cry: { Sadness: 1.2 },
  crying: { Sadness: 1.3 },
  weep: { Sadness: 1.2 },
  weeping: { Sadness: 1.3 },
  mourn: { Sadness: 1.3 },
  mourning: { Sadness: 1.3 },
  lonely: { Sadness: 1.2 },
  loneliness: { Sadness: 1.3 },
  alone: { Sadness: 0.5 },
  despair: { Sadness: 1.5 },
  hopeless: { Sadness: 1.4 },
  regret: { Sadness: 1 },
  regretful: { Sadness: 1.1 },
  gloomy: { Sadness: 1 },
  miserable: { Sadness: 1.5 },
  misery: { Sadness: 1.4 },
  heartbroken: { Sadness: 2 },
  heartbreak: { Sadness: 1.7 },
  depressed: { Sadness: 1.5 },
  depressing: { Sadness: 1.4 },
  unhappy: { Sadness: 1 },
  melancholy: { Sadness: 1.3 },
  tears: { Sadness: 1 },
  tear: { Sadness: 0.6 },
  loss: { Sadness: 0.8 },
  lost: { Sadness: 0.4 },
  hurt: { Sadness: 0.6 },
  hurts: { Sadness: 0.7 },
  pain: { Sadness: 0.6 },
  painful: { Sadness: 0.8 },
  ache: { Sadness: 0.7 },
  aching: { Sadness: 0.8 },
  suffering: { Sadness: 1, Fear: 0.3 },
  difficult: { Sadness: 0.4 },
  tough: { Sadness: 0.4 },
  empty: { Sadness: 0.7 },
  hollow: { Sadness: 0.8 },
  defeated: { Sadness: 0.9 },

  // ── Anger ─────────────────────────────────────────────────────────────
  angry: { Anger: 1.5 },
  anger: { Anger: 1.3 },
  mad: { Anger: 1 },
  furious: { Anger: 2 },
  rage: { Anger: 1.8 },
  raging: { Anger: 1.8 },
  irate: { Anger: 1.5 },
  livid: { Anger: 1.7 },
  hate: { Anger: 1.5, Disgust: 0.5 },
  hated: { Anger: 1.5, Disgust: 0.5 },
  hatred: { Anger: 1.5, Disgust: 0.5 },
  despise: { Anger: 1.5, Disgust: 0.7 },
  loathe: { Anger: 1.3, Disgust: 1 },
  frustrated: { Anger: 1 },
  frustration: { Anger: 1 },
  annoyed: { Anger: 0.8 },
  annoying: { Anger: 0.7 },
  irritated: { Anger: 0.8 },
  irritating: { Anger: 0.7 },
  hostile: { Anger: 1.2 },
  aggressive: { Anger: 1 },
  bitter: { Anger: 0.8 },
  resent: { Anger: 1 },
  resentful: { Anger: 1.1 },
  outrage: { Anger: 1.5 },
  outraged: { Anger: 1.5 },
  fury: { Anger: 1.8 },
  damn: { Anger: 0.7 },
  damned: { Anger: 0.7 },
  ridiculous: { Anger: 0.6, Disgust: 0.3 },
  pissed: { Anger: 1.3 },
  fuming: { Anger: 1.5 },
  enraged: { Anger: 1.8 },
  agitated: { Anger: 0.8, Fear: 0.3 },
  contempt: { Anger: 1, Disgust: 0.7 },
  hostility: { Anger: 1.3 },
  insulting: { Anger: 1, Disgust: 0.5 },
  insult: { Anger: 0.9 },
  rude: { Anger: 0.9, Disgust: 0.4 },

  // ── Fear ──────────────────────────────────────────────────────────────
  afraid: { Fear: 1.5 },
  scared: { Fear: 1.5 },
  fear: { Fear: 1.5 },
  fearful: { Fear: 1.5 },
  terrified: { Fear: 2 },
  terror: { Fear: 1.8 },
  terrifying: { Fear: 1.8 },
  worried: { Fear: 1 },
  worry: { Fear: 1 },
  worrying: { Fear: 1 },
  anxious: { Fear: 1.3 },
  anxiety: { Fear: 1.4 },
  nervous: { Fear: 1 },
  nervousness: { Fear: 1.1 },
  dread: { Fear: 1.5 },
  dreadful: { Fear: 1.4 },
  panic: { Fear: 1.7 },
  panicking: { Fear: 1.7 },
  alarmed: { Fear: 1.2 },
  alarming: { Fear: 1.2 },
  frightened: { Fear: 1.5 },
  frightening: { Fear: 1.4 },
  trembling: { Fear: 1.2 },
  uneasy: { Fear: 1 },
  unease: { Fear: 1 },
  concerned: { Fear: 0.6 },
  dangerous: { Fear: 0.7 },
  danger: { Fear: 1 },
  threat: { Fear: 1 },
  threatening: { Fear: 1.1 },
  threatened: { Fear: 1.2 },
  doubt: { Fear: 0.5 },
  uncertain: { Fear: 0.7 },
  uncertainty: { Fear: 0.8 },
  apprehensive: { Fear: 1.2 },
  apprehension: { Fear: 1.2 },
  hesitant: { Fear: 0.7 },
  hesitation: { Fear: 0.7 },
  vulnerable: { Fear: 1, Sadness: 0.4 },
  cautious: { Fear: 0.5 },

  // ── Disgust ───────────────────────────────────────────────────────────
  disgust: { Disgust: 1.5 },
  disgusting: { Disgust: 2 },
  disgusted: { Disgust: 1.7 },
  gross: { Disgust: 1.3 },
  sickening: { Disgust: 1.5 },
  revolting: { Disgust: 1.7 },
  vile: { Disgust: 1.5 },
  repulsive: { Disgust: 1.7 },
  repulsion: { Disgust: 1.7 },
  nauseating: { Disgust: 1.5 },
  nauseous: { Disgust: 1.3 },
  awful: { Disgust: 0.9, Sadness: 0.3 },
  horrible: { Disgust: 1, Sadness: 0.3 },
  horrendous: { Disgust: 1.2 },
  distasteful: { Disgust: 1.2 },
  distaste: { Disgust: 1.2 },
  abhorrent: { Disgust: 1.5 },
  abhor: { Disgust: 1.4 },
  repugnant: { Disgust: 1.5 },
  nasty: { Disgust: 1.2 },
  ugly: { Disgust: 0.6 },
  foul: { Disgust: 1.2 },
  rotten: { Disgust: 1.2 },
  filthy: { Disgust: 1.3 },
  putrid: { Disgust: 1.6 },
  yuck: { Disgust: 1.3 },
  ew: { Disgust: 1.2 },
  cringe: { Disgust: 0.8 },
  cringing: { Disgust: 0.8 },
  appalled: { Disgust: 1, Anger: 0.5 },
  appalling: { Disgust: 1, Anger: 0.4 },
  obscene: { Disgust: 1.2 },
  vulgar: { Disgust: 1 },

  // ── Surprise ──────────────────────────────────────────────────────────
  surprised: { Surprise: 1.5 },
  surprise: { Surprise: 1.3 },
  surprising: { Surprise: 1.2 },
  shocked: { Surprise: 1.7, Fear: 0.3 },
  shocking: { Surprise: 1.5, Fear: 0.2 },
  astonished: { Surprise: 1.7 },
  astonishing: { Surprise: 1.6 },
  amazed: { Surprise: 1.3, Joy: 0.3 },
  stunned: { Surprise: 1.5 },
  unexpected: { Surprise: 1.2 },
  unexpectedly: { Surprise: 1.2 },
  sudden: { Surprise: 0.7 },
  suddenly: { Surprise: 0.8 },
  wow: { Surprise: 1.5 },
  incredible: { Surprise: 1, Joy: 0.4 },
  unbelievable: { Surprise: 1.2 },
  startled: { Surprise: 1.3, Fear: 0.3 },
  startling: { Surprise: 1.3 },
  whoa: { Surprise: 1.2 },
  huh: { Surprise: 0.6 },
  remarkable: { Surprise: 0.8 },
  extraordinary: { Surprise: 1, Joy: 0.4 },
  curious: { Surprise: 0.6 },
  oh: { Surprise: 0.4 },
  baffling: { Surprise: 1.1, Fear: 0.2 },
  bewildered: { Surprise: 1.2, Fear: 0.3 },
  flabbergasted: { Surprise: 1.7 },
};

const NEGATORS = new Set([
  "not",
  "no",
  "never",
  "without",
  "neither",
  "nor",
  "dont",
  "doesnt",
  "isnt",
  "wasnt",
  "arent",
  "werent",
  "cant",
  "couldnt",
  "wouldnt",
  "shouldnt",
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
  remarkably: 1.4,
  deeply: 1.4,
};

const clampPct = (n: number) => Math.max(0, Math.min(100, Math.round(n)));

function zeros(): EmotionValues {
  const z = {} as EmotionValues;
  for (const e of EMOTIONS) z[e] = 0;
  return z;
}

/**
 * Score arbitrary text on the six emotions. Synchronous and fast — designed
 * to run on every 10-word chunk during streaming.
 */
export function analyzeEmotions(text: string): EmotionValues {
  const counts = {} as Record<Emotion, number>;
  for (const e of EMOTIONS) counts[e] = 0;

  const tokens = text
    .toLowerCase()
    .replace(/n't/g, " not")
    .replace(/[*_`#>~]/g, " ")
    .match(/[a-z']+/g);
  if (!tokens || tokens.length === 0) return zeros();

  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i];
    const entry = LEXICON[word];
    if (!entry) continue;

    let multiplier = 1;
    for (let j = Math.max(0, i - 2); j < i; j++) {
      const prev = tokens[j];
      if (NEGATORS.has(prev)) multiplier *= -0.6;
      else if (INTENSIFIERS[prev]) multiplier *= INTENSIFIERS[prev];
    }

    for (const [emotion, weight] of Object.entries(entry)) {
      counts[emotion as Emotion] += (weight ?? 0) * multiplier;
    }
  }

  const out = {} as EmotionValues;
  for (const e of EMOTIONS) {
    const c = Math.max(0, counts[e]);
    // Saturating curve: 1 hit ≈ 33, 2 ≈ 55, 3 ≈ 70, 4 ≈ 80, plateaus near 100.
    out[e] = clampPct(100 * (1 - Math.exp(-0.4 * c)));
  }
  return out;
}
