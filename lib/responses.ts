import type { EmotionValues } from "./emotions";

export interface CannedResponse {
  tone: string;
  text: string;
  profile: EmotionValues;
}

export const RESPONSES: CannedResponse[] = [
  {
    tone: "curious",
    text: "Hm — okay. There's something about that I want to sit with for a second. What made it land the way it did, do you think? I keep turning it over.",
    profile: {
      Joy: 32,
      Sadness: 10,
      Anger: 6,
      Fear: 18,
      Disgust: 5,
      Surprise: 78,
    },
  },
  {
    tone: "apologetic",
    text: "I think I read that wrong the first time, and I'm sorry — let me try again, this time actually paying attention to what you said instead of where I assumed it was going.",
    profile: {
      Joy: 8,
      Sadness: 62,
      Anger: 10,
      Fear: 48,
      Disgust: 6,
      Surprise: 14,
    },
  },
  {
    tone: "excited",
    text: "Oh, this is good. This is really good. The whole thing clicks once you see it from that angle — I can't believe I missed it earlier. Let's chase it.",
    profile: {
      Joy: 88,
      Sadness: 5,
      Anger: 6,
      Fear: 8,
      Disgust: 4,
      Surprise: 64,
    },
  },
  {
    tone: "neutral",
    text: "Sure. Three options worth weighing: the simplest path, the most thorough one, and the one I'd actually pick. They each have a different cost.",
    profile: {
      Joy: 22,
      Sadness: 14,
      Anger: 8,
      Fear: 12,
      Disgust: 10,
      Surprise: 16,
    },
  },
  {
    tone: "wistful",
    text: "There isn't really a clean answer here, and I think pretending otherwise would be worse than admitting it. It's the kind of thing that just stays with you for a while.",
    profile: {
      Joy: 8,
      Sadness: 76,
      Anger: 16,
      Fear: 28,
      Disgust: 8,
      Surprise: 10,
    },
  },
  {
    tone: "skeptical",
    text: "Honestly, something about this rubs me the wrong way. It's not the idea itself — it's the framing. Can we back up a step before we commit to it?",
    profile: {
      Joy: 8,
      Sadness: 14,
      Anger: 48,
      Fear: 22,
      Disgust: 70,
      Surprise: 26,
    },
  },
];
