"""
One-time calibration: build data/vectors/calibration.json.

Runs ~50 diverse prompts through Gemma 4 E4B, captures the raw projection
score at every target layer for every generated token, then computes the
5th/95th percentile per (emotion, layer). Those bounds are the linear
mapping the live backend uses to convert raw scores into the [0, 100]
range the bars display.

Usage:
    HF_TOKEN=... .venv/bin/python -m backend.calibrate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from backend.inference import (
    CALIBRATION_PATH,
    EMOTIONS,
    TARGET_LAYERS,
    EmotionEngine,
)


SEED = 42
MAX_NEW_TOKENS = 80


PROMPTS: list[str] = [
    # joy
    "Tell me about a moment of pure happiness.",
    "Write a short scene about someone surprising their best friend with a gift.",
    "Describe the feeling of getting accepted to your dream school.",
    "Why do people love watching their kids open presents on Christmas morning?",
    "Describe the emotion of accomplishing something you worked years for.",
    "Tell me about laughing until you cried with someone you love.",
    "Write a paragraph about a wedding day from the bride's perspective.",
    "What is the best part of reuniting with an old friend after years apart?",
    # sadness
    "Describe what grief feels like after losing a parent.",
    "Write a short scene where someone says goodbye to their childhood home for the last time.",
    "What is the worst part of a long relationship ending?",
    "Describe the loneliness of moving to a new city where you know no one.",
    "Tell me about saying goodbye to a dying pet.",
    "Write a paragraph from the perspective of a person watching their parents grow old.",
    "What does it feel like to be the last one out of a beloved restaurant on its closing night?",
    "Describe what it is like to miss someone who has passed away.",
    # anger
    "What does it feel like to be falsely accused of something at work?",
    "Describe the rage of finding out your partner has been cheating for years.",
    "Write a scene about someone confronting a contractor who scammed their elderly mother.",
    "Tell me about being repeatedly cut off in traffic during an already stressful day.",
    "Describe the anger of discovering a coworker has been taking credit for your work.",
    "What is it like when a referee makes a clearly wrong call against your team in a championship?",
    "Write a paragraph about a parent finding out a teacher mistreated their child.",
    "Describe how it feels when a stranger insults your family for no reason.",
    # fear
    "Describe waking up at three a.m. convinced someone is in the house.",
    "What goes through your head right before a job interview that could change your life?",
    "Tell me about hiking alone and realizing you are being followed.",
    "Describe the moment a doctor says 'we need to run more tests.'",
    "Write a scene about a parent realizing they have lost sight of their child in a crowded mall.",
    "What does it feel like to be on a plane during heavy turbulence at night?",
    "Describe driving home in a thick fog with no other cars around.",
    "Tell me about getting a phone call from an unknown number at four a.m.",
    # surprise
    "Describe walking into a surprise birthday party thrown in your honor.",
    "Write a scene about someone discovering a hidden room in the house they have lived in for ten years.",
    "What does it feel like to find out a long-lost sibling exists?",
    "Tell me about opening a letter that turns out to be from someone you have not spoken to in twenty years.",
    "Describe the moment of finding twenty thousand dollars in cash hidden in a thrift store coat pocket.",
    "Write a paragraph about meeting a celebrity you idolized as a child by chance at a coffee shop.",
    "What is it like the moment you realize you have won the lottery?",
    "Describe a friend showing up unexpectedly at your door after five years of silence.",
    # disgust
    "Describe the feeling of finding moldy food forgotten in the back of the fridge.",
    "Write a scene about discovering rotten meat in the kitchen of your favorite restaurant.",
    "What does it feel like to watch a politician lie smugly on television?",
    "Tell me about cleaning up after a sick child has thrown up everywhere.",
    "Describe the feeling of overhearing strangers gossiping cruelly about a close friend.",
    "Write a paragraph from the perspective of someone discovering their idol committed a horrible crime.",
    "What is it like to find rats in your kitchen one morning?",
    "Describe walking through a row of port-a-potties at the end of a music festival.",
    # neutral
    "Explain how a household refrigerator works.",
    "List the prime numbers less than thirty.",
]
assert len(PROMPTS) == 50, f"expected 50 prompts, got {len(PROMPTS)}"


def main() -> int:
    torch.manual_seed(SEED)

    engine = EmotionEngine()
    print(f"\n[calibrate] running {len(PROMPTS)} prompts × {MAX_NEW_TOKENS} tokens", flush=True)
    print(f"[calibrate] target layers: {TARGET_LAYERS}", flush=True)
    print(f"[calibrate] emotions: {EMOTIONS}\n", flush=True)

    # samples[emotion][layer] = list[float]
    samples: dict[str, dict[int, list[float]]] = {
        emo: {L: [] for L in TARGET_LAYERS} for emo in EMOTIONS
    }

    t_total = time.time()
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        n_tokens = 0
        for per_layer in engine.calibration_run(prompt, MAX_NEW_TOKENS):
            n_tokens += 1
            for L, per_emotion in per_layer.items():
                for emo, val in per_emotion.items():
                    samples[emo][L].append(val)
        dt = time.time() - t0
        print(
            f"[{i + 1:>2}/{len(PROMPTS)}] {n_tokens:>3} tokens in {dt:5.1f}s — "
            f"{prompt[:60]}{'...' if len(prompt) > 60 else ''}",
            flush=True,
        )

    print(
        f"\n[calibrate] all done in {time.time() - t_total:.1f}s "
        f"({sum(len(v) for d in samples.values() for v in d.values())} total samples)\n",
        flush=True,
    )

    # Compute percentiles
    out: dict = {}
    for emo in EMOTIONS:
        out[emo] = {}
        for L in TARGET_LAYERS:
            arr = np.array(samples[emo][L], dtype=np.float64)
            p5 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            out[emo][str(L)] = {
                "min": p5,
                "max": p95,
                "method": "percentile_5_95",
                "n": int(arr.size),
            }
            print(
                f"  {emo:<9} layer {L:>2}: p5={p5:7.3f}  p95={p95:7.3f}  "
                f"span={p95 - p5:6.3f}  (n={arr.size})",
                flush=True,
            )

    out["_meta"] = {
        "num_prompts": len(PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "method": "percentile_5_95",
        "seed": SEED,
        "model": "google/gemma-4-E4B-it",
        "target_layers": TARGET_LAYERS,
        "emotions": EMOTIONS,
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_PATH.with_suffix(CALIBRATION_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.replace(CALIBRATION_PATH)
    print(f"\n[calibrate] wrote {CALIBRATION_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
