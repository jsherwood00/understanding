"""
verify_dataset.py

Quick sanity check on the contrastive pipeline output.
Run anytime (even while contrastive_pipeline.py is still running) to verify
the dataset is generating properly.

Checks:
  1. File integrity: every .npz loads, has all expected layers, no NaN/Inf
  2. Distribution balance: counts across (emotion, topic, trial)
  3. Story length distribution: tokens per story, flag outliers
  4. Activation health: norm distribution per layer per emotion
  5. Prompt constraint check: does the model leak the emotion word into stories?
  6. Sample previews: random stories per emotion to eyeball quality

Run:
    python verify_dataset.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]
NUM_TOPICS = 100
STORIES_PER_TOPIC = 12

# Words that should NOT appear in the stories per the paper's prompt
# (the model is told never to use the emotion word or direct synonyms)
EMOTION_WORDS_TO_CHECK = {
    "joy":      ["joy", "joyful", "joyous", "happy", "happiness"],
    "sadness":  ["sadness", "sad", "sorrow", "sorrowful"],
    "anger":    ["anger", "angry", "rage", "furious"],
    "fear":     ["fear", "afraid", "scared", "fearful"],
    "surprise": ["surprise", "surprised", "shocked"],
    "disgust":  ["disgust", "disgusted", "revolted"],
}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    if not (DATA_DIR / "activations").exists():
        print("ERROR: data/activations not found. has the pipeline been run?")
        return

    # ============================================================================
    # 1. Inventory: which combos exist, which are missing
    # ============================================================================
    section("1. INVENTORY")

    activation_files = list((DATA_DIR / "activations").glob("*.npz"))
    story_files = list((DATA_DIR / "stories").glob("*.json"))

    print(f"activation files: {len(activation_files)}")
    print(f"story files:      {len(story_files)}")
    expected_total = len(EMOTIONS) * NUM_TOPICS * STORIES_PER_TOPIC
    print(f"expected total:   {expected_total}")
    pct = 100 * len(activation_files) / expected_total
    print(f"completion:       {pct:.1f}%")

    if len(activation_files) != len(story_files):
        print(f"\nWARNING: activation/story file count mismatch.")

    # ============================================================================
    # 2. Distribution: counts per emotion, per (emotion, trial)
    # ============================================================================
    section("2. DISTRIBUTION BALANCE")

    pattern = re.compile(r"^(\w+)_topic_(\d+)_story_(\d+)$")
    by_emotion = Counter()
    by_emotion_trial = defaultdict(Counter)

    for f in activation_files:
        m = pattern.match(f.stem)
        if not m:
            print(f"  WARNING: unexpected filename: {f.name}")
            continue
        emotion, t_idx, s_idx = m.group(1), int(m.group(2)), int(m.group(3))
        if emotion not in EMOTIONS:
            print(f"  WARNING: unknown emotion in {f.name}")
            continue
        by_emotion[emotion] += 1
        by_emotion_trial[emotion][s_idx] += 1

    print("counts per emotion (out of 1200 max):")
    for e in EMOTIONS:
        n = by_emotion[e]
        bar = "#" * int(40 * n / 1200) if n else ""
        print(f"  {e:<10s} {n:>4d}  {bar}")

    print("\ncounts per (emotion, trial) — should be balanced if midway through a trial pass:")
    print(f"  {'emotion':<10s} | " + " | ".join(f"t{i:>2d}" for i in range(STORIES_PER_TOPIC)))
    print("  " + "-" * (12 + 6 * STORIES_PER_TOPIC))
    for e in EMOTIONS:
        row = [f"{by_emotion_trial[e].get(i, 0):>3d}" for i in range(STORIES_PER_TOPIC)]
        print(f"  {e:<10s} | " + " | ".join(row))

    # ============================================================================
    # 3. File integrity: NaN/Inf checks, layer presence
    # ============================================================================
    section("3. FILE INTEGRITY")

    bad_files = []
    nan_files = []
    missing_layer_files = []
    norm_stats = defaultdict(lambda: defaultdict(list))  # emotion -> layer -> [norms]

    for f in activation_files:
        m = pattern.match(f.stem)
        if not m:
            continue
        emotion = m.group(1)
        try:
            d = np.load(f)
        except Exception as e:
            bad_files.append((f.name, str(e)))
            continue

        for L in TARGET_LAYERS:
            key = f"layer_{L}"
            if key not in d:
                missing_layer_files.append((f.name, L))
                continue
            arr = d[key]
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                nan_files.append((f.name, L))
                continue
            norm_stats[emotion][L].append(float(np.linalg.norm(arr)))

    if bad_files:
        print(f"WARNING: {len(bad_files)} corrupt files:")
        for name, err in bad_files[:5]:
            print(f"  {name}: {err}")
    else:
        print("all files load cleanly")

    if missing_layer_files:
        print(f"\nWARNING: {len(missing_layer_files)} files missing expected layers")
    else:
        print("all expected layers present in every file")

    if nan_files:
        print(f"\nWARNING: {len(nan_files)} files contain NaN/Inf:")
        for name, L in nan_files[:5]:
            print(f"  {name} layer {L}")
    else:
        print("no NaN/Inf values detected")

    # ============================================================================
    # 4. Activation health: norm distribution per layer per emotion
    # ============================================================================
    section("4. ACTIVATION NORM STATISTICS")

    print("expected: similar mean/std across emotions at the same layer")
    print("anomaly:  one emotion's norms way different from others would suggest a problem\n")

    print(f"{'emotion':<10s} | " + " | ".join(f"L{L:>2d} (mean +/- std)" for L in TARGET_LAYERS))
    print("-" * (12 + 22 * len(TARGET_LAYERS)))
    for e in EMOTIONS:
        cells = []
        for L in TARGET_LAYERS:
            vals = norm_stats[e][L]
            if not vals:
                cells.append(f"{'no data':>20s}")
            else:
                cells.append(f"{np.mean(vals):>7.2f} +/-{np.std(vals):>5.2f}  ")
        print(f"{e:<10s} | " + " | ".join(cells))

    # ============================================================================
    # 5. Story length distribution
    # ============================================================================
    section("5. STORY LENGTHS")

    lengths_by_emotion = defaultdict(list)
    too_short = []

    for f in story_files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        emotion = d.get("emotion")
        n_tok = d.get("gen_tokens")
        if emotion and n_tok is not None:
            lengths_by_emotion[emotion].append(n_tok)
            if n_tok < 80:
                too_short.append((f.name, n_tok))

    print("token count distribution per emotion:")
    print(f"{'emotion':<10s} | min   p25   median  p75   max    mean")
    print("-" * 60)
    for e in EMOTIONS:
        v = lengths_by_emotion[e]
        if not v:
            print(f"{e:<10s} | (no data)")
            continue
        v_arr = np.array(v)
        print(f"{e:<10s} | {v_arr.min():>4d}  {int(np.percentile(v_arr, 25)):>4d}  "
              f"{int(np.median(v_arr)):>5d}   {int(np.percentile(v_arr, 75)):>4d}  "
              f"{v_arr.max():>4d}   {v_arr.mean():>5.1f}")

    if too_short:
        print(f"\n{len(too_short)} stories under 80 tokens (would be skipped by extraction):")
        for name, n in too_short[:5]:
            print(f"  {name}: {n} tokens")

    # ============================================================================
    # 6. Prompt constraint: does the model leak the emotion word into stories?
    # ============================================================================
    section("6. PROMPT CONSTRAINT CHECK")

    print("paper's methodology requires the emotion word and synonyms NOT appear in stories")
    print("(stories should convey emotion through indirect means only)\n")

    leak_counts = Counter()
    leak_examples = defaultdict(list)
    total_per_emotion = Counter()

    for f in story_files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        emotion = d.get("emotion")
        story = d.get("story", "").lower()
        if not emotion or emotion not in EMOTION_WORDS_TO_CHECK:
            continue

        total_per_emotion[emotion] += 1
        forbidden = EMOTION_WORDS_TO_CHECK[emotion]
        leaked_words = []
        for word in forbidden:
            # word boundary match to avoid e.g. "fearless" hitting "fear"
            if re.search(rf"\b{re.escape(word)}\b", story):
                leaked_words.append(word)

        if leaked_words:
            leak_counts[emotion] += 1
            if len(leak_examples[emotion]) < 3:
                leak_examples[emotion].append((f.name, leaked_words))

    print(f"{'emotion':<10s} | leaked / total | %     | sample leaked words")
    print("-" * 70)
    for e in EMOTIONS:
        total = total_per_emotion[e]
        leaked = leak_counts[e]
        if total == 0:
            print(f"{e:<10s} | (no data)")
            continue
        pct = 100 * leaked / total
        sample = ""
        if leak_examples[e]:
            sample = ", ".join(f"{w}" for _, words in leak_examples[e][:1] for w in words)
        print(f"{e:<10s} | {leaked:>3d} / {total:<6d} | {pct:>4.1f}% | {sample}")

    print("\nsmall leak rates (~5-15%) are normal -- model is told 'never' but sometimes slips")
    print("very high leak rate (>40%) for one emotion would suggest the prompt isn't being followed")

    # ============================================================================
    # 7. Random story previews per emotion
    # ============================================================================
    section("7. RANDOM STORY PREVIEWS")

    rng = random.Random(42)
    by_emotion_files = defaultdict(list)
    for f in story_files:
        m = pattern.match(f.stem)
        if m:
            by_emotion_files[m.group(1)].append(f)

    for e in EMOTIONS:
        files = by_emotion_files[e]
        if not files:
            print(f"\n{e}: (no stories yet)")
            continue
        sample = rng.choice(files)
        try:
            with open(sample) as fh:
                d = json.load(fh)
            preview = d["story"][:300].replace("\n", " ")
            print(f"\n[{e}] {sample.name}")
            print(f"  topic: {d['topic']}")
            print(f"  preview: {preview}...")
        except Exception as ex:
            print(f"\n[{e}] {sample.name}: ERROR reading: {ex}")

    # ============================================================================
    # Summary
    # ============================================================================
    section("VERIFICATION COMPLETE")
    print(f"  total stories:      {len(activation_files)}")
    print(f"  corrupt files:      {len(bad_files)}")
    print(f"  files with NaN:     {len(nan_files)}")
    print(f"  too-short stories:  {len(too_short)}")
    print(f"  emotion-word leaks: {sum(leak_counts.values())} stories")

    if not (bad_files or nan_files):
        print("\nDataset looks healthy.")
    else:
        print("\nIssues detected; see warnings above.")


if __name__ == "__main__":
    main()
