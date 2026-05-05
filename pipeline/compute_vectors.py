"""
compute_vectors.py

Compute contrastive emotion vectors from the saved activations.
Implements the paper-accurate baseline: mean(emotion E) - mean(other emotions).

For each emotion at each layer:
    emotion_vector = mean(activations for that emotion) - mean(per-emotion means of other emotions)

The "mean of per-emotion means" baseline (rather than "mean across all other-emotion stories") prevents
emotions with more stories from dominating the baseline.

Run AFTER contrastive_pipeline.py finishes.

Output:
    data/vectors/{emotion}_layer_{L}.npy   # the contrastive vectors
    data/vectors/best_layer_per_emotion.json  # which layer has highest norm per emotion
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]


def load_activations_grouped() -> dict[str, dict[int, list[np.ndarray]]]:
    """Returns {emotion: {layer: [array, array, ...]}}"""
    grouped = {e: {L: [] for L in TARGET_LAYERS} for e in EMOTIONS}
    for emotion in EMOTIONS:
        prefix = f"{emotion}_topic_"
        files = sorted((DATA_DIR / "activations").glob(f"{prefix}*.npz"))
        for f in files:
            d = np.load(f)
            for L in TARGET_LAYERS:
                key = f"layer_{L}"
                if key in d:
                    grouped[emotion][L].append(d[key])
    return grouped


def main():
    print("loading saved activations...")
    grouped = load_activations_grouped()

    print("\ncounts per emotion:")
    for e in EMOTIONS:
        n = len(grouped[e][TARGET_LAYERS[0]])
        print(f"  {e:<10s}: {n}")

    # validate we have enough stories per emotion
    min_count = min(len(grouped[e][TARGET_LAYERS[0]]) for e in EMOTIONS)
    if min_count < 2:
        print(f"\nERROR: only {min_count} stories for some emotion. Need >=2 to compute means.")
        return

    # per-emotion mean per layer
    print("\ncomputing per-emotion mean activations...")
    means = {}
    for e in EMOTIONS:
        means[e] = {}
        for L in TARGET_LAYERS:
            stack = np.stack(grouped[e][L], axis=0)  # (num_stories, hidden_dim)
            means[e][L] = stack.mean(axis=0)         # (hidden_dim,)

    # contrastive vectors: mean(E) - mean(other emotion means)
    out_dir = DATA_DIR / "vectors"
    out_dir.mkdir(exist_ok=True)

    print("\ncomputing contrastive vectors and writing to disk...")
    print(f"\n{'emotion':<10} | " + " | ".join(f"L{L:>2}" for L in TARGET_LAYERS))
    print("-" * (12 + 8 * len(TARGET_LAYERS)))

    norms_table = {}
    for E in EMOTIONS:
        others = [e for e in EMOTIONS if e != E]
        norms = []
        for L in TARGET_LAYERS:
            other_mean = np.stack([means[o][L] for o in others], axis=0).mean(axis=0)
            vec = means[E][L] - other_mean
            np.save(out_dir / f"{E}_layer_{L}.npy", vec)
            norms.append(float(np.linalg.norm(vec)))
        norms_table[E] = norms
        norms_str = " | ".join(f"{n:>5.2f}" for n in norms)
        print(f"{E:<10s} | {norms_str}")

    # pick best layer per emotion (highest L2 norm = most discriminative)
    best = {}
    for E in EMOTIONS:
        norms = norms_table[E]
        best_idx = int(np.argmax(norms))
        best[E] = {
            "layer": TARGET_LAYERS[best_idx],
            "norm": norms[best_idx],
        }

    print("\nbest layer per emotion (highest contrastive vector norm):")
    for E, info in best.items():
        print(f"  {E:<10s} -> layer {info['layer']:>2} (norm={info['norm']:.2f})")

    with open(out_dir / "best_layer_per_emotion.json", "w") as f:
        json.dump(best, f, indent=2)

    # cosine similarity matrix between contrastive vectors at the most-common best layer
    from collections import Counter
    common_best = Counter(info["layer"] for info in best.values()).most_common(1)[0][0]
    print(f"\ncosine similarity between contrastive vectors at layer {common_best}:")
    print(f"(low/negative values = good; means emotions are well-separated)\n")

    vecs = {E: np.load(out_dir / f"{E}_layer_{common_best}.npy") for E in EMOTIONS}
    print(f"{'':<10} | " + " | ".join(f"{e[:6]:>6}" for e in vecs))
    print("-" * (12 + 9 * len(vecs)))
    for e1, v1 in vecs.items():
        sims = []
        for e2, v2 in vecs.items():
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            sims.append(f"{cos:>6.2f}")
        print(f"{e1:<10} | " + " | ".join(sims))

    print(f"\ndone. vectors saved to {out_dir}/")
    print(f"contrastive vectors: 6 emotions x {len(TARGET_LAYERS)} layers = {6*len(TARGET_LAYERS)} files")
    print(f"best layer per emotion saved to {out_dir}/best_layer_per_emotion.json")


if __name__ == "__main__":
    main()
