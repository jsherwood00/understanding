"""
validation.py

Post-vector-computation validation. Three checks, all CPU-only except
the logit lens which needs the model loaded for the unembedding matrix.

  1. logit_lens: project each emotion vector through the model's
     unembedding (lm_head.weight) and report which tokens light up most.
     A clean emotion direction should upweight emotion-related tokens.
  2. holdout_classification: per held-out story, compute its mean
     residual at each layer (using the same scope+cutoff that produced
     the vectors), classify it by max-cosine to the contrastive vectors,
     and compare to the true emotion label. Reports per-emotion
     precision/recall and a confusion matrix.
  3. topic_balance: count how many distinct topics contributed to each
     emotion's set of vectors. Flags any emotion whose top-1 topic is
     more than topic_dominance_threshold of its data.

USAGE
    python -m pipeline.validation --corpus no_thinking --scope full --cutoff 0
    python -m pipeline.validation --corpus thinking --scope reply --cutoff 25 --logit-lens
    python -m pipeline.validation --corpus thinking --all       # runs every cutoff x scope

The logit lens needs the model loaded — that's GPU work. Skip it with
--no-logit-lens if the GPU is busy with another pipeline phase.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


ROOT_DATA_DIR = Path("data")
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]
PHASE_THOUGHT = np.uint8(0)
PHASE_REPLY = np.uint8(1)
TRAIN_TRIALS = 12  # trials >= 12 are holdout

STORY_NPZ_RE = re.compile(
    r"^(\w+)_topic_(\d+)_split_(\w+)_story_(\d+)\.npz$"
)
THOUGHT_NPZ_RE = re.compile(
    r"^(\w+)_topic_(\d+)_split_(\w+)_thought\.npz$"
)


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(out_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    """Reads {emotion}_layer_{L}.npy from out_dir into nested dict."""
    out: dict[str, dict[int, np.ndarray]] = {e: {} for e in EMOTIONS}
    for E in EMOTIONS:
        for L in TARGET_LAYERS:
            p = out_dir / f"{E}_layer_{L}.npy"
            if p.exists():
                out[E][L] = np.load(p)
    return out


# ---------------------------------------------------------------------------
# Holdout classification (new batched-corpus layout)
# ---------------------------------------------------------------------------

def average_with_cutoff(layer_arr: np.ndarray, cutoff: int) -> Optional[np.ndarray]:
    n = layer_arr.shape[0]
    if n <= cutoff:
        return None
    return layer_arr[cutoff:].astype(np.float32).mean(axis=0)


def holdout_means(
    corpus_dir: Path,
    scope: str,
    cutoff: int,
) -> dict[int, list[tuple[str, np.ndarray]]]:
    """For each layer, returns list of (true_emotion, mean-vec) for every
    holdout NPZ matching `scope`:
        scope="full"     → corpus_dir/activations/*_split_holdout_story_*.npz
        scope="reply"    → same (story NPZs in thinking corpus)
        scope="thought"  → corpus_dir/activations/*_split_holdout_thought.npz
    """
    out: dict[int, list[tuple[str, np.ndarray]]] = {L: [] for L in TARGET_LAYERS}
    activations_dir = corpus_dir / "activations"
    if not activations_dir.exists():
        return out

    if scope in ("full", "reply"):
        pattern = "*_split_holdout_story_*.npz"
        regex = STORY_NPZ_RE
    elif scope == "thought":
        pattern = "*_split_holdout_thought.npz"
        regex = THOUGHT_NPZ_RE
    else:
        raise ValueError(f"unknown scope {scope!r}")

    for p in sorted(activations_dir.glob(pattern)):
        m = regex.match(p.name)
        if not m:
            continue
        emotion = m.group(1)
        if emotion not in EMOTIONS:
            continue
        try:
            d = np.load(p)
        except Exception:
            continue
        for L in TARGET_LAYERS:
            key = f"layer_{L}"
            if key not in d:
                continue
            avg = average_with_cutoff(d[key], cutoff)
            if avg is not None:
                out[L].append((emotion, avg))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = (np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / n)


def classify_holdout(
    holdout: list[tuple[str, np.ndarray]],
    vectors: dict[str, np.ndarray],
) -> dict:
    """Predict each holdout story's emotion by argmax cosine over the
    emotion vectors. Returns accuracy + confusion matrix + per-class
    precision/recall."""
    if not holdout or not vectors:
        return {"n": 0, "accuracy": None}
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for true_e, vec in holdout:
        scores = {E: cosine(vec, v) for E, v in vectors.items()}
        if not scores:
            continue
        pred = max(scores, key=lambda k: scores[k])
        confusion[true_e][pred] += 1
        if pred == true_e:
            correct += 1
    n = len(holdout)
    per_class: dict[str, dict[str, float]] = {}
    for E in EMOTIONS:
        tp = confusion[E].get(E, 0)
        total_true = sum(confusion[E].values())
        total_pred = sum(c.get(E, 0) for c in confusion.values())
        precision = tp / total_pred if total_pred else 0.0
        recall = tp / total_true if total_true else 0.0
        per_class[E] = {"precision": precision, "recall": recall, "n_true": total_true}
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "per_class": per_class,
        "confusion": {e: dict(c) for e, c in confusion.items()},
    }


# ---------------------------------------------------------------------------
# Topic balance
# ---------------------------------------------------------------------------

def topic_balance(
    corpus_dir: Path,
    train_trials: int = TRAIN_TRIALS,
    dominance_threshold: float = 0.05,
) -> dict:
    """Counts how many train stories per (emotion, topic). Flags any
    (emotion, topic) pair whose share exceeds dominance_threshold of
    that emotion's training set."""
    counts: dict[str, Counter] = {e: Counter() for e in EMOTIONS}
    activations_dir = corpus_dir / "activations"
    if not activations_dir.exists():
        return {"available": False}
    for p in activations_dir.glob("*_split_train_story_*.npz"):
        m = STORY_NPZ_RE.match(p.name)
        if not m:
            continue
        emotion, topic_idx = m.group(1), int(m.group(2))
        if emotion not in EMOTIONS:
            continue
        counts[emotion][topic_idx] += 1
    flags: list[dict] = []
    out: dict = {"available": True, "per_emotion": {}}
    for E in EMOTIONS:
        total = sum(counts[E].values())
        if total == 0:
            continue
        topic_share = [(t, n / total) for t, n in counts[E].most_common(5)]
        out["per_emotion"][E] = {
            "n_train": total,
            "distinct_topics": len(counts[E]),
            "top5_share": topic_share,
        }
        for t, share in topic_share:
            if share > dominance_threshold:
                flags.append({
                    "emotion": E,
                    "topic_idx": t,
                    "share": share,
                    "threshold": dominance_threshold,
                })
    out["flags"] = flags
    return out


# ---------------------------------------------------------------------------
# Logit lens (GPU)
# ---------------------------------------------------------------------------

def logit_lens(
    vectors: dict[str, np.ndarray],
    layer: int,
    top_k: int = 25,
) -> dict[str, list[tuple[str, float]]]:
    """For each emotion vector at the given layer, multiply by the
    model's unembedding matrix and report top_k tokens by raw logit
    score. Loads the model once on GPU; vectors have to come from the
    same architecture."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("  [logit_lens] loading model in 4-bit nf4 for unembedding...")
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it",
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()

    # lm_head.weight: (vocab, hidden). Casting to fp32 for clean dot.
    W = model.lm_head.weight.to(torch.float32).detach().cpu().numpy()
    out: dict[str, list[tuple[str, float]]] = {}
    for E, v in vectors.items():
        scores = W @ v.astype(np.float32)  # (vocab,)
        top = np.argsort(-scores)[:top_k]
        out[E] = [(tok.decode([int(i)]), float(scores[int(i)])) for i in top]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Validate contrastive emotion vectors.")
    p.add_argument(
        "--corpus",
        choices=("no_thinking", "thinking"),
        required=True,
    )
    p.add_argument(
        "--scope",
        choices=("full", "thought", "reply"),
        default="full",
        help="Which scope's vectors to validate. Default: full.",
    )
    p.add_argument(
        "--cutoff",
        type=int,
        default=0,
        help="Which cutoff's vectors to validate. Default: 0.",
    )
    p.add_argument("--all", action="store_true",
                   help="Run every (scope, cutoff) combination available "
                        "for this corpus.")
    p.add_argument("--logit-lens", action="store_true",
                   help="Run the GPU logit-lens projection (skipped by default).")
    p.add_argument("--no-classification", action="store_true",
                   help="Skip holdout classification (CPU but slow on big corpora).")
    return p.parse_args()


def run_one(corpus: str, scope: str, cutoff: int, do_logit_lens: bool, do_classification: bool):
    corpus_dir = ROOT_DATA_DIR / corpus
    out_dir = corpus_dir / "vectors" / f"cutoff_{cutoff}" / scope
    if not out_dir.exists():
        print(f"  no vectors at {out_dir} — skip.")
        return

    print(f"\n==== validate {corpus} / cutoff={cutoff} / scope={scope} ====")
    vectors = load_vectors(out_dir)
    if not any(v for v in vectors.values()):
        print("  no vectors loaded — skip.")
        return

    # Topic balance — fast.
    print("\n  topic balance:")
    bal = topic_balance(corpus_dir)
    if bal.get("available"):
        for E, info in bal["per_emotion"].items():
            print(f"    {E:<10}: n={info['n_train']:>4} across "
                  f"{info['distinct_topics']:>3} topics; "
                  f"top {info['top5_share'][0][1]:.2%} from one topic")
        if bal["flags"]:
            print(f"    !! {len(bal['flags'])} (emotion, topic) pairs over "
                  f"dominance threshold")

    # Holdout classification.
    if do_classification:
        print("\n  holdout classification:")
        holdout = holdout_means(corpus_dir, scope, cutoff)
        for L in TARGET_LAYERS:
            layer_holdout = holdout.get(L, [])
            layer_vecs = {E: vectors[E][L] for E in EMOTIONS if L in vectors[E]}
            res = classify_holdout(layer_holdout, layer_vecs)
            if res["n"] == 0:
                print(f"    layer {L:>2}: no holdout data")
                continue
            print(f"    layer {L:>2}: n={res['n']:>3}, acc={res['accuracy']:.2%}")
            with open(out_dir / f"holdout_classification_layer_{L}.json", "w") as f:
                json.dump(res, f, indent=2)

    # Logit lens — pick the most-common best layer.
    if do_logit_lens:
        # Find best layer per emotion → most common.
        best_layers = []
        for E in EMOTIONS:
            if not vectors[E]:
                continue
            best_layers.append(
                max(vectors[E].keys(), key=lambda L: float(np.linalg.norm(vectors[E][L])))
            )
        if not best_layers:
            return
        common = Counter(best_layers).most_common(1)[0][0]
        print(f"\n  logit lens at layer {common}:")
        layer_vecs = {E: vectors[E][common] for E in EMOTIONS if common in vectors[E]}
        results = logit_lens(layer_vecs, layer=common, top_k=25)
        for E, top in results.items():
            print(f"    {E:<10} top tokens: " + ", ".join(
                f"{tok!r}({s:.1f})" for tok, s in top[:8]
            ))
        with open(out_dir / f"logit_lens_layer_{common}.json", "w") as f:
            json.dump(results, f, indent=2)


def main():
    t0 = time.time()
    args = parse_args()

    if args.all:
        scopes = (
            ("full",) if args.corpus == "no_thinking"
            else ("full", "thought", "reply")
        )
        for cutoff in (0, 25, 50):
            for scope in scopes:
                run_one(
                    args.corpus, scope, cutoff,
                    do_logit_lens=args.logit_lens,
                    do_classification=not args.no_classification,
                )
    else:
        run_one(
            args.corpus, args.scope, args.cutoff,
            do_logit_lens=args.logit_lens,
            do_classification=not args.no_classification,
        )

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
