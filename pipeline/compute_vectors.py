"""
compute_vectors.py

CPU-only post-extraction step. Loads per-token NPZs produced by
extract_activations.py, builds three contrastive emotion-vector sets
matching the user spec:

    1. no_thinking_full       (from no_thinking/*_story_*.npz)
    2. thinking_reply         (from thinking/*_story_*.npz)
    3. thinking_thought       (from thinking/*_thought.npz)

Each set is computed at three CUTOFFS in parallel: 0 / 25 / 50. Cutoff k
means "skip the first k tokens of each NPZ before averaging" — research
will pick the most defensible cutoff post-hoc using sensitivity analysis.

Pre-registered synonym filter (commit 6cf25bf) is applied at vector-
compute time. By default we EXCLUDE leaky stories from the contrastive
mean — including them would let the literal emotion-word's residuals
pollute the direction we're trying to extract. With --include-leaky we
ALSO compute a `*_with_leaks` variant for comparison.

PROJECTION-OUT DENOISING: top PCs of the neutral corpus's residuals
that explain 50% of variance are projected out from each contrastive
vector, isolating the emotion-specific direction from the generic
"language is happening" subspace.

OUTPUTS
    data/no_thinking/vectors/cutoff_{N}/full/{emotion}_layer_{L}.npy
    data/thinking/vectors/cutoff_{N}/{thought,reply}/{emotion}_layer_{L}.npy
    plus best_layer_per_emotion.json + cosine_matrix.json per directory.
    With --include-leaky, sibling `*_with_leaks/` directories.

Holdout (story_idx >= 12) activations are NEVER included in the vector
mean — they're for downstream classification validation only.

USAGE
    python -m pipeline.compute_vectors
    python -m pipeline.compute_vectors --include-leaky
    python -m pipeline.compute_vectors --no-denoise
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
CUTOFFS = (0, 25, 50)
TRAIN_TRIALS = 12

# Variance fraction the neutral PCs need to explain before we stop
# accumulating components for the projection-out matrix.
NEUTRAL_PC_VAR_TARGET = 0.50

# Filename patterns produced by extract_activations.py.
STORY_NPZ_RE = re.compile(
    r"^(\w+)_topic_(\d+)_split_(\w+)_story_(\d+)\.npz$"
)
THOUGHT_NPZ_RE = re.compile(
    r"^(\w+)_topic_(\d+)_split_(\w+)_thought\.npz$"
)


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def load_npz_safely(p: Path) -> Optional[dict]:
    try:
        d = np.load(p)
    except Exception:
        return None
    out: dict = {}
    for L in TARGET_LAYERS:
        key = f"layer_{L}"
        if key in d:
            out[L] = d[key]
    if "metadata_json" in d:
        try:
            out["metadata"] = json.loads(str(d["metadata_json"]))
        except Exception:
            out["metadata"] = {}
    else:
        out["metadata"] = {}
    return out


def parse_story_filename(name: str) -> Optional[tuple[str, int, str, int]]:
    m = STORY_NPZ_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def parse_thought_filename(name: str) -> Optional[tuple[str, int, str]]:
    m = THOUGHT_NPZ_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


# ---------------------------------------------------------------------------
# Per-emotion aggregation (one item per NPZ per cutoff per layer)
# ---------------------------------------------------------------------------

def average_with_cutoff(
    layer_arr: np.ndarray, cutoff: int
) -> Optional[np.ndarray]:
    """Mean of layer_arr[cutoff:, :] in float32. None if too few tokens."""
    n = layer_arr.shape[0]
    if n <= cutoff:
        return None
    return layer_arr[cutoff:].astype(np.float32).mean(axis=0)


def aggregate_corpus_stories(
    corpus_dir: Path,
    cutoffs: tuple[int, ...],
    include_leaky: bool,
) -> dict:
    """For story NPZs in this corpus, returns
        out[cutoff][emotion][layer] = list of per-story means
    Train-only (story_idx < TRAIN_TRIALS). Skips leaky stories unless
    include_leaky=True. Returns also stats for logging."""
    activations_dir = corpus_dir / "activations"
    out: dict = {
        c: {e: {L: [] for L in TARGET_LAYERS} for e in EMOTIONS}
        for c in cutoffs
    }
    stats = {
        "total_npz": 0, "loaded": 0, "skipped_holdout": 0,
        "skipped_leaky": 0, "skipped_other": 0,
    }
    if not activations_dir.exists():
        return out, stats

    files = sorted(activations_dir.glob("*_story_*.npz"))
    stats["total_npz"] = len(files)
    print(f"  walking {len(files)} story NPZs in {activations_dir}...")

    for p in files:
        parsed = parse_story_filename(p.name)
        if not parsed:
            stats["skipped_other"] += 1
            continue
        emotion, _topic_idx, _split, story_idx = parsed
        if emotion not in EMOTIONS:
            stats["skipped_other"] += 1
            continue
        if story_idx >= TRAIN_TRIALS:
            stats["skipped_holdout"] += 1
            continue

        d = load_npz_safely(p)
        if d is None:
            stats["skipped_other"] += 1
            continue

        leaks = (d.get("metadata") or {}).get("contains_emotion_word") or []
        if leaks and not include_leaky:
            stats["skipped_leaky"] += 1
            continue

        for cutoff in cutoffs:
            for L in TARGET_LAYERS:
                if L not in d:
                    continue
                avg = average_with_cutoff(d[L], cutoff)
                if avg is not None:
                    out[cutoff][emotion][L].append(avg)
        stats["loaded"] += 1

    return out, stats


def aggregate_thinking_thoughts(
    cutoffs: tuple[int, ...],
) -> dict:
    """One thought NPZ per batch (thinking corpus). Train batches only."""
    activations_dir = ROOT_DATA_DIR / "thinking" / "activations"
    out: dict = {
        c: {e: {L: [] for L in TARGET_LAYERS} for e in EMOTIONS}
        for c in cutoffs
    }
    stats = {
        "total_npz": 0, "loaded": 0, "skipped_holdout": 0,
        "skipped_other": 0,
    }
    if not activations_dir.exists():
        return out, stats

    files = sorted(activations_dir.glob("*_thought.npz"))
    stats["total_npz"] = len(files)
    print(f"  walking {len(files)} thought NPZs in {activations_dir}...")

    for p in files:
        parsed = parse_thought_filename(p.name)
        if not parsed:
            stats["skipped_other"] += 1
            continue
        emotion, _topic_idx, split = parsed
        if emotion not in EMOTIONS:
            stats["skipped_other"] += 1
            continue
        if split == "holdout":
            stats["skipped_holdout"] += 1
            continue

        d = load_npz_safely(p)
        if d is None:
            stats["skipped_other"] += 1
            continue

        for cutoff in cutoffs:
            for L in TARGET_LAYERS:
                if L not in d:
                    continue
                avg = average_with_cutoff(d[L], cutoff)
                if avg is not None:
                    out[cutoff][emotion][L].append(avg)
        stats["loaded"] += 1

    return out, stats


def aggregate_neutral_per_token() -> dict[int, list[np.ndarray]]:
    """For PCA basis: keep all tokens (no scope, no cutoff). The neutral
    corpus has only `_story_*.npz` files (n=1 per prompt, no thought)."""
    out: dict[int, list[np.ndarray]] = {L: [] for L in TARGET_LAYERS}
    activations_dir = ROOT_DATA_DIR / "neutral" / "activations"
    if not activations_dir.exists():
        return out
    files = sorted(activations_dir.glob("*_story_*.npz"))
    print(f"  walking {len(files)} neutral NPZs in {activations_dir}...")
    for p in files:
        d = load_npz_safely(p)
        if d is None:
            continue
        for L in TARGET_LAYERS:
            if L in d:
                out[L].append(d[L].astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Contrastive subtraction + projection-out
# ---------------------------------------------------------------------------

def contrastive_vectors(
    per_emotion_means: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    """vec(E, L) = mean(E, L) - mean({mean(o, L) for o != E})"""
    out: dict[str, dict[int, np.ndarray]] = {e: {} for e in EMOTIONS}
    for E in EMOTIONS:
        others = [e for e in EMOTIONS if e != E]
        for L in TARGET_LAYERS:
            if L not in per_emotion_means.get(E, {}):
                continue
            other_stack = [
                per_emotion_means[o][L] for o in others
                if L in per_emotion_means.get(o, {})
            ]
            if not other_stack:
                continue
            other_mean = np.stack(other_stack, axis=0).mean(axis=0)
            out[E][L] = per_emotion_means[E][L] - other_mean
    return out


def neutral_pcs_per_layer(
    neutral_per_token: dict[int, list[np.ndarray]],
    var_target: float = NEUTRAL_PC_VAR_TARGET,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for L, samples in neutral_per_token.items():
        if not samples:
            continue
        X = np.concatenate(samples, axis=0).astype(np.float32)
        X -= X.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        var = (S ** 2) / max(1, len(S) - 1)
        var_explained = np.cumsum(var) / var.sum()
        k = int(np.searchsorted(var_explained, var_target) + 1)
        k = max(1, min(k, len(S)))
        out[L] = Vt[:k]
        print(
            f"    layer {L}: kept {k} PCs explaining "
            f"{var_explained[k-1]:.2%} of neutral variance"
        )
    return out


def project_out(vec: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    if pcs is None or len(pcs) == 0:
        return vec
    coeffs = pcs @ vec
    return vec - pcs.T @ coeffs


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_vector_set(
    out_dir: Path,
    vectors: dict[str, dict[int, np.ndarray]],
    label: str,
    counts: dict[str, int],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    norms_table: dict[str, list[tuple[int, float]]] = {}
    for E in EMOTIONS:
        rows: list[tuple[int, float]] = []
        for L in TARGET_LAYERS:
            if L not in vectors.get(E, {}):
                continue
            v = vectors[E][L]
            np.save(out_dir / f"{E}_layer_{L}.npy", v)
            rows.append((L, float(np.linalg.norm(v))))
        norms_table[E] = rows

    print(f"  -- {label} --")
    print(f"    {'emotion':<10} | n   | " + " | ".join(f"L{L:>2}" for L in TARGET_LAYERS))
    print("    " + "-" * (16 + 8 * len(TARGET_LAYERS)))
    for E in EMOTIONS:
        cells: list[str] = []
        rows = dict(norms_table[E])
        for L in TARGET_LAYERS:
            cells.append(f"{rows[L]:>5.2f}" if L in rows else "  -- ")
        print(f"    {E:<10} | {counts.get(E, 0):>3} | " + " | ".join(cells))

    best: dict[str, dict] = {}
    for E in EMOTIONS:
        rows = norms_table[E]
        if not rows:
            continue
        best_layer, best_norm = max(rows, key=lambda r: r[1])
        best[E] = {"layer": best_layer, "norm": best_norm}
    with open(out_dir / "best_layer_per_emotion.json", "w") as f:
        json.dump(best, f, indent=2)

    if not best:
        return

    common_best = Counter(b["layer"] for b in best.values()).most_common(1)[0][0]
    cos_at: dict[str, dict[str, float]] = {}
    vecs_at = {
        E: vectors[E][common_best]
        for E in EMOTIONS
        if common_best in vectors.get(E, {})
    }
    for e1, v1 in vecs_at.items():
        cos_at[e1] = {}
        n1 = np.linalg.norm(v1) + 1e-9
        for e2, v2 in vecs_at.items():
            cos_at[e1][e2] = float(np.dot(v1, v2) / (n1 * (np.linalg.norm(v2) + 1e-9)))
    with open(out_dir / "cosine_matrix.json", "w") as f:
        json.dump({"layer": common_best, "matrix": cos_at}, f, indent=2)


# ---------------------------------------------------------------------------
# One vector-set worker
# ---------------------------------------------------------------------------

def compute_set(
    *,
    nested: dict,
    cutoffs: tuple[int, ...],
    out_root: Path,
    set_label: str,
    neutral_pcs: Optional[dict[int, np.ndarray]],
):
    """`nested` is the {cutoff: {emotion: {layer: [arrays]}}} dict from an
    aggregate_*. Writes vectors per cutoff into out_root/cutoff_{N}/."""
    for cutoff in cutoffs:
        grouped = nested[cutoff]
        counts = {E: len(grouped[E][TARGET_LAYERS[0]]) for E in EMOTIONS}
        min_count = min(counts.values()) if counts else 0
        if min_count < 2:
            print(
                f"  cutoff={cutoff}: SKIP (min count {min_count} per "
                f"emotion < 2). Counts: {counts}"
            )
            continue

        means: dict[str, dict[int, np.ndarray]] = {e: {} for e in EMOTIONS}
        for E in EMOTIONS:
            for L in TARGET_LAYERS:
                if grouped[E][L]:
                    means[E][L] = np.stack(grouped[E][L], axis=0).mean(axis=0)

        raw = contrastive_vectors(means)
        if neutral_pcs:
            denoised: dict[str, dict[int, np.ndarray]] = {e: {} for e in EMOTIONS}
            for E in EMOTIONS:
                for L, v in raw[E].items():
                    pcs = neutral_pcs.get(L)
                    denoised[E][L] = project_out(v, pcs) if pcs is not None else v
        else:
            denoised = raw

        out_dir = out_root / f"cutoff_{cutoff}"
        write_vector_set(
            out_dir, denoised,
            label=f"{set_label} cutoff={cutoff} (denoised)",
            counts=counts,
        )
        write_vector_set(
            out_dir.with_name(f"cutoff_{cutoff}_raw"), raw,
            label=f"{set_label} cutoff={cutoff} (raw — no PC projection)",
            counts=counts,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute contrastive emotion vectors.")
    p.add_argument(
        "--include-leaky",
        action="store_true",
        help="Build a sibling vector set including leaky stories. "
             "Default is to exclude leaky stories per pre-reg filter.",
    )
    p.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip projection-out denoising. Useful when neutral corpus "
             "isn't ready yet.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Build neutral PCs (shared across all sets).
    neutral_pcs: Optional[dict[int, np.ndarray]] = None
    if not args.no_denoise:
        print("\n==== neutral PCs (for projection-out denoising) ====")
        per_token = aggregate_neutral_per_token()
        if any(per_token[L] for L in TARGET_LAYERS):
            neutral_pcs = neutral_pcs_per_layer(per_token)
        else:
            print("  no neutral activations found — proceeding without denoising.")
    else:
        print("[--no-denoise] skipping projection-out step.")

    runs: list[tuple[bool, str]] = [(False, "clean")]
    if args.include_leaky:
        runs.append((True, "with_leaks"))

    for include_leaky, suffix_tag in runs:
        out_suffix = "" if not include_leaky else "_with_leaks"

        # Set 1: no_thinking, full
        print(f"\n==== set 1: no_thinking_full ({suffix_tag}) ====")
        nested, stats = aggregate_corpus_stories(
            ROOT_DATA_DIR / "no_thinking", CUTOFFS, include_leaky,
        )
        print(f"  loaded {stats['loaded']}, "
              f"holdout-skipped {stats['skipped_holdout']}, "
              f"leak-skipped {stats['skipped_leaky']}, "
              f"other-skipped {stats['skipped_other']}")
        compute_set(
            nested=nested, cutoffs=CUTOFFS,
            out_root=ROOT_DATA_DIR / "no_thinking" / "vectors" / f"full{out_suffix}",
            set_label=f"no_thinking_full {suffix_tag}",
            neutral_pcs=neutral_pcs,
        )

        # Set 2: thinking, reply
        print(f"\n==== set 2: thinking_reply ({suffix_tag}) ====")
        nested, stats = aggregate_corpus_stories(
            ROOT_DATA_DIR / "thinking", CUTOFFS, include_leaky,
        )
        print(f"  loaded {stats['loaded']}, "
              f"holdout-skipped {stats['skipped_holdout']}, "
              f"leak-skipped {stats['skipped_leaky']}, "
              f"other-skipped {stats['skipped_other']}")
        compute_set(
            nested=nested, cutoffs=CUTOFFS,
            out_root=ROOT_DATA_DIR / "thinking" / "vectors" / f"reply{out_suffix}",
            set_label=f"thinking_reply {suffix_tag}",
            neutral_pcs=neutral_pcs,
        )

    # Set 3: thinking, thought (leak filter not applied to thoughts —
    # the model's reasoning often quotes the emotion word explicitly,
    # which is fine for thought-level analysis).
    print(f"\n==== set 3: thinking_thought ====")
    nested, stats = aggregate_thinking_thoughts(CUTOFFS)
    print(f"  loaded {stats['loaded']}, "
          f"holdout-skipped {stats['skipped_holdout']}, "
          f"other-skipped {stats['skipped_other']}")
    compute_set(
        nested=nested, cutoffs=CUTOFFS,
        out_root=ROOT_DATA_DIR / "thinking" / "vectors" / "thought",
        set_label="thinking_thought",
        neutral_pcs=neutral_pcs,
    )

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
