"""
Percentile-rank calibration against the contrastive corpus.

For each (scope, emotion, layer):
  1. Project every per-token activation from emotion-labeled train-split
     stories onto the final (denoised) emotion vector at that layer.
     Reply scope uses `*_story_*.npz` files; thought scope uses
     `*_thought.npz` files.
  2. Compute 101 percentile breakpoints (0%, 1%, ..., 100%) over the
     resulting per-token scalar distribution.
  3. Save the breakpoints into the scope's `calibration.json`.

At inference time the live backend reads these breakpoints and uses
`np.interp(raw, breakpoints, [0..100])` to convert a raw projection into
a percentile rank against the labeled corpus. The y-axis the user sees
then literally means "X% of this emotion's stories had a projection
≤ this value".

Usage:
    .venv/bin/python -m pipeline.percentile_calibrate
    .venv/bin/python -m pipeline.percentile_calibrate --scope reply
    .venv/bin/python -m pipeline.percentile_calibrate --scope thought
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.inference import (
    DATA_DIR,
    EMOTIONS,
    SCOPES,
    TARGET_LAYERS,
    calibration_path,
    vectors_dir,
)


ACTIVATIONS_DIR = DATA_DIR / "thinking" / "activations"

# Pre-registered topic split (commit 8a6f5f7). Calibration uses ONLY the
# train-topic half — including holdout topics here would leak the
# pre-registered held-out distribution into the percentile reference and
# overstate how "training-grounded" the y-axis actually is.
TOPIC_SPLIT_PATH = Path(__file__).parent / "topic_split.json"
_topic_split = json.loads(TOPIC_SPLIT_PATH.read_text())
TRAIN_TOPICS: set[int] = set(_topic_split["train_topics"])

# Filename topic-index parser. Matches both story and thought NPZ names.
TOPIC_IDX_RE = re.compile(r"_topic_(\d+)_split_")

# 101 anchor points spanning [0, 100] in 1% steps. Stored breakpoints
# at these positions; np.interp(v, breakpoints, PERCENTILE_GRID) gives
# the percentile rank of `v` against the calibration distribution.
PERCENTILE_GRID = np.linspace(0.0, 100.0, 101)


def _topic_idx_from_name(name: str) -> int | None:
    m = TOPIC_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def files_for(emotion: str, scope: str) -> list[Path]:
    """Train-split, TRAIN-TOPIC-only files for this emotion + scope.

    Filtering is enforced by two independent gates:
      1. Filename glob — selects the trial-level train split
         ("_split_train_") for the requested scope (story vs thought).
      2. Topic-index parse — drops any file whose topic_idx is in the
         pre-registered holdout half (topics 80-99). The earlier version
         of this script omitted gate 2, which let ~20% of each emotion's
         calibration corpus come from holdout topics.
    """
    if scope == "reply":
        pattern = f"{emotion}_topic_*_split_train_story_*.npz"
    elif scope == "thought":
        pattern = f"{emotion}_topic_*_split_train_thought.npz"
    else:
        raise ValueError(f"unknown scope: {scope}")
    matched = sorted(ACTIVATIONS_DIR.glob(pattern))
    out: list[Path] = []
    for p in matched:
        topic_idx = _topic_idx_from_name(p.name)
        if topic_idx is None or topic_idx not in TRAIN_TOPICS:
            continue
        out.append(p)
    return out


def load_emotion_vectors(scope: str) -> dict[str, dict[int, np.ndarray]]:
    """{emotion: {layer: vec(2560,) float32}} — the final denoised vectors
    the backend is also projecting against at inference time."""
    base = vectors_dir(scope)
    out: dict[str, dict[int, np.ndarray]] = {}
    for emo in EMOTIONS:
        out[emo] = {}
        for L in TARGET_LAYERS:
            out[emo][L] = np.load(base / f"{emo}_layer_{L}.npy").astype(np.float32)
    return out


def projections_from_file(
    npz_path: Path, emo_vectors: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """{layer: (seq_len,) projection-scalars} for one stored activations file.

    Projects each token's residual stream onto this emotion's vector at
    each target layer. Float16 → float32 promotion happens here to avoid
    accumulating reduced-precision noise.
    """
    data = np.load(npz_path)
    out: dict[int, np.ndarray] = {}
    for L in TARGET_LAYERS:
        acts = data[f"layer_{L}"].astype(np.float32)  # (seq_len, 2560)
        out[L] = acts @ emo_vectors[L]                # (seq_len,)
    return out


def calibrate_scope(scope: str) -> dict:
    print(f"\n=== scope: {scope} ===", flush=True)
    vec_table = load_emotion_vectors(scope)

    cal: dict = {}
    for emo in EMOTIONS:
        files = files_for(emo, scope)
        print(f"  {emo}: {len(files)} files", flush=True)
        if not files:
            print(f"  WARN: no files for {emo}/{scope} — skipping", flush=True)
            continue

        # Stream-aggregate per-layer projections across all files for this
        # emotion. Keep as a list of arrays and concatenate once at the end
        # to avoid O(N^2) appends.
        per_layer: dict[int, list[np.ndarray]] = {L: [] for L in TARGET_LAYERS}
        t0 = time.time()
        for i, f in enumerate(files):
            if i and i % 200 == 0:
                print(
                    f"    {i}/{len(files)} files ({time.time()-t0:.1f}s elapsed)",
                    flush=True,
                )
            projs = projections_from_file(f, vec_table[emo])
            for L, scores in projs.items():
                per_layer[L].append(scores)

        cal[emo] = {}
        for L in TARGET_LAYERS:
            merged = np.concatenate(per_layer[L]) if per_layer[L] else np.empty(0)
            if merged.size == 0:
                print(f"    {emo} layer {L}: 0 samples — skipping", flush=True)
                continue
            breakpoints = np.percentile(merged, PERCENTILE_GRID)
            cal[emo][str(L)] = {
                "method": "percentile_rank",
                "breakpoints": [float(b) for b in breakpoints],
                "n_samples": int(merged.size),
            }
        n_total = sum(
            cal[emo][str(L)]["n_samples"]
            for L in TARGET_LAYERS if str(L) in cal[emo]
        )
        # Median p95-relative anchor for human-readable diagnostics:
        ref = cal[emo].get(str(TARGET_LAYERS[2]), {})
        if ref.get("breakpoints"):
            bp = ref["breakpoints"]
            print(
                f"  {emo}: {n_total:>9,} samples; "
                f"L{TARGET_LAYERS[2]} p1={bp[1]:.2f} p50={bp[50]:.2f} p95={bp[95]:.2f}",
                flush=True,
            )

    cal["_meta"] = {
        "method": "percentile_rank",
        "scope": scope,
        "model": "google/gemma-4-E4B-it",
        "target_layers": TARGET_LAYERS,
        "emotions": EMOTIONS,
        "percentile_grid": PERCENTILE_GRID.tolist(),
        "source": "contrastive corpus, TRAIN topics only (pre-reg 8a6f5f7)",
        "topic_split_file": str(TOPIC_SPLIT_PATH),
        "n_train_topics": len(TRAIN_TOPICS),
    }
    return cal


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=SCOPES + ("both",),
        default="both",
        help="Which vector scope(s) to calibrate (default: both).",
    )
    args = parser.parse_args(list(argv) if argv else None)
    scopes_to_run = SCOPES if args.scope == "both" else (args.scope,)

    for scope in scopes_to_run:
        cal = calibrate_scope(scope)
        out = calibration_path(scope)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(cal, f, indent=2)
        tmp.replace(out)
        print(f"\n[calibrate] wrote {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
