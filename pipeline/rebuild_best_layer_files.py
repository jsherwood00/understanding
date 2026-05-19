"""Rewrite best_layer_per_emotion.json and cosine_matrix.json by HOLDOUT
ACCURACY instead of vector norm.

The original files (written by compute_vectors.py:386-409) selected the
"best" layer per emotion as argmax of the vector L2 norm. That criterion
disagrees with actual classification accuracy — for the reply scope,
norm-best is L13 (uniformly across emotions) while accuracy-best is L25
(0.536 topic-level), and for the thought scope norm-best is L32 while
accuracy-best is L13 (0.487 topic-level).

This script reads holdout_classification_topic_level.json (already saved
by validation.py) and rewrites the two summary files using accuracy.
Per-emotion: layer that maximizes that emotion's RECALL at the topic-
level holdout. Cosine matrix: computed at the global accuracy-best
layer (max of overall topic-level accuracy across layers).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/johnk/Documents/understanding")
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]


def find_cells():
    """Returns list of cutoff_50 dirs that have a holdout JSON to consult."""
    out = []
    for set_dir in (
        ROOT / "data" / "no_thinking" / "vectors" / "full",
        ROOT / "data" / "thinking" / "vectors" / "reply",
        ROOT / "data" / "thinking" / "vectors" / "thought",
    ):
        for cutoff in (0, 25, 50):
            d = set_dir / f"cutoff_{cutoff}"
            if (d / "holdout_classification_topic_level.json").exists():
                out.append(d)
    return out


def rebuild(cell_dir: Path):
    holdout = json.loads(
        (cell_dir / "holdout_classification_topic_level.json").read_text()
    )

    # Layer keys are strings; collect available layers
    layer_keys = sorted(
        (k for k in holdout if k.isdigit()), key=int
    )

    # Per-emotion best layer by recall on topic-level holdout
    per_emo: dict[str, dict] = {}
    for emo in EMOTIONS:
        rows: list[tuple[int, float, float]] = []
        for L_str in layer_keys:
            rec = holdout[L_str]
            pc = rec.get("per_class", {}).get(emo, {})
            recall = pc.get("recall")
            if recall is None:
                continue
            rows.append((int(L_str), float(recall), float(pc.get("precision", 0))))
        if not rows:
            continue
        best_L, best_recall, best_prec = max(rows, key=lambda r: r[1])
        per_emo[emo] = {
            "layer": best_L,
            "recall": best_recall,
            "precision": best_prec,
            "criterion": "max recall on topic-level holdout",
        }

    # Global best layer = layer with highest overall accuracy
    overall: list[tuple[int, float]] = []
    for L_str in layer_keys:
        a = holdout[L_str].get("accuracy")
        if a is None:
            continue
        overall.append((int(L_str), float(a)))
    if overall:
        global_best_L, global_best_acc = max(overall, key=lambda r: r[1])
    else:
        global_best_L, global_best_acc = None, None

    out = {
        "_criterion": "topic-level holdout (topics 80-99). "
                      "Per-emotion best layer = argmax recall on that emotion. "
                      "Global best layer = argmax overall 6-way accuracy.",
        "_source_file": "holdout_classification_topic_level.json",
        "_global_best_layer": global_best_L,
        "_global_best_overall_accuracy": global_best_acc,
        **per_emo,
    }

    out_path = cell_dir / "best_layer_per_emotion.json"
    # Keep the norm-based legacy version for any tooling that still reads it
    legacy = cell_dir / "best_layer_per_emotion.norm_legacy.json"
    if out_path.exists() and not legacy.exists():
        legacy.write_text(out_path.read_text())
    out_path.write_text(json.dumps(out, indent=2))

    print(f"  wrote {out_path.relative_to(ROOT)}")
    print(f"    global best L = {global_best_L} (acc {global_best_acc:.3f})")
    for emo, rec in per_emo.items():
        print(f"    {emo:>9} → L{rec['layer']} (recall {rec['recall']:.3f})")


def rebuild_cosine_at_best(cell_dir: Path):
    """Recompute cosine_matrix.json at the GLOBAL accuracy-best layer."""
    summary_path = cell_dir / "best_layer_per_emotion.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    L = summary.get("_global_best_layer")
    if L is None:
        return

    vecs: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        p = cell_dir / f"{emo}_layer_{L}.npy"
        if not p.exists():
            continue
        vecs[emo] = np.load(p).astype(np.float32)

    if len(vecs) != len(EMOTIONS):
        print(f"    skipping cosine_matrix at L{L}: missing vectors")
        return

    cos: dict[str, dict[str, float]] = {}
    for e1, v1 in vecs.items():
        cos[e1] = {}
        n1 = float(np.linalg.norm(v1)) + 1e-12
        for e2, v2 in vecs.items():
            n2 = float(np.linalg.norm(v2)) + 1e-12
            cos[e1][e2] = float(np.dot(v1, v2) / (n1 * n2))

    out = {
        "_criterion": "computed at the global accuracy-best layer "
                      "(max overall accuracy on topic-level holdout)",
        "layer": L,
        "matrix": cos,
    }
    out_path = cell_dir / "cosine_matrix.json"
    legacy = cell_dir / "cosine_matrix.norm_legacy.json"
    if out_path.exists() and not legacy.exists():
        legacy.write_text(out_path.read_text())
    out_path.write_text(json.dumps(out, indent=2))
    print(f"    rewrote cosine_matrix.json at L{L}")


def main():
    cells = find_cells()
    print(f"rebuilding {len(cells)} cells...")
    for c in cells:
        print(f"\n{c.relative_to(ROOT)}")
        rebuild(c)
        rebuild_cosine_at_best(c)


if __name__ == "__main__":
    main()
