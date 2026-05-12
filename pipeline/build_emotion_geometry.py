"""Build the 2D MDS embedding of the 6 emotion vectors per runtime scope.

Loads the denoised emotion vectors at each scope's runtime layer:
  reply scope   ← data/thinking/vectors/reply/cutoff_50/   layer 25  (V3)
  thought scope ← data/thinking/vectors/thought/cutoff_50/ layer 13  (V2)

For each scope:
  1. Cosine similarity matrix between the 6 emotion vectors.
  2. Distance matrix D = 1 - cosine_similarity.
  3. sklearn.manifold.MDS(dissimilarity="precomputed", n_components=2,
     random_state=42) → 2D coordinates.
  4. Normalize coordinates to fit inside the unit square [-1, 1] in each
     axis, centered on the origin.

Writes to lib/emotion_geometry.json so the Next.js frontend can import it
at build time. Coordinates are scope-keyed; the frontend picks the right
scope per phase (reply uses reply coords, thought uses thought coords).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from sklearn.manifold import MDS

ROOT = Path("/home/johnk/Documents/understanding")
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise"]

SCOPES = {
    "reply": {
        "dir": ROOT / "data" / "thinking" / "vectors" / "reply" / "cutoff_50",
        "layer": 25,
    },
    "thought": {
        "dir": ROOT / "data" / "thinking" / "vectors" / "thought" / "cutoff_50",
        "layer": 13,
    },
}

OUT_PATH = ROOT / "lib" / "emotion_geometry.json"


def load_vectors(d: Path, layer: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for e in EMOTIONS:
        p = d / f"{e}_layer_{layer}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        out[e] = np.load(p).astype(np.float32)
    return out


def cosine_matrix(vecs: dict[str, np.ndarray]) -> np.ndarray:
    n = len(EMOTIONS)
    C = np.zeros((n, n), dtype=np.float64)
    for i, e1 in enumerate(EMOTIONS):
        for j, e2 in enumerate(EMOTIONS):
            v1, v2 = vecs[e1], vecs[e2]
            C[i, j] = float(np.dot(v1, v2) /
                            (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return C


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Center on origin, scale so the max |coord| is 1. Preserves shape."""
    centered = coords - coords.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(centered)))
    if max_abs < 1e-12:
        return centered
    return centered / max_abs


def main():
    out: dict = {"_meta": {
        "method": "sklearn.manifold.MDS (metric, SMACOF)",
        "n_components": 2,
        "random_state": 42,
        "dissimilarity": "1 - cosine_similarity",
        "coords_normalization": "centered on origin, scaled so max |coord| == 1",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    for scope, cfg in SCOPES.items():
        print(f"\n=== {scope} (layer {cfg['layer']}) ===")
        vecs = load_vectors(cfg["dir"], cfg["layer"])
        C = cosine_matrix(vecs)
        print("cosine matrix:")
        print("        " + "  ".join(f"{e[:5]:>7}" for e in EMOTIONS))
        for i, e in enumerate(EMOTIONS):
            print(f"  {e[:7]:<7}" + "  ".join(f"{C[i,j]:>7.3f}" for j in range(len(EMOTIONS))))

        D = 1.0 - C
        # Enforce symmetry & zero diagonal (numerical hygiene)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",   # silences sklearn warning
            n_init=10,
        )
        coords_raw = mds.fit_transform(D)
        coords = normalize_coords(coords_raw)
        stress = float(mds.stress_)

        print(f"MDS stress: {stress:.4f}")
        print("coords (normalized to [-1, 1]):")
        for i, e in enumerate(EMOTIONS):
            print(f"  {e:<9} x={coords[i, 0]:+.3f}  y={coords[i, 1]:+.3f}")

        out[scope] = {
            "layer": cfg["layer"],
            "stress": stress,
            "coords": {e: [float(coords[i, 0]), float(coords[i, 1])]
                       for i, e in enumerate(EMOTIONS)},
            "cosine_matrix": {e1: {e2: float(C[i, j])
                                   for j, e2 in enumerate(EMOTIONS)}
                              for i, e1 in enumerate(EMOTIONS)},
        }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
