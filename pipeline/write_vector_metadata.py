"""Write metadata.json to each vector directory documenting provenance.

Run after compute_vectors.py and percentile_calibrate.py so the saved
metadata reflects the actual files on disk. The backend reads
metadata.json at startup as a sanity check that it's loading the
vectors it expects.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/johnk/Documents/understanding")
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def npz_count(corpus_subdir: str, glob: str) -> int:
    """Count files; ROOT-relative path includes activations dir."""
    d = ROOT / "data" / corpus_subdir / "activations"
    if not d.exists():
        return 0
    return len(list(d.glob(glob)))


def write_for(set_name: str, scope_dir: Path, cutoff: int, denoised: bool,
              corpus_subdir: str, train_npz_glob: str,
              neutral_corpus_subdir: str | None,
              neutral_glob: str | None):
    """Write metadata.json into `scope_dir / cutoff_{N}[_raw]`."""
    suffix = "" if denoised else "_raw"
    cell = scope_dir / f"cutoff_{cutoff}{suffix}"
    if not cell.exists():
        return
    # Sanity-check vector files
    expected = [cell / f"{e}_layer_{L}.npy" for e in EMOTIONS for L in TARGET_LAYERS]
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]

    has_calibration = (cell / "calibration.json").exists()
    cal_info = None
    if has_calibration:
        cal = json.loads((cell / "calibration.json").read_text())
        cal_info = {
            "method": cal.get("_meta", {}).get("method"),
            "source": cal.get("_meta", {}).get("source"),
            "n_train_topics": cal.get("_meta", {}).get("n_train_topics"),
        }

    metadata = {
        "set": set_name,
        "scope_directory": str(scope_dir.relative_to(ROOT)),
        "cutoff": cutoff,
        "denoised": denoised,
        "neutral_pc_variance_target": 0.50 if denoised else None,
        "neutral_corpus": neutral_corpus_subdir if denoised else None,
        "neutral_pattern": neutral_glob if denoised else None,
        "neutral_pc_count_at_build_time": "see compute_vectors.py log output",
        "training_source": {
            "corpus": corpus_subdir,
            "file_pattern": train_npz_glob,
            "n_files_found_at_writeback": npz_count(corpus_subdir, train_npz_glob),
            "train_topics": "0-79 (pre-registered, see pipeline/topic_split.json)",
            "train_trials": "0-11 within each topic",
        },
        "calibration": cal_info,
        "missing_vector_files": missing,
        "git_commit_at_write_time": git_head(),
        "wrote_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "written_by": "pipeline/write_vector_metadata.py",
        "consumed_by": [
            "backend/inference.py (live demo)",
            "pipeline/validation.py (holdout classification)",
            "pipeline/build_emotion_geometry.py (MDS embedding)",
        ],
    }
    out = cell / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    # V1 — no_thinking_full
    for cutoff in (0, 25, 50):
        for denoised in (True, False):
            write_for(
                "V1_no_thinking_full",
                ROOT / "data" / "no_thinking" / "vectors" / "full",
                cutoff,
                denoised,
                corpus_subdir="no_thinking",
                train_npz_glob="*_story_*.npz",
                neutral_corpus_subdir="neutral_no_thinking",
                neutral_glob="*_story_*.npz",
            )

    # V2 — thinking_thought
    for cutoff in (0, 25, 50):
        for denoised in (True, False):
            write_for(
                "V2_thinking_thought",
                ROOT / "data" / "thinking" / "vectors" / "thought",
                cutoff,
                denoised,
                corpus_subdir="thinking",
                train_npz_glob="*_thought.npz",
                neutral_corpus_subdir="neutral_thinking",
                neutral_glob="*_thought.npz",
            )

    # V3 — thinking_reply
    for cutoff in (0, 25, 50):
        for denoised in (True, False):
            write_for(
                "V3_thinking_reply",
                ROOT / "data" / "thinking" / "vectors" / "reply",
                cutoff,
                denoised,
                corpus_subdir="thinking",
                train_npz_glob="*_story_*.npz",
                neutral_corpus_subdir="neutral_thinking",
                neutral_glob="*_story_*.npz",
            )


if __name__ == "__main__":
    main()
