#!/usr/bin/env bash
# Overnight contrastive corpus generation. Each phase is resumable, so
# Ctrl+C between batches is safe — re-running picks up.
#
# Generation only. Activation extraction is a separate step
# (extract_activations.py) you run after generation finishes.
#
# Pause anytime: `touch /tmp/understanding_pause_user`. The /monitor
# frontend has a button that does this. The pipeline checks between
# batches and sleeps until the flag is gone.
#
# Logs land in data/<corpus>/generate.log. The /monitor page tails them.

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

phase() {
    echo
    echo "================================================================"
    echo "[$(date +%Y-%m-%dT%H:%M:%S)] PHASE $1"
    echo "================================================================"
}

phase "GEN-1 — no_thinking corpus (train + holdout)"
python -m pipeline.generate_stories --corpus no_thinking

phase "GEN-2 — thinking corpus (train + holdout)"
python -m pipeline.generate_stories --corpus thinking

phase "GEN-3a — neutral_no_thinking dialogues (Sofroniew template, n=5/topic, 100 topics)"
python -m pipeline.generate_stories --corpus neutral_no_thinking

phase "GEN-3b — neutral_thinking dialogues (thinking on, n=5/topic, 100 topics)"
python -m pipeline.generate_stories --corpus neutral_thinking

# Extraction can run after each gen phase to start producing activations
# sooner — but it needs the GPU exclusive of generation. Easiest is to
# do all gen first, then all extract.

phase "EXT-1 — no_thinking per-token activations"
python -m pipeline.extract_activations --corpus no_thinking

phase "EXT-2 — thinking per-token activations (story + thought NPZs)"
python -m pipeline.extract_activations --corpus thinking

phase "EXT-3a — neutral_no_thinking activations (V1 PCA basis)"
python -m pipeline.extract_activations --corpus neutral_no_thinking

phase "EXT-3b — neutral_thinking activations (V2/V3 PCA bases — thought + reply)"
python -m pipeline.extract_activations --corpus neutral_thinking

echo
echo "[$(date +%Y-%m-%dT%H:%M:%S)] all generation + extraction done."
echo "next (CPU only, fast):"
echo "  python -m pipeline.compute_vectors"
echo "  python -m pipeline.validation --corpus no_thinking --all"
echo "  python -m pipeline.validation --corpus thinking --all"
