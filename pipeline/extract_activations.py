"""
extract_activations.py

Pass 2 of the contrastive pipeline: per-token residual-stream extraction
on the BATCHED-corpus JSON layout produced by generate_stories.py.

Per the user spec: each generated story (a single batch JSON contains
many) gets its own forward pass, capturing per-token activations at
TARGET_LAYERS in float16. Thinking corpus also gets a per-batch
forward pass on the thought block.

Output layout (one .npz per forward pass):
    data/<corpus>/activations/<batch_tag>_story_<NN>.npz
    data/<corpus>/activations/<batch_tag>_thought.npz   (thinking only)

Each .npz keys:
    layer_{L}     float16 (seq_len, hidden_dim)   for L in TARGET_LAYERS
    metadata_json string-encoded JSON with:
        kind                      = "story" | "thought"
        batch_tag                 = parent batch tag
        token_count               = seq_len
        emotion / topic_idx / split / story_idx (story only)
        contains_emotion_word     (story only) — pre-reg leak labels
        filter_pre_registration_commit
        model_id, model_revision_sha, bnb_quant_config

USAGE
    python -m pipeline.extract_activations --corpus no_thinking
    python -m pipeline.extract_activations --corpus thinking
    python -m pipeline.extract_activations --corpus neutral

RESUMABLE: skips items whose .npz already exists with all expected
layer keys. Honors Ctrl+C between items.

PAUSE / THERMAL / DISK: same cooperative pause flag scheme as
generate_stories.py. Aborts before writing if the all-corpora
activation total would cross --max-disk-gb (default 200).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from nnsight import LanguageModel


MODEL_ID = "google/gemma-4-E4B-it"
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]
ROOT_DATA_DIR = Path("data")

# Cooperative-pause file contract — shared with generate_stories.py.
USER_PAUSE_FLAG = Path("/tmp/understanding_pause_user")
THERMAL_PAUSE_FLAG = Path("/tmp/understanding_pause_thermal")
TEMP_PAUSE_C = 87
TEMP_RESUME_C = 80
TEMP_FAIL_C = 92


# ---------------------------------------------------------------------------
# Graceful interrupts
# ---------------------------------------------------------------------------

_interrupt_count = 0


def _signal_handler(signum, frame):
    global _interrupt_count
    _interrupt_count += 1
    if _interrupt_count == 1:
        print(
            "\n\n[interrupt — finishing current item then exiting cleanly. "
            "press Ctrl+C again to force-exit.]\n",
            flush=True,
        )
    else:
        print("\n[force-exit]\n", flush=True)
        sys.exit(1)


def install_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def interrupted() -> bool:
    return _interrupt_count > 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH: Optional[Path] = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None and LOG_PATH.parent.exists():
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


def section(title: str) -> None:
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


# ---------------------------------------------------------------------------
# Cooperative pause + thermal monitoring
# ---------------------------------------------------------------------------

def get_gpu_temp_c() -> Optional[int]:
    import shutil
    import subprocess
    nvsmi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits", "--id=0"],
            timeout=5,
        ).decode().strip()
        return int(out)
    except Exception:
        return None


def cooperative_pause_if_needed():
    paused_for: Optional[str] = None
    while True:
        user = USER_PAUSE_FLAG.exists()
        temp = get_gpu_temp_c()
        too_hot = temp is not None and temp >= TEMP_PAUSE_C
        if too_hot and not THERMAL_PAUSE_FLAG.exists():
            try:
                THERMAL_PAUSE_FLAG.write_text(f"temp={temp}")
            except Exception:
                pass
        cool_enough = temp is None or temp <= TEMP_RESUME_C
        if not user and (not too_hot or cool_enough):
            if THERMAL_PAUSE_FLAG.exists():
                try:
                    THERMAL_PAUSE_FLAG.unlink()
                except Exception:
                    pass
            if paused_for is not None:
                log(f"resuming (was paused for {paused_for})")
            break
        if temp is not None and temp >= TEMP_FAIL_C:
            log(f"!! GPU TEMP {temp}°C >= {TEMP_FAIL_C}°C — aborting.")
            sys.exit(1)
        new_reason = (
            "user pause" if user
            else f"thermal ({temp}°C)" if too_hot
            else "unknown"
        )
        if new_reason != paused_for:
            log(f"PAUSE: {new_reason}")
            paused_for = new_reason
        time.sleep(5)


# ---------------------------------------------------------------------------
# Model loading + nnsight wrap
# ---------------------------------------------------------------------------

def load_model():
    log(f"loading {MODEL_ID} (4-bit nf4)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
    )
    hf_model.eval()
    nn_model = LanguageModel(hf_model, tokenizer=tokenizer)
    log(f"  loaded in {time.time() - t0:.1f}s on {hf_model.device}")
    return tokenizer, nn_model


# ---------------------------------------------------------------------------
# Per-text extraction (one forward pass)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_per_token(
    tokenizer,
    nn_model: LanguageModel,
    text: str,
    device: torch.device,
) -> Optional[tuple[dict[int, np.ndarray], int]]:
    """Run forward over `text` and capture all 6 target-layer residuals
    per token. Returns ({layer: float16 (seq_len, hidden)}, seq_len) or
    None on tokenization/empty-text failure. Caller catches OOM."""
    if not text.strip():
        return None
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    if len(input_ids) < 2:
        return None

    captures: dict[int, torch.Tensor] = {}
    ids_2d = input_ids.unsqueeze(0)
    with nn_model.trace(ids_2d):
        for L in TARGET_LAYERS:
            captures[L] = nn_model.model.language_model.layers[L].output[0].save()

    out: dict[int, np.ndarray] = {}
    for L in TARGET_LAYERS:
        t = captures[L]
        if not isinstance(t, torch.Tensor):
            t = t.value
        # nnsight returns hidden states with batch already squeezed
        # (seq_len, hidden_dim). Defensive in case that ever changes.
        if t.ndim == 3:
            t = t[0]
        out[L] = t.to(torch.float16).cpu().numpy()
    return out, int(len(input_ids))


# ---------------------------------------------------------------------------
# Storage layout helpers
# ---------------------------------------------------------------------------

def story_npz_path(corpus_dir: Path, batch_tag: str, story_idx: int) -> Path:
    return corpus_dir / "activations" / f"{batch_tag}_story_{story_idx:02d}.npz"


def thought_npz_path(corpus_dir: Path, batch_tag: str) -> Path:
    return corpus_dir / "activations" / f"{batch_tag}_thought.npz"


def npz_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = np.load(path)
        return all(f"layer_{L}" in d for L in TARGET_LAYERS)
    except Exception:
        return False


def save_npz_atomic(
    path: Path,
    layer_arrays: dict[int, np.ndarray],
    metadata: dict,
):
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {f"layer_{L}": arr for L, arr in layer_arrays.items()}
    payload["metadata_json"] = np.array(json.dumps(metadata))
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def cleanup_stale_tmp_files(corpus_dir: Path):
    activations_dir = corpus_dir / "activations"
    if not activations_dir.exists():
        return
    n = 0
    for tmp in activations_dir.glob("*.tmp"):
        tmp.unlink()
        n += 1
    if n > 0:
        log(f"cleaned up {n} stale .tmp files")


def all_activations_disk_gb_root() -> float:
    if not ROOT_DATA_DIR.exists():
        return 0.0
    total = 0
    for sub in ROOT_DATA_DIR.iterdir():
        if not sub.is_dir():
            continue
        d = sub / "activations"
        if d.exists():
            total += sum(p.stat().st_size for p in d.glob("*.npz"))
    return total / 1e9


# ---------------------------------------------------------------------------
# Plan: walk batches/, expand to per-story / per-thought items
# ---------------------------------------------------------------------------

def plan_items(corpus_dir: Path) -> list[dict]:
    """Returns list of items to extract, each {kind, batch_tag, story_idx?, batch_payload_subset}."""
    batches_dir = corpus_dir / "batches"
    if not batches_dir.exists():
        return []
    items: list[dict] = []
    for bp in sorted(batches_dir.glob("*.json")):
        try:
            with open(bp) as f:
                batch = json.load(f)
        except Exception:
            continue
        if batch.get("parse_status") != "ok":
            continue
        tag = batch["tag"]

        # Per-story items (always present in valid batches).
        stories = batch.get("stories", [])
        leaks = batch.get("contains_emotion_word_per_story", [[]] * len(stories))
        for i, story_text in enumerate(stories):
            items.append({
                "kind": "story",
                "batch_tag": tag,
                "story_idx": i,
                "text": story_text,
                "metadata": {
                    "kind": "story",
                    "batch_tag": tag,
                    "story_idx": i,
                    "emotion": batch.get("emotion"),
                    "topic_idx": batch.get("topic_idx"),
                    "split": batch.get("split"),
                    "thinking": batch.get("thinking", False),
                    "seed": batch.get("seed"),
                    "model_id": batch.get("model_id"),
                    "model_revision_sha": batch.get("model_revision_sha"),
                    "bnb_quant_config": batch.get("bnb_quant_config"),
                    "filter_pre_registration_commit": batch.get(
                        "filter_pre_registration_commit"
                    ),
                    "contains_emotion_word": leaks[i] if i < len(leaks) else [],
                },
            })

        # Per-thought item (thinking corpus only).
        thought = batch.get("thought")
        if thought:
            items.append({
                "kind": "thought",
                "batch_tag": tag,
                "text": thought,
                "metadata": {
                    "kind": "thought",
                    "batch_tag": tag,
                    "emotion": batch.get("emotion"),
                    "topic_idx": batch.get("topic_idx"),
                    "split": batch.get("split"),
                    "thinking": True,
                    "seed": batch.get("seed"),
                    "model_id": batch.get("model_id"),
                    "model_revision_sha": batch.get("model_revision_sha"),
                    "bnb_quant_config": batch.get("bnb_quant_config"),
                    "filter_pre_registration_commit": batch.get(
                        "filter_pre_registration_commit"
                    ),
                    # Thoughts aren't constrained by the no-emotion-word
                    # rule (the model's reasoning often quotes the word
                    # explicitly), so we don't run the leak detector on them.
                },
            })

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-token residuals from a batched corpus.")
    p.add_argument(
        "--corpus",
        choices=("no_thinking", "thinking", "neutral"),
        required=True,
    )
    p.add_argument(
        "--max-disk-gb",
        type=float,
        default=200.0,
        help="Hard cap on total activation .npz size across all corpora.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N items (for smoke testing).",
    )
    return p.parse_args()


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def main():
    global LOG_PATH

    args = parse_args()
    corpus_dir = ROOT_DATA_DIR / args.corpus
    activations_dir = corpus_dir / "activations"

    if not (corpus_dir / "batches").exists():
        print(f"ERROR: {corpus_dir}/batches does not exist. Generate stories first.")
        sys.exit(1)

    activations_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = corpus_dir / "extract.log"
    install_signal_handlers()

    section(f"EXTRACT — corpus={args.corpus}")
    log(f"batches dir:        {corpus_dir / 'batches'}")
    log(f"activations dir:    {activations_dir}")
    log(f"target layers:      {TARGET_LAYERS}")
    log(f"max disk (all):     {args.max_disk_gb} GB")

    cleanup_stale_tmp_files(corpus_dir)

    full_plan = plan_items(corpus_dir)
    log(f"items in plan:      {len(full_plan)}")

    todo: list[dict] = []
    already_done = 0
    for it in full_plan:
        if it["kind"] == "story":
            p = story_npz_path(corpus_dir, it["batch_tag"], it["story_idx"])
        else:
            p = thought_npz_path(corpus_dir, it["batch_tag"])
        if npz_is_complete(p):
            already_done += 1
        else:
            todo.append({**it, "out_path": p})
    if args.limit is not None:
        todo = todo[: args.limit]

    log(f"already extracted:  {already_done}")
    log(f"to extract this run:{len(todo)}")
    if not todo:
        log("nothing to do.")
        return

    log(f"rough wall-clock estimate: {format_eta(len(todo) * 1.0)}")
    log(f"starting disk used (data/*/activations): {all_activations_disk_gb_root():.2f} GB")

    tokenizer, nn_model = load_model()
    device = next(nn_model.model.parameters()).device

    section("EXTRACTING")
    pipeline_start = time.time()
    completed = 0
    failed = 0
    skipped_short = 0
    times: list[float] = []
    last_summary = 0

    for i, it in enumerate(todo, start=1):
        if interrupted():
            log("\nexiting cleanly due to interrupt")
            break

        cooperative_pause_if_needed()
        item_start = time.time()

        if times:
            recent = times[-min(50, len(times)):]
            eta = format_eta((len(todo) - i + 1) * (sum(recent) / len(recent)))
            eta_str = f" ETA {eta}"
        else:
            eta_str = ""

        try:
            result = extract_per_token(tokenizer, nn_model, it["text"], device)
            if result is None:
                log(f"[{i}/{len(todo)}] {it['out_path'].name}: SKIP (empty/short text)")
                skipped_short += 1
                continue
            layer_arrays, seq_len = result

            # Pre-write disk-cap check.
            estimated_bytes = sum(arr.nbytes for arr in layer_arrays.values())
            current_gb = all_activations_disk_gb_root()
            projected_gb = current_gb + estimated_bytes / 1e9
            if projected_gb > args.max_disk_gb:
                log(
                    f"\nDISK CAP HIT: {projected_gb:.2f} GB would exceed "
                    f"{args.max_disk_gb:.2f} GB cap. Stopping."
                )
                break

            metadata = {**it["metadata"], "token_count": seq_len}
            save_npz_atomic(it["out_path"], layer_arrays, metadata)

            elapsed = time.time() - item_start
            times.append(elapsed)
            completed += 1
            log(
                f"[{i}/{len(todo)}] {it['out_path'].name} "
                f"({it['kind']}, toks={seq_len}) in {elapsed:.2f}s{eta_str}"
            )

            if completed - last_summary >= 250:
                last_summary = completed
                total_elapsed = time.time() - pipeline_start
                rate = completed / total_elapsed
                disk = all_activations_disk_gb_root()
                log(
                    f"  -- progress: {completed}/{len(todo)} done, "
                    f"{rate:.2f} items/s, disk {disk:.2f} GB / {args.max_disk_gb} cap, "
                    f"elapsed {format_eta(total_elapsed)} --"
                )

        except torch.cuda.OutOfMemoryError as e:
            log(f"[{i}/{len(todo)}] {it['out_path'].name}: OOM — {e}")
            failed += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            import traceback
            log(f"[{i}/{len(todo)}] {it['out_path'].name}: ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            failed += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    section("SUMMARY")
    total = time.time() - pipeline_start
    log(f"completed:          {completed}")
    log(f"failed:             {failed}")
    log(f"skipped (short):    {skipped_short}")
    log(f"wall clock:         {format_eta(total)}")
    if completed:
        log(f"avg per item:       {total/completed:.2f}s")
    log(f"final disk (all):   {all_activations_disk_gb_root():.2f} GB")


if __name__ == "__main__":
    main()
