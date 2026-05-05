"""
contrastive_pipeline.py

Generate emotional stories + extract residual stream activations for emotion vector pipeline.
Implements the methodology from Sofroniew et al. 2026 ("Emotion Concepts and their Function
in a Large Language Model"), adapted for Gemma 4 E4B.

CONFIG: 6 emotions x 100 topics x 12 stories/topic = 7200 stories total
        (paper used 171 emotions x same 100 topics x 12 stories = 205,200 stories)

RESUMABLE: skips (emotion, topic_idx, story_idx) combos that already have valid activation
files. Just rerun this script after Ctrl+C / crash / shutdown and it will pick up where it
left off.

GRACEFUL INTERRUPTS:
    - First Ctrl+C: finish current story (save its activations), then exit cleanly.
    - Second Ctrl+C: force-exit immediately (current story is lost but everything before it
      is safe).
    - SIGTERM (e.g. systemd shutdown): same as first Ctrl+C.

OUTPUT STRUCTURE:
    data/
        topics.json                              # the 100 topics, in order
        config.json                              # run configuration for reproducibility
        run.log                                  # progress log (appended across runs)
        stories/
            joy_topic_000_story_00.json          # story text + metadata
            ...
        activations/
            joy_topic_000_story_00.npz           # one (2560,) vector per layer
            ...

USAGE:
    python contrastive_pipeline.py

    # to monitor progress in another terminal while running:
    tail -f data/run.log
"""

from __future__ import annotations

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


# ============================================================================
# Configuration -- match the paper for full replication on the 6 Ekman emotions
# ============================================================================

MODEL_ID = "google/gemma-4-E4B-it"
DATA_DIR = Path("data")

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

NUM_TOPICS = 100             # paper: 100
STORIES_PER_TOPIC = 12       # paper: 12

# Layer selection (paper sec 2.2 + 2.3):
#   - early-middle (~30-40% depth): emotional connotations of present content
#   - middle-late  (~60-65% depth): emotions relevant to upcoming tokens
# Gemma 4 E4B has 42 layers. Paper uses ~2/3 depth (=layer 28) for main analyses.
TARGET_LAYERS = [13, 17, 21, 25, 28, 32]

MAX_NEW_TOKENS = 280         # ~one paragraph
MIN_TOKEN_POSITION = 50      # paper: avg activations from this position onward
MIN_STORY_TOKENS = 80        # need >50 + buffer for the average to be meaningful

TEMPERATURE = 0.9            # diversity in generations
TOP_P = 0.95


# Anthropic's exact 100 topics, lifted from the paper appendix
TOPICS: list[str] = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they're converting to a different religion",
    "Someone's childhood imaginary friend appears in their niece's drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother's engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler's flight is delayed, causing them to miss an important event",
    "A student is accused of plagiarism",
    "A person discovers their mentor has retired without saying goodbye",
    "Two friends both apply for the same job",
    "A person runs into their ex at a mutual friend's wedding",
    "Someone discovers their friend has been lying about their job",
    "A person discovers their partner has been taking secret phone calls",
    "A person discovers their child has the same teacher they had",
    "A person's car is towed from their own driveway",
    "Two friends realize they remember a shared event completely differently",
    "Someone discovers their mother kept every school assignment",
    "A person discovers their teenage diary has been published online",
    "Someone finds out their medical records were mixed up with another patient's",
    "A person finds out their article was published under someone else's name",
    "An athlete doesn't make the team they expected to join",
    "An employee is transferred to a different department",
    "Someone receives a friend request from a childhood bully",
    "A person finds out their surprise party has been cancelled",
    "An employee finds out a junior colleague makes more money",
    "A person finds out their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "A person learns their favorite restaurant is closing",
    "Someone finds their childhood teddy bear at a yard sale",
    "A homeowner discovers previous residents left items in the attic",
    "Someone finds an unsigned birthday card in their mailbox",
    "Someone discovers a hidden room in their new house",
    "Two strangers realize they've been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother's house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone's childhood home is about to be demolished",
    "A person's invention is already patented by someone else",
    "A neighbor's dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn't remember them",
    "A musician hears their song being performed by someone else",
    "A person's manuscript is rejected by their dream publisher",
    "A person finds old photos that contradict family stories",
    "A person is asked to give a speech at their parent's retirement party",
    "A student discovers their teacher follows them on social media",
    "A parent finds an old letter they wrote but never sent",
    "An employee discovers the company is being sold",
    "A person accidentally sends a text to the wrong recipient",
    "Two coworkers are stuck in an elevator for three hours",
    "A student learns their thesis advisor is leaving the university",
    "A person's longtime hobby becomes their child's obsession",
    "Two colleagues are both considered for the same promotion",
    "Two coworkers discover they went to the same summer camp",
    "A tenant receives an eviction notice",
    "Someone finds their parent's draft letter of resignation from decades ago",
    "Someone finds out their best friend is moving across the country",
    "A neighbor's tree falls on their property",
    "Someone receives an apology letter years after the incident",
    "A person discovers the tree they planted as a child has been cut down",
    "Two siblings discover different versions of their inheritance",
    "A person finds their childhood home listed for sale online",
    "A homeowner learns their house was a former crime scene",
    "Someone finds out they have a half-sibling they never knew about",
    "A person learns their childhood bully became a therapist",
    "Two people discover they've been working on identical projects",
    "A person finds their spouse's secret savings account",
    "A neighbor complains about noise levels",
    "Someone finds their deceased parent's bucket list",
    "A teacher receives an unexpected gift from a former student",
    "An artist's work is displayed without their permission",
    "Someone discovers their neighbor is secretly wealthy",
    "A student receives a much lower grade than expected",
    "A person learns their college is closing down",
    "A neighbor asks to cut down a tree on the property line",
    "Two strangers discover they share the same rare medical condition",
    "Someone receives flowers with no card attached",
    "Someone discovers their partner has been writing a novel about them",
    "Someone finds a time capsule they don't remember burying",
    "Someone finds their partner's bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]
assert len(TOPICS) == 100, f"expected 100 topics, got {len(TOPICS)}"


STORY_PROMPT_TEMPLATE = """Write 1 short story based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.

The paragraph should be a fresh start. Use either third-person or first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# ============================================================================
# Graceful interrupt handling
# ============================================================================

# Set by signal handler. Loop checks this at safe points.
_interrupt_count = 0


def _signal_handler(signum, frame):
    global _interrupt_count
    _interrupt_count += 1
    if _interrupt_count == 1:
        print(
            "\n\n[interrupt received -- finishing current story then exiting cleanly. "
            "press Ctrl+C again to force-exit immediately]\n",
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


# ============================================================================
# Logging
# ============================================================================

LOG_PATH = DATA_DIR / "run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH.parent.exists():
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


def section(title: str) -> None:
    bar = "=" * 70
    log("")
    log(bar)
    log(title)
    log(bar)


# ============================================================================
# Model loading
# ============================================================================

def load_model():
    log(f"loading {MODEL_ID} (4-bit nf4)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s on {model.device}")
    return tokenizer, model


def get_layer_modules(model):
    """Verified path for Gemma 4 E4B with the current transformers version."""
    return model.model.language_model.layers


# ============================================================================
# Hook utilities
# ============================================================================

def attach_hooks(layers, captured: dict, target_layers: list[int]):
    handles = []
    for L in target_layers:
        def make_hook(idx):
            def hook(module, inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                captured[idx] = hidden.detach().cpu()
            return hook
        handles.append(layers[L].register_forward_hook(make_hook(L)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ============================================================================
# Generation + extraction
# ============================================================================

def generate_story(tokenizer, model, emotion: str, topic: str) -> tuple[str, int, float]:
    prompt = STORY_PROMPT_TEMPLATE.format(topic=topic, emotion=emotion)
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    gen_time = time.time() - t0
    story_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    gen_tokens = int(out[0].shape[0] - prompt_len)
    return story_text, gen_tokens, gen_time


def extract_activations(
    tokenizer, model, layers, story_text: str
) -> Optional[dict[str, np.ndarray]]:
    """Forward-pass on the story alone, average activations over positions >= MIN_TOKEN_POSITION."""
    inputs = tokenizer(story_text, return_tensors="pt").to(model.device)
    seq_len = int(inputs["input_ids"].shape[-1])
    if seq_len < MIN_STORY_TOKENS:
        return None

    captured: dict[int, torch.Tensor] = {}
    handles = attach_hooks(layers, captured, TARGET_LAYERS)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        remove_hooks(handles)

    out = {}
    for L in TARGET_LAYERS:
        if L not in captured:
            return None
        hidden = captured[L][0]
        avg = hidden[MIN_TOKEN_POSITION:].float().mean(dim=0).numpy()
        out[f"layer_{L}"] = avg
    return out


# ============================================================================
# Storage (atomic writes, resumable checking)
# ============================================================================

def file_tag(emotion: str, topic_idx: int, story_idx: int) -> str:
    return f"{emotion}_topic_{topic_idx:03d}_story_{story_idx:02d}"


def story_path(emotion: str, topic_idx: int, story_idx: int) -> Path:
    return DATA_DIR / "stories" / f"{file_tag(emotion, topic_idx, story_idx)}.json"


def activation_path(emotion: str, topic_idx: int, story_idx: int) -> Path:
    return DATA_DIR / "activations" / f"{file_tag(emotion, topic_idx, story_idx)}.npz"


def is_done(emotion: str, topic_idx: int, story_idx: int) -> bool:
    """A combo is done if the activation file exists AND loads cleanly with all expected layers."""
    p = activation_path(emotion, topic_idx, story_idx)
    if not p.exists():
        return False
    try:
        d = np.load(p)
        return all(f"layer_{L}" in d for L in TARGET_LAYERS)
    except Exception:
        return False


def save_story(emotion: str, topic_idx: int, story_idx: int, topic: str,
               story: str, gen_tokens: int) -> None:
    p = story_path(emotion, topic_idx, story_idx)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({
            "emotion": emotion,
            "topic": topic,
            "topic_idx": topic_idx,
            "story_idx": story_idx,
            "story": story,
            "gen_tokens": gen_tokens,
        }, f, indent=2)
    os.replace(tmp, p)


def save_activations(emotion: str, topic_idx: int, story_idx: int,
                     activations: dict[str, np.ndarray]) -> None:
    p = activation_path(emotion, topic_idx, story_idx)
    tmp = p.with_suffix(p.suffix + ".tmp")
    # np.savez auto-appends .npz to string paths, which breaks the rename.
    # Pass a file handle to bypass that behavior.
    with open(tmp, "wb") as f:
        np.savez(f, **activations)
    os.replace(tmp, p)


def cleanup_stale_tmp_files():
    """Remove any .tmp files left over from a previous interrupted run."""
    removed = 0
    for d in [DATA_DIR / "stories", DATA_DIR / "activations"]:
        if not d.exists():
            continue
        for tmp in d.glob("*.tmp"):
            tmp.unlink()
            removed += 1
    if removed > 0:
        log(f"cleaned up {removed} stale .tmp files from previous run")


# ============================================================================
# Main pipeline
# ============================================================================

def write_config():
    cfg = {
        "model_id": MODEL_ID,
        "emotions": EMOTIONS,
        "num_topics": NUM_TOPICS,
        "stories_per_topic": STORIES_PER_TOPIC,
        "target_layers": TARGET_LAYERS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_token_position": MIN_TOKEN_POSITION,
        "min_story_tokens": MIN_STORY_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    with open(DATA_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def main():
    install_signal_handlers()
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "stories").mkdir(exist_ok=True)
    (DATA_DIR / "activations").mkdir(exist_ok=True)
    write_config()
    with open(DATA_DIR / "topics.json", "w") as f:
        json.dump(TOPICS, f, indent=2)

    section("CONTRASTIVE PIPELINE")
    log(f"emotions:           {EMOTIONS}")
    log(f"num topics:         {NUM_TOPICS} (of 100 available)")
    log(f"stories per topic:  {STORIES_PER_TOPIC}")
    log(f"target layers:      {TARGET_LAYERS}")
    log(f"")
    log(f"press Ctrl+C anytime; current story will finish then exit cleanly")
    log(f"  (rerun this script to resume; already-saved stories will be skipped)")

    cleanup_stale_tmp_files()

    # plan: which (emotion, topic_idx, story_idx) combos still need to be done
    # Loop order: trial -> topic -> emotion (outer to inner). This means:
    #   - first 600 stories: 1 trial of every (emotion, topic) combo
    #   - next 600: a 2nd trial of every combo
    #   - etc.
    # If interrupted partway, the dataset stays balanced across emotions instead of
    # being heavily biased toward whichever emotion was iterated first.
    todo = []
    already_done = 0
    for s_idx in range(STORIES_PER_TOPIC):
        for t_idx in range(NUM_TOPICS):
            for emotion in EMOTIONS:
                if is_done(emotion, t_idx, s_idx):
                    already_done += 1
                else:
                    todo.append((emotion, t_idx, s_idx))

    total_target = len(EMOTIONS) * NUM_TOPICS * STORIES_PER_TOPIC
    log(f"")
    log(f"total target:       {total_target} stories")
    log(f"already done:       {already_done}")
    log(f"to generate:        {len(todo)}")

    if not todo:
        log("nothing to do! pipeline complete.")
        return

    # rough wall-clock estimate based on smoke test (~7s/story)
    rough_eta = format_eta(len(todo) * 7)
    log(f"rough wall-clock estimate: {rough_eta}")

    tokenizer, model = load_model()
    layers = get_layer_modules(model)

    section("GENERATING + EXTRACTING")
    pipeline_start = time.time()
    completed_this_run = 0
    failed = 0
    times: list[float] = []
    last_summary_at = 0  # index of last time we printed a summary

    for i, (emotion, t_idx, s_idx) in enumerate(todo, start=1):
        if interrupted():
            log("\nexiting cleanly due to interrupt request")
            break

        topic = TOPICS[t_idx]
        story_start = time.time()

        # adaptive ETA based on running average of recent stories
        if times:
            recent = times[-min(20, len(times)):]
            avg_time = sum(recent) / len(recent)
            remaining_secs = (len(todo) - i + 1) * avg_time
            eta_str = f" ETA {format_eta(remaining_secs)}"
        else:
            eta_str = ""

        try:
            story, gen_tokens, gen_time = generate_story(tokenizer, model, emotion, topic)

            t_ext = time.time()
            acts = extract_activations(tokenizer, model, layers, story)
            ext_time = time.time() - t_ext

            if acts is None:
                log(f"[{i}/{len(todo)}] {emotion} t{t_idx:03d} s{s_idx:02d}: "
                    f"SKIP (story too short, {gen_tokens} tokens)")
                failed += 1
                continue

            save_story(emotion, t_idx, s_idx, topic, story, gen_tokens)
            save_activations(emotion, t_idx, s_idx, acts)

            elapsed = time.time() - story_start
            times.append(elapsed)
            completed_this_run += 1

            log(f"[{i}/{len(todo)}] {emotion:<8s} t{t_idx:03d} s{s_idx:02d} "
                f"{gen_tokens:>3d}tok in {gen_time:.1f}s, "
                f"L21 norm={np.linalg.norm(acts['layer_21']):.1f}{eta_str}")

            # periodic summary every 50 stories
            if completed_this_run - last_summary_at >= 50:
                last_summary_at = completed_this_run
                total_elapsed = time.time() - pipeline_start
                rate = completed_this_run / total_elapsed
                log(f"  -- progress: {completed_this_run}/{len(todo)} done this run, "
                    f"{rate:.1f} stories/s, "
                    f"elapsed {format_eta(total_elapsed)} --")

        except Exception as e:
            log(f"[{i}/{len(todo)}] {emotion} t{t_idx:03d} s{s_idx:02d}: "
                f"ERROR {type(e).__name__}: {e}")
            failed += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    section("SUMMARY")
    total_time = time.time() - pipeline_start
    log(f"completed this run: {completed_this_run}")
    log(f"failures this run:  {failed}")
    log(f"total wall-clock:   {format_eta(total_time)}")
    if completed_this_run > 0:
        log(f"avg per story:      {total_time/completed_this_run:.1f}s")

    # final state check across the entire dataset
    remaining = sum(
        1 for emotion in EMOTIONS
        for t_idx in range(NUM_TOPICS)
        for s_idx in range(STORIES_PER_TOPIC)
        if not is_done(emotion, t_idx, s_idx)
    )
    log("")
    log(f"final state: {total_target - remaining}/{total_target} stories complete")
    if remaining == 0:
        log("ALL DONE. next: run compute_vectors.py to compute emotion vectors.")
    else:
        log(f"{remaining} still remaining. rerun this script to continue.")


if __name__ == "__main__":
    main()