"""
generate_stories.py

Pass 1 of the contrastive pipeline: generate emotional stories with
Gemma 4 E4B. BATCHED generation — one call produces n stories at once,
formatted with [story 1] / [story 2] / ... markers — replacing the old
1-call-per-story approach.

CONFIG
    6 emotions × 100 topics × 2 batches/topic = 1200 batches total
        train batch:    n_stories = 12 (trial indices 0..11)
        holdout batch:  n_stories =  3 (trial indices 12..14)
    yields 6 × 100 × 15 = 9000 stories per corpus.

THREE CORPORA
    --corpus no_thinking  (thinking off, vectors set 1: full text)
    --corpus thinking     (thinking on,  vectors sets 2+3: thought-only,
                           reply-only — split by <channel|> token)
    --corpus neutral      (50 emotionally-flat prompts, n=1 each, for the
                           projection-out denoising basis)

NO TOKEN CAP — generation runs until the model emits <eos>/<turn|> or
hits the model's context limit. If a batch OOMs or fails to parse, we
skip it and continue. CUDA cache is cleared between batches.

THOUGHT/REPLY SPLIT (thinking corpus only)
    The split is done at the TOKEN-ID level — we look for token id 101
    (<channel|>), the model's own channel-close marker. If a batch
    doesn't emit it, or emits it more than once, we mark the batch as
    failed and skip. No text-matching, no fallback to "guess from
    context". The boundary is unambiguous or it's not used.

STORAGE LAYOUT
    data/<corpus>/batches/<tag>.json          one per batch
        Contains: emotion, topic, split, n_stories, seed, full metadata,
        rendered_prompt, raw_output, parsed stories[], thought (or null),
        and a `parse_status` field (ok / fail_count / fail_no_channel /
        fail_two_channels) so failures are auditable post-hoc.
    The activation extractor reads batches[].stories and runs separate
    forward passes per individual story, plus one extra pass on the
    thought when thinking is enabled.

USAGE
    python -m pipeline.generate_stories --corpus no_thinking
    python -m pipeline.generate_stories --corpus thinking
    python -m pipeline.generate_stories --corpus neutral

RESUMABLE: skips (emotion, topic, split) batches that already have a
valid JSON. Honors Ctrl+C between batches.

PAUSE COOPERATION: between batches, checks /tmp/understanding_pause_user
(monitor frontend) and GPU temperature. Same contract as
extract_activations.py.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "google/gemma-4-E4B-it"
ROOT_DATA_DIR = Path("data")

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

NUM_TOPICS = 100
N_TRAIN = 12
N_HOLDOUT = 3

TARGET_LAYERS = [13, 17, 21, 25, 28, 32]  # for downstream awareness only

# Generation defaults. Temperature/top_p match the original pipeline; we
# leave max_new_tokens unset so the model stops at <eos>/<turn|> or the
# context boundary on its own. If a batch overruns and OOMs, we catch
# and skip it (clearing CUDA cache) — the user explicitly preferred this
# over an arbitrary cap that might truncate well-formed long generations.
TEMPERATURE = 0.9
TOP_P = 0.95
GEN_MAX_NEW_TOKENS_FALLBACK = 16384  # safety only — won't normally apply


# Pause flag files — same contract as extract_activations.py.
USER_PAUSE_FLAG = Path("/tmp/understanding_pause_user")
THERMAL_PAUSE_FLAG = Path("/tmp/understanding_pause_thermal")
TEMP_PAUSE_C = 87
TEMP_RESUME_C = 80
TEMP_FAIL_C = 92

# Pre-registered synonym filter (locked before corpus generation, commit
# 6cf25bf — see filter_synonyms.json). Each story gets a per-story
# `contains_emotion_word` list; stories that leak prohibited words are
# kept (research can filter at vector-compute time) but auditably tagged.
FILTER_SYNONYMS_PATH = Path(__file__).parent / "filter_synonyms.json"


# Anthropic's exact 100 topics, lifted from the paper appendix.
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
assert len(TOPICS) == NUM_TOPICS


# Batched prompt template, per the user's spec (May 2026). The {n_stories}
# placeholder is set at call time. The "[story 1]" etc. markers tell the
# model exactly how to delimit individual stories so we can reliably
# parse them post-hoc.
STORY_PROMPT_TEMPLATE = """Write {n_stories} different stories based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.

Format the stories like so:
[story 1]
[story 2]
[story 3]
etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# Neutral prompts for the projection-out denoising set.
NEUTRAL_PROMPTS: list[str] = [
    "Describe how to brew a pot of black tea in detail.",
    "Explain how a printer works step by step.",
    "Write a short paragraph describing a typical Tuesday morning at an office.",
    "Describe the process of changing a tire on a car.",
    "Write about how a library is organized.",
    "Explain the rules of basic chess play to a beginner.",
    "Describe what a typical kitchen looks like.",
    "Write a paragraph about how bread rises.",
    "Describe a route from a bus stop to a grocery store.",
    "Explain how to assemble a flat-pack bookshelf.",
    "Write about the layout of a small public park.",
    "Describe the steps to cook plain rice on a stovetop.",
    "Explain how a calendar app on a phone works.",
    "Write a paragraph about reading a printed map.",
    "Describe what is on a typical hardware-store shelf.",
    "Explain how to fold a fitted sheet.",
    "Describe the process of replacing a lightbulb.",
    "Write about the weather over a typical week in autumn in a temperate climate.",
    "Explain how to address a postal envelope correctly.",
    "Describe the layout of a pharmacy aisle.",
    "Write a paragraph about how dishwashers operate.",
    "Explain the function of a household thermostat.",
    "Describe what an empty airport terminal looks like at night.",
    "Write about the contents of a standard first-aid kit.",
    "Explain how shoelaces tie into a bow.",
    "Describe the process of taking the bus to work.",
    "Write a paragraph about how laundry is folded.",
    "Explain how to set the time on a microwave.",
    "Describe a wall of mailboxes in an apartment building.",
    "Write about the steps to make a peanut butter sandwich.",
    "Explain the function of a stoplight at an intersection.",
    "Describe the layout of a small office supply room.",
    "Write a paragraph about brushing teeth before bed.",
    "Explain how to change the batteries in a remote control.",
    "Describe a typical morning newspaper's front page sections.",
    "Write about the process of doing the dishes by hand.",
    "Explain how to wash a load of dark clothes.",
    "Describe the appearance of a public library reading room.",
    "Write a paragraph about how a coffee machine drips coffee.",
    "Explain how to organize a sock drawer.",
    "Describe a sidewalk in a residential neighborhood at noon.",
    "Write about how a ballpoint pen functions.",
    "Explain how to set up a basic spreadsheet.",
    "Describe the steps for sweeping a floor.",
    "Write a paragraph about how an elevator works.",
    "Explain how to fold a paper airplane.",
    "Describe the steps to mail a package at the post office.",
    "Write about the contents of a standard grocery cart.",
    "Explain how to take out the trash.",
    "Describe what happens at a bank teller's window during a deposit.",
]
assert len(NEUTRAL_PROMPTS) == 50


# ============================================================================
# Logging + interrupts
# ============================================================================

LOG_PATH: Optional[Path] = None
_interrupt_count = 0


def _signal_handler(signum, frame):
    global _interrupt_count
    _interrupt_count += 1
    if _interrupt_count == 1:
        print(
            "\n\n[interrupt — finishing current batch, exiting cleanly. "
            "press Ctrl+C again to force-exit.]\n", flush=True,
        )
    else:
        print("\n[force-exit]\n", flush=True)
        sys.exit(1)


def install_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def interrupted() -> bool:
    return _interrupt_count > 0


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


# ============================================================================
# Cooperative pause
# ============================================================================

def get_gpu_temp_c() -> Optional[int]:
    import shutil
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


# ============================================================================
# Reproducibility
# ============================================================================

def deterministic_seed(salt: str, *parts) -> int:
    h = hashlib.sha256(("|".join([salt, *map(str, parts)])).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def resolve_model_revision_sha() -> Optional[str]:
    home = Path(os.path.expanduser("~"))
    p = (home / ".cache" / "huggingface" / "hub"
         / "models--google--gemma-4-E4B-it" / "refs" / "main")
    if p.exists():
        try:
            return p.read_text().strip()
        except Exception:
            return None
    return None


def quant_config_dump(bnb: BitsAndBytesConfig) -> dict:
    return {
        "load_in_4bit": getattr(bnb, "load_in_4bit", None),
        "load_in_8bit": getattr(bnb, "load_in_8bit", None),
        "bnb_4bit_compute_dtype": str(getattr(bnb, "bnb_4bit_compute_dtype", None)),
        "bnb_4bit_quant_type": getattr(bnb, "bnb_4bit_quant_type", None),
        "bnb_4bit_use_double_quant": getattr(bnb, "bnb_4bit_use_double_quant", None),
    }


# ============================================================================
# Stop / channel token resolution
# ============================================================================

def resolve_stop_token_ids(tokenizer) -> list[int]:
    stops: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for name in ("<turn|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stops.add(tid)
    return sorted(stops)


def resolve_channel_token_ids(tokenizer) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, name in (("open", "<|channel>"), ("close", "<channel|>")):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            out[key] = tid
    label_ids = tokenizer("thought\n", add_special_tokens=False).input_ids
    out["label_count"] = len(label_ids)
    return out


# ============================================================================
# Batched output parser
# ============================================================================

# Patterns the model is most likely to use for delimiting individual
# stories within a batched output. Tried in order; whichever yields
# exactly n_expected non-empty parts wins.
_PARSE_PATTERNS = [
    # [story 1] / [Story 1] (the format the prompt explicitly asks for)
    re.compile(r"\[\s*[Ss]tory\s+\d+\s*\]\s*", re.MULTILINE),
    # **Story 1** / **Story 1:** markdown bold
    re.compile(r"\*\*\s*[Ss]tory\s+\d+[:\.]?\s*\*\*\s*", re.MULTILINE),
    # ## Story 1 / ### Story 1 markdown headers
    re.compile(r"(?:^|\n)#+\s*[Ss]tory\s+\d+[:\.]?\s*\n", re.MULTILINE),
    # Story 1: / Story 1.
    re.compile(r"(?:^|\n)\s*[Ss]tory\s+\d+[:\.]\s*\n", re.MULTILINE),
    # 1. / 1) numbered list at start of line
    re.compile(r"(?:^|\n)\s*\d+[\.\)]\s+", re.MULTILINE),
]


_FILTER_CACHE: Optional[dict[str, list[str]]] = None
_FILTER_REGEX_CACHE: Optional[dict[str, re.Pattern]] = None


def load_filter_synonyms() -> dict[str, list[str]]:
    """Read the pre-registered synonym filter once. The list is locked
    by the commit in filter_synonyms.json — never mutate at runtime."""
    global _FILTER_CACHE
    if _FILTER_CACHE is None:
        with open(FILTER_SYNONYMS_PATH) as f:
            _FILTER_CACHE = json.load(f)
    return _FILTER_CACHE


def filter_regex_for(emotion: str) -> re.Pattern:
    """Compiled \\bword\\b alternation for fast per-story leak checks."""
    global _FILTER_REGEX_CACHE
    if _FILTER_REGEX_CACHE is None:
        _FILTER_REGEX_CACHE = {}
    if emotion not in _FILTER_REGEX_CACHE:
        words = load_filter_synonyms().get(emotion, [])
        if not words:
            _FILTER_REGEX_CACHE[emotion] = re.compile(r"(?!.*)")  # match nothing
        else:
            # Sort longest-first so multi-word phrases (e.g. "taken aback")
            # win over their substrings if any. \b is at word boundaries.
            alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
            _FILTER_REGEX_CACHE[emotion] = re.compile(
                rf"\b(?:{alt})\b", re.IGNORECASE,
            )
    return _FILTER_REGEX_CACHE[emotion]


def detect_emotion_word_leaks(story: str, emotion: str) -> list[str]:
    """Returns the deduplicated list of prohibited words that appear in
    this story (case-insensitive, lowercase form). Empty list = clean."""
    if emotion not in load_filter_synonyms():
        return []
    pat = filter_regex_for(emotion)
    found = sorted({m.group(0).lower() for m in pat.finditer(story)})
    return found


def parse_batched_stories(text: str, n_expected: int) -> Optional[list[str]]:
    """Returns the n_expected stories parsed from a batched output, or
    None if no pattern produced the right count. None means "skip this
    batch" — we don't try to salvage partial / off-by-one parses."""
    text = text.strip()
    for pat in _PARSE_PATTERNS:
        parts = pat.split(text)
        # parts[0] is anything before the first marker (preamble or empty).
        candidates = [p.strip() for p in parts[1:] if p.strip()]
        if len(candidates) == n_expected:
            return candidates
    return None


# ============================================================================
# Channel-block splitting (thinking corpus)
# ============================================================================

def split_thinking_token_sequence(
    gen_ids: list[int],
    channel_open_id: int,
    channel_close_id: int,
    label_count: int,
) -> Optional[tuple[list[int], list[int]]]:
    """Returns (thought_ids, reply_ids) or None if the boundary is not
    unambiguous. Three failure modes treated as None:
      - <channel|> appears 0 times (no boundary)
      - <channel|> appears >1 time (ambiguous)
      - the chunk before <channel|> is too short to contain <|channel>
        + the literal "thought\\n" label

    No fallback to text-based splitting. Either we have a clean
    token-id boundary or the batch is rejected."""
    n_close = gen_ids.count(channel_close_id)
    if n_close != 1:
        return None
    close_idx = gen_ids.index(channel_close_id)
    prefix = gen_ids[:close_idx]
    if len(prefix) < label_count + 1:
        return None
    if prefix[0] != channel_open_id:
        return None
    thought_ids = prefix[1 + label_count:]  # drop <|channel> + "thought\n"
    reply_ids = gen_ids[close_idx + 1:]
    return thought_ids, reply_ids


# ============================================================================
# Generation
# ============================================================================

def load_model():
    log(f"loading {MODEL_ID} (4-bit nf4)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s on {model.device}")
    return tokenizer, model, bnb


def render_prompt(tokenizer, user_msg: str, thinking: bool) -> str:
    msgs = [{"role": "user", "content": user_msg}]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(msgs, **kwargs)


def generate_one_batch(
    tokenizer,
    model,
    rendered_prompt: str,
    seed: int,
    stop_ids: list[int],
) -> tuple[list[int], int, float]:
    """Returns (gen_ids_trimmed, raw_count, gen_seconds). gen_ids_trimmed
    has trailing stop tokens stripped so the parser doesn't have to deal
    with them. Raises RuntimeError on OOM (caller catches + skips)."""
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS_FALLBACK,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=stop_ids,
        )
    gen_secs = time.time() - t0

    gen_ids = out[0][prompt_len:].tolist()
    raw_count = len(gen_ids)
    while gen_ids and gen_ids[-1] in stop_ids:
        gen_ids.pop()
    return gen_ids, raw_count, gen_secs


# ============================================================================
# Storage
# ============================================================================

def batch_tag(emotion: str, topic_idx: int, split: str) -> str:
    return f"{emotion}_topic_{topic_idx:03d}_split_{split}"


def neutral_tag(prompt_idx: int) -> str:
    return f"neutral_topic_{prompt_idx:03d}_split_train"


def batch_path(corpus_dir: Path, tag: str) -> Path:
    return corpus_dir / "batches" / f"{tag}.json"


def is_done(corpus_dir: Path, tag: str) -> bool:
    p = batch_path(corpus_dir, tag)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("parse_status") == "ok"
    except Exception:
        return False


def save_batch(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def cleanup_stale_tmp_files(corpus_dir: Path):
    for sub in ("batches",):
        d = corpus_dir / sub
        if not d.exists():
            continue
        n = 0
        for tmp in d.glob("*.tmp"):
            tmp.unlink()
            n += 1
        if n:
            log(f"cleaned up {n} stale .tmp files in {sub}/")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Generate batched emotion-stories corpus.")
    p.add_argument(
        "--corpus",
        choices=("no_thinking", "thinking", "neutral"),
        required=True,
    )
    p.add_argument(
        "--splits",
        choices=("train", "holdout", "both"),
        default="both",
        help="For emotion corpora: which splits to generate. Default both.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N batches this run (for smoke testing).",
    )
    return p.parse_args()


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def plan_emotion_batches(splits_want: str) -> list[dict]:
    """Trial→topic→emotion ordering preserves balanced resumability."""
    plan: list[dict] = []
    pairs = []
    if splits_want in ("train", "both"):
        pairs.append(("train", N_TRAIN))
    if splits_want in ("holdout", "both"):
        pairs.append(("holdout", N_HOLDOUT))
    for split, n_stories in pairs:
        for t_idx in range(NUM_TOPICS):
            for emotion in EMOTIONS:
                plan.append({
                    "kind": "emotion",
                    "emotion": emotion,
                    "topic_idx": t_idx,
                    "split": split,
                    "n_stories": n_stories,
                    "tag": batch_tag(emotion, t_idx, split),
                })
    return plan


def plan_neutral_batches() -> list[dict]:
    return [
        {
            "kind": "neutral",
            "topic_idx": p_idx,
            "split": "train",
            "n_stories": 1,
            "prompt": NEUTRAL_PROMPTS[p_idx],
            "tag": neutral_tag(p_idx),
        }
        for p_idx in range(len(NEUTRAL_PROMPTS))
    ]


def main():
    global LOG_PATH
    args = parse_args()
    install_signal_handlers()

    corpus = args.corpus
    corpus_dir = ROOT_DATA_DIR / corpus
    (corpus_dir / "batches").mkdir(parents=True, exist_ok=True)
    LOG_PATH = corpus_dir / "generate.log"
    cleanup_stale_tmp_files(corpus_dir)

    thinking = (corpus == "thinking")

    if corpus == "neutral":
        full_plan = plan_neutral_batches()
    else:
        full_plan = plan_emotion_batches(args.splits)

    section(f"GENERATE — corpus={corpus} thinking={thinking}")
    log(f"corpus dir:         {corpus_dir}")
    log(f"target layers:      {TARGET_LAYERS}  (informational; extraction does the work)")
    if corpus != "neutral":
        log(f"split rule:         train=trials 0..{N_TRAIN-1}, holdout={N_TRAIN}..{N_TRAIN+N_HOLDOUT-1}")
        log(f"requested splits:   {args.splits}")

    todo = [b for b in full_plan if not is_done(corpus_dir, b["tag"])]
    if args.limit is not None:
        todo = todo[: args.limit]
    already_done = len(full_plan) - len([b for b in full_plan if not is_done(corpus_dir, b["tag"])])
    log(f"total in plan:      {len(full_plan)}")
    log(f"already done:       {already_done}")
    log(f"to generate:        {len(todo)}")
    if not todo:
        log("nothing to do.")
        return

    rough_per = 50 if thinking else 30
    log(f"rough wall-clock:   {format_eta(len(todo) * rough_per)}")

    tokenizer, model, bnb = load_model()
    stop_ids = resolve_stop_token_ids(tokenizer)
    channel_ids = resolve_channel_token_ids(tokenizer) if thinking else {}
    sha = resolve_model_revision_sha()
    bnb_dump = quant_config_dump(bnb)
    log(f"stop ids:           {stop_ids}")
    if thinking:
        log(f"channel ids:        {channel_ids}")
    log(f"model rev sha:      {sha}")

    # Clear any leftover allocations before starting.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    section("GENERATING")
    pipeline_start = time.time()
    completed = 0
    failed_oom = 0
    failed_parse = 0
    failed_channel = 0
    times: list[float] = []
    last_summary = 0

    for i, b in enumerate(todo, start=1):
        if interrupted():
            log("\nexiting cleanly due to interrupt")
            break

        cooperative_pause_if_needed()
        batch_start = time.time()

        if times:
            recent = times[-min(20, len(times)):]
            eta = format_eta((len(todo) - i + 1) * (sum(recent) / len(recent)))
            eta_str = f" ETA {eta}"
        else:
            eta_str = ""

        try:
            if b["kind"] == "neutral":
                user_msg = b["prompt"]
                seed_parts = ("neutral", b["topic_idx"], b["split"])
            else:
                user_msg = STORY_PROMPT_TEMPLATE.format(
                    n_stories=b["n_stories"],
                    topic=TOPICS[b["topic_idx"]],
                    emotion=b["emotion"],
                )
                seed_parts = (b["emotion"], b["topic_idx"], b["split"])
            seed = deterministic_seed(corpus, *seed_parts)
            rendered = render_prompt(tokenizer, user_msg, thinking)

            try:
                gen_ids, raw_count, gen_secs = generate_one_batch(
                    tokenizer, model, rendered, seed, stop_ids,
                )
            except torch.cuda.OutOfMemoryError as e:
                log(f"[{i}/{len(todo)}] {b['tag']}: OOM — {e}")
                failed_oom += 1
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Phase split (thinking only) BEFORE parse, since the parse
            # operates on the reply portion only.
            thought_text: Optional[str] = None
            if thinking:
                split_result = split_thinking_token_sequence(
                    gen_ids,
                    channel_open_id=channel_ids.get("open", -1),
                    channel_close_id=channel_ids.get("close", -1),
                    label_count=channel_ids.get("label_count", 0),
                )
                if split_result is None:
                    log(f"[{i}/{len(todo)}] {b['tag']}: REJECT (no clean <channel|> boundary)")
                    failed_channel += 1
                    # Save a failure stub so the batch isn't retried — this is
                    # a model-output property, not a transient error.
                    save_batch(batch_path(corpus_dir, b["tag"]), {
                        **{k: v for k, v in b.items() if k != "tag"},
                        "tag": b["tag"],
                        "thinking": True,
                        "seed": seed,
                        "model_id": MODEL_ID,
                        "model_revision_sha": sha,
                        "bnb_quant_config": bnb_dump,
                        "rendered_prompt": rendered,
                        "raw_output_token_count": raw_count,
                        "raw_output_text_with_specials": tokenizer.decode(
                            gen_ids, skip_special_tokens=False,
                        ),
                        "parse_status": "fail_no_channel",
                        "parse_error": "expected exactly one <channel|> token in generated output",
                    })
                    continue
                thought_ids, reply_ids = split_result
                thought_text = tokenizer.decode(
                    thought_ids, skip_special_tokens=True,
                ).strip()
                reply_text = tokenizer.decode(
                    reply_ids, skip_special_tokens=True,
                ).strip()
            else:
                reply_text = tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()

            # Parse stories from the reply portion.
            stories = parse_batched_stories(reply_text, b["n_stories"])
            if stories is None:
                log(
                    f"[{i}/{len(todo)}] {b['tag']}: REJECT "
                    f"(parser failed; expected {b['n_stories']} stories)"
                )
                failed_parse += 1
                save_batch(batch_path(corpus_dir, b["tag"]), {
                    **{k: v for k, v in b.items() if k != "tag"},
                    "tag": b["tag"],
                    "thinking": thinking,
                    "seed": seed,
                    "model_id": MODEL_ID,
                    "model_revision_sha": sha,
                    "bnb_quant_config": bnb_dump,
                    "rendered_prompt": rendered,
                    "raw_output_text": reply_text,
                    "thought": thought_text,
                    "raw_output_token_count": raw_count,
                    "parse_status": f"fail_count_expected_{b['n_stories']}",
                })
                continue

            # Pre-registered synonym filter: per-story leak labels. Apply
            # only to emotion corpora (neutral has no emotion to filter
            # against). Stories are KEPT regardless — leaks are research
            # data, not failures.
            if b["kind"] == "emotion":
                leaks_per_story = [
                    detect_emotion_word_leaks(s, b["emotion"]) for s in stories
                ]
            else:
                leaks_per_story = [[] for _ in stories]
            n_clean = sum(1 for L in leaks_per_story if not L)

            payload = {
                **{k: v for k, v in b.items() if k != "tag"},
                "tag": b["tag"],
                "topic": (
                    TOPICS[b["topic_idx"]] if b["kind"] == "emotion"
                    else (b.get("prompt") or "")
                ),
                "thinking": thinking,
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision_sha": sha,
                "bnb_quant_config": bnb_dump,
                "generation_config": {
                    "max_new_tokens": GEN_MAX_NEW_TOKENS_FALLBACK,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "do_sample": True,
                    "eos_token_id": stop_ids,
                    "eos_token_id_note": (
                        "Drops <|tool_response> from gen_config.eos_token_id "
                        "(agentic-mode-only stop)."
                    ),
                    "no_user_max_cap": True,
                    "max_new_tokens_note": (
                        "GEN_MAX_NEW_TOKENS_FALLBACK is a safety guard only; "
                        "user spec is no token cap. If a batch hits this, "
                        "it likely went runaway and should be inspected."
                    ),
                },
                "rendered_prompt": rendered,
                "raw_output_token_count": raw_count,
                "raw_output_text": reply_text,  # what the model produced (post-thought, post-decode)
                "thought": thought_text,
                "stories": stories,
                "filter_pre_registration_commit": "6cf25bf9952126dd84d1bcac910b933988c62539",
                "contains_emotion_word_per_story": leaks_per_story,
                "n_stories_clean": n_clean,
                "parse_status": "ok",
                "gen_seconds": round(gen_secs, 3),
            }
            save_batch(batch_path(corpus_dir, b["tag"]), payload)

            elapsed = time.time() - batch_start
            times.append(elapsed)
            completed += 1

            # Per-batch CUDA cache flush — user spec; keeps memory tidy
            # so the no-cap generation has clean room for each new batch.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            clean_str = (
                f" clean={n_clean}/{len(stories)}"
                if b["kind"] == "emotion" else ""
            )
            log(
                f"[{i}/{len(todo)}] {b['tag']} "
                f"n={b['n_stories']} toks={raw_count}{clean_str} "
                f"in {gen_secs:.1f}s{eta_str}"
            )

            if completed - last_summary >= 25:
                last_summary = completed
                total_elapsed = time.time() - pipeline_start
                rate = completed / total_elapsed
                log(
                    f"  -- progress: {completed}/{len(todo)} ok, "
                    f"{rate:.2f} batch/s, "
                    f"OOM={failed_oom}, parse_fail={failed_parse}, "
                    f"channel_fail={failed_channel}, "
                    f"elapsed {format_eta(total_elapsed)} --"
                )

        except torch.cuda.OutOfMemoryError as e:
            log(f"[{i}/{len(todo)}] {b['tag']}: OOM (outer) — {e}")
            failed_oom += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            import traceback
            log(f"[{i}/{len(todo)}] {b['tag']}: ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            failed_parse += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    section("SUMMARY")
    total = time.time() - pipeline_start
    log(f"completed:          {completed}")
    log(f"OOM failures:       {failed_oom}")
    log(f"parse failures:     {failed_parse}")
    log(f"channel failures:   {failed_channel}")
    log(f"wall clock:         {format_eta(total)}")
    if completed:
        log(f"avg per batch:      {total/completed:.2f}s")


if __name__ == "__main__":
    main()
