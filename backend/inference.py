"""
Live emotion-projection inference for Gemma 4 E4B.

Loads the model in 4-bit nf4, attaches forward hooks at all 6 target layers,
and exposes `generate_stream(...)` — an async generator that yields one event
per generated token: text delta + per-emotion projection scores at the
caller-chosen layer, normalized via calibration into [0, 100].

Methodology (vectors are the inputs from `data/vectors/`):
  Each emotion has a 2560-dim "direction" at each of 6 layers
  [13, 17, 21, 25, 28, 32] computed by contrastive prompting (Sofroniew et
  al. 2026). Per generated token we dot the residual-stream activation at
  the chosen layer with each emotion vector and normalize.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "google/gemma-4-E4B-it"
EMOTIONS: list[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
TARGET_LAYERS: list[int] = [13, 17, 21, 25, 28, 32]
DEFAULT_LAYER = 21  # paper's main-analysis depth (~2/3 of 42 layers)

# Generation defaults — match the pipeline (which produced the vectors) for
# distributional parity, just shorter to keep responses snappy.
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.95

# Soft cap on streamed tokens/sec. We never delay a slow token, but if the
# model produces tokens faster than this we sleep the difference so the
# frontend halo has time to render each step legibly.
MAX_TOKENS_PER_SEC = 40.0
MIN_TOKEN_INTERVAL_S = 1.0 / MAX_TOKENS_PER_SEC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VECTORS_DIR = DATA_DIR / "vectors"
CALIBRATION_PATH = VECTORS_DIR / "calibration.json"

# Fallback bounds when no calibration file is present. Raw projection scores
# at the chosen layer are clipped to this range, then linearly mapped to
# [0, 100]. Calibration replaces this with per-(emotion, layer) p5/p95 bounds.
FALLBACK_RAW_BOUND = 5.0


# ----------------------------------------------------------------------
# Vector loading
# ----------------------------------------------------------------------

def load_vectors(device: torch.device) -> dict[int, torch.Tensor]:
    """Returns {layer: tensor(6, 2560) on device}, rows ordered by EMOTIONS."""
    out: dict[int, torch.Tensor] = {}
    for L in TARGET_LAYERS:
        rows = []
        for emo in EMOTIONS:
            path = VECTORS_DIR / f"{emo}_layer_{L}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing vector: {path}")
            rows.append(np.load(path).astype(np.float32))
        stacked = np.stack(rows, axis=0)  # (6, 2560)
        out[L] = torch.from_numpy(stacked).to(device)
    return out


def load_calibration() -> Optional[dict]:
    if not CALIBRATION_PATH.exists():
        return None
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Engine — wraps tokenizer + model + hook-captured state
# ----------------------------------------------------------------------

class EmotionEngine:
    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN")

        print(f"[engine] loading {MODEL_ID} (4-bit nf4)...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=token,
        )
        self.model.eval()
        self.device = self.model.device
        print(
            f"[engine] loaded in {time.time() - t0:.1f}s on {self.device}",
            flush=True,
        )

        # Verified path for Gemma 4 E4B in transformers 5.7.
        self.layers = self.model.model.language_model.layers

        # Gemma chat ends a turn with <turn|> (end-of-turn), and the
        # global <eos> always terminates. We deliberately do NOT include
        # <|tool_response> even though Gemma's generation_config lists it
        # as an "eos" — that token is a stop only in agentic mode, where
        # a tool runtime is supposed to inject the response. In plain
        # chat we'd cut off the model mid-thought (e.g. its built-in
        # think/tool-style scaffolding) before the actual reply lands.
        stop_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(int(self.tokenizer.eos_token_id))
        for name in ("<turn|>", "<end_of_turn>"):
            tid = self.tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                stop_ids.add(tid)
        self.stop_token_ids = stop_ids
        print(f"[engine] stop tokens: {sorted(stop_ids)}", flush=True)

        # Channel markers — Gemma 4 wraps its thinking in
        # <|channel>thought\n[reasoning]<channel|>[reply]. We track the
        # marker token IDs so we can split the stream into "thought" and
        # "reply" phases, and so the frontend can render each phase
        # distinctly. We also pre-resolve the token IDs of the literal
        # "thought" channel label so we can skip emitting them as content.
        def _safe_id(name: str) -> Optional[int]:
            tid = self.tokenizer.convert_tokens_to_ids(name)
            return tid if isinstance(tid, int) and tid != self.tokenizer.unk_token_id else None
        self.channel_open_id = _safe_id("<|channel>")
        self.channel_close_id = _safe_id("<channel|>")
        # The label after <|channel> is the channel name + a newline,
        # tokenized. For the "thought" channel that's just `thought\n`,
        # which the tokenizer typically encodes as 1–3 tokens depending
        # on the BPE merges. We compute the count once at init.
        label_ids = self.tokenizer("thought\n", add_special_tokens=False).input_ids
        self.channel_label_token_count = len(label_ids)
        print(
            f"[engine] channel markers: open={self.channel_open_id} "
            f"close={self.channel_close_id} label_tokens={self.channel_label_token_count}",
            flush=True,
        )

        self.vectors = load_vectors(self.device)  # {layer: (6, 2560) tensor}
        self.calibration = load_calibration()
        if self.calibration is None:
            print(
                f"[engine] no calibration.json — using fallback "
                f"[-{FALLBACK_RAW_BOUND}, +{FALLBACK_RAW_BOUND}] linear map",
                flush=True,
            )
        else:
            n_emotions = sum(1 for k in self.calibration if not k.startswith("_"))
            print(
                f"[engine] loaded calibration for {n_emotions} emotions × "
                f"{len(TARGET_LAYERS)} layers",
                flush=True,
            )

        # Forward hooks at every target layer write the latest output here.
        # Captured shape is the full layer output: (batch, seq_len, 2560).
        # Subsequent KV-cached forwards produce (1, 1, 2560).
        self._captured: dict[int, torch.Tensor] = {}
        self._hook_handles: list = []
        self._attach_hooks()

        # Serialize requests: one user, one GPU. Avoid interleaved generations
        # smashing each other's hook state.
        self._lock = asyncio.Lock()

    def _attach_hooks(self) -> None:
        for L in TARGET_LAYERS:
            def make_hook(idx: int):
                def hook(module, inputs, outputs):
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    self._captured[idx] = hidden.detach()
                return hook
            self._hook_handles.append(
                self.layers[L].register_forward_hook(make_hook(L))
            )

    # ------------------------------------------------------------------
    # Projection + normalization
    # ------------------------------------------------------------------

    def _project(self, layer: int) -> dict[str, float]:
        """Raw dot products of last token's residual stream with each emotion
        vector at `layer`. Returns {emotion: float}."""
        hidden = self._captured[layer]                # (1, seq_len, 2560)
        last = hidden[0, -1, :].to(torch.float32)     # (2560,)
        vecs = self.vectors[layer]                    # (6, 2560)
        scores = (vecs @ last).cpu().numpy()          # (6,)
        return {emo: float(scores[i]) for i, emo in enumerate(EMOTIONS)}

    @torch.no_grad()
    def project_all_layers_raw(self) -> dict[int, dict[str, float]]:
        """Raw (un-normalized) projection scores at every target layer for
        the latest captured token. Used by the calibration script."""
        return {L: self._project(L) for L in TARGET_LAYERS}

    @torch.no_grad()
    def calibration_run(self, message: str, max_new_tokens: int = 80):
        """Sync generator: yields {layer: {emotion: raw_score}} per
        generated token. Mirrors generate_stream's inner loop without
        async / SSE / normalize. Used by calibrate.py."""
        input_ids = self._build_prompt(message)
        next_id, past_kv = self._step(input_ids, None, do_sample=True)
        for _ in range(max_new_tokens):
            if next_id in self.stop_token_ids:
                break
            yield self.project_all_layers_raw()
            next_input = torch.tensor(
                [[next_id]], dtype=input_ids.dtype, device=self.device,
            )
            next_id, past_kv = self._step(next_input, past_kv, do_sample=True)

    def _normalize(self, raw: dict[str, float], layer: int) -> dict[str, float]:
        """Map raw projection scores to display range [0, 100].

        Preferred schema (shift_and_scale): pins neutral text to 0 and a
        high percentile of matching-emotional text to 100. Negative shifted
        values clip to 0 — "less aligned than neutral" reads as no signal,
        not as a negative bar.

            display = clip(0, 100, (raw - neutral_mean) / scale * 100)

        Fallback schema (percentile_5_95, legacy): linear p5→0, p95→100.
        """
        out: dict[str, float] = {}
        for emo, val in raw.items():
            bounds = None
            if self.calibration:
                bounds = self.calibration.get(emo, {}).get(str(layer))

            if bounds and "neutral_mean" in bounds and "scale" in bounds:
                shifted = (val - bounds["neutral_mean"]) / bounds["scale"]
                out[emo] = max(0.0, min(100.0, shifted * 100.0))
                continue

            if bounds and "min" in bounds and "max" in bounds:
                lo, hi = bounds["min"], bounds["max"]
            else:
                lo, hi = -FALLBACK_RAW_BOUND, FALLBACK_RAW_BOUND
            span = hi - lo
            if span <= 0:
                out[emo] = 50.0
                continue
            pct = ((val - lo) / span) * 100.0
            out[emo] = max(0.0, min(100.0, pct))
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_prompt(self, message: str, history: Optional[list[dict]] = None) -> torch.Tensor:
        msgs: list[dict] = []
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": message})
        # `enable_thinking=True` injects a system-prompt-level <|think|>
        # signal in Gemma 4 E4B's chat template, which causes the model
        # to produce an explicit thinking block before the reply. Without
        # this flag the model goes straight to the answer with no visible
        # reasoning trace.
        text = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    @torch.no_grad()
    def _step(
        self, input_ids: torch.Tensor, past_key_values, do_sample: bool
    ) -> tuple[int, "object"]:
        out = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        if do_sample:
            logits = logits / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            # Nucleus (top_p) filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumprobs > TOP_P
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_idx.gather(-1, choice).item()
        else:
            next_id = int(torch.argmax(logits, dim=-1).item())
        return next_id, out.past_key_values

    async def generate_stream(
        self,
        message: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream tokens with the *full* layered thinking — all 6 layers,
        normalized — so the frontend can switch displayed layer without
        re-running generation."""

        async with self._lock:
            input_ids = self._build_prompt(message, history)
            past_key_values = None
            generated_ids: list[int] = []
            full_text_so_far = ""
            last_yield_at: Optional[float] = None
            # Phase tracking — see _build_prompt; the model emits
            # <|channel>thought\n[reasoning]<channel|>[reply] when
            # thinking is enabled, and we surface those two streams
            # separately to the frontend.
            phase = "reply"
            label_skip_remaining = 0

            try:
                # First forward (full prompt) — populates KV cache + hook
                # captures. We don't yield a projection for this step; the
                # prompt's own activations aren't a "thinking" event.
                next_id, past_key_values = self._step(
                    input_ids, None, do_sample=True,
                )

                for step in range(max_new_tokens):
                    if next_id in self.stop_token_ids:
                        break

                    # Channel markers steer phase but are not user-visible
                    # text. We still record them in generated_ids so the
                    # decoder's BPE state stays consistent, but we don't
                    # emit a token event for them.
                    is_marker = False
                    if next_id == self.channel_open_id:
                        phase = "thought"
                        label_skip_remaining = self.channel_label_token_count
                        is_marker = True
                    elif next_id == self.channel_close_id:
                        phase = "reply"
                        is_marker = True

                    generated_ids.append(next_id)

                    # Decode incrementally for correct multi-byte handling.
                    decoded = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True,
                    )
                    delta = decoded[len(full_text_so_far):]
                    full_text_so_far = decoded

                    # Project + normalize at every target layer. ~10us total
                    # extra over single-layer projection, negligible.
                    all_raw = self.project_all_layers_raw()
                    all_thinking = {
                        str(L): self._normalize(all_raw[L], L)
                        for L in TARGET_LAYERS
                    }

                    # Drop marker tokens (no visible text anyway, since
                    # decode strips specials) and the channel-label tokens
                    # that follow <|channel> ("thought\n").
                    suppress_emit = is_marker
                    if not is_marker and label_skip_remaining > 0:
                        label_skip_remaining -= 1
                        suppress_emit = True

                    # Always advance to next token even if we skipped emit.
                    next_input = torch.tensor(
                        [[next_id]], dtype=input_ids.dtype, device=self.device,
                    )
                    next_id, past_key_values = self._step(
                        next_input, past_key_values, do_sample=True,
                    )

                    if suppress_emit or not delta:
                        continue

                    if last_yield_at is not None:
                        target = last_yield_at + MIN_TOKEN_INTERVAL_S
                        now = time.monotonic()
                        if now < target:
                            await asyncio.sleep(target - now)
                    last_yield_at = time.monotonic()

                    yield {
                        "type": "token",
                        "text": delta,
                        "thinking": all_thinking,
                        "step": step,
                        "phase": phase,
                    }

                    await asyncio.sleep(0)

                # Debug: log the raw decoded sequence (special tokens
                # visible) so we can see what scaffolding the model
                # emitted (<|think|>, <|tool_response>, etc.).
                raw = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False,
                )
                print(
                    f"[engine] generated {len(generated_ids)} toks; "
                    f"raw sample: {raw[:400]!r}"
                    + (" ..." if len(raw) > 400 else ""),
                    flush=True,
                )

                yield {
                    "type": "done",
                    "fullText": full_text_so_far,
                    "tokens": len(generated_ids),
                }
            finally:
                # Release GPU memory before the next turn. Without this,
                # the KV cache from each turn lingers in PyTorch's
                # allocator and the cache fragments — eventually a new
                # turn fails to allocate even though plenty of bytes are
                # technically free. Hook captures hold per-layer tensors
                # too and need the same explicit drop.
                past_key_values = None
                self._captured.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
