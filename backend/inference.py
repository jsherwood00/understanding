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
MAX_NEW_TOKENS = 280
TEMPERATURE = 0.7
TOP_P = 0.95

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
        self.eos_token_id = self.tokenizer.eos_token_id

        self.vectors = load_vectors(self.device)  # {layer: (6, 2560) tensor}
        self.calibration = load_calibration()
        if self.calibration is None:
            print(
                f"[engine] no calibration.json — using fallback "
                f"[-{FALLBACK_RAW_BOUND}, +{FALLBACK_RAW_BOUND}] linear map",
                flush=True,
            )
        else:
            print(
                f"[engine] loaded calibration for "
                f"{len(self.calibration)} emotions",
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

    def _normalize(self, raw: dict[str, float], layer: int) -> dict[str, float]:
        """Map raw projection scores to display range [0, 100] using
        calibration if present, otherwise fallback linear map."""
        out: dict[str, float] = {}
        for emo, val in raw.items():
            if self.calibration:
                bounds = self.calibration.get(emo, {}).get(str(layer))
                if bounds:
                    lo, hi = bounds["min"], bounds["max"]
                else:
                    lo, hi = -FALLBACK_RAW_BOUND, FALLBACK_RAW_BOUND
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
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
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
        layer: int = DEFAULT_LAYER,
        max_new_tokens: int = MAX_NEW_TOKENS,
        history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        if layer not in TARGET_LAYERS:
            raise ValueError(f"layer must be one of {TARGET_LAYERS}, got {layer}")

        async with self._lock:
            input_ids = self._build_prompt(message, history)
            past_key_values = None
            generated_ids: list[int] = []
            full_text_so_far = ""

            # First forward (full prompt) — populates KV cache + hook captures.
            # Don't yield projection for this step; the prompt's own
            # activations aren't a "thinking" event for the user.
            next_id, past_key_values = self._step(
                input_ids, None, do_sample=True,
            )

            for step in range(max_new_tokens):
                if next_id == self.eos_token_id:
                    break

                generated_ids.append(next_id)

                # Decode incrementally: full decode on every step gives us
                # correct whitespace/multi-byte handling at the cost of
                # decoding a slightly bigger sequence each iteration.
                # Cheap enough to not matter.
                decoded = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                )
                delta = decoded[len(full_text_so_far):]
                full_text_so_far = decoded

                # Project the *just-generated* token's hidden state at the
                # chosen layer. _captured was written by the hook during
                # this step's forward.
                raw = self._project(layer)
                thinking = self._normalize(raw, layer)

                yield {
                    "type": "token",
                    "text": delta,
                    "thinking": thinking,
                    "step": step,
                }

                # Yield to the event loop so SSE can flush.
                await asyncio.sleep(0)

                # Next forward — feed only the new token, with KV cache.
                next_input = torch.tensor(
                    [[next_id]], dtype=input_ids.dtype, device=self.device,
                )
                next_id, past_key_values = self._step(
                    next_input, past_key_values, do_sample=True,
                )

            yield {
                "type": "done",
                "fullText": full_text_so_far,
                "tokens": len(generated_ids),
            }
