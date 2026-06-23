"""
VLM verifier v2 — 5 research-backed improvements over the original.

Improvements (all training-free):
  1. Logit-based confidence: softmax(P(YES), P(NO)) from token logits
     instead of unreliable verbalized "CONFIDENCE: 0.8" scores.
     (Ref: "Seeing is Believing" EMNLP 2025)
  2. Binary prompt with attribute hints: direct YES/NO question enriched
     with distinguishing attributes for confusable discrimination.
     Open-ended two-step prompting was tested but regressed (model
     identified "rice cooker" then still said YES to "thermos").
     (Ref: GUIDED, NeurIPS 2025; HA-FGOVD, TMM 2025)
  3. Multi-crop verification: verify top-K crops, not just top-1.
     If top-1 is wrong but top-3 is correct, the query is rescued.
     (Ref: Adaptive Detector-Verifier, arxiv 2512.12492)
  4. Asymmetric scoring: VLM YES is full strength, VLM NO is damped
     because verifying a crop != verifying the scene.

SAHI tiled detection is implemented separately in grounding_dino.py.
"""

import os
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ─── Configuration ─────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

NO_DAMPING = 0.3

# Binary verification prompt enriched with attribute hints.
# Open-ended two-step prompting (NegToMe, ICLR 2026) was tested but caused
# regressions: the model correctly identified "electric rice cooker" in step 1
# but then incorrectly concluded "thermos" was present in step 2. A direct
# binary question with attribute hints is simpler and more reliable for
# confusable discrimination.
_VERIFY_PROMPT = (
    "Look at this cropped image from a kitchen video.\n\n"
    "Target object: {target}{attr_hint}\n\n"
    "Is this specific object clearly visible as the main subject of this crop? "
    "Be strict: if you see a similar but different object, say NO.\n\n"
    "VERDICT: YES or NO\n"
    "REASON: <one sentence>"
)

# Attribute hints for common kitchen objects (Priority 5).
# Helps the VLM distinguish confusable pairs.
_ATTRIBUTE_HINTS = {
    "knife":         " (metal blade, sharp cutting edge, handle — not a fork or spatula)",
    "fork":          " (pronged metal utensil with tines — not a knife or spoon)",
    "spoon":         " (rounded bowl-shaped head — not a fork or ladle)",
    "spatula":       " (flat, wide blade for flipping — not a knife)",
    "kettle":        " (container for boiling water, with spout and handle — not a pot or thermos)",
    "thermos":       " (insulated cylindrical bottle, usually with screw cap — not a kettle or water bottle)",
    "pan":           " (flat cooking surface with handle — not a pot or wok)",
    "pot":           " (deep round cooking vessel — not a pan or bowl)",
    "wok":           " (large round-bottomed cooking pan — not a regular pan)",
    "bowl":          " (round open-top container for food — not a plate or cup)",
    "plate":         " (flat circular dish for serving — not a bowl or lid)",
    "lid":           " (flat cover for a pot or pan — not a plate)",
    "mug":           " (handled drinking cup, usually ceramic — not a glass)",
    "cup":           " (small drinking vessel — not a mug or bowl)",
    "glass":         " (transparent drinking vessel, no handle — not a cup or jar)",
    "bottle":        " (narrow-necked container — not a jar or thermos)",
    "jar":           " (wide-mouthed container, usually glass — not a bottle)",
    "sponge":        " (soft porous cleaning pad — not a cloth or towel)",
    "towel":         " (fabric for drying — not a cloth or sponge)",
    "cloth":         " (piece of fabric — not a towel or paper towel)",
    "cutting board": " (flat board surface for cutting — not a tray or plate)",
    "chopsticks":    " (pair of thin sticks for eating — not a skewer or tongs)",
    "tongs":         " (hinged gripping tool — not chopsticks or scissors)",
    "whisk":         " (wire loops on a handle for mixing — not a fork or spoon)",
    "ladle":         " (deep-bowled spoon with long handle — not a regular spoon)",
    "peeler":        " (tool with a blade for removing skin — not a knife)",
    "grater":        " (metal surface with sharp holes for shredding — not a sieve)",
    "toaster":       " (small appliance with slots for bread — not a microwave)",
    "microwave":     " (box-shaped appliance with door — not an oven or toaster)",
    "oven":          " (large built-in cooking appliance with door — not a microwave)",
    "fridge":        " (large cooling appliance — not a freezer or cabinet)",
    "dishwasher":    " (appliance for washing dishes, front-loading — not a washing machine)",
    "blender":       " (tall appliance with blades for mixing — not a food processor)",
    "rice cooker":   " (electric pot with lid for cooking rice — not a slow cooker)",
    "lighter":       " (small flame-producing device — not a match)",
    "bread":         " (baked food, loaf or sliced — not a pastry or cake)",
    "tap":           " (water faucet mounted on sink — not a handle or knob)",
    "sink":          " (basin with drain for washing — not a bowl or tub)",
    "handle":        " (gripping part of a door or drawer — not a knob)",
    "knob":          " (round turning control — not a handle or button)",
    "counter":       " (flat kitchen work surface — not a table or shelf)",
    "shelf":         " (horizontal storage surface mounted on wall — not a counter)",
    "cabinet":       " (enclosed storage with door — not a shelf or drawer)",
    "drawer":        " (sliding storage compartment — not a cabinet)",
    "stainless steel pot": " (metallic silvery deep cooking vessel — not a pan or bowl)",
    "wooden spoon":  " (spoon made of wood — not a metal spoon or spatula)",
    "frying pan":    " (flat pan for frying with handle — not a saucepan or wok)",
    "saucepan":      " (deep pan with long handle — not a pot or frying pan)",
    "paper towel":   " (disposable paper sheet for cleaning — not a cloth towel)",
    "salt shaker":   " (small container with holes for dispensing salt — not a pepper mill)",
    "plastic bottle": " (bottle made of plastic — not a glass bottle)",
    "white plate":   " (plate that is white in color — not a colored plate)",
    "red bowl":      " (bowl that is red in color — not a white or metal bowl)",
    "yellow sponge": " (sponge that is yellow — not a different colored sponge)",
    "silver knife":  " (knife with silver/metallic blade — not a plastic knife)",
    "blue cup":      " (cup that is blue in color — not a mug or glass)",
    "ceramic bowl":  " (bowl made of ceramic/pottery — not metal or plastic)",
    "glass jar":     " (jar made of clear glass — not a plastic container)",
    "metal kettle":  " (kettle made of metal — not plastic or ceramic)",
    "wooden cutting board": " (cutting board made of wood — not plastic)",
    "wok pan":       " (round-bottomed deep pan — not a flat frying pan)",
    "apron":         " (protective garment worn in front — not a towel or cloth)",
    "hob":           " (stovetop cooking surface with burners — not a counter)",
    "scale":         " (weighing device — not a cutting board or tray)",
}


def _get_attr_hint(target: str) -> str:
    """Look up attribute hint for a target, with fuzzy matching."""
    t = target.lower().strip()
    if t in _ATTRIBUTE_HINTS:
        return _ATTRIBUTE_HINTS[t]
    for key, hint in _ATTRIBUTE_HINTS.items():
        if key in t or t in key:
            return hint
    return ""


# ─── Model wrapper ─────────────────────────────────────────────────────

class VLMVerifier:

    def __init__(self,
                 model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.model_id = model_id
        self.device   = device
        self.dtype    = dtype
        self.model     = None
        self.processor = None
        self._yes_token_id = None
        self._no_token_id  = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                  AutoProcessor)
        print(f"[VLMVerifier] loading {self.model_id} (~6 GB VRAM)...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._cache_token_ids()

    def _cache_token_ids(self):
        """Pre-compute YES/NO token IDs for logit extraction."""
        tokenizer = self.processor.tokenizer
        for candidate in ["YES", "Yes", "yes"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                self._yes_token_id = ids[0]
                break
        for candidate in ["NO", "No", "no"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                self._no_token_id = ids[0]
                break
        if self._yes_token_id is not None and self._no_token_id is not None:
            print(f"  [VLM] logit extraction ready: YES={self._yes_token_id}, "
                  f"NO={self._no_token_id}")
        else:
            print(f"  [VLM] logit extraction unavailable — YES/NO not single tokens")

    def unload(self):
        """Free VRAM so other models (SigLIP, GDino) aren't starved."""
        if self.model is None:
            return
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._yes_token_id = None
        self._no_token_id = None
        torch.cuda.empty_cache()
        print("[VLMVerifier] unloaded, VRAM freed")

    @torch.no_grad()
    def verify(self, crop_pil: Image.Image, target: str,
               max_new_tokens: int = 120) -> Tuple[float, dict]:
        """
        Returns (s_vlm, diagnostics).

        s_vlm range: roughly [-0.3, +1.0]
          Positive = VLM confirmed target present (full confidence)
          Negative = VLM said target absent (damped by NO_DAMPING=0.3)
          Zero     = VLM unsure or parse failure
        """
        self._load()

        attr_hint = _get_attr_hint(target)
        prompt = _VERIFY_PROMPT.format(target=target, attr_hint=attr_hint)

        messages = [
            {"role": "user",
             "content": [
                 {"type": "image", "image": crop_pil},
                 {"type": "text",  "text":  prompt},
             ]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[crop_pil],
            padding=True, return_tensors="pt",
        ).to(self.device)

        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0,
            output_scores=True, return_dict_in_generate=True,
        )
        out_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(
            out_ids, skip_special_tokens=True)[0]

        verdict, conf_parsed, reason = self._parse(decoded)

        logit_conf = self._extract_logit_confidence(out, inputs)
        conf = logit_conf if logit_conf is not None else conf_parsed

        if verdict == "YES":
            s_vlm = +float(conf)
        elif verdict == "NO":
            s_vlm = -float(conf) * NO_DAMPING
        else:
            s_vlm = 0.0

        diag = {
            "raw": decoded, "verdict": verdict,
            "confidence": round(conf, 3),
            "confidence_source": "logit" if logit_conf is not None else "parsed",
            "reason": reason,
            "attr_hint_used": bool(attr_hint),
        }
        return s_vlm, diag

    def _extract_logit_confidence(self, gen_output, inputs) -> Optional[float]:
        """Extract P(YES) from token logits at the VERDICT position."""
        if not hasattr(gen_output, 'scores') or not gen_output.scores:
            return None
        if self._yes_token_id is None or self._no_token_id is None:
            return None

        try:
            decoded_ids = gen_output.sequences[0, inputs["input_ids"].shape[1]:]
            verdict_pos = None
            for i, tok_id in enumerate(decoded_ids):
                if tok_id.item() == self._yes_token_id:
                    verdict_pos = i
                    break
                if tok_id.item() == self._no_token_id:
                    verdict_pos = i
                    break

            if verdict_pos is None or verdict_pos >= len(gen_output.scores):
                return None

            logits = gen_output.scores[verdict_pos][0]
            yes_logit = logits[self._yes_token_id].float()
            no_logit = logits[self._no_token_id].float()
            probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
            p_yes = probs[0].item()
            return p_yes
        except Exception:
            return None

    @staticmethod
    def _parse(text: str) -> Tuple[str, float, str]:
        verdict = "UNSURE"
        conf = 0.5
        reason = ""
        m = re.search(r"VERDICT:\s*(YES|NO|UNSURE)", text, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
        else:
            first_word = text.strip().split()[0].strip(".,!:") if text.strip() else ""
            if first_word.upper() in ("YES", "NO", "UNSURE"):
                verdict = first_word.upper()
        m = re.search(r"CONFIDENCE:\s*([0-1](?:\.\d+)?)", text)
        if m:
            try: conf = float(m.group(1))
            except ValueError: conf = 0.5
            conf = max(0.0, min(1.0, conf))
        m = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
        return verdict, conf, reason


# ─── Multi-crop verification (Priority 3) ─────────────────────────────

def maybe_verify_multi(
    verifier: Optional[VLMVerifier],
    crops: List[Image.Image],
    target: str,
    m_conf: float,
    min_margin_for_skip: float = 0.05,
) -> Tuple[float, dict]:
    """
    Verify up to K crops instead of just top-1. If ANY crop gets YES,
    we take the max positive s_vlm (rescuing false negatives where top-1
    was wrong but top-2 or top-3 was correct).

    For NO verdicts, we take the least-negative s_vlm (least damaging),
    since one bad crop doesn't prove absence.
    """
    if verifier is None:
        return 0.0, {"skipped": True, "reason": "verifier disabled"}
    if m_conf >= min_margin_for_skip:
        return 0.0, {"skipped": True, "reason": f"m_conf={m_conf:.3f} clean"}
    if not crops:
        return 0.0, {"skipped": True, "reason": "no crops"}

    t0 = time.time()
    all_results = []

    for i, crop in enumerate(crops):
        s_vlm, diag = verifier.verify(crop, target)
        diag["crop_index"] = i
        all_results.append((s_vlm, diag))

        # Early exit: if VLM says YES on any crop, no need to check more
        if diag.get("verdict") == "YES":
            break

    verifier.unload()

    # Aggregation: take the best (most positive) result
    best_s_vlm = max(r[0] for r in all_results)
    best_idx = next(i for i, (s, _) in enumerate(all_results) if s == best_s_vlm)
    best_diag = all_results[best_idx][1]

    best_diag["n_crops_checked"] = len(all_results)
    best_diag["all_verdicts"] = [d.get("verdict", "?") for _, d in all_results]
    best_diag["latency_s"] = round(time.time() - t0, 2)
    best_diag["skipped"] = False

    return best_s_vlm, best_diag


# ─── Single-crop wrapper (backwards compatible) ───────────────────────

def maybe_verify(verifier: Optional[VLMVerifier],
                 crop_pil: Image.Image, target: str,
                 m_conf: float, min_margin_for_skip: float = 0.05
                 ) -> Tuple[float, dict]:
    """Single-crop verification. Use maybe_verify_multi for top-K."""
    return maybe_verify_multi(
        verifier, [crop_pil], target, m_conf, min_margin_for_skip)


# ─── Standalone smoke test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to crop image")
    ap.add_argument("--target", required=True, help="what to look for")
    args = ap.parse_args()
    v = VLMVerifier()
    crop = Image.open(args.image).convert("RGB")
    s, d = v.verify(crop, args.target)
    print(f"\ns_vlm = {s:+.3f}")
    print(f"diagnostics:")
    for k, val in d.items():
        print(f"  {k}: {val}")
