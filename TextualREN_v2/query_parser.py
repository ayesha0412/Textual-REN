"""
Query Parser: Domain-aware query understanding for egocentric video localization.

Parses ambiguous text queries into structured detection plans with:
  - Disambiguated target description
  - Grounding DINO detection prompts (multi-label)
  - Confusable class lists for negative suppression
  - CLIP retrieval prompts (richer text for CLS/patch matching)

Two modes:
  - "rule_based" (default): Fast, deterministic, no external deps.
    Uses a kitchen/egocentric domain ontology to resolve ambiguity.
  - "llm": Calls an LLM API for truly novel queries. Caches results
    to disk for reproducibility. Falls back to rule_based on failure.

Usage:
    parser = QueryParser(mode="rule_based")
    plan = parser.parse("pan")
    # -> QueryPlan(target="frying pan", detection_prompt="frying pan. cooking pan. saucepan",
    #              confusables=["plate", "lid", "tray", "bowl"], ...)
"""

import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class QueryPlan:
    """Structured detection plan for a text query."""
    original_query: str             # Raw user query
    target: str                     # Disambiguated target (e.g., "frying pan")
    detection_prompt: str           # Multi-label GDino prompt (e.g., "frying pan. cooking pan.")
    confusables: List[str]          # Objects that look similar (for negative suppression)
    retrieval_prompts: List[str]    # CLIP text prompts for CLS/patch retrieval
    context_hint: str               # Human-readable context (e.g., "metal cooking utensil on stove")
    domain: str = "kitchen"         # Detected domain
    confidence: float = 1.0         # Parser confidence (1.0 = dictionary match, 0.5 = heuristic)

    def to_dict(self) -> dict:
        return asdict(self)


class QueryParser:
    """
    Domain-aware query parser for egocentric video object localization.

    Resolves ambiguous queries using a kitchen/egocentric ontology:
      - "pan" -> "frying pan" (not camera pan, not washing pan)
      - "tap" -> "kitchen faucet" (not beer tap, not shoulder tap)
      - "board" -> "cutting board" (not whiteboard, not circuit board)

    LLM mode uses **open-source local models only** (no proprietary APIs):
      - Ollama (recommended): runs models like phi3, llama3.2, mistral locally
      - HuggingFace Transformers: loads small models like Phi-3-mini on GPU
    """

    def __init__(self, mode: str = "rule_based", cache_dir: str = None,
                 llm_backend: str = "ollama",
                 llm_model: str = "phi3:mini",
                 ollama_url: str = "http://localhost:11434"):
        """
        Args:
            mode: "rule_based" or "llm"
            cache_dir: Directory for LLM response cache (default: .query_cache/)
            llm_backend: "ollama" (local REST API) or "transformers" (HuggingFace)
            llm_model: Model name — for ollama: "phi3:mini", "llama3.2:3b",
                       "mistral:7b"; for transformers: HF model ID
            ollama_url: Ollama server URL (default: http://localhost:11434)
        """
        self.mode = mode
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '.query_cache'
        )
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.ollama_url = ollama_url.rstrip('/')
        self._hf_pipeline = None  # lazy-loaded for transformers backend

    def parse(self, text_query: str) -> QueryPlan:
        """
        Parse a text query into a structured detection plan.

        Always tries rule-based first (fast, deterministic).
        If mode="llm" and rule-based has low confidence, calls LLM.
        """
        # Try rule-based first
        plan = self._rule_based_parse(text_query)

        # If LLM mode and rule-based wasn't confident, enhance with LLM
        if self.mode == "llm" and plan.confidence < 0.8:
            llm_plan = self._llm_parse(text_query)
            if llm_plan is not None:
                return llm_plan

        return plan

    # ------------------------------------------------------------------ #
    # Rule-based parser (domain ontology)                                  #
    # ------------------------------------------------------------------ #

    # Kitchen/egocentric domain ontology
    # Each entry: query -> {target, detection_prompt, confusables, retrieval_prompts, context}
    _KITCHEN_ONTOLOGY = {
        # === Cookware ===
        'pan': {
            'target': 'frying pan',
            'detection_prompt': 'frying pan. cooking pan. saucepan with handle',
            'confusables': ['plate', 'lid', 'tray', 'bowl', 'wok', 'pot lid'],
            'retrieval_prompts': [
                'a frying pan', 'a cooking pan on the stove',
                'a saucepan with handle', 'a metal frying pan'
            ],
            'context': 'metal cooking utensil with handle, usually on stove or drying rack',
        },
        'pot': {
            'target': 'cooking pot',
            'detection_prompt': 'cooking pot. saucepan. stock pot',
            'confusables': ['vase', 'jar', 'mug', 'bucket', 'bowl'],
            'retrieval_prompts': [
                'a cooking pot', 'a large cooking pot on stove',
                'a saucepan', 'a metal pot with handles'
            ],
            'context': 'deep metal container for cooking, usually on stove',
        },
        'wok': {
            'target': 'wok',
            'detection_prompt': 'wok. stir fry pan. Chinese wok',
            'confusables': ['frying pan', 'bowl', 'pot'],
            'retrieval_prompts': ['a wok', 'a stir fry wok', 'a large round wok'],
            'context': 'large round-bottomed cooking pan',
        },

        # === Tableware ===
        'plate': {
            'target': 'dinner plate',
            'detection_prompt': 'dinner plate. ceramic plate. serving plate',
            'confusables': ['lid', 'tray', 'pan bottom', 'cutting board'],
            'retrieval_prompts': [
                'a dinner plate', 'a white plate', 'a ceramic plate on counter'
            ],
            'context': 'flat round dish for serving food',
        },
        'bowl': {
            'target': 'bowl',
            'detection_prompt': 'mixing bowl. cereal bowl. soup bowl',
            'confusables': ['cup', 'mug', 'pot', 'container'],
            'retrieval_prompts': ['a bowl', 'a mixing bowl', 'a round bowl on counter'],
            'context': 'round open-top container for food',
        },
        'cup': {
            'target': 'drinking cup',
            'detection_prompt': 'coffee cup. drinking cup. tea cup. mug',
            'confusables': ['mug', 'jar', 'glass', 'small bowl'],
            'retrieval_prompts': ['a cup', 'a coffee cup', 'a drinking cup', 'a tea cup'],
            'context': 'small container for hot or cold drinks',
        },
        'mug': {
            'target': 'mug',
            'detection_prompt': 'coffee mug. ceramic mug. drinking mug',
            'confusables': ['cup', 'jar', 'glass'],
            'retrieval_prompts': ['a mug', 'a coffee mug', 'a ceramic mug with handle'],
            'context': 'sturdy cup with handle, usually for hot drinks',
        },
        'glass': {
            'target': 'drinking glass',
            'detection_prompt': 'drinking glass. water glass. wine glass',
            'confusables': ['jar', 'bottle', 'cup', 'vase'],
            'retrieval_prompts': ['a drinking glass', 'a glass of water', 'a clear glass'],
            'context': 'transparent container for cold drinks',
        },

        # === Utensils ===
        'knife': {
            'target': 'kitchen knife',
            'detection_prompt': 'kitchen knife. chopping knife. chef knife. bread knife',
            'confusables': ['fork', 'spoon', 'spatula', 'peeler'],
            'retrieval_prompts': [
                'a kitchen knife', 'a chef knife', 'a sharp knife on cutting board'
            ],
            'context': 'sharp bladed utensil for cutting food',
        },
        'fork': {
            'target': 'fork',
            'detection_prompt': 'dinner fork. eating fork. metal fork',
            'confusables': ['spoon', 'knife', 'spatula', 'whisk'],
            'retrieval_prompts': ['a fork', 'a dinner fork', 'a metal fork'],
            'context': 'pronged utensil for eating',
        },
        'spoon': {
            'target': 'spoon',
            'detection_prompt': 'cooking spoon. tablespoon. serving spoon. wooden spoon',
            'confusables': ['fork', 'knife', 'spatula', 'ladle'],
            'retrieval_prompts': ['a spoon', 'a cooking spoon', 'a wooden spoon'],
            'context': 'rounded utensil for stirring or eating',
        },
        'spatula': {
            'target': 'spatula',
            'detection_prompt': 'spatula. cooking spatula. turner. flipper',
            'confusables': ['spoon', 'knife', 'fork', 'tongs'],
            'retrieval_prompts': ['a spatula', 'a cooking spatula', 'a kitchen turner'],
            'context': 'flat utensil for flipping or spreading food',
        },

        # === Surfaces / Boards ===
        'board': {
            'target': 'cutting board',
            'detection_prompt': 'cutting board. chopping board. wooden board',
            'confusables': ['shelf', 'counter', 'table', 'tray', 'wooden surface'],
            'retrieval_prompts': [
                'a cutting board', 'a chopping board', 'a wooden cutting board'
            ],
            'context': 'flat board for cutting food, usually wood or plastic',
        },
        'cutting board': {
            'target': 'cutting board',
            'detection_prompt': 'cutting board. chopping board. plastic cutting board. wooden chopping board',
            'confusables': ['counter', 'shelf', 'table', 'wooden surface', 'tray'],
            'retrieval_prompts': [
                'a cutting board', 'a chopping board on counter',
                'a plastic cutting board', 'a wooden cutting board'
            ],
            'context': 'flat board for cutting food, on kitchen counter',
        },

        # === Fixtures ===
        'tap': {
            'target': 'kitchen faucet',
            'detection_prompt': 'kitchen faucet. water tap. sink faucet. kitchen tap',
            'confusables': ['pipe', 'handle', 'shower head', 'hose nozzle'],
            'retrieval_prompts': [
                'a kitchen faucet', 'a water tap', 'a sink faucet',
                'a metal faucet above the sink'
            ],
            'context': 'metal fixture above kitchen sink for dispensing water',
        },
        'faucet': {
            'target': 'kitchen faucet',
            'detection_prompt': 'kitchen faucet. water tap. sink faucet',
            'confusables': ['pipe', 'handle', 'shower head'],
            'retrieval_prompts': ['a kitchen faucet', 'a water faucet', 'a sink faucet'],
            'context': 'metal fixture above kitchen sink',
        },
        'sink': {
            'target': 'kitchen sink',
            'detection_prompt': 'kitchen sink. washing sink. stainless steel sink',
            'confusables': ['basin', 'bathtub', 'bucket', 'washing bowl'],
            'retrieval_prompts': ['a kitchen sink', 'a stainless steel sink'],
            'context': 'basin for washing dishes, usually stainless steel',
        },

        # === Appliances ===
        'kettle': {
            'target': 'electric kettle',
            'detection_prompt': 'electric kettle. tea kettle. water kettle. stovetop kettle',
            'confusables': ['water filter pitcher', 'jug', 'coffee maker', 'teapot', 'thermos'],
            'retrieval_prompts': [
                'an electric kettle', 'a tea kettle', 'a water kettle on counter'
            ],
            'context': 'appliance for boiling water, usually electric with base station',
        },
        'toaster': {
            'target': 'toaster',
            'detection_prompt': 'toaster. bread toaster. pop-up toaster. toaster oven',
            'confusables': ['microwave', 'oven', 'air fryer'],
            'retrieval_prompts': ['a toaster', 'a bread toaster on counter'],
            'context': 'small appliance for toasting bread',
        },
        'microwave': {
            'target': 'microwave oven',
            'detection_prompt': 'microwave oven. microwave. countertop microwave',
            'confusables': ['oven', 'toaster oven', 'dishwasher'],
            'retrieval_prompts': ['a microwave', 'a microwave oven'],
            'context': 'countertop or built-in appliance for heating food',
        },
        'blender': {
            'target': 'blender',
            'detection_prompt': 'blender. kitchen blender. smoothie blender',
            'confusables': ['food processor', 'mixer', 'jar', 'bottle'],
            'retrieval_prompts': ['a blender', 'a kitchen blender on counter'],
            'context': 'appliance with blade for blending food or drinks',
        },

        # === Cleaning ===
        'sponge': {
            'target': 'kitchen sponge',
            'detection_prompt': 'kitchen sponge. dish sponge. cleaning sponge. scrub sponge',
            'confusables': ['cloth', 'towel', 'rag', 'scrubber', 'soap bar'],
            'retrieval_prompts': [
                'a kitchen sponge', 'a dish sponge', 'a yellow sponge near sink'
            ],
            'context': 'soft porous pad for washing dishes, often yellow/green',
        },
        'towel': {
            'target': 'kitchen towel',
            'detection_prompt': 'kitchen towel. dish towel. hand towel. tea towel',
            'confusables': ['cloth', 'rag', 'napkin', 'paper towel'],
            'retrieval_prompts': ['a kitchen towel', 'a dish towel hanging'],
            'context': 'fabric cloth for drying hands or dishes',
        },

        # === Containers ===
        'lid': {
            'target': 'pot lid',
            'detection_prompt': 'pot lid. pan lid. container lid. glass lid',
            'confusables': ['plate', 'tray', 'pan bottom', 'frisbee'],
            'retrieval_prompts': ['a pot lid', 'a glass lid', 'a pan lid'],
            'context': 'round cover for pots and pans',
        },
        'jar': {
            'target': 'glass jar',
            'detection_prompt': 'glass jar. storage jar. mason jar. food jar',
            'confusables': ['bottle', 'glass', 'vase', 'container'],
            'retrieval_prompts': ['a glass jar', 'a storage jar', 'a jar with lid'],
            'context': 'glass or plastic container with screw-top lid',
        },
        'bottle': {
            'target': 'bottle',
            'detection_prompt': 'bottle. water bottle. plastic bottle. glass bottle',
            'confusables': ['jar', 'glass', 'vase', 'thermos'],
            'retrieval_prompts': ['a bottle', 'a water bottle', 'a plastic bottle'],
            'context': 'narrow-necked container for liquids',
        },
        'box': {
            'target': 'box',
            'detection_prompt': 'cardboard box. storage box. food box. cereal box',
            'confusables': ['container', 'package', 'bag', 'carton'],
            'retrieval_prompts': ['a box', 'a cardboard box', 'a food box on counter'],
            'context': 'rectangular container, cardboard or plastic',
        },
        'bag': {
            'target': 'bag',
            'detection_prompt': 'plastic bag. shopping bag. food bag. paper bag',
            'confusables': ['container', 'wrapper', 'cloth', 'package'],
            'retrieval_prompts': ['a bag', 'a plastic bag', 'a shopping bag'],
            'context': 'flexible container for carrying items',
        },
        'tin': {
            'target': 'tin can',
            'detection_prompt': 'tin can. food tin. canned food. metal can',
            'confusables': ['jar', 'cup', 'container', 'mug'],
            'retrieval_prompts': ['a tin can', 'a canned food item', 'a metal tin'],
            'context': 'cylindrical metal container for preserved food',
        },
        'container': {
            'target': 'food container',
            'detection_prompt': 'food container. storage container. tupperware. plastic container',
            'confusables': ['box', 'bowl', 'jar', 'lid'],
            'retrieval_prompts': ['a food container', 'a plastic container', 'a storage box'],
            'context': 'sealed container for storing food, usually plastic',
        },
        'tray': {
            'target': 'tray',
            'detection_prompt': 'serving tray. baking tray. oven tray. food tray',
            'confusables': ['plate', 'cutting board', 'lid', 'pan'],
            'retrieval_prompts': ['a tray', 'a baking tray', 'a serving tray'],
            'context': 'flat surface for carrying or baking food',
        },

        # === Food ===
        'onion': {
            'target': 'onion',
            'detection_prompt': 'onion. yellow onion. red onion. whole onion',
            'confusables': ['apple', 'potato', 'garlic', 'tomato'],
            'retrieval_prompts': ['an onion', 'a yellow onion', 'an onion on cutting board'],
            'context': 'round layered vegetable',
        },
        'tomato': {
            'target': 'tomato',
            'detection_prompt': 'tomato. red tomato. cherry tomato',
            'confusables': ['apple', 'pepper', 'onion', 'ball'],
            'retrieval_prompts': ['a tomato', 'a red tomato', 'tomatoes on counter'],
            'context': 'round red fruit/vegetable',
        },
        'bread': {
            'target': 'bread',
            'detection_prompt': 'bread. bread loaf. sliced bread. toast',
            'confusables': ['cutting board', 'sponge', 'cake'],
            'retrieval_prompts': ['a loaf of bread', 'sliced bread', 'bread on counter'],
            'context': 'baked food made from flour',
        },
    }

    # Semantic category mapping for heuristic confusable generation
    _CATEGORY_MAP = {
        'cookware':   ['pan', 'pot', 'wok', 'lid'],
        'tableware':  ['plate', 'bowl', 'cup', 'mug', 'glass', 'tray'],
        'utensils':   ['knife', 'fork', 'spoon', 'spatula', 'tongs', 'whisk', 'ladle', 'peeler'],
        'containers': ['jar', 'bottle', 'box', 'bag', 'tin', 'container'],
        'appliances': ['kettle', 'toaster', 'microwave', 'blender', 'coffee maker'],
        'cleaning':   ['sponge', 'towel', 'cloth', 'scrubber'],
        'surfaces':   ['board', 'cutting board', 'counter', 'shelf', 'table'],
        'fixtures':   ['tap', 'faucet', 'sink'],
    }

    def _rule_based_parse(self, text_query: str) -> QueryPlan:
        """Parse query using kitchen domain ontology."""
        q = text_query.lower().strip()

        # Direct match in ontology
        if q in self._KITCHEN_ONTOLOGY:
            entry = self._KITCHEN_ONTOLOGY[q]
            return QueryPlan(
                original_query=text_query,
                target=entry['target'],
                detection_prompt=entry['detection_prompt'],
                confusables=entry['confusables'],
                retrieval_prompts=entry['retrieval_prompts'],
                context_hint=entry['context'],
                domain='kitchen',
                confidence=1.0,
            )

        # Multi-word query: check if any word matches
        words = q.split()
        for word in reversed(words):  # last word is often the object
            if word in self._KITCHEN_ONTOLOGY:
                entry = self._KITCHEN_ONTOLOGY[word]
                # Adjust detection prompt to include the full query
                det_prompt = f"{text_query}. {entry['detection_prompt']}"
                return QueryPlan(
                    original_query=text_query,
                    target=text_query,
                    detection_prompt=det_prompt,
                    confusables=entry['confusables'],
                    retrieval_prompts=[f"a {text_query}"] + entry['retrieval_prompts'],
                    context_hint=entry['context'],
                    domain='kitchen',
                    confidence=0.8,
                )

        # Heuristic: generate confusables from category
        confusables = self._heuristic_confusables(q)
        return QueryPlan(
            original_query=text_query,
            target=text_query,
            detection_prompt=text_query,
            confusables=confusables,
            retrieval_prompts=[f"a {text_query}", f"a {text_query} in a kitchen"],
            context_hint=f"object described as '{text_query}'",
            domain='kitchen',
            confidence=0.5,
        )

    def _heuristic_confusables(self, query: str) -> List[str]:
        """Generate confusable classes using category co-membership."""
        # Find which category the query belongs to
        for category, members in self._CATEGORY_MAP.items():
            if query in members:
                # Confusables = other members of the same category
                return [m for m in members if m != query][:5]

        # No category match — return generic kitchen objects that could confuse
        return []

    # ------------------------------------------------------------------ #
    # LLM-enhanced parser (open-source models only)                        #
    # ------------------------------------------------------------------ #

    _LLM_SYSTEM_PROMPT = """You are a query parser for an egocentric kitchen video object localization system.
Given a text query for an object to find in kitchen video, return a JSON object with:
- "target": the disambiguated object name (e.g., "pan" -> "frying pan")
- "detection_prompt": multi-label prompt for Grounding DINO detection (period-separated, e.g., "frying pan. cooking pan. saucepan")
- "confusables": list of 3-6 visually similar objects that could be confused (e.g., ["plate", "lid", "tray"])
- "retrieval_prompts": list of 3-4 CLIP text descriptions (e.g., ["a frying pan", "a cooking pan on the stove"])
- "context_hint": one sentence describing what the object looks like in a kitchen (e.g., "metal cooking utensil with handle, usually on stove")

The video is first-person (egocentric) from someone cooking in a kitchen.
Focus on VISUAL confusion — objects that look similar in shape, color, or texture.
Return ONLY valid JSON, no explanation."""

    def _llm_parse(self, text_query: str) -> Optional[QueryPlan]:
        """
        Parse query using a LOCAL open-source LLM with disk caching.

        Supported backends:
          - "ollama": Ollama REST API (recommended, zero-code setup)
                      Install: https://ollama.com  then `ollama pull phi3:mini`
          - "transformers": HuggingFace Transformers pipeline
                      Uses models like microsoft/Phi-3-mini-4k-instruct
        """
        # ---- Check disk cache first ----
        cache_key = hashlib.md5(text_query.lower().strip().encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return self._plan_from_dict(text_query, data)
            except (json.JSONDecodeError, KeyError):
                pass

        # ---- Call local LLM ----
        prompt = (
            f'{self._LLM_SYSTEM_PROMPT}\n\n'
            f'Query: "{text_query}"\n\n'
            f'Return ONLY a JSON object, no other text.'
        )

        raw_response = None
        if self.llm_backend == "ollama":
            raw_response = self._call_ollama(prompt)
        elif self.llm_backend == "transformers":
            raw_response = self._call_transformers(prompt)
        else:
            print(f"  [QueryParser] Unknown backend '{self.llm_backend}', "
                  f"falling back to rule-based")
            return None

        if raw_response is None:
            return None

        # ---- Parse JSON from response ----
        try:
            data = self._extract_json(raw_response)
            if data is None:
                print(f"  [QueryParser] Could not parse JSON from LLM response")
                return None

            # Cache result for reproducibility
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            return self._plan_from_dict(text_query, data)

        except Exception as e:
            print(f"  [QueryParser] Failed to parse LLM output: {e}")
            return None

    def _plan_from_dict(self, text_query: str, data: dict) -> QueryPlan:
        """Create a QueryPlan from a parsed JSON dict."""
        return QueryPlan(
            original_query=text_query,
            target=data.get('target', text_query),
            detection_prompt=data.get('detection_prompt', text_query),
            confusables=data.get('confusables', []),
            retrieval_prompts=data.get('retrieval_prompts', [f"a {text_query}"]),
            context_hint=data.get('context_hint', ''),
            domain='kitchen',
            confidence=0.9,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON object from LLM response (handles markdown fences)."""
        import re

        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ---- Ollama backend (recommended) ----

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call Ollama local REST API.

        Ollama runs open-source models locally with zero configuration:
          1. Install: https://ollama.com
          2. Pull a model: ollama pull phi3:mini
          3. It auto-starts a server at localhost:11434

        Recommended models (sorted by speed):
          - phi3:mini     (3.8B, ~2GB VRAM, fast, good JSON output)
          - llama3.2:3b   (3.2B, ~2GB VRAM, very fast)
          - qwen2.5:7b    (7B, ~4GB VRAM, excellent at JSON)
          - mistral:7b    (7B, ~4GB VRAM, high quality)
        """
        try:
            import requests
        except ImportError:
            print("  [QueryParser] 'requests' package not installed")
            return None

        try:
            # Check if Ollama is running
            health = requests.get(f"{self.ollama_url}/", timeout=2)
            if health.status_code != 200:
                print(f"  [QueryParser] Ollama not responding at {self.ollama_url}")
                return None
        except requests.ConnectionError:
            print(f"  [QueryParser] Ollama not running at {self.ollama_url}")
            print(f"    Install: https://ollama.com")
            print(f"    Then: ollama pull {self.llm_model}")
            return None

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 512,
                    },
                    "format": "json",
                },
                timeout=60,
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                err = response.text[:200]
                print(f"  [QueryParser] Ollama error ({response.status_code}): {err}")
                return None

        except requests.Timeout:
            print(f"  [QueryParser] Ollama timeout (model may be loading)")
            return None
        except Exception as e:
            print(f"  [QueryParser] Ollama call failed: {e}")
            return None

    # ---- HuggingFace Transformers backend ----

    def _call_transformers(self, prompt: str) -> Optional[str]:
        """
        Call a local HuggingFace model via transformers pipeline.

        Uses small instruction-tuned models that fit alongside CLIP+GDino:
          - microsoft/Phi-3-mini-4k-instruct (3.8B, ~4GB in fp16)
          - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, ~2GB in fp16)

        The model is loaded once and cached for subsequent queries.
        """
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError:
            print("  [QueryParser] transformers not installed")
            return None

        try:
            if self._hf_pipeline is None:
                print(f"  [QueryParser] Loading HuggingFace model: {self.llm_model}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._hf_pipeline = hf_pipeline(
                    "text-generation",
                    model=self.llm_model,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                print(f"  [QueryParser] Model loaded on {device}")

            messages = [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f'Query: "{prompt.split("Query:")[1].split(chr(10))[0].strip()}"'
                 if 'Query:' in prompt else f'Query: "{prompt}"'},
            ]

            output = self._hf_pipeline(
                messages,
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                return_full_text=False,
            )
            return output[0]['generated_text']

        except Exception as e:
            print(f"  [QueryParser] Transformers call failed: {e}")
            return None

    def offload_llm(self):
        """Free GPU memory used by the HuggingFace LLM (if loaded)."""
        if self._hf_pipeline is not None:
            import torch
            del self._hf_pipeline
            self._hf_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  [QueryParser] LLM offloaded from GPU")


# Quick test
if __name__ == '__main__':
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Test QueryParser")
    _p.add_argument('--mode', default='rule_based', choices=['rule_based', 'llm'])
    _p.add_argument('--backend', default='ollama', choices=['ollama', 'transformers'])
    _p.add_argument('--model', default='phi3:mini',
                    help='Model name (ollama: phi3:mini, llama3.2:3b; '
                         'transformers: microsoft/Phi-3-mini-4k-instruct)')
    _p.add_argument('--query', default=None, help='Single query to test')
    _args = _p.parse_args()

    parser = QueryParser(
        mode=_args.mode,
        llm_backend=_args.backend,
        llm_model=_args.model,
    )

    test_queries = [_args.query] if _args.query else [
        'pan', 'tap', 'knife', 'cutting board', 'kettle',
        'sponge', 'glass', 'potato masher',
    ]

    print(f"Mode: {_args.mode}  Backend: {_args.backend}  Model: {_args.model}")

    for q in test_queries:
        plan = parser.parse(q)
        print(f"\n{'='*60}")
        print(f"Query: '{q}'")
        print(f"  Target:     {plan.target}")
        print(f"  Detection:  {plan.detection_prompt}")
        print(f"  Confusable: {plan.confusables}")
        print(f"  Retrieval:  {plan.retrieval_prompts}")
        print(f"  Context:    {plan.context_hint}")
        print(f"  Confidence: {plan.confidence}")
