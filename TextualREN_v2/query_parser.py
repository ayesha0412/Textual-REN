"""
Query Parser: Domain-aware query understanding for egocentric video localization.

Parses ambiguous text queries into structured detection plans with:
  - Disambiguated target description
  - Grounding DINO detection prompts (multi-label)
  - Confusable class lists for negative suppression
  - CLIP retrieval prompts (richer text for CLS/patch matching)

Two modes:
  - "rule_based": Fast, deterministic, no external deps.
    Uses the built-in ontology cache to resolve ambiguity.
  - "llm" (default): A local LLM generates plans for ANY query — domain-agnostic.
    The built-in ontology acts as a warm cache of pre-verified plans for common
    objects; LLM outputs are disk-cached in the same format for reproducibility.
    Set use_ontology=False to bypass the cache entirely (pure-LLM ablation).

Usage:
    parser = QueryParser(mode="rule_based")
    plan = parser.parse("pan")
    # -> QueryPlan(target="frying pan", detection_prompt="frying pan. cooking pan. saucepan",
    #              confusables=["plate", "lid", "tray", "bowl"], ...)
"""

import os
import json
import time
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
    domain: str = "general"         # Detected domain
    confidence: float = 1.0         # Parser confidence (1.0 = ontology cache, 0.9 = LLM, 0.5 = heuristic)
    source: str = "heuristic"       # Plan provenance: "ontology" | "llm" | "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


class QueryParser:
    """
    Domain-agnostic query parser for egocentric video object localization.

    Detection plans come from a local LLM (works for any object, any domain).
    The built-in ontology is a warm cache of pre-verified plans for common
    objects — a cache hit resolves ambiguity instantly:
      - "pan" -> "frying pan" (not camera pan, not washing pan)
      - "tap" -> "kitchen faucet" (not beer tap, not shoulder tap)
      - "board" -> "cutting board" (not whiteboard, not circuit board)

    LLM mode uses **open-source local models only** (no proprietary APIs):
      - HuggingFace Transformers (default): Qwen3-0.6B — only 1.2GB VRAM,
        highest tool-calling score (0.880) among sub-4B models
      - Ollama: runs models like qwen3:0.6b, phi3:mini locally via REST API
    """

    # Default HuggingFace model: Qwen3-0.6B
    # - 0.6B params, ~1.2GB VRAM in bf16, ~0.5GB in 4-bit
    # - Highest tool-calling/JSON score (0.880) among sub-4B models
    # - Fits alongside CLIP ViT-g-14 (5GB) + GDino (0.8GB) on a single GPU
    # Alternatives: "Qwen/Qwen3-1.7B", "microsoft/Phi-4-mini-instruct"
    DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"

    def __init__(self, mode: str = "rule_based", cache_dir: str = None,
                 llm_backend: str = "transformers",
                 llm_model: str = None,
                 ollama_url: str = "http://localhost:11434",
                 use_ontology: bool = True):
        """
        Args:
            mode: "rule_based" or "llm"
            cache_dir: Directory for LLM response cache (default: .query_cache/)
            llm_backend: "transformers" (HuggingFace, default) or "ollama"
            llm_model: Model name — for transformers: HF model ID
                       (default: Qwen/Qwen3-0.6B); for ollama: "qwen3:0.6b"
            ollama_url: Ollama server URL (default: http://localhost:11434)
            use_ontology: if False, bypass the built-in ontology cache so every
                       query goes through the LLM (pure-LLM ablation mode)
        """
        self.mode = mode
        self.use_ontology = use_ontology
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '.query_cache'
        )
        self.llm_backend = llm_backend
        self.llm_model = llm_model or self.DEFAULT_HF_MODEL
        self.ollama_url = ollama_url.rstrip('/')
        self._hf_model = None       # lazy-loaded
        self._hf_tokenizer = None   # lazy-loaded

    def parse(self, text_query: str) -> QueryPlan:
        """
        Parse a text query into a structured detection plan.

        Always tries rule-based first (fast, deterministic).
        If mode="llm" and rule-based has low confidence, calls LLM.
        """
        # Rule-based first: an ontology cache hit is instant and pre-verified
        plan = self._rule_based_parse(text_query)

        # LLM mode: generate a plan for anything the cache didn't cover
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
            # NOTE: no 'wok' here — a wok IS an acceptable answer to "pan"
            # (co-hyponym, not a confusable). Confusable = an object the
            # user would NOT accept as the answer.
            'confusables': ['plate', 'pot lid', 'tray', 'bowl',
                            'rice cooker', 'pot'],
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

        'strainer': {
            'target': 'mesh strainer',
            'detection_prompt': 'mesh strainer. kitchen sieve with handle. wire mesh sieve',
            'confusables': ['drain cover', 'sink plug', 'stove burner', 'hob ring', 'lid', 'pan'],
            'retrieval_prompts': [
                'a mesh strainer', 'a kitchen sieve with handle',
                'a wire strainer near sink', 'a metal mesh sieve'
            ],
            'context': 'metal mesh bowl with handle for draining liquids from food',
        },
        'sieve': {
            'target': 'mesh strainer',
            'detection_prompt': 'mesh strainer. kitchen sieve with handle. wire mesh sieve',
            'confusables': ['drain cover', 'sink plug', 'stove burner', 'hob ring', 'lid', 'pan'],
            'retrieval_prompts': [
                'a mesh sieve', 'a kitchen sieve with handle',
                'a wire strainer near sink'
            ],
            'context': 'metal mesh bowl with handle for draining liquids from food',
        },
        'cooker': {
            'target': 'cooker',
            'detection_prompt': 'stove top. cooker. gas hob. electric hob. cooking range',
            'confusables': ['oven', 'microwave', 'hot plate', 'toaster'],
            'retrieval_prompts': [
                'a kitchen cooker', 'a stove top with burners', 'a gas hob'
            ],
            'context': 'appliance with burners/hobs for cooking on top',
        },
        'can opener': {
            'target': 'can opener',
            'detection_prompt': 'can opener. tin opener. manual can opener',
            'confusables': ['bottle opener', 'corkscrew', 'peeler', 'scissors'],
            'retrieval_prompts': [
                'a can opener', 'a tin opener', 'a manual can opener on counter'
            ],
            'context': 'handheld tool with cutting wheel for opening tin cans',
        },
        'colander': {
            'target': 'colander',
            'detection_prompt': 'colander. pasta strainer. large strainer. draining bowl',
            'confusables': ['bowl', 'pot', 'strainer', 'sieve', 'lid'],
            'retrieval_prompts': [
                'a colander', 'a pasta colander', 'a colander with holes'
            ],
            'context': 'large bowl-shaped vessel with holes for draining pasta or vegetables',
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
        """Parse query via the ontology cache (if enabled), else heuristic."""
        q = text_query.lower().strip()

        # Direct match in ontology cache
        if self.use_ontology and q in self._KITCHEN_ONTOLOGY:
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
                source='ontology',
            )

        # Multi-word query: check if any word matches the cache
        words = q.split()
        for word in reversed(words):  # last word is often the object
            if self.use_ontology and word in self._KITCHEN_ONTOLOGY:
                entry = self._KITCHEN_ONTOLOGY[word]
                # Referring-expression pass-through: full query stays first
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
                    source='ontology',
                )

        # Heuristic: generate confusables from category co-membership
        confusables = self._heuristic_confusables(q)
        return QueryPlan(
            original_query=text_query,
            target=text_query,
            detection_prompt=text_query,
            confusables=confusables,
            retrieval_prompts=[f"a {text_query}", f"a photo of a {text_query}"],
            context_hint=f"object described as '{text_query}'",
            domain='general',
            confidence=0.5,
            source='heuristic',
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

    def export_ontology_cache(self, out_dir: str = None) -> int:
        """
        Serialize every ontology entry into the unified plan-cache format.

        Makes the "ontology = pre-verified warm cache of detection plans"
        framing literal: each entry becomes the same JSON an LLM plan would
        produce, tagged source="ontology". Files are prefixed "ontology_" so
        they never shadow runtime LLM cache lookups.

        Returns the number of files written.
        """
        out_dir = out_dir or self.cache_dir
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for query, entry in self._KITCHEN_ONTOLOGY.items():
            data = {
                'query': query,
                'target': entry['target'],
                'detection_prompt': entry['detection_prompt'],
                'confusables': entry['confusables'],
                'retrieval_prompts': entry['retrieval_prompts'],
                'context_hint': entry['context'],
                'source': 'ontology',
                'model': None,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            key = hashlib.md5(query.lower().strip().encode()).hexdigest()
            with open(os.path.join(out_dir, f"ontology_{key}.json"), 'w') as f:
                json.dump(data, f, indent=2)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # LLM-enhanced parser (open-source models only)                        #
    # ------------------------------------------------------------------ #

    _LLM_SYSTEM_PROMPT = """You parse object queries into structured JSON for a visual localization system.

Output JSON with these 5 fields:
- "target": disambiguated object name (the concrete physical object the query refers to). Keep the queried object itself — query "zebra" means the animal zebra. NEVER substitute a scene, place, or broader category (not "zoo", not "wildlife").
- "detection_prompt": exactly 3 different synonyms/names for the object, period-separated. Each must be a DIFFERENT phrase. Use the object's common names, alternative names, or descriptive names.
- "confusables": list of 3-5 real objects with similar visual appearance (shape, color, size) that could plausibly appear in the same scene. A confusable must be an object the user would NOT accept as the answer — never a synonym, sub-type, or variant of the target.
- "retrieval_prompts": list of 3 short visual descriptions starting with "a" or "the"
- "context_hint": one sentence describing what the object looks like and where it typically appears

Important: detection_prompt must have 3 UNIQUE labels. confusables must be real objects.
Return ONLY valid JSON."""

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

            # Cache result for reproducibility — unified plan-cache format
            # (same fields whether the plan came from ontology or LLM)
            data['query'] = text_query
            data['source'] = 'llm'
            data['model'] = self.llm_model
            data['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            return self._plan_from_dict(text_query, data)

        except Exception as e:
            print(f"  [QueryParser] Failed to parse LLM output: {e}")
            return None

    def _plan_from_dict(self, text_query: str, data: dict) -> QueryPlan:
        """Create a QueryPlan from a parsed JSON dict with quality cleanup."""
        # Normalize keys: small LLMs occasionally misspell field names
        # (observed: "retrieival_prompts" from Qwen3-0.6B) — fuzzy-match
        # unknown keys onto the expected schema so no field is silently lost
        import difflib
        _expected = ['target', 'detection_prompt', 'confusables',
                     'retrieval_prompts', 'context_hint']
        for k in list(data.keys()):
            if k not in _expected:
                match = difflib.get_close_matches(k, _expected, n=1, cutoff=0.8)
                if match and match[0] not in data:
                    data[match[0]] = data[k]

        # Clean up detection_prompt: deduplicate labels
        det = data.get('detection_prompt', text_query)
        if isinstance(det, list):
            det = ". ".join(det)
        # Split, deduplicate, rejoin.  The full original query is always kept
        # as the FIRST label — referring expressions ("red mug on the shelf")
        # carry attributes/relations that Grounding DINO can ground directly.
        labels = [l.strip().strip('.') for l in det.split('.') if l.strip()]
        labels = [text_query.strip()] + labels
        seen = set()
        unique_labels = []
        for l in labels:
            lk = l.lower()
            if lk not in seen and lk:
                seen.add(lk)
                unique_labels.append(l)
        det = ". ".join(unique_labels) if unique_labels else text_query

        # Clean up confusables: deduplicate, remove target name
        raw_conf = data.get('confusables', [])
        if isinstance(raw_conf, list):
            target_lower = data.get('target', text_query).lower()
            query_lower = text_query.lower()
            seen_conf = set()
            clean_conf = []
            for c in raw_conf:
                if isinstance(c, str):
                    ck = c.lower().strip()
                    # Skip if it's the target itself, or already seen
                    if ck and ck not in seen_conf and ck != target_lower and ck != query_lower:
                        seen_conf.add(ck)
                        clean_conf.append(c.strip())
            raw_conf = clean_conf

        # Retrieval prompts: make sure the verbatim query is represented so
        # CLIP retrieval sees the attributes of a referring expression too
        retr = data.get('retrieval_prompts', [])
        if not isinstance(retr, list):
            retr = [str(retr)]
        retr = [str(r).strip() for r in retr if str(r).strip()]
        ql = text_query.lower().strip()
        if not any(ql in r.lower() for r in retr):
            retr = [f"a {text_query}"] + retr

        return QueryPlan(
            original_query=text_query,
            target=data.get('target', text_query),
            detection_prompt=det,
            confusables=raw_conf,
            retrieval_prompts=retr,
            context_hint=data.get('context_hint', ''),
            domain='general',
            confidence=0.9,
            source='llm',
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

    # Fallback model if transformers is too old for Qwen3
    _FALLBACK_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def _load_hf_model(self):
        """
        Lazy-load the HuggingFace model and tokenizer.

        Default: Qwen/Qwen3-0.6B — 0.6B params, ~1.2GB VRAM in bf16.
        Fits alongside CLIP ViT-g-14 (5GB) + Grounding DINO (0.8GB).

        Fallback: Qwen/Qwen2.5-0.5B-Instruct if transformers < 4.51
        (Qwen3 architecture not yet supported).
        """
        if self._hf_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.llm_model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # transformers 4.x expects torch_dtype=; 5.x renamed it to dtype=
        import transformers as _transformers
        _tf_major = int(_transformers.__version__.split('.')[0])
        dtype_kw = ({'dtype': torch.bfloat16} if _tf_major >= 5
                    else {'torch_dtype': torch.bfloat16})

        # Check if transformers supports the requested model
        try:
            print(f"  [QueryParser] Loading {model_id} ...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                **dtype_kw,
            )
        except (ValueError, KeyError) as e:
            if "does not recognize this architecture" in str(e) or "qwen3" in str(e).lower():
                import transformers
                print(f"  [QueryParser] transformers {transformers.__version__} "
                      f"does not support {model_id}.")
                print(f"  [QueryParser] Upgrade: pip install --upgrade transformers")
                print(f"  [QueryParser] Falling back to {self._FALLBACK_HF_MODEL}")
                model_id = self._FALLBACK_HF_MODEL
                self.llm_model = model_id
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    **dtype_kw,
                )
            else:
                raise

        self._hf_model.eval()

        # Report VRAM usage
        param_bytes = sum(p.numel() * p.element_size() for p in self._hf_model.parameters())
        print(f"  [QueryParser] Loaded {model_id} on {device} "
              f"({param_bytes / 1024**2:.0f} MB, "
              f"{sum(p.numel() for p in self._hf_model.parameters()) / 1e6:.0f}M params)")

    def _call_transformers(self, prompt: str) -> Optional[str]:
        """
        Generate structured JSON using a local HuggingFace model.

        Default model: Qwen/Qwen3-0.6B
          - 0.6B params, ~1.2GB VRAM (bf16)
          - Tool-calling score: 0.880 (highest among sub-4B models)
          - Native JSON/tool-calling support via Qwen3 chat template

        Alternatives (set via llm_model):
          - Qwen/Qwen3-1.7B   (~3.4GB VRAM, better quality but heavier)
          - microsoft/Phi-4-mini-instruct  (~7.6GB VRAM, best reasoning)
        """
        try:
            import torch
        except ImportError:
            print("  [QueryParser] torch not installed")
            return None

        try:
            self._load_hf_model()

            # Build chat messages
            messages = [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f'Query: "{prompt.split("Query:")[1].split(chr(10))[0].strip().strip(chr(34))}"'
                 if 'Query:' in prompt else f'Query: "{prompt}"'},
            ]

            # Apply chat template
            # Qwen3 supports enable_thinking=False for fast JSON;
            # Qwen2.5 and others don't have this param — use try/except
            try:
                text = self._hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Qwen3: skip thinking for fast JSON
                )
            except TypeError:
                text = self._hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self._hf_tokenizer(text, return_tensors="pt").to(
                self._hf_model.device
            )

            with torch.no_grad():
                output_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.3,  # penalize repeated phrases
                )

            # Decode only the new tokens (skip the prompt)
            new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            print(f"  [QueryParser] Transformers call failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def offload_llm(self):
        """Free GPU memory used by the HuggingFace LLM (if loaded)."""
        if self._hf_model is not None:
            import torch
            del self._hf_model
            del self._hf_tokenizer
            self._hf_model = None
            self._hf_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  [QueryParser] LLM offloaded from GPU")


# Quick test
if __name__ == '__main__':
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Test QueryParser")
    _p.add_argument('--mode', default='llm', choices=['rule_based', 'llm'])
    _p.add_argument('--backend', default='transformers', choices=['ollama', 'transformers'])
    _p.add_argument('--model', default=None,
                    help='Model name (transformers default: Qwen/Qwen3-0.6B; '
                         'ollama: qwen3:0.6b, phi3:mini)')
    _p.add_argument('--query', default=None, help='Single query to test')
    _p.add_argument('--no-ontology', action='store_true',
                    help='Bypass the ontology cache — every query goes through '
                         'the LLM (pure-LLM ablation mode)')
    _p.add_argument('--export-cache', action='store_true',
                    help='Write all ontology entries to the plan cache and exit')
    _args = _p.parse_args()

    parser = QueryParser(
        mode=_args.mode,
        llm_backend=_args.backend,
        llm_model=_args.model,
        use_ontology=not _args.no_ontology,
    )

    if _args.export_cache:
        n = parser.export_ontology_cache()
        print(f"Exported {n} ontology entries to {parser.cache_dir}")
        raise SystemExit(0)

    test_queries = [_args.query] if _args.query else [
        'pan', 'tap', 'knife', 'cutting board', 'kettle',
        'sponge', 'glass', 'potato masher',
    ]

    print(f"Mode: {_args.mode}  Backend: {_args.backend}  "
          f"Model: {_args.model or QueryParser.DEFAULT_HF_MODEL}  "
          f"Ontology: {'ON' if not _args.no_ontology else 'OFF (pure LLM)'}")

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
        print(f"  Source:     {plan.source}")
