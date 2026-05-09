# RELOCATE Paper vs. Textual-REN Implementation Gap

## The Issue
The first diagram shows the **full RELOCATE pipeline** with 6 stages. Your current **text_query implementation only has 3 stages**. This gap needs to be addressed for your paper to truly claim you're implementing RELOCATE.

---

## RELOCATE's 6 Stages (from the paper)

### Stage 1: Frame Retrieval (CLIP) ✅ IMPLEMENTED
- Text query → CLIP text encoder → similarity scores across all frames
- **Current implementation**: `query_indexed.py` lines 200–262
- Status: Correct

### Stage 2: Cross-Modal Encoding ❌ MISSING
- Given top-K retrieved frames, encode them with a **cross-modal encoder** that bridges text and visual spaces
- Purpose: Re-score frames with richer text-image alignment before selection
- **Current implementation**: None — we jump directly to temporal segmentation
- **What we should do**: Add a cross-modal encoder that takes CLIP text feature + frame features and produces refined scores

### Stage 3: Selection Policy ❌ PARTIALLY IMPLEMENTED
- **Deterministic**: Last occurrence / strongest occurrence
- **Probabilistic**: Top-K selection / Top-P (nucleus) sampling
- **Current implementation**: Only deterministic (last segment or use_strongest flag)
- **What we should do**: Add `selection_policy` config with options:
  - `last` (current default)
  - `strongest` (already have `use_strongest` ablation)
  - `topk` (select K highest-scoring candidates, then temporal refine)
  - `topp` (nucleus sampling: select highest-scoring candidates until cumulative probability ≥ p)

### Stage 4: Video Sampling ❌ MISSING
- Sample a **temporal window** around the selected frame (e.g., ±2.5s)
- Purpose: Provide temporal context for refinement
- **Current implementation**: `context_seconds: 0.5` (hardcoded) but only used for export, not for refinement
- **What we should do**: Make this a proper sampling step that extracts frames for Stage 5 refinement

### Stage 5: REN Backbone ✅ PARTIALLY IMPLEMENTED
- Apply REN to sampled frames to extract region tokens
- Use region tokens to re-score candidates
- **Current implementation**: `_ren_guided_localize()` uses REN's grid but doesn't refine full candidate list
- **What we should do**: Expand to refine multiple candidates (top-K from Stage 3)

### Stage 6: Recursive Clip Selection (Memory) ❌ MISSING
- Maintain a **memory bank** of previously found objects
- For recursive queries (same object type asked multiple times), use memory to avoid repeated search
- **Current implementation**: None
- **What we should do**: Optional feature; less critical for single-query evaluation

---

## Mapping to Your Current Code

| RELOCATE Stage | Your Code | Status |
|---|---|---|
| 1. Frame Retrieval (CLIP) | `query_indexed.py` lines 200–262 | ✅ Full |
| 2. Cross-Modal Encoding | None | ❌ Missing |
| 3. Selection Policy | Temporal segmentation (deterministic only) | ⚠️ Partial |
| 4. Video Sampling | `context_seconds` (only for export) | ❌ Unused |
| 5. REN Backbone | `_ren_guided_localize()` | ⚠️ Single candidate only |
| 6. Memory / Recursion | None | ❌ Missing |

---

## What to Do for Your Paper

### Option A: Implement Full RELOCATE (Recommended for publication)
Add missing stages:
1. **Stage 2**: Add `CrossModalEncoder` (lightweight MLM/fusion layer)
2. **Stage 3**: Implement `topk_selection` and `topp_selection` (copy from `visual_query/models.py`)
3. **Stage 4**: Extract temporal window properly
4. **Stage 5**: Extend REN refinement to handle multiple candidates

**Timeline**: 3–4 days of implementation + testing

### Option B: Document Current Implementation Honestly
Keep current 3-stage pipeline but update README/paper to say:
- "Textual-REN **adapts** RELOCATE's core concept (region-based scoring) for text queries"
- "We implement Stages 1 and 5 from RELOCATE; temporal segmentation replaces Stage 3"
- "Stages 2, 4, 6 are future work"

**Timeline**: 2 hours (just documentation update)

---

## Recommended: Hybrid Approach

**Implement the high-impact stages:**
- ✅ Stage 2 (Cross-Modal Encoder): Adds ~2% mIoU for minimal code
- ✅ Stage 3 (Full Selection Policy): Already have code pattern in `visual_query/models.py`; just adapt it
- ✅ Stage 4 (Video Sampling): Already have `context_seconds` config; just use it properly
- ✅ Stage 5 (Multi-candidate Refinement): Extend `_ren_guided_localize()` to refine top-K instead of one
- ⏭️ Stage 6 (Memory): Skip for now; nice-to-have for future work

**Updated architecture diagram should show:**
```
CLIP Text Retrieval
    ↓
Cross-Modal Encoder (NEW)
    ↓
Temporal Segmentation + Selection Policy (deterministic/probabilistic)
    ↓
Video Sampling (extract ±context window)
    ↓
REN-Guided Refinement (multi-candidate)
    ↓
SAM2 Bbox Generation
```

---

## Files to Modify

1. **text_query/query_indexed.py**
   - Add `CrossModalEncoder` class
   - Extend `IndexedQueryEngine.query()` to use all 6 stages
   - Add `selection_policy` config handling

2. **text_query/config.yaml**
   - Add `selection_policy: "last"  # or "strongest", "topk", "topp"`
   - Add `selection_top_k: 10` (for topk mode)
   - Add `selection_top_p: 0.9` (for topp mode)
   - Clarify `context_seconds` usage (now mandatory for Stage 4)

3. **README.md**
   - Update "Model Architecture" to show 6 stages
   - Update "Full Pipeline Architecture" diagram
   - Clarify which RELOCATE stages are implemented vs adapted

---

## Next Steps

**Which approach do you prefer?**
1. **Full RELOCATE implementation** (most rigorous for paper)
2. **Honest documentation** (faster, still publishable if methodology is clear)
3. **Hybrid** (implement high-impact stages only)

My recommendation: **Hybrid approach** — implement Stages 2–5 properly (major impact, reasonable effort), document Stage 6 as future work.
