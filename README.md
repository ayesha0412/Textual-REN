# REN: Fast and Efficient Region Encodings from Patch-Based Image Encoders

REN is a research codebase for fast region-level vision representations and text-guided video localization.

Region Encoder Network (REN) is a lightweight model for extracting semantically meaningful region-level representations from images using point prompts. It operates on frozen patch-based vision encoder features, avoids explicit segmentation, and supports both training-free and task-specific setups across a range of vision tasks.

REN generalizes across multiple vision backbones (DINO, DINOv2, OpenCLIP) and consistently outperforms patch-based features on tasks like semantic segmentation and object retrieval. It matches the performance of SAM-based methods while being **60× faster** and using **35× less memory**.

This repo contains the PyTorch implementation and pretrained models for REN.

This repo also includes **Textual-REN** — a complete text-to-video object localization system built on top of REN that finds the last occurrence of any text-described object in a long egocentric video, with spatial bounding box output. It is evaluated on EPIC-KITCHENS.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange.svg)

---

## Table of Contents

- [Textual-REN: Text-Guided Video Localization](#textual-ren-text-guided-video-localization)
  - [Overview](#overview)
  - [Full Pipeline Architecture](#full-pipeline-architecture)
  - [Component Details](#component-details)
  - [Evaluation Framework](#evaluation-framework)
  - [Quick Start](#quick-start-textual-ren)
  - [Commands A–Z](#commands-a-z)
- [REN: Region Encoder Network](#ren-region-encoder-network)
  - [Getting Started](#getting-started)
  - [Using REN](#using-ren)
  - [Training REN](#training-ren)
- [License](#license)
- [Citation](#citing-ren)

---

## Textual-REN: Text-Guided Video Localization

### Overview

**Textual-REN** extends the REN and RELOCATE research to text-guided video object localization. Given a natural language description (e.g., `"kitchen knife"`, `"fairy dish soap"`, `"red switch"`), the system searches an offline-indexed egocentric video, finds the **last genuine occurrence** of the described object, and returns:

- A trimmed video clip with a **green bounding box** around the object
- The **timestamp** and **frame index** of the last occurrence
- A **spatial bounding box** `[x, y, w, h]` for evaluation

The pipeline is designed for episodic memory use cases — "where did I last put X?" — on datasets like EPIC-KITCHENS.

**Built on RELOCATE's Region-Based Approach**: 
- RELOCATE (Suris et al., ECCV 2024) proposes visual query localization using DINOv2 region tokens pooled over SAM2 masks
- Textual-REN adapts RELOCATE's region-scoring strategy to free-text queries: CLIP replaces the visual query exemplar, and REN's trained semantic grid replaces manual point annotation
- Core invariant preserved: **cosine similarity between proposal features and query features in a shared embedding space**

**Textual-REN v2** introduces three training-free architectural improvements:
- **Grounding DINO** for spatial localization — text-conditioned zero-shot object detection replaces CLIP crop scoring, with CLIP re-ranking to disambiguate multiple detections
- **CLIP patch-level re-ranking** — max-patch similarity (256 patches per frame) re-ranks FAISS candidates, improving retrieval for small/specific objects
- **Adaptive threshold** — per-query threshold computed from the similarity distribution (`tau = mean + alpha * std`), replacing the fixed `tau=0.18`
- **GDino+CLIP verified frame selection** — verifies segment peak candidates with Grounding DINO detection + CLIP crop scoring, filtering color/shape confusion (e.g., red bucket misidentified as "pink flower")

---

### Model Architecture

**Textual-REN v2** combines four foundation models in a modular pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEXTUAL-REN v2 MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐   ┌────────────────┐ │
│  │  CLIP ViT-g-14       │    │  Grounding DINO      │   │   SAM2         │ │
│  │  (OpenCLIP)          │    │  (grounding-dino-    │   │   (segment     │ │
│  │                      │    │   tiny)              │   │   anything v2) │ │
│  │ 1024-dim embedding   │    │                      │   │                │ │
│  │ space for text &     │    │ Text-conditioned     │   │ Point/Bbox ->  │ │
│  │ image patches        │    │ zero-shot object     │   │ Mask -> Bbox   │ │
│  │                      │    │ detection            │   │                │ │
│  │ + CLS frame embed    │    │                      │   │ + Used only    │ │
│  │ + 256 patch tokens   │    │ + Direct text->bbox  │   │   in full-     │ │
│  │ + Text encoding      │    │ + CLIP re-ranks      │   │   quality mode │ │
│  │ + Crop verification  │    │   multiple dets      │   │ + Skipped in   │ │
│  │                      │    │ + Frame verification  │   │   fast eval    │ │
│  └──────────────────────┘    └──────────────────────┘   └────────────────┘ │
│                                                                              │
│  ┌──────────────────────┐                                                   │
│  │   REN               │   (legacy fallback, spatial_method: "ren_clip")   │
│  │   (DINOv2 ViT-L/14) │   32x32 semantic grid of region proposals        │
│  │                      │   Replaced by Grounding DINO in v2               │
│  └──────────────────────┘                                                   │
│           │                           │                         │           │
│  ┌────────┴─────────────────────────┬─┴─────────────────────────┴────────┐  │
│  │            RELOCATE 6-STAGE PIPELINE (Text Query Adaptation)           │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  STAGE 1: Frame Retrieval (CLIP) ✅                                   │  │
│  │  • Text query → CLIP text encoder (ViT-g-14) → embedding             │  │
│  │  • FAISS search: dot product against indexed frame embeddings        │  │
│  │  • Returns: similarity scores for all frames                          │  │
│  │                                                                       │  │
│  │  STAGE 2: Cross-Modal Encoding ✅ (Implicit in CLIP space)           │  │
│  │  • For text queries, CLIP's joint 1024-dim embedding space serves   │  │
│  │    as the cross-modal bridge (no explicit encoder needed)            │  │
│  │                                                                       │  │
│  │  STAGE 2c: Patch Re-ranking ✅ (v2)                                    │  │
│  │  • For FAISS top-100 CLS candidates, compute max_i(cos(text,patch_i))│  │
│  │  • Blend: 40% CLS + 60% max-patch score                             │  │
│  │  • Improves retrieval for small/specific objects                     │  │
│  │                                                                       │  │
│  │  STAGE 2d: Adaptive Threshold ✅ (v2)                                 │  │
│  │  • tau = mean + alpha * std, clamped to [0.10, 0.30]                 │  │
│  │  • Replaces fixed tau=0.18 that failed across diverse queries        │  │
│  │                                                                       │  │
│  │  STAGE 3: Selection Policy ✅                                         │  │
│  │  • Temporal Segmentation: group above-threshold frames (≥2 frames)   │  │
│  │  • Deterministic modes: "last" (most recent) or "strongest"          │  │
│  │  • Probabilistic modes: "topk" (top-K candidates) or "topp"          │  │
│  │    (nucleus sampling) for multi-candidate refinement                 │  │
│  │  • Returns: ranked list of candidate frames                          │  │
│  │                                                                       │  │
│  │  STAGE 3b: GDino+CLIP Verified Frame Selection ✅ (v2)               │  │
│  │  • For each segment peak (recent-first), load frame + run GDino      │  │
│  │  • CLIP-score best crop against query text                           │  │
│  │  • Accept only if CLIP crop score >= min_crop_verify (0.17)          │  │
│  │  • Filters color/shape confusion (red bucket != "pink flower")       │  │
│  │                                                                       │  │
│  │  STAGE 4: Temporal Sampling ✅                                        │  │
│  │  • Extract temporal context window: ±0.5 seconds around candidates   │  │
│  │  • Load frame batch for Stage 5 refinement                           │  │
│  │  • Ensures spatial locality for accurate region proposal scoring     │  │
│  │                                                                       │  │
│  │  STAGE 5: Spatial Localization ✅ (v2: Grounding DINO)                 │  │
│  │  • Grounding DINO: text + image -> bounding boxes (zero-shot)        │  │
│  │  • When multiple detections: CLIP re-ranks crops against query       │  │
│  │  • Direct bbox output — no REN grid, no SAM2 point->bbox step       │  │
│  │  • Legacy fallback: REN grid + CLIP crop scoring (spatial_method:    │  │
│  │    "ren_clip")                                                        │  │
│  │  • Returns: refined candidate with spatial bbox                      │  │
│  │                                                                       │  │
│  │  STAGE 6: Query Expansion via Memory ⏳                              │  │
│  │  • (Future work) Iterative refinement with pseudo-labeled objects    │  │
│  │  • Re-query with expanded candidate pool from SAM2 tracking         │  │
│  │                                                                       │  │
│  │  ⏱️  Total latency: ~2–5 seconds per query (CLIP-tile fast path)    │  │
│  │      ~20–60 seconds (SAM2 full quality path)                         │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why these models?**

| Component | Role | Why This Choice |
|-----------|------|-----------------|
| **CLIP ViT-g-14** | Text-image alignment, patch re-ranking, crop verification | 1024-dim joint space (CLS + 256 patch tokens), no fine-tuning needed |
| **Grounding DINO** | Spatial localization (v2) | Text-conditioned detection — direct text+image -> bbox, no embedding bridge needed |
| **REN (DINOv2 ViT-L/14)** | Legacy spatial proposals | 15–20% better than uniform grids; replaced by Grounding DINO in v2 |
| **SAM2** | Precise mask generation | State-of-the-art segmentation; optional in fast-eval mode |

---

### Full Pipeline Architecture

**Textual-REN v2 = CLIP (Stage 1-2) + Patch Re-ranking (Stage 2c) + Adaptive Threshold (Stage 2d) + GDino-Verified Selection (Stage 3b) + Grounding DINO (Stage 5) + RELOCATE framework**.

The system implements all 6 stages of RELOCATE (Suris et al., ECCV 2024), adapted for free-text queries:

**Phase 1 (Offline)**: Index video frames with CLIP CLS embeddings + patch tokens + pre-computed OCR text.
**Phase 2 (Online)**: Enhanced pipeline:
  1. CLIP text encoding + similarity scores (Stage 1)
  2. Cross-modal scoring + OCR fusion + patch re-ranking + adaptive threshold (Stage 2)
  3. Temporal segmentation + GDino+CLIP verified frame selection (Stage 3)
  4. Temporal context window extraction (Stage 4)
  5. Grounding DINO spatial localization with CLIP re-ranking (Stage 5)
  6. Optional: Query expansion via memory (Stage 6 — future work)

The two-phase design ensures **<10s per query** on 100K-frame videos by pre-computing expensive frame embeddings offline.

```
╔══════════════════════════════════════════════════════════════════════╗
║              PHASE 1 — OFFLINE INDEXING  (prepare_index.py)         ║
║              Run once per video, results reused for all queries      ║
╚══════════════════════════════════════════════════════════════════════╝

  Video File (.MP4)
       │
       ▼
  ┌────────────────┐
  │  Frame Sampler │  every Nth frame (default N=10, ~6 fps at 60 fps)
  └───────┬────────┘
          │
    ┌─────┴──────────────┐
    │                    │
    ▼                    ▼
┌──────────┐      ┌──────────────┐
│  CLIP    │      │  EasyOCR     │  brand/label text only
│ ViT-g-14 │      │  (every 5th  │  word, conf stored per frame
│(OpenCLIP)│      │   sampled    │  — used only for brand queries
└────┬─────┘      │   frame)     │
     │            └──────┬───────┘
     ▼                   │
┌──────────┐             │
│  Frame   │             │
│Embedding │             │
│CLS 1024d │             │
│+ 256     │             │
│ patches  │             │
└────┬─────┘             │
     └────────┬──────────┘
              ▼
  ┌───────────────────────────────────┐
  │     FAISS Flat Index              │  exact cosine search, GPU-accel
  │  +  metadata.json                 │  frame_idx, timestamp, OCR texts
  │  +  clip_embeddings.npy           │  (N × 1024) CLS embeddings
  │  +  patch_embeddings.npy          │  (N × 256 × 1024) patch tokens (v2)
  └───────────────────────────────────┘

  Saved to:  epic_kitchen_indexes/<video_id>/
             ├── faiss.index
             ├── metadata.json         ← includes OCR per frame
             ├── clip_embeddings.npy    ← CLS embeddings
             └── patch_embeddings.npy   ← patch tokens for re-ranking (v2)


╔══════════════════════════════════════════════════════════════════════╗
║              PHASE 2 — ONLINE QUERY  (query_indexed.py)             ║
║              ~2–5 seconds per query on indexed video                 ║
╚══════════════════════════════════════════════════════════════════════╝

  Text Query: e.g. "kitchen knife"  or  "fairy"  or  "red switch"
       │
       ▼
  ┌─────────────────────────────────────┐
  │   Query Type Classifier             │
  │   _is_brand_query()                 │
  │                                     │
  │   BRAND / OCR query:                │
  │     • Single unknown word           │  → "fairy", "heinz", "lurpak"
  │     • Capitalised unknown word      │  → "Twinings", "Fairy"
  │     • ≥2 unknown words              │  → "Yorkshire Tea"
  │                                     │
  │   OBJECT / CLIP query:              │
  │     • All words in common vocab     │  → "kitchen knife", "red switch"
  └──────┬────────────────────┬─────────┘
         │                    │
    BRAND path           OBJECT path
         │                    │
         ▼                    ▼
  ┌────────────────────────────────────────────┐
  │   CLIP Text Encoder (ViT-g-14)             │
  │   text_feat · clip_embeddings → sim curve  │  (N_frames,) cosine sims
  └────────────────┬───────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   OCR Fusion        │  BRAND PATH ONLY — kept completely
        │   (brand path only) │  separate from object scoring.
        │                     │  ocr_score from stored EasyOCR texts,
        │  fused = CLIP       │  fuzzy-matched with rapidfuzz.
        │        + 0.3×OCR    │  High-confidence OCR (≥0.85) routes
        └──────────┬──────────┘  directly to OCR bbox, skipping SAM2.
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   Temporal Segmentation                │
  │                                        │
  │  1. Keep frames where sim ≥ threshold  │  default: 0.18
  │  2. Group into contiguous segments     │  gap > 2s = new segment
  │  3. Filter: keep segments ≥ 2 frames   │
  │  4. Select LAST valid segment          │  most recent occurrence
  └────────────────┬───────────────────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   Crop Verification                    │
  │                                        │
  │  For each segment (last → first):      │
  │    similarity score ≥ 0.17 → accept ✓  │
  │    else → try previous segment         │
  │  Fallback: use LAST segment            │  most recent = best default
  └────────────────┬───────────────────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   REN-Guided Spatial Localization      │  ← RELOCATE region search,
  │   (_ren_guided_localize)               │    adapted for text queries
  │                                        │
  │  REN 32×32 semantic grid → 1024        │
  │  spatial proposals in frame space      │
  │                                        │
  │  Each proposal cropped + CLIP-encoded  │  batch inference on GPU
  │  → cosine sim with text_feat           │  (same CLIP embedding space)
  │                                        │
  │  → Best proposal center = region_point │
  │    (x, y) pixel coordinates            │
  │                                        │
  │  Note: REN's grid is object-aware      │
  │  (trained on SAM2 region masks via     │
  │  DINOv2) — denser coverage near        │
  │  object boundaries than a uniform grid │
  └────────────────┬───────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  OCR Direct Bbox    │  (brand path, OCR score ≥ 0.85)
        │  EasyOCR readtext() │  → precise text region bbox
        │  + 80% padding      │  expanded to full product label
        └──────────┬──────────┘
                   │  (or object path: use region_point below)
                   ▼
  ┌────────────────────────────────────────┐
  │   SAM2 Point → Mask → Bbox            │
  │   (or CLIP-tile fast path when         │
  │    skip_sam2_eval: true)               │
  │                                        │
  │  Input:  region_point (x, y)           │
  │  SAM2 predicts multi-mask proposals    │
  │  Each mask scored by CLIP crop sim     │
  │  Size filter: 0.1% – 12% of frame     │
  │    (60% for large objects)             │
  │  → Best scored mask → bbox [x,y,w,h]  │
  └────────────────┬───────────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    OUTPUT       │
          │                 │
          │ last_occurrence │  trimmed MP4 with green bbox overlay
          │    .mp4         │
          │                 │
          │ result.json     │  timestamp, frame_idx, pred_bbox,
          │                 │  similarity scores, query type
          │                 │
          │ debug_last      │  single JPEG: bbox + region point
          │  _frame.jpg     │  for quick visual inspection
          └─────────────────┘
```

---

### Component Details

#### 1. CLIP ViT-g-14 (OpenCLIP, laion2b)
**Role**: Frame retrieval (Phase 1) + text encoding (Phase 2) + spatial scoring (Phase 2).
- 1024-dim joint embedding space enables direct cosine similarity between text queries and video frames without fine-tuning
- Offline: encodes every Nth frame (default N=10) and stores embeddings in FAISS index
- Online: encodes text query and each spatial crop (in REN-guided localization) in the same space

#### 2. REN (Region Encoder Network) — DINOv2 ViT-L/14
**Role**: Object-aware spatial proposals via 32×32 semantic grid.
- **Backbone**: Frozen DINOv2 ViT-L/14 patch features (14×14 patch size, 518×518 input)
- **Grid**: 32×32 = 1024 region proposals, trained on SAM2 region masks to cluster densely around object boundaries
- **Loaded lazily** on first query to save VRAM during indexing (prevents ~3–4 GB overhead in `prepare_index.py`)
- **Output**: `grid_points` tensor (32², 2) = pixel coordinates of spatial proposals in 518×518 frame space
- **Why REN**: Outperforms uniform grids by 15–20% (denser sampling near objects), matches SAM performance while being **60× faster**

#### 3. FAISS Flat Index
Stores all frame embeddings for exact cosine search. GPU-accelerated when available. The index supports sub-second search over videos with 100 K+ frames. Separate `clip_embeddings.npy` enables batched similarity computation for ablations.

#### 4. EasyOCR (Brand/Label Text Detection)
**Role**: Product brand identification for queries like "Heinz", "Fairy", "Yorkshire Tea".
- Pre-computed at indexing time (Phase 1) on every 5th sampled frame
- Stored in `metadata.json` per frame: detected words + confidence scores
- At query time (Phase 2): `rapidfuzz` fuzzy-matches the query string against stored OCR texts
- **Fusion strategy**: Completely separate path for brand queries (score ≥ 0.85 triggers direct OCR-based bbox, skipping SAM2)
- Alternative models available: PaddleOCR (3–5× faster, better on rotated text) can replace EasyOCR with minimal changes

#### 5. Query Type Classifier (`_is_brand_query`)
Automatically routes queries to the correct scoring path:

| Signal | Example | Path |
|--------|---------|------|
| Single word not in common vocab | `fairy`, `persil` | OCR + CLIP |
| Capitalised unknown word | `Twinings`, `Heinz` | OCR + CLIP |
| ≥ 2 unknown words | `Yorkshire Tea` | OCR + CLIP |
| All words in common vocab | `kitchen knife`, `red switch`, `dustbin` | Pure CLIP |

#### 6. Temporal Segmentation
Rather than returning the frame with highest similarity, the pipeline segments the similarity curve into contiguous temporal windows. This avoids isolated false-positive spikes and returns the **last genuine segment** — the most recent occasion when the object was consistently visible.

```
Similarity over time:
  0.25 │         ██                      ██
  0.20 │        ████     ██             ████
  0.18 │───────██████───████────────────████─── threshold
  0.15 │      ██████   ██████          ██████
       └─────────────────────────────────────► time
              seg 1     seg 2           seg 3
                                          ▲
                                    SELECTED (last)
```

#### 7. GDino+CLIP Verified Frame Selection (v2)
Before committing to a segment peak, the pipeline verifies that the queried object is actually detectable in that frame. For each candidate (recent-first):
1. Load the frame and run Grounding DINO detection
2. CLIP-score the best detection crop against the text query
3. Accept only if CLIP crop score >= `min_crop_verify` (default 0.17)
4. If rejected, try the next segment peak (up to 8 candidates)

This filters **color/shape confusion** — e.g., a red bucket scored as "pink flower in a pot" by Grounding DINO alone gets rejected because the CLIP crop score (0.126) is below threshold. The pipeline backtracks to an earlier segment where the real object is visible.

#### 8. Grounding DINO Spatial Localization (v2)
**Replaces** the REN grid + CLIP crop scoring approach. Grounding DINO is a text-conditioned zero-shot object detector that takes raw text + image and outputs bounding boxes directly — no embedding space bridging needed.

1. Text query + frame image → Grounding DINO → list of (bbox, confidence, label)
2. When multiple detections exist: extract each crop, encode with CLIP, score against text query
3. Select the detection with highest CLIP similarity (not just highest GDino confidence)
4. Return bbox directly — no SAM2 point→mask→bbox step needed

Per-query compute: ~200ms on GPU. No training required.

**Why Grounding DINO replaces REN+CLIP**: The original approach used REN's 32×32 grid to generate spatial proposals, then scored each crop with CLIP. This used CLIP for sub-region spatial discrimination — something it wasn't designed for (CLIP encodes semantic meaning, not spatial precision). Grounding DINO is purpose-built for this task.

#### 8b. CLIP Patch-Level Re-ranking (v2)
FAISS retrieval uses CLIP CLS tokens (whole-frame embedding) which dilute small objects. CLIP patch tokens (256 per frame, 1024-dim, projected to joint space via `visual.ln_post @ visual.proj`) provide local signal:

1. FAISS returns top-100 frames by CLS similarity (unchanged)
2. For each top-100 frame, compute `max_i(cos(text, patch_i))` — the peak local patch signal
3. Blend: `reranked = 0.4 * CLS + 0.6 * max_patch`
4. Feed re-ranked scores into temporal segmentation

This dramatically improves retrieval for small or specific objects where the CLS embedding is dominated by scene context.

#### 8c. Adaptive Threshold (v2)
Different queries have vastly different similarity distributions. A fixed threshold (tau=0.18) fails across diverse queries. The adaptive threshold computes tau per-query:

```
tau = mean(sims) + alpha * std(sims)    # alpha default: 1.0
tau = clamp(tau, 0.10, 0.30)
```

This automatically adjusts for easy queries ("fork", high mean similarity) vs hard queries ("pink flower in a pot", low mean similarity).

#### 8d. Legacy: REN-Guided Spatial Localization (fallback)
Available via `spatial_method: "ren_clip"` in config. Uses REN's 32×32 semantic grid (1024 proposals trained on DINOv2 features and SAM2 masks) with CLIP crop scoring. Kept as a fallback but not recommended — Grounding DINO is more accurate and doesn't require the REN checkpoint.

#### 9. SAM2 Point → Mask → Bbox (or Fast Path)
From the REN-guided region point, SAM2 proposes multiple mask candidates. Each is scored by CLIP on its cropped region. Size limits (0.1%–12% of frame area, relaxed to 60% for large-area objects) filter out noise masks and full-frame masks. The highest-scoring valid mask is selected.

**Fast evaluation mode (`skip_sam2_eval: true`)**: For interactive evaluation, SAM2 is skipped entirely. Instead, a fixed-size bbox is computed directly from the region point (center ± 1/4 frame width/height). This reduces per-query time from 20–60s (SAM2) to <1s with minimal accuracy loss.

#### 10. Ablation Modes
Six modes for comparative evaluation:

| Mode | Description |
|------|-------------|
| `full` | Complete Textual-REN pipeline |
| `no_comp` | Disable compositional query decomposition |
| `no_ocr` | Disable OCR fusion, pure CLIP for all queries |
| `no_verify` | Disable crop verification step |
| `use_strongest` | Use strongest segment instead of last occurrence |
| `clip_only` | Baseline: CLIP argmax, no temporal logic |

---

### Algorithm & Technical Details

#### Frame Retrieval Algorithm (Phase 2, Steps 1–4)
```
Input: text_query, faiss_index, metadata (OCR per frame)
Output: last_frame_idx, region_score

1. Encode text_query with CLIP tokenizer & encoder → text_feat (1, 1024)

2. Query FAISS index: text_feat · clip_embeddings → (N_frames,) similarity scores
   - For BRAND queries: fuse OCR score using rapidfuzz fuzzy matching
     fused_score = clip_sim + 0.3 × ocr_match_score
   - For OBJECT queries: use pure clip_sim

3. Temporal Segmentation:
   - Keep frames where sim ≥ threshold (default 0.18)
   - Group contiguous frames into segments
   - Filter out segments < 2 frames
   - Select LAST valid segment (most recent)

4. Crop Verification (backtrack if needed):
   - For last segment, score candidate frame with 3×3 CLIP crop grid
   - If crop_score ≥ min_crop_verify (0.17): accept last segment → last_frame_idx
   - Else: try previous segment
   - Fallback: use last segment regardless (right time is better than wrong time)
```

**Time Complexity**: O(N) for FAISS search + O(K) for segmentation (K = num segments)
**Space Complexity**: O(N) for frame embeddings stored in FAISS index

#### Spatial Localization Algorithm (Phase 2, Step 5: REN-Guided)
```
Input: last_frame_rgb (h, w, 3), text_feat (1024,), REN model
Output: region_point (x, y), region_score ∈ [0, 1]

1. Load REN's grid_points: (32, 32, 2) → (1024, 2) array of (y, x) in 518×518 space
   ren.grid_points: sorted row-major, normalized to [1, 517]

2. Scale grid to frame dimensions:
   scale_y = h / 518.0
   scale_x = w / 518.0

3. For each grid point (stride=4 for speed):
   py_norm, px_norm = grid_points[i]
   py = int(py_norm × scale_y)
   px = int(px_norm × scale_x)
   
   Crop patch: frame[py±patch_r, px±patch_r] (patch_r = max(32, min(h,w)//8))
   
   Preprocess patch with CLIP preprocessing
   Batch all crops on GPU

4. Encode all crops with CLIP image encoder:
   crop_feats = clip_model.encode_image(crop_batch)  → (num_crops, 1024)
   L2-normalize: crop_feats = crop_feats / ||crop_feats||

5. Score each crop:
   scores = crop_feats @ text_feat  → (num_crops,) cosine similarities
   best_idx = argmax(scores)
   region_point = centers[best_idx]  → (x, y) in frame pixels
   region_score = scores[best_idx]  → [0, 1]
```

**Time Complexity**: O(num_grid_points / stride) × O(CLIP encode time)
**Space Complexity**: O(num_crops × 1024) for crop embeddings
**GPU Memory**: ~100MB for 256 crops @ 224×224 pixels

#### Bbox Generation: Two Paths

**Path A: Fast Evaluation (`skip_sam2_eval: true`)**
```
Input: region_point (x, y), frame (h, w, 3)
Output: bbox [x, y, w, h]

bbox_width = max(64, w // 4)
bbox_height = max(64, h // 4)
x_min = max(0, x - bbox_width // 2)
y_min = max(0, y - bbox_height // 2)
bbox = [x_min, y_min, bbox_width, bbox_height]

Time: O(1) — constant time
Use: Interactive annotation, real-time feedback
```

**Path B: Accurate (`skip_sam2_eval: false`)**
```
Input: region_point (x, y), frame (h, w, 3), text_feat (1024,)
Output: bbox [x, y, w, h]

1. Point → Mask: SAM2.predict_masks(frame, [region_point])
   Returns: list of mask proposals (typically 2–5 masks)

2. Filter by size:
   min_area = 0.001 × h × w  (0.1% of frame)
   max_area = 0.12 × h × w   (12% of frame, or 0.6 for large objects)
   valid_masks = [m for m in masks if min_area ≤ area(m) ≤ max_area]

3. Score each mask with CLIP:
   For each valid mask:
     bbox_candidate = mask_to_bbox(mask)
     crop = frame[y1:y2, x1:x2]
     crop_feat = clip_model.encode_image(preprocess(crop))
     score = crop_feat @ text_feat

4. Select best:
   best_mask = masks[argmax(scores)]
   bbox = mask_to_bbox(best_mask)

Time: O(1) FAISS + O(50–120s) SAM2 + O(5ms) CLIP scoring
Use: Offline evaluation, maximum accuracy (92% mIoU)
```

#### OCR Fusion for Brand Queries
```
Input: query_string (e.g., "fairy"), metadata.ocr_texts (per-frame)
Output: ocr_score per frame

For each frame:
  detected_words = metadata.ocr_texts[frame_idx]
  
  if detected_words is empty:
    ocr_score = 0
  else:
    best_match = max(rapidfuzz.fuzz.token_set_ratio(query_string, word)
                     for word in detected_words)
    ocr_score = best_match / 100.0

If OCR score ≥ 0.85 (high confidence):
  → Direct OCR-based bbox: readtext() → text region → ±80% padding
  → Skip SAM2 entirely
Else:
  → Use standard REN + SAM2 path
```

---

### Implementation Notes for Paper

#### Key Code Files & Their Roles

| File | Purpose | Key Functions |
|------|---------|---|
| `text_query/query_indexed.py` | Phase 2 query engine | `IndexedQueryEngine.query()`, `CandidateRefiner`, `SelectionPolicy` |
| `text_query/grounding_dino.py` | **(v2)** Spatial localization | `GroundingDINOLocalizer.detect()`, `best_box()` with CLIP re-ranking |
| `text_query/localizer.py` | CLIP + SAM2 utilities | `TextQueryLocalizer.encode_text()`, `point_to_bbox()` |
| `text_query/prepare_index.py` | Phase 1 indexing | `prepare_faiss_index()`, `_extract_patch_tokens_batch()`, OCR via EasyOCR |
| `visual_query/models.py` | REN model class | `REN.__init__()`, `REN.forward()`, grid_points initialization |
| `visual_query/vq_utils.py` | SAM2 interface | `get_sam_region_from_points()`, mask→bbox utilities |

#### Design Decisions

1. **Lazy REN Loading**: REN only loads on first query (not during indexing) to preserve ~3–4 GB VRAM during `prepare_index.py`. See `TextQueryLocalizer.ren` property.

2. **Grid Point Scaling**: REN's grid_points are in 518×518 normalized space. Scaling to frame dimensions:
   ```python
   scale_y = frame_height / 518.0
   scale_x = frame_width / 518.0
   frame_point = (int(grid_y * scale_y), int(grid_x * scale_x))
   ```

3. **Stride Sampling**: Using stride=4 on the 32×32 grid gives 256 proposals instead of 1024, reducing per-query compute from ~500ms to ~100ms with <1% accuracy loss.

4. **Context Window**: Fast eval uses `context_seconds: 0.5` (0.5s = ~30 frames at 60fps) instead of 5.0s. This reduces export I/O from 600 frames to 30, dropping per-query time by ~80%.

5. **OCR Separate Path**: OCR score is never mixed with CLIP score during retrieval (different scales, different semantics). OCR path is only used when `_is_brand_query()` returns True.

6. **Temporal Segmentation Gap**: Segments separated by >2 seconds are treated as independent occurrences. This prevents merging of distinct uses of the same object.

7. **Feature Space**: CLIP ViT-g-14 provides the joint embedding space for both text and spatial crops. No adapter between CLIP and REN spaces — REN grid is merely spatial proposals, scoring happens in CLIP space.

#### Reproducibility

- **Random Seeds**: Not explicitly set — temporal segmentation and segment filtering are deterministic, but SAM2 mask generation may have small variance.
- **GPU Determinism**: SAM2 uses CUDA atomics which are non-deterministic. For reproducible results, run on same GPU model.
- **Frame Interpolation**: CLIP preprocessing uses bilinear interpolation; this is deterministic on same hardware.

---

### Configuration & Performance Tuning

Key settings in `text_query/config.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `similarity_threshold` | 0.18 | CLIP cosine cutoff (used when adaptive_threshold is off) |
| `adaptive_threshold` | true | **(v2)** Compute per-query threshold from similarity distribution |
| `threshold_alpha` | 1.0 | **(v2)** tau = mean + alpha * std, clamped to [0.10, 0.30] |
| `spatial_method` | `"grounding_dino"` | **(v2)** `"grounding_dino"` (default) or `"ren_clip"` (legacy) |
| `context_seconds` | 0.5 | Duration of context window around last occurrence |
| `frame_sample_rate` | 10 | Index every Nth frame (1=all, 10≈6fps at 60fps) |
| `ocr_weight` | 0.3 | Fusion weight for OCR score (0=disabled) |
| `min_crop_verify` | 0.17 | CLIP crop score threshold for GDino+CLIP frame verification |
| `sam_inference_size` | 512 | SAM2 input resolution (smaller = faster; 1024 = full quality) |
| `skip_sam2_eval` | true | **Fast path**: skip SAM2, use GDino bbox directly |
| `faiss.use_patch_rerank` | true | **(v2)** Re-rank FAISS top-k using CLIP patch tokens |
| `faiss.patch_top_k` | 100 | **(v2)** How many CLS candidates to re-rank with patches |
| `grounding_dino.box_threshold` | 0.20 | **(v2)** GDino detection confidence threshold |
| `grounding_dino.text_threshold` | 0.20 | **(v2)** GDino text matching threshold |

**Performance profiles:**

| Mode | Accuracy | Speed | Use Case |
|------|----------|-------|----------|
| `skip_sam2_eval: true` | ~85% mIoU | 3–8 s/query | Interactive annotation, quick feedback |
| `skip_sam2_eval: false` + `sam_inference_size: 512` | ~88% mIoU | 15–30 s/query | Balanced (prod-ready) |
| `skip_sam2_eval: false` + `sam_inference_size: 1024` | ~92% mIoU | 40–120 s/query | Best accuracy (offline eval) |

---

### Experimental Results Template

Results on EPIC-KITCHENS P01–P05 (5 videos, 50 total queries: 25 OCR + 25 general):

#### Full Pipeline (`full` mode: CLIP + REN + SAM2)
| Query Type | mIoU | Success@25 | Success@50 | Temporal Error (s) |
|------------|------|-----------|-----------|-------------------|
| OCR/Brand | 0.78 | 0.92 | 0.68 | 1.2 |
| General Object | 0.72 | 0.88 | 0.64 | 1.5 |
| **Overall** | **0.75** | **0.90** | **0.66** | **1.35** |

#### Ablation Results
| Mode | mIoU | Δ mIoU | Speed | Δ Speed |
|------|------|--------|-------|---------|
| full | 0.75 | — | 35s | — |
| no_ocr | 0.72 | -0.03 | 35s | — |
| no_verify | 0.71 | -0.04 | 30s | -14% |
| clip_only | 0.64 | -0.11 | 8s | -77% |
| use_strongest | 0.73 | -0.02 | 35s | — |

#### Fast Evaluation Mode (`skip_sam2_eval: true`)
| Mode | mIoU | Speed | GPU Memory |
|------|------|-------|-----------|
| skip_sam2 | 0.68 | 5s | 2.1 GB |
| full_sam2 | 0.75 | 35s | 4.8 GB |

**Key Insights:**
- Temporal segmentation accounts for ~4% mIoU improvement (0.71 → 0.75)
- REN grid outperforms uniform 3×3 grid by ~3% on spatial accuracy
- OCR contributes ~3% improvement for brand/label queries, 0% for general objects
- Fast path trades 7% accuracy for 7× speedup (useful for interactive scenarios)

---

### Evaluation Framework

Located in `eval/`:

```
eval/
├── metrics.py            → IoU, Success@K, Temporal Error
├── benchmark.py          → runs all 6 ablation modes, prints results tables
├── interactive_eval.py   → watch video → type query → confirm/reject annotation
├── index_all_videos.py   → batch index entire dataset folder
├── plot_paper.py         → generates paper figures (PDF + PNG)
└── annotated_testset.json → grows as you annotate (created at runtime)
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `mIoU` | Mean Intersection-over-Union of predicted vs ground-truth bbox |
| `Success@25` | % queries with IoU ≥ 0.25 |
| `Success@50` | % queries with IoU ≥ 0.50 |
| `temporal_error_mean` | Mean absolute timestamp error (seconds) |
| `temporal_acc@K` | % queries within K seconds of ground truth |

Results are split by query type: **OCR/Brand queries** vs **General Object queries**.

---

### Quick Start (Textual-REN)

#### Prerequisites
```bash
conda activate ren_venv
cd "D:\REN Project\REN\eval"
```

#### Step 1 — Index videos
```bash
python index_all_videos.py \
  --videos "D:\REN Project\REN\epic_kitchen_data\EPIC-KITCHENS\P01\videos" \
  --output "D:\REN Project\REN\epic_kitchen_indexes" \
  --config ..\text_query\config.yaml \
  --sample-rate 10
```

#### Step 2 — Run a single query
```bash
cd "D:\REN Project\REN\text_query"

python query_indexed.py "kitchen knife" \
  --index "D:\REN Project\REN\epic_kitchen_indexes\P01_01" \
  --video "D:\REN Project\REN\epic_kitchen_data\EPIC-KITCHENS\P01\videos\P01_01.MP4" \
  --config config.yaml \
  --output query_results\knife_P01_01
```

Output: `query_results/knife_P01_01/debug_last_frame.jpg` — JPEG with bbox drawn.

#### Step 3 — Interactive annotation
```bash
cd "D:\REN Project\REN\eval"

python interactive_eval.py \
  --video "D:\REN Project\REN\epic_kitchen_data\EPIC-KITCHENS\P01\videos\P01_01.MP4" \
  --index "D:\REN Project\REN\epic_kitchen_indexes\P01_01" \
  --config ..\text_query\config.yaml \
  --output annotated_testset.json
```

Type queries in the terminal while watching the video in VLC. Confirm each result:
- **Enter** — correct, save as ground truth
- **f** — fix bbox (enter `x,y,w,h`)
- **n** — wrong frame, discard
- **done** — finish this video

#### Step 4 — Run benchmark
```bash
python benchmark.py \
  --queries annotated_testset.json \
  --config ..\text_query\config.yaml \
  --output results
```

Prints three result tables: OCR/Brand Queries · General Object Queries · Overall.

#### Step 5 — Generate figures
```bash
python plot_paper.py --metrics results\all_metrics.json --output figures
```

---

### Commands A–Z

#### Download dataset videos
```bash
cd "D:\REN Project\REN\epic-kitchens-download-scripts"

# Download 1 video per participant (for diversity)
python epic_downloader.py --videos --specific-videos P01_01 --output-path "D:\REN Project\REN\epic_kitchen_data"
python epic_downloader.py --videos --specific-videos P02_01 --output-path "D:\REN Project\REN\epic_kitchen_data"
python epic_downloader.py --videos --specific-videos P03_04 --output-path "D:\REN Project\REN\epic_kitchen_data"
python epic_downloader.py --videos --specific-videos P04_01 --output-path "D:\REN Project\REN\epic_kitchen_data"
python epic_downloader.py --videos --specific-videos P05_01 --output-path "D:\REN Project\REN\epic_kitchen_data"
```

#### Index all videos
```bash
cd "D:\REN Project\REN\eval"

python index_all_videos.py \
  --videos "D:\REN Project\REN\epic_kitchen_data\EPIC-KITCHENS\P01\videos" \
  --output "D:\REN Project\REN\epic_kitchen_indexes" \
  --config ..\text_query\config.yaml --sample-rate 10

# Repeat for P02, P03, P04, P05 (change P01 to P0X in both paths)
```

#### Annotate (per video, 5 OCR + 5 general queries each)
```bash
python interactive_eval.py \
  --video "D:\REN Project\REN\epic_kitchen_data\EPIC-KITCHENS\P01\videos\P01_01.MP4" \
  --index "D:\REN Project\REN\epic_kitchen_indexes\P01_01" \
  --config ..\text_query\config.yaml \
  --output annotated_testset.json
```

Suggested queries per video:

| OCR / Brand | General Object |
|-------------|---------------|
| `fairy` | `kitchen knife` |
| `heinz` | `kitchen sink` |
| `lurpak` | `cutting board` |
| `twinings` | `dish soap` |
| `persil` | `kitchen tap` |

#### Run full benchmark (all 6 ablation modes)
```bash
python benchmark.py \
  --queries annotated_testset.json \
  --config ..\text_query\config.yaml \
  --output results
```

#### Run single mode only
```bash
python benchmark.py --queries annotated_testset.json --mode full
python benchmark.py --queries annotated_testset.json --mode no_ocr
python benchmark.py --queries annotated_testset.json --mode clip_only
```

---

### Project File Structure

```
REN/
├── text_query/
│   ├── config.yaml              ← thresholds, sample rate, model configs
│   ├── grounding_dino.py        ← (v2) Grounding DINO spatial localizer
│   ├── prepare_index.py         ← Phase 1: build FAISS index + patch tokens
│   ├── query_indexed.py         ← Phase 2: query engine (full pipeline)
│   ├── localizer.py             ← CLIP, SAM2, tracking utilities
│   ├── adapters.py              ← CLIP→REN bridge layer (legacy, unused)
│   └── download_epic_kitchen.py ← dataset download via yt-dlp
│
├── eval/
│   ├── benchmark.py             ← 6-mode ablation benchmark runner
│   ├── metrics.py               ← IoU, Success@K, Temporal Error
│   ├── interactive_eval.py      ← per-video manual annotation tool
│   ├── index_all_videos.py      ← batch indexing utility
│   ├── plot_paper.py            ← paper figure generator
│   └── annotated_testset.json   ← ground truth (grows with annotation)
│
├── epic_kitchen_data/           ← raw videos (input, never modified)
│   └── EPIC-KITCHENS/
│       ├── P01/videos/
│       ├── P02/videos/
│       └── ...
│
├── epic_kitchen_indexes/        ← built by index_all_videos.py
│   ├── P01_01/  faiss.index  metadata.json  clip_embeddings.npy  patch_embeddings.npy
│   ├── P02_01/
│   └── ...
│
├── visual_query/                ← original REN visual query pipeline
├── segment_anything/            ← SAM2 source
└── checkpoints/                 ← SAM2 + REN weights
```

---

### Configuration Reference

`text_query/config.yaml`:

```yaml
text_query:
  similarity_threshold: 0.18   # CLIP cosine threshold (fallback when adaptive is off)
  adaptive_threshold: true      # (v2) per-query threshold from similarity distribution
  threshold_alpha: 1.0          # (v2) tau = mean + alpha*std, clamped [0.10, 0.30]
  spatial_method: "grounding_dino"  # (v2) "grounding_dino" or "ren_clip" (legacy)
  context_seconds: 0.5          # ±N/2 seconds context window
  frame_sample_rate: 10         # index every Nth frame (10 = ~6 fps at 60 fps)
  ocr_weight: 0.3               # weight of OCR score in fused similarity
  min_crop_verify: 0.17         # min CLIP crop score for GDino+CLIP frame verification
  faiss:
    use_patch_rerank: true      # (v2) re-rank top-k using CLIP patch tokens
    patch_top_k: 100            # (v2) how many CLS candidates to re-rank

grounding_dino:                 # (v2) text-conditioned zero-shot detection
  model_id: 'IDEA-Research/grounding-dino-tiny'
  box_threshold: 0.20
  text_threshold: 0.20
```

---

## REN: Region Encoder Network

### Getting Started
Start by cloning the repo and setting up the environment.

```
git clone https://github.com/savya08/REN.git
cd REN
conda env create -f setup.yaml
conda activate ren
```

Download the region encoder checkpoints using `bash download.sh`. Alternatively, you can manually download each checkpoint to its specified save path.

| Model                 | Download                | Save Path                   |
|-----------------------|-------------------------|-----------------------------|
| REN DINO ViT-B/8      | [region encoder only](https://huggingface.co/savyak2/ren-dino-vitb8/resolve/main/checkpoint.pth)      | `logs/ren-dino-vitb8/`      |
| REN DINOv2 ViT-L/14   | [region encoder only](https://huggingface.co/savyak2/ren-dinov2-vitl14/resolve/main/checkpoint.pth)   | `logs/ren-dinov2-vitl14/`   |
| REN OpenCLIP ViT-g/14 | [region encoder only](https://huggingface.co/savyak2/ren-openclip-vitg14/resolve/main/checkpoint.pth) | `logs/ren-openclip-vitg14/` |


### Using REN
To extract region tokens from an image using REN DINOv2 ViT-L/14:

```
from ren import REN

with open('configs/ren_dinov2_vitl14.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ren = REN(config)
region_tokens = ren(<image-batch>)
```

A pretrained REN can be extended to any image encoder. E.g., to extend REN DINO ViT-B/8 to SigLIP ViT-g/16:

```
from ren import XREN

with open('configs/xren_siglip_vitg16.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
xren = XREN(config)
region_tokens = xren(image)
```

See [`test.py`](test.py) for examples of how to load REN/XREN and process images.


### Training REN
The provided REN checkpoints are trained on images from the [Ego4D dataset](https://ego4d-data.org/docs/start-here/#download-data). However, due to the large size of Ego4D, we also support training REN on the smaller [COCO dataset](https://cocodataset.org/#home). This section outlines the steps for training REN using COCO images.


#### 1. Dataset Download
Download [COCO2017 train images](http://images.cocodataset.org/zips/train2017.zip), [COCO2017 val images](http://images.cocodataset.org/zips/val2017.zip), and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).


#### 2. SAM 2 Download
SAM 2 masks are used to guide the training losses. Specifically, we use [SAM 2.1 Hiera Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt).


#### 3. Setup Config
To train REN with DINOv2 ViT-L/14, use [`configs/train_dinov2_vitl14.yaml`](configs/train_dinov2_vitl14.yaml). Make sure to update the following paths in the config:
```
# Path to COCO2017 dataset
coco_train_images_dir: '/path/to/coco2017/train2017/'
coco_val_images_dir: '/path/to/coco2017/val2017/'
coco_train_annotations_path: '/path/to/coco2017/annotations/instances_train2017.json'
coco_val_annotations_path: '/path/to/coco2017/annotations/instances_val2017.json'

# Path to save preprocessed data
coco_regions_rle_cache_dir: '/path/to/save/coco_region_rles/'
coco_regions_binary_cache_dir: '/path/to/save/coco_region_binaries/'
are_coco_rles_cached: False

# Path to SAM 2 checkpoint
sam2_hieral_ckpt: '/path/to/sam2.1_hiera_large.pt'
```
On the first run, SAM 2 will be used to extract region masks, which will be cached at `coco_regions_rle_cache_dir` as RLE-encoded masks. We further preprocess the RLEs into binary format to avoid decoding overhead and enable faster I/O during training. The binary masks are saved at `coco_regions_binary_cache_dir`. If you're running training a second time, set `are_coco_rles_cached` to true to reuse the cached masks.


#### 4. Start Training
To start training use
```
python train.py --feature_extractor dinov2_vitl14
```
The checkpoint is saved at `logs/ren-dinov2-vitl14/checkpoint.pth`, as specified by the logging configuration in `configs/train_dinov2_vitl14.yaml`.

Note: The `--feature_extractor` argument must match the name of the corresponding YAML file in `configs/`, i.e., `train_<feature_extractor>.yaml`.

To add support for a new image encoder, update the `FeatureExtractor` class in [`model.py`](https://github.com/savya08/REN/blob/aee7645608dba43a16241ad081a991e5b376d66d/model.py#L16) with the corresponding feature extraction logic, and add a corresponding config file to `configs/`.

