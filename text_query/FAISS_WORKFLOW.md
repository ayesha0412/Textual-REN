# Text Query Episodic Localization: FAISS-Based Workflow

## Overview

This document describes the full two-phase FAISS-based architecture for text-query episodic memory localization. The system finds the last occurrence of a text-described object in long egocentric videos efficiently.

## Architecture

### Phase 1: Offline Indexing (Preprocessing)

```
Video → Frame Sampling → CLIP Embeddings + REN Region Tokens → FAISS Index + Metadata
```

**Process:**
1. Load video (can be 40+ minutes, 80K+ frames)
2. Sample frames at configurable rate (e.g., every 2nd frame for 2× speedup)
3. Extract per-frame CLIP image embeddings (1280-dim, laion2b pretrained)
4. Extract per-frame REN region tokens (multiple 1024-dim vectors per frame via grid-based cross-attention)
5. Build FAISS flat index on CLIP embeddings
6. Persist:
   - `faiss.index`: FAISS flat L2 index (N frames × 1280 dims)
   - `metadata.json`: Frame mapping (frame_idx, timestamp, region_count, region_start_idx)
   - `regions.pkl`: Serialized REN region tokens (variable-length per frame)
   - `clip_embeddings.npy`: NumPy array of CLIP embeddings for reference

**Files:**
- `prepare_index.py`: Main indexing script
- `VideoIndexer` class: Orchestrates sampling, feature extraction, index building

**Time Complexity:** O(N) per video, where N = total frames
**Space Complexity:** O(N × 1280) for CLIP + O(R × 1024) for regions, where R = total regions

### Phase 2: Online Query (Inference)

```
Text Query → CLIP Text Embedding → FAISS Search (top-K) → REN Refinement → 
SAM2 Bbox + Tracking → Export Result Clip
```

**Process:**
1. Encode text query with CLIP text encoder (1280-dim)
2. Search FAISS index for top-K candidate frames (fast, ~0.1s for 100K frames)
3. Score each candidate's REN region tokens against text query via TextRegionAdapter:
   - Project text embedding: CLIP 1280-dim → REN 1024-dim via linear adapter
   - Compute cosine similarity with per-region tokens
   - Keep highest-scoring region
4. Find "last occurrence": highest frame index where similarity ≥ threshold
5. Use SAM2 to convert best region to bounding box
6. Track bbox forward/backward in context window (±5s around last frame)
7. Export trimmed MP4 with green bbox overlay + metadata JSON

**Files:**
- `query_indexed.py`: Main query script
- `IndexedQueryEngine` class: Loads index, handles FAISS search + refinement
- `adapters.py`: TextRegionAdapter (CLIP→REN bridging)

**Time Complexity:** O(K × R) for refinement, where K = top_k candidates, R = avg regions/frame
**Query Latency:** ~1-2 seconds (FAISS: 0.1s, REN refinement: 0.5-1s, SAM2: 0.2-0.5s)

## Components

### 1. CLIP Text Encoder (OpenCLIP ViT-g-14)
- **Purpose**: Encode text queries into shared vision-language space
- **Output**: 1280-dim embedding, L2-normalized
- **Pretrained**: laion2b weights (robust to diverse objects)
- **Load time**: ~2 seconds

### 2. CLIP Image Encoder (OpenCLIP ViT-g-14)
- **Purpose**: Encode frames for fast similarity matching
- **Output**: 1280-dim per frame, L2-normalized
- **Why CLIP**: Trained on image-text pairs (better object understanding than DINO)
- **Frame extraction**: 1 embedding per frame, L2-norm → FAISS quantization

### 3. REN (DINOv2 ViT-L/14)
- **Purpose**: Extract spatial region tokens for fine-grained localization
- **Output**: Variable-length list of 1024-dim tokens per frame
- **Architecture**: Grid-based cross-attention from fixed grid points to patch features
- **grid_size**: 32×32 grid → up to 1024 region tokens per frame
- **Why REN**: Unsupervised spatial feature extraction (no human annotations needed)

### 4. TextRegionAdapter (Linear Projection)
- **Purpose**: Bridge CLIP text space (1280-dim) to REN region space (1024-dim)
- **Architecture**: Single linear layer, no bias
- **Initialization**: Normal distribution (σ=0.02) for stable training
- **Training**: Frozen for now (can be fine-tuned on labeled video+query pairs)
- **Usage**: Projects text embedding, scores each region via cosine similarity

### 5. FAISS Index
- **Type**: `IndexFlatL2` (exact search, no approximation)
- **Distance metric**: L2 (compatible with normalized embeddings)
- **Alternative**: `IndexIVF` for faster approximate search on large indices
- **Query**: Returns top-K closest CLIP embeddings + distances
- **Conversion**: L2 distance → similarity via: `sim = 1 - 0.5 × dist²`

### 6. SAM2 (Segment Anything Model 2)
- **Purpose**: Point → segmentation mask → bounding box
- **Input**: RGB frame + point (region center)
- **Output**: Mask + bbox
- **Tracking**: Forward/backward video segmentation
- **Checkpoint**: sam2.1_hiera_large (856 MB)

## Data Flow Diagram

```
OFFLINE PHASE (prepare_index.py)
═════════════════════════════════

Video File
    ↓
[CV2 VideoCapture] Load frame sequence (streaming, no memory overload)
    ↓
[Frame Sampling] Every sample_rate-th frame
    ↓
┌─────────────────┬─────────────────┐
│ CLIP Encoder    │ REN Encoder     │
│ ViT-g-14        │ DINOv2 ViT-L/14 │
└─────────────────┴─────────────────┘
    ↓                 ↓
[1280-dim embed]  [1024-dim tokens × N]
    ↓                 ↓
────────────────────────────────────────
    ↓
[FAISS Index] + [Metadata] + [Regions PKL]
    ↓
Saved to: ../epic_kitchen_indexes/P01_01/
    ├── faiss.index             (FAISS flat L2 index)
    ├── metadata.json           (Frame mapping)
    ├── regions.pkl             (Region tokens)
    └── clip_embeddings.npy     (CLIP embeddings)


ONLINE PHASE (query_indexed.py)
═══════════════════════════════

Text Query ("knife in hand")
    ↓
[CLIP Text Encoder]
    ↓
[1280-dim text embedding]
    ↓
[FAISS Search]
    ├── Load index from disk
    ├── Query: top-K closest CLIP embeddings
    └── Convert L2 dist → similarity
    ↓
[Top-K candidate frames + scores]
    ↓
[REN Region Refinement]
    ├── Load region tokens for each candidate
    ├── TextRegionAdapter: CLIP 1280 → REN 1024
    ├── Cosine similarity with regions
    └── Keep best region per frame
    ↓
[Find Last Occurrence]
    ├── Iterate candidates in reverse frame order
    ├── Find highest-indexed frame ≥ threshold
    └── Best region from that frame
    ↓
[SAM2: Point → Bbox]
    ├── Get center point of best region
    ├── Run SAM2 multi-mask prediction
    ├── Score masks with CLIP
    └── Extract bbox
    ↓
[SAM2: Track Bbox]
    ├── Forward tracking from best frame
    ├── Backward tracking from best frame
    └── Smooth bbox across context window
    ↓
[Export Clip]
    ├── Trim video ±context_seconds around best frame
    ├── Draw green bbox overlay
    ├── Add "LAST OCCURRENCE" label
    ├── Encode MP4
    └── Save metadata JSON
    ↓
Results: ../results/
    ├── last_occurrence.mp4      (Trimmed clip with bbox)
    ├── result.json              (Metadata)
    └── similarity.png           (Optional: similarity vs frame plot)
```

## Usage Examples

### 1. Quick Test with Synthetic Video

```bash
conda activate ren_venv
cd "D:/REN Project/REN/text_query"

# Generate 10-second synthetic egocentric video
python download_epic_kitchen.py --synthetic --duration 10 --output test_video.mp4

# Index the video
python prepare_index.py test_video.mp4 --output test_index/ --sample-rate 2

# Query
python query_indexed.py "cup" --index test_index/ --video test_video.mp4 --output test_results/

# Validate
python test_epic_kitchen.py --index test_index/ --video test_video.mp4 --output test_validation/
```

### 2. Validate on Epic Kitchen

```bash
# Download one full Epic Kitchen video (~200 MB, 5-10 min download)
python download_epic_kitchen.py --single-video P01_01

# Index
python prepare_index.py P01_01.mp4 --output epic_indexes/P01_01/ --sample-rate 2

# Query with standard test suite (13 queries, 15-20 min total)
python test_epic_kitchen.py --index epic_indexes/P01_01/ --video P01_01.mp4 --batch

# Run custom query
python query_indexed.py "holding a knife" --index epic_indexes/P01_01/ --video P01_01.mp4
```

### 3. Scale to Ego4D

```bash
# Download Ego4D VQ2D subset (500 GB recommended for initial validation)
# See: https://ego4d-data.org/

# Index a single long Ego4D video (45 min = 87K frames)
python prepare_index.py ego4d_video.mp4 --output ego4d_indexes/video_1/ --sample-rate 3

# Query
python query_indexed.py "coffee mug being used" --index ego4d_indexes/video_1/ \
  --video ego4d_video.mp4 --threshold 0.20

# Batch index multiple videos
for video in ego4d_videos/*.mp4; do
  python prepare_index.py "$video" --output "ego4d_indexes/$(basename $video .mp4)/" --sample-rate 3
done
```

## Configuration

Edit `config.yaml` to tune behavior:

```yaml
text_query:
  similarity_threshold: 0.20     # CLIP similarity cutoff (0.15-0.30 typical)
  context_seconds: 5.0            # Clip length around match (3-10 typical)
  frame_sample_rate: 2            # Process every Nth frame (1=all, higher=faster)
  
  faiss:
    clip_dim: 1280                # CLIP embedding dimension (fixed)
    top_k: 100                    # Candidates to refine (50-200 typical)
    index_type: 'flat'            # 'flat' (exact) or 'ivf' (approximate)
  
  adapter:
    input_dim: 1280               # CLIP text dim (fixed)
    output_dim: 1024              # REN region dim (fixed)
    temperature: 0.1              # Softmax temperature (0.05-0.5)
```

**Key Tuning Tips:**
- **Lower threshold** (0.15): More permissive, may include false positives
- **Higher frame_sample_rate** (3-4): Faster indexing/search, less accurate
- **Lower top_k** (50): Faster refinement, may miss best match
- **Increase context_seconds** (7-10): Longer output clip, more tracking

## Performance Metrics

### Expected Timings

| Phase | Component | Time |
|-------|-----------|------|
| **Index** | Load video (1hr, 87K frames) | ~5 min |
| | CLIP embedding | ~2 min |
| | REN region extraction | ~8 min |
| | FAISS index build | ~1 min |
| | **Total** | **~16 min per video** |
| **Query** | FAISS search (top-100) | 0.1 s |
| | REN refinement | 0.5-1.0 s |
| | SAM2 bbox + tracking | 0.2-0.5 s |
| | **Total** | **~1-2 seconds per query** |

### Expected Accuracy

| Dataset | Recall@1 | MRR | Notes |
|---------|----------|-----|-------|
| Synthetic | 95% | 0.95 | Trivial (4 objects, 10s) |
| Epic Kitchen | 70-80% | 0.75 | Moderate diversity, shorter videos |
| Ego4D | 50-60% | 0.55 | High diversity, longer context |

**Recall@1**: % queries where last occurrence is found above threshold
**MRR**: Mean Reciprocal Rank (how high is first correct match)

## Common Issues & Fixes

### 1. FAISS Search Slow (>2s per query)

**Symptom**: FAISS search returns in <0.5s but total query time is >2s
**Cause**: REN refinement or SAM2 is slow
**Fix**:
```bash
# Reduce top_k in config or command line
python query_indexed.py query --top-k 50 ...

# Reduce regions per frame by increasing sample_rate during indexing
python prepare_index.py video.mp4 --sample-rate 4 ...
```

### 2. All Queries Fail (No frames above threshold)

**Symptom**: "No frames found above similarity threshold 0.20"
**Cause**: Object not in video, or threshold too high
**Fix**:
```bash
# Lower threshold
python query_indexed.py query --threshold 0.15 ...

# Verify object is actually in video
# Try different query text
python query_indexed.py "similar object description" ...
```

### 3. Out of Memory During Indexing

**Symptom**: "CUDA out of memory" or "Failed to allocate X bytes"
**Cause**: Trying to load all frames at once
**Fix**: Already handled by streaming architecture, but:
```bash
# Use higher sample_rate
python prepare_index.py video.mp4 --sample-rate 4 ...

# Process on CPU
# (modify code to use device='cpu')
```

### 4. Index File Corrupt

**Symptom**: FileNotFoundError or "corrupted index"
**Fix**:
```bash
# Rebuild index
rm -rf index_dir/
python prepare_index.py video.mp4 --output index_dir/ ...
```

## Extending the System

### 1. Train TextRegionAdapter

Currently the adapter is frozen (linear identity projection). To improve region scoring:

```python
# In adapters.py, implement:
class TrainableTextRegionAdapter(TextRegionAdapter):
    def train_on_labeled_pairs(self, text_embeds, region_tokens, labels):
        # labels[i] = 1 if region_tokens[i] matches text_embeds[i]
        # Use contrastive loss (e.g., NT-Xent)
        pass
```

### 2. Use GPU FAISS

For faster search on large indices:

```python
# In query_indexed.py
import faiss
self.faiss_index = faiss.read_index(index_path)
if faiss.get_num_gpus() > 0:
    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
```

### 3. Approximate FAISS Index (IVF)

Trade search accuracy for speed on 1M+ frames:

```python
# In prepare_index.py
faiss_config = self.config['text_query']['faiss']
if faiss_config.get('index_type') == 'ivf':
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(1280), 1280, 100)
    index.train(clip_embeddings.astype(np.float32))
```

### 4. Multi-Modal Queries

Extend to include images as queries:

```python
# In localizer.py or query_indexed.py
def query_image(self, image_path, video_path, output_dir):
    # Load image, extract CLIP embedding (same as frame)
    # Rest of pipeline unchanged
    pass
```

## References

- CLIP: https://github.com/openai/CLIP
- OpenCLIP: https://github.com/mlfoundry/open_clip
- REN: [Paper] Region Encoder Network for Spatial Feature Extraction
- SAM2: https://github.com/facebookresearch/segment-anything-2
- FAISS: https://github.com/facebookresearch/faiss
- Epic Kitchen: https://epic-kitchens.github.io/
- Ego4D: https://ego4d-data.org/

## Next Steps

1. ✓ Implement adapters.py (TextRegionAdapter)
2. ✓ Implement prepare_index.py (Phase 1 offline indexing)
3. ✓ Implement query_indexed.py (Phase 2 online querying)
4. ✓ Create Epic Kitchen validation suite
5. → Test on synthetic video (quick sanity check)
6. → Validate on Epic Kitchen (medium-scale, ~10 videos)
7. → Scale to Ego4D (long-tail validation)
8. → Optimize adapter weights (labeled video data)
9. → Publish results & code
