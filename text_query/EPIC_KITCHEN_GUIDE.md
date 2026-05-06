# Epic Kitchen Dataset Guide

## Overview

Epic Kitchen is a large-scale egocentric video dataset with action annotations, objects, and natural language descriptions. It's significantly smaller than Ego4D (~100 GB vs ~9 TB) and ideal for validating the text-query episodic localization pipeline before scaling to Ego4D.

## Dataset Details

- **Size**: ~100 GB total (vs Ego4D's 9 TB)
- **Videos**: ~700 egocentric videos from 32 participants
- **Duration**: 55 hours total (~5 min per video)
- **Annotations**: Actions, object interactions, scene descriptions
- **Format**: MP4 video + JSON metadata

## Quickstart

### Option A: Download Full Epic Kitchen (recommended for validation)

```bash
# Note: Requires ~100 GB disk space and ~30 min download time

# 1. Create Epic Kitchen directory
mkdir -p ../epic_kitchen_data/videos
cd ../epic_kitchen_data

# 2. Download metadata
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git annotations

# 3. Download videos using provided script
python ../text_query/download_epic_kitchen.py --split train --limit 50
```

### Option B: Use Pre-sampled Subset (fastest for testing)

Epic Kitchen provides pre-extracted frames and optical flow. Use frames from a single video:

```bash
# Download one full video for quick testing (~200 MB, 3-5 min)
python ../text_query/download_epic_kitchen.py --single-video P01_01 --output ../epic_kitchen_data/single_video
```

### Option C: Synthetic Test Video (no download needed)

For immediate testing without downloads:

```bash
# Generate synthetic egocentric video (~50 frames, various objects)
python ../text_query/generate_test_video.py --duration 10 --output test_video.mp4
```

## Validation Workflow

### 1. Prepare Index from Epic Kitchen Video

```bash
conda activate ren_venv
cd "D:/REN Project/REN/text_query"

# Index a single video
python prepare_index.py ../epic_kitchen_data/P01_01.mp4 \
  --output ../epic_kitchen_indexes/P01_01 \
  --sample-rate 2
```

**Expected output:**
```
Indexing video: ../epic_kitchen_data/P01_01.mp4
  Video: 5400 frames, 60 FPS, 1920x1080
  Sampled 2700 frames
  Total regions extracted: 250000+ regions
Building FAISS index...
  Index built with 2700 frames
Index saved to: ../epic_kitchen_indexes/P01_01
```

### 2. Query Against Index

```bash
# Query: "a person picking up a knife"
python query_indexed.py "knife in hand" \
  --index ../epic_kitchen_indexes/P01_01 \
  --video ../epic_kitchen_data/P01_01.mp4 \
  --output results/knife_test \
  --threshold 0.20
```

**Expected output:**
```
Query: 'knife in hand'
  Text embedding shape: torch.Size([1, 1280])

Searching FAISS index for top-100 candidates...
  Top-5 similarities: [0.45, 0.42, 0.39, 0.37, 0.35]
  Last occurrence above 0.20: frame 2847 (similarity: 0.28)

Refining top candidates with REN region tokens...
  Best match: frame 2847, region 142 (score: 0.89)

SAM2: estimating bbox from region point...
SAM2: tracking bbox through context window...

Exporting result...
  Output: results/knife_test/last_occurrence.mp4

=== Query Complete ===
  last_frame_idx: 2847
  best_frame_timestamp: 47.45 seconds
```

### 3. Validate Results

```bash
# Run test suite on validation queries
python test_epic_kitchen.py \
  --index ../epic_kitchen_indexes/P01_01 \
  --video ../epic_kitchen_data/P01_01.mp4 \
  --output validation_results/
```

## Sample Test Queries

Use these queries to test different object types in Epic Kitchen videos:

### Common Kitchen Objects
- "cutting board"
- "knife on counter"
- "plate being held"
- "hand picking up a mug"
- "water in sink"
- "food on table"

### Hand-Object Interactions
- "holding a pan"
- "stirring with a spoon"
- "pouring liquid"
- "reaching for item"
- "placing object down"

### Scene Descriptions
- "inside kitchen"
- "looking at stove"
- "person standing"
- "near refrigerator"

## Troubleshooting

### Out of Memory During Indexing
If you see "CUDA out of memory" or "Failed to allocate":

```bash
# Use streaming frame processor (automatic in prepare_index.py)
# Or reduce sample rate for fewer frames
python prepare_index.py video.mp4 --output index_dir --sample-rate 4
```

### Slow FAISS Search
If FAISS search takes >5s per query:

```bash
# Check that index uses flat (exact) search in config.yaml
# For faster but approximate search, switch to:
faiss:
  index_type: 'ivf'
```

### Low Similarity Scores on All Queries
Indicates query/video mismatch or model issues:

```bash
# Lower threshold to be more permissive
python query_indexed.py "knife" --threshold 0.15 ...

# Or check that the object actually appears in video
# (Epic Kitchen is more structured; Ego4D is more diverse)
```

## Next Steps: Scaling to Ego4D

Once validation passes on Epic Kitchen:

1. **Download Ego4D VQ2D subset** (~500 GB initially, full 9 TB optional)
2. **Index a single long Ego4D video** (~45 min, ~87K frames)
3. **Query with diverse text prompts** (testing long-tail objects)
4. **Measure latency & accuracy** (target: <2 sec per query)
5. **Batch index multiple videos** (full training set)

## References

- Epic Kitchen Dataset: https://epic-kitchens.github.io/
- Annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- Paper: *Rescaling Egocentric Vision* (Damen et al., 2022)

## Data Licensing

Epic Kitchen is released under the Creative Commons Attribution 4.0 License. When using the dataset, please cite:

```bibtex
@inproceedings{damen2022rescaling,
  title={Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
  author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria and and others},
  booktitle={ICCV},
  year={2022}
}
```
