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

**Textual-REN** extends the REN visual query pipeline to accept free-text queries instead of visual exemplars. Given a natural language description (e.g., `"kitchen knife"`, `"fairy dish soap"`, `"red switch"`), the system searches an offline-indexed egocentric video, finds the **last genuine occurrence** of the described object, and returns:

- A trimmed video clip with a **green bounding box** around the object
- The **timestamp** and **frame index** of the last occurrence
- A **spatial bounding box** `[x, y, w, h]` for evaluation

The pipeline is designed for episodic memory use cases — "where did I last put X?" — on datasets like EPIC-KITCHENS.

---

### Full Pipeline Architecture

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
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌──────────┐  ┌──────────────┐
│  CLIP    │  │  EasyOCR     │
│ ViT-g-14 │  │  per frame   │
│(OpenCLIP)│  │              │
└────┬─────┘  └──────┬───────┘
     │                │
     ▼                ▼
┌──────────┐  ┌──────────────┐
│  Image   │  │  OCR Text    │  word, confidence, bbox
│Embedding │  │  Metadata    │  stored per sampled frame
│(1024-dim)│  │              │
└────┬─────┘  └──────┬───────┘
     │                │
     └────────┬───────┘
              ▼
  ┌───────────────────────┐
  │     FAISS Flat Index  │  exact cosine search, GPU-accelerated
  │  +  metadata.json     │  frame_idx → timestamp, OCR texts
  │  +  clip_embeddings   │  (N_frames × 1024) numpy array
  └───────────────────────┘

  Saved to:  epic_kitchen_indexes/<video_id>/
             ├── faiss.index
             ├── metadata.json
             └── clip_embeddings.npy


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
  ┌────────────┐       ┌────────────────┐
  │ CLIP Text  │       │  CLIP Text     │
  │  Encoder   │       │  Encoder       │
  │ (ViT-g-14) │       │  (ViT-g-14)   │
  └─────┬──────┘       └───────┬────────┘
        │                      │
        ▼                      ▼
  ┌─────────────────────────────────┐
  │   Cosine Similarity             │
  │   text_feat · clip_embeddings   │  shape: (N_frames,)
  └────────────────┬────────────────┘
                   │
        ┌──────────▼──────────┐
        │   OCR Fusion        │  (brand path only)
        │                     │
        │  ocr_score = max(   │  rapidfuzz partial_ratio
        │    partial_ratio,   │  + token_set_ratio
        │    token_set_ratio  │  over ±2 frame window
        │  ) / 100            │
        │                     │
        │  fused = CLIP       │
        │        + 0.3×OCR    │
        └──────────┬──────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   Temporal Segmentation                │
  │                                        │
  │  1. Keep frames where sim ≥ threshold  │  default: 0.18
  │  2. Group into contiguous segments     │  gap > 2s = new segment
  │     (consecutive sampled frames        │
  │      within 2-second gap)              │
  │  3. Filter: keep segments ≥ 2 frames   │
  │  4. Select LAST valid segment          │  most recent occurrence
  └────────────────┬───────────────────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   Crop Verification  (3×3 CLIP grid)   │
  │                                        │
  │  For each segment (last → first):      │
  │    crop_score = best 3×3 tile score    │
  │    if crop_score ≥ 0.17 → accept ✓    │
  │    else → try previous segment         │
  │                                        │
  │  Fallback: if ALL segments fail,       │
  │  use LAST segment anyway               │  most recent = best default
  └────────────────┬───────────────────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   Spatial Localization                 │
  │   Multi-Scale CLIP Crop Scoring        │
  │                                        │
  │  Crops evaluated:                      │
  │    • Full frame (global context)       │
  │    • 3×3 overlapping halves            │
  │    • 6×6 fine grid, 50% stride         │
  │                                        │
  │  → Best crop center = region_point     │
  │    (x, y) pixel coordinates            │
  └────────────────┬───────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  OCR Direct Bbox    │  (brand path, OCR score ≥ 0.85)
        │  EasyOCR readtext() │  → precise text region bbox
        │  + 80% padding      │  expanded to full product label
        └──────────┬──────────┘
                   │  (or CLIP path: use region_point below)
                   ▼
  ┌────────────────────────────────────────┐
  │   SAM2 Point → Mask → Bbox            │
  │                                        │
  │  Input:  region_point (x, y)           │
  │  SAM2 predicts multi-mask proposals    │
  │  Each mask scored by CLIP crop sim     │
  │  Size filter: 0.1% – 12% of frame     │
  │    (60% for large objects:             │
  │     painting, screen, door, window)    │
  │  → Best scored mask within size range  │
  │  → Bounding box [x, y, w, h]          │
  └────────────────┬───────────────────────┘
                   │
                   ▼
  ┌────────────────────────────────────────┐
  │   SAM2 Video Tracking                  │
  │                                        │
  │  Context window: ±2.5 s around frame   │
  │  SAM2 forward propagation              │
  │  bbox tracked through all frames       │
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
Used for both offline frame embedding and online text encoding. The 1024-dim joint embedding space enables direct cosine similarity between text queries and video frames without any fine-tuning.

#### 2. FAISS Flat Index
Stores all frame embeddings for exact cosine search. GPU-accelerated when available. The index supports sub-second search over videos with 100 K+ frames. Separate `clip_embeddings.npy` enables batched similarity computation for ablations.

#### 3. EasyOCR Fusion
Brand/label queries (fairy, heinz, lurpak, twinings, persil) benefit from OCR — the text label visible on packaging gives a direct signal that CLIP alone may miss. OCR is pre-computed at indexing time and stored in `metadata.json`. At query time, `rapidfuzz` fuzzy-matches the query string against stored OCR texts to produce a per-frame OCR score, which is fused with the CLIP score.

#### 4. Query Type Classifier (`_is_brand_query`)
Automatically routes queries to the correct scoring path:

| Signal | Example | Path |
|--------|---------|------|
| Single word not in common vocab | `fairy`, `persil` | OCR + CLIP |
| Capitalised unknown word | `Twinings`, `Heinz` | OCR + CLIP |
| ≥ 2 unknown words | `Yorkshire Tea` | OCR + CLIP |
| All words in common vocab | `kitchen knife`, `red switch`, `dustbin` | Pure CLIP |

#### 5. Temporal Segmentation
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

#### 6. Crop Verification
Before committing to a segment, a fast 3×3 CLIP crop grid scores whether the object is actually visible in the candidate frame. If the score is too low, the pipeline backtracks to an earlier segment. If all segments fail, the **last segment** is used as a fallback (right timeframe with uncertain spatial location is better than a wrong timeframe).

#### 7. Multi-Scale CLIP Spatial Localisation
The spatial location of the object within the frame is found by scoring overlapping crops at three scales:
- Full frame
- 3×3 grid of half-frame tiles (50% overlap)
- 6×6 fine grid tiles (50% overlap)

The center of the highest-scoring crop becomes the input point to SAM2.

#### 8. SAM2 Mask Selection
SAM2 proposes multiple mask candidates from the input point. Each is scored by CLIP on its cropped region. Size limits (0.1%–12% of frame area, relaxed to 60% for large-area objects) filter out noise masks and full-frame masks. The highest-scoring valid mask is selected.

#### 9. Ablation Modes
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
│   ├── config.yaml              ← thresholds, sample rate, OCR weight
│   ├── prepare_index.py         ← Phase 1: build FAISS index from video
│   ├── query_indexed.py         ← Phase 2: query engine (full pipeline)
│   ├── localizer.py             ← CLIP, SAM2, tracking utilities
│   ├── adapters.py              ← CLIP→REN bridge layer
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
│   ├── P01_01/  faiss.index  metadata.json  clip_embeddings.npy
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
  similarity_threshold: 0.18   # CLIP cosine threshold for candidate frames
  use_compositional: false      # query decomposition (future work)
  context_seconds: 5.0          # ±N/2 seconds context window for tracking
  frame_sample_rate: 10         # index every Nth frame (10 = ~6 fps at 60 fps)
  ocr_weight: 0.3               # weight of OCR score in fused similarity
  min_crop_verify: 0.17         # min crop score to accept a candidate segment
  last_segment_min_peak: 0.18   # min peak to trust last segment
  last_segment_min_rel: 0.60    # last segment must be ≥60% of global peak
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


## License
This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.


## Citing REN
```
@inproceedings{khosla2025ren,
      title={REN: Fast and Efficient Region Encodings from Patch-Based Image Encoders}, 
      author={Savya Khosla and Sethuraman TV and Barnett Lee and Alexander Schwing and Derek Hoiem},
      booktitle={Neural Information Processing Systems},
      year={2025},
}
```
