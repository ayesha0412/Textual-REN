# Next Steps: Validation & Deployment Checklist

## ✓ What's Been Completed

### Core Implementation
- ✓ **adapters.py**: TextRegionAdapter + ClipRegionScorer
- ✓ **prepare_index.py**: Phase 1 offline indexing pipeline
- ✓ **query_indexed.py**: Phase 2 online query engine
- ✓ **config.yaml**: FAISS + adapter parameters
- ✓ **test_epic_kitchen.py**: Validation suite with 13 standard queries
- ✓ **download_epic_kitchen.py**: Synthetic + real data download tools

### Documentation
- ✓ **README_FAISS.md**: System overview & quick reference
- ✓ **QUICKSTART.md**: 5-min & 15-min tutorials
- ✓ **FAISS_WORKFLOW.md**: Full architecture documentation
- ✓ **EPIC_KITCHEN_GUIDE.md**: Dataset setup & validation
- ✓ **IMPLEMENTATION_SUMMARY.md**: Implementation details
- ✓ **NEXT_STEPS.md**: This checklist

## → Your Immediate Action Items

### Phase 1: Quick Sanity Check (5 minutes)

**Goal**: Verify system loads correctly

```bash
conda activate ren_venv
cd "D:/REN Project/REN/text_query"

# Check imports
python -c "
import torch
import faiss
from adapters import TextRegionAdapter, ClipRegionScorer
from prepare_index import VideoIndexer
from query_indexed import IndexedQueryEngine
print('✓ All imports successful')
"

# Check config
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print('✓ Config loads successfully')
print(f'  FAISS top_k: {config[\"text_query\"][\"faiss\"][\"top_k\"]}')
print(f'  Adapter dims: {config[\"text_query\"][\"adapter\"][\"input_dim\"]}->{config[\"text_query\"][\"adapter\"][\"output_dim\"]}')
"
```

**Expected output**: Two success messages

---

### Phase 2: Synthetic Video Test (5 minutes)

**Goal**: End-to-end test with generated data

```bash
# 1. Generate test video
echo "[Step 1/4] Generating synthetic video..."
python download_epic_kitchen.py --synthetic --duration 10 --output test_video.mp4
# Expected: test_video.mp4 (~50 MB)

# 2. Build index
echo "[Step 2/4] Building FAISS index..."
python prepare_index.py test_video.mp4 --output test_index/ --sample-rate 2
# Expected: test_index/ with 4 files
# Time: 2-3 minutes

# 3. Run single query
echo "[Step 3/4] Querying index..."
python query_indexed.py "cup" \
  --index test_index/ \
  --video test_video.mp4 \
  --output test_results/
# Expected: test_results/last_occurrence.mp4
# Time: 1-2 seconds

# 4. Check results
echo "[Step 4/4] Verifying results..."
ls -lh test_results/
cat test_results/result.json | python -m json.tool
```

**Success criteria:**
- ✓ test_video.mp4 created
- ✓ test_index/ built (4 files)
- ✓ test_results/last_occurrence.mp4 exists
- ✓ result.json shows valid frame numbers

---

### Phase 3: Validation Suite (20 minutes)

**Goal**: Run standard test queries, measure latency

```bash
# Quick test (3 queries, 5 minutes)
echo "[1/2] Running quick validation (3 queries)..."
python test_epic_kitchen.py \
  --index test_index/ \
  --video test_video.mp4 \
  --output validation_quick/
# Expected: success_rate >= 90%, avg_latency < 2s

# Full test (13 queries, 20 minutes)
echo "[2/2] Running full validation (13 queries)..."
python test_epic_kitchen.py \
  --index test_index/ \
  --video test_video.mp4 \
  --output validation_full/ \
  --batch
# Expected: success_rate >= 70%, bottleneck analysis

# Inspect results
cat validation_full/validation_results.json | python -m json.tool
```

**Expected output:**
```json
{
  "success_rate": 0.85,
  "avg_latency": 1.5,
  "num_queries": 13,
  "queries": [
    {"query": "cup", "success": true, "latency": 1.4},
    ...
  ]
}
```

---

### Phase 4: Epic Kitchen Validation (30-60 minutes)

**Goal**: Validate on real egocentric data

**Option A: Quick (with generated video)**
```bash
# Use 30-second synthetic video instead of real download
python download_epic_kitchen.py --synthetic --duration 30 --output epic_test.mp4

# Index
python prepare_index.py epic_test.mp4 --output epic_index/ --sample-rate 2

# Validate
python test_epic_kitchen.py --index epic_index/ --video epic_test.mp4 --batch
```

**Option B: Real Data (requires download)**
```bash
# Instructions in EPIC_KITCHEN_GUIDE.md
# For now, use generated video above

# When ready for real data:
# 1. Install yt-dlp: pip install yt-dlp
# 2. Follow instructions in download_epic_kitchen.py
# 3. Download one full Epic Kitchen video (~200 MB)
# 4. Repeat indexing & validation above
```

**Success criteria:**
- Success rate >= 70% (real data more challenging)
- Average latency 1-2 seconds
- Bottleneck analysis shows "none" or "region_refinement"

---

### Phase 5: Ego4D Scaling (optional, requires ~500 GB)

**Goal**: Validate on long-form diverse videos

```bash
# 1. Download Ego4D VQ2D subset
# See: https://ego4d-data.org/docs/data/download/
# Recommended: ~500 GB to start (10-20 videos)
# Extract to: ../ego4d_data/

# 2. Index one long video (45 min)
python prepare_index.py ../ego4d_data/long_video.mp4 \
  --output ego4d_index/ \
  --sample-rate 3  # Higher rate for speed on long videos
# Time: 15-20 minutes

# 3. Query with diverse prompts
python query_indexed.py "coffee mug" \
  --index ego4d_index/ \
  --video ../ego4d_data/long_video.mp4 \
  --threshold 0.20

python query_indexed.py "person reaching for item" \
  --index ego4d_index/ \
  --video ../ego4d_data/long_video.mp4 \
  --threshold 0.18

# 4. Batch validation
python test_epic_kitchen.py --index ego4d_index/ --video ../ego4d_data/long_video.mp4 --batch
```

**Success criteria:**
- Success rate >= 50% (Ego4D is harder)
- Latency stable at 1-2 seconds (scales with video length)
- Bottleneck analysis guides optimization

---

## How to Interpret Results

### Success Rate
- **>90%**: Excellent, ready to deploy
- **70-90%**: Good, acceptable for early validation
- **50-70%**: Moderate, may need threshold tuning
- **<50%**: Poor, investigate failure cases

### Average Latency
- **<1 second**: Excellent
- **1-2 seconds**: Good
- **2-5 seconds**: Acceptable but slow
- **>5 seconds**: Needs optimization

### Bottleneck Analysis
- **"none"**: All systems performant, ready for production
- **"faiss_search"**: Reduce --top-k or use IVF index
- **"region_refinement"**: Reduce frame_sample_rate or use fewer regions
- **"sam2"**: GPU acceleration recommended for many queries

---

## Debugging Failed Queries

If a query fails, follow these steps:

```bash
# 1. Check that object is actually in video
ffplay <video_path>  # Watch video manually

# 2. Try different query phrasing
python query_indexed.py "similar_object" --index <index_dir> --video <video_path>

# 3. Lower similarity threshold
python query_indexed.py <query> --index <index_dir> --video <video_path> --threshold 0.15

# 4. Increase refinement candidates
python query_indexed.py <query> --index <index_dir> --video <video_path> --top-k 200

# 5. Check index validity
python -c "
import faiss
idx = faiss.read_index('<index_dir>/faiss.index')
print(f'Index valid: {idx.ntotal} vectors loaded')
"
```

---

## Optimization Opportunities

Once basic validation passes, consider these optimizations:

### For Speed
1. **Use IVF index** (approximate FAISS search)
   ```yaml
   # In config.yaml
   faiss:
     index_type: 'ivf'  # Instead of 'flat'
   ```

2. **Increase frame sampling rate** (fewer features to process)
   ```bash
   python prepare_index.py video.mp4 --sample-rate 4  # Instead of 2
   ```

3. **Reduce refinement candidates**
   ```bash
   python query_indexed.py query --index idx --video vid --top-k 50  # Instead of 100
   ```

4. **Use GPU FAISS** (if NVIDIA GPU available)
   ```python
   # In query_indexed.py, modify _load_index():
   import faiss
   if faiss.get_num_gpus() > 0:
       self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
   ```

### For Accuracy
1. **Train TextRegionAdapter** on labeled video+query pairs
2. **Lower similarity threshold** (0.15-0.18 instead of 0.20)
3. **Increase frame_sample_rate** during indexing (more features)
4. **Use multilingual CLIP** (for non-English queries)

---

## Testing Checklist

Use this checklist to track your progress:

```
SYNTHETIC VIDEO TEST
□ Generate test_video.mp4
□ Build test_index/
□ Query "cup" succeeds
□ result.json shows valid frame
□ last_occurrence.mp4 plays

VALIDATION SUITE
□ Quick mode (3 queries) >= 90% success
□ Full mode (13 queries) >= 70% success
□ Average latency < 2 seconds
□ Bottleneck analysis helpful

EPIC KITCHEN (if attempting)
□ Generate or download epic_test.mp4
□ Build epic_index/
□ Run test_epic_kitchen.py
□ Success rate >= 70%
□ 13 queries complete in <30 minutes

EGO4D SCALING (if attempting)
□ Download Ego4D subset
□ Index one long video
□ Query with diverse prompts
□ Success rate >= 50%
□ Latency stable at 1-2 seconds
□ Identify optimization targets
```

---

## Common Issues & Quick Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "No frames found above threshold" | Threshold too high / object not in video | Lower --threshold to 0.15 |
| Query takes >2 seconds | REN refinement bottleneck | Reduce --top-k to 50 |
| CUDA out of memory | Video too long (shouldn't happen) | Use --sample-rate 4 |
| Index file corrupt | Interrupted indexing | Delete index, rebuild |
| Query fails silently | Missing dependencies | Run import check (see above) |
| Results inaccurate | Threshold too low | Raise --threshold to 0.25 |

---

## What to Do If Validation Fails

**Success rate < 50%:**
1. Manually inspect video (does object actually appear?)
2. Try different query phrasing (more descriptive)
3. Lower similarity threshold (0.15 minimum)
4. Increase frame_sample_rate during indexing (more features)

**Latency > 5 seconds:**
1. Check bottleneck analysis (which step is slow?)
2. Reduce --top-k to 50
3. Increase frame_sample_rate during indexing (fewer frames to process)
4. Consider GPU FAISS if available

**Crashes or errors:**
1. Check conda environment activated (`conda activate ren_venv`)
2. Verify dependencies installed (`pip install faiss-cpu`)
3. Check config.yaml is valid (YAML syntax)
4. Inspect error trace (usually indicates missing file or memory)

---

## Documentation Navigation

- **"How do I get started?"** → `QUICKSTART.md`
- **"What was implemented?"** → `IMPLEMENTATION_SUMMARY.md`
- **"How does it work?"** → `FAISS_WORKFLOW.md`
- **"How do I set up Epic Kitchen?"** → `EPIC_KITCHEN_GUIDE.md`
- **"What are the hyperparameters?"** → `config.yaml` + `FAISS_WORKFLOW.md`
- **"How do I debug?"** → `FAISS_WORKFLOW.md` → "Troubleshooting"
- **"Quick reference?"** → `README_FAISS.md`

---

## Timeline Estimate

| Phase | Time | Status |
|-------|------|--------|
| Imports check | 1 min | ✓ Before you test |
| Synthetic test | 5 min | ✓ Before Epic Kitchen |
| Validation suite | 20 min | → After synthetic |
| Epic Kitchen | 30 min | → After validation |
| Ego4D (optional) | 2 hours | → After Epic Kitchen |

**Total time to deployment-ready: ~1 hour**

---

## When to Stop Testing & Declare Success

✓ **System is ready when:**
- Synthetic test: 100% success, <1 second latency
- Validation suite: ≥70% success, <2 second latency
- Epic Kitchen: ≥70% success (if attempted)
- Bottleneck analysis: "none" or acceptable

✓ **Next major milestone: Deploy to Ego4D**

---

## Questions?

Refer to the documentation:
- `README_FAISS.md` — System overview
- `QUICKSTART.md` — Getting started
- `FAISS_WORKFLOW.md` — Architecture details
- `EPIC_KITCHEN_GUIDE.md` — Dataset & validation
- `config.yaml` — Configuration reference

---

**Ready to begin? Start with the Imports Check above, then move to Synthetic Video Test.**

**Estimated total time: 1 hour to deployment-ready**

Good luck! 🚀
