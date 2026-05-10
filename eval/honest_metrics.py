"""
HONEST evaluation: compute the metrics that are genuinely valid
from real system outputs, without circular GT contamination.

Valid metrics we CAN compute:
  1. CLIP retrieval statistics (similarity scores, coverage)
  2. Temporal segmentation quality (segments found, frames above threshold)
  3. OCR hit rate for brand queries vs object queries
  4. Cross-video consistency (same query on multiple videos)
  5. Region localization score (CLIP grid crop score)
  6. One genuinely independent IoU: blue kettle (gt from different source)

All outputs saved to: paper_figures/honest_metrics_dashboard.png
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = r"D:\REN Project\REN"
sys.path.insert(0, os.path.join(ROOT, "eval"))
from metrics import compute_iou

OUT_DIR = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
})

# ================================================================== #
# Load ALL predictions (full picture across all videos)
# ================================================================== #

ALL_RESULTS = []

def add(path, video_id, query_type_override=None):
    if not os.path.exists(path):
        return
    with open(path) as f:
        r = json.load(f)
    qt = query_type_override or ("brand" if r.get("ocr_score", 0) > 0.5 else "object")
    ALL_RESULTS.append({
        "video_id": video_id,
        "query": r.get("query", ""),
        "query_type": qt,
        "pred_timestamp": r.get("last_frame_timestamp"),
        "pred_frame_idx": r.get("last_frame_idx"),
        "clip_similarity": r.get("clip_similarity", 0),
        "fused_similarity": r.get("fused_similarity", r.get("clip_similarity", 0)),
        "ocr_score": r.get("ocr_score", 0),
        "ocr_frames_hit": r.get("ocr_frames_hit", 0),
        "region_clip_score": r.get("region_clip_score", 0),
        "valid_segments": r.get("valid_segments"),
        "frames_above_threshold": r.get("frames_above_threshold", 0),
        "pred_bbox": r.get("pred_bbox"),
    })

# epic_results (P01 videos)
for folder, vid in [("black dustbin", "P01_03"), ("fork utensil", "P01_02"),
                    ("fork", "P01_02"), ("loaf of bread", "P01_05"), ("strainer", "P01_05")]:
    add(os.path.join(ROOT, f"epic_results/{folder}/result.json"), vid)

# query_results/P02_01
for folder in os.listdir(os.path.join(ROOT, "query_results/P02_01")):
    path = os.path.join(ROOT, f"query_results/P02_01/{folder}/result.json")
    brand_queries = {"yorkshire_tea", "yorkshire tea", "fairy", "twinings_camomile_tea",
                     "twinings camomile tea", "twinings_peppermint", "twinings peppermint"}
    qt = "brand" if folder.lower().replace(" ", "_") in brand_queries else "object"
    add(path, "P02_01", qt)

# eval/results/full P02_01
full_dir = os.path.join(ROOT, "eval/results/full/P02_01")
if os.path.exists(full_dir):
    for folder in os.listdir(full_dir):
        add(os.path.join(full_dir, folder, "result.json"), "P02_01")

# P04_01 plate
add(os.path.join(ROOT, "query_results/P04_01_plate/result.json"), "P04_01")

# P01_01 queries
add(os.path.join(ROOT, "eval/_tmp_eval/P01_01/orange_juice_bottle/result.json"), "P01_01")
add(os.path.join(ROOT, "query_results/test_cup/result.json"), "P01_01")
add(os.path.join(ROOT, "query_results/final_demo_test/result.json"), "P01_01")

# P01_02
for q in ["dustbin", "sponge"]:
    add(os.path.join(ROOT, f"eval/evaluation_results/P01_02/{q}/result.json"), "P01_02")

# Deduplicate by (video_id, query_lower)
seen = set()
UNIQUE = []
for r in ALL_RESULTS:
    key = (r["video_id"], r["query"].lower().strip())
    if key not in seen:
        seen.add(key)
        UNIQUE.append(r)

ALL_RESULTS = UNIQUE
print(f"Total unique predictions: {len(ALL_RESULTS)}")

# ================================================================== #
# One GENUINE independent IoU: blue kettle (P02_01)
# The GT from test_queries_annotated.json was annotated independently
# (gt_bbox=[497, 175, 753, 530] vs pred_bbox from benchmark run)
# ================================================================== #
GENUINE_IOUs = []

# Load blue kettle from full_predictions.json (has DIFFERENT gt than pred)
with open(os.path.join(ROOT, "eval/results/full_predictions.json")) as f:
    full_preds = json.load(f)

for row in full_preds:
    pred = row.get("prediction")
    gt = row.get("ground_truth")
    if pred and gt and gt.get("gt_bbox") and pred.get("pred_bbox"):
        gt_bbox = gt["gt_bbox"]
        pred_bbox = pred["pred_bbox"]
        # Only count if gt_bbox != pred_bbox (genuinely independent)
        if gt_bbox != pred_bbox:
            iou = compute_iou(pred_bbox, gt_bbox)
            GENUINE_IOUs.append({
                "query": row["query"],
                "video_id": row["video_id"],
                "pred_bbox": pred_bbox,
                "gt_bbox": gt_bbox,
                "iou": iou,
            })

print(f"Genuinely independent IoU pairs: {len(GENUINE_IOUs)}")
for r in GENUINE_IOUs:
    print(f"  {r['query']} ({r['video_id']}): pred={r['pred_bbox']} gt={r['gt_bbox']} IoU={r['iou']:.3f}")

# ================================================================== #
# Analytics
# ================================================================== #

videos = sorted(set(r["video_id"] for r in ALL_RESULTS))
brands = [r for r in ALL_RESULTS if r["query_type"] == "brand"]
objects = [r for r in ALL_RESULTS if r["query_type"] == "object"]

clip_sims = [r["clip_similarity"] for r in ALL_RESULTS if r["clip_similarity"]]
ocr_hits  = [r["ocr_frames_hit"] for r in ALL_RESULTS]
segs      = [r["valid_segments"] for r in ALL_RESULTS if r["valid_segments"] is not None]
region_scores = [r["region_clip_score"] for r in ALL_RESULTS if r["region_clip_score"] > 0]
above_thresh  = [r["frames_above_threshold"] for r in ALL_RESULTS if r["frames_above_threshold"]]

# ================================================================== #
# Figure: honest metrics dashboard
# ================================================================== #
fig = plt.figure(figsize=(18, 12))

from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ---- (1) Coverage per video ----
ax1 = fig.add_subplot(gs[0, 0])
vid_counts = {v: sum(1 for r in ALL_RESULTS if r["video_id"] == v) for v in videos}
cols_v = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(videos)]
bars = ax1.bar(vid_counts.keys(), vid_counts.values(), color=cols_v, alpha=0.85, edgecolor='k', lw=0.6)
for b, v in zip(bars, vid_counts.values()):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, str(v), ha='center', va='bottom', fontsize=10)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("Number of Queries")
ax1.set_title("(a) Queries per Video")

# ---- (2) CLIP similarity distribution ----
ax2 = fig.add_subplot(gs[0, 1])
obj_sims   = [r["clip_similarity"] for r in objects if r["clip_similarity"]]
brand_sims = [r["clip_similarity"] for r in brands if r["clip_similarity"]]
ax2.hist(obj_sims,   bins=12, alpha=0.7, label=f'Object (n={len(obj_sims)})',   color='#1f77b4', edgecolor='k', lw=0.4)
ax2.hist(brand_sims, bins=12, alpha=0.7, label=f'Brand (n={len(brand_sims)})',  color='#ff7f0e', edgecolor='k', lw=0.4)
ax2.axvline(0.18, color='red', linestyle='--', lw=1.5, label='τ=0.18')
ax2.axvline(0.20, color='purple', linestyle='--', lw=1.5, label='τ=0.20')
ax2.set_xlabel("CLIP Similarity Score")
ax2.set_ylabel("Count")
ax2.set_title("(b) CLIP Similarity Distribution")
ax2.legend(fontsize=8.5)

# ---- (3) Segments found per query ----
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(segs, bins=range(0, max(segs)+2) if segs else [0, 1], color='#2ca02c', alpha=0.85, edgecolor='k', lw=0.5)
ax3.axvline(np.mean(segs) if segs else 0, color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(segs):.1f}')
ax3.set_xlabel("Number of Temporal Segments Found")
ax3.set_ylabel("Count")
ax3.set_title("(c) Temporal Segments per Query")
ax3.legend()

# ---- (4) OCR hit rate: brand vs object ----
ax4 = fig.add_subplot(gs[1, 0])
brand_ocr = [r["ocr_frames_hit"] for r in brands]
obj_ocr   = [r["ocr_frames_hit"] for r in objects]
brand_any = np.mean([1 if h > 0 else 0 for h in brand_ocr]) * 100 if brand_ocr else 0
obj_any   = np.mean([1 if h > 0 else 0 for h in obj_ocr]) * 100 if obj_ocr else 0
xticks = ['Object\nQueries', 'Brand\nQueries']
bars = ax4.bar(xticks, [obj_any, brand_any], color=['#1f77b4', '#ff7f0e'], alpha=0.85, edgecolor='k', lw=0.8, width=0.4)
for b, v in zip(bars, [obj_any, brand_any]):
    ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_ylabel("OCR Hit Rate (%)")
ax4.set_title("(d) OCR Detection Rate\n(Brand vs. Object Queries)")
ax4.set_ylim(0, 115)

# ---- (5) Region CLIP score vs frame CLIP sim ----
ax5 = fig.add_subplot(gs[1, 1])
frame_sims_r = [r["clip_similarity"] for r in ALL_RESULTS if r["region_clip_score"] > 0]
region_scores_r = [r["region_clip_score"] for r in ALL_RESULTS if r["region_clip_score"] > 0]
types_r = [r["query_type"] for r in ALL_RESULTS if r["region_clip_score"] > 0]
colors_r = ['#ff7f0e' if t == 'brand' else '#1f77b4' for t in types_r]
ax5.scatter(frame_sims_r, region_scores_r, c=colors_r, alpha=0.7, edgecolor='k', lw=0.5, s=70)
ax5.plot([0.14, 0.35], [0.14, 0.35], 'r--', lw=1.2, alpha=0.6, label='y=x (no improvement)')
ax5.set_xlabel("Frame CLIP Similarity")
ax5.set_ylabel("Region CLIP Score (after crop scoring)")
ax5.set_title("(e) Frame vs. Region Score\n(Region > Frame = better localization)")
brand_patch = mpatches.Patch(color='#ff7f0e', label='Brand')
obj_patch   = mpatches.Patch(color='#1f77b4', label='Object')
ax5.legend(handles=[brand_patch, obj_patch, plt.Line2D([0], [0], color='red', linestyle='--', lw=1.2, label='y=x')],
           fontsize=8.5)

# ---- (6) Frames above threshold ----
ax6 = fig.add_subplot(gs[1, 2])
above_log = [np.log10(f + 1) for f in above_thresh]
ax6.hist(above_log, bins=15, color='#9467bd', alpha=0.85, edgecolor='k', lw=0.5)
xtick_vals = [1, 5, 10, 50, 100, 500, 1000]
ax6.set_xticks([np.log10(v) for v in xtick_vals])
ax6.set_xticklabels([str(v) for v in xtick_vals])
ax6.set_xlabel("Frames Above Threshold (log scale)")
ax6.set_ylabel("Count")
ax6.set_title("(f) Retrieval Coverage\n(frames above τ per query)")

# ---- (7) Genuine IoU bar (independent annotations) ----
ax7 = fig.add_subplot(gs[2, :2])
if GENUINE_IOUs:
    g_labels = [f"{r['query']}\n({r['video_id']})" for r in GENUINE_IOUs]
    g_ious = [r['iou'] for r in GENUINE_IOUs]
    colors_g = ['#2ca02c' if iou >= 0.5 else ('#ff7f0e' if iou >= 0.25 else '#d62728') for iou in g_ious]
    gb = ax7.bar(g_labels, g_ious, color=colors_g, alpha=0.85, edgecolor='k', lw=0.8)
    ax7.axhline(0.25, color='gray', linestyle='--', lw=1.2, label='IoU=0.25')
    ax7.axhline(0.50, color='black', linestyle='--', lw=1.2, label='IoU=0.50')
    for b, v in zip(gb, g_ious):
        ax7.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax7.set_ylim(0, 1.1)
    ax7.set_ylabel("IoU Score")
    ax7.set_title("(g) Genuine Independent IoU (GT annotated separately from predictions)")
    ax7.legend()
else:
    ax7.text(0.5, 0.5, "No independently annotated GT\nbboxes available for IoU computation.\n\nGT was derived from predictions —\ncircular evaluation.",
             transform=ax7.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.set_title("(g) Genuine IoU — Requires Independent GT Annotation")

# ---- (8) Summary table ----
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
tdata = [
    ["Videos evaluated", f"{len(videos)} ({', '.join(videos)})"],
    ["Total queries", f"{len(ALL_RESULTS)}"],
    ["Object queries", f"{len(objects)}"],
    ["Brand queries", f"{len(brands)}"],
    ["Mean CLIP similarity", f"{np.mean(clip_sims):.3f}"],
    ["Mean segments found", f"{np.mean(segs):.1f}"],
    ["OCR hit rate (brand)", f"{brand_any:.0f}%"],
    ["OCR hit rate (object)", f"{obj_any:.0f}%"],
    ["Region > Frame score", f"{np.mean([1 if r>f else 0 for r,f in zip(region_scores_r, frame_sims_r)])*100:.0f}% of queries"],
    ["Genuine IoU pairs", f"{len(GENUINE_IOUs)}"],
]
if GENUINE_IOUs:
    tdata.append(["Mean genuine IoU", f"{np.mean([r['iou'] for r in GENUINE_IOUs]):.3f}"])
table = ax8.table(cellText=tdata, colLabels=["Metric", "Value"],
                  cellLoc='left', loc='center',
                  colWidths=[0.48, 0.52])
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)
ax8.set_title("(h) Summary Statistics", fontsize=10)

fig.suptitle(
    f"Textual-REN: Honest Multi-Video Evaluation\n"
    f"({len(ALL_RESULTS)} queries across {len(videos)} videos: {', '.join(videos)})",
    fontsize=13, fontweight='bold'
)

out = os.path.join(OUT_DIR, "honest_metrics_dashboard.png")
plt.savefig(out, bbox_inches='tight')
plt.close()
print(f"\nDashboard saved: {out}")

# ================================================================== #
# Print final honest assessment
# ================================================================== #
print("\n" + "="*70)
print("HONEST ASSESSMENT FOR THE PAPER")
print("="*70)
print(f"Videos covered: {', '.join(videos)}")
print(f"Total queries: {len(ALL_RESULTS)} ({len(objects)} object, {len(brands)} brand)")
print(f"Mean CLIP similarity: {np.mean(clip_sims):.3f}")
print(f"Mean temporal segments found: {np.mean(segs):.1f}")
print(f"OCR hit rate — brand: {brand_any:.0f}% | object: {obj_any:.0f}%")
print(f"Region scoring improved over frame similarity: "
      f"{np.mean([1 if r>f else 0 for r,f in zip(region_scores_r, frame_sims_r)])*100:.0f}% of queries")
print(f"\nGenuinely independent IoU pairs: {len(GENUINE_IOUs)}")
for r in GENUINE_IOUs:
    print(f"  {r['query']}: IoU={r['iou']:.3f}")
print("\nACTION REQUIRED for proper paper:")
print("  1. Annotate GT bboxes INDEPENDENTLY (without looking at model output)")
print("  2. Use Ego4D VQ2D official test set with public annotations")
print("  3. OR: conduct a user study — ask annotators 'is this correct?' (Yes/No)")
print("="*70)
