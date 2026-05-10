"""
Aggregate ALL available Textual-REN predictions + ground truth
across every video and compute real evaluation metrics.

Sources:
  epic_results/         - P01_03, P01_02, P01_05 predictions
  query_results/P02_01/ - P02_01 predictions (11 queries)
  query_results/P04_01_plate/ - P04_01 prediction
  eval/results/full/    - P02_01 benchmark predictions with GT
  eval/annotated_testset.json  - Gold GT for P01 videos
  eval/test_queries_annotated.json - GT for P02_01 queries
  eval/results/full_predictions.json - P02_01 predictions+GT

Outputs:
  eval/results/aggregated_metrics.json
  eval/results/aggregated_per_query.json
  paper_figures/real_fig*.png  (replaces estimated figures)
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = r"D:\REN Project\REN"
sys.path.insert(0, os.path.join(ROOT, "eval"))
from metrics import compute_iou, temporal_error, success_rate_curve

OUT_DIR = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10.5,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
})

# ================================================================== #
# STEP 1: Build unified ground truth index
# ================================================================== #

def load_gt():
    gt = {}  # (video_id, query_lower) -> gt dict

    # --- Gold annotations (P01 videos) ---
    with open(os.path.join(ROOT, "eval/annotated_testset.json")) as f:
        for entry in json.load(f):
            vid = entry["video_id"]
            q = entry["query"].lower().strip()
            gt[(vid, q)] = entry

    # --- P02_01 annotated queries ---
    with open(os.path.join(ROOT, "eval/test_queries_annotated.json")) as f:
        for entry in json.load(f):
            vid = entry["video_id"]
            q = entry["query"].lower().strip()
            # only overwrite if GT is more complete
            key = (vid, q)
            if key not in gt or (gt[key].get("gt_bbox") is None and entry.get("gt_bbox")):
                gt[key] = entry

    # --- full_predictions.json also carries GT ---
    with open(os.path.join(ROOT, "eval/results/full_predictions.json")) as f:
        for row in json.load(f):
            if row.get("ground_truth"):
                entry = row["ground_truth"]
                vid = entry.get("video_id", "P02_01")
                q = entry["query"].lower().strip()
                key = (vid, q)
                if key not in gt or (gt[key].get("gt_bbox") is None and entry.get("gt_bbox")):
                    gt[key] = entry

    return gt


# ================================================================== #
# STEP 2: Collect all predictions
# ================================================================== #

def load_all_predictions():
    preds = []  # list of dicts with pred + gt_key

    def add_from_result_json(path, video_id):
        if not os.path.exists(path):
            return
        with open(path) as f:
            result = json.load(f)
        query = result.get("query", "").lower().strip()
        preds.append({
            "video_id": video_id,
            "query": query,
            "query_raw": result.get("query", ""),
            "query_type": "brand" if result.get("ocr_score", 0) > 0.5 else "object",
            "pred_frame_idx": result.get("last_frame_idx"),
            "pred_timestamp": result.get("last_frame_timestamp"),
            "pred_bbox": result.get("pred_bbox"),
            "clip_similarity": result.get("clip_similarity"),
            "fused_similarity": result.get("fused_similarity", result.get("clip_similarity")),
            "ocr_score": result.get("ocr_score", 0),
            "ocr_frames_hit": result.get("ocr_frames_hit", 0),
            "valid_segments": result.get("valid_segments"),
            "frames_above_threshold": result.get("frames_above_threshold"),
            "region_clip_score": result.get("region_clip_score", 0),
            "source": path,
        })

    # --- epic_results (P01 videos) ---
    epic_map = {
        "black dustbin": "P01_03",
        "fork utensil":  "P01_02",
        "fork":          "P01_02",
        "loaf of bread": "P01_05",
        "strainer":      "P01_05",
    }
    epic_results_dir = os.path.join(ROOT, "epic_results")
    for folder in os.listdir(epic_results_dir):
        rjson = os.path.join(epic_results_dir, folder, "result.json")
        if os.path.exists(rjson):
            with open(rjson) as f:
                result = json.load(f)
            query = result.get("query", folder).lower().strip()
            vid = epic_map.get(query, "P01_03")
            add_from_result_json(rjson, vid)

    # --- query_results/P02_01/ ---
    p02_dir = os.path.join(ROOT, "query_results/P02_01")
    for folder in os.listdir(p02_dir):
        rjson = os.path.join(p02_dir, folder, "result.json")
        add_from_result_json(rjson, "P02_01")

    # --- query_results/P04_01_plate ---
    add_from_result_json(os.path.join(ROOT, "query_results/P04_01_plate/result.json"), "P04_01")

    # --- eval/results/full/ (P02_01 benchmark run — use these as authoritative for P02_01) ---
    full_dir = os.path.join(ROOT, "eval/results/full/P02_01")
    if os.path.exists(full_dir):
        for folder in os.listdir(full_dir):
            rjson = os.path.join(full_dir, folder, "result.json")
            if os.path.exists(rjson):
                with open(rjson) as f:
                    result = json.load(f)
                q = result.get("query", "").lower().strip()
                # Check if we already have this P02_01 query; if so, overwrite with benchmark version
                preds = [p for p in preds if not (p["video_id"] == "P02_01" and p["query"] == q)]
                add_from_result_json(rjson, "P02_01")

    # --- annotated_testset: orange juice bottle (P01_01) ---
    oj_json = os.path.join(ROOT, "eval/_tmp_eval/P01_01/orange_juice_bottle/result.json")
    add_from_result_json(oj_json, "P01_01")

    # --- eval/evaluation_results/ (P01_02: dustbin, sponge) ---
    for q_folder in ["dustbin", "sponge"]:
        rjson = os.path.join(ROOT, f"eval/evaluation_results/P01_02/{q_folder}/result.json")
        add_from_result_json(rjson, "P01_02")

    return preds


# ================================================================== #
# STEP 3: Match + compute metrics
# ================================================================== #

def match_and_compute(preds, gt_index):
    per_query = []
    iou_thresholds = [0.10, 0.25, 0.50, 0.75]

    for pred in preds:
        vid = pred["video_id"]
        q = pred["query"]
        gt = gt_index.get((vid, q)) or gt_index.get((vid, q.replace("_", " ")))

        gt_bbox = gt.get("gt_bbox") if gt else None
        gt_ts   = gt.get("gt_timestamp") if gt else None
        gt_frame = gt.get("gt_frame_idx") if gt else None

        pred_bbox = pred.get("pred_bbox")
        pred_ts   = pred.get("pred_timestamp")
        pred_frame = pred.get("pred_frame_idx")

        has_spatial_gt = gt_bbox is not None
        has_temporal_gt = gt_ts is not None or gt_frame is not None

        # IoU
        iou = compute_iou(pred_bbox, gt_bbox) if (pred_bbox and gt_bbox) else None

        # Temporal error (seconds)
        terr = None
        if pred_ts is not None and gt_ts is not None:
            terr = abs(pred_ts - gt_ts)
        elif pred_frame is not None and gt_frame is not None:
            fps = pred.get("fps", 59.94) or 59.94
            terr = abs(pred_frame - gt_frame) / fps

        successes = {}
        for t in iou_thresholds:
            if iou is not None:
                successes[f"S@{int(t*100)}"] = 1 if iou >= t else 0
            else:
                successes[f"S@{int(t*100)}"] = None

        per_query.append({
            **pred,
            "gt_bbox": gt_bbox,
            "gt_timestamp": gt_ts,
            "gt_frame_idx": gt_frame,
            "has_spatial_gt": has_spatial_gt,
            "has_temporal_gt": has_temporal_gt,
            "iou": iou,
            "temporal_error_s": terr,
            **successes,
        })

    return per_query


# ================================================================== #
# STEP 4: Aggregate and print
# ================================================================== #

def aggregate(per_query):
    spatial = [r for r in per_query if r["has_spatial_gt"] and r["iou"] is not None]
    temporal = [r for r in per_query if r["temporal_error_s"] is not None]

    ious = [r["iou"] for r in spatial]

    thresholds = [0.10, 0.25, 0.50, 0.75]
    by_type = {"object": [], "brand": [], "compositional": []}
    for r in spatial:
        qt = r.get("query_type", "object")
        if qt not in by_type:
            qt = "object"
        by_type[qt].append(r["iou"])

    metrics = {
        "n_total": len(per_query),
        "n_with_spatial_gt": len(spatial),
        "n_with_temporal_gt": len(temporal),
        "mIoU": float(np.mean(ious)) if ious else None,
        "median_IoU": float(np.median(ious)) if ious else None,
    }
    for t in thresholds:
        vals = [r[f"S@{int(t*100)}"] for r in spatial if r[f"S@{int(t*100)}"] is not None]
        metrics[f"Success@{int(t*100)}"] = float(np.mean(vals)) * 100 if vals else None

    terrs = [r["temporal_error_s"] for r in temporal]
    metrics["temporal_error_mean_s"] = float(np.mean(terrs)) if terrs else None
    metrics["temporal_error_median_s"] = float(np.median(terrs)) if terrs else None
    for w in [1, 2, 5, 10]:
        acc = [1 if e <= w else 0 for e in terrs]
        metrics[f"temporal_acc@{w}s"] = float(np.mean(acc)) * 100 if acc else None

    # Per query type
    for qt, qt_ious in by_type.items():
        if qt_ious:
            metrics[f"mIoU_{qt}"] = float(np.mean(qt_ious))
            metrics[f"Success@50_{qt}"] = float(np.mean([1 if v >= 0.5 else 0 for v in qt_ious])) * 100

    return metrics


# ================================================================== #
# STEP 5: Plot REAL figures
# ================================================================== #

def plot_real_results(per_query, metrics):
    spatial = [r for r in per_query if r["has_spatial_gt"] and r["iou"] is not None]
    ious = [r["iou"] for r in spatial]
    labels = [r["query_raw"] for r in spatial]
    types = [r.get("query_type", "object") for r in spatial]
    temporal = [r for r in per_query if r["temporal_error_s"] is not None]
    terrs = [r["temporal_error_s"] for r in temporal]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- (1) Per-query IoU bar chart ----
    ax = axes[0, 0]
    colors = ['#d62728' if iou < 0.25 else ('#ff7f0e' if iou < 0.50 else '#2ca02c') for iou in ious]
    bars = ax.bar(range(len(ious)), ious, color=colors, alpha=0.85, edgecolor='k', linewidth=0.6)
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=1.2, label='IoU=0.25')
    ax.axhline(0.50, color='black', linestyle='--', linewidth=1.2, label='IoU=0.50')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel("IoU Score")
    ax.set_title("(a) Per-Query IoU (Real Results)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # ---- (2) Success rate curve ----
    ax = axes[0, 1]
    thresholds = np.arange(0.05, 1.0, 0.05)
    sr = [np.mean([1 if iou >= t else 0 for iou in ious]) * 100 for t in thresholds]
    ax.plot(thresholds, sr, 'o-', color='#1f77b4', linewidth=2.5, markersize=5, label='Textual-REN (Ours)')
    ax.axvline(0.25, color='gray', linestyle=':', linewidth=1.2)
    ax.axvline(0.50, color='gray', linestyle=':', linewidth=1.2)
    ax.text(0.26, 95, 'IoU=0.25', fontsize=9, color='gray')
    ax.text(0.51, 95, 'IoU=0.50', fontsize=9, color='gray')
    # Annotate key values
    for t, s in zip([0.25, 0.50, 0.75], [metrics.get('Success@25'), metrics.get('Success@50'), metrics.get('Success@75')]):
        if s is not None:
            ax.plot(t, s, 'ro', markersize=8, zorder=5)
            ax.annotate(f'{s:.0f}%', (t, s), textcoords="offset points", xytext=(5, -15), fontsize=9, color='red')
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(b) Success Rate Curve (Real Data)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 105)
    ax.legend()

    # ---- (3) IoU by query type ----
    ax = axes[0, 2]
    type_groups = {}
    for r in spatial:
        qt = r.get("query_type", "object")
        type_groups.setdefault(qt, []).append(r["iou"])
    qt_names = list(type_groups.keys())
    qt_means = [np.mean(v) * 100 for v in type_groups.values()]
    qt_s50   = [np.mean([1 if v >= 0.5 else 0 for v in vals]) * 100 for vals in type_groups.values()]
    x = np.arange(len(qt_names))
    w = 0.38
    b1 = ax.bar(x - w/2, qt_means, w, label='mIoU (%)', color='#1f77b4', alpha=0.85, edgecolor='k', lw=0.6)
    b2 = ax.bar(x + w/2, qt_s50,   w, label='Success@50 (%)', color='#d62728', alpha=0.85, edgecolor='k', lw=0.6)
    for b, v in [(bb, vv) for bs, vs in [(b1, qt_means), (b2, qt_s50)] for bb, vv in zip(bs, vs)]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(n={len(type_groups[n])})' for n in qt_names], fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_title("(c) Performance by Query Type")
    ax.legend()
    ax.set_ylim(0, 115)

    # ---- (4) IoU distribution histogram ----
    ax = axes[1, 0]
    bins = np.arange(0, 1.1, 0.1)
    ax.hist(ious, bins=bins, color='#1f77b4', alpha=0.8, edgecolor='k', linewidth=0.8)
    ax.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious):.2f}')
    ax.axvline(np.median(ious), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.2f}')
    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Number of Queries")
    ax.set_title("(d) IoU Distribution")
    ax.legend()
    ax.set_xlim(0, 1.0)

    # ---- (5) Temporal error distribution ----
    ax = axes[1, 1]
    finite_terrs = [e for e in terrs if e < 100]
    if finite_terrs:
        ax.hist(finite_terrs, bins=20, color='#ff7f0e', alpha=0.8, edgecolor='k', linewidth=0.8)
        ax.axvline(np.mean(finite_terrs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(finite_terrs):.1f}s')
        ax.axvline(np.median(finite_terrs), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(finite_terrs):.1f}s')
        # temporal acc windows
        for w, col in [(1, 'blue'), (5, 'purple')]:
            acc = np.mean([1 if e <= w else 0 for e in finite_terrs]) * 100
            ax.axvline(w, color=col, linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'Acc@{w}s: {acc:.0f}%')
    ax.set_xlabel("Temporal Error (seconds)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("(e) Temporal Error Distribution")
    ax.legend(fontsize=8.5)

    # ---- (6) Summary metrics bar ----
    ax = axes[1, 2]
    metric_names = ['mIoU\n(×100)', 'Success\n@25', 'Success\n@50', 'Success\n@75', 'Temp.Acc\n@5s']
    vals = [
        metrics.get('mIoU', 0) * 100 if metrics.get('mIoU') else 0,
        metrics.get('Success@25', 0) or 0,
        metrics.get('Success@50', 0) or 0,
        metrics.get('Success@75', 0) or 0,
        metrics.get('temporal_acc@5s', 0) or 0,
    ]
    cols = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax.bar(metric_names, vals, color=cols, alpha=0.85, edgecolor='k', linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Score (%)")
    ax.set_title("(f) Summary Metrics (Real Results)")
    ax.set_ylim(0, 115)

    fig.suptitle(
        f"Textual-REN: Real Evaluation Results\n"
        f"n={len(per_query)} queries | {len(spatial)} with spatial GT | {len(temporal)} with temporal GT",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "real_results_dashboard.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Dashboard saved: {out}")


# ================================================================== #
# STEP 6: Print a publication-ready table
# ================================================================== #

def print_full_table(per_query, metrics):
    print("\n" + "="*90)
    print("TEXTUAL-REN: COMPLETE EVALUATION RESULTS (Real Data)")
    print("="*90)
    print(f"{'Query':<30} {'Video':>6} {'Type':>12} {'Pred TS':>8} {'GT TS':>7} {'ΔT(s)':>7} {'Pred BBox':>22} {'IoU':>6}")
    print("-"*90)
    for r in per_query:
        iou_str = f"{r['iou']:.3f}" if r['iou'] is not None else "  N/A"
        terr_str = f"{r['temporal_error_s']:.1f}" if r['temporal_error_s'] is not None else "  N/A"
        pts = f"{r['pred_timestamp']:.1f}" if r.get('pred_timestamp') else "  N/A"
        gts = f"{r.get('gt_timestamp', 'N/A') or 'N/A'}"
        bbox_str = str(r.get('pred_bbox', 'N/A'))[:21]
        print(f"{r['query_raw']:<30} {r['video_id']:>6} {r.get('query_type','object'):>12} "
              f"{pts:>8} {gts:>7} {terr_str:>7} {bbox_str:>22} {iou_str:>6}")

    print("="*90)
    print(f"\n{'AGGREGATE METRICS':}")
    print(f"  Total queries        : {metrics['n_total']}")
    print(f"  With spatial GT      : {metrics['n_with_spatial_gt']}")
    print(f"  With temporal GT     : {metrics['n_with_temporal_gt']}")
    print(f"  mIoU                 : {metrics['mIoU']*100:.1f}%" if metrics.get('mIoU') else "  mIoU: N/A (no spatial GT)")
    for t in [10, 25, 50, 75]:
        v = metrics.get(f'Success@{t}')
        print(f"  Success@{t}          : {v:.1f}%" if v is not None else f"  Success@{t}: N/A")
    print(f"  Temporal Error (mean): {metrics.get('temporal_error_mean_s', 'N/A'):.2f}s" if metrics.get('temporal_error_mean_s') else "  Temporal Error: N/A")
    for w in [1, 2, 5]:
        v = metrics.get(f'temporal_acc@{w}s')
        print(f"  Temporal Acc@{w}s     : {v:.1f}%" if v is not None else f"  Temporal Acc@{w}s: N/A")
    print("="*90)

    # Per-type breakdown
    for qt in ['object', 'brand', 'compositional']:
        miou = metrics.get(f'mIoU_{qt}')
        s50  = metrics.get(f'Success@50_{qt}')
        if miou is not None:
            print(f"  {qt:>14} queries: mIoU={miou*100:.1f}%  Success@50={s50:.0f}%")


# ================================================================== #
# MAIN
# ================================================================== #

if __name__ == "__main__":
    print("Loading ground truth...")
    gt_index = load_gt()
    print(f"  {len(gt_index)} GT entries loaded")

    print("Loading all predictions...")
    preds = load_all_predictions()
    print(f"  {len(preds)} predictions loaded")

    print("Matching predictions to GT...")
    per_query = match_and_compute(preds, gt_index)

    print("Computing metrics...")
    metrics = aggregate(per_query)

    print_full_table(per_query, metrics)

    print("\nGenerating dashboard figure...")
    plot_real_results(per_query, metrics)

    # Save JSON
    out_metrics = os.path.join(ROOT, "eval/results/aggregated_metrics.json")
    out_perq = os.path.join(ROOT, "eval/results/aggregated_per_query.json")
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(out_perq, 'w') as f:
        json.dump(per_query, f, indent=2, default=str)
    print(f"\nResults saved:")
    print(f"  {out_metrics}")
    print(f"  {out_perq}")
    print(f"  {OUT_DIR}/real_results_dashboard.png")
