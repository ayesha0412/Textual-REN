"""
Evaluation metrics for Textual-REN video object localization.

Metrics:
  - IoU (bbox overlap)
  - Success Rate @ IoU threshold (mAP-style)
  - Temporal Error (seconds between predicted and GT timestamp)
  - Temporal Accuracy (within N seconds of GT)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_iou(pred_bbox: List[int], gt_bbox: List[int]) -> float:
    """
    Compute IoU between two bboxes in [x, y, w, h] format.
    Returns 0.0 if either bbox is invalid.
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox
    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 0.0

    inter_x1 = max(px, gx)
    inter_y1 = max(py, gy)
    inter_x2 = min(px + pw, gx + gw)
    inter_y2 = min(py + ph, gy + gh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = pw * ph + gw * gh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def temporal_error(pred_ts: float, gt_ts: float) -> float:
    """Absolute temporal error in seconds."""
    if pred_ts is None or gt_ts is None:
        return float('inf')
    return abs(pred_ts - gt_ts)


def compute_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Compute all evaluation metrics from a list of predictions vs ground truth.

    predictions: list of result dicts from query_indexed.py (result.json)
    ground_truth: list of test_queries.json entries with gt_bbox, gt_timestamp

    Returns dict of aggregated metrics.
    """
    ious, temp_errors, successes = [], [], {}
    iou_thresholds = [0.10, 0.25, 0.50, 0.75]
    for t in iou_thresholds:
        successes[t] = []

    temporal_windows = [1, 2, 5, 10]
    temporal_acc = {w: [] for w in temporal_windows}

    valid = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred is None:
            # Query failed (no frames above threshold)
            for t in iou_thresholds:
                successes[t].append(0)
            for w in temporal_windows:
                temporal_acc[w].append(0)
            ious.append(0.0)
            temp_errors.append(float('inf'))
            continue

        gt_bbox = gt.get('gt_bbox')
        gt_ts   = gt.get('gt_timestamp')
        pred_bbox = pred.get('pred_bbox')
        pred_ts   = pred.get('last_frame_timestamp')

        iou = compute_iou(pred_bbox, gt_bbox) if gt_bbox else None
        terr = temporal_error(pred_ts, gt_ts) if gt_ts is not None else None

        if iou is not None:
            ious.append(iou)
            for t in iou_thresholds:
                successes[t].append(1 if iou >= t else 0)
            valid += 1

        if terr is not None:
            temp_errors.append(terr)
            for w in temporal_windows:
                temporal_acc[w].append(1 if terr <= w else 0)

    n = len(predictions)
    metrics = {
        'n_queries': n,
        'n_valid_gt': valid,
        'mIoU': float(np.mean(ious)) if ious else 0.0,
    }
    for t in iou_thresholds:
        key = f'Success@{int(t*100)}'
        metrics[key] = float(np.mean(successes[t])) * 100 if successes[t] else 0.0

    finite_errors = [e for e in temp_errors if e != float('inf')]
    metrics['temporal_error_mean'] = float(np.mean(finite_errors)) if finite_errors else float('inf')
    metrics['temporal_error_median'] = float(np.median(finite_errors)) if finite_errors else float('inf')
    for w in temporal_windows:
        metrics[f'temporal_acc@{w}s'] = float(np.mean(temporal_acc[w])) * 100 if temporal_acc[w] else 0.0

    return metrics


def success_rate_curve(ious: List[float], thresholds=None) -> Tuple[List[float], List[float]]:
    """
    Compute success rate at every IoU threshold for plotting.
    Returns (thresholds, success_rates).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05).tolist()
    rates = [float(np.mean([1 if iou >= t else 0 for iou in ious])) * 100
             for t in thresholds]
    return thresholds, rates


def print_metrics_table(metrics: Dict, label: str = ""):
    """Pretty-print metrics as a table row."""
    print(f"\n{'─'*70}")
    if label:
        print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  mIoU          : {metrics.get('mIoU', 0)*100:.1f}%")
    print(f"  Success@25    : {metrics.get('Success@25', 0):.1f}%")
    print(f"  Success@50    : {metrics.get('Success@50', 0):.1f}%")
    print(f"  Success@75    : {metrics.get('Success@75', 0):.1f}%")
    print(f"  Temp. Error   : {metrics.get('temporal_error_mean', 0):.2f}s (mean) | "
          f"{metrics.get('temporal_error_median', 0):.2f}s (median)")
    print(f"  Temp. Acc@2s  : {metrics.get('temporal_acc@2s', 0):.1f}%")
    print(f"  Temp. Acc@5s  : {metrics.get('temporal_acc@5s', 0):.1f}%")
    print(f"{'─'*70}")
