"""
Ground-truth annotation tool for Textual-REN evaluation.

Step 1: Run the full model on all queries to get candidate frames.
Step 2: This script opens each debug_last_frame.jpg, shows the predicted bbox,
        and lets you type the corrected [x, y, w, h] and timestamp.

Usage:
    # First run benchmark to generate predictions
    python benchmark.py --queries test_queries.json --mode full --output results/

    # Then annotate ground truth interactively
    python annotate_gt.py --queries test_queries.json \
                          --predictions results/full_predictions.json \
                          --output test_queries_annotated.json

Controls during annotation:
    Enter: accept predicted bbox as GT
    n    : skip (no GT for this query)
    <x>,<y>,<w>,<h>: type corrected bbox manually
    t<timestamp>: type corrected timestamp (e.g. t174.5)
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np


def draw_bbox(img: np.ndarray, bbox, color=(0, 255, 0), label='') -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(out, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def annotate_interactive(query_entry: dict, pred: dict,
                         results_root: str) -> dict:
    """Show predicted frame, get user correction. Returns updated entry."""
    q = query_entry['query']
    video_id = query_entry['video_id']

    debug_path = os.path.join(
        results_root, 'full', video_id,
        q.replace(' ', '_'), 'debug_last_frame.jpg'
    )

    print(f"\n{'─'*60}")
    print(f"Query: '{q}'  ({query_entry['query_type']})")

    if pred is None:
        print("  Pipeline failed — no prediction.")
        inp = input("  Enter GT bbox x,y,w,h and timestamp (or 'n' to skip): ").strip()
        if inp == 'n':
            return query_entry
        try:
            parts = inp.split()
            bbox = [int(v) for v in parts[0].split(',')]
            ts = float(parts[1]) if len(parts) > 1 else None
            query_entry['gt_bbox'] = bbox
            query_entry['gt_timestamp'] = ts
        except Exception:
            print("  Invalid input — skipped.")
        return query_entry

    pred_bbox = pred.get('pred_bbox')
    pred_ts   = pred.get('last_frame_timestamp')
    print(f"  Predicted bbox: {pred_bbox}  timestamp: {pred_ts}s")

    if os.path.exists(debug_path):
        if pred_bbox:
            # Draw bbox on a copy and save it, then open in Windows Photos
            img = cv2.imread(debug_path)
            if img is not None:
                vis = draw_bbox(img, pred_bbox, color=(0, 255, 0), label='predicted')
                vis_path = debug_path.replace('.jpg', '_annotate.jpg')
                cv2.imwrite(vis_path, vis)
                print(f"  Opening: {vis_path}")
                os.startfile(os.path.abspath(vis_path))
            else:
                print(f"  [warn] Could not read image: {debug_path}")
        else:
            os.startfile(os.path.abspath(debug_path))
            print(f"  Opened: {debug_path}")
        input("  (Look at the image in Photos, then press Enter here to continue...)")
    else:
        print(f"  [warn] Debug image not found: {debug_path}")
        print(f"  Expected: {debug_path}")

    print("  Options:")
    print("    Enter       → accept predicted bbox & timestamp as GT")
    print("    x,y,w,h     → enter correct bbox")
    print("    x,y,w,h t<s>→ correct bbox + timestamp")
    print("    n           → skip (no GT)")
    inp = input("  > ").strip()

    if inp == '':
        query_entry['gt_bbox'] = pred_bbox
        query_entry['gt_timestamp'] = pred_ts
        print(f"  Accepted: bbox={pred_bbox}  ts={pred_ts}s")
    elif inp == 'n':
        print("  Skipped.")
    else:
        try:
            parts = inp.split()
            bbox = [int(v) for v in parts[0].split(',')]
            query_entry['gt_bbox'] = bbox
            ts = None
            for part in parts[1:]:
                if part.startswith('t'):
                    ts = float(part[1:])
            query_entry['gt_timestamp'] = ts if ts is not None else pred_ts
            print(f"  Saved: bbox={bbox}  ts={query_entry['gt_timestamp']}s")
        except Exception as e:
            print(f"  Parse error ({e}) — skipped.")

    return query_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries',     default='test_queries.json')
    parser.add_argument('--predictions', default='results/full_predictions.json')
    parser.add_argument('--results-root', default='results/', dest='results_root')
    parser.add_argument('--output',      default='test_queries_annotated.json')
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    queries_path = os.path.join(base, args.queries)
    preds_path   = os.path.join(base, args.predictions)

    with open(queries_path) as f:
        queries = json.load(f)

    preds = [None] * len(queries)
    if os.path.exists(preds_path):
        with open(preds_path) as f:
            data = json.load(f)
        preds = [d['prediction'] for d in data]
    else:
        print(f"[warn] Predictions file not found: {preds_path}")
        print("       Run benchmark.py --mode full first.")

    print(f"Annotating {len(queries)} queries...")
    print("For each query: check the predicted frame, correct if needed.\n")

    updated = []
    for i, (q, pred) in enumerate(zip(queries, preds)):
        print(f"[{i+1}/{len(queries)}]", end='')
        q_updated = annotate_interactive(q, pred, args.results_root)
        updated.append(q_updated)

    out_path = os.path.join(base, args.output)
    with open(out_path, 'w') as f:
        json.dump(updated, f, indent=2)
    print(f"\nAnnotated queries saved: {out_path}")
    n_gt = sum(1 for q in updated if q.get('gt_bbox') is not None)
    print(f"Queries with GT bbox: {n_gt}/{len(updated)}")


if __name__ == '__main__':
    main()
