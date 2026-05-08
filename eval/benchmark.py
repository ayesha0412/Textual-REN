"""
Textual-REN Evaluation Benchmark.

Runs all queries in test_queries.json across multiple configurations:
  - full          : complete Textual-REN pipeline
  - no_comp       : ablation — no compositional scoring
  - no_ocr        : ablation — no OCR fusion
  - no_verify     : ablation — no frame verification
  - use_strongest : ablation — strongest segment instead of last occurrence
  - clip_only     : baseline — single CLIP argmax, no temporal logic

Usage:
    python benchmark.py --queries test_queries.json \
                        --config ../text_query/config.yaml \
                        --output results/

    # Run a single mode:
    python benchmark.py --queries test_queries.json --mode full
    python benchmark.py --queries test_queries.json --mode no_comp
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text_query'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np

from metrics import compute_metrics, print_metrics_table


MODES = {
    'full':          dict(ablation_no_compositional=False, ablation_no_ocr=False,
                          ablation_no_verification=False, ablation_use_strongest=False),
    'no_comp':       dict(ablation_no_compositional=True,  ablation_no_ocr=False,
                          ablation_no_verification=False, ablation_use_strongest=False),
    'no_ocr':        dict(ablation_no_compositional=False, ablation_no_ocr=True,
                          ablation_no_verification=False, ablation_use_strongest=False),
    'no_verify':     dict(ablation_no_compositional=False, ablation_no_ocr=False,
                          ablation_no_verification=True,  ablation_use_strongest=False),
    'use_strongest': dict(ablation_no_compositional=False, ablation_no_ocr=False,
                          ablation_no_verification=False, ablation_use_strongest=True),
    'clip_only':     dict(ablation_no_compositional=True,  ablation_no_ocr=True,
                          ablation_no_verification=True,  ablation_use_strongest=True),
}

MODE_LABELS = {
    'full':          'Textual-REN (Ours)',
    'no_comp':       'w/o Compositional Scoring',
    'no_ocr':        'w/o OCR Fusion',
    'no_verify':     'w/o Frame Verification',
    'use_strongest': 'w/o Last-Occurrence (Strongest)',
    'clip_only':     'CLIP Baseline',
}


def run_query_safe(engine, query_entry: Dict, output_dir: str,
                   mode_flags: Dict) -> Optional[Dict]:
    """Run a single query, return result dict or None on failure."""
    try:
        from query_indexed import IndexedQueryEngine
        result = engine.query(
            text_query=query_entry['query'],
            video_path=query_entry['video_path'],
            output_dir=os.path.join(output_dir, query_entry['query'].replace(' ', '_')),
            **mode_flags,
        )
        # Attach the predicted bbox from debug info (SAM2 result)
        # The bbox is saved in result.json; we also need it here.
        result_path = os.path.join(
            output_dir,
            query_entry['query'].replace(' ', '_'),
            'result.json'
        )
        if os.path.exists(result_path):
            with open(result_path) as f:
                saved = json.load(f)
            result['pred_bbox'] = saved.get('pred_bbox')
        return result
    except RuntimeError as e:
        print(f"  [FAILED] {query_entry['query']}: {e}")
        return None
    except Exception:
        traceback.print_exc()
        return None


def run_mode(mode: str, queries: List[Dict], config: Dict,
             output_root: str) -> List[Optional[Dict]]:
    """Run all queries in a single mode, return predictions."""
    from query_indexed import IndexedQueryEngine

    print(f"\n{'='*70}")
    print(f"  MODE: {MODE_LABELS[mode]}")
    print(f"{'='*70}")

    mode_flags = MODES[mode]
    predictions = []

    # Group queries by index_dir so we load each index once
    by_index: Dict[str, List] = {}
    for q in queries:
        by_index.setdefault(q['index_dir'], []).append(q)

    for index_dir, index_queries in by_index.items():
        print(f"\nLoading index: {index_dir}")
        try:
            engine = IndexedQueryEngine(config, index_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] Index not found: {e}")
            predictions.extend([None] * len(index_queries))
            continue

        for q in index_queries:
            print(f"\n  Query: '{q['query']}' ({q['query_type']})")
            out_dir = os.path.join(output_root, mode, q['video_id'])
            result = run_query_safe(engine, q, out_dir, mode_flags)
            predictions.append(result)

    return predictions


def save_results(mode: str, predictions: List, queries: List, output_root: str):
    path = os.path.join(output_root, f'{mode}_predictions.json')
    data = []
    for pred, gt in zip(predictions, queries):
        data.append({'query': gt['query'], 'video_id': gt['video_id'],
                     'prediction': pred, 'ground_truth': gt})
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='test_queries.json')
    parser.add_argument('--config', default='../text_query/config.yaml')
    parser.add_argument('--output', default='results')
    parser.add_argument('--mode', default='all',
                        help='all | full | no_comp | no_ocr | no_verify | use_strongest | clip_only')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    queries_path = args.queries
    if not os.path.isabs(queries_path):
        queries_path = os.path.join(os.path.dirname(__file__), queries_path)
    with open(queries_path) as f:
        queries = json.load(f)

    # Skip queries without ground truth for metric computation
    eval_queries = [q for q in queries if q.get('gt_bbox') is not None or
                    q.get('gt_timestamp') is not None]
    print(f"Loaded {len(queries)} queries ({len(eval_queries)} with ground truth)")

    os.makedirs(args.output, exist_ok=True)

    modes_to_run = list(MODES.keys()) if args.mode == 'all' else [args.mode]

    all_metrics = {}
    for mode in modes_to_run:
        predictions = run_mode(mode, queries, config, args.output)
        save_results(mode, predictions, queries, args.output)

        # Split queries into OCR / General for separate tables
        def has_gt(q): return q.get('gt_bbox') or q.get('gt_timestamp')

        all_gt   = [(p, q) for p, q in zip(predictions, queries) if has_gt(q)]
        ocr_gt   = [(p, q) for p, q in all_gt if q.get('query_type') == 'ocr']
        gen_gt   = [(p, q) for p, q in all_gt if q.get('query_type') == 'general']

        m_all    = compute_metrics([p for p,q in all_gt],  [q for p,q in all_gt])
        m_ocr    = compute_metrics([p for p,q in ocr_gt],  [q for p,q in ocr_gt])
        m_gen    = compute_metrics([p for p,q in gen_gt],  [q for p,q in gen_gt])

        all_metrics[mode] = {
            'overall': m_all,
            'ocr':     m_ocr,
            'general': m_gen,
        }

        lbl = MODE_LABELS[mode]
        print_metrics_table(m_all, label=f"{lbl} — ALL ({len(all_gt)} queries)")
        print_metrics_table(m_ocr, label=f"{lbl} — OCR/Brand ({len(ocr_gt)} queries)")
        print_metrics_table(m_gen, label=f"{lbl} — General ({len(gen_gt)} queries)")

    # Save aggregated metrics
    metrics_path = os.path.join(args.output, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({MODE_LABELS[k]: v for k, v in all_metrics.items()}, f, indent=2)
    print(f"\nAll metrics saved: {metrics_path}")

    # ── Final summary tables ─────────────────────────────────────────────
    def _row(mode, split):
        m = all_metrics[mode][split]
        return (f"{m.get('mIoU',0)*100:.1f}",
                f"{m.get('Success@25',0):.1f}",
                f"{m.get('Success@50',0):.1f}",
                f"{m.get('temporal_error_mean',0):.1f}")

    for split, label in [('ocr','OCR / Brand Queries'),
                         ('general','General Object Queries'),
                         ('overall','Overall')]:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'─'*80}")
        print(f"  {'Method':<32} {'mIoU':>7} {'S@25':>7} {'S@50':>7} {'T.Err':>8}")
        print(f"{'─'*80}")
        for mode in modes_to_run:
            miou, s25, s50, terr = _row(mode, split)
            star = ' ★' if mode == 'full' else ''
            print(f"  {MODE_LABELS[mode]:<32} {miou:>6}% {s25:>6}% {s50:>6}% {terr:>7}s{star}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
