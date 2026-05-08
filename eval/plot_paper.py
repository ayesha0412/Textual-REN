"""
Publication-ready figure generation for Textual-REN paper.

Generates (300 DPI, PDF + PNG):
  Fig 1  — Success Rate curves (IoU threshold sweep)
  Fig 2  — Method comparison bar chart
  Fig 3  — Ablation bar chart
  Fig 4  — Temporal error CDF
  Fig 5  — Score distribution (violin/box) by query type
  Fig 6  — Qualitative results grid
  Table 1 — Main comparison table (LaTeX)
  Table 2 — Ablation table (LaTeX)

Usage:
    python plot_paper.py --metrics results/all_metrics.json \
                         --predictions results/ \
                         --queries test_queries.json \
                         --output figures/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'full':          '#1f77b4',   # blue  — our method
    'no_comp':       '#ff7f0e',   # orange
    'no_ocr':        '#2ca02c',   # green
    'no_verify':     '#d62728',   # red
    'use_strongest': '#9467bd',   # purple
    'clip_only':     '#8c564b',   # brown — baseline
}
LABELS = {
    'full':          'Textual-REN (Ours)',
    'no_comp':       'w/o Compositional',
    'no_ocr':        'w/o OCR Fusion',
    'no_verify':     'w/o Frame Verification',
    'use_strongest': 'w/o Last-Occurrence',
    'clip_only':     'CLIP Baseline',
}
HATCHES = {
    'full': '', 'no_comp': '//', 'no_ocr': '\\\\',
    'no_verify': 'xx', 'use_strongest': '..', 'clip_only': '--',
}


def savefig(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(output_dir, f'{name}.{ext}')
        fig.savefig(path, format=ext)
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


# ── Figure 1: Success Rate Curves ────────────────────────────────────────────
def plot_success_curves(predictions_by_mode: Dict, queries: List, output_dir: str):
    """
    % queries with IoU > threshold, swept from 0.05 to 0.95.
    One curve per method — standard in visual grounding papers.
    """
    thresholds = np.arange(0.05, 1.0, 0.05)
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for mode, preds in predictions_by_mode.items():
        ious = []
        for pred, gt in zip(preds, queries):
            if pred and gt.get('gt_bbox') and pred.get('pred_bbox'):
                from metrics import compute_iou
                ious.append(compute_iou(pred['pred_bbox'], gt['gt_bbox']))

        if not ious:
            continue
        rates = [np.mean([1 if iou >= t else 0 for iou in ious]) * 100
                 for t in thresholds]
        lw = 2.5 if mode == 'full' else 1.5
        ls = '-' if mode == 'full' else '--'
        ax.plot(thresholds, rates, label=LABELS[mode],
                color=COLORS[mode], linewidth=lw, linestyle=ls)

    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs. IoU Threshold')
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axvline(0.5, color='gray', linewidth=0.8, linestyle=':')
    ax.text(0.51, 5, 'IoU=0.5', color='gray', fontsize=9)
    savefig(fig, output_dir, 'fig1_success_curves')


# ── Figure 2: Comparison Bar Chart ───────────────────────────────────────────
def plot_comparison_bars(all_metrics: Dict, output_dir: str,
                         modes=None, metric_keys=None):
    """
    Grouped bar chart: methods × metrics.
    Highlights our method with a different edge color.
    """
    if modes is None:
        modes = ['clip_only', 'use_strongest', 'no_verify',
                 'no_ocr', 'no_comp', 'full']
    if metric_keys is None:
        metric_keys = ['Success@25', 'Success@50', 'Success@75', 'mIoU']
    metric_labels = ['S@25 (%)', 'S@50 (%)', 'S@75 (%)', 'mIoU (%)']

    x = np.arange(len(metric_keys))
    n = len(modes)
    width = 0.13
    offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mode in enumerate(modes):
        m = all_metrics.get(LABELS[mode]) or all_metrics.get(mode) or {}
        vals = []
        for k in metric_keys:
            v = m.get(k, 0)
            vals.append(v * 100 if k == 'mIoU' else v)
        edge = 'black' if mode == 'full' else 'gray'
        lw = 1.5 if mode == 'full' else 0.7
        bars = ax.bar(x + offsets[i], vals, width,
                      label=LABELS[mode], color=COLORS[mode],
                      edgecolor=edge, linewidth=lw,
                      hatch=HATCHES[mode], alpha=0.88)
        if mode == 'full':
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=8,
                        fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score (%)')
    ax.set_title('Method Comparison — Localization Performance')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    savefig(fig, output_dir, 'fig2_comparison_bars')


# ── Figure 3: Ablation Bar Chart ─────────────────────────────────────────────
def plot_ablation_bars(all_metrics: Dict, output_dir: str):
    """
    Horizontal bar chart showing drop in S@50 and mIoU per ablation.
    Full model is reference; each bar shows relative change.
    """
    ablation_modes = ['no_comp', 'no_ocr', 'no_verify', 'use_strongest']
    full_label = LABELS['full']
    full_m = all_metrics.get(full_label) or {}
    ref_s50  = full_m.get('Success@50', 0)
    ref_miou = full_m.get('mIoU', 0) * 100

    labels, drops_s50, drops_miou = [], [], []
    for mode in ablation_modes:
        m = all_metrics.get(LABELS[mode]) or {}
        labels.append(LABELS[mode].replace('w/o ', ''))
        drops_s50.append(ref_s50 - m.get('Success@50', 0))
        drops_miou.append(ref_miou - m.get('mIoU', 0) * 100)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    h1 = ax.barh(y - 0.18, drops_s50,  0.35, label='ΔSuccess@50 (%)',
                 color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.7)
    h2 = ax.barh(y + 0.18, drops_miou, 0.35, label='ΔmIoU (%)',
                 color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.7)

    for bar in list(h1) + list(h2):
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height()/2,
                f'{w:+.1f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Performance Drop from Full Model (%)')
    ax.set_title('Ablation Study — Component Contribution')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    savefig(fig, output_dir, 'fig3_ablation_bars')


# ── Figure 4: Temporal Error CDF ─────────────────────────────────────────────
def plot_temporal_cdf(predictions_by_mode: Dict, queries: List, output_dir: str):
    """
    Cumulative distribution of temporal error (seconds).
    Shows what % of queries are localized within N seconds.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    max_err = 30

    for mode, preds in predictions_by_mode.items():
        errors = []
        for pred, gt in zip(preds, queries):
            if pred and gt.get('gt_timestamp') and pred.get('last_frame_timestamp'):
                e = abs(pred['last_frame_timestamp'] - gt['gt_timestamp'])
                errors.append(min(e, max_err))
        if not errors:
            continue
        errors_sorted = np.sort(errors)
        cdf = np.arange(1, len(errors_sorted)+1) / len(errors_sorted) * 100
        lw = 2.5 if mode == 'full' else 1.5
        ls = '-' if mode == 'full' else '--'
        ax.plot(errors_sorted, cdf, label=LABELS[mode],
                color=COLORS[mode], linewidth=lw, linestyle=ls)

    ax.axvline(2, color='gray', linewidth=0.8, linestyle=':')
    ax.axvline(5, color='gray', linewidth=0.8, linestyle=':')
    ax.text(2.1, 5, '2s', color='gray', fontsize=9)
    ax.text(5.1, 5, '5s', color='gray', fontsize=9)
    ax.set_xlabel('Temporal Error (seconds)')
    ax.set_ylabel('Cumulative % of Queries')
    ax.set_title('Temporal Localization Error CDF')
    ax.set_xlim(0, max_err)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', framealpha=0.9)
    savefig(fig, output_dir, 'fig4_temporal_cdf')


# ── Figure 5: Per Query-Type Performance ─────────────────────────────────────
def plot_query_type_breakdown(predictions: List, queries: List, output_dir: str):
    """
    Bar chart of S@50 broken down by query type (object / brand / compositional).
    Shows where Textual-REN excels.
    """
    from metrics import compute_iou
    types = ['object', 'brand', 'compositional']
    type_ious: Dict[str, List] = {t: [] for t in types}

    for pred, gt in zip(predictions, queries):
        qtype = gt.get('query_type', 'object')
        if pred and gt.get('gt_bbox') and pred.get('pred_bbox'):
            iou = compute_iou(pred['pred_bbox'], gt['gt_bbox'])
            type_ious.setdefault(qtype, []).append(iou)

    labels, s50, miou = [], [], []
    for t in types:
        ious = type_ious.get(t, [])
        if ious:
            labels.append(t.capitalize())
            s50.append(np.mean([1 if iou >= 0.5 else 0 for iou in ious]) * 100)
            miou.append(np.mean(ious) * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - 0.18, s50,  0.35, label='Success@50', color='#1f77b4',
           edgecolor='black', linewidth=0.7)
    ax.bar(x + 0.18, miou, 0.35, label='mIoU',       color='#ff7f0e',
           edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance by Query Type')
    ax.set_ylim(0, 105)
    ax.legend()
    savefig(fig, output_dir, 'fig5_query_type_breakdown')


# ── Figure 6: Qualitative Grid ───────────────────────────────────────────────
def plot_qualitative_grid(pred_dirs: Dict[str, str], queries: List,
                          output_dir: str, n_examples: int = 4):
    """
    Paper-style qualitative grid: rows = examples, cols = methods.
    Reads debug_last_frame.jpg from each query result directory.

    pred_dirs: {mode: base_output_dir}
    """
    import cv2

    selected = queries[:n_examples]
    modes_show = ['clip_only', 'full']
    col_labels = [LABELS[m] for m in modes_show]

    n_rows = len(selected)
    n_cols = len(modes_show) + 1  # +1 for query text column

    fig = plt.figure(figsize=(3.5 * n_cols, 2.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.05)

    for row, q in enumerate(selected):
        for col, mode in enumerate(modes_show):
            ax = fig.add_subplot(gs[row, col])
            img_path = os.path.join(
                pred_dirs.get(mode, ''), mode, q['video_id'],
                q['query'].replace(' ', '_'), 'debug_last_frame.jpg'
            )
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='red')
            ax.axis('off')
            if row == 0:
                ax.set_title(col_labels[col], fontsize=10, fontweight='bold')

        # Query label column
        ax_txt = fig.add_subplot(gs[row, -1])
        ax_txt.text(0.5, 0.5,
                    f'Query:\n"{q["query"]}"\n\nType: {q["query_type"]}',
                    ha='center', va='center', fontsize=10,
                    transform=ax_txt.transAxes, wrap=True,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
                              edgecolor='#aaaaaa', linewidth=0.8))
        ax_txt.axis('off')
        if row == 0:
            ax_txt.set_title('Query', fontsize=10, fontweight='bold')

    fig.suptitle('Qualitative Comparison: CLIP Baseline vs. Textual-REN',
                 fontsize=13, fontweight='bold', y=1.01)
    savefig(fig, output_dir, 'fig6_qualitative_grid')


# ── LaTeX Table Generators ───────────────────────────────────────────────────
def latex_comparison_table(all_metrics: Dict, output_dir: str):
    """Table 1: Main comparison table for paper."""
    modes = ['clip_only', 'use_strongest', 'no_verify', 'no_ocr', 'no_comp', 'full']
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Comparison of Textual-REN with baseline and ablation variants on EPIC-Kitchens P02.}',
        r'\label{tab:comparison}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Method & S@25 & S@50 & S@75 & mIoU & T.Err (s) & T@2s \\',
        r'\midrule',
    ]
    for mode in modes:
        m = all_metrics.get(LABELS[mode]) or all_metrics.get(mode) or {}
        s25  = m.get('Success@25', 0)
        s50  = m.get('Success@50', 0)
        s75  = m.get('Success@75', 0)
        miou = m.get('mIoU', 0) * 100
        terr = m.get('temporal_error_mean', 0)
        t2s  = m.get('temporal_acc@2s', 0)
        label = LABELS[mode]

        bold = mode == 'full'
        def fmt(v, is_lower=False):
            s = f'{v:.1f}'
            return r'\textbf{' + s + r'}' if bold else s

        row = f'{label} & {fmt(s25)} & {fmt(s50)} & {fmt(s75)} & ' \
              f'{fmt(miou)} & {fmt(terr, True)} & {fmt(t2s)} \\\\'
        if mode == 'no_comp':
            lines.append(r'\midrule')
        lines.append(row)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\end{table}',
    ]
    path = os.path.join(output_dir, 'table1_comparison.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: table1_comparison.tex")


def latex_ablation_table(all_metrics: Dict, output_dir: str):
    """Table 2: Compact ablation table."""
    full_m = all_metrics.get(LABELS['full']) or {}
    components = [
        ('full',          '✓', '✓', '✓', '✓'),
        ('no_comp',       '✗', '✓', '✓', '✓'),
        ('no_ocr',        '✓', '✗', '✓', '✓'),
        ('no_verify',     '✓', '✓', '✗', '✓'),
        ('use_strongest', '✓', '✓', '✓', '✗'),
    ]
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Ablation study. Each row removes one component from the full model.}',
        r'\label{tab:ablation}',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Comp. & OCR & Verify & Last-Occ & S@50 & mIoU \\',
        r'\midrule',
    ]
    for mode, *flags in components:
        m = all_metrics.get(LABELS[mode]) or {}
        s50  = m.get('Success@50', 0)
        miou = m.get('mIoU', 0) * 100
        bold = mode == 'full'
        def fmt(v): return r'\textbf{' + f'{v:.1f}' + r'}' if bold else f'{v:.1f}'
        row = ' & '.join(flags) + f' & {fmt(s50)} & {fmt(miou)} \\\\'
        lines.append(row)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = os.path.join(output_dir, 'table2_ablation.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: table2_ablation.tex")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics',     default='results/all_metrics.json')
    parser.add_argument('--predictions', default='results/')
    parser.add_argument('--queries',     default='test_queries.json')
    parser.add_argument('--output',      default='figures/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load metrics
    metrics_path = args.metrics
    if not os.path.isabs(metrics_path):
        metrics_path = os.path.join(os.path.dirname(__file__), metrics_path)
    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Load queries
    queries_path = args.queries
    if not os.path.isabs(queries_path):
        queries_path = os.path.join(os.path.dirname(__file__), queries_path)
    with open(queries_path) as f:
        queries = json.load(f)

    # Load per-mode predictions
    predictions_by_mode = {}
    for mode in MODES_LIST := ['full', 'no_comp', 'no_ocr', 'no_verify',
                                'use_strongest', 'clip_only']:
        pred_path = os.path.join(args.predictions, f'{mode}_predictions.json')
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                data = json.load(f)
            predictions_by_mode[mode] = [d['prediction'] for d in data]

    print("Generating figures...")

    plot_success_curves(predictions_by_mode, queries, args.output)
    plot_comparison_bars(all_metrics, args.output)
    plot_ablation_bars(all_metrics, args.output)
    plot_temporal_cdf(predictions_by_mode, queries, args.output)

    if 'full' in predictions_by_mode:
        plot_query_type_breakdown(
            predictions_by_mode['full'], queries, args.output
        )
        plot_qualitative_grid(
            {m: args.predictions for m in predictions_by_mode},
            queries, args.output
        )

    latex_comparison_table(all_metrics, args.output)
    latex_ablation_table(all_metrics, args.output)

    print(f"\nAll figures saved to: {args.output}")
    print("Files: fig1_success_curves, fig2_comparison_bars, fig3_ablation_bars,")
    print("       fig4_temporal_cdf, fig5_query_type_breakdown, fig6_qualitative_grid")
    print("       table1_comparison.tex, table2_ablation.tex")


if __name__ == '__main__':
    main()
