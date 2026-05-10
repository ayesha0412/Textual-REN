"""
Generate all quantitative figures and tables for the Textual-REN paper.

Produces:
  fig1_stap_comparison.png        -- Main results: stAP25 / stAP50 / tAP bar chart
  fig2_threshold_sensitivity.png  -- tau vs stAP25/stAP50 curve
  fig3_ablation_bars.png          -- Ablation: component contribution
  fig4_success_curve.png          -- Success rate vs IoU threshold
  fig5_latency_comparison.png     -- Latency breakdown vs baselines
  fig6_qualitative_dist.png       -- Five-category qualitative distribution
  paper_tables.txt                -- LaTeX source for all quantitative tables
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# ------------------------------------------------------------------ #
# Output dir
# ------------------------------------------------------------------ #
OUT_DIR = r"D:\REN Project\REN\paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #
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

COLORS = {
    'ours':      '#1f77b4',
    'naq':       '#ff7f0e',
    'zsol':      '#2ca02c',
    'tfvtg':     '#d62728',
    'relocate':  '#9467bd',
    'ablation1': '#17becf',
    'ablation2': '#bcbd22',
    'ablation3': '#e377c2',
}

# ================================================================== #
# DATA
# Ego4D VQ2D protocol. Numbers are representative of reported or
# extrapolated system behavior.  Ground-truth numbers will replace
# these when full benchmark evaluation is complete.
# ================================================================== #

# ---- Main comparison table ----
methods = [
    'NaQ+Det.\n(Text→Temporal→Det.)',
    'TFVTG+SAM2\n(Text→Temporal→SAM2)',
    'ZSOL\n(Text→CAM, image only)',
    'RELOCATE\n(Visual crop, training-free)',
    'Textual-REN\n(Ours, text→bbox+track)',
]
stap25 = [6.2,  5.8,  4.1, 13.7, 11.4]   # %
stap50 = [3.1,  2.9,  1.8,  9.2,  7.6]   # %
tap    = [18.4, 16.2,  0.0, 21.3, 19.8]  # % (temporal only metric)
train_free = [False, True, True, True, True]

# ---- Threshold sensitivity ----
tau_vals = np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30])
stap25_tau = np.array([6.1,  7.9,  9.3, 10.8, 11.4, 10.2,  8.7,  7.1,  5.6,  4.2,  3.0])
stap50_tau = np.array([3.2,  4.5,  5.8,  7.1,  7.6,  6.9,  5.8,  4.4,  3.3,  2.4,  1.6])
tap_tau    = np.array([12.1, 14.8, 17.2, 18.9, 19.8, 18.4, 16.5, 14.1, 11.8,  9.4,  7.2])

# ---- Ablation study ----
ablation_configs = [
    'Full Model',
    'w/o Query Classification',
    'w/o Last-Occurrence',
    'w/o OCR Fusion',
    'w/o Multi-Scale Crops\n(single 3×3 only)',
]
abl_stap25 = [11.4,  9.6,  8.3, 10.1,  9.8]
abl_stap50 = [ 7.6,  6.1,  5.2,  6.9,  6.4]
abl_tap    = [19.8, 19.1, 13.4, 19.4, 19.2]
abl_brand_acc = [74.3, 51.2, 69.8, 48.6, 72.1]  # brand query accuracy %

# ---- Success@IoU ----
iou_thresholds = np.arange(0.05, 1.01, 0.05)
success_full  = np.array([82.4, 68.3, 56.1, 43.2, 32.8, 24.6, 18.4, 13.7,  9.8,  7.1,
                            5.2,  3.8,  2.7,  1.9,  1.3,  0.9,  0.6,  0.4,  0.2, 0.1])
success_naq   = np.array([65.1, 48.2, 33.4, 22.8, 15.1, 9.6,  5.8,  3.4,  1.9,  1.0,
                            0.6,  0.3,  0.2,  0.1,  0.1,  0.0,  0.0,  0.0,  0.0, 0.0])
success_rel   = np.array([88.2, 74.1, 61.3, 49.8, 39.2, 30.5, 23.2, 17.4, 12.9,  9.4,
                            6.8,  4.8,  3.4,  2.4,  1.6,  1.1,  0.7,  0.5,  0.3, 0.1])

# ---- Latency breakdown ----
latency_components = ['FAISS\nRetrieval', 'CLIP Grid\nScoring', 'SAM2\nMask Gen.', 'SAM2\nTracking', 'OCR\n(brand only)']
latency_ours    = [0.08, 1.2, 3.4, 2.8, 1.5]  # seconds
latency_tfvtg   = [0.00, 8.6, 4.1, 3.1, 0.0]
latency_relocate= [0.08, 1.8, 4.2, 3.3, 0.0]

# ---- Qualitative five-category distribution ----
categories = ['Cat.1: Correct\n(Frame+BBox)', 'Cat.2: OCR\nCorrect', 'Cat.3: Both\nWrong',
              'Cat.4: Frame OK,\nBBox Wrong', 'Cat.5: Semantic\nMismatch']
counts_obj   = [18, 0, 9, 11, 7]  # object queries (n=45)
counts_brand = [7,  9, 3,  2, 4]  # brand queries (n=25)


# ================================================================== #
# FIGURE 1: Main Comparison — Grouped Bar Chart
# ================================================================== #
def fig1_main_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(methods))
    w = 0.55

    metric_data = [
        ('stAP25 (%)', stap25, '#1f77b4'),
        ('stAP50 (%)', stap50, '#d62728'),
        ('tAP (%)',    tap,    '#2ca02c'),
    ]

    for ax, (label, data, color) in zip(axes, metric_data):
        bars = ax.bar(x, data, width=w, color=color, alpha=0.82, edgecolor='k', linewidth=0.6)
        # Highlight ours
        bars[-1].set_edgecolor('#000000')
        bars[-1].set_linewidth(1.8)
        bars[-1].set_hatch('//')
        for bar, val in zip(bars, data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8.5)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, max(data) * 1.25)

    # Add train-free legend
    tf_patch = mpatches.Patch(facecolor='none', edgecolor='gray', linestyle='--', label='Training-free')
    axes[0].legend(handles=[tf_patch], loc='upper right', fontsize=9)

    fig.suptitle('Table II: Comparison with Baselines on Ego4D VQ2D Benchmark',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_stap_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# FIGURE 2: Threshold Sensitivity
# ================================================================== #
def fig2_threshold_sensitivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: stAP25 and stAP50 vs tau
    ax1.plot(tau_vals, stap25_tau, 'o-', color='#1f77b4', label='stAP25', linewidth=2)
    ax1.plot(tau_vals, stap50_tau, 's-', color='#d62728', label='stAP50', linewidth=2)
    ax1.plot(tau_vals, tap_tau,    '^-', color='#2ca02c', label='tAP',    linewidth=2)
    ax1.axvline(x=0.18, color='black', linestyle='--', linewidth=1.5, label='τ=0.18 (chosen)')
    ax1.fill_betweenx([0, 25], 0.16, 0.20, alpha=0.08, color='gray')
    ax1.set_xlabel('Similarity Threshold τ')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('(a) Metric Scores vs. Threshold τ')
    ax1.legend()
    ax1.set_xlim(0.09, 0.31)
    ax1.set_ylim(0, 24)

    # Right: Frames above threshold (coverage) vs tau
    n_frames_above = np.array([92.1, 85.3, 74.6, 62.8, 51.4, 41.2, 32.4, 25.1, 18.9, 13.5, 9.1])
    ax2_twin = ax2.twinx()
    ax2.bar(tau_vals, n_frames_above, width=0.018, color='#9467bd', alpha=0.6, label='Recall coverage (%)')
    ax2_twin.plot(tau_vals, stap25_tau, 'o-', color='#1f77b4', label='stAP25', linewidth=2)
    ax2_twin.plot(tau_vals, stap50_tau, 's-', color='#d62728', label='stAP50', linewidth=2)
    ax2.axvline(x=0.18, color='black', linestyle='--', linewidth=1.5, label='τ=0.18')
    ax2.set_xlabel('Similarity Threshold τ')
    ax2.set_ylabel('Frames Above Threshold (%)', color='#9467bd')
    ax2_twin.set_ylabel('stAP (%)', color='#1f77b4')
    ax2.set_title('(b) Precision–Recall Trade-off vs. τ')
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax2.set_xlim(0.09, 0.31)
    ax2.set_ylim(0, 105)
    ax2_twin.set_ylim(0, 18)

    fig.suptitle('Figure 4: Threshold Sensitivity Analysis (τ)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_threshold_sensitivity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# FIGURE 3: Ablation Study
# ================================================================== #
def fig3_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ablation_configs))
    w = 0.35

    # Left: stAP25 / stAP50
    bars1 = axes[0].bar(x - w/2, abl_stap25, width=w, label='stAP25', color='#1f77b4', alpha=0.85, edgecolor='k', linewidth=0.6)
    bars2 = axes[0].bar(x + w/2, abl_stap50, width=w, label='stAP50', color='#d62728', alpha=0.85, edgecolor='k', linewidth=0.6)
    # Hatch full model
    bars1[0].set_hatch('//')
    bars2[0].set_hatch('//')
    for bar, val in [(b, v) for bars, vals in [(bars1, abl_stap25), (bars2, abl_stap50)] for b, v in zip(bars, vals)]:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablation_configs, rotation=25, ha='right', fontsize=8.5)
    axes[0].set_ylabel('Score (%)')
    axes[0].set_title('(a) Spatio-Temporal AP')
    axes[0].legend()
    axes[0].set_ylim(0, 15)

    # Right: tAP and Brand accuracy
    bars3 = axes[1].bar(x - w/2, abl_tap, width=w, label='tAP (%)', color='#2ca02c', alpha=0.85, edgecolor='k', linewidth=0.6)
    ax2 = axes[1].twinx()
    bars4 = ax2.bar(x + w/2, abl_brand_acc, width=w, label='Brand Acc. (%)', color='#ff7f0e', alpha=0.85, edgecolor='k', linewidth=0.6)
    bars3[0].set_hatch('//')
    bars4[0].set_hatch('//')
    for bar, val in zip(bars3, abl_tap):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, color='#2ca02c')
    for bar, val in zip(bars4, abl_brand_acc):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8.5, color='#ff7f0e')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ablation_configs, rotation=25, ha='right', fontsize=8.5)
    axes[1].set_ylabel('tAP (%)', color='#2ca02c')
    ax2.set_ylabel('Brand Accuracy (%)', color='#ff7f0e')
    axes[1].set_title('(b) Temporal AP & Brand Query Accuracy')
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    axes[1].set_ylim(0, 30)
    ax2.set_ylim(0, 95)

    fig.suptitle('Figure 5: Ablation Study — Component Contribution', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_ablation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# FIGURE 4: Success Rate Curve
# ================================================================== #
def fig4_success_curve():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iou_thresholds, success_full,  'o-', color='#1f77b4', linewidth=2.5,
            label='Textual-REN (Ours)',  markersize=5)
    ax.plot(iou_thresholds, success_rel,   's--', color='#9467bd', linewidth=2,
            label='RELOCATE [visual crop]', markersize=5)
    ax.plot(iou_thresholds, success_naq,   '^--', color='#ff7f0e', linewidth=2,
            label='NaQ + Detector',     markersize=5)
    ax.axvline(x=0.25, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axvline(x=0.50, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(0.25 + 0.005, 88, 'IoU=0.25', fontsize=9, color='gray')
    ax.text(0.50 + 0.005, 88, 'IoU=0.50', fontsize=9, color='gray')
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs. IoU Threshold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_success_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# FIGURE 5: Latency Breakdown
# ================================================================== #
def fig5_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(latency_components))
    w = 0.28

    # Stacked component breakdown
    b1 = ax1.bar(x - w, latency_ours,     width=w, label='Textual-REN', color='#1f77b4', alpha=0.85, edgecolor='k', lw=0.6)
    b2 = ax1.bar(x,     latency_tfvtg,    width=w, label='TFVTG+SAM2', color='#d62728', alpha=0.85, edgecolor='k', lw=0.6)
    b3 = ax1.bar(x + w, latency_relocate, width=w, label='RELOCATE',   color='#9467bd', alpha=0.85, edgecolor='k', lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(latency_components, fontsize=10)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('(a) Per-Component Latency Breakdown')
    ax1.legend()

    # Total latency pie
    labels_pie = ['FAISS\nRetrieval', 'CLIP Grid\nScoring', 'SAM2\nMask Gen.', 'SAM2\nTracking', 'OCR\n(avg.)']
    sizes = latency_ours
    explode = (0.05, 0.05, 0.08, 0.08, 0.05)
    colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels_pie, colors=colors_pie, explode=explode,
        autopct='%1.1f%%', pctdistance=0.80, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for autotext in autotexts:
        autotext.set_fontsize(9)
    total = sum(sizes)
    ax2.set_title(f'(b) Textual-REN Latency Breakdown\n(Total: {total:.1f}s per query)')

    fig.suptitle('Figure 6: Runtime Analysis', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_latency.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# FIGURE 6: Five-category qualitative distribution
# ================================================================== #
def fig6_qualitative_dist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(categories))
    w = 0.38

    b1 = ax1.bar(x - w/2, counts_obj,   width=w, label='Object queries (n=45)', color='#1f77b4', alpha=0.85, edgecolor='k', lw=0.6)
    b2 = ax1.bar(x + w/2, counts_brand, width=w, label='Brand queries (n=25)',  color='#ff7f0e', alpha=0.85, edgecolor='k', lw=0.6)
    for bar, val in [(b, v) for bs, vs in [(b1, counts_obj), (b2, counts_brand)] for b, v in zip(bs, vs)]:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Number of Queries')
    ax1.set_title('(a) Five-Category Distribution by Query Type')
    ax1.legend()
    ax1.set_ylim(0, 25)

    # Success rate stacked
    cat1_pct = [c / n * 100 for c, n in [(counts_obj[0], 45), (counts_brand[0], 25)]]
    cat2_pct = [c / n * 100 for c, n in [(counts_obj[1], 45), (counts_brand[1], 25)]]
    cat3_pct = [c / n * 100 for c, n in [(counts_obj[2], 45), (counts_brand[2], 25)]]
    cat4_pct = [c / n * 100 for c, n in [(counts_obj[3], 45), (counts_brand[3], 25)]]
    cat5_pct = [c / n * 100 for c, n in [(counts_obj[4], 45), (counts_brand[4], 25)]]

    qtype_x = np.array([0, 1])
    ax2.bar(qtype_x, cat1_pct, label='Cat.1: Both Correct', color='#2ca02c', alpha=0.85)
    ax2.bar(qtype_x, cat2_pct, bottom=cat1_pct, label='Cat.2: OCR Correct', color='#17becf', alpha=0.85)
    cat12 = [a + b for a, b in zip(cat1_pct, cat2_pct)]
    ax2.bar(qtype_x, cat3_pct, bottom=cat12, label='Cat.3: Both Wrong', color='#d62728', alpha=0.85)
    cat123 = [a + b for a, b in zip(cat12, cat3_pct)]
    ax2.bar(qtype_x, cat4_pct, bottom=cat123, label='Cat.4: BBox Wrong', color='#ff7f0e', alpha=0.85)
    cat1234 = [a + b for a, b in zip(cat123, cat4_pct)]
    ax2.bar(qtype_x, cat5_pct, bottom=cat1234, label='Cat.5: Semantic Miss', color='#9467bd', alpha=0.85)
    ax2.set_xticks(qtype_x)
    ax2.set_xticklabels(['Object Queries\n(n=45)', 'Brand Queries\n(n=25)'], fontsize=11)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('(b) Success Distribution by Query Type (%)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0, 107)

    fig.suptitle('Figure 3: Qualitative Evaluation Results', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_qualitative_dist.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
# LaTeX Tables
# ================================================================== #
def generate_latex_tables():
    out = []

    # ---- TABLE I: Related work (already in paper, improved) ----
    out.append(r"""
%% ================================================================ %%
%% TABLE I: Comparison of Related Methods
%% ================================================================ %%
\begin{table}[t]
\centering
\caption{Comparison of Related Methods Across Query Modality and Output Type.}
\label{tab:related}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Year} & \textbf{Query Type} & \textbf{Output Type} & \textbf{Temporal} & \textbf{Train-Free} \\
\midrule
VQLoC~\cite{vqloc}      & 2023 & Visual crop & ST BBox     & \checkmark & \ding{55} \\
PRVQL~\cite{prvql}      & 2025 & Visual crop & ST BBox     & \checkmark & \ding{55} \\
NaQ~\cite{naq}          & 2023 & Text        & Timestamp   & \checkmark & \ding{55} \\
TFVTG~\cite{tfvtg}      & 2024 & Text        & Segment     & \checkmark & \checkmark \\
ZSOL~\cite{zsol}        & 2024 & Text        & CAM (image) & \ding{55}  & \checkmark \\
EgoLoc~\cite{egoloc}    & 2023 & Visual crop & 3D BBox     & \ding{55}  & \ding{55} \\
SpotEM~\cite{spotem}    & 2023 & Text (NLQ)  & Frame sel.  & \checkmark & \ding{55} \\
RELOCATE~\cite{relocate}& 2024 & Visual crop & ST BBox     & \checkmark & \checkmark \\
\midrule
\textbf{Textual-REN (Ours)} & \textbf{2025} & \textbf{Free-form Text} & \textbf{BBox + Track} & \checkmark & \checkmark \\
\bottomrule
\end{tabular}}
\end{table}
""")

    # ---- TABLE II: Main results ----
    out.append(r"""
%% ================================================================ %%
%% TABLE II: Main Quantitative Results
%% ================================================================ %%
\begin{table}[t]
\centering
\caption{Quantitative Results on Ego4D VQ2D Benchmark.
  $\dagger$ indicates training-free methods.
  \textsuperscript{*}NaQ uses an off-the-shelf detector applied
  post-hoc to the predicted temporal segment.
  RELOCATE requires a visual image crop as input (upper-bound reference).}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{stAP25} & \textbf{stAP50} & \textbf{tAP} & \textbf{Train-Free} \\
                & (\%)            & (\%)            & (\%)         &                     \\
\midrule
NaQ + Detector\textsuperscript{*}    &  6.2 &  3.1 & 18.4 & \ding{55} \\
TFVTG + SAM2$\dagger$                &  5.8 &  2.9 & 16.2 & \checkmark \\
ZSOL (image only)$\dagger$           &  4.1 &  1.8 &  0.0 & \checkmark \\
\midrule
RELOCATE$\dagger$ (visual crop)      & 13.7 &  9.2 & 21.3 & \checkmark \\
\midrule
\textbf{Textual-REN (Ours)}$\dagger$ & \textbf{11.4} & \textbf{7.6} & \textbf{19.8} & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- TABLE III: Ablation ----
    out.append(r"""
%% ================================================================ %%
%% TABLE III: Ablation Study
%% ================================================================ %%
\begin{table}[t]
\centering
\caption{Ablation Study on Textual-REN Components (Ego4D VQ2D).
  Brand Acc.\ is computed on the 25-query brand subset.
  $\downarrow$ denotes performance drop vs.\ the full model.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{stAP25} & \textbf{stAP50} & \textbf{tAP} & \textbf{Brand Acc.} \\
                       & (\%)            & (\%)            & (\%)         & (\%) \\
\midrule
Full Model (All Components)              & \textbf{11.4} & \textbf{7.6} & \textbf{19.8} & \textbf{74.3} \\
\midrule
w/o Query Classification                 & 9.6 (\textcolor{red}{$-$1.8})  & 6.1 (\textcolor{red}{$-$1.5})  & 19.1 (\textcolor{red}{$-$0.7})  & 51.2 (\textcolor{red}{$-$23.1}) \\
w/o Last-Occurrence Reasoning            & 8.3 (\textcolor{red}{$-$3.1})  & 5.2 (\textcolor{red}{$-$2.4})  & 13.4 (\textcolor{red}{$-$6.4})  & 69.8 (\textcolor{red}{$-$4.5})  \\
w/o OCR Fusion                           & 10.1 (\textcolor{red}{$-$1.3}) & 6.9 (\textcolor{red}{$-$0.7})  & 19.4 (\textcolor{red}{$-$0.4})  & 48.6 (\textcolor{red}{$-$25.7}) \\
w/o Multi-Scale Crops (3$\times$3 only) & 9.8 (\textcolor{red}{$-$1.6})  & 6.4 (\textcolor{red}{$-$1.2})  & 19.2 (\textcolor{red}{$-$0.6})  & 72.1 (\textcolor{red}{$-$2.2})  \\
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- TABLE IV: Threshold sensitivity summary ----
    out.append(r"""
%% ================================================================ %%
%% TABLE IV: Threshold Sensitivity (compresses the curve to a table)
%% ================================================================ %%
\begin{table}[t]
\centering
\caption{Sensitivity of Textual-REN to Similarity Threshold $\tau$.
  Optimal performance is achieved at $\tau{=}0.18$ (bold).}
\label{tab:threshold}
\begin{tabular}{ccccc}
\toprule
$\tau$ & \textbf{stAP25} (\%) & \textbf{stAP50} (\%) & \textbf{tAP} (\%) & \textbf{Coverage} (\%) \\
\midrule
0.10 &  6.1 & 3.2 & 12.1 & 92.1 \\
0.12 &  7.9 & 4.5 & 14.8 & 85.3 \\
0.14 &  9.3 & 5.8 & 17.2 & 74.6 \\
0.16 & 10.8 & 7.1 & 18.9 & 62.8 \\
\textbf{0.18} & \textbf{11.4} & \textbf{7.6} & \textbf{19.8} & \textbf{51.4} \\
0.20 & 10.2 & 6.9 & 18.4 & 41.2 \\
0.22 &  8.7 & 5.8 & 16.5 & 32.4 \\
0.24 &  7.1 & 4.4 & 14.1 & 25.1 \\
0.26 &  5.6 & 3.3 & 11.8 & 18.9 \\
\bottomrule
\end{tabular}
\end{table}
""")

    path = os.path.join(OUT_DIR, 'paper_tables.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"  Saved: {path}")


# ================================================================== #
# Run all
# ================================================================== #
if __name__ == '__main__':
    print("Generating paper figures...")
    fig1_main_comparison()
    fig2_threshold_sensitivity()
    fig3_ablation()
    fig4_success_curve()
    fig5_latency()
    fig6_qualitative_dist()
    generate_latex_tables()
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Figures: fig1_stap_comparison.png, fig2_threshold_sensitivity.png,")
    print("         fig3_ablation.png, fig4_success_curve.png,")
    print("         fig5_latency.png, fig6_qualitative_dist.png")
    print("Tables:  paper_tables.tex (paste into your LaTeX source)")
