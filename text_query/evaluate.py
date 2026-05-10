"""
Comprehensive Evaluation Script for Textual-REN

Computes quantitative metrics:
- mIoU (mean Intersection-over-Union)
- Success@IoU (Success Rate at various IoU thresholds)
- Average Precision (AP) at IoU=0.5, 0.75
- Latency Analysis (per-component breakdown)
- Per-Query-Type Performance
- Failure Analysis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
from query_indexed import IndexedQueryEngine
import yaml


class EvaluationMetrics:
    """Compute standard object detection metrics."""

    @staticmethod
    def compute_iou(pred_bbox: List[int], gt_bbox: List[int]) -> float:
        """
        Compute Intersection-over-Union (IoU).

        Bboxes format: [x, y, width, height]
        """
        pred_x, pred_y, pred_w, pred_h = pred_bbox
        gt_x, gt_y, gt_w, gt_h = gt_bbox

        # Convert to [x1, y1, x2, y2]
        pred_x1, pred_y1 = pred_x, pred_y
        pred_x2, pred_y2 = pred_x + pred_w, pred_y + pred_h

        gt_x1, gt_y1 = gt_x, gt_y
        gt_x2, gt_y2 = gt_x + gt_w, gt_y + gt_h

        # Intersection
        inter_x1 = max(pred_x1, gt_x1)
        inter_y1 = max(pred_y1, gt_y1)
        inter_x2 = min(pred_x2, gt_x2)
        inter_y2 = min(pred_y2, gt_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        pred_area = pred_w * pred_h
        gt_area = gt_w * gt_h
        union_area = pred_area + gt_area - inter_area

        if union_area == 0:
            return 0.0

        return float(inter_area / union_area)

    @staticmethod
    def compute_success_rate(ious: List[float], threshold: float) -> float:
        """Compute Success@IoU_threshold."""
        if not ious:
            return 0.0
        return float(np.mean(np.array(ious) >= threshold)) * 100

    @staticmethod
    def compute_average_precision(ious: List[float], iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision (AP).

        AP = sum(precision * recall_delta) for all detections
        Simplified: fraction of queries with IoU >= threshold
        """
        if not ious:
            return 0.0
        return float(np.mean(np.array(ious) >= iou_threshold)) * 100


class TextualRENEvaluator:
    """Complete evaluation pipeline for Textual-REN."""

    def __init__(self, config_path: str, index_dir: str, video_path: str):
        """Initialize evaluator."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.engine = IndexedQueryEngine(self.config, index_dir)
        self.video_path = video_path
        self.metrics = EvaluationMetrics()
        self.results = []

    def evaluate_queries(self, queries_file: str, output_dir: str) -> Dict:
        """
        Evaluate on a set of test queries.

        queries_file format:
        [
            {
                "query": "cup",
                "gt_frame_idx": 64790,
                "gt_bbox": [720, 350, 200, 180],
                "query_type": "object",  # object, brand, attribute
                "difficulty": "easy"  # easy, medium, hard
            },
            ...
        ]
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(queries_file) as f:
            queries = json.load(f)

        print(f"\n📊 Evaluating {len(queries)} queries...")

        ious = []
        latencies = []
        per_type_results = {}
        per_difficulty_results = {}

        for query_data in tqdm(queries, desc="Processing queries"):
            query = query_data['query']
            gt_bbox = query_data['gt_bbox']
            query_type = query_data.get('query_type', 'object')
            difficulty = query_data.get('difficulty', 'medium')

            # Run query
            start_time = time.time()
            try:
                result = self.engine.query(
                    query,
                    self.video_path,
                    os.path.join(output_dir, f"query_{query}"),
                    threshold=self.config['text_query'].get('similarity_threshold', 0.18)
                )
                latency = time.time() - start_time

                # Compute IoU
                pred_bbox = result['pred_bbox']
                iou = self.metrics.compute_iou(pred_bbox, gt_bbox)

                ious.append(iou)
                latencies.append(latency)

                # Store result
                eval_result = {
                    'query': query,
                    'query_type': query_type,
                    'difficulty': difficulty,
                    'gt_bbox': gt_bbox,
                    'pred_bbox': pred_bbox,
                    'iou': float(iou),
                    'latency': float(latency),
                    'clip_similarity': result.get('clip_similarity', 0),
                    'region_point': result.get('region_point', [0, 0]),
                }
                self.results.append(eval_result)

                # Aggregate by type
                if query_type not in per_type_results:
                    per_type_results[query_type] = []
                per_type_results[query_type].append(iou)

                # Aggregate by difficulty
                if difficulty not in per_difficulty_results:
                    per_difficulty_results[difficulty] = []
                per_difficulty_results[difficulty].append(iou)

            except Exception as e:
                print(f"  ❌ Query '{query}' failed: {e}")
                self.results.append({
                    'query': query,
                    'query_type': query_type,
                    'difficulty': difficulty,
                    'error': str(e),
                    'iou': 0.0,
                    'latency': time.time() - start_time
                })

        # Compute metrics
        metrics_dict = {
            'total_queries': len(queries),
            'successful_queries': len(ious),
            'mIoU': float(np.mean(ious)) if ious else 0.0,
            'mIoU_std': float(np.std(ious)) if ious else 0.0,
            'median_iou': float(np.median(ious)) if ious else 0.0,
            'success_at_0.3': self.metrics.compute_success_rate(ious, 0.3),
            'success_at_0.5': self.metrics.compute_success_rate(ious, 0.5),
            'success_at_0.75': self.metrics.compute_success_rate(ious, 0.75),
            'ap_at_0.5': self.metrics.compute_average_precision(ious, 0.5),
            'ap_at_0.75': self.metrics.compute_average_precision(ious, 0.75),
            'mean_latency': float(np.mean(latencies)) if latencies else 0.0,
            'median_latency': float(np.median(latencies)) if latencies else 0.0,
            'std_latency': float(np.std(latencies)) if latencies else 0.0,
            'min_latency': float(np.min(latencies)) if latencies else 0.0,
            'max_latency': float(np.max(latencies)) if latencies else 0.0,
            'per_type': {
                query_type: {
                    'mIoU': float(np.mean(iou_list)),
                    'count': len(iou_list),
                    'success_at_0.5': self.metrics.compute_success_rate(iou_list, 0.5),
                }
                for query_type, iou_list in per_type_results.items()
            },
            'per_difficulty': {
                difficulty: {
                    'mIoU': float(np.mean(iou_list)),
                    'count': len(iou_list),
                    'success_at_0.5': self.metrics.compute_success_rate(iou_list, 0.5),
                }
                for difficulty, iou_list in per_difficulty_results.items()
            }
        }

        return metrics_dict, ious, latencies

    def generate_visualizations(self, ious: List[float], latencies: List[float], output_dir: str):
        """Generate evaluation plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. mIoU Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ious, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
        ax.axvline(np.median(ious), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.3f}')
        ax.set_xlabel('IoU Score', fontsize=12)
        ax.set_ylabel('Number of Queries', fontsize=12)
        ax.set_title('mIoU Distribution Across Test Set', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_miou_distribution.png'), dpi=300)
        plt.close()

        # 2. Success Rate Curve
        thresholds = np.arange(0, 1.01, 0.05)
        success_rates = [self.metrics.compute_success_rate(ious, t) for t in thresholds]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, success_rates, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.fill_between(thresholds, success_rates, alpha=0.3, color='lightblue')
        ax.set_xlabel('IoU Threshold', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate vs IoU Threshold', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

        # Annotate key points
        for t in [0.3, 0.5, 0.75]:
            sr = self.metrics.compute_success_rate(ious, t)
            ax.annotate(f'@{t}: {sr:.1f}%', xy=(t, sr), xytext=(t, sr+5),
                       fontsize=10, ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_success_curve.png'), dpi=300)
        plt.close()

        # 3. Latency Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(latencies, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(latencies):.2f}s')
        ax.axvline(np.median(latencies), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(latencies):.2f}s')
        ax.set_xlabel('Latency (seconds)', fontsize=12)
        ax.set_ylabel('Number of Queries', fontsize=12)
        ax.set_title('Query Latency Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_latency_distribution.png'), dpi=300)
        plt.close()

        # 4. Per-Query-Type Performance
        if self.results:
            query_types = {}
            for r in self.results:
                qtype = r.get('query_type', 'unknown')
                if qtype not in query_types:
                    query_types[qtype] = []
                if 'iou' in r:
                    query_types[qtype].append(r['iou'])

            types = list(query_types.keys())
            miou_by_type = [np.mean(query_types[t]) for t in types]
            success_by_type = [self.metrics.compute_success_rate(query_types[t], 0.5) for t in types]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # mIoU by type
            bars1 = ax1.bar(types, miou_by_type, color=['steelblue', 'coral', 'green'][:len(types)], alpha=0.7, edgecolor='black')
            ax1.set_ylabel('mIoU', fontsize=12)
            ax1.set_title('mIoU by Query Type', fontsize=12, fontweight='bold')
            ax1.set_ylim([0, 1])
            ax1.grid(alpha=0.3, axis='y')
            for i, (bar, val) in enumerate(zip(bars1, miou_by_type)):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                        ha='center', fontsize=10, fontweight='bold')

            # Success@0.5 by type
            bars2 = ax2.bar(types, success_by_type, color=['steelblue', 'coral', 'green'][:len(types)], alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Success Rate (%)', fontsize=12)
            ax2.set_title('Success@0.5 by Query Type', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 105])
            ax2.grid(alpha=0.3, axis='y')
            for i, (bar, val) in enumerate(zip(bars2, success_by_type)):
                ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                        ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '04_per_type_performance.png'), dpi=300)
            plt.close()

        # 5. IoU vs Latency Scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(latencies, ious, alpha=0.6, s=100, c=ious, cmap='RdYlGn', edgecolors='black')
        ax.set_xlabel('Latency (seconds)', fontsize=12)
        ax.set_ylabel('IoU Score', fontsize=12)
        ax.set_title('IoU vs Latency (Speed-Accuracy Trade-off)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('IoU Score', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_iou_vs_latency.png'), dpi=300)
        plt.close()

        # 6. Per-Difficulty Performance
        if self.results:
            difficulties = {}
            for r in self.results:
                diff = r.get('difficulty', 'unknown')
                if diff not in difficulties:
                    difficulties[diff] = []
                if 'iou' in r:
                    difficulties[diff].append(r['iou'])

            diffs = ['easy', 'medium', 'hard']
            diffs = [d for d in diffs if d in difficulties]
            miou_by_diff = [np.mean(difficulties[d]) for d in diffs]
            success_by_diff = [self.metrics.compute_success_rate(difficulties[d], 0.5) for d in diffs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            colors = ['green', 'orange', 'red']

            # mIoU by difficulty
            bars1 = ax1.bar(diffs, miou_by_diff, color=colors[:len(diffs)], alpha=0.7, edgecolor='black')
            ax1.set_ylabel('mIoU', fontsize=12)
            ax1.set_title('mIoU by Difficulty', fontsize=12, fontweight='bold')
            ax1.set_ylim([0, 1])
            ax1.grid(alpha=0.3, axis='y')
            for bar, val in zip(bars1, miou_by_diff):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                        ha='center', fontsize=10, fontweight='bold')

            # Success@0.5 by difficulty
            bars2 = ax2.bar(diffs, success_by_diff, color=colors[:len(diffs)], alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Success Rate (%)', fontsize=12)
            ax2.set_title('Success@0.5 by Difficulty', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 105])
            ax2.grid(alpha=0.3, axis='y')
            for bar, val in zip(bars2, success_by_diff):
                ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                        ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '06_per_difficulty_performance.png'), dpi=300)
            plt.close()

        print(f"✅ Visualizations saved to {output_dir}")


def create_sample_test_set() -> str:
    """Create a sample test set for evaluation."""
    test_queries = [
        {
            "query": "cup",
            "gt_frame_idx": 64790,
            "gt_bbox": [720, 350, 200, 180],
            "query_type": "object",
            "difficulty": "easy"
        },
        {
            "query": "knife",
            "gt_frame_idx": 45230,
            "gt_bbox": [450, 200, 150, 400],
            "query_type": "object",
            "difficulty": "medium"
        },
        {
            "query": "plate",
            "gt_frame_idx": 52100,
            "gt_bbox": [600, 400, 250, 150],
            "query_type": "object",
            "difficulty": "easy"
        },
        {
            "query": "spoon",
            "gt_frame_idx": 38500,
            "gt_bbox": [800, 450, 80, 120],
            "query_type": "object",
            "difficulty": "hard"
        },
        {
            "query": "Fairy",
            "gt_frame_idx": 30000,
            "gt_bbox": [800, 300, 120, 200],
            "query_type": "brand",
            "difficulty": "medium"
        },
    ]

    test_file = "eval_data/sample_test_set.json"
    os.makedirs("eval_data", exist_ok=True)

    with open(test_file, 'w') as f:
        json.dump(test_queries, f, indent=2)

    return test_file


def main():
    parser = argparse.ArgumentParser(description='Evaluate Textual-REN')
    parser.add_argument('--config', default='text_query/config.yaml', help='Config file path')
    parser.add_argument('--index', default='epic_kitchen_indexes/P01_01', help='Index directory')
    parser.add_argument('--video', default='epic_kitchen_data/EPIC-KITCHENS/P01/videos/P01_01.MP4', help='Video path')
    parser.add_argument('--queries', default='eval_data/sample_test_set.json', help='Test queries JSON')
    parser.add_argument('--output', default='evaluation_results', help='Output directory')

    args = parser.parse_args()

    # Create sample test set if it doesn't exist
    if not os.path.exists(args.queries):
        print(f"Creating sample test set...")
        args.queries = create_sample_test_set()

    # Run evaluation
    evaluator = TextualRENEvaluator(args.config, args.index, args.video)
    metrics, ious, latencies = evaluator.evaluate_queries(args.queries, args.output)

    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total Queries:        {metrics['total_queries']}")
    print(f"Successful Queries:   {metrics['successful_queries']}")
    print(f"\nmIoU (mean):          {metrics['mIoU']:.4f} ± {metrics['mIoU_std']:.4f}")
    print(f"Median IoU:           {metrics['median_iou']:.4f}")
    print(f"\nSuccess@0.3:          {metrics['success_at_0.3']:.2f}%")
    print(f"Success@0.5:          {metrics['success_at_0.5']:.2f}%")
    print(f"Success@0.75:         {metrics['success_at_0.75']:.2f}%")
    print(f"\nAverage Precision:")
    print(f"  AP@0.5:             {metrics['ap_at_0.5']:.2f}%")
    print(f"  AP@0.75:            {metrics['ap_at_0.75']:.2f}%")
    print(f"\nLatency:")
    print(f"  Mean:               {metrics['mean_latency']:.2f}s")
    print(f"  Median:             {metrics['median_latency']:.2f}s")
    print(f"  Std Dev:            {metrics['std_latency']:.2f}s")
    print(f"  Range:              {metrics['min_latency']:.2f}s - {metrics['max_latency']:.2f}s")

    if metrics['per_type']:
        print(f"\nPer Query Type:")
        for qtype, stats in metrics['per_type'].items():
            print(f"  {qtype:10s}: mIoU={stats['mIoU']:.4f}  Success@0.5={stats['success_at_0.5']:.2f}%  (n={stats['count']})")

    if metrics['per_difficulty']:
        print(f"\nPer Difficulty:")
        for diff, stats in metrics['per_difficulty'].items():
            print(f"  {diff:10s}: mIoU={stats['mIoU']:.4f}  Success@0.5={stats['success_at_0.5']:.2f}%  (n={stats['count']})")

    print("="*60)

    # Generate visualizations
    evaluator.generate_visualizations(ious, latencies, os.path.join(args.output, 'visualizations'))

    # Save detailed results
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output, 'per_query_results.json'), 'w') as f:
        json.dump(evaluator.results, f, indent=2)

    print(f"✅ Results saved to {args.output}")
    print(f"   - metrics.json: Overall metrics")
    print(f"   - per_query_results.json: Per-query details")
    print(f"   - visualizations/: Plots and graphs")


if __name__ == '__main__':
    main()
