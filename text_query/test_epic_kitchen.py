"""
Validation suite for text-query pipeline on Epic Kitchen videos.

Tests the full FAISS index → query workflow with standard queries
to identify bottlenecks and measure latency before Ego4D scaling.

Usage:
    python test_epic_kitchen.py \
        --index ../epic_kitchen_indexes/P01_01 \
        --video ../epic_kitchen_data/P01_01.mp4 \
        --output validation_results/ \
        --batch
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import time

import cv2
import numpy as np

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from query_indexed import IndexedQueryEngine


class ValidationSuite:
    """
    Validate text-query pipeline on Epic Kitchen with standard test queries.
    """

    # Standard test queries for Epic Kitchen (common objects/interactions)
    STANDARD_QUERIES = [
        # Kitchen objects
        ("knife", 0.20, "sharp cutting implement"),
        ("cup", 0.20, "drinking vessel"),
        ("plate", 0.20, "food serving dish"),
        ("pan", 0.20, "cooking implement"),
        ("water", 0.25, "liquid in sink/glass"),

        # Interactions
        ("hand holding something", 0.18, "grasp interaction"),
        ("cutting food", 0.18, "chopping action"),
        ("pouring", 0.20, "liquid transfer"),
        ("stirring", 0.22, "mixing with utensil"),

        # Scenes
        ("kitchen counter", 0.25, "work surface"),
        ("food on table", 0.22, "dining scene"),
        ("inside kitchen", 0.28, "location"),
    ]

    def __init__(self, index_dir: str, video_path: str, output_dir: str):
        self.index_dir = index_dir
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load engine once (faster than reloading for each query)
        self.engine = IndexedQueryEngine({}, index_dir)

    def run_query(
        self,
        query: str,
        threshold: float = None,
        top_k: int = 100
    ) -> Tuple[Dict, float]:
        """
        Run single query and return result + latency.

        Returns:
            (result_dict, query_time_seconds)
        """
        start_time = time.time()

        try:
            result = self.engine.query(
                query,
                self.video_path,
                os.path.join(self.output_dir, query.replace(" ", "_")),
                threshold=threshold,
                top_k=top_k
            )
            latency = time.time() - start_time
            return result, latency, None
        except Exception as e:
            latency = time.time() - start_time
            return None, latency, str(e)

    def run_standard_suite(self, batch: bool = False) -> Dict:
        """
        Run all standard test queries.

        Args:
            batch: If True, run all queries; else just test first 3

        Returns:
            results_dict with per-query metrics
        """
        queries = self.STANDARD_QUERIES if batch else self.STANDARD_QUERIES[:3]

        results = {
            'timestamp': time.time(),
            'video_path': self.video_path,
            'index_dir': self.index_dir,
            'num_queries': len(queries),
            'queries': []
        }

        total_time = 0
        success_count = 0

        print(f"\n{'='*70}")
        print(f"{"VALIDATION SUITE": ^70}")
        print(f"{'='*70}")
        print(f"Video: {os.path.basename(self.video_path)}")
        print(f"Index: {os.path.basename(self.index_dir)}")
        print(f"Queries: {len(queries)}")
        print(f"{'='*70}\n")

        for i, (query, threshold, description) in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Query: '{query}'")
            print(f"      Threshold: {threshold}, Description: {description}")

            result, latency, error = self.run_query(query, threshold=threshold)

            query_result = {
                'query': query,
                'threshold': threshold,
                'description': description,
                'latency': latency,
                'success': error is None
            }

            if error is None:
                query_result.update({
                    'last_frame_idx': result['last_frame_idx'],
                    'last_frame_timestamp': result['last_frame_timestamp'],
                    'best_region_score': result['best_region_score'],
                    'context_seconds': result['context_seconds'],
                })
                print(f"      ✓ Success | Frame: {result['best_frame_idx']} | Score: {result['best_region_score']:.3f} | Time: {latency:.1f}s")
                success_count += 1
                total_time += latency
            else:
                query_result['error'] = error
                print(f"      ✗ Error: {error[:60]}")

            results['queries'].append(query_result)
            print()

        # Summary
        results['success_rate'] = success_count / len(queries)
        results['avg_latency'] = total_time / success_count if success_count > 0 else None
        results['total_time'] = total_time

        print(f"{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Success rate: {results['success_rate']*100:.0f}% ({success_count}/{len(queries)})")
        if results['avg_latency']:
            print(f"Average latency: {results['avg_latency']:.1f}s per query")
            print(f"Total time: {results['total_time']:.0f}s")
        print(f"{'='*70}\n")

        return results

    def save_results(self, results: Dict):
        """Save validation results to JSON."""
        result_path = os.path.join(self.output_dir, 'validation_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {result_path}")

    def analyze_bottlenecks(self, results: Dict) -> Dict:
        """
        Analyze which steps are slowest.

        Returns:
            bottleneck_analysis dict
        """
        if results['success_rate'] < 0.5:
            return {
                'bottleneck': 'high_failure_rate',
                'recommendation': 'Check that index was built correctly and objects are actually in video'
            }

        avg_latency = results['avg_latency']
        if avg_latency is None:
            return {'bottleneck': 'no_successful_queries'}

        analysis = {'avg_latency': avg_latency}

        if avg_latency > 10:
            analysis['bottleneck'] = 'faiss_search'
            analysis['recommendation'] = 'FAISS search is slow. Try: (1) Smaller top_k, (2) IVF index for approximate search, (3) GPU FAISS'
        elif avg_latency > 5:
            analysis['bottleneck'] = 'region_refinement'
            analysis['recommendation'] = 'REN region refinement is slow. Try: (1) Batch region scoring, (2) Reduce top_k, (3) Lower sample_rate during indexing'
        else:
            analysis['bottleneck'] = 'none'
            analysis['recommendation'] = 'Performance is good! Ready to scale to Ego4D.'

        return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Validate text-query pipeline on Epic Kitchen videos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--index', type=str, required=True, help='Path to FAISS index directory')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='validation_results/',
                       help='Output directory for results')
    parser.add_argument('--batch', action='store_true',
                       help='Run full test suite (default: test first 3 queries)')
    parser.add_argument('--custom-query', type=str, nargs='+',
                       help='Run custom queries instead of standard suite')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.index):
        print(f"Error: Index directory not found: {args.index}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Run validation
    suite = ValidationSuite(args.index, args.video, args.output)

    if args.custom_query:
        # Custom queries
        print(f"\nRunning {len(args.custom_query)} custom queries...")
        results = {
            'timestamp': time.time(),
            'video_path': args.video,
            'index_dir': args.index,
            'num_queries': len(args.custom_query),
            'queries': []
        }

        for query in args.custom_query:
            result, latency, error = suite.run_query(query)
            results['queries'].append({
                'query': query,
                'latency': latency,
                'success': error is None,
                'error': error
            })

    else:
        # Standard suite
        results = suite.run_standard_suite(batch=args.batch)

    # Save and analyze
    suite.save_results(results)
    bottlenecks = suite.analyze_bottlenecks(results)

    print("BOTTLENECK ANALYSIS")
    print(f"{'='*70}")
    print(f"Bottleneck: {bottlenecks.get('bottleneck', 'unknown')}")
    print(f"Recommendation: {bottlenecks.get('recommendation', 'N/A')}")
    print(f"{'='*70}\n")

    # Next steps
    if results['success_rate'] >= 0.7:
        print("✓ Pipeline is ready for Ego4D scaling!")
        print("  Next steps:")
        print("  1. Download Ego4D VQ2D dataset (500 GB recommended)")
        print("  2. Index several long videos (45 min each)")
        print("  3. Measure latency & accuracy on diverse queries")
        print("  4. Optimize hyperparameters if needed")
    else:
        print("✗ Pipeline needs debugging before Ego4D scaling")
        print("  Check:")
        print("  - Index was built correctly (prepare_index.py succeeded)")
        print("  - Objects in video match queries (check video content)")
        print("  - Threshold is reasonable (0.15-0.25 typical)")
        print("  - CLIP/REN models are loaded correctly")


if __name__ == '__main__':
    main()
