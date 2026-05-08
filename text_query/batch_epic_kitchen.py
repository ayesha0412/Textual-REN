"""
Batch index + validation for Epic Kitchen videos (P01/P02).

Usage:
  python batch_epic_kitchen.py \
    --video-root ../epic_kitchen_data/EPIC-KITCHENS \
    --index-root ../epic_kitchen_indexes \
    --output-root ../epic_kitchen_validation \
    --config config.yaml \
    --rebuild
"""

import argparse
import os
from pathlib import Path
from typing import List

import yaml

from prepare_index import VideoIndexer
from test_epic_kitchen import ValidationSuite


def _find_videos(video_root: Path) -> List[Path]:
    videos = []
    for person_dir in sorted(video_root.iterdir()):
        if not person_dir.is_dir():
            continue
        videos_dir = person_dir / "videos"
        if not videos_dir.exists():
            continue
        for video_path in sorted(videos_dir.glob("*.MP4")):
            videos.append(video_path)
    return videos


def _parse_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Batch index + validate Epic Kitchen videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="../epic_kitchen_data/EPIC-KITCHENS",
        help="Root directory containing P01/P02 folders",
    )
    parser.add_argument(
        "--index-root",
        type=str,
        default="../epic_kitchen_indexes",
        help="Root directory to store FAISS indexes",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="../epic_kitchen_validation",
        help="Root directory to store validation outputs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override frame sample rate (default: config)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild indexes even if they already exist",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Disable OCR when building indexes",
    )
    parser.add_argument(
        "--ocr-queries",
        type=str,
        default=None,
        help="Comma-separated OCR brand queries to run after standard suite",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    video_root = Path(args.video_root).resolve()
    index_root = Path(args.index_root).resolve()
    output_root = Path(args.output_root).resolve()

    videos = _find_videos(video_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {video_root}")

    indexer = VideoIndexer(config)

    for video_path in videos:
        stem = video_path.stem
        index_dir = index_root / stem
        output_dir = output_root / stem
        index_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        index_files = [
            index_dir / "faiss.index",
            index_dir / "metadata.json",
            index_dir / "clip_embeddings.npy",
        ]
        need_rebuild = args.rebuild or not all(p.exists() for p in index_files)

        if need_rebuild:
            print(f"\n=== Indexing {stem} ===")
            indexer.index_video(
                str(video_path),
                str(index_dir),
                sample_rate=args.sample_rate,
                skip_ren=True,
                skip_ocr=args.skip_ocr,
            )
        else:
            print(f"\n=== Index exists for {stem}, skipping rebuild ===")

        print(f"\n=== Validating {stem} ===")
        suite = ValidationSuite(str(index_dir), str(video_path), str(output_dir), config)
        results = suite.run_standard_suite(batch=True)
        suite.save_results(results)

        if args.ocr_queries:
            ocr_queries = _parse_list(args.ocr_queries)
            if ocr_queries:
                print(f"\n=== OCR queries for {stem} ===")
                for query in ocr_queries:
                    suite.run_query(query)


if __name__ == "__main__":
    main()
