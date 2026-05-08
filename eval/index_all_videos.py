"""
Batch index all videos in a folder for Textual-REN evaluation.

Usage:
    python index_all_videos.py \
        --videos "D:/REN Project/epic_kitchen_data/EPIC-KITCHENS" \
        --output "D:/REN Project/epic_kitchen_indexes" \
        --config ../text_query/config.yaml

It will find every .MP4 / .mp4 file recursively and index each one,
skipping any that already have a completed index (faiss.index exists).

Output structure:
    epic_kitchen_indexes/
    ├── P01_01/   faiss.index  metadata.json  clip_embeddings.npy
    ├── P01_04/
    ├── P02_01/
    └── ...
"""

import os
import sys
import glob
import argparse
import yaml
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text_query'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def find_videos(root: str):
    """Find all MP4 files recursively."""
    videos = []
    for ext in ('*.MP4', '*.mp4', '*.avi', '*.AVI'):
        videos.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos',  required=True,
                        help='Root folder containing video files (searched recursively)')
    parser.add_argument('--output',  required=True,
                        help='Root folder where indexes will be saved')
    parser.add_argument('--config',  default='../text_query/config.yaml')
    parser.add_argument('--sample-rate', type=int, default=None, dest='sample_rate',
                        help='Override frame sample rate from config')
    parser.add_argument('--skip-ocr', action='store_true', dest='skip_ocr',
                        help='Skip OCR extraction (faster, no brand recognition)')
    parser.add_argument('--force', action='store_true',
                        help='Re-index even if index already exists')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    videos = find_videos(args.videos)
    if not videos:
        print(f"No videos found under: {args.videos}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s):\n")
    for v in videos:
        print(f"  {v}")
    print()

    from prepare_index import VideoIndexer
    indexer = VideoIndexer(config)

    results = []
    for i, video_path in enumerate(videos):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir    = os.path.join(args.output, video_name)
        index_file = os.path.join(out_dir, 'faiss.index')

        print(f"\n[{i+1}/{len(videos)}]  {video_name}")

        if os.path.exists(index_file) and not args.force:
            print(f"  ✓ Already indexed — skipping  (use --force to reindex)")
            results.append({'video': video_name, 'status': 'skipped'})
            continue

        t0 = time.time()
        try:
            meta = indexer.index_video(
                video_path, out_dir,
                sample_rate=args.sample_rate,
                skip_ocr=args.skip_ocr,
            )
            elapsed = time.time() - t0
            print(f"  ✓ Done in {elapsed/60:.1f} min  "
                  f"({meta['num_frames']} frames indexed)")
            results.append({'video': video_name, 'status': 'ok',
                            'frames': meta['num_frames'], 'time_min': round(elapsed/60, 1)})
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({'video': video_name, 'status': 'failed', 'error': str(e)})

    print(f"\n{'='*60}")
    print(f"  Indexing Summary")
    print(f"{'='*60}")
    for r in results:
        if r['status'] == 'ok':
            print(f"  ✓ {r['video']:20s}  {r['frames']} frames  {r['time_min']} min")
        elif r['status'] == 'skipped':
            print(f"  - {r['video']:20s}  (already indexed)")
        else:
            print(f"  ✗ {r['video']:20s}  FAILED: {r.get('error','')}")
    print(f"{'='*60}")

    print(f"\nAll indexes saved to: {args.output}")
    print("\nNext step — update test_queries_multi.json with your video IDs,")
    print("then run: python benchmark.py --queries test_queries_multi.json --mode full")


if __name__ == '__main__':
    main()
