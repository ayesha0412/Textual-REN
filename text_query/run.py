"""
Text-query episodic memory localization — CLI entry point.

Usage (run from D:\\REN Project\\REN\\text_query\\):

    conda activate ren_venv
    cd "D:/REN Project/REN/text_query"

    python run.py "my coffee mug" ../ego4d_data/v2/full_scale/3534864b-2289-4aaf-b3ed-10eeeee7acd2.mp4

Options:
    --output      Output directory  (default: output/<slug>)
    --config      Config file path  (default: config.yaml)
    --threshold   CLIP similarity threshold 0-1  (default: from config)
    --context     Clip context ± seconds          (default: from config)
    --sample-rate Process every Nth frame for CLIP (default: from config)
"""

import os
import sys
import re
import yaml
import argparse

# Add text_query directory to path so localizer.py is importable when run
# from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from localizer import TextQueryLocalizer


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:40]


def main():
    parser = argparse.ArgumentParser(
        description='Find the last occurrence of a text-described object in a video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('query',  type=str, help='Natural language query  e.g. "my coffee mug"')
    parser.add_argument('video',  type=str, help='Path to video file')
    parser.add_argument('--config',      type=str,   default='config.yaml')
    parser.add_argument('--output',      type=str,   default=None,
                        help='Output directory (auto-named from query if omitted)')
    parser.add_argument('--threshold',   type=float, default=None,
                        help='CLIP similarity threshold (0-1)')
    parser.add_argument('--context',     type=float, default=None,
                        help='Context window in seconds around last occurrence')
    parser.add_argument('--sample-rate', type=int,   default=None, dest='sample_rate',
                        help='Process every Nth frame for CLIP retrieval')
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Apply CLI overrides
    if args.threshold is not None:
        config['text_query']['similarity_threshold'] = args.threshold
    if args.context is not None:
        config['text_query']['context_seconds'] = args.context
    if args.sample_rate is not None:
        config['text_query']['frame_sample_rate'] = args.sample_rate

    # Resolve output directory
    output_dir = args.output
    if output_dir is None:
        save_root  = config['text_query'].get('save_dir', 'logs/')
        output_dir = os.path.join(save_root, _slug(args.query))

    # Validate video path
    if not os.path.exists(args.video):
        print(f'Error: video not found: {args.video}')
        sys.exit(1)

    # Run localization
    localizer = TextQueryLocalizer(config)
    result = localizer.localize(args.query, args.video, output_dir=output_dir)

    print('\n=== Result ===')
    for k, v in result.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
