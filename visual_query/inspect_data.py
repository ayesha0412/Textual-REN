"""
Inspect the preprocessed dataset — show frames and annotations.
Usage: python inspect_data.py --data_dir D:/ego4d_records/val --num_clips 3
"""
import os
import pickle
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:/ego4d_records/val')
parser.add_argument('--num_clips', type=int, default=3, help='How many clips to inspect')
args = parser.parse_args()

pkl_files = sorted(Path(args.data_dir).glob('*.pkl'))[:args.num_clips]

for pkl_file in pkl_files:
    print(f"\n{'='*70}")
    print(f"Clip: {pkl_file.name}")
    print(f"{'='*70}")

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    print(f"  Video UID:  {data['video_uid'].decode()}")
    print(f"  Clip UID:   {data['clip_uid'].decode()}")
    print(f"  Frames:     {data['num_frames']}")
    print(f"  Resolution: {data['frame_width']}×{data['frame_height']}")
    print(f"  Annotations: {len(data['annotations'])}")

    frames = data['frames']
    annotations = data['annotations']

    # Show first annotation
    if len(annotations) > 0:
        ann = annotations[0]
        query_frame_num = ann['visual_crop']['frame_number']
        query_bbox = ann['visual_crop']

        print(f"\n  First annotation:")
        print(f"    Object: {ann.get('object_title', 'unknown')}")
        print(f"    Query frame: {query_frame_num} / {data['num_frames']}")
        print(f"    Query bbox: x={query_bbox['x']}, y={query_bbox['y']}, "
              f"w={query_bbox['width']}, h={query_bbox['height']}")
        print(f"    Response track frames: {len(ann['response_track'])} frames")

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Query frame with query crop
        query_frame = frames[query_frame_num]
        axes[0].imshow(query_frame)
        x, y, w, h = query_bbox['x'], query_bbox['y'], query_bbox['width'], query_bbox['height']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Query crop')
        axes[0].add_patch(rect)
        axes[0].set_title(f'Query frame #{query_frame_num}\n(object: {ann.get("object_title", "?")})')
        axes[0].legend()
        axes[0].axis('off')

        # Right: First response frame with ground truth bbox
        response_tracks = ann['response_track']
        if len(response_tracks) > 0:
            first_response = response_tracks[0]
            resp_frame_num = first_response['frame_number']
            if 0 <= resp_frame_num < len(frames):
                resp_frame = frames[resp_frame_num]
                axes[1].imshow(resp_frame)
                rx, ry, rw, rh = first_response['x'], first_response['y'], first_response['width'], first_response['height']
                rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='g', facecolor='none', label='Ground truth')
                axes[1].add_patch(rect)
                axes[1].set_title(f'First response frame #{resp_frame_num}')
                axes[1].legend()
                axes[1].axis('off')

        plt.tight_layout()
        save_path = f'inspect_{pkl_file.stem}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n  → Saved visualization: {save_path}")
        plt.close()

print(f"\n{'='*70}\nDone. Check the .png files in current directory.\n")
