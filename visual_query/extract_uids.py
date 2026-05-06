"""
Selects the most clip-dense video UIDs from the VQ2D val annotations
and prints the ego4d full_scale download command.

Usage:
    python extract_uids.py --data_dir D:/ego4d_data --target_clips 30
"""
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='D:/ego4d_data')
parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
parser.add_argument('--target_clips', type=int, default=30,
                    help='Approx number of clips to cover (picks fewest videos to reach this)')
args = parser.parse_args()

split_file = {
    'train': f'{args.data_dir}/v2/annotations/vq_train.json',
    'val':   f'{args.data_dir}/v2/annotations/vq_val.json',
    'test':  f'{args.data_dir}/v2/annotations/vq_test_unannotated.json',
}[args.split]

with open(split_file) as f:
    data = json.load(f)

# Count valid clips per video
clips_per_video = defaultdict(list)
for video in data['videos']:
    for clip in video['clips']:
        has_valid = any(
            qset['is_valid']
            for ann in clip.get('annotations', [])
            for qset in ann['query_sets'].values()
        )
        if has_valid:
            clips_per_video[video['video_uid']].append(clip['clip_uid'])

# Sort videos by clip count descending (most clips per video = best efficiency)
sorted_videos = sorted(clips_per_video.items(), key=lambda x: len(x[1]), reverse=True)

# Pick fewest videos that cover target_clips
selected_videos, selected_clips = [], []
for vid, clips in sorted_videos:
    if len(selected_clips) >= args.target_clips:
        break
    selected_videos.append(vid)
    selected_clips.extend(clips)

print(f"\n{'='*60}")
print(f"Split: {args.split}  |  Videos: {len(selected_videos)}  |  Clips: {len(selected_clips)}")
print(f"{'='*60}")
print(f"\nClips per selected video:")
for vid in selected_videos:
    print(f"  {vid}  →  {len(clips_per_video[vid])} clips")

print(f"\nApprox download size: {len(selected_videos)} videos × ~1.5 GB avg = ~{len(selected_videos)*15//10} GB")
print(f"(actual size varies by video length)")

uid_str = ' '.join(selected_videos)
print(f"\n── Run this to download (~{len(selected_videos)*15//10} GB) ──")
print(f"""
ego4d `
  --output_directory "D:/ego4d_data" `
  --datasets full_scale `
  --version v2 `
  --video_uids {uid_str}
""")
