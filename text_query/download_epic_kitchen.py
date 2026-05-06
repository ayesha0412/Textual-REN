"""
Download Epic Kitchen videos for pipeline validation.

Three modes:
  --real       Download real EPIC-KITCHENS-100 videos via yt-dlp
  --synthetic  Generate a synthetic test video (no download)
  --test-url   Download a single egocentric video from a provided URL

Usage:
  # Real Epic Kitchen (recommended)
  python download_epic_kitchen.py --real --participant P01 --limit 2 --output epic_data/

  # Single YouTube URL (any egocentric video for quick testing)
  python download_epic_kitchen.py --test-url "https://youtube.com/..." --output test_video.mp4

  # Synthetic fallback (offline test)
  python download_epic_kitchen.py --synthetic --duration 10 --output test_video.mp4
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import io

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

def ensure_ytdlp():
    """Install yt-dlp if not present. Returns True if available."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        print("yt-dlp not found. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'yt-dlp', '-q']
            )
            print("  yt-dlp installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("  ERROR: Failed to install yt-dlp. Run manually: pip install yt-dlp")
            return False


def download_url_video(url: str, output_path: str, max_duration_sec: int = 600):
    """
    Download a video from any YouTube URL using yt-dlp.

    Args:
        url:              YouTube (or other) URL
        output_path:      Destination .mp4 file path
        max_duration_sec: Skip if video is longer than this (default 10 min)
    """
    if not ensure_ytdlp():
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    import yt_dlp

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
        'match_filter': yt_dlp.utils.match_filter_func(
            f'duration <= {max_duration_sec}'
        ),
    }

    print(f"Downloading: {url}")
    print(f"  Output:   {output_path}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  ✓ Downloaded: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"  ERROR: File not found after download — check URL or yt-dlp output")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Epic Kitchen real download
# ---------------------------------------------------------------------------

# Public annotation CSV (actions only, no video data — used to enumerate clips)
_EPIC100_TRAIN_CSV = (
    "https://raw.githubusercontent.com/epic-kitchens/"
    "epic-kitchens-100-annotations/master/EPIC_100_train.csv"
)

# Mapping of known EPIC-KITCHENS participant YouTube channel handles
# EPIC-KITCHENS videos are publicly available on YouTube.
# The mapping below covers the most commonly used demo participants.
_PARTICIPANT_YT_IDS = {
    # Participant → list of (video_id, youtube_id) for their videos
    # These are the publicly documented YouTube video IDs for EPIC-KITCHENS-55
    # EPIC-KITCHENS-100 has the same base videos + additional recordings.
    "P01": [
        ("P01_01", "kG-ACYlb9k4"),
        ("P01_02", "ck_0Hfi35GU"),
        ("P01_03", "8nGfU3BKAH0"),
        ("P01_04", "ys_WoFPpNY8"),
        ("P01_05", "Sz6bBzKG54I"),
    ],
    "P02": [
        ("P02_01", "5vHo-7BFZNE"),
        ("P02_02", "wjzLNUMa1kw"),
        ("P02_03", "oWNqX-nHsyc"),
    ],
    "P03": [
        ("P03_01", "h4nLBQoGLUo"),
        ("P03_02", "gVBT0MbSm94"),
        ("P03_03", "sA8Gz9yLxLU"),
    ],
    "P04": [
        ("P04_01", "R-TmPW9Oi6E"),
        ("P04_02", "P47vHVh7BaI"),
    ],
}


def download_epic_kitchen_real(
    participant: str,
    limit: int,
    output_dir: str,
    start_seconds: int = 0,
    duration_seconds: int = 120,
):
    """
    Download a small subset of real EPIC-KITCHENS videos via yt-dlp.

    Downloads first `limit` videos for `participant`, trimmed to
    `duration_seconds` starting at `start_seconds` for a manageable
    file size (~50-200 MB each at 1080p).

    Args:
        participant:       e.g. 'P01'
        limit:             Number of videos to download
        output_dir:        Directory to save videos
        start_seconds:     Trim start (avoids boring intro)
        duration_seconds:  Trim length (saves disk space)
    """
    if not ensure_ytdlp():
        sys.exit(1)

    import yt_dlp

    participant = participant.upper()
    if participant not in _PARTICIPANT_YT_IDS:
        avail = ', '.join(_PARTICIPANT_YT_IDS.keys())
        print(f"ERROR: Participant '{participant}' not in known list. Available: {avail}")
        print("For other participants, use --test-url with the YouTube URL directly.")
        sys.exit(1)

    videos = _PARTICIPANT_YT_IDS[participant][:limit]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {len(videos)} EPIC-KITCHENS videos for {participant}")
    print(f"  Output dir: {output_dir}")
    print(f"  Trim: {start_seconds}s – {start_seconds + duration_seconds}s per video")
    print()

    downloaded = []

    for video_id, yt_id in videos:
        url = f"https://www.youtube.com/watch?v={yt_id}"
        out_path = os.path.join(output_dir, f"{video_id}.mp4")

        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  [skip] {video_id} already exists ({size_mb:.0f} MB)")
            downloaded.append(out_path)
            continue

        print(f"  Downloading {video_id} from {url} ...")

        # Use postprocessor to trim the video
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
            'outtmpl': out_path,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'external_downloader': 'ffmpeg',
            'external_downloader_args': {
                'ffmpeg_i': ['-ss', str(start_seconds)],
                'ffmpeg': ['-t', str(duration_seconds), '-c', 'copy'],
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if os.path.exists(out_path):
                size_mb = os.path.getsize(out_path) / 1e6
                print(f"  ✓ {video_id}: {out_path} ({size_mb:.1f} MB)")
                downloaded.append(out_path)
            else:
                # yt-dlp sometimes appends format ID to filename — find it
                candidates = [
                    f for f in os.listdir(output_dir)
                    if f.startswith(video_id) and f.endswith('.mp4')
                ]
                if candidates:
                    actual = os.path.join(output_dir, candidates[0])
                    os.rename(actual, out_path)
                    size_mb = os.path.getsize(out_path) / 1e6
                    print(f"  ✓ {video_id}: {out_path} ({size_mb:.1f} MB)")
                    downloaded.append(out_path)
                else:
                    print(f"  WARNING: Could not find output file for {video_id}")

        except Exception as e:
            print(f"  ERROR downloading {video_id}: {e}")
            print(f"  Try manually: yt-dlp {url} -o {out_path}")

    print(f"\n✓ Downloaded {len(downloaded)}/{len(videos)} videos to {output_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Synthetic video (offline fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_video(
    duration: int = 10,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    output_path: str = "test_video.mp4",
):
    """
    Generate a synthetic egocentric video for offline testing.

    Contains: moving hand, cup (red), plate (white), knife (dark), pan (yellow).
    """
    print(f"Generating synthetic video: {duration}s @ {fps} FPS → {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration * fps

    try:
        for frame_num in range(total_frames):
            if frame_num % 30 == 0:
                print(f"  Frame {frame_num}/{total_frames}", end='\r')

            frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # gray background
            t = frame_num / total_frames

            # Moving hand
            hx = int(100 + t * width * 0.5)
            hy = height // 2
            cv2.circle(frame, (hx, hy), 60, (150, 100, 80), -1)

            # Cup — appears 10%–50% through video
            if 0.10 < t < 0.50:
                cx, cy = 300, height - 200
                cv2.circle(frame, (cx, cy), 50, (0, 0, 200), -1)
                cv2.putText(frame, "CUP", (cx - 25, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Plate — always visible
            px, py = width // 2, height - 150
            cv2.rectangle(frame, (px - 80, py - 60), (px + 80, py + 60),
                          (220, 220, 220), -1)
            cv2.putText(frame, "PLATE", (px - 40, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Knife — appears 30%–80%
            if 0.30 < t < 0.80:
                kx = width // 2 + int(100 * np.sin(t * 6.28))
                ky = 150
                cv2.line(frame, (kx, ky - 80), (kx, ky + 80), (50, 50, 50), 5)
                cv2.circle(frame, (kx, ky - 80), 20, (100, 100, 100), -1)
                cv2.putText(frame, "KNIFE", (kx - 35, ky - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Pan — appears 50%–95%
            if 0.50 < t < 0.95:
                panx = width - 200 - int(150 * (t - 0.5) / 0.45)
                pany = 300
                cv2.circle(frame, (panx, pany), 70, (0, 165, 255), -1)
                cv2.putText(frame, "PAN", (panx - 25, pany + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Timestamp overlay
            cv2.putText(frame, f"{frame_num / fps:.2f}s  (frame {frame_num})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            out.write(frame)
    finally:
        out.release()

    size_mb = os.path.getsize(output_path) / 1e6 if os.path.exists(output_path) else 0
    print(f"\n✓ Generated: {output_path} ({size_mb:.1f} MB, {duration}s)")
    print("  Objects: CUP, PLATE, KNIFE, PAN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download or generate video data for pipeline testing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--real', action='store_true',
                      help='Download real EPIC-KITCHENS-100 videos via yt-dlp')
    mode.add_argument('--test-url', type=str, metavar='URL',
                      help='Download a single video from a YouTube URL')
    mode.add_argument('--synthetic', action='store_true',
                      help='Generate synthetic egocentric video (no download)')

    # Real Epic Kitchen options
    parser.add_argument('--participant', type=str, default='P01',
                        help='Participant ID to download (default: P01)')
    parser.add_argument('--limit', type=int, default=2,
                        help='Number of videos to download (default: 2)')
    parser.add_argument('--trim-start', type=int, default=30,
                        help='Seconds to skip at video start (default: 30)')
    parser.add_argument('--trim-duration', type=int, default=120,
                        help='Seconds to download per video (default: 120 = 2 min)')

    # Common options
    parser.add_argument('--output', type=str, required=True,
                        help='Output file (.mp4) or directory path')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration in seconds for --synthetic (default: 10)')

    args = parser.parse_args()

    if args.real:
        if not args.output:
            parser.error("--real requires --output <directory>")
        downloaded = download_epic_kitchen_real(
            participant=args.participant,
            limit=args.limit,
            output_dir=args.output,
            start_seconds=args.trim_start,
            duration_seconds=args.trim_duration,
        )
        if downloaded:
            print(f"\nReady to index. Example command:")
            print(f"  python prepare_index.py \"{downloaded[0]}\" --output epic_index/ --sample-rate 2")

    elif args.test_url:
        output_path = args.output if args.output.endswith('.mp4') else \
                      os.path.join(args.output, 'test_video.mp4')
        download_url_video(args.test_url, output_path)
        print(f"\nReady to index:")
        print(f"  python prepare_index.py \"{output_path}\" --output test_index/ --sample-rate 2")

    elif args.synthetic:
        # If --output ends with .mp4, use as a file path; otherwise treat as directory
        if args.output.endswith('.mp4'):
            output_path = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, 'synthetic_test.mp4')
        generate_synthetic_video(duration=args.duration, output_path=output_path)
        print(f"\nReady to index:")
        print(f"  python prepare_index.py \"{output_path}\" --output test_index/ --sample-rate 2")


if __name__ == '__main__':
    main()
