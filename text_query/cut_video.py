"""Extract a short clip from a video for testing."""

import cv2
import argparse
import os

def cut_video(input_video, output_video, duration_seconds=300, sample_rate=1):
    """
    Extract a clip from a video.
    
    Args:
        input_video: Path to input video
        output_video: Path to output clip
        duration_seconds: Clip duration in seconds (default 300 = 5 min)
        sample_rate: Sample every Nth frame (1 = all frames, 3 = every 3rd frame)
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frames to extract
    max_frames = int(fps * duration_seconds)
    frames_to_read = min(max_frames * sample_rate, total_frames)
    
    print(f"Input: {input_video}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames / fps:.1f} seconds")
    print(f"  Resolution: {width}x{height}")
    print()
    print(f"Output: {output_video}")
    print(f"  Extracting {max_frames} frames ({duration_seconds}s)")
    print(f"  Sample rate: every {sample_rate} frame(s)")
    print(f"  Output FPS: {fps / sample_rate:.1f}")
    
    # Create video writer
    out_fps = fps / sample_rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, out_fps, (width, height))
    
    os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
    
    frame_count = 0
    written = 0
    
    while frame_count < frames_to_read and written < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            out.write(frame)
            written += 1
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames, wrote {written}...")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Done! Wrote {written} frames to {output_video}")
    print(f"  File size: {os.path.getsize(output_video) / 1e6:.1f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut a short test clip from a video")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--output", "-o", default="test_clip.mp4", help="Output video path")
    parser.add_argument("--duration", "-d", type=int, default=300, help="Clip duration in seconds (default 300 = 5 min)")
    parser.add_argument("--sample-rate", "-s", type=int, default=1, help="Sample every Nth frame (default 1 = all)")
    
    args = parser.parse_args()
    
    cut_video(args.input, args.output, args.duration, args.sample_rate)
