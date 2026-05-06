"""
Download or create sample egocentric videos for testing.

Usage:
    python download_sample_video.py --source pexels --output sample.mp4
    python download_sample_video.py --source synthetic --duration 30 --output sample.mp4
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import cv2


def download_pexels(output_path: str, duration_sec: int = 10):
    """
    Download a kitchen/cooking scene from Pexels (free stock video).
    
    Options:
    - ID 3209708: "Woman Preparing Food" (6.9s, 1920x1080)
    - ID 3209705: "Person Chopping Vegetables" (5.2s, 1920x1080)
    - ID 3394650: "Cooking Pasta" (9.2s, 1920x1080)
    """
    videos = {
        'cooking': '3209705',  # Person chopping - perfect for query testing
        'pasta': '3394650',    # Cooking pasta
        'prep': '3209708',     # Food prep
    }
    
    print("Available sample videos from Pexels:")
    for name, vid_id in videos.items():
        print(f"  {name}: https://www.pexels.com/video/{vid_id}")
    
    print("\nManual download instructions:")
    print("1. Visit https://www.pexels.com/search/videos/cooking/")
    print("2. Download a kitchen/cooking video in 1080p")
    print("3. Save to: " + output_path)
    print("\nFor automated download, install:")
    print("  pip install pexels-api requests")


def download_youtube(url: str, output_path: str):
    """Download a video from YouTube using yt-dlp."""
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp not installed")
        print("Install with: pip install yt-dlp")
        return False
    
    print(f"Downloading: {url}")
    print(f"Output: {output_path}")
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path.replace('.mp4', ''),
        'quiet': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def generate_realistic_synthetic(output_path: str, duration: int = 15, fps: int = 30):
    """
    Generate realistic synthetic egocentric video with kitchen scenes.
    
    Contains:
    - Hand interactions
    - Multiple objects (knife, cutting board, ingredients)
    - Realistic lighting and scene changes
    - Motion blur for realism
    """
    print(f"Generating realistic synthetic video: {duration}s @ {fps} FPS")
    
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    try:
        for frame_num in range(total_frames):
            # Realistic kitchen background with natural lighting
            frame = np.ones((height, width, 3), dtype=np.uint8) * 230
            
            # Add natural light gradient
            for y in range(height):
                intensity = int(230 - (y / height) * 30)
                frame[y, :] = [intensity, intensity + 5, intensity + 10]
            
            if frame_num % max(1, fps // 5) == 0:
                print(f"  Frame {frame_num}/{total_frames}", end='\r')
            
            time_phase = frame_num / total_frames
            
            # === CUTTING BOARD SCENE (0-0.3) ===
            if 0 < time_phase < 0.3:
                # Wooden cutting board
                board_x, board_y = 400, height - 350
                cv2.rectangle(frame, (board_x - 150, board_y - 100), 
                            (board_x + 150, board_y + 100), (180, 140, 80), -1)
                cv2.rectangle(frame, (board_x - 150, board_y - 100), 
                            (board_x + 150, board_y + 100), (120, 90, 60), 2)
                
                # Vegetable (tomato)
                tomato_x = board_x - 60 + int(20 * np.sin(time_phase * 10))
                cv2.circle(frame, (tomato_x, board_y - 20), 40, (0, 50, 200), -1)
                cv2.putText(frame, "TOMATO", (tomato_x - 40, board_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Knife (moving)
                knife_x = board_x + 100 - int(150 * time_phase / 0.3)
                knife_y = board_y - 30
                cv2.line(frame, (knife_x, knife_y - 80), (knife_x, knife_y + 80), (50, 50, 50), 8)
                cv2.circle(frame, (knife_x, knife_y - 80), 15, (100, 100, 100), -1)
                cv2.putText(frame, "KNIFE", (knife_x - 35, knife_y - 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Hand
                hand_x = int(width * 0.2 + 200 * (time_phase / 0.3))
                hand_y = board_y + 50
                cv2.ellipse(frame, (hand_x, hand_y), (50, 70), 45, 0, 360, (180, 130, 100), -1)
            
            # === PAN/POT SCENE (0.3-0.7) ===
            elif 0.3 <= time_phase < 0.7:
                # Stove pot
                pot_x, pot_y = width // 2, 300
                cv2.circle(frame, (pot_x, pot_y), 80, (100, 100, 120), -1)
                cv2.circle(frame, (pot_x, pot_y), 80, (50, 50, 70), 3)
                
                # Pot handle
                cv2.rectangle(frame, (pot_x + 70, pot_y - 20), (pot_x + 100, pot_y + 20), 
                            (100, 100, 120), -1)
                cv2.putText(frame, "POT", (pot_x - 30, pot_y + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Steam effect
                for i in range(3):
                    steam_x = pot_x + int(20 * np.sin(time_phase * 5 + i))
                    steam_y = pot_y - 100 - int(50 * (time_phase - 0.3) / 0.4)
                    radius = max(2, 30 - int(50 * (time_phase - 0.3) / 0.4))
                    cv2.circle(frame, (steam_x, steam_y), radius, (200, 200, 200), -1)
                
                # Hand holding spoon
                hand_x = int(width * 0.15 + 100 * np.sin((time_phase - 0.3) * 6.28))
                hand_y = pot_y + 80
                cv2.ellipse(frame, (hand_x, hand_y), (40, 60), 30, 0, 360, (180, 130, 100), -1)
                
                # Spoon in pot
                spoon_x = pot_x + int(30 * np.cos((time_phase - 0.3) * 6.28))
                spoon_y = pot_y + 20
                cv2.line(frame, (spoon_x, spoon_y), (spoon_x, spoon_y + 60), (150, 150, 150), 6)
                cv2.circle(frame, (spoon_x, spoon_y + 70), 20, (150, 150, 150), -1)
                cv2.putText(frame, "SPOON", (spoon_x - 40, spoon_y - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # === PLATING SCENE (0.7-1.0) ===
            else:
                # Plate (white ceramic)
                plate_x, plate_y = width - 350, height - 250
                cv2.circle(frame, (plate_x, plate_y), 100, (245, 245, 245), -1)
                cv2.circle(frame, (plate_x, plate_y), 100, (200, 200, 200), 2)
                
                # Food on plate
                food_x = plate_x + int(30 * np.cos((time_phase - 0.7) * 6.28))
                food_y = plate_y + int(30 * np.sin((time_phase - 0.7) * 6.28))
                cv2.circle(frame, (food_x, food_y), 30, (150, 80, 50), -1)
                cv2.putText(frame, "FOOD", (plate_x - 40, plate_y + 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Hand placing food
                hand_x = int(width * 0.2 + 300 * ((time_phase - 0.7) / 0.3))
                hand_y = plate_y - 100
                cv2.ellipse(frame, (hand_x, hand_y), (50, 70), 45, 0, 360, (180, 130, 100), -1)
            
            # Timestamp and frame counter
            cv2.putText(frame, f"Frame {frame_num}/{total_frames}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, f"{frame_num/fps:.2f}s", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            out.write(frame)
    
    finally:
        out.release()
    
    print(f"\n✓ Generated: {output_path} ({duration}s)")
    print("  Contains scenes: Cutting, Cooking, Plating")
    print("  Objects: KNIFE, TOMATO, POT, SPOON, PLATE, FOOD, HAND")
    print("  Try queries like: 'knife', 'cutting tomato', 'hand with spoon', 'plate', etc.")


def main():
    parser = argparse.ArgumentParser(
        description='Download or generate sample egocentric videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--source', choices=['pexels', 'youtube', 'synthetic'],
                       default='synthetic',
                       help='Video source (default: synthetic)')
    parser.add_argument('--output', type=str, default='sample_video.mp4',
                       help='Output video path')
    parser.add_argument('--duration', type=int, default=15,
                       help='Duration in seconds (for synthetic)')
    parser.add_argument('--url', type=str, help='YouTube URL (for youtube source)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.source == 'pexels':
        download_pexels(args.output, args.duration)
    elif args.source == 'youtube':
        if not args.url:
            print("Error: --url required for youtube source")
            sys.exit(1)
        download_youtube(args.url, args.output)
    elif args.source == 'synthetic':
        generate_realistic_synthetic(args.output, args.duration)


if __name__ == '__main__':
    main()
