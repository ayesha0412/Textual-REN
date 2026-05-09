"""
Phase 1: Offline video indexing.

Samples frames from video, extracts CLIP embeddings + REN region tokens,
builds FAISS index for fast nearest-neighbor search during query time.

Usage:
    python prepare_index.py <video_path> --output <index_dir> --config config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Must be set before any cv2.VideoCapture is opened — EPIC Kitchen / Ego4D videos
# have interleaved audio+video streams that exhaust OpenCV's default packet limit.
os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '65536')

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.transforms as T

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    import faiss
    HAS_GPU = faiss.get_num_gpus() > 0
    if HAS_GPU:
        print(f"GPU-accelerated FAISS enabled (GPUs available: {faiss.get_num_gpus()})")
except ImportError:
    print("Error: faiss not installed.")
    print("")
    print("Install via conda (recommended):")
    print("  conda install -c conda-forge faiss-gpu -y     # GPU support")
    print("  conda install -c conda-forge faiss-cpu -y     # CPU-only")
    print("")
    print("Note: faiss-gpu/faiss-cpu are not available via pip.")
    sys.exit(1)

try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: easyocr not installed — brand/text recognition disabled.")
    print("  pip install easyocr")

from localizer import TextQueryLocalizer


class VideoIndexer:
    """
    Build FAISS index from video frames for efficient text-query retrieval.

    Pipeline:
    1. Sample frames at configurable rate
    2. Extract CLIP image embeddings (1280-dim)
    3. Extract REN region tokens (1024-dim each)
    4. Store metadata (frame_idx, timestamp, region_count)
    5. Build FAISS index on CLIP embeddings
    6. Persist index + metadata for query phase
    """

    def __init__(self, config: Dict):
        self.config = config
        self.localizer = TextQueryLocalizer(config)

        # CLIP embedding dim: read from config (ViT-g-14 joint space = 1024, ViT-bigG-14 = 1280)
        self.clip_dim = config.get('text_query', {}).get('faiss', {}).get('clip_dim', 1024)
        # REN region token dim (DINOv2 ViT-L/14)
        self.ren_dim = 1024
        
        # Device for GPU/CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # OCR reader: initialized lazily on first frame that needs it
        self.ocr_reader = None

        # GPU support
        self.use_gpu = HAS_GPU
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

    # How many frames to encode in one GPU CLIP forward pass.
    # Larger = faster (fewer kernel launches), but uses more VRAM.
    # 64 is safe on 8 GB+; lower to 32 if you hit OOM.
    CLIP_BATCH_SIZE = 64

    # Run EasyOCR only every Nth *sampled* frame.
    # At sample_rate=10 + OCR_STRIDE=5, OCR fires every 50 raw frames
    # (~0.8 s at 60 fps) — enough to catch label text without reading
    # thousands of nearly-identical frames.
    OCR_STRIDE = 5

    def index_video(
        self,
        video_path: str,
        output_dir: str,
        sample_rate: int = None,
        skip_ren: bool = True,
        skip_ocr: bool = False,
    ) -> Dict:
        """
        Index a video: extract CLIP image embeddings and build FAISS index.

        Performance notes
        -----------------
        * CLIP is processed in batches of CLIP_BATCH_SIZE (default 64) so the
          GPU runs at full utilisation instead of stalling between single-frame
          calls.
        * torch.cuda.empty_cache() is NOT called inside the frame loop — doing
          so forces a GPU pipeline flush on every frame, costing minutes on
          long videos.
        * EasyOCR is run only every OCR_STRIDE sampled frames (default every 5)
          because label text changes far more slowly than the frame rate.
        * Frames are read sequentially (no per-frame seek) which is 10–50×
          faster than cap.set(POS_FRAMES) on compressed video files.

        Args:
            video_path:  Path to video file
            output_dir:  Directory to save index files
            sample_rate: Process every Nth frame (default: from config)
            skip_ren:    Skip expensive REN token extraction (default: True)
            skip_ocr:    Skip EasyOCR extraction (default: False)

        Returns:
            metadata dict with index_path, num_frames, fps, etc.
        """
        if sample_rate is None:
            sample_rate = self.config['text_query'].get('frame_sample_rate', 10)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Indexing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        n_sampled = max(1, total_frames // sample_rate)
        print(f"  Source  : {total_frames} frames @ {fps:.2f} fps  "
              f"({frame_width}×{frame_height})")
        print(f"  Sampling: every {sample_rate}th frame → ~{n_sampled} frames")
        print(f"  CLIP batch size : {self.CLIP_BATCH_SIZE}")

        do_ocr = not skip_ocr and HAS_OCR
        if skip_ren:
            print("  REN   : skipped (not needed for text-query mode)")
        if not do_ocr:
            print("  OCR   : skipped")
        else:
            print(f"  OCR   : every {self.OCR_STRIDE} sampled frames "
                  f"(~{self.OCR_STRIDE * sample_rate / fps:.1f} s intervals)")
            if self.ocr_reader is None:
                print("  Initializing EasyOCR …")
                self.ocr_reader = easyocr.Reader(
                    ['en'], gpu=self.device.type == 'cuda', verbose=False
                )
                print("  EasyOCR ready.")

        clip_embeddings   = []   # will hold (N, D) float32
        frame_metadata    = []
        all_region_tokens = []

        # ── accumulation buffers for CLIP batching ──────────────────────────
        batch_frames: List[np.ndarray] = []   # BGR frames pending CLIP
        batch_meta:   List[dict]       = []   # metadata stubs for each frame

        sampled_frame_idx = 0   # counter of frames we actually keep

        # Sequential read — much faster than seeking to every Nth frame on
        # compressed video, and avoids packet-budget exhaustion on EPIC-Kitchens.
        raw_frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_frame_idx % sample_rate == 0:
                # ── OCR (low-frequency) ──────────────────────────────────────
                ocr_texts: list = []
                if do_ocr and sampled_frame_idx % self.OCR_STRIDE == 0:
                    ocr_texts = self._extract_ocr(frame)

                stub = {
                    'frame_idx':         raw_frame_idx,
                    'sampled_frame_idx': sampled_frame_idx,
                    'timestamp':         raw_frame_idx / fps,
                    'ocr_texts':         ocr_texts,
                }
                batch_frames.append(frame)
                batch_meta.append(stub)

                # ── flush CLIP batch ─────────────────────────────────────────
                if len(batch_frames) >= self.CLIP_BATCH_SIZE:
                    feats = self._extract_clip_batch(batch_frames)
                    clip_embeddings.append(feats)
                    frame_metadata.extend(batch_meta)
                    pct = 100 * sampled_frame_idx / max(n_sampled, 1)
                    print(f"  [{pct:5.1f}%] frame {raw_frame_idx}/{total_frames}  "
                          f"({sampled_frame_idx} sampled)", end='\r')
                    batch_frames = []
                    batch_meta   = []

                sampled_frame_idx += 1

            raw_frame_idx += 1

        # flush remaining frames
        if batch_frames:
            feats = self._extract_clip_batch(batch_frames)
            clip_embeddings.append(feats)
            frame_metadata.extend(batch_meta)

        cap.release()

        if not clip_embeddings:
            raise RuntimeError("No frames could be read from the video.")

        clip_embeddings_np = np.vstack(clip_embeddings)   # (N, D)
        print(f"\n  Done — {len(frame_metadata)} frames indexed")
        if not skip_ren:
            print(f"  Total regions: {sum(m.get('region_count', 0) for m in frame_metadata)}")

        # Build FAISS index on CLIP embeddings
        clip_embeddings = clip_embeddings_np
        actual_dim = clip_embeddings.shape[1]
        if actual_dim != self.clip_dim:
            print(f"  [Warning] Actual CLIP embedding dim ({actual_dim}) differs from "
                  f"config clip_dim ({self.clip_dim}). Using actual dim.")
            self.clip_dim = actual_dim
        print(f"\nBuilding FAISS index (dim={self.clip_dim})...")
        if self.use_gpu:
            # Create GPU index for fast search
            print("  Using GPU-accelerated index")
            cpu_index = faiss.IndexFlatIP(self.clip_dim)
            cpu_index.add(clip_embeddings.astype(np.float32))
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        else:
            # Fallback to CPU index
            print("  Using CPU-based index")
            index = faiss.IndexFlatIP(self.clip_dim)
            index.add(clip_embeddings.astype(np.float32))
        print(f"  Index built with {index.ntotal} frames")

        # Save index and metadata
        index_path = os.path.join(output_dir, 'faiss.index')
        metadata_path = os.path.join(output_dir, 'metadata.json')
        regions_path = os.path.join(output_dir, 'regions.pkl')
        embeddings_path = os.path.join(output_dir, 'clip_embeddings.npy')

        # For GPU indices, convert back to CPU for serialization
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(
                {
                    'video_path': video_path,
                    'fps': fps,
                    'total_frames': total_frames,
                    'sampled_frames': len(clip_embeddings),
                    'sample_rate': sample_rate,
                    'frame_metadata': frame_metadata,
                },
                f,
                indent=2
            )
        if not skip_ren:
            with open(regions_path, 'wb') as f:
                pickle.dump(all_region_tokens, f)
        np.save(embeddings_path, clip_embeddings)

        print(f"Index saved to: {output_dir}")
        print(f"  - FAISS index: {index_path}")
        print(f"  - Metadata: {metadata_path}")
        if not skip_ren:
            print(f"  - Region tokens: {regions_path}")
        print(f"  - CLIP embeddings: {embeddings_path}")

        return {
            'index_path': index_path,
            'metadata_path': metadata_path,
            'regions_path': regions_path,
            'embeddings_path': embeddings_path,
            'num_frames': len(clip_embeddings),
            'fps': fps,
        }

    def _extract_ocr(self, frame: np.ndarray) -> list:
        """Extract visible text from a BGR frame. Reader must be initialized before calling."""
        if self.ocr_reader is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = self.ocr_reader.readtext(rgb, detail=1, paragraph=False)
            return [
                {'text': text.lower().strip(), 'conf': round(float(conf), 3)}
                for (_, text, conf) in detections
                if conf >= 0.3 and len(text.strip()) > 1
            ]
        except Exception:
            return []

    def _extract_clip_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract L2-normalised CLIP embeddings for a batch of BGR frames.

        Processing a batch in one GPU call is ~CLIP_BATCH_SIZE× faster than
        calling encode_image() once per frame, and avoids the per-call kernel
        launch overhead that caused 6-hour indexing times.

        Args:
            frames: list of BGR numpy arrays

        Returns:
            (len(frames), D) float32 array of L2-normalised embeddings
        """
        imgs = torch.stack([
            self.localizer.clip_preprocess(
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            )
            for f in frames
        ]).to(self.device)

        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            feats = self.localizer.clip_model.encode_image(imgs).float()

        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def _extract_region_tokens(self, frame: np.ndarray) -> torch.Tensor:
        """
        Extract REN region tokens from a frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            Region tokens (num_regions, 1024)
        """
        # Convert BGR to RGB to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        img_res = self.config['ren']['parameters']['image_resolution']
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_res, img_res), antialias=True),
        ])
        transformed_image = transform(pil_image)

        # Extract features using REN's feature extractor
        image_batch = transformed_image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get feature maps from extractor
            upsample_features = self.config['ren']['parameters'].get('upsample_features', False)
            _, feature_maps = self.localizer.ren.feature_extractor(
                self.localizer.ren.extractor_name,
                image_batch,
                resize=upsample_features
            )

            # Get region tokens from region encoder
            # Ensure grid_points are on correct device
            grid_points = self.localizer.ren.grid_points.to(self.device)
            outputs = self.localizer.ren.region_encoder(
                feature_maps,
                [grid_points for _ in range(image_batch.shape[0])]
            )
            
            # Extract tokens from outputs
            region_tokens = outputs['pred_tokens'].cpu()  # (batch, num_regions, 1024)
            region_tokens = region_tokens.squeeze(0)  # (num_regions, 1024)

        return region_tokens


def main():
    parser = argparse.ArgumentParser(
        description='Index a video for efficient text-query retrieval.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output directory for index (required)')
    parser.add_argument('--sample-rate', type=int, default=None, dest='sample_rate',
                        help='Process every Nth frame (default: from config)')
    parser.add_argument('--skip-ocr', action='store_true', dest='skip_ocr',
                        help='Skip OCR extraction (faster, no brand recognition)')

    args = parser.parse_args()

    if args.output is None:
        print("Error: --output directory is required")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        # Help the user recover if they accidentally passed a directory
        if os.path.isdir(args.video):
            candidates = [f for f in os.listdir(args.video) if f.endswith('.mp4')]
            if candidates:
                print(f"  Hint: '{args.video}' is a directory. Did you mean one of:")
                for c in candidates:
                    print(f"    python prepare_index.py \"{os.path.join(args.video, c)}\" --output test_index/")
        sys.exit(1)

    if os.path.isdir(args.video):
        # User passed a directory instead of a file — find MP4 inside and suggest
        candidates = [f for f in os.listdir(args.video) if f.endswith('.mp4')]
        print(f"Error: '{args.video}' is a directory, not a video file.")
        if candidates:
            print(f"  Hint: Did you mean one of these files inside it?")
            for c in candidates:
                print(f"    python prepare_index.py \"{os.path.join(args.video, c)}\" --output test_index/")
        sys.exit(1)

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Build index
    indexer = VideoIndexer(config)
    metadata = indexer.index_video(args.video, args.output, sample_rate=args.sample_rate,
                                   skip_ocr=args.skip_ocr)

    print("\n=== Indexing Complete ===")
    for k, v in metadata.items():
        if k != 'index_path':
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
