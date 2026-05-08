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

        REN region tokens live in DINOv2 space (vision-only) and cannot be compared
        to CLIP text embeddings — so they are skipped by default for text-query mode.
        Pass skip_ren=False only if you need the tokens for a separate visual-query pipeline.

        Args:
            video_path:  Path to video file
            output_dir:  Directory to save index files
            sample_rate: Process every Nth frame (default: from config)
            skip_ren:    Skip expensive REN token extraction (default: True)

        Returns:
            metadata: Dict with index_path, num_frames, fps, etc.
        """
        if sample_rate is None:
            sample_rate = self.config['text_query'].get('frame_sample_rate', 2)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Indexing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frames we will actually process
        sampled_indices = list(range(0, total_frames, sample_rate))
        print(f"  Source : {total_frames} frames @ {fps:.2f} fps, {frame_width}x{frame_height}")
        print(f"  Sampling: every {sample_rate}th frame → {len(sampled_indices)} frames to process")

        clip_embeddings  = []
        frame_metadata   = []
        all_region_tokens = []

        if skip_ren:
            print("  REN token extraction skipped (not needed for text-query mode)")
        if skip_ocr or not HAS_OCR:
            print("  OCR text extraction skipped (brand recognition disabled)")
        else:
            print("  OCR text extraction enabled (brand/label recognition)")
            if self.ocr_reader is None:
                print("  Initializing EasyOCR model on GPU...")
                self.ocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=self.device.type == 'cuda',
                    verbose=False,
                )
                print("  EasyOCR ready.")

        # Direct-seek: jump to each target frame index rather than reading all
        # frames sequentially.  Avoids exhausting the multi-stream packet budget.
        for sampled_frame_idx, frame_idx in enumerate(sampled_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue   # skip unreadable frames, keep going

            if sampled_frame_idx % 50 == 0:
                pct = 100 * sampled_frame_idx / len(sampled_indices)
                print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}  "
                      f"({sampled_frame_idx}/{len(sampled_indices)} sampled)", end='\r')

            clip_feat = self._extract_clip_embedding(frame)
            clip_embeddings.append(clip_feat)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            ocr_texts = self._extract_ocr(frame) if (not skip_ocr and HAS_OCR) else []
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if not skip_ren:
                region_tokens = self._extract_region_tokens(frame)
                region_count  = region_tokens.shape[0] if region_tokens.numel() > 0 else 0
                frame_metadata.append({
                    'frame_idx':        frame_idx,
                    'sampled_frame_idx': sampled_frame_idx,
                    'timestamp':        frame_idx / fps,
                    'region_count':     region_count,
                    'region_start_idx': len(all_region_tokens),
                    'ocr_texts':        ocr_texts,
                })
                if region_count > 0:
                    all_region_tokens.append(region_tokens.cpu().numpy())
            else:
                frame_metadata.append({
                    'frame_idx':        frame_idx,
                    'sampled_frame_idx': sampled_frame_idx,
                    'timestamp':        frame_idx / fps,
                    'ocr_texts':        ocr_texts,
                })

        cap.release()
        print(f"\n  Sampled {len(clip_embeddings)} frames")
        if not skip_ren:
            print(f"  Total regions: {sum(m.get('region_count', 0) for m in frame_metadata)}")

        # Stack embeddings
        clip_embeddings = np.vstack(clip_embeddings)  # (num_frames, clip_dim)

        # Build FAISS index on CLIP embeddings
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

    def _extract_clip_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract CLIP image embedding from a frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            L2-normalized embedding (1280,)
        """
        # Convert BGR to RGB and PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Extract CLIP embedding using localizer's preprocess
        img_tensor = self.localizer.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            clip_embedding = self.localizer.clip_model.encode_image(img_tensor).float()
        
        # Normalize and return
        clip_embedding = torch.nn.functional.normalize(clip_embedding, p=2, dim=-1)
        return clip_embedding.cpu().numpy().astype(np.float32)[0]

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
