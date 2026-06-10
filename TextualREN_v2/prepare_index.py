"""
Phase 1: Offline video indexing (Textual-REN v2 — no OCR).

Samples frames from video, extracts CLIP CLS embeddings + patch tokens,
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
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '65536'

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
    sys.exit(1)

from localizer import TextQueryLocalizer


class VideoIndexer:
    """
    Build FAISS index from video frames for efficient text-query retrieval.

    Pipeline:
    1. Sample frames at configurable rate
    2. Extract CLIP CLS embeddings (1024-dim) + 256 patch tokens (1024-dim each)
    3. Store metadata (frame_idx, timestamp)
    4. Build FAISS index on CLS embeddings
    5. Persist index + metadata + patch embeddings for query phase
    """

    def __init__(self, config: Dict):
        self.config = config
        self.localizer = TextQueryLocalizer(config)

        # CLIP embedding dim: read from config (ViT-g-14 joint space = 1024)
        self.clip_dim = config.get('text_query', {}).get('faiss', {}).get('clip_dim', 1024)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GPU support
        self.use_gpu = HAS_GPU
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

    # How many frames to encode in one GPU CLIP forward pass.
    CLIP_BATCH_SIZE = 64

    def index_video(
        self,
        video_path: str,
        output_dir: str,
        sample_rate: int = None,
    ) -> Dict:
        """
        Index a video: extract CLIP image embeddings and build FAISS index.

        Performance notes
        -----------------
        * CLIP is processed in batches of CLIP_BATCH_SIZE (default 64) so the
          GPU runs at full utilisation.
        * Frames are read sequentially (no per-frame seek) which is 10-50x
          faster than cap.set(POS_FRAMES) on compressed video files.
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
              f"({frame_width}x{frame_height})")
        print(f"  Sampling: every {sample_rate}th frame -> ~{n_sampled} frames")
        print(f"  CLIP batch size : {self.CLIP_BATCH_SIZE}")

        clip_embeddings   = []
        patch_embeddings  = []
        frame_metadata    = []

        do_patches = self.config.get('text_query', {}).get('faiss', {}).get('use_patch_rerank', True)

        # ── accumulation buffers for CLIP batching ──
        batch_frames: List[np.ndarray] = []
        batch_meta:   List[dict]       = []

        sampled_frame_idx = 0

        # Sequential read — much faster than seeking
        raw_frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_frame_idx % sample_rate == 0:
                stub = {
                    'frame_idx':         raw_frame_idx,
                    'sampled_frame_idx': sampled_frame_idx,
                    'timestamp':         raw_frame_idx / fps,
                }
                batch_frames.append(frame)
                batch_meta.append(stub)

                # ── flush CLIP batch ──
                if len(batch_frames) >= self.CLIP_BATCH_SIZE:
                    feats = self._extract_clip_batch(batch_frames)
                    clip_embeddings.append(feats)
                    if do_patches:
                        patches = self._extract_patch_tokens_batch(batch_frames)
                        patch_embeddings.append(patches)
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
            if do_patches:
                patches = self._extract_patch_tokens_batch(batch_frames)
                patch_embeddings.append(patches)
            frame_metadata.extend(batch_meta)

        cap.release()

        if not clip_embeddings:
            raise RuntimeError("No frames could be read from the video.")

        clip_embeddings_np = np.vstack(clip_embeddings)   # (N, D)
        print(f"\n  Done — {len(frame_metadata)} frames indexed")

        # Build FAISS index on CLIP embeddings
        clip_embeddings = clip_embeddings_np
        actual_dim = clip_embeddings.shape[1]
        if actual_dim != self.clip_dim:
            print(f"  [Warning] Actual CLIP embedding dim ({actual_dim}) differs from "
                  f"config clip_dim ({self.clip_dim}). Using actual dim.")
            self.clip_dim = actual_dim
        print(f"\nBuilding FAISS index (dim={self.clip_dim})...")
        if self.use_gpu:
            print("  Using GPU-accelerated index")
            cpu_index = faiss.IndexFlatIP(self.clip_dim)
            cpu_index.add(clip_embeddings.astype(np.float32))
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        else:
            print("  Using CPU-based index")
            index = faiss.IndexFlatIP(self.clip_dim)
            index.add(clip_embeddings.astype(np.float32))
        print(f"  Index built with {index.ntotal} frames")

        # Save index and metadata
        index_path = os.path.join(output_dir, 'faiss.index')
        metadata_path = os.path.join(output_dir, 'metadata.json')
        embeddings_path = os.path.join(output_dir, 'clip_embeddings.npy')

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
        np.save(embeddings_path, clip_embeddings)

        patch_emb_path = os.path.join(output_dir, 'patch_embeddings.npy')
        if do_patches and patch_embeddings:
            patch_embeddings_np = np.concatenate(patch_embeddings, axis=0)
            np.save(patch_emb_path, patch_embeddings_np)
            size_mb = patch_embeddings_np.nbytes / (1024 * 1024)
            print(f"  Patch embeddings: {patch_embeddings_np.shape} ({size_mb:.0f} MB)")

        print(f"Index saved to: {output_dir}")
        print(f"  - FAISS index: {index_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - CLIP embeddings: {embeddings_path}")
        if do_patches and patch_embeddings:
            print(f"  - Patch embeddings: {patch_emb_path}")

        return {
            'index_path': index_path,
            'metadata_path': metadata_path,
            'embeddings_path': embeddings_path,
            'num_frames': len(clip_embeddings),
            'fps': fps,
        }

    def _extract_clip_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract L2-normalised CLIP CLS embeddings for a batch of BGR frames."""
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

    def _extract_patch_tokens_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract CLIP patch tokens for a batch of BGR frames.
        Each frame produces 256 patch tokens of dim 1024 (projected to joint space).
        """
        imgs = torch.stack([
            self.localizer.clip_preprocess(
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            )
            for f in frames
        ]).to(self.device)

        visual = self.localizer.clip_model.visual

        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            out = visual.forward_intermediates(
                imgs, indices=[-1], intermediates_only=False, output_fmt='NLC'
            )
            patches = out['image_intermediates'][-1].float()  # (B, 256, 1408)
            patches = visual.ln_post(patches) @ visual.proj.float()  # (B, 256, 1024)

        patches = torch.nn.functional.normalize(patches, p=2, dim=-1)
        return patches.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Index a video for efficient text-query retrieval (v2, no OCR).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output directory for index (required)')
    parser.add_argument('--sample-rate', type=int, default=None, dest='sample_rate',
                        help='Process every Nth frame (default: from config)')

    args = parser.parse_args()

    if args.output is None:
        print("Error: --output directory is required")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Error: video not found: {args.video}")
        if os.path.isdir(args.video):
            candidates = [f for f in os.listdir(args.video) if f.endswith('.mp4')]
            if candidates:
                print(f"  Hint: '{args.video}' is a directory. Did you mean one of:")
                for c in candidates:
                    print(f"    python prepare_index.py \"{os.path.join(args.video, c)}\" --output test_index/")
        sys.exit(1)

    if os.path.isdir(args.video):
        candidates = [f for f in os.listdir(args.video) if f.endswith('.mp4')]
        print(f"Error: '{args.video}' is a directory, not a video file.")
        if candidates:
            print(f"  Hint: Did you mean one of these files inside it?")
            for c in candidates:
                print(f"    python prepare_index.py \"{os.path.join(args.video, c)}\" --output test_index/")
        sys.exit(1)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    indexer = VideoIndexer(config)
    metadata = indexer.index_video(args.video, args.output, sample_rate=args.sample_rate)

    print("\n=== Indexing Complete ===")
    for k, v in metadata.items():
        if k != 'index_path':
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
