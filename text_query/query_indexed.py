"""
Phase 2: Online query against indexed video.

Pipeline
--------
1. Load FAISS index + clip_embeddings.npy (built by prepare_index.py)
2. Encode text query with CLIP  →  same joint space as indexed image embeddings
3. Full scan: cosine similarity of text vs every indexed frame
4. Temporal segmentation: group above-threshold frames into contiguous segments;
   take the LAST segment, then the peak-similarity frame within it.
   (Avoids isolated false-positive frames at the tail of the video.)
5. CLIP crop scoring on the last-occurrence frame: N×N grid → best region point
6. SAM2: point → bbox
7. SAM2: forward/backward tracking in context window
8. Export result clip with bbox overlay

Usage:
    python query_indexed.py "coffee mug" --index <index_dir> --video <video_path> --output <output_dir>
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    import faiss
    HAS_GPU = faiss.get_num_gpus() > 0
    if HAS_GPU:
        print(f"GPU-accelerated FAISS enabled (GPUs available: {faiss.get_num_gpus()})")
except ImportError:
    print("Error: faiss not installed.")
    print("  conda install -c conda-forge faiss-gpu -y  # GPU")
    print("  conda install -c conda-forge faiss-cpu -y  # CPU-only")
    sys.exit(1)

from localizer import TextQueryLocalizer


class IndexedQueryEngine:
    """Query against a FAISS-indexed video using CLIP text-image retrieval."""

    def __init__(self, config: Dict, index_dir: str):
        self.config = config
        self.localizer = TextQueryLocalizer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_gpu = HAS_GPU
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        self.index_dir = index_dir
        self._load_index()

    # ------------------------------------------------------------------ #
    # Index loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_index(self):
        index_path = os.path.join(self.index_dir, 'faiss.index')
        metadata_path = os.path.join(self.index_dir, 'metadata.json')
        embeddings_path = os.path.join(self.index_dir, 'clip_embeddings.npy')

        for p in [index_path, metadata_path, embeddings_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Index file not found: {p}")

        print(f"Loading index from {self.index_dir}...")
        cpu_index = faiss.read_index(index_path)
        if self.use_gpu:
            print("  Moving index to GPU for fast search...")
            self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        else:
            self.faiss_index = cpu_index

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # L2-normalised CLIP image embeddings, shape (N, D).  ~12 MB at sample_rate=10.
        self.clip_embeddings = np.load(embeddings_path)

        print(f"  Loaded {self.faiss_index.ntotal} frames  (d={self.faiss_index.d})")
        print(f"  FPS: {self.metadata['fps']}")
        print(f"  Total video frames: {self.metadata['total_frames']}")

    # ------------------------------------------------------------------ #
    # Main query                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        text_query: str,
        video_path: str,
        output_dir: str,
        threshold: float = None,
        region_grid: int = 6,
    ) -> Dict:
        """
        Find the last genuine occurrence of text_query in the indexed video.

        'Last occurrence' = the peak-similarity frame inside the latest temporal
        segment of frames that all score above `threshold`.  Using the segment
        peak (not the chronologically last above-threshold frame) avoids isolated
        false-positive frames that appear after the object has left the scene.
        """
        if threshold is None:
            threshold = self.config['text_query'].get('similarity_threshold', 0.20)

        os.makedirs(output_dir, exist_ok=True)
        fps = self.metadata['fps']
        frame_metadata = self.metadata['frame_metadata']
        sample_rate = self.metadata.get('sample_rate', 10)

        # ---- Step 1: encode text query ----
        print(f"\nQuery: '{text_query}'")
        text_feat = self.localizer.encode_text(text_query)    # (1, D)
        text_np = text_feat.cpu().numpy().astype(np.float32)  # (1, D)
        print(f"  Embedding dim: {text_feat.shape[-1]}  |  index dim: {self.faiss_index.d}")

        if text_feat.shape[-1] != self.faiss_index.d:
            raise RuntimeError(
                f"Dimension mismatch: text={text_feat.shape[-1]}, "
                f"index={self.faiss_index.d}. Delete the index and rebuild."
            )

        # ---- Step 2: cosine similarity over ALL indexed frames ----
        # clip_embeddings is L2-normalised → dot product = cosine similarity.
        all_sims = (self.clip_embeddings @ text_np.T).squeeze()   # (N,)
        n_above = int((all_sims >= threshold).sum())
        print(f"\n  Similarities — max={all_sims.max():.4f}  "
              f"mean={all_sims.mean():.4f}  "
              f"above {threshold}: {n_above}/{len(all_sims)}")

        if n_above == 0:
            print(f"  [Hint] Try --threshold {max(0.05, float(all_sims.max()) - 0.02):.2f}")
            raise RuntimeError(
                f"No frames found above similarity threshold {threshold}"
            )

        # ---- Step 3: temporal segmentation → last genuine segment ----
        # Sort above-threshold frame indices by video time.
        above = np.where(all_sims >= threshold)[0]
        sorted_above = sorted(above.tolist(),
                              key=lambda i: frame_metadata[i]['frame_idx'])

        # Two consecutive sampled frames belong to the same segment when the gap
        # between their video frame indices is ≤ gap_frames.
        # At 60 fps / sample_rate=10, 2 seconds = 12 sampled frames gap.
        gap_frames = max(1, int(2.0 * fps / sample_rate)) * sample_rate

        segments: List[List[int]] = []
        current: List[int] = [sorted_above[0]]
        for idx in sorted_above[1:]:
            prev_vidx = frame_metadata[current[-1]]['frame_idx']
            curr_vidx = frame_metadata[idx]['frame_idx']
            if curr_vidx - prev_vidx > gap_frames:
                segments.append(current)
                current = [idx]
            else:
                current.append(idx)
        segments.append(current)

        # Filter to segments with ≥ 2 sampled frames (suppresses single-frame spikes).
        valid = [s for s in segments if len(s) >= 2]
        if not valid:
            valid = segments   # fallback: accept singletons if nothing else

        print(f"  {len(segments)} segment(s) found, {len(valid)} valid (≥2 frames):")
        for s in valid[-5:]:
            pk = max(s, key=lambda i: all_sims[i])
            t0 = frame_metadata[s[0]]['frame_idx'] / fps
            t1 = frame_metadata[s[-1]]['frame_idx'] / fps
            print(f"    t={t0:.1f}–{t1:.1f}s  n={len(s)}  peak_sim={all_sims[pk]:.3f}")

        # Last valid segment → its peak-similarity frame is the "last occurrence"
        last_segment = valid[-1]
        last_meta_idx = max(last_segment, key=lambda i: all_sims[i])
        last_frame_idx = frame_metadata[last_meta_idx]['frame_idx']
        last_sim = float(all_sims[last_meta_idx])
        print(f"\n  Last occurrence → frame {last_frame_idx}  "
              f"(t={last_frame_idx/fps:.2f}s, sim={last_sim:.3f})")

        # ---- Step 4: CLIP crop scoring on the last-occurrence frame ----
        print(f"\nCLIP crop scoring ({region_grid}×{region_grid} grid)...")
        last_frame_rgb = self._load_single_frame(video_path, last_frame_idx)
        region_point, region_score = self._find_best_region_clip(
            last_frame_rgb, text_feat, grid_size=region_grid
        )
        print(f"  Best region point: {region_point}  (CLIP score: {region_score:.3f})")

        # ---- Step 5: context window centred on the last-occurrence frame ----
        context_seconds = self.config['text_query'].get('context_seconds', 5.0)
        half_span = int(context_seconds * fps / 2)

        print(f"\nLoading ±{context_seconds/2:.1f}s context window...")
        frames, frame_indices = self._load_frame_window(
            video_path, last_frame_idx, half_span
        )
        start_idx = max(0, last_frame_idx - half_span)
        center_local = last_frame_idx - start_idx   # may be < half_span near video edges

        # ---- Step 6: SAM2 point → bbox ----
        print("SAM2: estimating bbox from region point...")
        bbox = self.localizer.point_to_bbox(
            frames[center_local], region_point, text_feat
        )
        print(f"  Bbox: {bbox}")

        # Save a debug image so you can verify the detected region visually
        debug_frame = frames[center_local].copy()
        px, py = int(region_point[0]), int(region_point[1])
        bx, by, bw, bh = [int(v) for v in bbox]
        cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
        cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        debug_path = os.path.join(output_dir, 'debug_last_frame.jpg')
        cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        print(f"  Debug frame saved: {debug_path}")

        # ---- Step 7: SAM2 tracking ----
        print("SAM2: tracking bbox through context window...")
        track = self.localizer.track_from_bbox(
            frames, center_local, bbox, half_span
        )
        print(f"  Tracked {len(track)} frames")
        if len(track) > 0:
            # Map local window indices to absolute video frame indices.
            for t in track:
                t['frame_idx'] = int(t['frame_idx']) + start_idx
        if len(track) == 0:
            track = [{'frame_idx': last_frame_idx,
                      'bbox': [bx, by, bw, bh]}]
            print("  [Warn] SAM2 produced no masks; exporting single-frame bbox.")

        # ---- Step 8: export clip ----
        print("\nExporting result clip...")
        self.localizer.export_clip(
            video_path, track, last_frame_idx,
            fps, os.path.join(output_dir, 'last_occurrence.mp4')
        )

        result = {
            'query': text_query,
            'video_path': video_path,
            'last_frame_idx': last_frame_idx,
            'last_frame_timestamp': round(last_frame_idx / fps, 3),
            'clip_similarity': round(last_sim, 4),
            'region_point': list(region_point),
            'region_clip_score': round(region_score, 4),
            'similarity_threshold': threshold,
            'valid_segments': len(valid),
            'frames_above_threshold': int(n_above),
            'context_seconds': context_seconds,
            'fps': fps,
        }
        with open(os.path.join(output_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        return result

    # ------------------------------------------------------------------ #
    # CLIP crop scoring                                                    #
    # ------------------------------------------------------------------ #

    def _find_best_region_clip(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        grid_size: int = 6,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Divide the frame into a grid_size×grid_size grid, encode each crop with
        the CLIP image encoder, return the center of the highest-scoring crop.
        Valid because CLIP text and CLIP image share the same embedding space.
        """
        h, w = frame_rgb.shape[:2]
        ph, pw = h // grid_size, w // grid_size

        crops, centers = [], []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, x1 = i * ph, j * pw
                y2, x2 = min(h, y1 + ph), min(w, x1 + pw)
                crops.append(self.localizer.clip_preprocess(
                    Image.fromarray(frame_rgb[y1:y2, x1:x2])
                ))
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

        crop_batch = torch.stack(crops).to(self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            crop_feats = self.localizer.clip_model.encode_image(crop_batch).float()
        crop_feats = F.normalize(crop_feats, p=2, dim=-1)           # (N², D)
        scores = (crop_feats @ text_feat.T).squeeze(-1).cpu().numpy()
        best = int(np.argmax(scores))
        return centers[best], float(scores[best])

    # ------------------------------------------------------------------ #
    # Video helpers                                                        #
    # ------------------------------------------------------------------ #

    def _load_single_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_frame_window(
        self,
        video_path: str,
        center_idx: int,
        half_span: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Load every frame in [center_idx−half_span, center_idx+half_span] via direct seek."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_idx = max(0, center_idx - half_span)
        end_idx = min(total - 1, center_idx + half_span)

        frames, frame_indices = [], []
        for idx in range(start_idx, end_idx + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_indices.append(idx)
        cap.release()
        return frames, frame_indices


def main():
    parser = argparse.ArgumentParser(
        description='Query indexed video for text-described objects.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('query', type=str, help='Text query (e.g., "fork")')
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float, default=None,
                        help='CLIP cosine similarity threshold (default: from config, 0.20)')
    parser.add_argument('--region-grid', type=int, default=6, dest='region_grid',
                        help='N×N grid for CLIP crop scoring (default: 6)')

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path
        )
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = (
        args.output or
        f"query_results/{args.query.lower().replace(' ', '_')}"
    )

    engine = IndexedQueryEngine(config, args.index)
    result = engine.query(
        args.query, args.video, output_dir,
        threshold=args.threshold,
        region_grid=args.region_grid,
    )

    print("\n=== Query Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
