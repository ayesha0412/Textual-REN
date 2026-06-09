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

try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Warning: rapidfuzz not installed -- OCR fusion disabled.")
    print("  pip install rapidfuzz")

try:
    import easyocr as _easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

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
from grounding_dino import GroundingDINOLocalizer


# ====================================================================== #
# STAGE 3: SELECTION POLICY                                              #
# ====================================================================== #
# Adapted from visual_query/models.py CandidateSelector                   #

class SelectionPolicy:
    """RELOCATE Stage 3: Select candidates using temporal segmentation + ranking."""

    def __init__(self, config: Dict):
        self.config = config
        self.policy_name = config.get('text_query', {}).get('selection_policy', 'last')
        self.top_k = config.get('text_query', {}).get('selection_top_k', 10)
        self.top_p = config.get('text_query', {}).get('selection_top_p', 0.9)
        self.nms_threshold = config.get('text_query', {}).get('nms_threshold', 0.5)
        self.nms_window = config.get('text_query', {}).get('nms_window', None)

    def temporal_segmentation(self, all_sims, frame_metadata, fps, sample_rate, threshold=0.18):
        """
        Group above-threshold frames into contiguous temporal segments.
        Returns: list of segments, each segment = set of metadata indices
        """
        # Find frames above threshold
        above_threshold = {i for i, sim in enumerate(all_sims) if sim >= threshold}
        if not above_threshold:
            return [], []

        # Group into contiguous segments (gap tolerance: 2 seconds)
        gap_tolerance_frames = max(1, int(2.0 * fps / sample_rate)) * sample_rate
        sorted_indices = sorted(above_threshold)

        segments = []
        current_segment = {sorted_indices[0]}

        for i in range(1, len(sorted_indices)):
            frame_idx_curr = frame_metadata[sorted_indices[i]]['frame_idx']
            frame_idx_prev = frame_metadata[sorted_indices[i-1]]['frame_idx']

            if frame_idx_curr - frame_idx_prev <= gap_tolerance_frames:
                current_segment.add(sorted_indices[i])
            else:
                segments.append(current_segment)
                current_segment = {sorted_indices[i]}

        if current_segment:
            segments.append(current_segment)

        # Filter: keep segments with ≥2 frames (suppress single-frame spikes)
        valid_segments = [s for s in segments if len(s) >= 2]

        return valid_segments, sorted_indices

    def inter_frame_nms(self, all_sims, frame_metadata, fps, sample_rate,
                        nms_threshold=0.5, nms_window=None):
        """
        Suppress temporally nearby detections (NMS or windowing).
        """
        if nms_window is not None:
            # Window-based: group frames, pick max per window
            window_size = int(nms_window * fps / sample_rate)
            selected = []
            for i in range(0, len(all_sims), window_size):
                window = range(i, min(i + window_size, len(all_sims)))
                if window:
                    best_idx = max(window, key=lambda j: all_sims[j])
                    selected.append(best_idx)
            return selected
        else:
            # Threshold-based NMS: suppress within nms_threshold seconds
            nms_frames = int(nms_threshold * fps / sample_rate)
            selected = []
            sorted_idx = sorted(range(len(all_sims)), key=lambda i: all_sims[i], reverse=True)
            suppressed = set()

            for idx in sorted_idx:
                if idx in suppressed:
                    continue
                selected.append(idx)
                frame_idx = frame_metadata[idx]['frame_idx']
                # Suppress nearby frames
                for other_idx in range(len(all_sims)):
                    if abs(frame_metadata[other_idx]['frame_idx'] - frame_idx) <= nms_frames:
                        suppressed.add(other_idx)

            return sorted(selected)

    def topk_selection(self, candidates, top_k):
        """Select top-K candidates by similarity score."""
        if len(candidates) <= top_k:
            return candidates
        sorted_cands = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
        return sorted_cands[:top_k]

    def topp_selection(self, candidates, top_p):
        """Nucleus sampling: select candidates until cumulative prob ≥ top_p."""
        if not candidates:
            return []

        sims = np.array([c['similarity'] for c in candidates])
        sims_normalized = sims / (sims.sum() + 1e-10)

        sorted_idx = np.argsort(-sims)
        cumsum = 0.0
        selected_idx = []

        for idx in sorted_idx:
            cumsum += sims_normalized[idx]
            selected_idx.append(idx)
            if cumsum >= top_p:
                break

        return [candidates[i] for i in selected_idx]

    def select(self, all_sims, frame_metadata, fps, sample_rate, threshold=0.18):
        """
        RELOCATE Stage 3: Temporal segmentation + deterministic/probabilistic selection.

        Returns: (candidates, num_valid_segments)
            candidates: list of candidate dicts: [{'frame_idx': idx, 'similarity': sim}, ...]
            num_valid_segments: number of valid temporal segments found
        """
        # Temporal segmentation
        valid_segments, _ = self.temporal_segmentation(all_sims, frame_metadata, fps, sample_rate, threshold)
        num_valid = len(valid_segments)

        if not valid_segments:
            # Fallback: use frame with highest similarity
            best_idx = np.argmax(all_sims)
            return [{'frame_idx': frame_metadata[best_idx]['frame_idx'], 'similarity': float(all_sims[best_idx])}], 0

        # Build candidate list from segments (peak frame per segment)
        candidates = []
        for segment in valid_segments:
            peak_idx = max(segment, key=lambda i: all_sims[i])
            candidates.append({
                'frame_idx': frame_metadata[peak_idx]['frame_idx'],
                'meta_idx': peak_idx,
                'similarity': float(all_sims[peak_idx]),
                'segment': segment
            })

        # Rank candidates (last segment first for temporal preference)
        candidates = sorted(candidates, key=lambda x: -x['meta_idx'])  # Recent first

        # Apply selection policy
        if self.policy_name == 'last':
            result = candidates  # All peaks, recent-first; caller picks verified best
        elif self.policy_name == 'strongest':
            best = max(candidates, key=lambda x: x['similarity'])
            result = [best]
        elif self.policy_name == 'topk':
            result = self.topk_selection(candidates, self.top_k)
        elif self.policy_name == 'topp':
            result = self.topp_selection(candidates, self.top_p)
        else:
            result = candidates[:1]  # Default to last

        return result, num_valid


# ====================================================================== #
# STAGE 5: CANDIDATE REFINER (Multi-candidate REN refinement)             #
# ====================================================================== #
# Adapted from visual_query/models.py CandidateRefiner                    #

class CandidateRefiner:
    """RELOCATE Stage 5: Refine multiple candidates with spatial localization."""

    def __init__(self, query_engine, config: Dict, text_query: str = ""):
        self.query_engine = query_engine
        self.config = config
        self.text_query = text_query
        self.max_candidates = config.get('text_query', {}).get('max_candidates_to_refine', 5)
        self.skip_sam2 = config.get('text_query', {}).get('skip_sam2_eval', True)
        self.spatial_method = config.get('text_query', {}).get('spatial_method', 'grounding_dino')

    def _localize_grounding_dino(self, frame_rgb, text_feat, h, w):
        """Use Grounding DINO for direct text->bbox detection, CLIP re-ranks ties."""
        gdino_cfg = self.config.get('grounding_dino', {})
        result = self.query_engine.grounding_dino.best_box(
            frame_rgb, self.text_query,
            box_threshold=gdino_cfg.get('box_threshold', 0.30),
            text_threshold=gdino_cfg.get('text_threshold', 0.25),
            clip_model=self.query_engine.localizer.clip_model,
            clip_preprocess=self.query_engine.localizer.clip_preprocess,
            text_feat=text_feat,
        )
        if result is not None:
            bbox, score = result
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            return (cx, cy), score, bbox
        return (w // 2, h // 2), 0.0, None

    def _localize_ren_clip(self, frame_rgb, text_feat, h, w):
        """Legacy: REN grid proposals scored with CLIP crops."""
        try:
            region_point, region_score = self.query_engine._ren_guided_localize(frame_rgb, text_feat)
            return region_point, region_score, None
        except Exception as e:
            print(f"  REN localization failed: {e}")
            return (w // 2, h // 2), 0.0, None

    def refine(self, selected_candidates, frames, frame_indices, text_feat):
        """
        Refine top-K candidates using spatial localization.

        spatial_method controls the localizer:
          - "grounding_dino": text-conditioned detection (default, training-free)
          - "ren_clip": REN grid proposals + CLIP crop scoring (legacy)
        """
        refined = []
        h, w = frames[0].shape[:2] if frames else (0, 0)
        method = self.spatial_method

        if method == 'ren_clip':
            try:
                _ = self.query_engine.localizer.ren
            except Exception as e:
                print(f"  REN loading failed ({e}); falling back to grounding_dino")
                method = 'grounding_dino'

        print(f"  Spatial method: {method}")

        for cand in selected_candidates[:self.max_candidates]:
            frame_idx = cand['frame_idx']

            try:
                local_idx = frame_indices.index(frame_idx)
                frame_rgb = frames[local_idx]
            except (ValueError, IndexError):
                print(f"  Warning: frame {frame_idx} not in context window, skipping")
                continue

            # Spatial localization
            if method == 'grounding_dino':
                region_point, region_score, gdino_bbox = self._localize_grounding_dino(frame_rgb, text_feat, h, w)
            else:
                region_point, region_score, gdino_bbox = self._localize_ren_clip(frame_rgb, text_feat, h, w)

            # Bbox: use Grounding DINO bbox directly if available
            if gdino_bbox is not None:
                bbox = gdino_bbox
            elif self.skip_sam2:
                bbox_width = max(64, w // 4)
                bbox_height = max(64, h // 4)
                x_min = max(0, region_point[0] - bbox_width // 2)
                y_min = max(0, region_point[1] - bbox_height // 2)
                bbox = [x_min, y_min, bbox_width, bbox_height]
            else:
                try:
                    bbox_tensor = self.query_engine.localizer.point_to_bbox(
                        frame_rgb,
                        np.array([region_point[1], region_point[0]]),
                        text_feat,
                        inference_size=self.config.get('text_query', {}).get('sam_inference_size', 512)
                    )
                    bbox = bbox_tensor.tolist()
                except Exception as e:
                    print(f"  SAM2 failed for frame {frame_idx}: {e}, using fast path")
                    bbox_width = max(64, w // 4)
                    bbox_height = max(64, h // 4)
                    x_min = max(0, region_point[0] - bbox_width // 2)
                    y_min = max(0, region_point[1] - bbox_height // 2)
                    bbox = [x_min, y_min, bbox_width, bbox_height]

            refined.append({
                'frame_idx': frame_idx,
                'region_point': region_point,
                'region_score': region_score,
                'bbox': bbox,
                'refined_score': region_score
            })

        refined.sort(key=lambda x: x['refined_score'], reverse=True)
        return refined


class IndexedQueryEngine:
    """Query against a FAISS-indexed video using CLIP text-image retrieval."""

    def __init__(self, config: Dict, index_dir: str):
        self.config = config
        self.localizer = TextQueryLocalizer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_gpu = HAS_GPU
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        self._ocr_reader = None   # lazily initialized for query-time OCR bbox
        self._grounding_dino = None  # lazily initialized for spatial localization

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

        # Patch-level embeddings for re-ranking (optional, from prepare_index.py)
        patch_path = os.path.join(self.index_dir, 'patch_embeddings.npy')
        use_patches = self.config.get('text_query', {}).get('faiss', {}).get('use_patch_rerank', True)
        if use_patches and os.path.exists(patch_path):
            self.patch_embeddings = np.load(patch_path)  # (N, 256, 1024)
            print(f"  Patch embeddings loaded: {self.patch_embeddings.shape}")
        else:
            self.patch_embeddings = None

        print(f"  Loaded {self.faiss_index.ntotal} frames  (d={self.faiss_index.d})")
        print(f"  FPS: {self.metadata['fps']}")
        print(f"  Total video frames: {self.metadata['total_frames']}")

    @property
    def grounding_dino(self) -> GroundingDINOLocalizer:
        if self._grounding_dino is None:
            gdino_cfg = self.config.get('grounding_dino', {})
            self._grounding_dino = GroundingDINOLocalizer(
                model_id=gdino_cfg.get('model_id', 'IDEA-Research/grounding-dino-tiny'),
                device=self.device,
            )
        return self._grounding_dino

    # ------------------------------------------------------------------ #
    # Main query                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        text_query: str,
        video_path: str,
        output_dir: str,
        threshold: float = None,
        region_grid: int = 3,
        ocr_weight: float = None,
        neg_queries: List[str] = None,
        neg_weight: float = None,
        # Ablation flags — disable individual components for comparative study
        ablation_no_compositional: bool = False,
        ablation_no_ocr: bool = False,
        ablation_no_verification: bool = False,
        ablation_use_strongest: bool = False,
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
        if ocr_weight is None:
            ocr_weight = self.config.get('text_query', {}).get('ocr_weight', 0.3)
        if neg_weight is None:
            neg_weight = self.config.get('text_query', {}).get('neg_weight', 0.0)
        if neg_queries is None:
            neg_queries = []

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

        # ---- Step 2: compositional CLIP scoring ----
        _comp_enabled = self.config.get('text_query', {}).get('use_compositional', False)
        sub_queries = self._decompose_query(text_query) if _comp_enabled else [text_query]
        if ablation_no_compositional:
            sub_queries = [text_query]   # ablation: force disable
        if len(sub_queries) > 1:
            print(f"  Compositional sub-queries: {sub_queries}")

        sub_sims_list: List[np.ndarray] = []
        for sq in sub_queries:
            sq_np = text_np if sq == text_query else (
                self.localizer.encode_text(sq).cpu().numpy().astype(np.float32)
            )
            sub_sims_list.append((self.clip_embeddings @ sq_np.T).squeeze())

        if len(sub_sims_list) == 1:
            clip_sims = sub_sims_list[0]
        else:
            noun_sims = sub_sims_list[1]
            # Only apply compositional weighting when the noun sub-query has
            # meaningful signal.  If "painting" or "picture" max out at 0.16
            # across all frames, weighting them in drags correct frames below
            # threshold instead of suppressing false positives.
            # Heuristic: noun must hit at least the same threshold as the full
            # query to be trusted as a discriminator.
            noun_signal = float(noun_sims.max())
            noun_useful = noun_signal >= threshold
            if noun_useful:
                clip_sims = (0.7 * sub_sims_list[0] + 0.3 * noun_sims).astype(np.float32)
                print(f"  Compositional: active -- noun '{sub_queries[1]}' max={noun_signal:.3f}")
            else:
                clip_sims = sub_sims_list[0]
                print(f"  Compositional: skipped -- noun '{sub_queries[1]}' signal too weak "
                      f"({noun_signal:.3f} < {threshold}), using full query only")

        # ---- Step 2a: optional negative query suppression ----
        # Penalize frames that also strongly match known confounders.
        if neg_queries and neg_weight > 0.0:
            neg_text = self.localizer.encode_text(', '.join(neg_queries))
            neg_np = neg_text.cpu().numpy().astype(np.float32)
            neg_sims = (self.clip_embeddings @ neg_np.T).squeeze()  # (N,)
            clip_sims = clip_sims - (neg_weight * neg_sims)
            print(f"  Negative prompt: {neg_queries} (weight={neg_weight})")

        # ---- Step 2b: OCR brand/text fusion ----
        # Auto-detect whether this is a brand/text query or a plain object query.
        # Brand queries  ("Yorkshire Tea", "Twinings") → OCR fusion + OCR bbox.
        # Object queries ("knife", "keys", "fork")     → pure CLIP, OCR skipped.
        is_brand = self._is_brand_query(text_query) and not ablation_no_ocr
        print(f"  Query type: {'BRAND/TEXT -- OCR fusion enabled' if is_brand else 'OBJECT -- pure CLIP (OCR skipped)'}")

        if is_brand:
            ocr_scores = self._compute_ocr_scores(text_query, frame_metadata, window=2)
            # Only trust very high OCR matches (≥0.85) to avoid spurious text hits
            if ocr_scores.max() < 0.85:
                ocr_scores[:] = 0.0
            n_ocr_hits = int((ocr_scores >= 0.85).sum())
            if n_ocr_hits > 0:
                print(f"  OCR hits (≥0.85 match): {n_ocr_hits}/{len(ocr_scores)} frames")
                print(f"  OCR weight: {ocr_weight}  (fused = CLIP + {ocr_weight} * OCR)")
            else:
                print(f"  OCR: index has no clear text match -- falling back to CLIP only")
        else:
            ocr_scores = np.zeros(len(frame_metadata), dtype=np.float32)
            n_ocr_hits = 0

        # ---- Step 2c: patch-level re-ranking ----
        if self.patch_embeddings is not None:
            clip_sims = self._patch_rerank(text_np, clip_sims)
            print(f"  Patch re-ranking: applied (top-{self.config.get('text_query', {}).get('faiss', {}).get('patch_top_k', 100)} frames)")

        all_sims = clip_sims + ocr_weight * ocr_scores

        # ---- Step 2d: adaptive threshold ----
        use_adaptive = self.config.get('text_query', {}).get('adaptive_threshold', False)
        if use_adaptive:
            alpha = self.config.get('text_query', {}).get('threshold_alpha', 1.0)
            adaptive_tau = self._adaptive_threshold(all_sims, alpha=alpha)
            print(f"  Adaptive threshold: tau={adaptive_tau:.4f} (alpha={alpha}, "
                  f"mean={all_sims.mean():.4f}, std={all_sims.std():.4f}, "
                  f"fixed was {threshold})")
            threshold = adaptive_tau

        n_above = int((all_sims >= threshold).sum())
        print(f"\n  Fused scores -- max={all_sims.max():.4f}  "
              f"mean={all_sims.mean():.4f}  "
              f"above {threshold}: {n_above}/{len(all_sims)}")

        if n_above == 0:
            print(f"  [Hint] Try --threshold {max(0.05, float(all_sims.max()) - 0.02):.2f}")
            raise RuntimeError(
                f"No frames found above similarity threshold {threshold}"
            )

        # ========================================================================= #
        # RELOCATE STAGE 3: Selection Policy (temporal segmentation + ranking)    #
        # ========================================================================= #
        selection_policy = SelectionPolicy(self.config)
        selected_candidates, num_valid_segments = selection_policy.select(
            all_sims, frame_metadata, fps, sample_rate, threshold=threshold
        )

        print(f"  {num_valid_segments} segment(s) found, {len(selected_candidates)} candidate(s) selected (policy={selection_policy.policy_name})")

        # ---- Stage 3b: Grounding DINO + CLIP verified frame selection ----
        # GDino alone can confuse visually similar objects (red bucket -> "pink
        # flower in a pot"). For each candidate, run GDino then CLIP-score the
        # best crop against the text query. Accept only if CLIP crop score
        # exceeds min_crop_verify — this filters color/shape confusion.
        spatial_method = self.config.get('text_query', {}).get('spatial_method', 'grounding_dino')
        if spatial_method == 'grounding_dino' and len(selected_candidates) > 1:
            gdino_cfg = self.config.get('grounding_dino', {})
            box_thresh = gdino_cfg.get('box_threshold', 0.20)
            text_thresh = gdino_cfg.get('text_threshold', 0.20)
            min_clip_crop = self.config.get('text_query', {}).get('min_crop_verify', 0.17)
            max_verify = min(len(selected_candidates), 8)

            print(f"\n  Verifying up to {max_verify} candidates with GDino+CLIP "
                  f"(min_crop={min_clip_crop})...")
            verified_idx = None
            best_unverified = (None, -1.0)  # (idx, clip_score) fallback
            for ci, cand in enumerate(selected_candidates[:max_verify]):
                fidx = cand['frame_idx']
                try:
                    probe = self._load_single_frame(video_path, fidx)
                except Exception:
                    continue
                result = self.grounding_dino.best_box(
                    probe, text_query,
                    box_threshold=box_thresh, text_threshold=text_thresh,
                    clip_model=self.localizer.clip_model,
                    clip_preprocess=self.localizer.clip_preprocess,
                    text_feat=text_feat,
                )
                if result is None:
                    print(f"    candidate {ci}: frame {fidx} "
                          f"(t={fidx/fps:.1f}s) -- no detection")
                    continue
                bbox, clip_score = result
                if clip_score >= min_clip_crop:
                    print(f"    candidate {ci}: frame {fidx} "
                          f"(t={fidx/fps:.1f}s) -- CLIP crop={clip_score:.3f} >= "
                          f"{min_clip_crop} VERIFIED")
                    verified_idx = ci
                    break
                else:
                    print(f"    candidate {ci}: frame {fidx} "
                          f"(t={fidx/fps:.1f}s) -- CLIP crop={clip_score:.3f} < "
                          f"{min_clip_crop} (color/shape confusion?)")
                    if clip_score > best_unverified[1]:
                        best_unverified = (ci, clip_score)

            if verified_idx is not None and verified_idx != 0:
                old_frame = selected_candidates[0]['frame_idx']
                selected_candidates = [selected_candidates[verified_idx]] + \
                    [c for i, c in enumerate(selected_candidates) if i != verified_idx]
                print(f"  Verified: frame {selected_candidates[0]['frame_idx']} "
                      f"(was {old_frame})")
            elif verified_idx is None and best_unverified[0] is not None:
                bi = best_unverified[0]
                if bi != 0:
                    old_frame = selected_candidates[0]['frame_idx']
                    selected_candidates = [selected_candidates[bi]] + \
                        [c for i, c in enumerate(selected_candidates) if i != bi]
                    print(f"  No frame passed CLIP threshold; using best "
                          f"({best_unverified[1]:.3f}): frame "
                          f"{selected_candidates[0]['frame_idx']} (was {old_frame})")
                else:
                    print(f"  No frame passed CLIP threshold; keeping original "
                          f"(best CLIP={best_unverified[1]:.3f})")
            elif verified_idx is None:
                print(f"  GDino found nothing on any candidate; keeping original")

        best_candidate = selected_candidates[0]
        last_frame_idx = best_candidate['frame_idx']
        last_sim = best_candidate['similarity']
        last_meta_idx = best_candidate.get('meta_idx', None)

        # Find the meta_idx if not provided (lookup in frame_metadata)
        if last_meta_idx is None:
            for i, fm in enumerate(frame_metadata):
                if fm['frame_idx'] == last_frame_idx:
                    last_meta_idx = i
                    break

        print(f"\n  Best candidate -> frame {last_frame_idx}  "
              f"(t={last_frame_idx/fps:.2f}s, sim={last_sim:.3f})")

        # ========================================================================= #
        # STAGE 4: Temporal Sampling (extract ±context window for refinement)      #
        # ========================================================================= #
        context_seconds = self.config['text_query'].get('context_seconds', 0.5)
        half_span = int(context_seconds * fps / 2)
        print(f"\nLoading ±{context_seconds/2:.1f}s context window...")
        frames, frame_indices = self._load_frame_window(
            video_path, last_frame_idx, half_span
        )
        center_local = frame_indices.index(last_frame_idx) if last_frame_idx in frame_indices else 0

        # ========================================================================= #
        # STAGE 5: Spatial Localization (Grounding DINO or legacy REN+CLIP)       #
        # ========================================================================= #
        refiner = CandidateRefiner(self, self.config, text_query=text_query)
        refined_candidates = refiner.refine(selected_candidates, frames, frame_indices, text_feat)

        if refined_candidates:
            best_refined = refined_candidates[0]
            region_point = tuple(best_refined['region_point'])
            region_score = best_refined['region_score']
            bbox_from_refiner = best_refined['bbox']
        else:
            print("  [Warning] REN refinement failed, using CLIP-tile fallback")
            # Fallback: simple CLIP-tile bbox around best candidate
            if frames:
                fh, fw = frames[center_local].shape[:2]
                region_point = (fw // 2, fh // 2)  # frame center
                region_score = 0.0
                bbox_width = max(64, fw // 4)
                bbox_height = max(64, fh // 4)
                bbox_from_refiner = [max(0, fw//2 - bbox_width//2), max(0, fh//2 - bbox_height//2),
                                     bbox_width, bbox_height]
            else:
                raise RuntimeError("No frames loaded in temporal window")

        spatial_method = self.config.get('text_query', {}).get('spatial_method', 'grounding_dino')
        print(f"  Spatial location ({spatial_method}): {region_point}  score={region_score:.3f}")

        # ---- Stage 5b: bbox generation (OCR direct, CLIP-tile fast path, or SAM2) ----
        _skip_sam2 = self.config['text_query'].get('skip_sam2_eval', True)

        # Check for OCR brand match on the best refined candidate
        ocr_bbox_xywh = None
        last_ocr_score = float(ocr_scores[last_meta_idx]) if last_meta_idx < len(ocr_scores) else 0.0

        if is_brand and last_ocr_score >= 0.85 and HAS_EASYOCR:
            print(f"\nOCR score high ({last_ocr_score:.2f}) -- locating brand bbox...")
            # Try to get OCR bbox from the best frame
            try:
                frame_rgb = frames[center_local]
                ocr_bbox_result = self._get_ocr_bbox(frame_rgb, text_query)
                if ocr_bbox_result is not None:
                    region_point, ocr_bbox_xywh = ocr_bbox_result
                    print(f"  OCR bbox found: {ocr_bbox_xywh}")
            except Exception as e:
                print(f"  OCR bbox extraction failed: {e}")

        # Use OCR bbox if available, otherwise use REN-refined bbox
        if ocr_bbox_xywh is not None:
            bbox = torch.tensor(ocr_bbox_xywh)
            print(f"SAM2 skipped -- using OCR bbox: {bbox.tolist()}")
        else:
            bbox = torch.tensor(bbox_from_refiner, dtype=torch.int32)
            if _skip_sam2:
                print(f"SAM2 skipped (skip_sam2_eval=true) -- refined bbox: {bbox.tolist()}")
            else:
                print(f"SAM2 will refine REN bbox: {bbox.tolist()}")

        print(f"  Final Bbox: {bbox.tolist()}")

        # Save a debug image
        debug_frame = frames[center_local].copy()
        px, py = int(region_point[0]), int(region_point[1])
        bx, by, bw, bh = [int(v) for v in bbox]
        cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
        cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        debug_path = os.path.join(output_dir, 'debug_last_frame.jpg')
        cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        print(f"  Debug frame saved: {debug_path}")

        # ---- Stage 6: SAM2 tracking (skip for interactive mode - too slow) ----
        track = [{'frame_idx': last_frame_idx, 'bbox': [bx, by, bw, bh]}]
        print("  [Interactive mode] Skipping SAM2 tracking (too slow)")
        print(f"  Using single-frame bbox: {[bx, by, bw, bh]}")

        # ---- Step 8: export clip ----
        print("\nExporting result clip...")
        self.localizer.export_clip(
            video_path, track, last_frame_idx,
            fps, os.path.join(output_dir, 'last_occurrence.mp4'),
            context_seconds=context_seconds,
        )

        result = {
            'query': text_query,
            'sub_queries': sub_queries,
            'video_path': video_path,
            'last_frame_idx': last_frame_idx,
            'last_frame_timestamp': round(last_frame_idx / fps, 3),
            'pred_bbox': [int(bx), int(by), int(bw), int(bh)],
            'clip_similarity': round(float(clip_sims[last_meta_idx]), 4),
            'sub_query_scores': [round(float(s[last_meta_idx]), 4) for s in sub_sims_list],
            'ocr_score': round(float(ocr_scores[last_meta_idx]), 4),
            'fused_similarity': round(last_sim, 4),
            'ocr_weight': ocr_weight,
            'ocr_frames_hit': int(n_ocr_hits),
            'region_point': list(region_point),
            'region_clip_score': round(region_score, 4),
            'similarity_threshold': threshold,
            'valid_segments': num_valid_segments,
            'frames_above_threshold': int(n_above),
            'context_seconds': context_seconds,
            'fps': fps,
        }
        with open(os.path.join(output_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        return result

    # ------------------------------------------------------------------ #
    # OCR-guided bbox (query time)                                         #
    # ------------------------------------------------------------------ #

    def _get_ocr_bbox(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
    ) -> Tuple[Tuple[int, int], List[int]]:
        """
        Run EasyOCR on the detected frame, find the text region that best
        matches the query, and return its center point + expanded bbox.

        Used instead of SAM2 when OCR score is high — direct text detection
        is more precise than SAM2 mask selection for brand-name queries.

        Returns (region_point, bbox_xywh) or None if no match found.
        """
        if not HAS_EASYOCR or not HAS_RAPIDFUZZ:
            return None

        if self._ocr_reader is None:
            print("  Initializing EasyOCR for query-time bbox...")
            self._ocr_reader = _easyocr.Reader(
                ['en'], gpu=self.device.type == 'cuda', verbose=False
            )

        query_lower = text_query.lower().strip()
        detections = self._ocr_reader.readtext(frame_rgb, detail=1, paragraph=False)

        best_score, best_det = 0.0, None
        for (bbox_pts, text, conf) in detections:
            if conf < 0.3 or len(text.strip()) < 2:
                continue
            score = max(
                rfuzz.partial_ratio(query_lower, text.lower()),
                rfuzz.token_set_ratio(query_lower, text.lower()),
            ) / 100.0
            if score > best_score:
                best_score, best_det = score, (bbox_pts, text, conf)

        if best_det is None or best_score < 0.7:
            return None

        bbox_pts = best_det[0]
        xs = [int(p[0]) for p in bbox_pts]
        ys = [int(p[1]) for p in bbox_pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        h, w = frame_rgb.shape[:2]
        text_w = x2 - x1
        text_h = y2 - y1
        # Product packaging is always much larger than the text label on it.
        # Expand by 80% on each side, but enforce a minimum box of 3× the text size
        # so small/distant detections still produce a usable bbox.
        pad_x = max(int(text_w * 0.8), text_w)
        pad_y = max(int(text_h * 0.8), text_h)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        print(f"  OCR bbox: '{best_det[1]}' conf={best_det[2]:.2f} "
              f"match={best_score:.2f} -> [{x1},{y1},{x2-x1},{y2-y1}]")
        return (cx, cy), [x1, y1, x2 - x1, y2 - y1]

    # ------------------------------------------------------------------ #
    # Query type classifier                                                #
    # ------------------------------------------------------------------ #

    # Queries about large-area objects need a relaxed SAM2 mask size limit.
    _LARGE_OBJECT_WORDS = {
        'painting', 'picture', 'photo', 'photograph', 'poster', 'print',
        'mural', 'artwork', 'canvas', 'frame', 'mirror', 'screen',
        'monitor', 'tv', 'television', 'window', 'door', 'floor', 'ceiling',
        'board', 'whiteboard', 'chalkboard', 'wall', 'shelf', 'cabinet',
        'fridge', 'refrigerator', 'cupboard', 'wardrobe', 'mat', 'rug',
    }

    # Stopwords stripped before building the noun sub-query.
    _STOPWORDS = {
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'with', 'and',
        'or', 'for', 'by', 'as', 'is', 'are', 'was', 'that', 'this',
    }

    # Modifier words used for compositional decomposition.
    _COLOR_WORDS = {
        'red', 'green', 'blue', 'black', 'white', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'grey', 'gray', 'beige', 'silver', 'gold', 'golden',
        'transparent', 'clear', 'dark', 'light', 'bright',
    }
    _SIZE_WORDS = {
        'small', 'large', 'big', 'tiny', 'huge', 'tall', 'short', 'wide',
        'narrow', 'thin', 'thick',
    }
    _MATERIAL_WORDS = {
        'metal', 'metallic', 'plastic', 'wooden', 'wood', 'glass', 'ceramic',
        'cloth', 'rubber', 'steel', 'iron', 'paper', 'cardboard',
    }
    _MODIFIER_WORDS = (
        _COLOR_WORDS | _SIZE_WORDS | _MATERIAL_WORDS |
        {'open', 'closed', 'empty', 'full', 'hot', 'cold', 'wet', 'dry', 'clean', 'dirty'}
    )

    # Common everyday objects/actions that are never brand names.
    # Add any object word here that should NOT trigger OCR/brand detection.
    _COMMON_OBJECTS = {
        # articles / prepositions / conjunctions
        'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'with', 'and', 'or',
        # kitchen utensils & cutlery
        'knife', 'knives', 'fork', 'forks', 'spoon', 'spoons', 'spatula',
        'ladle', 'tongs', 'whisk', 'grater', 'peeler', 'opener', 'corkscrew',
        'colander', 'strainer', 'sieve', 'skimmer', 'masher', 'roller',
        # crockery & glassware
        'cup', 'cups', 'mug', 'mugs', 'plate', 'plates', 'bowl', 'bowls',
        'dish', 'dishes', 'glass', 'glasses', 'jug', 'pitcher', 'ramekin',
        # cookware & bakeware
        'pan', 'pans', 'pot', 'pots', 'wok', 'skillet', 'saucepan', 'casserole',
        'tray', 'baking', 'sheet', 'tin', 'rack',
        # containers & packaging
        'bottle', 'bottles', 'box', 'boxes', 'bag', 'bags', 'jar', 'jars',
        'can', 'cans', 'tube', 'packet', 'carton', 'container', 'tin', 'lid',
        'cap', 'wrapper', 'sachet', 'pouch',
        # kitchen appliances & fixtures
        'kettle', 'toaster', 'microwave', 'oven', 'hob', 'cooker', 'stove',
        'fridge', 'freezer', 'refrigerator', 'dishwasher', 'blender', 'mixer',
        'sink', 'tap', 'faucet', 'drain', 'counter', 'worktop', 'table',
        # cleaning & household
        'sponge', 'cloth', 'towel', 'rag', 'brush', 'mop', 'broom', 'duster',
        'soap', 'detergent', 'bleach', 'spray', 'bucket', 'dustbin', 'bin',
        'trash', 'rubbish', 'garbage', 'waste',
        # food & drink (generic)
        'food', 'water', 'milk', 'coffee', 'tea', 'juice', 'bread', 'rice',
        'pasta', 'egg', 'eggs', 'apple', 'onion', 'tomato', 'pepper', 'garlic',
        'salt', 'sugar', 'flour', 'oil', 'butter',
        # furniture & room features
        'shelf', 'shelves', 'drawer', 'drawer', 'door', 'window', 'wall',
        'floor', 'ceiling', 'cabinet', 'cupboard', 'hook', 'handle', 'knob',
        'switch', 'light', 'bulb', 'socket', 'plug', 'cord', 'cable', 'remote',
        'board', 'cutting', 'chopping',
        # storage & misc
        'key', 'keys', 'phone', 'book', 'books', 'pen', 'pens', 'scissors',
        'tape', 'charger', 'mat', 'rug', 'paper', 'napkin', 'foil', 'wrap',
        # people & body
        'hand', 'hands', 'finger', 'fingers', 'person', 'people', 'man',
        'woman', 'child',
        # colours (so "red switch" never triggers brand OCR)
        'red', 'green', 'blue', 'black', 'white', 'yellow', 'orange', 'purple',
        'pink', 'brown', 'grey', 'gray', 'silver', 'gold', 'dark', 'light',
        # sizes / descriptors
        'small', 'large', 'big', 'tiny', 'tall', 'short', 'open', 'closed',
        'hot', 'cold', 'wet', 'dry', 'empty', 'full', 'clean', 'dirty',
        'wooden', 'metal', 'plastic', 'glass', 'ceramic', 'stainless', 'steel',
    }

    def _is_brand_query(self, text_query: str) -> bool:
        """
        Decide whether the query refers to a brand/label (OCR useful) or a
        generic object (pure CLIP is better).

        Brand signals:
          - Contains a word not in the common-object vocabulary
          - AND that word starts with a capital letter (user typed it as a
            proper noun), OR the whole query has ≥2 non-common words
            (e.g. "Yorkshire Tea", "Cafe Royal", "Twinings Peppermint")

        Object signals:
          - All words are in the common-object vocabulary
          - OR the query is a single short common noun
        """
        words = text_query.split()
        non_common = [
            w for w in words
            if w.lower().strip('.,!?-') not in self._COMMON_OBJECTS
        ]

        if not non_common:
            return False  # every word is a known common object

        # Strong signal: user capitalized a non-common word (proper noun / brand)
        has_capitalized_brand = any(
            w[0].isupper() and w.lower() not in self._COMMON_OBJECTS
            for w in words
        )
        if has_capitalized_brand:
            return True

        # Moderate signal: multiple unknown words (e.g. "twinings peppermint")
        if len(non_common) >= 2:
            return True

        # Single unknown word typed as a standalone query → likely a brand name
        # ("fairy", "persil", "hovis", "heinz").  OCR will score 0 if no label
        # text matches, so false positives here silently fall back to CLIP.
        if len(words) == 1:
            return True

        return False

    def _decompose_query(self, text_query: str) -> List[str]:
        """
        Split a compositional query into independent sub-queries.

        "blue kettle"  → ["blue kettle", "kettle", "blue object"]
        "red cup"      → ["red cup", "cup", "red object"]
        "knife"        → ["knife"]   (single noun, no split)
        "Yorkshire Tea"→ ["Yorkshire Tea"]   (brand, no split)

        Returns a list; first element is always the original full query.
        """
        words = text_query.lower().split()
        if len(words) == 1 or self._is_brand_query(text_query):
            return [text_query]

        modifiers = [w for w in words if w in self._MODIFIER_WORDS]
        nouns = [w for w in words
                 if w not in self._MODIFIER_WORDS and w not in self._STOPWORDS]

        if not modifiers or not nouns:
            return [text_query]

        # Only add noun sub-query — attribute-only prompts ("blue object") score
        # poorly with CLIP in cluttered kitchen frames and shrink the combined
        # score below threshold.  The noun sub-query alone discriminates
        # "kettle" from "cutting board" which is the key failure case.
        return [text_query, ' '.join(nouns)]

    # ------------------------------------------------------------------ #
    # OCR brand/text fusion                                                #
    # ------------------------------------------------------------------ #

    def _compute_ocr_scores(
        self,
        text_query: str,
        frame_metadata: List[Dict],
        window: int = 2,
    ) -> np.ndarray:
        """
        For each indexed frame, compute how well its OCR-detected text matches
        the text query.  Aggregates OCR results over ±window neighboring sampled
        frames so that a blurry or partially occluded label is covered by a
        sharper nearby frame.

        Scoring uses rapidfuzz partial_ratio + token_set_ratio so that
        'Yorkshire Tea' matches 'yorksh...' (partial) and word-order variants.

        Returns (N,) float32 array in [0, 1].  All zeros if rapidfuzz is not
        installed or if the index was built without OCR.
        """
        n = len(frame_metadata)
        scores = np.zeros(n, dtype=np.float32)

        if not HAS_RAPIDFUZZ:
            return scores

        # Check if any frame has OCR data at all
        has_ocr = any('ocr_texts' in m and m['ocr_texts'] for m in frame_metadata)
        if not has_ocr:
            return scores

        query_lower = text_query.lower().strip()

        for i in range(n):
            # Aggregate text from ±window sampled frames
            window_texts: List[str] = []
            for j in range(max(0, i - window), min(n, i + window + 1)):
                for item in frame_metadata[j].get('ocr_texts', []):
                    if item.get('conf', 0) >= 0.3:
                        window_texts.append(item['text'])

            if not window_texts:
                continue

            combined = ' '.join(window_texts)
            # partial_ratio: best substring match (handles cropped/occluded text)
            # token_set_ratio: order-independent token match (handles word rearrangement)
            score = max(
                rfuzz.partial_ratio(query_lower, combined),
                rfuzz.token_set_ratio(query_lower, combined),
            ) / 100.0
            scores[i] = score

        return scores

    # ------------------------------------------------------------------ #
    # Adaptive threshold                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adaptive_threshold(sims: np.ndarray, alpha: float = 1.0,
                            min_tau: float = 0.10, max_tau: float = 0.30) -> float:
        """Compute query-conditioned threshold from the similarity distribution."""
        mu = float(sims.mean())
        sigma = float(sims.std())
        return float(np.clip(mu + alpha * sigma, min_tau, max_tau))

    # ------------------------------------------------------------------ #
    # Patch-level re-ranking                                               #
    # ------------------------------------------------------------------ #

    def _patch_rerank(self, text_feat: np.ndarray, clip_sims: np.ndarray) -> np.ndarray:
        """
        Re-rank frame similarities using max-patch scoring.

        For each frame, compute max_i(cos(text, patch_i)) — the strongest
        local patch signal — and blend it with the CLS score.

        Args:
            text_feat: (1, D) L2-normalized text embedding
            clip_sims: (N,) CLS-based similarities for all frames

        Returns:
            (N,) re-ranked similarity scores
        """
        if self.patch_embeddings is None:
            return clip_sims

        top_k = self.config.get('text_query', {}).get('faiss', {}).get('patch_top_k', 100)
        top_indices = np.argsort(clip_sims)[::-1][:top_k]

        reranked = clip_sims.copy()
        for idx in top_indices:
            if idx >= len(self.patch_embeddings):
                continue
            patches = self.patch_embeddings[idx]       # (256, 1024)
            patch_sims = patches @ text_feat.T         # (256, 1)
            max_patch = float(patch_sims.max())
            # Blend: 40% CLS + 60% max-patch (patch is more discriminative for small objects)
            reranked[idx] = 0.4 * clip_sims[idx] + 0.6 * max_patch

        return reranked

    # ------------------------------------------------------------------ #
    # CLIP crop scoring                                                    #
    # ------------------------------------------------------------------ #

    def _find_best_region_clip(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        grid_size: int = 6,
        fast_mode: bool = False,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Score crops at multiple scales with 50% overlap so objects near tile
        boundaries and corners are fully captured in at least one crop.

        Crops generated:
          - Full frame (global context)
          - 4 quadrant halves (if not fast_mode)
          - grid_size×grid_size tiles at 50% stride (if not fast_mode, skipped if fast_mode)

        fast_mode: if True, only score the full frame + 2×2 quadrants (≈5 crops vs 50+)
        """
        h, w = frame_rgb.shape[:2]

        crops, centers = [], []

        # Full frame
        crops.append(self.localizer.clip_preprocess(Image.fromarray(frame_rgb)))
        centers.append((w // 2, h // 2))

        # Fast mode: only 2×2 quadrants, skip fine grid
        scales = (2,) if fast_mode else (2, 3)

        # Quadrant halves (2×2 with 50% overlap = 3×3 positions)
        for scale_div in scales:
            ph, pw = h // scale_div, w // scale_div
            stride_h, stride_w = ph // 2, pw // 2
            y1 = 0
            while y1 + ph <= h:
                x1 = 0
                while x1 + pw <= w:
                    y2, x2 = min(h, y1 + ph), min(w, x1 + pw)
                    crops.append(self.localizer.clip_preprocess(
                        Image.fromarray(frame_rgb[y1:y2, x1:x2])
                    ))
                    centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                    x1 += stride_w
                y1 += stride_h

        # Fine grid at 50% overlap (skip in fast_mode)
        if not fast_mode:
            ph, pw = h // grid_size, w // grid_size
            stride_h, stride_w = max(1, ph // 2), max(1, pw // 2)
            y1 = 0
            while y1 + ph <= h:
                x1 = 0
                while x1 + pw <= w:
                    y2, x2 = min(h, y1 + ph), min(w, x1 + pw)
                    crops.append(self.localizer.clip_preprocess(
                        Image.fromarray(frame_rgb[y1:y2, x1:x2])
                    ))
                    centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                    x1 += stride_w
                y1 += stride_h

        crop_batch = torch.stack(crops).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            crop_feats = self.localizer.clip_model.encode_image(crop_batch).float()
        crop_feats = F.normalize(crop_feats, p=2, dim=-1)
        scores = (crop_feats @ text_feat.T).squeeze(-1).cpu().numpy()
        best = int(np.argmax(scores))
        return centers[best], float(scores[best])

    # ------------------------------------------------------------------ #
    # REN-guided spatial localization                                      #
    # ------------------------------------------------------------------ #

    def _ren_guided_localize(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        stride: int = 4,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Use REN's 32×32 semantic grid as region proposals, score each crop with
        CLIP, and return the (x, y) center of the highest-scoring region.

        This bridges RELOCATE's region-based localization to text queries without
        any cross-space adapter: REN provides object-aware spatial proposals (its
        grid_points) and CLIP scores them in a text-compatible embedding space.

        stride: take every Nth grid point (1024 total → stride=4 gives 256 proposals)
        """
        h, w = frame_rgb.shape[:2]
        img_res = self.config['ren']['parameters']['image_resolution']

        # Accessing .ren triggers the lazy REN load (DINOv2 ViT-L/14).
        # grid_points: (grid_size², 2) in 518×518 normalised coords (y, x).
        grid_points = self.localizer.ren.grid_points.cpu().numpy()
        scale_y = h / float(img_res)
        scale_x = w / float(img_res)

        patch_r = max(32, min(h, w) // 8)
        crops: List = []
        centers: List[Tuple[int, int]] = []

        for py_norm, px_norm in grid_points[::stride]:
            py = int(py_norm * scale_y)
            px = int(px_norm * scale_x)
            y1 = max(0, py - patch_r)
            y2 = min(h, py + patch_r)
            x1 = max(0, px - patch_r)
            x2 = min(w, px + patch_r)
            patch = frame_rgb[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            crops.append(self.localizer.clip_preprocess(Image.fromarray(patch)))
            centers.append((px, py))

        if not crops:
            return (w // 2, h // 2), 0.0

        crop_batch = torch.stack(crops).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            feats = self.localizer.clip_model.encode_image(crop_batch).float()
        feats = F.normalize(feats, p=2, dim=-1)
        scores = (feats @ text_feat.T).squeeze(-1).cpu().numpy()
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
    parser.add_argument('--region-grid', type=int, default=3, dest='region_grid',
                        help='N×N grid for CLIP crop scoring (default: 3 for speed, use 6 for max accuracy)')
    parser.add_argument('--ocr-weight', type=float, default=None, dest='ocr_weight',
                        help='Weight for OCR brand match score (default: 0.3). '
                             'Set 0 to disable OCR fusion.')
    parser.add_argument('--neg', type=str, default=None,
                        help='Comma-separated negative prompts to suppress confounders')
    parser.add_argument('--neg-weight', type=float, default=None, dest='neg_weight',
                        help='Weight for negative prompt suppression (default: from config)')

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

    neg_queries = [q.strip() for q in args.neg.split(',')] if args.neg else []

    engine = IndexedQueryEngine(config, args.index)
    result = engine.query(
        args.query, args.video, output_dir,
        threshold=args.threshold,
        region_grid=args.region_grid,
        ocr_weight=args.ocr_weight,
        neg_queries=neg_queries,
        neg_weight=args.neg_weight,
    )

    print("\n=== Query Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
