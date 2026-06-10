"""
Textual-REN v2: Online query against indexed video (no OCR).

Pipeline
--------
1. Load FAISS index + clip_embeddings.npy + patch_embeddings.npy
2. Encode text query with CLIP → same joint space as indexed image embeddings
3. Full scan: cosine similarity of text vs every indexed frame
4. Patch re-ranking: max-patch similarity blended with CLS score
5. Adaptive threshold: per-query τ = mean + α·std
6. Temporal segmentation: group above-threshold frames into contiguous segments
7. GDino+CLIP verified frame selection: verify each segment peak with
   Grounding DINO detection + CLIP crop scoring
8. Grounding DINO spatial localization: text + image → bbox directly
9. Export result clip with bbox overlay

Usage:
    python query_indexed.py "coffee mug" --index <index_dir> --video <video_path> --output <output_dir>
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '65536'

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
from grounding_dino import GroundingDINOLocalizer


# ====================================================================== #
# STAGE 3: SELECTION POLICY                                              #
# ====================================================================== #

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
        above_threshold = {i for i, sim in enumerate(all_sims) if sim >= threshold}
        if not above_threshold:
            return [], []

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

        valid_segments = [s for s in segments if len(s) >= 2]
        return valid_segments, sorted_indices

    def select(self, all_sims, frame_metadata, fps, sample_rate, threshold=0.18):
        """
        RELOCATE Stage 3: Temporal segmentation + deterministic/probabilistic selection.

        Returns: (candidates, num_valid_segments)
        """
        valid_segments, _ = self.temporal_segmentation(all_sims, frame_metadata, fps, sample_rate, threshold)
        num_valid = len(valid_segments)

        if not valid_segments:
            best_idx = np.argmax(all_sims)
            return [{'frame_idx': frame_metadata[best_idx]['frame_idx'], 'similarity': float(all_sims[best_idx])}], 0

        candidates = []
        for segment in valid_segments:
            peak_idx = max(segment, key=lambda i: all_sims[i])
            candidates.append({
                'frame_idx': frame_metadata[peak_idx]['frame_idx'],
                'meta_idx': peak_idx,
                'similarity': float(all_sims[peak_idx]),
                'segment': segment
            })

        candidates = sorted(candidates, key=lambda x: -x['meta_idx'])  # Recent first

        if self.policy_name == 'last':
            result = candidates
        elif self.policy_name == 'strongest':
            best = max(candidates, key=lambda x: x['similarity'])
            result = [best]
        elif self.policy_name == 'topk':
            sorted_cands = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            result = sorted_cands[:self.top_k]
        else:
            result = candidates

        return result, num_valid


# ====================================================================== #
# STAGE 5: CANDIDATE REFINER                                            #
# ====================================================================== #

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
        # Use expanded query for GDino detection (e.g. "pan" → "frying pan. cooking pan.")
        det_query = self.query_engine._detection_query(self.text_query)
        result = self.query_engine.grounding_dino.best_box(
            frame_rgb, det_query,
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

            if method == 'grounding_dino':
                region_point, region_score, gdino_bbox = self._localize_grounding_dino(frame_rgb, text_feat, h, w)
            else:
                region_point, region_score, gdino_bbox = self._localize_ren_clip(frame_rgb, text_feat, h, w)

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


# ====================================================================== #
# MAIN QUERY ENGINE                                                      #
# ====================================================================== #

class IndexedQueryEngine:
    """Query against a FAISS-indexed video using CLIP text-image retrieval."""

    def __init__(self, config: Dict, index_dir: str):
        self.config = config
        self.localizer = TextQueryLocalizer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_gpu = HAS_GPU
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        self._grounding_dino = None

        self.index_dir = index_dir
        self._load_index()

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

        self.clip_embeddings = np.load(embeddings_path)

        patch_path = os.path.join(self.index_dir, 'patch_embeddings.npy')
        use_patches = self.config.get('text_query', {}).get('faiss', {}).get('use_patch_rerank', True)
        if use_patches and os.path.exists(patch_path):
            self.patch_embeddings = np.load(patch_path)
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
        neg_queries: List[str] = None,
        neg_weight: float = None,
        # Ablation flags
        ablation_no_verification: bool = False,
        ablation_use_strongest: bool = False,
    ) -> Dict:
        """
        Find the last genuine occurrence of text_query in the indexed video.

        Pipeline: CLIP retrieval → patch re-ranking → adaptive threshold →
        temporal segmentation → GDino+CLIP verification → GDino localization.
        """
        if threshold is None:
            threshold = self.config['text_query'].get('similarity_threshold', 0.20)
        if neg_weight is None:
            neg_weight = self.config.get('text_query', {}).get('neg_weight', 0.0)
        if neg_queries is None:
            neg_queries = []

        os.makedirs(output_dir, exist_ok=True)
        fps = self.metadata['fps']
        frame_metadata = self.metadata['frame_metadata']
        sample_rate = self.metadata.get('sample_rate', 10)

        # ---- Stage 1: encode text query ----
        print(f"\nQuery: '{text_query}'")
        text_feat = self.localizer.encode_text(text_query)
        text_np = text_feat.cpu().numpy().astype(np.float32)
        print(f"  Embedding dim: {text_feat.shape[-1]}  |  index dim: {self.faiss_index.d}")

        if text_feat.shape[-1] != self.faiss_index.d:
            raise RuntimeError(
                f"Dimension mismatch: text={text_feat.shape[-1]}, "
                f"index={self.faiss_index.d}. Delete the index and rebuild."
            )

        # ---- Stage 2: CLIP similarity scoring ----
        clip_sims = (self.clip_embeddings @ text_np.T).squeeze()

        # ---- Stage 2a: optional negative query suppression ----
        if neg_queries and neg_weight > 0.0:
            neg_text = self.localizer.encode_text(', '.join(neg_queries))
            neg_np = neg_text.cpu().numpy().astype(np.float32)
            neg_sims = (self.clip_embeddings @ neg_np.T).squeeze()
            clip_sims = clip_sims - (neg_weight * neg_sims)
            print(f"  Negative prompt: {neg_queries} (weight={neg_weight})")

        # ---- Stage 2c: patch-level re-ranking ----
        if self.patch_embeddings is not None:
            clip_sims = self._patch_rerank(text_np, clip_sims)
            print(f"  Patch re-ranking: applied (top-{self.config.get('text_query', {}).get('faiss', {}).get('patch_top_k', 100)} frames)")

        all_sims = clip_sims

        # ---- Stage 2d: adaptive threshold ----
        use_adaptive = self.config.get('text_query', {}).get('adaptive_threshold', False)
        if use_adaptive:
            alpha = self.config.get('text_query', {}).get('threshold_alpha', 1.0)
            adaptive_tau = self._adaptive_threshold(all_sims, alpha=alpha)
            print(f"  Adaptive threshold: tau={adaptive_tau:.4f} (alpha={alpha}, "
                  f"mean={all_sims.mean():.4f}, std={all_sims.std():.4f}, "
                  f"fixed was {threshold})")
            threshold = adaptive_tau

        n_above = int((all_sims >= threshold).sum())
        print(f"\n  Scores -- max={all_sims.max():.4f}  "
              f"mean={all_sims.mean():.4f}  "
              f"above {threshold}: {n_above}/{len(all_sims)}")

        if n_above == 0:
            print(f"  [Hint] Try --threshold {max(0.05, float(all_sims.max()) - 0.02):.2f}")
            raise RuntimeError(
                f"No frames found above similarity threshold {threshold}"
            )

        # ---- Stage 3: Selection Policy (temporal segmentation) ----
        selection_policy = SelectionPolicy(self.config)
        selected_candidates, num_valid_segments = selection_policy.select(
            all_sims, frame_metadata, fps, sample_rate, threshold=threshold
        )

        print(f"  {num_valid_segments} segment(s) found, {len(selected_candidates)} candidate(s) selected (policy={selection_policy.policy_name})")

        # ---- Stage 3b: GDino+CLIP verified frame selection ----
        # Improvement: Score ALL candidates and pick the best, rather than
        # stopping at the first above-threshold frame.  This avoids accepting
        # a borderline detection (plate→pan confusion at 0.19) when a much
        # stronger match exists on a different segment peak.
        spatial_method = self.config.get('text_query', {}).get('spatial_method', 'grounding_dino')
        if spatial_method == 'grounding_dino' and len(selected_candidates) > 1:
            gdino_cfg = self.config.get('grounding_dino', {})
            box_thresh = gdino_cfg.get('box_threshold', 0.20)
            text_thresh = gdino_cfg.get('text_threshold', 0.20)
            min_clip_crop = self.config.get('text_query', {}).get('min_crop_verify', 0.17)
            max_verify = min(len(selected_candidates), 12)
            det_query = self._detection_query(text_query)

            # Pre-encode confusable class texts for negative suppression
            conf_names, conf_feats = self._encode_confusable_texts(text_query)

            print(f"\n  Verifying ALL {max_verify} candidates with GDino+CLIP "
                  f"(min_crop={min_clip_crop})...")
            if det_query != text_query:
                print(f"  Detection query expanded: '{text_query}' -> '{det_query}'")
            if conf_names:
                print(f"  Confusable classes: {conf_names}")

            # Score every candidate — don't stop early
            scored_candidates = []   # (ci, clip_crop_score, patch_verify_score, combined)
            already_checked = set()

            def _verify_range(start, end):
                """Verify candidates in [start, end) range."""
                for ci in range(start, min(end, len(selected_candidates))):
                    if ci in already_checked:
                        continue
                    already_checked.add(ci)
                    cand = selected_candidates[ci]
                    fidx = cand['frame_idx']
                    meta_idx = cand.get('meta_idx', None)
                    try:
                        probe = self._load_single_frame(video_path, fidx)
                    except Exception:
                        continue
                    result = self.grounding_dino.best_box(
                        probe, det_query,
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

                    # Negative-class suppression: check if the crop matches
                    # a confusable object better than the target query
                    confused = False
                    if conf_feats is not None:
                        bx, by, bw, bh = bbox
                        crop = probe[by:by+bh, bx:bx+bw]
                        is_conf, conf_score, conf_name = self._confusable_check(
                            crop, clip_score, conf_names, conf_feats
                        )
                        if is_conf:
                            confused = True
                            print(f"    candidate {ci}: frame {fidx} "
                                  f"(t={fidx/fps:.1f}s) -- CLIP crop={clip_score:.3f} "
                                  f"BUT '{conf_name}'={conf_score:.3f} > target "
                                  f"-> CONFUSED")

                    # Patch spatial verification
                    patch_score = self._patch_spatial_verify(
                        meta_idx, bbox, text_np, probe.shape
                    ) if meta_idx is not None else clip_score

                    # Combined score — heavy penalty if confused
                    if confused:
                        combined = (0.6 * clip_score + 0.4 * patch_score) * 0.3
                    else:
                        combined = 0.6 * clip_score + 0.4 * patch_score

                    if not confused:
                        tag = "PASS" if clip_score >= min_clip_crop else "FAIL"
                        print(f"    candidate {ci}: frame {fidx} "
                              f"(t={fidx/fps:.1f}s) -- CLIP crop={clip_score:.3f}  "
                              f"patch={patch_score:.3f}  combined={combined:.3f} {tag}")
                    scored_candidates.append((ci, clip_score, patch_score, combined))

            # First pass: check top-N candidates
            _verify_range(0, max_verify)

            # Auto-expand: if ALL candidates so far are confused, try more
            passed = [(ci, cs, ps, cb) for ci, cs, ps, cb in scored_candidates
                      if cs >= min_clip_crop and cb > 0.10]
            if not passed and conf_feats is not None and max_verify < len(selected_candidates):
                expand_to = min(len(selected_candidates), max_verify + 8)
                print(f"  All {max_verify} candidates confused -- expanding to {expand_to}...")
                _verify_range(max_verify, expand_to)
                passed = [(ci, cs, ps, cb) for ci, cs, ps, cb in scored_candidates
                          if cs >= min_clip_crop and cb > 0.10]

            # Pick the candidate with highest combined score
            if scored_candidates:
                if passed:
                    best_ci, best_cs, best_ps, best_cb = max(passed, key=lambda x: x[3])
                    if best_ci != 0:
                        old_frame = selected_candidates[0]['frame_idx']
                        selected_candidates = [selected_candidates[best_ci]] + \
                            [c for i, c in enumerate(selected_candidates) if i != best_ci]
                        print(f"  Best verified: frame {selected_candidates[0]['frame_idx']} "
                              f"(combined={best_cb:.3f}, was {old_frame})")
                    else:
                        print(f"  Best verified: frame {selected_candidates[0]['frame_idx']} "
                              f"(combined={best_cb:.3f}, already first)")
                else:
                    # No candidate passed — use highest combined anyway
                    best_ci, best_cs, best_ps, best_cb = max(scored_candidates, key=lambda x: x[3])
                    if best_ci != 0:
                        old_frame = selected_candidates[0]['frame_idx']
                        selected_candidates = [selected_candidates[best_ci]] + \
                            [c for i, c in enumerate(selected_candidates) if i != best_ci]
                        print(f"  No clean candidate; best combined: "
                              f"frame {selected_candidates[0]['frame_idx']} "
                              f"({best_cb:.3f}, was {old_frame})")
                    else:
                        print(f"  No clean candidate; keeping first "
                              f"(combined={best_cb:.3f})")
            else:
                print(f"  GDino found nothing on any candidate; keeping original")

        # Track whether the final result is a confident or confused detection
        is_confused = False
        if spatial_method == 'grounding_dino' and len(selected_candidates) > 1:
            # Check if the winning candidate was from the "no clean candidate" fallback
            if scored_candidates:
                passed_final = [(ci, cs, ps, cb) for ci, cs, ps, cb in scored_candidates
                                if cs >= min_clip_crop and cb > 0.10]
                if not passed_final:
                    is_confused = True

        best_candidate = selected_candidates[0]
        last_frame_idx = best_candidate['frame_idx']
        last_sim = best_candidate['similarity']
        last_meta_idx = best_candidate.get('meta_idx', None)

        if last_meta_idx is None:
            for i, fm in enumerate(frame_metadata):
                if fm['frame_idx'] == last_frame_idx:
                    last_meta_idx = i
                    break

        print(f"\n  Best candidate -> frame {last_frame_idx}  "
              f"(t={last_frame_idx/fps:.2f}s, sim={last_sim:.3f})")

        # ---- Stage 4: Temporal Sampling ----
        context_seconds = self.config['text_query'].get('context_seconds', 0.5)
        half_span = int(context_seconds * fps / 2)
        print(f"\nLoading ±{context_seconds/2:.1f}s context window...")
        frames, frame_indices = self._load_frame_window(
            video_path, last_frame_idx, half_span
        )
        center_local = frame_indices.index(last_frame_idx) if last_frame_idx in frame_indices else 0

        # ---- Stage 5: Spatial Localization (Grounding DINO) ----
        refiner = CandidateRefiner(self, self.config, text_query=text_query)
        refined_candidates = refiner.refine(selected_candidates, frames, frame_indices, text_feat)

        if refined_candidates:
            best_refined = refined_candidates[0]
            region_point = tuple(best_refined['region_point'])
            region_score = best_refined['region_score']
            bbox_from_refiner = best_refined['bbox']
        else:
            print("  [Warning] Refinement failed, using center fallback")
            if frames:
                fh, fw = frames[center_local].shape[:2]
                region_point = (fw // 2, fh // 2)
                region_score = 0.0
                bbox_width = max(64, fw // 4)
                bbox_height = max(64, fh // 4)
                bbox_from_refiner = [max(0, fw//2 - bbox_width//2), max(0, fh//2 - bbox_height//2),
                                     bbox_width, bbox_height]
            else:
                raise RuntimeError("No frames loaded in temporal window")

        spatial_method = self.config.get('text_query', {}).get('spatial_method', 'grounding_dino')
        print(f"  Spatial location ({spatial_method}): {region_point}  score={region_score:.3f}")

        # ---- Bbox generation ----
        bbox = torch.tensor(bbox_from_refiner, dtype=torch.int32)
        _skip_sam2 = self.config['text_query'].get('skip_sam2_eval', True)
        if _skip_sam2:
            print(f"  Using Grounding DINO bbox directly: {bbox.tolist()}")
        else:
            print(f"  SAM2 will refine bbox: {bbox.tolist()}")

        bx, by, bw, bh = [int(v) for v in bbox]
        print(f"  Final Bbox: {[bx, by, bw, bh]}")
        if is_confused:
            print(f"  ** CONFUSED: all candidates matched confusable objects better "
                  f"than '{text_query}' -- bbox is best guess but likely wrong **")

        # Save debug image
        debug_frame = frames[center_local].copy()
        px, py = int(region_point[0]), int(region_point[1])
        if is_confused:
            # Red bbox + CONFUSED label for uncertain results
            bbox_color = (255, 0, 0)   # Red in RGB
            cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), bbox_color, 2)
            # Add CONFUSED label above bbox
            label = f"CONFUSED: '{text_query}' (best guess)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 10, by - 10)
            # Background rectangle for text readability
            cv2.rectangle(debug_frame, (bx, label_y - th - 6),
                          (bx + tw + 6, label_y + 4), (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (bx + 3, label_y - 2),
                        font, font_scale, (255, 80, 80), thickness)
        else:
            # Green bbox for confident results
            cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        debug_path = os.path.join(output_dir, 'debug_last_frame.jpg')
        cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        print(f"  Debug frame saved: {debug_path}")

        # ---- Export clip ----
        track = [{'frame_idx': last_frame_idx, 'bbox': [bx, by, bw, bh]}]
        print("\nExporting result clip...")
        self.localizer.export_clip(
            video_path, track, last_frame_idx,
            fps, os.path.join(output_dir, 'last_occurrence.mp4'),
            context_seconds=context_seconds,
        )

        result = {
            'query': text_query,
            'video_path': video_path,
            'last_frame_idx': last_frame_idx,
            'last_frame_timestamp': round(last_frame_idx / fps, 3),
            'pred_bbox': [bx, by, bw, bh],
            'confused': is_confused,
            'clip_similarity': round(float(clip_sims[last_meta_idx]), 4),
            'fused_similarity': round(last_sim, 4),
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
    # Detection query expansion                                            #
    # ------------------------------------------------------------------ #

    # Ambiguous single-word queries that confuse GDino with visually similar
    # objects.  Expanding to multi-label GDino prompts ("frying pan. cooking
    # pan.") produces more diverse detections for CLIP to re-rank.
    _QUERY_EXPANSIONS = {
        'pan':   'frying pan. cooking pan. saucepan',
        'pot':   'cooking pot. saucepan',
        'board': 'cutting board. chopping board',
        'cup':   'coffee cup. drinking cup. tea cup',
        'glass': 'drinking glass. wine glass',
        'plate': 'dinner plate. serving plate',
        'bowl':  'mixing bowl. cereal bowl',
        'knife': 'kitchen knife. chopping knife',
        'spoon': 'cooking spoon. tablespoon',
        'lid':   'pot lid. pan lid. container lid',
        'jar':   'glass jar. storage jar',
        'box':   'cardboard box. storage box',
        'bag':   'plastic bag. shopping bag',
        'tin':   'tin can. baking tin',
    }

    def _detection_query(self, text_query: str) -> str:
        """
        Expand ambiguous short queries for Grounding DINO detection.

        Single-word queries like "pan" are ambiguous — GDino may detect
        plates, lids, or other round objects.  Expanding to "frying pan.
        cooking pan." gives GDino more specific text conditioning and
        produces more diverse candidates for CLIP to re-rank.

        Multi-word queries (e.g., "cutting board") are already specific
        and pass through unchanged.
        """
        q = text_query.lower().strip()
        if ' ' not in q and q in self._QUERY_EXPANSIONS:
            return self._QUERY_EXPANSIONS[q]
        return text_query

    # ------------------------------------------------------------------ #
    # Negative-class (confusable) suppression                              #
    # ------------------------------------------------------------------ #

    # For each query, list objects that look similar and confuse GDino.
    # If a detection crop scores higher for a confusable than for the
    # target query in CLIP space, the detection is likely a false positive.
    _CONFUSABLE_CLASSES = {
        'pan':           ['plate', 'lid', 'tray', 'bowl'],
        'pot':           ['vase', 'jar', 'mug', 'bucket'],
        'plate':         ['lid', 'tray', 'pan'],
        'bowl':          ['cup', 'mug', 'pot'],
        'cup':           ['mug', 'jar', 'glass'],
        'glass':         ['jar', 'bottle', 'cup'],
        'lid':           ['plate', 'tray', 'pan bottom'],
        'board':         ['shelf', 'counter', 'table', 'tray'],
        'cutting board': ['counter', 'shelf', 'table', 'wooden surface'],
        'sponge':        ['cloth', 'towel', 'rag', 'scrubber'],
        'knife':         ['fork', 'spoon', 'spatula'],
        'spoon':         ['fork', 'knife', 'spatula'],
        'kettle':        ['water filter pitcher', 'jug', 'coffee maker'],
    }

    def _encode_confusable_texts(self, text_query: str):
        """
        Pre-encode confusable class texts for negative suppression.

        Returns (confusable_names, confusable_feats) or (None, None).
        confusable_feats: (C, 1024) tensor of L2-normalized text features.
        """
        q = text_query.lower().strip()
        confusables = self._CONFUSABLE_CLASSES.get(q, [])
        if not confusables:
            return None, None

        import open_clip
        tokens = open_clip.tokenize(confusables).to(self.device)
        with torch.no_grad():
            feats = self.localizer.clip_model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
        return confusables, feats

    def _confusable_check(
        self,
        crop_rgb: np.ndarray,
        target_clip_score: float,
        confusable_names: list,
        confusable_feats: torch.Tensor,
    ) -> Tuple[bool, float, str]:
        """
        Check if a detection crop is more similar to a confusable class
        than to the target query.

        Returns (is_confused, max_confusable_score, worst_confusable_name).
        is_confused=True means the crop matched a confusable better.
        """
        if confusable_feats is None or crop_rgb.size == 0:
            return False, 0.0, ""

        crop_t = self.localizer.clip_preprocess(
            Image.fromarray(crop_rgb)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == 'cuda'
        ):
            crop_feat = self.localizer.clip_model.encode_image(crop_t).float()
        crop_feat = F.normalize(crop_feat, dim=-1)

        sims = (crop_feat @ confusable_feats.T).squeeze(0).cpu()
        max_idx = int(sims.argmax())
        max_conf = float(sims[max_idx])
        worst_name = confusable_names[max_idx]

        # Confused if the best confusable score exceeds the target score
        is_confused = max_conf > target_clip_score
        return is_confused, max_conf, worst_name

    # ------------------------------------------------------------------ #
    # Patch-level spatial verification                                     #
    # ------------------------------------------------------------------ #

    def _patch_spatial_verify(
        self,
        meta_idx: int,
        bbox: list,
        text_np: np.ndarray,
        frame_shape: tuple,
    ) -> float:
        """
        Verify a detection bbox using pre-indexed CLIP patch tokens.

        The 256 patch tokens form a 16×16 grid over the image.  We find
        which patches overlap with the bbox and compute their max similarity
        to the text query.  If the detection is correct, patches at that
        location should strongly match the query.

        Returns max patch similarity at the bbox location (0.0 if no patches).
        """
        if self.patch_embeddings is None or meta_idx is None:
            return 0.0
        if meta_idx >= len(self.patch_embeddings):
            return 0.0

        patches = self.patch_embeddings[meta_idx]  # (256, 1024)
        h, w = frame_shape[:2]

        # CLIP ViT-g-14: 16×16 patch grid
        grid_h, grid_w = 16, 16

        bx, by, bw, bh = bbox
        # Map bbox to patch grid coordinates
        px1 = max(0, int(bx / w * grid_w))
        py1 = max(0, int(by / h * grid_h))
        px2 = min(grid_w, int((bx + bw) / w * grid_w) + 1)
        py2 = min(grid_h, int((by + bh) / h * grid_h) + 1)

        # Collect patches inside the bbox
        bbox_indices = []
        for py in range(py1, py2):
            for px in range(px1, px2):
                idx = py * grid_w + px
                if idx < len(patches):
                    bbox_indices.append(idx)

        if not bbox_indices:
            return 0.0

        bbox_patches = patches[bbox_indices]         # (K, 1024)
        sims = (bbox_patches @ text_np.T).squeeze()  # (K,)
        return float(sims.max()) if sims.size > 0 else 0.0

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
        """
        if self.patch_embeddings is None:
            return clip_sims

        top_k = self.config.get('text_query', {}).get('faiss', {}).get('patch_top_k', 100)
        top_indices = np.argsort(clip_sims)[::-1][:top_k]

        reranked = clip_sims.copy()
        for idx in top_indices:
            if idx >= len(self.patch_embeddings):
                continue
            patches = self.patch_embeddings[idx]
            patch_sims = patches @ text_feat.T
            max_patch = float(patch_sims.max())
            reranked[idx] = 0.4 * clip_sims[idx] + 0.6 * max_patch

        return reranked

    # ------------------------------------------------------------------ #
    # CLIP crop scoring (legacy utility)                                   #
    # ------------------------------------------------------------------ #

    def _find_best_region_clip(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        grid_size: int = 6,
    ) -> Tuple[Tuple[int, int], float]:
        """Score crops at multiple scales with 50% overlap."""
        h, w = frame_rgb.shape[:2]
        crops, centers = [], []

        crops.append(self.localizer.clip_preprocess(Image.fromarray(frame_rgb)))
        centers.append((w // 2, h // 2))

        for scale_div in (2, 3):
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
    # REN-guided spatial localization (legacy fallback)                     #
    # ------------------------------------------------------------------ #

    def _ren_guided_localize(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        stride: int = 4,
    ) -> Tuple[Tuple[int, int], float]:
        """Legacy: Use REN's 32x32 semantic grid as proposals, score with CLIP."""
        h, w = frame_rgb.shape[:2]
        img_res = self.config['ren']['parameters']['image_resolution']
        grid_points = self.localizer.ren.grid_points.cpu().numpy()
        scale_y = h / float(img_res)
        scale_x = w / float(img_res)

        patch_r = max(32, min(h, w) // 8)
        crops, centers = [], []

        for py_norm, px_norm in grid_points[::stride]:
            py = int(py_norm * scale_y)
            px = int(px_norm * scale_x)
            y1, y2 = max(0, py - patch_r), min(h, py + patch_r)
            x1, x2 = max(0, px - patch_r), min(w, px + patch_r)
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
        """Load every frame in [center_idx-half_span, center_idx+half_span]."""
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
        description='Textual-REN v2: Query indexed video for text-described objects.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('query', type=str, help='Text query (e.g., "fork")')
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str)
    parser.add_argument('--threshold', type=float, default=None,
                        help='CLIP cosine similarity threshold (default: from config)')
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
        neg_queries=neg_queries,
        neg_weight=args.neg_weight,
    )

    print("\n=== Query Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
