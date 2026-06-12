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
import time
import hashlib
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

from evidence_fusion import EvidenceVector, PresenceModel

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
from query_parser import QueryParser, QueryPlan


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

        # Query parser: domain-agnostic query understanding
        # (ontology = warm cache of pre-verified plans; LLM covers the rest)
        tq_cfg = config.get('text_query', {})
        self.query_parser = QueryParser(
            mode=tq_cfg.get('query_parser', 'rule_based'),
            llm_backend=tq_cfg.get('llm_backend', 'transformers'),
            llm_model=tq_cfg.get('llm_model', 'Qwen/Qwen3-0.6B'),
            use_ontology=tq_cfg.get('use_ontology', True),
        )

        # Evidence fusion: presence posterior over all verification signals
        _pw = tq_cfg.get('presence_weights', 'configs/presence_weights.json')
        if not os.path.isabs(_pw):
            _pw = os.path.join(os.path.dirname(os.path.abspath(__file__)), _pw)
        self.presence_model = PresenceModel(_pw)

        self._exemplar_reid = None  # lazy (Contribution A)

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

        # Load pre-computed blur scores for quality-aware frame selection
        blur_path = os.path.join(self.index_dir, 'blur_scores.npy')
        if os.path.exists(blur_path):
            self.blur_scores = np.load(blur_path)
            print(f"  Blur scores loaded: {self.blur_scores.shape} "
                  f"(mean={self.blur_scores.mean():.1f})")
        else:
            self.blur_scores = None

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
        json_out: str = None,
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

        _t_start = time.time()
        _timing = {}

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
        # v3 improvements:
        #   - Query parser for detection prompts + confusable classes
        #   - Blur-aware frame quality gating (Laplacian variance)
        #   - Multi-frame consensus (IoU across top-3 verified frames)
        #   - Calibrated confidence score
        spatial_method = self.config.get('text_query', {}).get('spatial_method', 'grounding_dino')
        scored_candidates = []
        consensus_iou = 0.0
        blur_quality = 1.0
        confusable_margin = 0.0
        consensus_app = 0.0       # appearance consensus (CLIP-crop cosine)
        best_evidence = None      # best candidate's raw signals for fusion
        _timing['retrieve_s'] = round(time.time() - _t_start, 2)
        _t_mark = time.time()

        if spatial_method == 'grounding_dino' and len(selected_candidates) > 1:
            gdino_cfg = self.config.get('grounding_dino', {})
            box_thresh = gdino_cfg.get('box_threshold', 0.20)
            text_thresh = gdino_cfg.get('text_threshold', 0.20)
            min_clip_crop = self.config.get('text_query', {}).get('min_crop_verify', 0.17)
            blur_threshold = self.config.get('text_query', {}).get('blur_threshold', 50.0)
            max_verify = min(len(selected_candidates), 12)

            # Use query parser for detection prompt + confusables
            query_plan = self.query_parser.parse(text_query)
            det_query = query_plan.detection_prompt
            conf_names = query_plan.confusables

            # Pre-encode confusable class texts for negative suppression,
            # plus the disambiguated target phrase for a fair comparison
            conf_margin_delta = self.config.get('text_query', {}).get(
                'confusable_margin_delta', 0.03)
            if conf_names:
                import open_clip
                tokens = open_clip.tokenize(conf_names).to(self.device)
                with torch.no_grad():
                    conf_feats = self.localizer.clip_model.encode_text(tokens).float()
                conf_feats = F.normalize(conf_feats, dim=-1)
                tgt_tokens = open_clip.tokenize([query_plan.target]).to(self.device)
                with torch.no_grad():
                    target_conf_feat = self.localizer.clip_model.encode_text(
                        tgt_tokens).float()
                target_conf_feat = F.normalize(target_conf_feat, dim=-1)
            else:
                conf_feats = None
                target_conf_feat = None

            print(f"\n  [QueryParser] target='{query_plan.target}' "
                  f"(confidence={query_plan.confidence})")
            print(f"  Verifying ALL {max_verify} candidates with GDino+CLIP "
                  f"(min_crop={min_clip_crop}, blur_thresh={blur_threshold})...")
            if det_query != text_query:
                print(f"  Detection query: '{det_query}'")
            if conf_names:
                print(f"  Confusable classes: {conf_names}")

            # Score every candidate
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

                    # ---- Blur quality check ----
                    frame_blur = self._get_blur_score(meta_idx, video_path, fidx)
                    is_blurry = frame_blur < blur_threshold

                    try:
                        probe = self._load_single_frame(video_path, fidx)
                    except Exception:
                        continue

                    boxes = self.grounding_dino.best_boxes(
                        probe, det_query,
                        box_threshold=box_thresh, text_threshold=text_thresh,
                        clip_model=self.localizer.clip_model,
                        clip_preprocess=self.localizer.clip_preprocess,
                        text_feat=text_feat,
                        top_k=1,
                    )
                    if not boxes:
                        blur_tag = f" BLURRY({frame_blur:.0f})" if is_blurry else ""
                        print(f"    candidate {ci}: frame {fidx} "
                              f"(t={fidx/fps:.1f}s) -- no detection{blur_tag}")
                        continue

                    bbox, clip_score, gdino_score, _label = boxes[0]

                    # ---- Negative-class suppression ----
                    confused = False
                    conf_margin = clip_score  # margin = target - max_confusable
                    crop_feat = None
                    if conf_feats is not None:
                        bx, by, bw, bh = bbox
                        crop = probe[by:by+bh, bx:bx+bw]
                        is_conf, conf_score, conf_name, target_eff, crop_feat = \
                            self._confusable_check(
                                crop, clip_score, conf_names, conf_feats,
                                target_feat=target_conf_feat,
                                margin=conf_margin_delta,
                            )
                        # Honest margin: effective target (max of raw query
                        # and disambiguated phrase) minus best confusable
                        conf_margin = target_eff - conf_score
                        if is_conf:
                            confused = True
                            print(f"    candidate {ci}: frame {fidx} "
                                  f"(t={fidx/fps:.1f}s) -- CLIP={clip_score:.3f} "
                                  f"BUT '{conf_name}'={conf_score:.3f} > target "
                                  f"-> CONFUSED")

                    # ---- Patch spatial verification ----
                    patch_score = self._patch_spatial_verify(
                        meta_idx, bbox, text_np, probe.shape
                    ) if meta_idx is not None else clip_score

                    # ---- Blur penalty ----
                    # Normalize blur to [0, 1]: 0=very blurry, 1=sharp
                    blur_norm = min(1.0, frame_blur / 500.0)

                    # ---- Combined score with all signals ----
                    if confused:
                        combined = (0.5 * clip_score + 0.3 * patch_score
                                    + 0.2 * blur_norm) * 0.3
                    else:
                        combined = (0.5 * clip_score + 0.3 * patch_score
                                    + 0.2 * blur_norm)

                    if not confused:
                        tag = "PASS" if clip_score >= min_clip_crop else "FAIL"
                        blur_tag = f" BLURRY({frame_blur:.0f})" if is_blurry else f" sharp({frame_blur:.0f})"
                        print(f"    candidate {ci}: frame {fidx} "
                              f"(t={fidx/fps:.1f}s) -- CLIP={clip_score:.3f} "
                              f"patch={patch_score:.3f} blur={blur_norm:.2f}{blur_tag} "
                              f"combined={combined:.3f} {tag}")

                    scored_candidates.append({
                        'ci': ci, 'clip_score': clip_score,
                        'patch_score': patch_score, 'combined': combined,
                        'confused': confused, 'blur_score': frame_blur,
                        'blur_norm': blur_norm, 'conf_margin': conf_margin,
                        'bbox': bbox, 'frame_idx': fidx,
                        'gdino': gdino_score, 'crop_feat': crop_feat,
                    })

            # First pass: check top-N candidates
            _verify_range(0, max_verify)

            # Auto-expand: if ALL candidates so far are confused, try more
            passed = [s for s in scored_candidates
                      if s['clip_score'] >= min_clip_crop and s['combined'] > 0.10]
            if not passed and conf_feats is not None and max_verify < len(selected_candidates):
                expand_to = min(len(selected_candidates), max_verify + 8)
                print(f"  All {max_verify} candidates confused -- expanding to {expand_to}...")
                _verify_range(max_verify, expand_to)
                passed = [s for s in scored_candidates
                          if s['clip_score'] >= min_clip_crop and s['combined'] > 0.10]

            # ---- Multi-frame consensus ----
            # Box-IoU agreement (legacy: measures camera motion as much as
            # agreement) + APPEARANCE consensus: do the top verified crops
            # look like the same kind of object? (mean pairwise CLIP cosine)
            if len(scored_candidates) >= 2:
                top3 = sorted(scored_candidates, key=lambda x: x['combined'], reverse=True)[:3]
                consensus_iou = self._compute_consensus_iou([s['bbox'] for s in top3 if s['bbox'] is not None])
                _feats3 = [s.get('crop_feat') for s in top3
                           if s.get('crop_feat') is not None]
                if len(_feats3) >= 2:
                    import itertools
                    _pair_sims = [float((a @ b.T).squeeze().cpu())
                                  for a, b in itertools.combinations(_feats3, 2)]
                    consensus_app = sum(_pair_sims) / len(_pair_sims)
                print(f"  Multi-frame consensus: IoU={consensus_iou:.3f}, "
                      f"appearance={consensus_app:.3f} "
                      f"(from {min(3, len(top3))} frames)")

            # Pick among verified candidates.
            # "latest" preserves last-occurrence semantics (episodic memory /
            # VQ2D): every candidate that passed verification is trusted, so
            # return the most RECENT one. "best" picks the highest combined
            # score — which can return an occurrence minutes earlier purely
            # on score noise (observed: 0.007 score gap = 130s temporal error).
            verified_selection = self.config.get('text_query', {}).get(
                'verified_selection', 'latest')
            if scored_candidates:
                if passed:
                    if verified_selection == 'latest':
                        best = max(passed, key=lambda x:
                                   selected_candidates[x['ci']]['frame_idx'])
                    else:
                        best = max(passed, key=lambda x: x['combined'])
                else:
                    best = max(scored_candidates, key=lambda x: x['combined'])

                best_ci = best['ci']
                blur_quality = best['blur_norm']
                confusable_margin = best['conf_margin']
                best_evidence = best

                if best_ci != 0:
                    old_frame = selected_candidates[0]['frame_idx']
                    selected_candidates = [selected_candidates[best_ci]] + \
                        [c for i, c in enumerate(selected_candidates) if i != best_ci]
                    tag = (("Latest verified" if verified_selection == 'latest'
                            else "Best verified") if passed
                           else "No clean candidate; best combined")
                    print(f"  {tag}: frame {selected_candidates[0]['frame_idx']} "
                          f"(combined={best['combined']:.3f}, blur={best['blur_score']:.0f}, "
                          f"was {old_frame})")
                else:
                    tag = "already first"
                    print(f"  Best verified: frame {selected_candidates[0]['frame_idx']} "
                          f"(combined={best['combined']:.3f}, {tag})")
            else:
                print(f"  GDino found nothing on any candidate; keeping original")

        # ---- Confidence score ----
        # Calibrated confidence combining all signals
        is_confused = False
        confidence_score = 0.5  # default

        if spatial_method == 'grounding_dino' and scored_candidates:
            passed_final = [s for s in scored_candidates
                            if s['clip_score'] >= min_clip_crop and s['combined'] > 0.10]
            if not passed_final:
                is_confused = True

            # Compute calibrated confidence
            if scored_candidates:
                best = max(scored_candidates, key=lambda x: x['combined'])
                confidence_score = self._compute_confidence(
                    clip_score=best['clip_score'],
                    patch_score=best['patch_score'],
                    confusable_margin=confusable_margin,
                    blur_quality=blur_quality,
                    consensus_iou=consensus_iou,
                    is_confused=is_confused,
                )
                print(f"  Confidence score: {confidence_score:.3f} "
                      f"({'HIGH' if confidence_score >= 0.7 else 'MEDIUM' if confidence_score >= 0.4 else 'LOW'})")

        best_candidate = selected_candidates[0]
        last_frame_idx = best_candidate['frame_idx']
        last_sim = best_candidate['similarity']
        last_meta_idx = best_candidate.get('meta_idx', None)

        if last_meta_idx is None:
            for i, fm in enumerate(frame_metadata):
                if fm['frame_idx'] == last_frame_idx:
                    last_meta_idx = i
                    break

        _timing['verify_s'] = round(time.time() - _t_mark, 2)
        _t_mark = time.time()
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

        # ---- Multi-instance detection: find ALL matching objects on the
        # response frame (primary bbox stays instance 0) ----
        _multi = self.config['text_query'].get('multi_instance', False)
        _max_inst = int(self.config['text_query'].get('max_instances', 5))
        instances = [{'bbox': [bx, by, bw, bh],
                      'score': round(float(region_score), 4)}]
        if _multi and not is_confused:
            try:
                gdino_cfg = self.config.get('grounding_dino', {})
                _boxes = self.grounding_dino.best_boxes(
                    frames[center_local],
                    self._detection_query(text_query),
                    box_threshold=gdino_cfg.get('box_threshold', 0.30),
                    text_threshold=gdino_cfg.get('text_threshold', 0.25),
                    clip_model=self.localizer.clip_model,
                    clip_preprocess=self.localizer.clip_preprocess,
                    text_feat=text_feat,
                    top_k=_max_inst,
                )
                # Per-instance confusable suppression: a secondary instance
                # must also beat the negative classes, not just match the
                # query (else "pan" happily collects the rice cooker)
                _plan = self.query_parser.parse(text_query)
                _conf_names = _plan.confusables
                _conf_feats = None
                _tgt_feat = None
                _margin = self.config.get('text_query', {}).get(
                    'confusable_margin_delta', 0.03)
                if _conf_names:
                    import open_clip
                    _tokens = open_clip.tokenize(_conf_names).to(self.device)
                    with torch.no_grad():
                        _conf_feats = self.localizer.clip_model.encode_text(
                            _tokens).float()
                    _conf_feats = F.normalize(_conf_feats, dim=-1)
                    _tt = open_clip.tokenize([_plan.target]).to(self.device)
                    with torch.no_grad():
                        _tgt_feat = self.localizer.clip_model.encode_text(
                            _tt).float()
                    _tgt_feat = F.normalize(_tgt_feat, dim=-1)

                # Secondary instances need a minimum score floor too — the
                # primary went through full verification, these did not
                # (observed: a bowl of eggs accepted as pan #3 at score 0.11)
                _min_inst = self.config.get('text_query', {}).get(
                    'min_crop_verify', 0.17)

                for _bb, _cs, _gs, _lb in _boxes:
                    if len(instances) >= _max_inst:
                        break
                    if any(self.grounding_dino._iou_xywh(_bb, inst['bbox']) >= 0.5
                           for inst in instances):
                        continue
                    _x, _y, _w2, _h2 = [int(v) for v in _bb]
                    if float(_cs) < _min_inst:
                        print(f"  instance candidate {[_x, _y, _w2, _h2]} rejected: "
                              f"score={float(_cs):.3f} < min_crop_verify={_min_inst}")
                        continue
                    _crop = frames[center_local][_y:_y+_h2, _x:_x+_w2]
                    _is_conf, _cscore, _cname, _teff, _cfeat = \
                        self._confusable_check(
                            _crop, float(_cs), _conf_names, _conf_feats,
                            target_feat=_tgt_feat, margin=_margin)
                    if _is_conf:
                        print(f"  instance candidate {[_x, _y, _w2, _h2]} rejected: "
                              f"'{_cname}'={_cscore:.3f} > target={_cs:.3f}")
                        continue
                    instances.append({'bbox': [_x, _y, _w2, _h2],
                                      'score': round(float(_cs), 4)})
                if len(instances) > 1:
                    print(f"  Multi-instance: {len(instances)} instance(s) of "
                          f"'{text_query}' on response frame")
            except Exception as e:
                print(f"  Multi-instance detection failed: {e}; keeping primary only")

        # Save debug image
        debug_frame = frames[center_local].copy()
        px, py = int(region_point[0]), int(region_point[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        if is_confused:
            # Red bbox + CONFUSED label for uncertain results
            bbox_color = (255, 0, 0)   # Red in RGB
            cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), bbox_color, 3)
            label = f"CONFUSED: '{text_query}' conf={confidence_score:.2f}"
            font_scale, thickness = 0.7, 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 10, by - 10)
            cv2.rectangle(debug_frame, (bx, label_y - th - 6),
                          (bx + tw + 6, label_y + 4), (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (bx + 3, label_y - 2),
                        font, font_scale, (255, 80, 80), thickness)
        elif confidence_score < 0.4:
            # Orange bbox for low confidence
            bbox_color = (255, 165, 0)  # Orange in RGB
            cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), bbox_color, 2)
            label = f"LOW CONF: '{text_query}' conf={confidence_score:.2f}"
            font_scale, thickness = 0.6, 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 10, by - 10)
            cv2.rectangle(debug_frame, (bx, label_y - th - 6),
                          (bx + tw + 6, label_y + 4), (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (bx + 3, label_y - 2),
                        font, font_scale, (255, 180, 50), thickness)
        else:
            # Green bbox for confident results
            cv2.circle(debug_frame, (px, py), 8, (255, 0, 0), 2)
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = f"'{text_query}' conf={confidence_score:.2f}"
            font_scale, thickness = 0.6, 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(th + 10, by - 10)
            cv2.rectangle(debug_frame, (bx, label_y - th - 6),
                          (bx + tw + 6, label_y + 4), (0, 0, 0), -1)
            cv2.putText(debug_frame, label, (bx + 3, label_y - 2),
                        font, font_scale, (0, 255, 0), thickness)
        # Secondary instances in cyan (multi-instance mode)
        for _k, _inst in enumerate(instances[1:], start=2):
            _ix, _iy, _iw, _ih = [int(v) for v in _inst['bbox']]
            cv2.rectangle(debug_frame, (_ix, _iy), (_ix + _iw, _iy + _ih),
                          (0, 200, 255), 2)
            cv2.putText(debug_frame, f"#{_k} {_inst['score']:.2f}",
                        (_ix + 3, max(15, _iy - 6)), font, 0.55, (0, 200, 255), 2)
        debug_path = os.path.join(output_dir, 'debug_last_frame.jpg')
        cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        print(f"  Debug frame saved: {debug_path}")

        # ---- Evidence fusion: presence posterior (Contribution C) ----
        # All verification signals fuse into ONE calibrated decision —
        # computed BEFORE tracking so abstention short-circuits the most
        # expensive stage (observed: 7 min spent tracking a nonexistent
        # "zebra" because the decision came after tracking).
        try:
            _s_ret = float((last_sim - float(all_sims.mean()))
                           / (float(all_sims.std()) + 1e-6))
        except Exception:
            _s_ret = 0.0
        _frame_area = float(max(1, frames[0].shape[0] * frames[0].shape[1]))
        evidence = EvidenceVector(
            s_clip=float(best_evidence['clip_score']) if best_evidence else float(last_sim),
            s_patch=float(best_evidence['patch_score']) if best_evidence else 0.0,
            m_conf=float(confusable_margin),
            q_blur=float(blur_quality),
            c_app=float(consensus_app),
            s_gdino=float(best_evidence.get('gdino', 0.0)) if best_evidence else 0.0,
            s_ret=_s_ret,
            a_frac=float(bw * bh) / _frame_area,
            s_reid=0.0,  # populated by exemplar re-ID (Contribution A, 1.5)
        )
        presence = self.presence_model.presence(evidence)
        tau_abstain = self.config['text_query'].get('tau_abstain', 0.2)
        tau_accept = self.config['text_query'].get('tau_accept', 0.5)
        decision = ('not_found' if presence < tau_abstain
                    else 'uncertain' if presence < tau_accept
                    else 'found')
        if presence < tau_abstain:
            print(f"\n  Presence p={presence:.3f} < tau_abstain={tau_abstain} "
                  f"-> NOT FOUND (calibrated abstention)")
        elif presence < tau_accept:
            print(f"\n  Presence p={presence:.3f} in uncertainty zone "
                  f"[{tau_abstain}, {tau_accept}) -> candidate for "
                  f"plan refinement (1.6)")
        else:
            print(f"\n  Presence p={presence:.3f} >= tau_accept={tau_accept} "
                  f"-> FOUND")

        # ---- SAM2 Tracking or single-frame export ----
        # Detect once, then ONE SAM2 track per instance across the window
        _skip_sam2 = self.config['text_query'].get('skip_sam2_eval', True)
        if presence < tau_abstain and not _skip_sam2:
            print("  Skipping SAM2 tracking — presence below abstention "
                  "threshold (no point tracking an absent object)")
            _skip_sam2 = True
        _timing['localize_s'] = round(time.time() - _t_mark, 2)
        _t_mark = time.time()
        # Masks (COCO RLE) are only encoded when a JSON export is requested
        _save_masks = (json_out is not None) and \
            self.config['text_query'].get('save_masks', True)

        all_tracks = []
        if not _skip_sam2:
            _seeds = [[int(v) for v in _inst['bbox']] for _inst in instances]
            print(f"\n  SAM2 tracking {len(_seeds)} instance(s) in ONE session "
                  f"from frame {last_frame_idx}...")
            try:
                all_tracks = self.localizer.track_from_bboxes(
                    frames, center_local, _seeds,
                    half_span=len(frames) // 2,
                    return_masks=_save_masks,
                )
                # Remap local frame indices back to global
                for _tr in all_tracks:
                    for t in _tr:
                        local_idx = t['frame_idx']
                        if local_idx < len(frame_indices):
                            t['frame_idx'] = frame_indices[local_idx]
                print("  SAM2 tracked: " + ", ".join(
                    f"{len(_tr)} frames (instance {_i})"
                    for _i, _tr in enumerate(all_tracks)))
            except Exception as e:
                print(f"  SAM2 tracking failed: {e}, using single-frame bbox(es)")
                all_tracks = [[{'frame_idx': last_frame_idx, 'bbox': _s}]
                              for _s in _seeds]
        else:
            all_tracks = [[{'frame_idx': last_frame_idx,
                            'bbox': [int(v) for v in _inst['bbox']]}]
                          for _inst in instances]
        # Primary instance renders the result clip (multi-track export in 1.4)
        track = all_tracks[0]

        # ---- Exemplar re-ID sweep (Contribution A) ----
        # The verified detection bootstraps an instance-level exemplar bank
        # (views harvested from the SAM2 track) that sweeps the WHOLE index
        # for other occurrences of the same object instance.
        reid_occurrences = []
        _reid_on = self.config['text_query'].get('exemplar_reid', True)
        if _reid_on and presence >= tau_abstain:
            _t_reid = time.time()
            try:
                from exemplar_reid import ExemplarReID
                if self._exemplar_reid is None:
                    self._exemplar_reid = ExemplarReID(
                        self.localizer.clip_model,
                        self.localizer.clip_preprocess,
                        device=self.device,
                    )
                _g2l = {g: i for i, g in enumerate(frame_indices)}
                bank = self._exemplar_reid.build_bank(
                    frames, _g2l, center_local, [bx, by, bw, bh],
                    track if not _skip_sam2 else [],
                    max_views=int(self.config['text_query'].get(
                        'exemplar_views', 8)),
                )
                if bank is not None:
                    print(f"\n  [ReID] Exemplar bank: {bank['num_views']} views; "
                          f"sweeping index for other occurrences...")
                    _ctx_frames = int(context_seconds * fps)
                    _exclude = {i for i, m in enumerate(frame_metadata)
                                if abs(m['frame_idx'] - last_frame_idx) <= _ctx_frames}
                    rows = self._exemplar_reid.candidate_rows(
                        bank, self.clip_embeddings, _exclude,
                        top_k=int(self.config['text_query'].get(
                            'exemplar_top_k', 12)),
                    )
                    _reid_thresh = float(self.config['text_query'].get(
                        'reid_sim_threshold', 0.6))
                    _max_occ = int(self.config['text_query'].get(
                        'max_reid_occurrences', 5))
                    gdino_cfg = self.config.get('grounding_dino', {})
                    _det_q = self._detection_query(text_query)
                    for _row, _row_score in rows:
                        if len(reid_occurrences) >= _max_occ:
                            break
                        _fidx = frame_metadata[_row]['frame_idx']
                        # one occurrence per temporal neighbourhood
                        if any(abs(_fidx - occ['frame_idx']) <= _ctx_frames
                               for occ in reid_occurrences):
                            continue
                        try:
                            _probe = self._load_single_frame(video_path, _fidx)
                        except Exception:
                            continue
                        _dets = self.grounding_dino.best_boxes(
                            _probe, _det_q,
                            box_threshold=gdino_cfg.get('box_threshold', 0.30),
                            text_threshold=gdino_cfg.get('text_threshold', 0.25),
                            top_k=3,
                        )
                        if not _dets:
                            continue
                        _hit = self._exemplar_reid.verify_candidate(
                            _probe, [d[0] for d in _dets], bank,
                            sim_threshold=_reid_thresh,
                        )
                        if _hit is not None:
                            _rb, _rs = _hit
                            reid_occurrences.append({
                                'frame_idx': int(_fidx),
                                'timestamp': round(_fidx / fps, 3),
                                'bbox': _rb,
                                's_reid': round(_rs, 4),
                            })
                            print(f"  [ReID] occurrence at t={_fidx/fps:.1f}s "
                                  f"(frame {_fidx}, s_reid={_rs:.3f})")
                            # Annotated thumbnail: visual verification of
                            # every re-ID claim (and paper figure material)
                            _ann = _probe.copy()
                            cv2.rectangle(_ann, (_rb[0], _rb[1]),
                                          (_rb[0] + _rb[2], _rb[1] + _rb[3]),
                                          (0, 200, 255), 3)
                            cv2.putText(_ann, f"reid {_rs:.2f}",
                                        (_rb[0] + 3, max(20, _rb[1] - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 200, 255), 2)
                            cv2.imwrite(
                                os.path.join(output_dir,
                                             f"reid_occ_t{int(_fidx/fps)}s.jpg"),
                                cv2.cvtColor(_ann, cv2.COLOR_RGB2BGR))
                    if reid_occurrences:
                        # Strongest re-ID agreement feeds back into fusion
                        evidence.s_reid = max(o['s_reid']
                                              for o in reid_occurrences)
                        presence = self.presence_model.presence(evidence)
                        decision = ('not_found' if presence < tau_abstain
                                    else 'uncertain' if presence < tau_accept
                                    else 'found')
                        print(f"  [ReID] {len(reid_occurrences)} additional "
                              f"occurrence(s); presence updated -> "
                              f"{presence:.3f} ({decision})")
                    else:
                        print("  [ReID] no additional occurrences above "
                              "threshold")
            except Exception as e:
                print(f"  [ReID] sweep failed: {e}")
            _timing['reid_s'] = round(time.time() - _t_reid, 2)

        _timing['track_s'] = round(time.time() - _t_mark, 2)

        print("\nExporting result clip...")
        self.localizer.export_clip(
            video_path, track, last_frame_idx,
            fps, os.path.join(output_dir, 'last_occurrence.mp4'),
            context_seconds=context_seconds,
        )
        _timing['total_s'] = round(time.time() - _t_start, 2)

        # Run provenance: which config and code produced these numbers
        _config_hash = hashlib.md5(
            json.dumps(self.config, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        try:
            import subprocess
            _git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            _git_hash = None

        result = {
            'query': text_query,
            'video_path': video_path,
            'last_frame_idx': last_frame_idx,
            'last_frame_timestamp': round(last_frame_idx / fps, 3),
            'pred_bbox': [bx, by, bw, bh],
            'num_instances': len(instances),
            'instances': instances,
            'tracked_frames_per_instance': [len(t) for t in all_tracks],
            'reid_occurrences': reid_occurrences,
            'confused': is_confused,
            'confidence': round(confidence_score, 4),
            'presence': round(presence, 4),
            'found': bool(presence >= tau_abstain),
            'decision': decision,
            'evidence': evidence.to_dict(),
            'clip_similarity': round(float(clip_sims[last_meta_idx]), 4),
            'fused_similarity': round(last_sim, 4),
            'region_point': list(region_point),
            'region_clip_score': round(region_score, 4),
            'blur_quality': round(blur_quality, 4),
            'consensus_iou': round(consensus_iou, 4),
            'confusable_margin': round(confusable_margin, 4),
            'similarity_threshold': threshold,
            'valid_segments': num_valid_segments,
            'frames_above_threshold': int(n_above),
            'context_seconds': context_seconds,
            'fps': fps,
            'timing': _timing,
            'config_hash': _config_hash,
            'git_hash': _git_hash,
        }
        with open(os.path.join(output_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        # ---- Full machine-readable export (benchmark adapters read this) ----
        if json_out:
            full = {
                'query': text_query,
                'plan': self.query_parser.parse(text_query).to_dict(),
                'found': bool(presence >= tau_abstain),
                'decision': decision,
                'presence': round(presence, 4),
                'confidence': round(confidence_score, 4),
                # Evidence vector — the dataset for presence-model fitting
                # (Phase 4 dev labels) and the ROC abstention figure
                'evidence': evidence.to_dict(),
                'last_frame_idx': last_frame_idx,
                'last_frame_timestamp': round(last_frame_idx / fps, 3),
                'temporal_window_s': [
                    round(max(0.0, last_frame_idx / fps - context_seconds / 2), 3),
                    round(last_frame_idx / fps + context_seconds / 2, 3),
                ],
                'tracks': [
                    {
                        'instance_id': i,
                        'source': 'text',
                        'seed_bbox': instances[i]['bbox'],
                        'score': instances[i]['score'],
                        'num_frames': len(tr),
                        'frames': tr,
                    }
                    for i, tr in enumerate(all_tracks)
                ] + [
                    {
                        'instance_id': 100 + i,
                        'source': 'reid',
                        'seed_bbox': occ['bbox'],
                        'score': occ['s_reid'],
                        'num_frames': 1,
                        'frames': [{'frame_idx': occ['frame_idx'],
                                    'bbox': occ['bbox']}],
                    }
                    for i, occ in enumerate(reid_occurrences)
                ],
                'timing': _timing,
                'config_hash': _config_hash,
                'git_hash': _git_hash,
                'video_path': video_path,
                'fps': fps,
            }
            _json_dir = os.path.dirname(json_out)
            if _json_dir:
                os.makedirs(_json_dir, exist_ok=True)
            with open(json_out, 'w') as f:
                json.dump(full, f, indent=2)
            print(f"  Full JSON export: {json_out}")

        return result

    # ------------------------------------------------------------------ #
    # Detection query expansion (delegates to QueryParser)                 #
    # ------------------------------------------------------------------ #

    def _detection_query(self, text_query: str) -> str:
        """Expand query for Grounding DINO detection via QueryParser."""
        plan = self.query_parser.parse(text_query)
        return plan.detection_prompt

    def _confusable_check(
        self,
        crop_rgb: np.ndarray,
        target_clip_score: float,
        confusable_names: list,
        confusable_feats: torch.Tensor,
        target_feat: torch.Tensor = None,
        margin: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """
        Check if a detection crop is more similar to a confusable class
        than to the target query.

        target_feat: CLIP text embedding of the DISAMBIGUATED target phrase
        (e.g. "frying pan"). The raw query ("pan") is a vague single word
        and systematically loses to specific confusable phrases ("pot lid")
        on the same crop — scoring both sides with specific phrases makes
        the comparison fair. The higher of raw/disambiguated is used.

        margin: required winning margin before rejecting. CLIP similarity
        differences below ~0.03 are noise for near-synonym categories.

        Returns (is_confused, max_confusable_score, worst_confusable_name).
        is_confused=True means the crop matched a confusable better.
        """
        if confusable_feats is None or crop_rgb.size == 0:
            return False, 0.0, "", target_clip_score, None

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

        # Score the SAME crop against the disambiguated target phrase
        if target_feat is not None:
            t_spec = float((crop_feat @ target_feat.T).squeeze().cpu())
            target_clip_score = max(target_clip_score, t_spec)

        # Confused only if the best confusable wins by more than the margin
        is_confused = max_conf > target_clip_score + margin
        # Also return the effective target score (for honest margin logging)
        # and the crop embedding (reused for appearance consensus)
        return is_confused, max_conf, worst_name, target_clip_score, crop_feat

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
    # Blur-aware frame quality gating                                      #
    # ------------------------------------------------------------------ #

    def _get_blur_score(self, meta_idx: int, video_path: str = None,
                        frame_idx: int = None) -> float:
        """
        Get blur score (Laplacian variance) for a frame.

        Uses pre-computed scores from index if available, otherwise computes
        on-the-fly. Higher = sharper. Typical values:
          - < 30: severe motion blur
          - 30-100: moderate blur
          - 100-500: acceptable sharpness
          - > 500: very sharp / high-frequency texture
        """
        # Try pre-computed scores first (fast path)
        if self.blur_scores is not None and meta_idx is not None:
            if meta_idx < len(self.blur_scores):
                return float(self.blur_scores[meta_idx])

        # Compute on-the-fly (slow path — only for non-indexed frames)
        if video_path is not None and frame_idx is not None:
            try:
                frame = self._load_single_frame(video_path, frame_idx)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                return float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                pass

        return 500.0  # Default to "sharp" if we can't compute

    # ------------------------------------------------------------------ #
    # Multi-frame consensus (IoU agreement)                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bbox_iou(box1: list, box2: list) -> float:
        """Compute IoU between two [x, y, w, h] bboxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_consensus_iou(bboxes: list) -> float:
        """
        Compute pairwise IoU consensus across multiple bboxes.

        Returns the average pairwise IoU. High IoU (>0.3) means the
        detection is spatially consistent across frames — strong signal
        that the object was correctly localized.
        """
        if len(bboxes) < 2:
            return 0.0

        ious = []
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if bboxes[i] is not None and bboxes[j] is not None:
                    ious.append(IndexedQueryEngine._bbox_iou(bboxes[i], bboxes[j]))

        return float(np.mean(ious)) if ious else 0.0

    # ------------------------------------------------------------------ #
    # Calibrated confidence score                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_confidence(
        clip_score: float,
        patch_score: float,
        confusable_margin: float,
        blur_quality: float,
        consensus_iou: float,
        is_confused: bool,
    ) -> float:
        """
        Compute a calibrated confidence score in [0, 1].

        Combines five signals:
          - CLIP crop score (how well the crop matches the text query)
          - Patch verification score (local spatial match)
          - Confusable margin (target_score - max_confusable_score)
          - Blur quality (normalized Laplacian variance, 0=blurry, 1=sharp)
          - Consensus IoU (spatial agreement across top-3 frames)

        If confused, the score is capped at 0.25.
        """
        if is_confused:
            return min(0.25, 0.3 * clip_score + 0.1 * blur_quality)

        # Normalize confusable margin: positive = target wins
        margin_norm = min(1.0, max(0.0, (confusable_margin + 0.1) / 0.3))

        # Weighted combination
        score = (
            0.30 * min(1.0, clip_score / 0.30)     # CLIP crop (normalize to ~0.30 max)
            + 0.20 * min(1.0, patch_score / 0.30)   # Patch verify
            + 0.20 * margin_norm                     # Confusable margin
            + 0.15 * blur_quality                    # Frame sharpness
            + 0.15 * consensus_iou                   # Multi-frame agreement
        )

        return float(np.clip(score, 0.0, 1.0))

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
    parser.add_argument('--json-out', type=str, default=None, dest='json_out',
                        help='Write full machine-readable result (plan, evidence, '
                             'per-instance tracks with COCO RLE masks, timing) to '
                             'this path — benchmark adapters consume this file')

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
        json_out=args.json_out,
    )

    print("\n=== Query Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
