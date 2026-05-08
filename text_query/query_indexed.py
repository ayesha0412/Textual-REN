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
    print("Warning: rapidfuzz not installed — OCR fusion disabled.")
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
                print(f"  Compositional: active — noun '{sub_queries[1]}' max={noun_signal:.3f}")
            else:
                clip_sims = sub_sims_list[0]
                print(f"  Compositional: skipped — noun '{sub_queries[1]}' signal too weak "
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
        print(f"  Query type: {'BRAND/TEXT — OCR fusion enabled' if is_brand else 'OBJECT — pure CLIP (OCR skipped)'}")

        if is_brand:
            ocr_scores = self._compute_ocr_scores(text_query, frame_metadata, window=2)
            # Only trust very high OCR matches (≥0.85) to avoid spurious text hits
            if ocr_scores.max() < 0.85:
                ocr_scores[:] = 0.0
            n_ocr_hits = int((ocr_scores >= 0.85).sum())
            if n_ocr_hits > 0:
                print(f"  OCR hits (≥0.85 match): {n_ocr_hits}/{len(ocr_scores)} frames")
                print(f"  OCR weight: {ocr_weight}  (fused = CLIP + {ocr_weight} × OCR)")
            else:
                print(f"  OCR: index has no clear text match — falling back to CLIP only")
        else:
            ocr_scores = np.zeros(len(frame_metadata), dtype=np.float32)
            n_ocr_hits = 0

        all_sims = clip_sims + ocr_weight * ocr_scores

        n_above = int((all_sims >= threshold).sum())
        print(f"\n  Fused scores — max={all_sims.max():.4f}  "
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

        # Identify global peak segment (used by ablation_use_strongest mode).
        segment_peaks = [(max(s, key=lambda i: all_sims[i]), s) for s in valid]
        global_peak_idx, global_peak_segment = max(
            segment_peaks, key=lambda t: all_sims[t[0]]
        )
        global_peak_sim = float(all_sims[global_peak_idx])

        if ablation_use_strongest:
            # Ablation: bypass temporal logic, use the globally strongest segment.
            last_segment = global_peak_segment
            last_meta_idx = global_peak_idx
        else:
            # Default: prefer the LAST (most recent) segment.
            # Step 3b crop-verification below will refine this choice — if the
            # last segment clearly doesn't contain the object an earlier segment
            # with a better crop score will be accepted instead.
            last_segment = valid[-1]
            last_meta_idx = max(last_segment, key=lambda i: all_sims[i])

        last_frame_idx = frame_metadata[last_meta_idx]['frame_idx']
        last_sim = float(all_sims[last_meta_idx])

        if not ablation_use_strongest:
            min_peak = self.config['text_query'].get('last_segment_min_peak', threshold)
            min_rel  = self.config['text_query'].get('last_segment_min_rel', 0.0)
            if last_sim < min_peak or last_sim < min_rel * global_peak_sim:
                # Last segment is genuinely weak — fall back to the strongest.
                last_meta_idx  = global_peak_idx
                last_segment   = global_peak_segment
                last_frame_idx = frame_metadata[last_meta_idx]['frame_idx']
                last_sim       = global_peak_sim
                print(f"\n  Last segment weak (peak={last_sim:.3f} vs min_peak={min_peak}, "
                      f"global={global_peak_sim:.3f}) — falling back to strongest segment")

        # ---- Step 3b: verify the chosen frame actually contains the object ----
        crop_score = 0.0
        MIN_CROP_VERIFY = self.config['text_query'].get('min_crop_verify', 0.17)
        if ablation_no_verification:
            MIN_CROP_VERIFY = 0.0   # always accept first candidate

        # Walk from the latest segment backward.  If a segment passes the crop
        # verify threshold we accept it immediately (it is the last good
        # occurrence).  If *no* segment passes we fall back to the latest
        # segment — a low crop score at the right time beats a high crop score
        # at the wrong time for general-object queries.
        last_seg_fallback = None   # (segment, meta_idx, frame_idx, sim, frame_rgb, score)

        for attempt in range(len(valid) - 1, -1, -1):
            cand_segment = valid[attempt]
            cand_meta_idx = max(cand_segment, key=lambda i: all_sims[i])
            cand_frame_idx = frame_metadata[cand_meta_idx]['frame_idx']
            cand_frame_rgb = self._load_single_frame(video_path, cand_frame_idx)
            _, crop_score = self._find_best_region_clip(cand_frame_rgb, text_feat, grid_size=3)
            t = cand_frame_idx / fps
            print(f"  Verify segment {attempt+1}/{len(valid)}: "
                  f"frame {cand_frame_idx} (t={t:.1f}s) crop_score={crop_score:.3f}", end='')

            # Remember the last (most recent) segment as our ultimate fallback.
            if last_seg_fallback is None:
                last_seg_fallback = (cand_segment, cand_meta_idx, cand_frame_idx,
                                     float(all_sims[cand_meta_idx]), cand_frame_rgb, crop_score)

            if crop_score >= MIN_CROP_VERIFY:
                last_segment = cand_segment
                last_meta_idx = cand_meta_idx
                last_frame_idx = cand_frame_idx
                last_sim = float(all_sims[last_meta_idx])
                last_frame_rgb = cand_frame_rgb
                print(f"  ✓ accepted")
                break
            print(f"  ✗ too low — trying previous segment")
        else:
            # No segment passed verification — use the LAST (most recent) segment,
            # not the earliest one.  Falling back to the earliest frame caused wrong
            # objects to be localized when the target object was visible only later.
            (last_segment, last_meta_idx, last_frame_idx,
             last_sim, last_frame_rgb, crop_score) = last_seg_fallback
            print(f"\n  No segment passed crop verify — defaulting to last occurrence "
                  f"(t={last_frame_idx/fps:.1f}s, crop={crop_score:.3f})")

        print(f"\n  Last occurrence → frame {last_frame_idx}  "
              f"(t={last_frame_idx/fps:.2f}s, sim={last_sim:.3f}, crop={crop_score:.3f})")

        # ---- Step 4: spatial localization on the last-occurrence frame ----

        ocr_bbox_result = None
        ocr_bbox_frame_idx = None
        last_ocr_score = float(ocr_scores[last_meta_idx])

        # When this is a brand query with high OCR confidence, read the bbox
        # directly from the frame rather than relying on SAM2 mask selection.
        if is_brand and last_ocr_score >= 0.85 and HAS_EASYOCR:
            print(f"\nOCR score high ({last_ocr_score:.2f}) — locating brand bbox...")
            # Build candidate frame indices: last-occurrence first, then neighbors
            # sorted by their OCR score descending so we try the clearest frame first.
            segment_indices = sorted(
                last_segment,
                key=lambda i: ocr_scores[i],
                reverse=True,
            )[:6]
            for cand_meta_idx in segment_indices:
                cand_frame_idx = frame_metadata[cand_meta_idx]['frame_idx']
                cand_frame_rgb = self._load_single_frame(video_path, cand_frame_idx)
                ocr_bbox_result = self._get_ocr_bbox(cand_frame_rgb, text_query)
                if ocr_bbox_result is not None:
                    ocr_bbox_frame_idx = cand_frame_idx
                    print(f"  OCR bbox found on frame {cand_frame_idx} "
                          f"(t={cand_frame_idx/fps:.1f}s)")
                    break
            if ocr_bbox_result is None:
                print("  OCR bbox not found on any nearby frame — falling back to CLIP")

        if ocr_bbox_result is not None:
            region_point, ocr_bbox_xywh = ocr_bbox_result
            region_score = last_ocr_score
            if ocr_bbox_frame_idx != last_frame_idx:
                print("  OCR bbox came from a nearby frame — using its center as SAM2 point")
                ocr_bbox_xywh = None
            print(f"  Spatial location (OCR): {region_point}")
        else:
            print(f"\nSpatial localization via CLIP crop scoring ({region_grid}×{region_grid} grid)...")
            region_point, region_score = self._find_best_region_clip(
                last_frame_rgb, text_feat, grid_size=region_grid
            )
            print(f"  Spatial location (CLIP): {region_point}  (score: {region_score:.3f})")
            ocr_bbox_xywh = None

        # ---- Step 5: context window centred on the last-occurrence frame ----
        context_seconds = self.config['text_query'].get('context_seconds', 5.0)
        half_span = int(context_seconds * fps / 2)

        print(f"\nLoading ±{context_seconds/2:.1f}s context window...")
        frames, frame_indices = self._load_frame_window(
            video_path, last_frame_idx, half_span
        )
        start_idx = max(0, last_frame_idx - half_span)
        center_local = last_frame_idx - start_idx   # may be < half_span near video edges

        # ---- Step 6: bbox — OCR direct or SAM2 ----
        if ocr_bbox_xywh is not None:
            bbox = torch.tensor(ocr_bbox_xywh)
            print(f"SAM2 skipped — using OCR bbox: {bbox.tolist()}")
        else:
            print("SAM2: estimating bbox from region point...")
            region_point_yx = np.array([region_point[1], region_point[0]], dtype=np.float32)
            # Large-area objects (paintings, screens, doors) need a higher mask
            # area ceiling so SAM2 doesn't filter them out in favour of small
            # foreground objects that happen to be in the same frame.
            words_lower = set(text_query.lower().split())
            is_large_obj = bool(words_lower & self._LARGE_OBJECT_WORDS)
            max_area_frac = 0.60 if is_large_obj else 0.12
            if is_large_obj:
                print(f"  Large-area object detected — SAM2 mask limit raised to {max_area_frac:.0%}")
            bbox = self.localizer.point_to_bbox(
                frames[center_local], region_point_yx, text_feat,
                max_area_frac=max_area_frac,
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
            'valid_segments': len(valid),
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
              f"match={best_score:.2f} → [{x1},{y1},{x2-x1},{y2-y1}]")
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
    # CLIP crop scoring                                                    #
    # ------------------------------------------------------------------ #

    def _find_best_region_clip(
        self,
        frame_rgb: np.ndarray,
        text_feat: torch.Tensor,
        grid_size: int = 6,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Score crops at multiple scales with 50% overlap so objects near tile
        boundaries and corners are fully captured in at least one crop.

        Crops generated:
          - Full frame (global context)
          - 4 quadrant halves
          - grid_size×grid_size tiles at 50% stride (overlapping)
        """
        h, w = frame_rgb.shape[:2]

        crops, centers = [], []

        # Full frame
        crops.append(self.localizer.clip_preprocess(Image.fromarray(frame_rgb)))
        centers.append((w // 2, h // 2))

        # Quadrant halves (2×2 with 50% overlap = 3×3 positions)
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

        # Fine grid at 50% overlap
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
