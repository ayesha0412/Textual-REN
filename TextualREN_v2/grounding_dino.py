"""
Grounding DINO spatial localizer for text-conditioned object detection.

Replaces CLIP crop scoring with a purpose-built open-vocabulary detector
that directly outputs bounding boxes from text queries.
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional


class GroundingDINOLocalizer:
    """Lazy-loaded Grounding DINO for text → bbox detection."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: torch.device = None):
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
        print(f"  Loading Grounding DINO ({self.model_id})...")
        self._processor = GroundingDinoProcessor.from_pretrained(self.model_id)
        self._model = GroundingDinoForObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device).eval()

    def offload(self):
        """Move model to CPU to free VRAM."""
        if self._model is not None:
            self._model.cpu()
            torch.cuda.empty_cache()

    def _detect_single(
        self,
        image: Image.Image,
        query: str,
        box_threshold: float,
        text_threshold: float,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Tuple[List[int], float, str]]:
        """Run detection on a single image tile, offsetting boxes."""
        inputs = self._processor(images=image, text=query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        w_img, h_img = image.size
        dets = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            x1 = max(0, int(x1)) + offset_x
            y1 = max(0, int(y1)) + offset_y
            x2 = min(w_img, int(x2)) + offset_x
            y2 = min(h_img, int(y2)) + offset_y
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(score), label))
        return dets

    def detect(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        use_sahi: bool = False,
        sahi_slice_size: int = 480,
        sahi_overlap_ratio: float = 0.25,
    ) -> List[Tuple[List[int], float, str]]:
        """
        Detect objects matching text_query in a frame.

        Returns list of (bbox_xywh, confidence, label) sorted by confidence desc.
        bbox_xywh is [x, y, w, h] in pixel coordinates.

        When use_sahi=True, also runs detection on overlapping tiles to catch
        small objects that are missed at full resolution (SAHI, arxiv 2202.06934).
        """
        self._load()
        self._model.to(self.device)

        image = Image.fromarray(frame_rgb)
        query = text_query.strip().rstrip(".") + "."
        h, w = frame_rgb.shape[:2]

        # Full-frame detection (always)
        detections = self._detect_single(image, query, box_threshold, text_threshold)

        # Tiled detection for small objects
        if use_sahi and (h > sahi_slice_size or w > sahi_slice_size):
            stride = int(sahi_slice_size * (1 - sahi_overlap_ratio))
            n_tiles = 0
            n_tile_dets = 0
            for y0 in range(0, h, stride):
                for x0 in range(0, w, stride):
                    x1 = min(x0 + sahi_slice_size, w)
                    y1 = min(y0 + sahi_slice_size, h)
                    if (x1 - x0) < sahi_slice_size // 2 or (y1 - y0) < sahi_slice_size // 2:
                        continue
                    tile = frame_rgb[y0:y1, x0:x1]
                    tile_img = Image.fromarray(tile)
                    tile_dets = self._detect_single(
                        tile_img, query, box_threshold, text_threshold,
                        offset_x=x0, offset_y=y0)
                    detections.extend(tile_dets)
                    n_tiles += 1
                    n_tile_dets += len(tile_dets)

            if n_tile_dets > 0:
                # NMS across full-frame + tile detections
                detections.sort(key=lambda d: d[1], reverse=True)
                kept = []
                for det in detections:
                    if all(self._iou_xywh(det[0], k[0]) < 0.5 for k in kept):
                        kept.append(det)
                if len(kept) < len(detections):
                    print(f"  [SAHI] {n_tiles} tiles -> {n_tile_dets} extra dets, "
                          f"NMS merged {len(detections)} -> {len(kept)}")
                detections = kept

        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    @staticmethod
    def _iou_xywh(a: List[int], b: List[int]) -> float:
        """IoU between two [x, y, w, h] boxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def best_boxes(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        clip_model=None,
        clip_preprocess=None,
        text_feat: torch.Tensor = None,
        max_area_frac: float = 0.25,
        top_k: int = 5,
        nms_iou: float = 0.5,
        use_sahi: bool = False,
    ) -> List[Tuple[List[int], float, float, str]]:
        """
        Multi-instance detection: up to top_k detections per frame as
        (bbox_xywh, score, gdino_score, label), greedy-NMS deduplicated,
        sorted by score descending.

        score is the CLIP crop similarity when clip_model + text_feat are
        provided (so callers can threshold uniformly), else GDino confidence.

        max_area_frac: reject detections larger than this fraction of the
        frame (prevents whole-counter / scene-level false positives).
        """
        detections = self.detect(frame_rgb, text_query, box_threshold,
                                 text_threshold, use_sahi=use_sahi)
        if not detections:
            return []

        # Size filter — a single object shouldn't cover more than
        # max_area_frac of the frame
        frame_h, frame_w = frame_rgb.shape[:2]
        frame_area = frame_h * frame_w
        sized = [d for d in detections
                 if (d[0][2] * d[0][3]) / frame_area <= max_area_frac]
        n_filtered = len(detections) - len(sized)
        if n_filtered > 0:
            print(f"  [size filter] removed {n_filtered} oversized detection(s) "
                  f"(>{max_area_frac*100:.0f}% of frame)")
        # If EVERY detection is oversized, return nothing rather than the
        # rejected giants — a box covering >max_area_frac of an egocentric
        # frame is a scene-level false positive, and surfacing it poisons the
        # crop score and area evidence. Empty -> evidence collapses ->
        # abstention decides (the correct outcome for an absent object).
        detections = sized
        if not detections:
            return []

        # Score every crop with CLIP (if available) so ranking is semantic
        if clip_model is not None and text_feat is not None:
            import torch.nn.functional as F

            crops = []
            for bbox, _, _ in detections:
                x, y, w, h = bbox
                crop = frame_rgb[y:y+h, x:x+w]
                crops.append(None if crop.size == 0
                             else clip_preprocess(Image.fromarray(crop)))
            valid = [(i, c) for i, c in enumerate(crops) if c is not None]
            if valid:
                batch = torch.stack([c for _, c in valid]).to(self.device)
                with torch.no_grad(), torch.autocast(
                    self.device.type, dtype=torch.bfloat16,
                    enabled=self.device.type == 'cuda'
                ):
                    feats = clip_model.encode_image(batch).float()
                feats = F.normalize(feats, p=2, dim=-1)
                clip_sims = (feats @ text_feat.T).squeeze(-1).cpu()
                scored = [
                    (detections[i][0], float(clip_sims[j]),
                     detections[i][1], detections[i][2])
                    for j, (i, _) in enumerate(valid)
                ]
            else:
                scored = [(b, s, s, l) for b, s, l in detections]
        else:
            scored = [(b, s, s, l) for b, s, l in detections]

        # Greedy NMS on the primary score, then cut to top_k
        scored.sort(key=lambda d: d[1], reverse=True)
        kept = []
        for det in scored:
            if all(self._iou_xywh(det[0], k[0]) < nms_iou for k in kept):
                kept.append(det)
            if len(kept) >= top_k:
                break

        if clip_model is not None and text_feat is not None and kept:
            print(f"  CLIP re-ranked {len(detections)} detections: "
                  f"kept {len(kept)} instance(s), "
                  f"top CLIP={kept[0][1]:.3f} (GDino={kept[0][2]:.3f})")
        return kept

    def best_box(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        clip_model=None,
        clip_preprocess=None,
        text_feat: torch.Tensor = None,
        max_area_frac: float = 0.25,
    ) -> Optional[Tuple[List[int], float]]:
        """
        Single best detection as ([x, y, w, h], score).
        Thin top-1 wrapper around best_boxes() — see there for semantics.
        """
        boxes = self.best_boxes(
            frame_rgb, text_query, box_threshold, text_threshold,
            clip_model, clip_preprocess, text_feat, max_area_frac,
            top_k=1,
        )
        if not boxes:
            return None
        bbox, score, _gdino, _label = boxes[0]
        return bbox, score
