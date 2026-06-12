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

    def detect(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> List[Tuple[List[int], float, str]]:
        """
        Detect objects matching text_query in a frame.

        Returns list of (bbox_xywh, confidence, label) sorted by confidence desc.
        bbox_xywh is [x, y, w, h] in pixel coordinates.
        """
        self._load()
        self._model.to(self.device)

        image = Image.fromarray(frame_rgb)
        # Grounding DINO expects the query to end with a period
        query = text_query.strip().rstrip(".") + "."

        inputs = self._processor(images=image, text=query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        h, w = frame_rgb.shape[:2]
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(score), label))

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
        detections = self.detect(frame_rgb, text_query, box_threshold, text_threshold)
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
        detections = sized if sized else detections  # keep originals as fallback

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
