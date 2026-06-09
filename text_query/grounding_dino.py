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

    def best_box(
        self,
        frame_rgb: np.ndarray,
        text_query: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        clip_model=None,
        clip_preprocess=None,
        text_feat: torch.Tensor = None,
    ) -> Optional[Tuple[List[int], float]]:
        """
        Return the single best detection as ([x, y, w, h], confidence).

        When clip_model + text_feat are provided and multiple detections exist,
        each crop is scored with CLIP against the full text query to pick the
        best semantic match (not just the highest Grounding DINO confidence).
        """
        detections = self.detect(frame_rgb, text_query, box_threshold, text_threshold)
        if not detections:
            return None

        if clip_model is None or text_feat is None:
            bbox, score, label = detections[0]
            return bbox, score

        # CLIP re-ranks all detections (even a single one) so the returned
        # score is always a CLIP similarity — callers can threshold uniformly.
        import torch.nn.functional as F

        crops = []
        for bbox, _, _ in detections:
            x, y, w, h = bbox
            crop = frame_rgb[y:y+h, x:x+w]
            if crop.size == 0:
                crops.append(None)
                continue
            crops.append(clip_preprocess(Image.fromarray(crop)))

        valid = [(i, c) for i, c in enumerate(crops) if c is not None]
        if not valid:
            bbox, score, _ = detections[0]
            return bbox, score

        batch = torch.stack([c for _, c in valid]).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            feats = clip_model.encode_image(batch).float()
        feats = F.normalize(feats, p=2, dim=-1)
        clip_sims = (feats @ text_feat.T).squeeze(-1).cpu()

        best_valid_idx = int(clip_sims.argmax())
        best_det_idx = valid[best_valid_idx][0]
        bbox, gdino_score, label = detections[best_det_idx]
        clip_score = float(clip_sims[best_valid_idx])

        n = len(detections)
        print(f"  CLIP re-ranked {n} detections: "
              f"picked #{best_det_idx+1}/{n} (CLIP={clip_score:.3f}, GDino={gdino_score:.3f})")

        return bbox, clip_score
