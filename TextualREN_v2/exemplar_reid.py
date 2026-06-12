"""
Exemplar bootstrapping + episodic re-ID (Contribution A).

"Language opens the door, regions remember": text grounding is category-level
("a mug") and degrades under blur, viewpoint change, and long absences.
Instance-level visual features survive all three — but no user can provide a
visual exemplar of their lost object. So language grounds the object ONCE,
and the verified detection bootstraps an instance-specific exemplar bank
(DINOv2 features over multiple views) that sweeps the whole CLIP index for
every OTHER occurrence of the same object.

Pipeline position: runs after the primary response is accepted; its results
feed the s_reid evidence feature and add re-ID occurrences to the output.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ExemplarReID:
    """Builds an exemplar bank from a verified detection and re-IDs it
    across the indexed video."""

    def __init__(self, clip_model, clip_preprocess, device=None):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._dino = None          # lazy DINOv2 ViT-L/14
        self._dino_failed = False

    # ------------------------------------------------------------------ #
    # Embedding backends                                                   #
    # ------------------------------------------------------------------ #

    def _load_dino(self):
        """Lazy-load DINOv2 — instance-discriminative features (CLIP is
        category-level; two different mugs look identical to it)."""
        if self._dino is not None or self._dino_failed:
            return
        try:
            print("  [ReID] Loading DINOv2 ViT-L/14 ...")
            self._dino = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitl14')
            self._dino.to(self.device).eval()
        except Exception as e:
            print(f"  [ReID] DINOv2 load failed ({e}); "
                  f"falling back to CLIP features for verification")
            self._dino_failed = True

    @staticmethod
    def _dino_preprocess(crop_rgb: np.ndarray) -> torch.Tensor:
        """Resize to 224 and ImageNet-normalize (DINOv2 convention)."""
        import torchvision.transforms as T
        tf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        return tf(Image.fromarray(crop_rgb))

    def embed_clip(self, crops: list) -> torch.Tensor:
        """(N, D) L2-normalized CLIP image embeddings."""
        batch = torch.stack([
            self.clip_preprocess(Image.fromarray(c)) for c in crops
        ]).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == 'cuda'
        ):
            feats = self.clip_model.encode_image(batch).float()
        return F.normalize(feats, p=2, dim=-1)

    def embed_instance(self, crops: list) -> torch.Tensor:
        """(N, D) L2-normalized instance features: DINOv2 if available,
        CLIP otherwise."""
        self._load_dino()
        if self._dino is None:
            return self.embed_clip(crops)
        batch = torch.stack([self._dino_preprocess(c) for c in crops]
                            ).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=self.device.type == 'cuda'
        ):
            feats = self._dino(batch).float()
        return F.normalize(feats, p=2, dim=-1)

    # ------------------------------------------------------------------ #
    # Exemplar bank                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _crop(frame: np.ndarray, bbox, scale: float = 1.0):
        """Crop bbox from frame with optional context scaling."""
        h, w = frame.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        if scale != 1.0:
            cx, cy = x + bw / 2, y + bh / 2
            bw2, bh2 = bw * scale, bh * scale
            x, y = cx - bw2 / 2, cy - bh2 / 2
            bw, bh = bw2, bh2
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2]

    def build_bank(self, frames: list, global_to_local: dict,
                   seed_local_idx: int, seed_bbox, track: list,
                   max_views: int = 8) -> dict:
        """
        Harvest exemplar views and embed them.

        Views = the seed detection crop (1.0x and 1.3x context) plus crops
        from SAM2-tracked frames sampled evenly across the track — free
        temporal augmentation: the tracker already followed the object
        through viewpoint and blur changes.
        """
        crops = []
        for scale in (1.0, 1.3):
            c = self._crop(frames[seed_local_idx], seed_bbox, scale)
            if c is not None:
                crops.append(c)

        if track:
            step = max(1, len(track) // max(1, max_views - len(crops)))
            for entry in track[::step]:
                local = global_to_local.get(entry['frame_idx'])
                if local is None:
                    continue
                c = self._crop(frames[local], entry['bbox'])
                if c is not None:
                    crops.append(c)
                if len(crops) >= max_views:
                    break

        if not crops:
            return None
        return {
            'clip': self.embed_clip(crops),          # retrieval (index space)
            'instance': self.embed_instance(crops),  # verification (DINOv2)
            'num_views': len(crops),
        }

    # ------------------------------------------------------------------ #
    # Re-ID sweep                                                          #
    # ------------------------------------------------------------------ #

    def candidate_rows(self, bank: dict, clip_embeddings: np.ndarray,
                       exclude_rows: set, top_k: int = 50) -> list:
        """
        Sweep the CLIP index with the exemplar bank: per index row, take the
        max similarity over views; return top rows outside excluded regions,
        grouped so neighbouring rows don't produce duplicate candidates.
        """
        bank_np = bank['clip'].cpu().numpy().astype(np.float32)
        sims = clip_embeddings @ bank_np.T            # (N_rows, n_views)
        row_scores = sims.max(axis=1)

        order = np.argsort(-row_scores)
        picked, last_row = [], None
        for r in order[: top_k * 4]:
            if int(r) in exclude_rows:
                continue
            # collapse near-duplicate rows (within ~5 index rows)
            if any(abs(int(r) - p[0]) < 5 for p in picked):
                continue
            picked.append((int(r), float(row_scores[r])))
            if len(picked) >= top_k:
                break
        return picked

    def verify_candidate(self, frame_rgb: np.ndarray, detections: list,
                         bank: dict, sim_threshold: float = 0.6):
        """
        Score each detection crop against the exemplar bank with instance
        features. Returns (bbox, s_reid) of the best match above threshold,
        else None.
        """
        crops, boxes = [], []
        for det in detections:
            bbox = det[0] if isinstance(det, tuple) else det
            c = self._crop(frame_rgb, bbox)
            if c is not None:
                crops.append(c)
                boxes.append([int(v) for v in bbox])
        if not crops:
            return None

        feats = self.embed_instance(crops)            # (M, D)
        sims = feats @ bank['instance'].T              # (M, n_views)
        best_per_box = sims.max(dim=1).values
        best = int(best_per_box.argmax())
        s = float(best_per_box[best])
        if s < sim_threshold:
            return None
        return boxes[best], s
