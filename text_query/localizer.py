"""
TextQueryLocalizer — episodic memory localization from natural language.

Pipeline
--------
1. CLIP ViT-g-14 coarse retrieval : text → frame similarity → last matching frame
2. REN (DINOv2 ViT-L/14) fine localization : region tokens on candidate window
3. SAM2 bbox : point → segmentation mask → bounding box
4. SAM2 tracking : forward + backward propagation from last-occurrence frame
5. Clip export : trimmed mp4 with bbox overlay
"""

import os
import sys
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip

# Resolve sibling paths regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_VQ   = os.path.join(_ROOT, 'visual_query')
_SAM  = os.path.join(_ROOT, 'segment_anything')

for _p in (_ROOT, _VQ, _SAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import REN                                          # visual_query/models.py
from vq_utils import get_sam_region_from_points, mask_to_bbox  # visual_query/vq_utils.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------- #

class TextQueryLocalizer:
    """Localize the last occurrence of a text-described object in a video."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(device)
        self.sam2_ckpt    = config['data']['sam2_ckpt']
        self.tracker_param = config['text_query']['tracker_param']
        self.patch_size   = config['text_query']['patch_size']

        print('[TextQueryLocalizer] Loading OpenCLIP ViT-g-14 (laion2b)…')
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-g-14', pretrained='laion2b_s34b_b88k', device=device
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')

        # REN loaded lazily on first use — indexing never needs it,
        # so this keeps ~3–4 GB VRAM free during prepare_index.py.
        self._ren = None

    @property
    def ren(self):
        if self._ren is None:
            print('[TextQueryLocalizer] Loading REN (DINOv2 ViT-L/14)…')
            self._ren = REN(self.config['ren'])
        return self._ren

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def encode_text(self, query: str) -> torch.Tensor:
        """Returns L2-normalised CLIP text embedding, shape (1, D)."""
        tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            feat = self.clip_model.encode_text(tokens).float()
        return F.normalize(feat, p=2, dim=-1)

    def encode_frames_clip(self, frames: list, batch_size: int = 32) -> torch.Tensor:
        """Returns stacked L2-normalised CLIP image embeddings, shape (N, D)."""
        all_feats = []
        for i in tqdm(range(0, len(frames), batch_size), desc='  CLIP frames', leave=False):
            imgs = torch.stack([
                self.clip_preprocess(Image.fromarray(f)) for f in frames[i:i + batch_size]
            ]).to(self.device)
            with torch.no_grad(), torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
            ):
                feats = self.clip_model.encode_image(imgs).float()
            all_feats.append(F.normalize(feats, p=2, dim=-1).cpu())
        return torch.cat(all_feats)  # (N, D)

    # ------------------------------------------------------------------ #
    # Coarse retrieval                                                     #
    # ------------------------------------------------------------------ #

    def find_last_occurrence(self, text_feat: torch.Tensor,
                              frame_feats: torch.Tensor,
                              threshold: float) -> tuple:
        """
        Returns (sampled_idx, score, similarities).
        sampled_idx is an index into the subsampled frame list.
        'Last occurrence' = the highest-indexed frame that exceeds the threshold.
        Falls back to argmax if nothing clears the threshold.
        """
        sims = (frame_feats @ text_feat.cpu().T).squeeze(-1)  # (N,)
        above = (sims > threshold).nonzero(as_tuple=True)[0]
        idx = above[-1].item() if len(above) > 0 else sims.argmax().item()
        return idx, sims[idx].item(), sims

    def find_last_occurrence_temporal(
        self,
        sims: torch.Tensor,
        frame_metadata: list,
        fps: float,
        sample_rate: int,
        threshold: float,
    ) -> tuple:
        """
        Temporal segmentation over similarity scores.
        Returns (sampled_idx, score, sims).
        """
        above = np.where(sims.numpy() >= threshold)[0]
        if len(above) == 0:
            idx = int(sims.argmax())
            return idx, float(sims[idx]), sims

        sorted_above = sorted(above.tolist(),
                              key=lambda i: frame_metadata[i]['frame_idx'])
        gap_frames = max(1, int(2.0 * fps / sample_rate)) * sample_rate

        segments = []
        current = [sorted_above[0]]
        for idx in sorted_above[1:]:
            prev_vidx = frame_metadata[current[-1]]['frame_idx']
            curr_vidx = frame_metadata[idx]['frame_idx']
            if curr_vidx - prev_vidx > gap_frames:
                segments.append(current)
                current = [idx]
            else:
                current.append(idx)
        segments.append(current)

        valid = [s for s in segments if len(s) >= 2] or segments
        last_segment = valid[-1]
        best_idx = max(last_segment, key=lambda i: sims[i])
        return int(best_idx), float(sims[best_idx]), sims

    # ------------------------------------------------------------------ #
    # Region localization (REN + CLIP bridge)                             #
    # ------------------------------------------------------------------ #

    def find_best_region(self, frame: np.ndarray,
                          region_tokens: torch.Tensor,
                          local_frame_idx: int,
                          text_feat: torch.Tensor) -> tuple:
        """
        Among REN region tokens for local_frame_idx, score each region crop with CLIP.
        Returns (best_point [y,x], best_clip_score).
        """
        h, w = frame.shape[:2]

        if region_tokens is None or region_tokens.numel() == 0:
            return np.array([h / 2.0, w / 2.0], dtype=np.float32), 0.0

        tokens = region_tokens[local_frame_idx]
        grid_points = self.ren.grid_points.cpu().numpy()
        img_res = self.config['ren']['parameters']['image_resolution']
        scale_y = h / float(img_res)
        scale_x = w / float(img_res)
        max_regions = min(tokens.shape[0], grid_points.shape[0])

        patch_r = max(32, min(h, w) // 8)
        best_score, best_point = -1.0, None

        for i in range(max_regions):
            py, px = grid_points[i]
            py = int(py * scale_y)
            px = int(px * scale_x)
            y1, y2 = max(0, py - patch_r), min(h, py + patch_r)
            x1, x2 = max(0, px - patch_r), min(w, px + patch_r)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            img = self.clip_preprocess(Image.fromarray(patch)).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
            ):
                pfeat = F.normalize(self.clip_model.encode_image(img).float(), p=2, dim=-1)

            score = (pfeat @ text_feat.T).item()
            if score > best_score:
                best_score = score
                best_point = np.array([float(py), float(px)], dtype=np.float32)

        if best_point is None:
            best_point = np.array([h / 2.0, w / 2.0], dtype=np.float32)

        return best_point, best_score

    # ------------------------------------------------------------------ #
    # SAM2 bounding box from a point                                       #
    # ------------------------------------------------------------------ #

    def point_to_bbox(self, frame: np.ndarray,
                       point: np.ndarray,
                       text_feat: torch.Tensor,
                       max_area_frac: float = 0.12) -> torch.Tensor:
        """
        Runs SAM2 multi-mask prediction at the given point.
        Picks the mask whose crop is most similar to the text query via CLIP.
        Returns bbox as [x, y, w, h] tensor.

        max_area_frac: upper bound on mask area as fraction of frame.
        Pass a larger value (e.g. 0.60) for large-area objects like paintings.
        """
        masks_list = get_sam_region_from_points(
            self.sam2_ckpt, self.tracker_param, [frame], [point]
        )
        h, w = frame.shape[:2]
        fallback = torch.tensor([
            max(0, int(point[1]) - w // 8),
            max(0, int(point[0]) - h // 8),
            w // 4, h // 4,
        ])

        if not masks_list or not masks_list[0]:
            return fallback

        h, w = frame.shape[:2]
        frame_area = h * w
        MIN_AREA_FRAC = 0.001   # reject masks < 0.1% of frame (noise)
        MAX_AREA_FRAC = max_area_frac

        candidates = []
        for mask in masks_list[0]:
            rows, cols = np.where(mask == 1)
            if len(rows) == 0:
                continue
            area_frac = len(rows) / frame_area
            y1, y2, x1, x2 = rows.min(), rows.max(), cols.min(), cols.max()
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            img = self.clip_preprocess(Image.fromarray(patch)).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
            ):
                mfeat = F.normalize(self.clip_model.encode_image(img).float(), p=2, dim=-1)
            score = (mfeat @ text_feat.T).item()
            bbox = torch.tensor([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            candidates.append((score, area_frac, bbox))

        if not candidates:
            return fallback

        # Prefer masks within the acceptable size range; fall back to all candidates
        sized = [(s, a, b) for s, a, b in candidates if MIN_AREA_FRAC <= a <= MAX_AREA_FRAC]
        pool = sized if sized else candidates
        best_score, _, best_bbox = max(pool, key=lambda t: t[0])
        return best_bbox

    # ------------------------------------------------------------------ #
    # SAM2 tracking                                                        #
    # ------------------------------------------------------------------ #

    def track_from_bbox(self, frames: list,
                         frame_idx: int,
                         bbox: torch.Tensor,
                         half_span: int = 150) -> list:
        """
        Runs SAM2 video predictor forward from frame_idx within a ±half_span window.
        Returns list of {'frame_idx': int, 'bbox': [x,y,w,h]}.
        """
        from sam2.build_sam import build_sam2_video_predictor

        transform = T.Compose([
            T.ToTensor(),
            T.Resize((1024, 1024), antialias=True),
            lambda x: x.unsqueeze(0),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        start = max(0, frame_idx - half_span)
        end   = min(len(frames), frame_idx + half_span + 1)
        window = frames[start:end]

        transformed = torch.cat([transform(np.array(f)) for f in window]).to(self.device)
        h, w = np.array(window[0]).shape[:2]
        local_idx = frame_idx - start

        x, y, bw, bh = [float(v) for v in bbox]
        tracker = build_sam2_video_predictor(self.tracker_param, self.sam2_ckpt, device=self.device)
        track = []

        with torch.inference_mode(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'
        ):
            state = tracker.init_state(images=transformed, video_height=h, video_width=w)
            tracker.reset_state(state)
            tracker.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_idx,
                obj_id=0,
                box=[x, y, x + bw, y + bh],
            )
            for fid, _, mask_logits in tracker.propagate_in_video(state):
                mask = (mask_logits[0] > 0.0)[0].cpu().numpy()
                if mask.sum() == 0:
                    continue
                rows, cols = np.where(mask)
                track.append({
                    'frame_idx': start + fid,
                    'bbox': [int(cols.min()), int(rows.min()),
                              int(cols.max() - cols.min()), int(rows.max() - rows.min())],
                })

        torch.cuda.empty_cache()
        return track

    # ------------------------------------------------------------------ #
    # Clip export                                                          #
    # ------------------------------------------------------------------ #

    def export_clip(self, video_path: str,
                     track: list,
                     last_idx: int,
                     fps: float,
                     output_path: str,
                     context_seconds: float = 5.0):
        frame_to_bbox = {t['frame_idx']: t['bbox'] for t in track}

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ctx   = int(context_seconds * fps)
        start = max(0, last_idx - ctx)
        end   = min(total, last_idx + ctx + 1)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_to_bbox:
                bx, by, bw, bh = frame_to_bbox[i]
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            if i == last_idx:
                cv2.putText(frame, 'LAST OCCURRENCE', (10, 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            out.write(frame)

        cap.release()
        out.release()
        print(f'  Clip saved: {output_path}  ({end - start} frames, {(end - start) / fps:.1f}s)')

    # ------------------------------------------------------------------ #
    # Main API                                                             #
    # ------------------------------------------------------------------ #

    def localize(self, query_text: str,
                  video_path: str,
                  output_dir: str = 'output') -> dict:
        os.makedirs(output_dir, exist_ok=True)
        cfg     = self.config['text_query']
        thr     = cfg['similarity_threshold']
        ctx_sec = cfg['context_seconds']
        srate   = cfg.get('frame_sample_rate', 1)

        # ---- Step 1 : get video metadata (don't load all frames yet) ------
        print(f'\n[1/6] Opening video: {video_path}')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'      {total_frames} frames @ {fps:.2f} fps')

        # Stream frames in batches for CLIP (memory-efficient)
        sampled_frames, sampled_indices, all_sims = [], [], []

        # ---- Step 2 : encode text query -----------------------------------
        print(f'[2/6] Encoding text query: "{query_text}"')
        text_feat = self.encode_text(query_text)

        # ---- Step 3 : CLIP coarse retrieval (stream frames) ----------------
        print(f'[3/6] CLIP coarse retrieval (streaming frames, sample_rate={srate})…')
        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % srate == 0:
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sampled_indices.append(frame_idx)
            frame_idx += 1

        frame_feats = self.encode_frames_clip(sampled_frames)
        frame_metadata = [{'frame_idx': idx} for idx in sampled_indices]
        sims = (frame_feats @ text_feat.cpu().T).squeeze(-1)
        sampled_idx, clip_score, sims = self.find_last_occurrence_temporal(
            sims, frame_metadata, fps, srate, thr
        )
        last_idx = sampled_indices[sampled_idx]
        print(f'      Last occurrence → frame {last_idx}  '
              f'(t={last_idx / fps:.2f}s,  CLIP score={clip_score:.4f})')
        self._save_similarity_plot(sims, sampled_idx, sampled_indices, fps, output_dir)

        # ---- Step 4 : REN fine localization on candidate window -----------
        print('[4/6] REN region encoding on candidate window…')
        win_size   = 5  # frames before last_idx fed to REN
        win_start  = max(0, sampled_idx - win_size)
        img_res = self.config['ren']['parameters']['image_resolution']
        ren_transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_res, img_res), antialias=True),
        ])
        frame_tensors = torch.stack([
            ren_transform(Image.fromarray(f))
            for f in sampled_frames[win_start:sampled_idx + 1]
        ]).to(self.device)
        region_tokens = self.ren(frame_tensors)

        # Map local key → absolute sampled index
        best_point, region_score = self.find_best_region(
            sampled_frames[sampled_idx], region_tokens, sampled_idx - win_start, text_feat
        )
        print(f'      Best region point: (y={best_point[0]:.1f}, x={best_point[1]:.1f})  '
              f'score={region_score:.4f}')

        # ---- Step 5 : SAM2 bounding box -----------------------------------
        print('[5/6] SAM2 bounding box…')
        bbox = self.point_to_bbox(sampled_frames[sampled_idx], best_point, text_feat)
        print(f'      bbox [x,y,w,h]: {bbox.tolist()}')

        # ---- Step 6 : SAM2 tracking + clip export -------------------------
        print('[6/6] SAM2 tracking + clip export…')
        half_span = int(ctx_sec * fps)
        # Load frames for tracking (window around last_idx)
        track_frames = self._load_frame_window(video_path, last_idx, half_span)
        track = self.track_from_bbox(track_frames, last_idx - (max(0, last_idx - half_span)),
                                      bbox, half_span=half_span)
        print(f'      Tracked {len(track)} frames')

        clip_path = os.path.join(output_dir, 'last_occurrence.mp4')
        self.export_clip(video_path, track, last_idx, fps, clip_path, ctx_sec)
        cap.release()

        # Save metadata
        result = {
            'query':                  query_text,
            'video':                  os.path.abspath(video_path),
            'last_occurrence_frame':  last_idx,
            'last_occurrence_time_s': round(last_idx / fps, 3),
            'clip_similarity_score':  round(clip_score, 4),
            'region_score':           round(region_score, 4),
            'bbox_xywh':              bbox.tolist(),
            'clip_output':            os.path.abspath(clip_path),
            'tracked_frames':         len(track),
        }
        meta_path = os.path.join(output_dir, 'result.json')
        with open(meta_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Metadata saved: {meta_path}')
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load_frame_window(self, video_path: str, center_idx: int, half_span: int) -> list:
        """Load frames in a window [center_idx - half_span, center_idx + half_span]."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, center_idx - half_span)
        end = min(total, center_idx + half_span + 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _save_similarity_plot(self, sims: torch.Tensor,
                               last_sampled_idx: int,
                               sampled_indices: list,
                               fps: float,
                               output_dir: str):
        times = [i / fps for i in sampled_indices]
        last_t = sampled_indices[last_sampled_idx] / fps
        plt.figure(figsize=(14, 3))
        plt.plot(times, sims.numpy(), linewidth=0.8, color='steelblue', alpha=0.85)
        plt.axvline(x=last_t, color='red', linestyle='--', linewidth=1.5,
                    label=f'last occurrence  t={last_t:.2f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('CLIP similarity')
        plt.title('Text-video similarity over time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity.png'), dpi=120)
        plt.close()
