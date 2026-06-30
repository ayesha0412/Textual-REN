"""
EgoMask adapter — runs our text-VQL pipeline on EgoMask samples and writes
predictions in EgoMask's per-video JSON format, ready for their official
eval script.

Two modes:

  --full-video  (recommended for EgoMask)
    Phase 1: Pipeline detection (CLIP retrieval + Grounding DINO + LLM
             query parsing + calibrated abstention). ~1 min/sample.
    Phase 2: SAM2 propagation from the primary detection.
    Phase 3: Adaptive recovery — if coverage < 15%, scan additional frames
             with Grounding DINO, add size-consistent seeds, re-propagate.
             Only ~30% of samples trigger recovery, keeping precision high.

  default (windowed tracking)
    Uses our pipeline's built-in SAM2 windowed tracking.  Lower temporal
    coverage but useful for ablation.

Frame index mapping (the critical detail):
  EgoMask frame name e.g. "00045" → integer 45 → source frame 45 * 6 = 270.
  JPEG frames live at: dataset/egomask/JPEGImages/egotracks/<vid>/00045.jpg

Prerequisites:
  - CLIP (or SigLIP) indexes for every clip in the tier
  - EgoMask annotation directory populated
  - EgoTracks clips + extracted JPEG frames on disk

Usage:
  # Full-video propagation (recommended)
  python eval/egomask_adapter.py --tier long --full-video
  # Smoke-test
  python eval/egomask_adapter.py --tier long --full-video --limit 1
  # Windowed tracking (ablation)
  python eval/egomask_adapter.py --tier long --no-reid
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "segment_anything"))
sys.path.insert(0, str(PROJECT_ROOT / "TextualREN_v2"))
EVAL_DIR     = PROJECT_ROOT / "eval"
TEXTUAL_DIR  = PROJECT_ROOT / "TextualREN_v2"
EGOMASK_ROOT = EVAL_DIR / "EgoMask" / "dataset" / "egomask"
JPEG_ROOT    = EGOMASK_ROOT / "JPEGImages" / "egotracks"
DEFAULT_CLIP_DIR  = EVAL_DIR / "EgoMask" / "dataset" / "tmp" / "ego4d" / "v2" / "clips"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "epic_kitchen_indexes"

EGOMASK_FRAME_STRIDE = 6
NEAREST_FRAME_TOLERANCE_SRC = 15


# ─── Helpers ───────────────────────────────────────────────────────────

def gt_key_to_source_frame(key: str) -> int:
    return int(key) * EGOMASK_FRAME_STRIDE


def encoder_suffix(config_path: Path) -> str:
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enc = cfg.get("text_query", {}).get("encoder", "clip")
    return "v2" if enc == "clip" else enc


def find_nearest_track_mask(target_src_frame: int, tracks: list,
                             tol: int = NEAREST_FRAME_TOLERANCE_SRC):
    best, best_dist = None, tol + 1
    for tr in tracks:
        for f in tr.get("frames", []):
            d = abs(int(f["frame_idx"]) - target_src_frame)
            m = f.get("mask") or f.get("mask_rle")
            if d < best_dist and m is not None:
                best_dist = d
                best = m
    return best


def get_clip_dimensions(clip_path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return [H, W]


def source_frame_to_jpeg_name(src_frame: int, available_names: list) -> str:
    """Map a 30fps source frame index to the nearest EgoMask JPEG frame name."""
    five_fps_idx = src_frame / EGOMASK_FRAME_STRIDE
    best_name, best_dist = None, float('inf')
    for name in available_names:
        d = abs(int(name) - five_fps_idx)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


# ─── Detection-only pipeline call ─────────────────────────────────────

def run_detection(video_uid: str, expression: str, obj_id: str,
                  expr_idx: int, clip_dir: Path, index_dir: Path,
                  enc_suffix: str, config_path: Path,
                  work_dir: Path) -> dict:
    """
    Run our full text-VQL pipeline in detection-only mode:
    CLIP retrieval → candidate verification → Grounding DINO bbox.
    No SAM2 tracking, no re-ID — just find the object and return its
    location. This is where our pipeline's advantage lives.
    """
    clip_path  = clip_dir / f"{video_uid}.mp4"
    index_path = index_dir / f"{video_uid}_{enc_suffix}"
    if not clip_path.exists():
        return {"_error": f"clip missing: {clip_path}"}
    if not index_path.exists():
        return {"_error": f"index missing: {index_path}"}

    qdir = work_dir / f"{video_uid}__exp{expr_idx}__obj{obj_id}"
    qdir.mkdir(parents=True, exist_ok=True)
    full_json = qdir / "result.json"

    if full_json.exists():
        try:
            out = json.loads(full_json.read_text(encoding="utf-8"))
            if out.get("decision") in ("found", "not_found", "uncertain"):
                out["_wall_detect_s"] = 0
                out["_cached"] = True
                return out
        except (json.JSONDecodeError, KeyError):
            pass

    cmd = [
        sys.executable, "query_indexed.py", expression,
        "--index",    str(index_path),
        "--video",    str(clip_path),
        "--config",   str(config_path),
        "--output",   str(qdir),
        "--json-out", str(full_json),
        "--skip-tracking",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(TEXTUAL_DIR),
                           capture_output=True, text=True)
    wall = round(time.time() - t0, 1)

    if proc.returncode != 0:
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        (qdir / "stdout.txt").write_text(proc.stdout[-8000:], encoding="utf-8")
        return {"_error": f"subprocess exit {proc.returncode}", "_wall_s": wall}

    if not full_json.exists():
        (qdir / "stdout.txt").write_text(proc.stdout[-8000:], encoding="utf-8")
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        return {"_error": f"result.json not produced (logs in {qdir})", "_wall_s": wall}

    try:
        out = json.loads(full_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"result.json bad: {e}", "_wall_s": wall}

    out["_wall_detect_s"] = wall
    return out


# ─── Multi-seed detection ─────────────────────────────────────────────

_gdino = None  # reuse across expressions

def detect_additional_seeds(jpeg_dir: Path, frame_names: list,
                             detection_prompt: str,
                             primary_frame_name: str,
                             primary_bbox_xywh: list,
                             min_seed_gap: int = 30,
                             sample_stride: int = 15,
                             max_seeds: int = 3,
                             box_threshold: float = 0.40,
                             size_ratio_max: float = 3.0) -> list:
    """
    Sample frames across the video and run Grounding DINO to find
    additional detection seeds for SAM2 propagation.

    Seeds are filtered by confidence, temporal spread, and bbox size
    consistency with the primary detection (rejects false positives
    that are drastically different in scale).

    Returns list of (frame_name, bbox_xywh, confidence) sorted by
    confidence, excluding seeds too close to existing ones.
    """
    import numpy as np
    from PIL import Image as PILImage
    from grounding_dino import GroundingDINOLocalizer

    global _gdino
    if _gdino is None:
        _gdino = GroundingDINOLocalizer()

    primary_idx = frame_names.index(primary_frame_name)
    primary_area = primary_bbox_xywh[2] * primary_bbox_xywh[3]

    sample_indices = list(range(0, len(frame_names), sample_stride))
    sample_indices = [i for i in sample_indices
                      if abs(i - primary_idx) >= min_seed_gap]

    if not sample_indices:
        return []

    candidates = []
    for idx in sample_indices:
        fname = frame_names[idx]
        jpg_path = jpeg_dir / f"{fname}.jpg"
        if not jpg_path.exists():
            continue
        img = np.array(PILImage.open(jpg_path).convert('RGB'))
        dets = _gdino.detect(img, detection_prompt,
                             box_threshold=box_threshold, text_threshold=0.25)
        if dets:
            best_bbox, best_conf, best_label = dets[0]
            seed_area = best_bbox[2] * best_bbox[3]
            ratio = max(seed_area, primary_area) / max(min(seed_area, primary_area), 1)
            if ratio <= size_ratio_max:
                candidates.append((fname, best_bbox, best_conf, idx))

    candidates.sort(key=lambda c: c[2], reverse=True)

    seeds = []
    used_indices = [primary_idx]
    for fname, bbox, conf, idx in candidates:
        if all(abs(idx - ui) >= min_seed_gap for ui in used_indices):
            seeds.append((fname, bbox, conf))
            used_indices.append(idx)
            if len(seeds) >= max_seeds:
                break

    return seeds


# ─── Full-video SAM2 propagation ──────────────────────────────────────

_sam2_predictor = None  # reuse across expressions

def fullvideo_sam2_propagate(jpeg_dir: Path, frame_names: list,
                              seeds: list,
                              sam2_cfg: str, sam2_ckpt: str) -> dict:
    """
    Propagate SAM2 from one or more detection seeds through ALL video frames.

    Multi-seed propagation: when SAM2 loses an object after the primary
    detection (occlusion, camera motion), additional seeds at other time
    points re-anchor the tracker. SAM2 conditions on all prompts during
    propagation, so a seed in the middle of a tracking gap recovers the mask.

    Args:
        jpeg_dir:   Directory of JPEG frames (00000.jpg, 00005.jpg, ...)
        frame_names: Sorted list of frame name strings without .jpg
        seeds:      List of (frame_name, bbox_xywh) tuples — at least one
        sam2_cfg:   SAM2 config file path
        sam2_ckpt:  SAM2 checkpoint path

    Returns:
        dict: {frame_name: {'size': [H,W], 'counts': rle_string}, ...}
    """
    import torch
    import numpy as np
    from pycocotools import mask as mask_utils
    from sam2.build_sam import build_sam2_video_predictor

    global _sam2_predictor
    if _sam2_predictor is None:
        _sam2_predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt)

    jpg_files = [f.name for f in jpeg_dir.iterdir() if f.suffix == '.jpg']
    jpg_files.sort(key=lambda p: int(p.split('.')[0]))
    jpg_names = [f.split('.')[0] for f in jpg_files]

    seed_indices = []
    for fname, bbox in seeds:
        idx = jpg_names.index(fname)
        x, y, w, h = bbox
        box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
        seed_indices.append((idx, box_xyxy))

    earliest = min(si[0] for si in seed_indices)
    latest = max(si[0] for si in seed_indices)

    masks = {}
    with torch.inference_mode(), torch.autocast(
        'cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        state = _sam2_predictor.init_state(
            video_path=str(jpeg_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        _sam2_predictor.reset_state(state)

        for idx, box_xyxy in seed_indices:
            _sam2_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=idx,
                obj_id=1,
                box=box_xyxy,
            )

        seen = set()
        for reverse in (False, True):
            start = latest if reverse else earliest
            for fid, obj_ids, mask_logits in _sam2_predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=reverse):
                if fid in seen:
                    continue
                seen.add(fid)
                mask = (mask_logits[0] > 0.0)[0].cpu().numpy()
                if mask.sum() == 0:
                    continue
                fname = jpg_names[fid]
                rle = mask_utils.encode(
                    np.asfortranarray(mask.astype(np.uint8)))
                masks[fname] = {
                    'size': list(rle['size']),
                    'counts': rle['counts'].decode('ascii'),
                }

        _sam2_predictor.reset_state(state)

    torch.cuda.empty_cache()
    return masks


# ─── Full-video mode: detect + propagate ──────────────────────────────

def run_one_sample_fullvideo(video_uid: str, expression: str, obj_id: str,
                              expr_idx: int, all_frames: list,
                              clip_dir: Path, index_dir: Path,
                              enc_suffix: str, config_path: Path,
                              output_dir: Path, work_dir: Path,
                              sam2_cfg: str, sam2_ckpt: str) -> dict:
    """
    Full-video mode with adaptive recovery.

    Phase 1 — Primary detection (our pipeline's strength):
      CLIP temporal retrieval → candidate verification → Grounding DINO
      → calibrated presence decision.

    Phase 2 — Single-seed SAM2 propagation:
      Propagate from the primary detection through all frames.

    Phase 3 — Adaptive recovery (only when coverage < 15%):
      When SAM2 loses the object (common in egocentric video with
      occlusion and camera motion), scan additional frames with
      Grounding DINO, add size-consistent detections as new seeds,
      and re-propagate. This targets the ~30% of samples where
      single-seed tracking fails without hurting the ~70% where
      it succeeds.
    """
    t0 = time.time()

    # Phase 1: Detection
    det = run_detection(video_uid, expression, obj_id, expr_idx,
                        clip_dir, index_dir, enc_suffix, config_path,
                        work_dir)
    if "_error" in det:
        return det

    decision = det.get("decision", "not_found")
    presence = det.get("presence", 0.0)
    bbox = det.get("pred_bbox")
    if bbox is None and det.get("tracks"):
        bbox = det["tracks"][0].get("seed_bbox")
    src_frame = det.get("last_frame_idx")
    detect_wall = det.get("_wall_detect_s", 0)

    if decision == "not_found" or bbox is None or src_frame is None:
        return {
            "_meta": {
                "expression": expression,
                "n_all_frames": len(all_frames),
                "n_hits": 0,
                "presence": presence,
                "decision": decision,
                "wall_s": round(time.time() - t0, 1),
            }
        }

    # Phase 2: Single-seed SAM2 propagation
    jpeg_dir = JPEG_ROOT / video_uid
    if not jpeg_dir.exists():
        return {"_error": f"JPEG frame dir missing: {jpeg_dir}"}

    det_frame_name = source_frame_to_jpeg_name(src_frame, all_frames)
    cached_tag = " [cached]" if det.get("_cached") else ""
    print(f"        detect: t={src_frame/30:.1f}s frame={src_frame} "
          f"→ JPEG {det_frame_name}  bbox={bbox}  ({detect_wall}s){cached_tag}")

    t_prop = time.time()
    seeds = [(det_frame_name, bbox)]
    pred_masks = fullvideo_sam2_propagate(
        jpeg_dir, all_frames, seeds, sam2_cfg, sam2_ckpt,
    )
    prop_wall = round(time.time() - t_prop, 1)
    n_hits = len(pred_masks)
    coverage = n_hits / len(all_frames) if all_frames else 0

    # Phase 3: Adaptive recovery — if coverage is poor, add more seeds
    RECOVERY_THRESHOLD = 0.15
    ms_wall = 0
    n_extra = 0
    if coverage < RECOVERY_THRESHOLD:
        det_prompt = expression
        plan = det.get("plan")
        if plan and plan.get("detection_prompt"):
            det_prompt = plan["detection_prompt"]

        t_ms = time.time()
        extra_seeds = detect_additional_seeds(
            jpeg_dir, all_frames, det_prompt, det_frame_name,
            primary_bbox_xywh=bbox,
            min_seed_gap=30, sample_stride=15, max_seeds=3,
        )
        n_extra = len(extra_seeds)
        ms_wall = round(time.time() - t_ms, 1)

        if _gdino is not None:
            _gdino.offload()

        if extra_seeds:
            for fname, sbbox, conf in extra_seeds:
                seeds.append((fname, sbbox))
            t_prop2 = time.time()
            pred_masks = fullvideo_sam2_propagate(
                jpeg_dir, all_frames, seeds, sam2_cfg, sam2_ckpt,
            )
            prop_wall += round(time.time() - t_prop2, 1)
            n_hits = len(pred_masks)
            print(f"        recovery: {n_extra} extra seeds "
                  f"({ms_wall}s scan) → {n_hits}/{len(all_frames)} masked")
        else:
            print(f"        recovery: no extra seeds found ({ms_wall}s scan)")
    else:
        print(f"        SAM2:   {n_hits}/{len(all_frames)} frames masked  "
              f"({prop_wall}s, coverage {coverage:.0%} — no recovery needed)")

    total_wall = round(time.time() - t0, 1)

    # Write grounding_ret.json for detection metrics
    x, y, w, h = bbox
    grounding_ret = {
        "frame_name": det_frame_name,
        "frame_idx": all_frames.index(det_frame_name) if det_frame_name in all_frames else -1,
        "input_boxes": [float(x), float(y), float(x + w), float(y + h)],
        "labels": [expression],
    }

    result = dict(pred_masks)
    result["_meta"] = {
        "expression": expression,
        "n_all_frames": len(all_frames),
        "n_hits": n_hits,
        "presence": presence,
        "decision": decision,
        "wall_s": total_wall,
        "detect_s": detect_wall,
        "propagate_s": prop_wall,
        "n_seeds": len(seeds),
        "recovery": coverage < RECOVERY_THRESHOLD,
    }
    result["_grounding_ret"] = grounding_ret
    return result


# ─── Windowed tracking mode (original) ────────────────────────────────

def run_one_sample_windowed(video_uid: str, expression: str, obj_id: str,
                             expr_idx: int, all_frames: list, clip_dir: Path,
                             index_dir: Path, enc_suffix: str,
                             config_path: Path, output_dir: Path,
                             work_dir: Path, context_seconds: float,
                             reid_context_seconds: float, max_reid: int,
                             exemplar_top_k: int, no_reid: bool) -> dict:
    clip_path  = clip_dir / f"{video_uid}.mp4"
    index_path = index_dir / f"{video_uid}_{enc_suffix}"
    if not clip_path.exists():
        return {"_error": f"clip missing: {clip_path}"}
    if not index_path.exists():
        return {"_error": f"index missing: {index_path}"}

    qdir = work_dir / f"{video_uid}__exp{expr_idx}__obj{obj_id}"
    qdir.mkdir(parents=True, exist_ok=True)
    full_json = qdir / "result.json"

    cmd = [
        sys.executable, "query_indexed.py", expression,
        "--index",    str(index_path),
        "--video",    str(clip_path),
        "--config",   str(config_path),
        "--output",   str(qdir),
        "--json-out", str(full_json),
        "--context-seconds", str(context_seconds),
        "--reid-context-seconds", str(reid_context_seconds),
        "--max-reid", str(0 if no_reid else max_reid),
        "--exemplar-top-k", str(0 if no_reid else exemplar_top_k),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(TEXTUAL_DIR),
                           capture_output=True, text=True)
    wall = round(time.time() - t0, 1)

    if proc.returncode != 0:
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        (qdir / "stdout.txt").write_text(proc.stdout[-8000:], encoding="utf-8")
        return {"_error": f"subprocess exit {proc.returncode}", "_wall_s": wall}

    if not full_json.exists():
        (qdir / "stdout.txt").write_text(proc.stdout[-8000:], encoding="utf-8")
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        return {"_error": f"result.json not produced (logs in {qdir})",
                "_wall_s": wall}

    try:
        out = json.loads(full_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"result.json bad: {e}", "_wall_s": wall}

    tracks = out.get("tracks", [])
    size   = get_clip_dimensions(clip_path)

    pred_masks = {}
    n_hits = 0
    for k in all_frames:
        target_src = gt_key_to_source_frame(k)
        m = find_nearest_track_mask(target_src, tracks)
        if m is not None:
            pred_masks[k] = {"size": m.get("size", size), "counts": m["counts"]}
            n_hits += 1

    pred_masks["_meta"] = {
        "expression": expression,
        "n_all_frames": len(all_frames),
        "n_hits": n_hits,
        "presence": out.get("presence"),
        "decision": out.get("decision"),
        "wall_s": wall,
    }
    return pred_masks


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str,
                    default=str(TEXTUAL_DIR / "config.yaml"),
                    help="Pipeline config (CLIP or SigLIP).")
    p.add_argument("--tier", choices=["long", "medium", "short"],
                    default="long")
    p.add_argument("--limit", type=int, default=None,
                    help="Process only the first N samples.")
    p.add_argument("--clip-dir", type=str, default=str(DEFAULT_CLIP_DIR))
    p.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--output-dir", type=str, default=None)

    # Full-video propagation (recommended)
    p.add_argument("--full-video", action="store_true",
                    help="Full-video SAM2 propagation: detect once, propagate "
                         "through all frames. Recommended for EgoMask.")

    # Windowed tracking options (ablation)
    p.add_argument("--context-seconds", type=float, default=20.0)
    p.add_argument("--reid-context-seconds", type=float, default=6.0)
    p.add_argument("--max-reid", type=int, default=10)
    p.add_argument("--exemplar-top-k", type=int, default=25)
    p.add_argument("--no-reid", action="store_true")
    p.add_argument("--work-dir", type=str, default=None,
                    help="Reuse detection cache from a prior run's _work dir")

    args = p.parse_args()

    config_path = Path(args.config).resolve()
    enc_suffix  = encoder_suffix(config_path)
    enc_name    = "clip" if enc_suffix == "v2" else enc_suffix
    tier_dir    = EGOMASK_ROOT / "subset" / args.tier

    if args.output_dir:
        output_base = Path(args.output_dir).resolve()
    else:
        mode = "fullvid" if args.full_video else "windowed"
        output_base = EVAL_DIR / "egomask_preds" / \
                      f"{enc_name}_{mode}_{time.strftime('%Y-%m-%d')}"
    output_dir = output_base / args.tier
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
    else:
        work_dir = output_base / "_work"
    work_dir.mkdir(exist_ok=True)

    # Resolve SAM2 paths from config (needed for full-video mode)
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # SAM2 config is resolved by Hydra internally — pass as-is
    sam2_cfg = cfg['text_query'].get(
        'tracker_param', 'configs/sam2.1/sam2.1_hiera_l.yaml')
    # SAM2 checkpoint is relative to TextualREN_v2/
    sam2_ckpt_rel = cfg['data'].get('sam2_ckpt', '../checkpoints/sam2.1_hiera_large.pt')
    sam2_ckpt = str((TEXTUAL_DIR / sam2_ckpt_rel).resolve())

    meta_exp = json.loads(
        (tier_dir / "meta_expressions.json").read_text(encoding="utf-8"))

    samples = []
    for video_uid, vrec in meta_exp.get("videos", {}).items():
        all_frames = vrec.get("frames", [])
        for expr_key, erec in vrec.get("expressions", {}).items():
            samples.append((video_uid, int(expr_key), erec["obj_id"],
                            erec["exp"], all_frames))

    if args.limit:
        samples = samples[:args.limit]

    print(f"EgoMask adapter")
    print(f"  tier      : {args.tier}")
    print(f"  encoder   : {enc_name}  (suffix: _{enc_suffix})")
    print(f"  mode      : {'FULL-VIDEO propagation' if args.full_video else 'windowed tracking'}")
    print(f"  samples   : {len(samples)}")
    print(f"  output    : {output_dir}")
    if not args.full_video:
        print(f"  context_s : {args.context_seconds}")
        if not args.no_reid:
            print(f"  max_reid  : {args.max_reid}")
    print(f"  eval cmd  : python eval_egomask.py --pred_path {output_base} "
          f"--dataset_type {args.tier}")
    print()

    summary_rows = []

    for i, (video_uid, expr_idx, obj_id, expression, all_frames) in enumerate(samples, 1):
        print(f"[{i:3}/{len(samples)}] {video_uid[:8]}.. expr#{expr_idx} "
              f"obj={obj_id} \"{expression[:60]}{'…' if len(expression)>60 else ''}\"")
        if not all_frames:
            print(f"        SKIP: no frames")
            continue

        if args.full_video:
            result = run_one_sample_fullvideo(
                video_uid, expression, obj_id, expr_idx, all_frames,
                clip_dir=Path(args.clip_dir),
                index_dir=Path(args.index_dir),
                enc_suffix=enc_suffix, config_path=config_path,
                output_dir=output_dir, work_dir=work_dir,
                sam2_cfg=sam2_cfg, sam2_ckpt=sam2_ckpt,
            )
        else:
            result = run_one_sample_windowed(
                video_uid, expression, obj_id, expr_idx, all_frames,
                clip_dir=Path(args.clip_dir),
                index_dir=Path(args.index_dir),
                enc_suffix=enc_suffix, config_path=config_path,
                output_dir=output_dir, work_dir=work_dir,
                context_seconds=args.context_seconds,
                reid_context_seconds=args.reid_context_seconds,
                max_reid=args.max_reid,
                exemplar_top_k=args.exemplar_top_k,
                no_reid=args.no_reid,
            )

        if "_error" in result:
            print(f"        ERROR: {result['_error']}")
            summary_rows.append({"video": video_uid, "exp": expr_idx,
                                  "obj": obj_id, "expr": expression,
                                  "error": result["_error"]})
            continue

        meta = result.pop("_meta", {})
        grounding_ret = result.pop("_grounding_ret", None)

        # Write prediction masks
        exp_dir = output_dir / video_uid / str(expr_idx)
        exp_dir.mkdir(parents=True, exist_ok=True)
        pred_file = exp_dir / f"{expr_idx}-{obj_id}.json"
        pred_file.write_text(json.dumps(result), encoding="utf-8")

        # Write grounding_ret.json for detection metrics
        if grounding_ret:
            gret_file = exp_dir / f"{expr_idx}-{obj_id}_grounding_ret.json"
            gret_file.write_text(json.dumps(grounding_ret), encoding="utf-8")

        n_hits = meta.get('n_hits', 0)
        n_frames = meta.get('n_all_frames', 0)
        print(f"        OK: decision={meta.get('decision')} "
              f"presence={meta.get('presence'):.4f} "
              f"hits={n_hits}/{n_frames} "
              f"wall={meta.get('wall_s')}s")

        summary_rows.append({"video": video_uid, "exp": expr_idx,
                              "obj": obj_id, "expr": expression,
                              "decision": meta.get("decision"),
                              "presence": meta.get("presence"),
                              "n_hits": n_hits, "n_frames": n_frames,
                              "wall_s": meta.get("wall_s")})

    # Summary CSV
    summary_csv = output_base / f"summary_{args.tier}.csv"
    if summary_rows:
        cols = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    print(f"\nDone. {len(summary_rows)} samples processed.")
    print(f"  predictions: {output_dir}")
    print(f"  summary    : {summary_csv}")
    print(f"\nTo evaluate:")
    print(f"  cd eval/EgoMask")
    print(f"  python evaluation/eval_egomask.py --pred_path {output_base} "
          f"--dataset_type {args.tier}")


if __name__ == "__main__":
    main()
