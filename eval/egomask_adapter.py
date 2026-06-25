"""
EgoMask adapter — runs our text-VQL pipeline on EgoMask samples and writes
predictions in EgoMask's per-video JSON format, ready for their official
eval script.

What it does:
  - Loads expressions from egomask/subset/{tier}/meta_expressions.json
  - For each (video_uid, expression_index, obj_id, expression):
      1. Check the source clip + its pre-built index exist
      2. Invoke query_indexed.py with the expression as the text query
      3. Parse the resulting tracks (per-frame bboxes + RLE masks)
      4. For each video frame, look up our mask at the corresponding
         source-video frame index (EgoMask frame "00045" ↔ source 270,
         since EgoMask kept frames live at 5 FPS = stride 6 of 30 FPS).
      5. Write to <output_dir>/<tier>/<vid>/<exp_id>/<exp_id>-<obj_id>.json
         matching the eval script's expected layout.

Frame index mapping (the critical detail):
  EgoMask frame name e.g. "00045" → integer 45 → source frame 45 * 6 = 270.
  We then find the nearest track frame within ±15 source frames (0.5 sec
  at 30 FPS); outside that window → empty mask (honest = we didn't see it).

Prerequisites:
  - Indexes for every clip in the tier, built with the active encoder:
        epic_kitchen_indexes/<video_uid>_<v2|siglip2>/    (or override --index-dir)
  - EgoMask annotation directory populated:
        eval/EgoMask/dataset/egomask/subset/<tier>/...
  - EgoTracks clips on disk:
        eval/EgoMask/dataset/tmp/ego4d/v2/clips/<video_uid>.mp4
    (path overridable via --clip-dir)

Usage:
  # Smoke-test on a single sample with CLIP
  python eval/egomask_adapter.py --tier long --limit 1
  # Same with SigLIP 2
  python eval/egomask_adapter.py --tier long --limit 1 \
      --config TextualREN_v2/config_siglip2.yaml
  # Full long-tier eval (15 videos × few expressions each)
  python eval/egomask_adapter.py --tier long
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
TEXTUAL_DIR  = PROJECT_ROOT / "TextualREN_v2"
EGOMASK_ROOT = EVAL_DIR / "EgoMask" / "dataset" / "egomask"
DEFAULT_CLIP_DIR  = EVAL_DIR / "EgoMask" / "dataset" / "tmp" / "ego4d" / "v2" / "clips"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "epic_kitchen_indexes"  # we'll add a tier subdir

# Frame-rate constants for EgoMask (from preprocess/egotracks/extract_clip_frames.py)
EGOMASK_FRAME_STRIDE = 6   # 30 FPS source → 5 FPS subsampling; EgoMask frame names are 5-FPS indices
NEAREST_FRAME_TOLERANCE_SRC = 15   # ±0.5 sec at 30 FPS — how far to look for a nearby prediction


# ─── Helpers ───────────────────────────────────────────────────────────

def gt_key_to_source_frame(key: str) -> int:
    """EgoMask frame name like '00045' → source-video frame index (270)."""
    return int(key) * EGOMASK_FRAME_STRIDE


def encoder_suffix(config_path: Path) -> str:
    """Read encoder from config so we look in the right index dir.
       'clip' → '_v2'  (legacy-compatible);  'siglip2' → '_siglip2'."""
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enc = cfg.get("text_query", {}).get("encoder", "clip")
    return "v2" if enc == "clip" else enc


def find_nearest_track_mask(target_src_frame: int, tracks: list,
                             tol: int = NEAREST_FRAME_TOLERANCE_SRC):
    """
    Across all tracks (all instances), find the prediction whose frame_idx
    is closest to target_src_frame. Returns the mask dict (with 'size' and
    'counts') if within tolerance, else None.
    """
    best, best_dist = None, tol + 1
    for tr in tracks:
        for f in tr.get("frames", []):
            d = abs(int(f["frame_idx"]) - target_src_frame)
            if d < best_dist and "mask" in f:
                best_dist = d
                best = f["mask"]
    return best


def empty_mask(size: list) -> dict:
    """Empty COCO RLE — used when we have no prediction for a GT frame.
       This is the honest 'we didn't see anything here' output, not a guess."""
    H, W = size
    # COCO RLE for all-zero mask: one run of length H*W
    return {"size": [H, W], "counts": f"{H*W}"}


def get_clip_dimensions(clip_path: Path) -> list:
    """Quickly read frame H, W from the mp4 without decoding video."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return [H, W]


# ─── Per-sample execution ──────────────────────────────────────────────

def run_one_sample(video_uid: str, expression: str, obj_id: str,
                    expr_idx: int, all_frames: list, clip_dir: Path,
                    index_dir: Path, enc_suffix: str, config_path: Path,
                    output_dir: Path, work_dir: Path) -> dict:
    """
    Invoke the pipeline once for one (video, expression) and return a dict
    mapping EgoMask frame_id → {size, counts} (RLE mask).

    all_frames: ALL video frame names from meta_expressions (not just GT).
    The eval iterates over every frame and checks membership in the prediction
    JSON — we only include frames where our pipeline actually produced a mask.
    """
    clip_path  = clip_dir / f"{video_uid}.mp4"
    index_path = index_dir / f"{video_uid}_{enc_suffix}"
    if not clip_path.exists():
        return {"_error": f"clip missing: {clip_path}"}
    if not index_path.exists():
        return {"_error": f"index missing: {index_path}  "
                          f"(build with prepare_index.py)"}

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
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(TEXTUAL_DIR),
                           capture_output=True, text=True)
    wall = round(time.time() - t0, 1)
    if proc.returncode != 0:
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        return {"_error": f"subprocess exit {proc.returncode} (stderr saved)",
                "_wall_s": wall}

    if not full_json.exists():
        return {"_error": "result.json not produced", "_wall_s": wall}

    try:
        out = json.loads(full_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"result.json bad: {e}", "_wall_s": wall}

    tracks = out.get("tracks", [])
    size   = get_clip_dimensions(clip_path)

    # Scan ALL video frames for predictions (not just GT frames).
    pred_masks = {}
    n_hits = 0
    for k in all_frames:
        target_src = gt_key_to_source_frame(k)
        m = find_nearest_track_mask(target_src, tracks)
        if m is not None:
            pred_masks[k] = {"size": m.get("size", size),
                              "counts": m["counts"]}
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
                    default="long",
                    help="EgoMask subset tier (default: long = smallest).")
    p.add_argument("--limit", type=int, default=None,
                    help="Smoke-test mode: only process the first N samples.")
    p.add_argument("--clip-dir", type=str, default=str(DEFAULT_CLIP_DIR),
                    help="Where the source .mp4 clips live.")
    p.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR),
                    help="Where <video_uid>_<enc>/ index dirs live.")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Where predictions go (default: eval/egomask_preds/<tier>_<enc>_<date>).")
    args = p.parse_args()

    config_path = Path(args.config)
    enc_suffix  = encoder_suffix(config_path)
    enc_name    = "clip" if enc_suffix == "v2" else enc_suffix
    tier_dir    = EGOMASK_ROOT / "subset" / args.tier

    # Output dir keyed by encoder + date.  The tier becomes a subdirectory
    # so the eval script's --pred_path / --dataset_type combo works directly:
    #   python eval_egomask.py --pred_path <output_base> --dataset_type long
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = EVAL_DIR / "egomask_preds" / \
                      f"{enc_name}_{time.strftime('%Y-%m-%d')}"
    output_dir = output_base / args.tier
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_base / "_work"
    work_dir.mkdir(exist_ok=True)

    # Load EgoMask metadata.
    meta_exp = json.loads((tier_dir / "meta_expressions.json").read_text(encoding="utf-8"))

    # Flatten into a list of (video_uid, expression_idx, obj_id, expression, all_frames).
    # all_frames = every frame in the video (from meta_expressions), so the
    # prediction covers all frames the eval iterates over.
    samples = []
    for video_uid, vrec in meta_exp.get("videos", {}).items():
        all_frames = vrec.get("frames", [])
        for expr_key, erec in vrec.get("expressions", {}).items():
            obj_id = erec["obj_id"]
            expression = erec["exp"]
            samples.append((video_uid, int(expr_key), obj_id, expression, all_frames))

    if args.limit:
        samples = samples[:args.limit]

    print(f"EgoMask adapter")
    print(f"  tier      : {args.tier}")
    print(f"  encoder   : {enc_name}  (index dir suffix: _{enc_suffix})")
    print(f"  samples   : {len(samples)}  (of {sum(len(v.get('expressions',{})) for v in meta_exp.get('videos',{}).values())} in tier)")
    print(f"  clip dir  : {args.clip_dir}")
    print(f"  index dir : {args.index_dir}")
    print(f"  output    : {output_dir}")
    print(f"  eval cmd  : python eval_egomask.py --pred_path {output_base} "
          f"--dataset_type {args.tier}")
    print()

    summary_rows = []

    for i, (video_uid, expr_idx, obj_id, expression, all_frames) in enumerate(samples, 1):
        print(f"[{i:3}/{len(samples)}] {video_uid[:8]}.. expr#{expr_idx} "
              f"obj={obj_id} \"{expression[:60]}{'…' if len(expression)>60 else ''}\"")
        if not all_frames:
            print(f"        SKIP: no frames listed for this video")
            continue
        result = run_one_sample(
            video_uid, expression, obj_id, expr_idx, all_frames,
            clip_dir=Path(args.clip_dir),
            index_dir=Path(args.index_dir),
            enc_suffix=enc_suffix,
            config_path=config_path,
            output_dir=output_dir,
            work_dir=work_dir,
        )
        if "_error" in result:
            print(f"        ERROR: {result['_error']}")
            summary_rows.append({"video": video_uid, "exp": expr_idx,
                                  "obj": obj_id, "expr": expression,
                                  "error": result["_error"]})
            continue
        meta = result.pop("_meta", {})

        # Write in EgoMask's eval-expected layout:
        #   <output_dir>/<vid>/<exp_id>/<exp_id>-<obj_id>.json
        exp_dir = output_dir / video_uid / str(expr_idx)
        exp_dir.mkdir(parents=True, exist_ok=True)
        pred_file = exp_dir / f"{expr_idx}-{obj_id}.json"
        pred_file.write_text(json.dumps(result), encoding="utf-8")

        print(f"        OK: decision={meta.get('decision')} "
              f"presence={meta.get('presence')} "
              f"hits={meta.get('n_hits')}/{meta.get('n_all_frames')} "
              f"wall={meta.get('wall_s')}s")
        summary_rows.append({"video": video_uid, "exp": expr_idx,
                              "obj": obj_id, "expr": expression,
                              "decision": meta.get("decision"),
                              "presence": meta.get("presence"),
                              "n_hits": meta.get("n_hits"),
                              "n_frames": meta.get("n_all_frames"),
                              "wall_s": meta.get("wall_s")})

    # Summary CSV for quick eyeballing.
    import csv
    summary_csv = output_base / f"summary_{args.tier}.csv"
    if summary_rows:
        cols = sorted({k for r in summary_rows for k in r.keys()})
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    print(f"\nDone.")
    print(f"  predictions: {output_dir}/<vid>/<exp_id>/<exp_id>-<obj_id>.json")
    print(f"  summary    : {summary_csv}")
    print(f"\nTo evaluate:")
    print(f"  cd eval/EgoMask")
    print(f"  python evaluation/eval_egomask.py --pred_path {output_base} "
          f"--dataset_type {args.tier}")


if __name__ == "__main__":
    main()
