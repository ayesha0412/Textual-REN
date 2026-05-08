"""
Interactive evaluation tool for Textual-REN.

You watch the video, type what you see, the system finds it,
shows you the result — you confirm correct / fix bbox / reject.
All confirmed results saved to a growing annotated test set.

Usage:
    python interactive_eval.py \
        --video  "D:/REN Project/epic_kitchen_data/EPIC-KITCHENS/P02/videos/P02_01.MP4" \
        --index  "D:/REN Project/epic_kitchen_indexes/P02_01" \
        --config ../text_query/config.yaml \
        --output annotated_testset.json

Run once per video. Results accumulate in the same output file across sessions.

Commands during session:
    <query>        type any text query and press Enter
    show           open last result image again
    done           finish this video and save
    quit / exit    save and exit immediately
"""

import os
import sys
import json
import argparse
import subprocess
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text_query'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml


def draw_bbox(img, bbox, color=(0, 255, 0), label=''):
    x, y, w, h = [int(v) for v in bbox]
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    if label:
        cv2.putText(out, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def open_image(path: str):
    """Open image in Windows Photos."""
    os.startfile(os.path.abspath(path))


def load_existing(output_path: str):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return []


def save_results(results, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def run_query(engine, query, video_path, tmp_dir, config):
    """Run pipeline for one query. Returns result dict or None."""
    import tempfile, shutil
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        result = engine.query(
            text_query=query,
            video_path=video_path,
            output_dir=tmp_dir,
        )
        debug_src = os.path.join(tmp_dir, 'debug_last_frame.jpg')
        return result, debug_src
    except RuntimeError as e:
        print(f"\n  [FAILED] {e}")
        hint = str(e)
        return None, None


def annotate_result(result, debug_path, query, video_id, video_path, index_dir):
    """
    Show the found frame, ask user to confirm/correct.
    Returns a finished annotated entry or None to discard.
    """
    pred_bbox = result.get('pred_bbox')
    pred_ts   = result.get('last_frame_timestamp')
    sim       = result.get('fused_similarity', 0)
    qtype     = 'ocr' if result.get('ocr_score', 0) > 0.5 else 'general'

    print(f"\n  ─── Result ───────────────────────────────────────")
    print(f"  Timestamp : {pred_ts:.1f}s")
    print(f"  Bbox      : {pred_bbox}")
    print(f"  Similarity: {sim:.3f}  |  type: {qtype}")

    # Draw bbox on image and open
    if debug_path and os.path.exists(debug_path):
        img = cv2.imread(debug_path)
        if img is not None and pred_bbox:
            vis = draw_bbox(img, pred_bbox, color=(0, 255, 0), label=query)
            vis_path = debug_path.replace('.jpg', '_vis.jpg')
            cv2.imwrite(vis_path, vis)
            open_image(vis_path)
            print(f"  (Image opened in Photos)")

    print(f"\n  Is this correct?")
    print(f"  y / Enter  → correct, accept as ground truth")
    print(f"  f          → wrong bbox only — type corrected x,y,w,h")
    print(f"  n          → wrong frame entirely — discard this query")
    print(f"  s          → skip (don't save, try next query)")

    while True:
        inp = input("  > ").strip().lower()

        if inp in ('y', ''):
            return {
                'video_id':    video_id,
                'video_path':  video_path,
                'index_dir':   index_dir,
                'query':       query,
                'query_type':  qtype,
                'gt_bbox':     pred_bbox,
                'gt_timestamp': pred_ts,
                'gt_frame_idx': result.get('last_frame_idx'),
                'confirmed':   'accepted',
            }

        elif inp == 'f':
            print("  Enter corrected bbox as x,y,w,h:")
            raw = input("  > ").strip()
            try:
                bbox = [int(v) for v in raw.split(',')]
                assert len(bbox) == 4
                return {
                    'video_id':    video_id,
                    'video_path':  video_path,
                    'index_dir':   index_dir,
                    'query':       query,
                    'query_type':  qtype,
                    'gt_bbox':     bbox,
                    'gt_timestamp': pred_ts,
                    'gt_frame_idx': result.get('last_frame_idx'),
                    'confirmed':   'bbox_fixed',
                }
            except Exception:
                print("  Invalid format — try again (x,y,w,h e.g. 100,200,300,150)")

        elif inp == 'n':
            print("  Discarded — wrong frame.")
            return None

        elif inp == 's':
            print("  Skipped.")
            return None

        else:
            print("  Type y, f, n, or s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  required=True, help='Path to video file')
    parser.add_argument('--index',  required=True, help='Path to FAISS index directory')
    parser.add_argument('--config', default='../text_query/config.yaml')
    parser.add_argument('--output', default='annotated_testset.json',
                        help='Growing output file — appends across sessions')
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    video_id  = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    # Load existing results (accumulate across sessions)
    all_results = load_existing(output_path)
    existing_keys = {(r['video_id'], r['query']) for r in all_results}
    session_results = []

    print(f"\n{'='*60}")
    print(f"  Textual-REN Interactive Evaluator")
    print(f"{'='*60}")
    print(f"  Video : {os.path.basename(args.video)}")
    print(f"  Index : {args.index}")
    print(f"  Output: {output_path}")
    print(f"  Existing annotated queries: {len(all_results)}")
    print(f"\n  Type a query to search. Commands: done | quit | show")
    print(f"{'='*60}\n")

    # Load engine
    from query_indexed import IndexedQueryEngine
    engine = IndexedQueryEngine(config, args.index)

    tmp_dir = os.path.join(os.path.dirname(output_path), '_tmp_eval')
    last_vis_path = None

    while True:
        try:
            raw = input(f"\n[{video_id}] Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            raw = 'quit'

        if not raw:
            continue

        if raw.lower() in ('done', 'quit', 'exit'):
            break

        if raw.lower() == 'show':
            if last_vis_path and os.path.exists(last_vis_path):
                open_image(last_vis_path)
            else:
                print("  No image to show yet.")
            continue

        query = raw
        key   = (video_id, query)
        if key in existing_keys:
            print(f"  Already annotated: '{query}' in {video_id} — skipping.")
            print(f"  (Delete from {output_path} to re-annotate)")
            continue

        print(f"\n  Searching: '{query}' ...")
        q_tmp = os.path.join(tmp_dir, video_id, query.replace(' ', '_'))
        result, debug_path = run_query(engine, query, args.video, q_tmp, config)

        if result is None:
            print(f"  Query failed — try a different description or different object.")
            continue

        last_vis_path = debug_path.replace('.jpg', '_vis.jpg') if debug_path else None

        entry = annotate_result(result, debug_path, query, video_id,
                                args.video, args.index)
        if entry is not None:
            session_results.append(entry)
            existing_keys.add(key)
            all_results.append(entry)
            save_results(all_results, output_path)
            print(f"  ✓ Saved ({len(session_results)} new this session, "
                  f"{len(all_results)} total)")

    # Final save
    save_results(all_results, output_path)

    print(f"\n{'='*60}")
    print(f"  Session complete")
    print(f"  New entries this session : {len(session_results)}")
    print(f"  Total annotated queries  : {len(all_results)}")
    print(f"  Saved to: {output_path}")

    # Summary by type
    ocr_n = sum(1 for r in all_results if r.get('query_type') == 'ocr')
    gen_n = sum(1 for r in all_results if r.get('query_type') == 'general')
    print(f"\n  OCR queries   : {ocr_n}")
    print(f"  General queries: {gen_n}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
