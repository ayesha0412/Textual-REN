"""
Batch runner for the v4 query set. Loops `query_indexed.py --json-out` over
every query in eval/query_set_v4.yaml and collects results.

Design notes (why it's shaped this way):
  - Subprocess per query, not one long-lived process. query_indexed.py is
    CLI-shaped; making it importable would be invasive surgery on working
    code. Cost is ~30s of model loading per query (~50 min over 100 queries),
    which is small next to actual query work.
  - result.json is the resume marker (it's the last file query_indexed writes).
    A query is "done" iff that file exists with a non-error decision.
  - Output layout: eval/results/<run_id>/<qid>/ — one dir per query, mirrors
    query_indexed's own --output convention so the JPEGs/MP4s land next to
    the JSON without surprising paths.
  - CWD = TextualREN_v2 for the subprocess so config.yaml resolves and the
    pipeline's own relative paths behave as if launched from the script's home.
  - Summary CSV is rewritten after every query, so an interrupted run still
    leaves a usable artifact.

Usage:
  python eval/run_query_set.py                         # everything
  python eval/run_query_set.py --split dev             # dev only
  python eval/run_query_set.py --tag absent            # abstention anchors
  python eval/run_query_set.py --query-id q020         # single query
  python eval/run_query_set.py --video P04_01          # all queries for one video
  python eval/run_query_set.py --dry-run               # show plan, run nothing
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ─── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
TEXTUAL_DIR  = PROJECT_ROOT / "TextualREN_v2"
INDEX_DIR    = PROJECT_ROOT / "epic_kitchen_indexes"
VIDEOS_DIR   = PROJECT_ROOT / "epic_kitchen_data" / "EPIC-KITCHENS"
RESULTS_DIR  = EVAL_DIR / "results"

QUERY_SET    = EVAL_DIR / "query_set_v4.yaml"


# ─── Helpers ─────────────────────────────────────────────────────────────

def video_path(vid: str) -> Path:
    """P04_01 -> .../EPIC-KITCHENS/P04/videos/P04_01.MP4"""
    return VIDEOS_DIR / vid.split("_")[0] / "videos" / f"{vid}.MP4"


def index_path(vid: str, enc: str = "clip") -> Path:
    """CLIP   -> .../epic_kitchen_indexes/P04_01_v2
       SigLIP -> .../epic_kitchen_indexes/P04_01_siglip2
    Different encoders write incompatible patch embeddings; we never share dirs."""
    suffix = "v2" if enc == "clip" else enc
    return INDEX_DIR / f"{vid}_{suffix}"


def config_hash(config_path: Path) -> str:
    """First 8 chars of canonical config SHA-256, EXCLUDING the encoder
    fields (they're captured in the run_id prefix instead). This means:
      - Adding `encoder:` to the YAML doesn't invalidate existing CLIP runs.
      - Changing OTHER config fields still produces a new run_id.
      - CLIP and SigLIP runs land in distinct dirs via the prefix only.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg.get("text_query", {}).pop("encoder", None)
    cfg.get("text_query", {}).pop("siglip_model", None)
    canonical = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:8]


def encoder_name(config_path: Path) -> str:
    """Read the active encoder from config — used in run_id prefix so CLIP
    and SigLIP results never share a directory by accident."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return cfg.get("text_query", {}).get("encoder", "clip")


def load_query_set() -> dict:
    with open(QUERY_SET, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_queries(queries, split=None, tag=None, video=None, qid=None):
    out = queries
    if qid:
        out = [q for q in out if q["id"] == qid]
    if split:
        out = [q for q in out if q["split"] == split]
    if tag:
        out = [q for q in out if tag in q.get("tags", [])]
    if video:
        out = [q for q in out if q["video"] == video]
    return out


def is_done(qdir: Path) -> bool:
    """A query is done iff its result.json exists and parses with a decision."""
    rj = qdir / "result.json"
    if not rj.exists():
        return False
    try:
        d = json.loads(rj.read_text(encoding="utf-8"))
        return "decision" in d
    except (json.JSONDecodeError, OSError):
        return False


def run_one(q: dict, run_dir: Path, config_path: Path, enc: str,
             verbose: bool = True) -> dict:
    """
    Invoke query_indexed.py for one query. Returns a dict suitable for the
    summary CSV. Does NOT raise on subprocess failure — records error instead.
    """
    qid    = q["id"]
    vid    = q["video"]
    text   = q["text"]
    qdir   = run_dir / qid
    qdir.mkdir(parents=True, exist_ok=True)
    json_out = qdir / "result.json"

    idx = index_path(vid, enc)

    # Pre-flight checks before spending GPU minutes.
    if not video_path(vid).exists():
        return {**_csv_row_skel(q), "decision": "ERROR", "error": f"video missing: {video_path(vid)}"}
    if not idx.exists():
        return {**_csv_row_skel(q), "decision": "ERROR", "error": f"index missing: {idx}"}

    cmd = [
        sys.executable, "query_indexed.py", text,
        "--index",    str(idx),
        "--video",    str(video_path(vid)),
        "--config",   str(config_path),
        "--output",   str(qdir),
        "--json-out", str(json_out),
    ]
    t0 = time.time()
    try:
        # cwd = TextualREN_v2 so config.yaml resolves
        proc = subprocess.run(cmd, cwd=str(TEXTUAL_DIR), capture_output=True, text=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {**_csv_row_skel(q), "decision": "ERROR", "error": f"subprocess raised: {e}",
                "wall_s": round(time.time() - t0, 1)}

    wall = round(time.time() - t0, 1)
    if proc.returncode != 0:
        # Persist stderr tail for debugging — useful when this happens overnight.
        (qdir / "stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
        return {**_csv_row_skel(q), "decision": "ERROR",
                "error": f"exit {proc.returncode} (stderr saved)", "wall_s": wall}

    if not json_out.exists():
        return {**_csv_row_skel(q), "decision": "ERROR",
                "error": "result.json not produced", "wall_s": wall}

    try:
        res = json.loads(json_out.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {**_csv_row_skel(q), "decision": "ERROR",
                "error": f"result.json bad: {e}", "wall_s": wall}

    return _summarize(q, res, wall)


def _csv_row_skel(q: dict) -> dict:
    """Empty row pre-populated with the YAML fields (for ERROR cases)."""
    return {
        "qid":        q["id"],
        "video":      q["video"],
        "text":       q["text"],
        "expected":   q.get("expected", "unknown"),
        "split":      q.get("split", ""),
        "tags":       "|".join(q.get("tags", [])),
        "decision":   "",
        "found":      "",
        "presence":   "",
        "confidence": "",
        "refine_iters": "",
        "num_instances": "",
        "num_reid":   "",
        "wall_s":     "",
        "error":      "",
    }


def _summarize(q: dict, res: dict, wall: float) -> dict:
    row = _csv_row_skel(q)
    row.update({
        "decision":      res.get("decision", ""),
        "found":         res.get("found", ""),
        "presence":      round(res.get("presence", 0.0), 4),
        "confidence":    round(res.get("confidence", 0.0), 4),
        "refine_iters":  res.get("refine_iters", 0),
        "num_instances": res.get("num_instances", 0),
        "num_reid":      len(res.get("reid_occurrences", []) or []),
        "wall_s":        wall,
        "error":         "",
    })
    return row


def write_summary(rows: list, csv_path: Path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query-set", type=str, default=str(QUERY_SET),
                   help="Path to query set YAML (default: eval/query_set_v4.yaml)")
    p.add_argument("--config",    type=str, default=str(TEXTUAL_DIR / "config.yaml"),
                   help="Pipeline config (default: TextualREN_v2/config.yaml). "
                        "Pass config_siglip2.yaml to run the SigLIP 2 encoder.")
    p.add_argument("--run-id",    type=str, default=None,
                   help="Run identifier (default: YYYY-MM-DD_<encoder?>_<configHash[:8]>)")
    p.add_argument("--split",     choices=["dev", "test"], default=None)
    p.add_argument("--tag",       type=str, default=None)
    p.add_argument("--video",     type=str, default=None)
    p.add_argument("--query-id",  type=str, default=None)
    p.add_argument("--force",     action="store_true",
                   help="Re-run even if result.json already exists")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print plan, do nothing")
    args = p.parse_args()

    # ─── Pre-flight import check ───────────────────────────────────────
    # Catches the dependency-broken-mid-run scenario (e.g. someone ran
    # `pip install -U huggingface_hub` and broke transformers/tokenizers
    # compatibility). Without this check, we'd discover that 38 of 102
    # subprocesses all crash with the same ImportError — which is what
    # happened on 2026-06-16. Spend 2 sec here to avoid 8 hr of wasted
    # GPU. The check runs the imports query_indexed.py needs in a
    # subprocess so the result reflects the actual run environment.
    if not args.dry_run:
        print("[pre-flight] checking imports …", end=" ", flush=True)
        smoke_proc = subprocess.run(
            [sys.executable, "-c",
             "import torch; "
             "import transformers; "
             "from transformers import AutoModel, AutoProcessor; "
             "import open_clip"],
            capture_output=True, text=True, cwd=str(TEXTUAL_DIR),
        )
        if smoke_proc.returncode != 0:
            print("FAIL")
            print(smoke_proc.stderr[-2000:], file=sys.stderr)
            print("\nFix the import error above, then re-run. Common fix: "
                  "`pip install \"huggingface_hub>=0.30.0,<1.0\"` "
                  "(transformers 4.51 requires hub < 1.0).", file=sys.stderr)
            sys.exit(3)
        print("OK")

    # Load + filter
    qs = load_query_set()
    queries = filter_queries(qs["queries"],
                             split=args.split, tag=args.tag,
                             video=args.video, qid=args.query_id)

    if not queries:
        print("No queries match the filters.", file=sys.stderr)
        sys.exit(2)

    # Run id format:
    #   CLIP   (default backend): "<YYYY-MM-DD>_<configHash>"  (legacy-compatible,
    #          so the existing CLIP run_dir 2026-06-14_358d195e/ keeps matching).
    #   SigLIP (or any non-clip): "<YYYY-MM-DD>_<encoder>_<configHash>"
    config_path = Path(args.config).resolve()
    enc = encoder_name(config_path)
    h   = config_hash(config_path)
    if args.run_id:
        run_id = args.run_id
    elif enc == "clip":
        run_id = f"{time.strftime('%Y-%m-%d')}_{h}"
    else:
        run_id = f"{time.strftime('%Y-%m-%d')}_{enc}_{h}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Plan
    todo, skip = [], []
    for q in queries:
        if not args.force and is_done(run_dir / q["id"]):
            skip.append(q)
        else:
            todo.append(q)

    print(f"\n=== run_id: {run_id} ===")
    print(f"  results dir : {run_dir}")
    print(f"  matched     : {len(queries)} queries")
    print(f"  to run      : {len(todo)}")
    print(f"  already done: {len(skip)}\n")

    if args.dry_run:
        for q in todo:
            print(f"  TODO {q['id']:5} {q['video']:6}  {q['text']}")
        return

    # Stamp the run with the source YAML + git hash for reproducibility
    stamp = {
        "run_id":     run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query_set":  str(QUERY_SET),
        "n_queries":  len(queries),
        "filters":    {"split": args.split, "tag": args.tag,
                       "video": args.video, "query_id": args.query_id},
    }
    (run_dir / "run_stamp.json").write_text(json.dumps(stamp, indent=2))

    # Execute. Summary is rewritten after EVERY query so an interrupt still
    # leaves a usable artifact — important for an overnight run.
    rows = []
    summary_csv = run_dir / "summary.csv"

    # Re-include already-done queries in the summary by reading their results
    for q in skip:
        rj = run_dir / q["id"] / "result.json"
        try:
            res = json.loads(rj.read_text(encoding="utf-8"))
            rows.append(_summarize(q, res, wall=float("nan")))
        except Exception:
            rows.append({**_csv_row_skel(q), "decision": "STALE",
                         "error": "could not re-parse cached result.json"})

    for i, q in enumerate(todo, 1):
        print(f"[{i:3}/{len(todo)}] {q['id']} {q['video']:6} \"{q['text']}\" …", flush=True)
        row = run_one(q, run_dir, config_path, enc)
        rows.append(row)
        write_summary(rows, summary_csv)  # checkpoint after every query
        flag = "ok " if row["decision"] not in ("ERROR", "") else "ERR"
        print(f"        -> {flag}  decision={row['decision']:10}  "
              f"p={row['presence']}  wall={row['wall_s']}s", flush=True)

    print(f"\nDone. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
