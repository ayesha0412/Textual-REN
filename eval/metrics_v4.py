"""
Metrics for the v4 harness. Two modes:

  pre-annotation  (no labels yet) -> system-only stats: decision distribution,
                                    presence histogram, timing, refinement
                                    use, per-tag/per-video breakdowns.
                                    These tell you "what did the system do"
                                    but NOT "was it right".

  post-annotation (labels.yaml present) -> precision, abstention precision/
                                    recall, ROC-AUC on presence, reliability
                                    bins (calibration). These tell you
                                    "was it right" — which is the headline.

Verdict vocabulary in labels_v4.yaml (per query):
  correct           — system localized the right object (or correctly said
                      NOT FOUND when the object is absent).
  wrong_object      — system found a DIFFERENT object than queried
                      (e.g. queried "thermos", returned a kettle).
  wrong_instance    — right kind of object, wrong specific instance
                      (e.g. queried "my mug", returned someone else's mug).
  partial           — right object, bad spatial bbox / mask.
  false_negative    — system said NOT FOUND but the object IS in the video.
  false_positive    — system said FOUND but the object is actually absent
                      (this is wrong_object's sibling for absent-expected queries).

Usage:
  python eval/metrics_v4.py --run-id 2026-06-14_442faa40
  python eval/metrics_v4.py --run-id 2026-06-14_442faa40 --labels eval/labels_v4.yaml
  python eval/metrics_v4.py --run-id 2026-06-14_442faa40 --split test
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
RESULTS_DIR  = EVAL_DIR / "results"
QUERY_SET    = EVAL_DIR / "query_set_v4.yaml"


# ─── Loaders ─────────────────────────────────────────────────────────────

def load_run(run_id: str) -> dict:
    """Returns {qid: result_dict} for every query that produced a result."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")
    out = {}
    for sub in sorted(run_dir.iterdir()):
        rj = sub / "result.json"
        if rj.exists():
            try:
                out[sub.name] = json.loads(rj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return out


def load_query_set() -> dict:
    return yaml.safe_load(open(QUERY_SET, "r", encoding="utf-8"))


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    return data.get("labels", {})


# ─── Stats helpers ───────────────────────────────────────────────────────

def histogram(values, n_bins=10, lo=0.0, hi=1.0):
    """Returns (edges, counts) — closed-left, open-right except last bin."""
    if not values:
        return [], []
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        if v is None:
            continue
        idx = min(int((v - lo) / width), n_bins - 1)
        idx = max(idx, 0)
        counts[idx] += 1
    edges = [lo + i * width for i in range(n_bins + 1)]
    return edges, counts


def fmt_bar(count, total, width=24):
    if total == 0:
        return ""
    filled = int(round(width * count / total))
    return "#" * filled + "." * (width - filled)


def roc_auc(y_true, y_score):
    """
    Returns AUC of the ROC curve. y_true is 0/1, y_score is continuous.
    Uses the rank-sum (Mann-Whitney U) form so it's stable for small N
    and does not require numpy.
    """
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    n_pos, n_neg = len(pos), len(neg)
    # Rank everything together (average ranks on ties).
    items = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    ranks = [0] * len(items)
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][0] == items[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1   # 1-indexed average
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, t) in zip(ranks, items) if t == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ─── Pre-annotation report ───────────────────────────────────────────────

def pre_annotation_report(results, queries):
    """
    Prints what the system DID. No claim about correctness — labels handle that.
    """
    n = len(results)
    if n == 0:
        print("No results in this run.")
        return

    decisions = Counter(r.get("decision", "?") for r in results.values())
    presences = [r.get("presence", 0.0) for r in results.values()]
    walls     = [r.get("timing", {}).get("total_s", 0.0) for r in results.values()]
    refines   = Counter(r.get("refine_iters", 0) for r in results.values())
    multi     = sum(1 for r in results.values() if r.get("num_instances", 0) > 1)
    reid_any  = sum(1 for r in results.values() if len(r.get("reid_occurrences", []) or []) > 0)

    print(f"\n------------ pre-annotation report ({n} queries) ------------")

    print("\nDecisions:")
    for d, c in decisions.most_common():
        print(f"  {d:14}  {c:3}   {fmt_bar(c, n)}")

    print("\nPresence histogram (10 bins, 0.0 -> 1.0):")
    edges, counts = histogram(presences)
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        print(f"  [{lo:.1f}-{hi:.1f})  {c:3}   {fmt_bar(c, n)}")

    print("\nRefinement iterations fired:")
    for k in sorted(refines):
        print(f"  {k} iter(s):  {refines[k]:3}   {fmt_bar(refines[k], n)}")

    print(f"\nMulti-instance returns      : {multi}/{n}")
    print(f"Re-ID found extra occurrences: {reid_any}/{n}")
    if walls:
        print(f"\nTiming (total_s)            : "
              f"median {statistics.median(walls):.0f}  "
              f"min {min(walls):.0f}  max {max(walls):.0f}  "
              f"sum {sum(walls)/60:.0f} min")

    # Per-video and per-tag breakdowns — quick sanity scan
    by_video = defaultdict(list)
    by_tag   = defaultdict(list)
    qs_by_id = {q["id"]: q for q in queries}
    for qid, r in results.items():
        q = qs_by_id.get(qid)
        if not q:
            continue
        by_video[q["video"]].append(r.get("decision", "?"))
        for t in q.get("tags", []):
            by_tag[t].append(r.get("decision", "?"))

    print("\nPer video decision counts:")
    for v in sorted(by_video):
        c = Counter(by_video[v])
        line = "  ".join(f"{k}:{n}" for k, n in c.most_common())
        print(f"  {v}  {line}")

    print("\nPer tag decision counts:")
    for t in sorted(by_tag):
        c = Counter(by_tag[t])
        line = "  ".join(f"{k}:{n}" for k, n in c.most_common())
        print(f"  {t:12}  {line}")


# ─── Post-annotation report ──────────────────────────────────────────────

CORRECT_VERDICTS = {"correct"}
WRONG_VERDICTS   = {"wrong_object", "wrong_instance", "false_negative",
                    "false_positive", "partial"}

def post_annotation_report(results, queries, labels):
    qs_by_id = {q["id"]: q for q in queries}

    labeled = []
    for qid, r in results.items():
        if qid not in labels:
            continue
        q = qs_by_id.get(qid)
        if not q:
            continue
        v = (labels[qid] or {}).get("verdict")
        if v is None:
            continue
        labeled.append((qid, q, r, v))

    n_labeled = len(labeled)
    n_total   = len(results)
    print(f"\n------------ post-annotation report ({n_labeled}/{n_total} labeled) ------------")
    if n_labeled == 0:
        print("(no labels yet — write verdicts into the labels file and re-run)")
        return

    verdict_counts = Counter(v for _, _, _, v in labeled)
    print("\nVerdict mix:")
    for v, c in verdict_counts.most_common():
        marker = " <-correct" if v in CORRECT_VERDICTS else " <-wrong" if v in WRONG_VERDICTS else ""
        print(f"  {v:18}  {c:3}   {fmt_bar(c, n_labeled)}{marker}")

    # Overall precision (excludes unknowns/partials handled separately)
    n_correct = sum(1 for _, _, _, v in labeled if v in CORRECT_VERDICTS)
    n_judged  = sum(1 for _, _, _, v in labeled if v in CORRECT_VERDICTS | WRONG_VERDICTS)
    if n_judged:
        print(f"\nOverall precision (correct / judged):  "
              f"{n_correct}/{n_judged} = {n_correct/n_judged:.3f}")

    # Abstention metrics: among queries the system DECLARED absent vs DECLARED present,
    # how often was it right (against the ground-truth `expected` field)?
    abst_correct = abst_wrong = found_correct = found_wrong = 0
    for _, q, r, v in labeled:
        expected = q.get("expected", "unknown")
        decision = r.get("decision", "")
        if expected == "absent":
            if decision in ("not_found", "abstain"):
                abst_correct += 1
            else:
                abst_wrong += 1
        elif expected == "present":
            if decision == "found":
                found_correct += 1
            elif v in CORRECT_VERDICTS:
                found_correct += 1  # we trust the human label over system's word
            else:
                found_wrong += 1
    if abst_correct + abst_wrong:
        print(f"Abstention precision (absent->NOT FOUND): "
              f"{abst_correct}/{abst_correct + abst_wrong} = "
              f"{abst_correct/(abst_correct + abst_wrong):.3f}")
    if found_correct + found_wrong:
        print(f"Detection precision  (present->FOUND):    "
              f"{found_correct}/{found_correct + found_wrong} = "
              f"{found_correct/(found_correct + found_wrong):.3f}")

    # ROC-AUC: does the presence score separate correct from wrong?
    y_true  = [1 if v in CORRECT_VERDICTS else 0 for _, _, _, v in labeled
               if v in CORRECT_VERDICTS | WRONG_VERDICTS]
    y_score = [r.get("presence", 0.0) for _, _, r, v in labeled
               if v in CORRECT_VERDICTS | WRONG_VERDICTS]
    if y_true and 0 < sum(y_true) < len(y_true):
        auc = roc_auc(y_true, y_score)
        print(f"\nROC-AUC (presence vs correctness):  {auc:.3f}   "
              f"(0.5 = chance, 1.0 = perfect ranking)")

    # Reliability diagram: bin by presence, show fraction correct per bin.
    print("\nReliability (calibration):")
    print("  presence bin    n   frac_correct  vs midpoint")
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    by_bin = defaultdict(list)
    for _, _, r, v in labeled:
        if v not in CORRECT_VERDICTS | WRONG_VERDICTS:
            continue
        p = r.get("presence", 0.0)
        idx = min(int(p * 10), 9)
        by_bin[idx].append(1 if v in CORRECT_VERDICTS else 0)
    for i, (lo, hi) in enumerate(bins):
        bs = by_bin.get(i, [])
        if not bs:
            print(f"  [{lo:.1f}-{hi:.1f})    -   -")
            continue
        frac = sum(bs) / len(bs)
        mid  = (lo + hi) / 2
        gap  = frac - mid
        sign = "+" if gap >= 0 else ""
        print(f"  [{lo:.1f}-{hi:.1f})   {len(bs):2}   {frac:.2f}        "
              f"mid={mid:.2f}  gap={sign}{gap:+.2f}")


# ─── Intervention A: Risk-Coverage (selective classification) ──────────────
#
# Following Geifman & El-Yaniv 2017 (NeurIPS) "Selective Classification for
# Deep Neural Networks" and Galil et al. 2023 (ICLR) "What can we learn from
# selective prediction... of 523 ImageNet classifiers".
#
# The Risk-Coverage (RC) curve is the right metric for systems with abstention.
# Sort predictions by confidence (high to low); at each coverage level c, the
# risk is the error rate on the top-c-fraction. A well-behaved selective
# classifier shows monotonically increasing risk with coverage. The reportable
# numbers are: precision@k (for k = 10%, 20%, ..., 100%) and AURRC (area
# under the risk-coverage curve, lower is better).
#
# This is the *correct* metric for v4's calibrated abstention claim, replacing
# the binary-correctness ROC-AUC we (correctly) abandoned during the fit
# diagnostic.

def risk_coverage_report(results, queries, labels):
    """
    Compute Risk-Coverage curve for the (presence, label) pairs.
    Prints precision@c for c in {10%, 20%, ...} and AURRC.

    The RC curve answers: "if the system abstains on the lowest-confidence
    1-c fraction of queries, what's the error rate on the remaining c?"
    """
    qs_by_id = {q["id"]: q for q in queries}

    pairs = []  # list of (presence, label) where label is binary correctness
    for qid, r in results.items():
        if qid not in labels:
            continue
        v = (labels[qid] or {}).get("verdict")
        if v in CORRECT_VERDICTS:
            y = 1
        elif v in WRONG_VERDICTS:
            y = 0
        else:
            continue   # skip / unlabeled
        pairs.append((float(r.get("presence", 0.0)), y))

    if len(pairs) < 5:
        print("\n(too few labeled samples for RC curve)")
        return

    # Sort descending by confidence so coverage = top-fraction.
    pairs.sort(key=lambda p: -p[0])
    n = len(pairs)

    print(f"\n----------- Risk-Coverage curve (selective classification) -----------")
    print(f"  N = {n} labeled samples")
    print(f"  Below: at each coverage c, what fraction of the top-c is *wrong*.")
    print(f"  Lower risk at low coverage = the system is confident-precise.")
    print()
    print(f"  coverage   n_kept   precision   risk")

    aurrc = 0.0
    prev_c = 0.0
    for c_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        k = max(1, int(round(n * c_pct / 100)))
        kept = pairs[:k]
        n_correct = sum(1 for p, y in kept if y == 1)
        precision = n_correct / k
        risk      = 1.0 - precision
        # Trapezoidal integration for AURRC
        c = c_pct / 100.0
        aurrc += (c - prev_c) * risk
        prev_c = c
        print(f"  {c_pct:>3}%       {k:>3}      {precision:.3f}       {risk:.3f}")

    print(f"\n  AURRC (lower = better) : {aurrc:.4f}")
    print(f"  Baseline (random)      : {1.0 - sum(y for _, y in pairs)/n:.4f}")
    # Identify precision plateau — the largest coverage with precision = 1.0
    plateau_c = 0
    for c_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        k = max(1, int(round(n * c_pct / 100)))
        if all(y == 1 for _, y in pairs[:k]):
            plateau_c = c_pct
    if plateau_c > 0:
        print(f"  Perfect-precision coverage : {plateau_c}%  "
              f"(useful for the paper: 'at {plateau_c}% coverage we are 100% precise')")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="run directory under eval/results/")
    p.add_argument("--labels", type=str, default=str(EVAL_DIR / "labels_v4.yaml"),
                   help="Labels YAML (default: eval/labels_v4.yaml). If missing, only "
                        "pre-annotation stats are printed.")
    p.add_argument("--split",  choices=["dev", "test"], default=None,
                   help="Restrict to dev or test queries.")
    args = p.parse_args()

    qs = load_query_set()
    queries = qs["queries"]
    if args.split:
        queries = [q for q in queries if q.get("split") == args.split]
    keep_ids = {q["id"] for q in queries}

    results = {qid: r for qid, r in load_run(args.run_id).items() if qid in keep_ids}
    labels  = {qid: v for qid, v in load_labels(Path(args.labels)).items() if qid in keep_ids}

    print(f"run_id : {args.run_id}")
    print(f"queries: {len(queries)} in scope  |  results: {len(results)}  |  labels: {len(labels)}")

    pre_annotation_report(results, queries)
    if labels:
        post_annotation_report(results, queries, labels)
        risk_coverage_report(results, queries, labels)
    else:
        print("\n(no labels file — annotate verdicts in eval/labels_v4.yaml for full metrics)")


if __name__ == "__main__":
    main()
