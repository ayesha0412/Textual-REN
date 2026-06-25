"""
Fit logistic presence weights on labeled v4 results.

What it does:
  1. Loads a labels JSON (exported from annotate_v4.py's HTML tool) and the
     corresponding result.json files from a run dir.
  2. Joins by qid → (evidence_vector, label) pairs.
  3. Splits by the query_set's `split` field (dev = training, test = held out).
  4. Calls PresenceModel.fit() on dev only.
  5. Evaluates on test: ROC-AUC, accuracy, reliability bins, per-feature
     weight importances.
  6. Writes the fitted weights to TextualREN_v2/configs/presence_weights.json
     (unless --dry-run).
  7. Prints a report comparing fitted weights to hand-set defaults.

Verdict → binary label mapping:
  correct           → 1   (system right)
  wrong_object, wrong_localization, partial, false_negative → 0  (system wrong)
  skip              → excluded

Usage:
  # Dry-run (don't overwrite weights):
  python eval/fit_presence_weights.py \
      --labels eval/labels_2026-06-17_siglip2_e01ce275.json \
      --run-id 2026-06-17_siglip2_e01ce275 --dry-run
  # Real fit + save:
  python eval/fit_presence_weights.py \
      --labels eval/labels_2026-06-17_siglip2_e01ce275.json \
      --run-id 2026-06-17_siglip2_e01ce275
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
RESULTS_DIR  = EVAL_DIR / "results"
TEXTUAL_DIR  = PROJECT_ROOT / "TextualREN_v2"
QUERY_SET    = EVAL_DIR / "query_set_v4.yaml"
WEIGHTS_PATH = TEXTUAL_DIR / "configs" / "presence_weights.json"


# Map annotation verdicts to binary labels for the fit.
CORRECT_VERDICTS = {"correct"}
WRONG_VERDICTS   = {"wrong_object", "wrong_localization", "partial",
                    "false_negative"}


def load_join(labels_path: Path, run_dir: Path, query_set: dict):
    """Returns list of dicts: {qid, label, vector, split, video, query, decision}"""
    sys.path.insert(0, str(TEXTUAL_DIR))
    from evidence_fusion import EvidenceVector, FEATURE_NAMES  # noqa

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = labels_data.get("labels", labels_data)  # accept both formats
    qmap = {q["id"]: q for q in query_set["queries"]}

    joined = []
    counts = Counter()
    for qid, lbl in labels.items():
        verdict = lbl.get("verdict") if isinstance(lbl, dict) else str(lbl)
        counts[verdict] += 1
        if verdict not in CORRECT_VERDICTS | WRONG_VERDICTS:
            continue  # skip / unknown
        rj = run_dir / qid / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ev = r.get("evidence", {})
        vec = EvidenceVector(**{k: float(ev.get(k, 0.0)) for k in FEATURE_NAMES})
        q = qmap.get(qid, {})
        joined.append({
            "qid": qid,
            "label": 1 if verdict in CORRECT_VERDICTS else 0,
            "vector": vec,
            "split": q.get("split", "dev"),
            "video": q.get("video", ""),
            "query": q.get("text", ""),
            "decision": r.get("decision", ""),
            "presence": r.get("presence", 0.0),
            "verdict": verdict,
        })
    return joined, dict(counts), FEATURE_NAMES


def compute_metrics(rows, model):
    """Returns precision, recall, ROC-AUC, mean cross-entropy on rows."""
    if not rows:
        return None
    sys.path.insert(0, str(TEXTUAL_DIR))
    y_true = [r["label"] for r in rows]
    y_score = [model.presence(r["vector"]) for r in rows]
    # Confusion at the current decision (decisions were already taken by the
    # system; the fitted model's accuracy is a separate question — we re-decide
    # using the fitted posterior with the same τ_accept threshold.)
    tau = 0.5
    y_pred = [1 if s >= tau else 0 for s in y_score]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy  = (tp + tn) / len(rows)

    # ROC-AUC via rank-sum / Mann-Whitney
    pos_scores = [s for s, t in zip(y_score, y_true) if t == 1]
    neg_scores = [s for s, t in zip(y_score, y_true) if t == 0]
    if pos_scores and neg_scores:
        items = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores])
        ranks = list(range(1, len(items) + 1))
        # average-ranks on ties
        i = 0
        while i < len(items):
            j = i
            while j + 1 < len(items) and items[j + 1][0] == items[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[k] = avg
            i = j + 1
        rank_sum_pos = sum(r for r, (_, t) in zip(ranks, items) if t == 1)
        u = rank_sum_pos - len(pos_scores) * (len(pos_scores) + 1) / 2
        auc = u / (len(pos_scores) * len(neg_scores))
    else:
        auc = float("nan")

    # Mean cross-entropy
    eps = 1e-9
    ce = -sum(
        t * math.log(max(eps, s)) + (1 - t) * math.log(max(eps, 1 - s))
        for s, t in zip(y_score, y_true)
    ) / len(rows)

    # Reliability: per-decile fraction correct
    bins = defaultdict(list)
    for s, t in zip(y_score, y_true):
        bins[min(int(s * 10), 9)].append(t)
    rel = []
    for b in range(10):
        vals = bins.get(b, [])
        if vals:
            rel.append((b * 0.1, (b + 1) * 0.1, len(vals),
                        sum(vals) / len(vals)))
    return {
        "n": len(rows), "accuracy": accuracy,
        "precision": precision, "recall": recall, "auc": auc,
        "cross_entropy": ce, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "reliability": rel,
    }


def print_metrics(name: str, m):
    if m is None:
        print(f"  {name}: (no samples)")
        return
    print(f"  {name}: n={m['n']}")
    print(f"    accuracy   : {m['accuracy']:.3f}")
    print(f"    precision  : {m['precision']:.3f}")
    print(f"    recall     : {m['recall']:.3f}")
    print(f"    ROC-AUC    : {m['auc']:.3f}   (0.5 = chance, 1.0 = perfect)")
    print(f"    cross-ent  : {m['cross_entropy']:.3f}")
    print(f"    confusion  : TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"    reliability (presence-bin → frac correct):")
    for lo, hi, n, frac in m["reliability"]:
        midpoint = (lo + hi) / 2
        gap = frac - midpoint
        print(f"      [{lo:.1f}–{hi:.1f})  n={n:3}  frac={frac:.2f}  "
              f"gap_vs_mid={gap:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="Exported labels JSON.")
    p.add_argument("--run-id", required=True,
                    help="Subdir under eval/results/ where the result.jsons live.")
    p.add_argument("--dry-run", action="store_true",
                    help="Fit and report; DO NOT write presence_weights.json.")
    p.add_argument("--output", type=str, default=str(WEIGHTS_PATH),
                    help="Where to save fitted weights.")
    args = p.parse_args()

    sys.path.insert(0, str(TEXTUAL_DIR))
    from evidence_fusion import EvidenceVector, PresenceModel, FEATURE_NAMES

    labels_path = Path(args.labels)
    run_dir     = RESULTS_DIR / args.run_id
    qs          = yaml.safe_load(QUERY_SET.read_text(encoding="utf-8"))

    joined, counts, feature_names = load_join(labels_path, run_dir, qs)
    print(f"\n=== Loaded labels ===")
    print(f"  labels file    : {labels_path}")
    print(f"  run dir        : {run_dir}")
    print(f"  total labeled  : {sum(counts.values())}")
    print(f"  verdict counts : {counts}")
    print(f"  used for fit   : {len(joined)} "
          f"(positives: {sum(r['label'] for r in joined)}, "
          f"negatives: {sum(1-r['label'] for r in joined)})")

    if len(joined) < 10:
        sys.exit("\nNeed at least 10 labeled samples to fit reliably. "
                 "Label more queries and re-run.")

    # Stratified by split.
    dev  = [r for r in joined if r["split"] == "dev"]
    test = [r for r in joined if r["split"] == "test"]
    print(f"\n  dev set   : {len(dev)}  (used for fit)")
    print(f"  test set  : {len(test)} (held out)")
    if not dev:
        sys.exit("No dev-split labels — every query needs split:'dev'|'test' in the YAML.")

    # Hand-set baseline metrics (before refit) on test split.
    print(f"\n=== Hand-set weights baseline (before refit) ===")
    baseline = PresenceModel(weights_path=None)  # forces hand-init defaults
    print(f"  weights: {baseline.w}")
    print(f"  bias   : {baseline.b}")
    print_metrics("dev ", compute_metrics(dev,  baseline))
    print_metrics("test", compute_metrics(test, baseline))

    # Fit on dev with class balancing + stronger regularization.
    # Class balancing is essential because positives outnumber negatives
    # ~4:1 in our annotation data — without it, sklearn finds the trivial
    # solution: predict the majority class (everything "correct"), which
    # gives recall=1.0 but ROC-AUC near chance. C=0.1 forces the fit to
    # use features parsimoniously instead of letting bias do all the work.
    print(f"\n=== Fitting logistic regression on dev split "
          f"(class_weight='balanced', C=0.1) ===")
    fit_model = PresenceModel(weights_path=None)
    fit_data = fit_model.fit([r["vector"] for r in dev],
                              [r["label"] for r in dev], save=False,
                              C=0.1, class_weight='balanced')
    print(f"  fitted weights:")
    for k in feature_names:
        delta = fit_model.w[k] - baseline.w[k]
        marker = " (decreased)" if delta < -0.5 else " (increased)" if delta > 0.5 else ""
        print(f"    {k:8} = {fit_model.w[k]:+.3f}  "
              f"(was {baseline.w[k]:+.2f}, Δ={delta:+.2f}){marker}")
    print(f"  bias   = {fit_model.b:+.3f}  (was {baseline.b:+.2f}, "
          f"Δ={fit_model.b - baseline.b:+.2f})")

    # Re-score with fitted weights.
    print(f"\n=== Fitted-weight performance ===")
    print_metrics("dev ", compute_metrics(dev,  fit_model))
    print_metrics("test", compute_metrics(test, fit_model))

    # Write the weights (unless dry-run).
    if not args.dry_run:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Add provenance metadata.
        fit_data["fitted"] = True
        fit_data["fit_run_id"] = args.run_id
        fit_data["fit_labels"] = str(labels_path)
        fit_data["fit_n_dev"]  = len(dev)
        fit_data["fit_n_test"] = len(test)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(fit_data, f, indent=2)
        print(f"\nFitted weights saved to: {out}")
        print(f"   (back up the old presence_weights.json if you want a baseline copy.)")
    else:
        print(f"\n[dry-run] Did NOT write weights. Re-run without --dry-run to commit.")


if __name__ == "__main__":
    main()
