"""
Refit evidence fusion weights from human-annotated labels.

Uses PresenceModel.fit() (logistic regression on evidence vectors) to learn
discriminative weights from labeled data, replacing hand-initialized defaults.

Usage:
  python eval/fit_weights_v4.py \
      --labels eval/labels_2026-06-23_siglip2_78a97229.json \
      --run-id 2026-06-23_siglip2_78a97229
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_ROOT / "eval"
RESULTS_DIR  = EVAL_DIR / "results"
TEXTUAL_DIR  = PROJECT_ROOT / "TextualREN_v2"
QUERY_SET    = EVAL_DIR / "query_set_v4.yaml"

CORRECT_VERDICTS = {"correct"}
WRONG_VERDICTS   = {"wrong_object", "wrong_localization", "partial", "false_negative"}

sys.path.insert(0, str(TEXTUAL_DIR))
from evidence_fusion import PresenceModel, EvidenceVector, FEATURE_NAMES


def load_data(labels_path, run_dir, query_set):
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = labels_data.get("labels", labels_data)
    qmap = {q["id"]: q for q in query_set["queries"]}

    rows = []
    for qid, lbl in labels.items():
        verdict = lbl.get("verdict") if isinstance(lbl, dict) else str(lbl)
        if verdict not in CORRECT_VERDICTS | WRONG_VERDICTS:
            continue
        rj = run_dir / qid / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        q = qmap.get(qid, {})
        ev_raw = r.get("evidence", {})
        ev = EvidenceVector(**{k: float(ev_raw.get(k, 0.0)) for k in FEATURE_NAMES})
        rows.append({
            "qid": qid,
            "label": 1 if verdict in CORRECT_VERDICTS else 0,
            "vector": ev,
            "split": q.get("split", "dev"),
            "old_presence": float(r.get("presence", 0.0)),
            "decision": r.get("decision", ""),
            "verdict": verdict,
        })
    return rows


def simulate_decisions(rows, model, tau_accept=0.50, tau_abstain=0.20):
    results = []
    for r in rows:
        p = model.presence(r["vector"])
        if p >= tau_accept:
            dec = "found"
        elif p < tau_abstain:
            dec = "not_found"
        else:
            dec = "uncertain"
        results.append({**r, "new_presence": p, "new_decision": dec})
    return results


def print_comparison(rows, old_label="hand-init", new_label="fitted"):
    print(f"\n{'qid':>5} {'verdict':>18} {'old_p':>7} {'new_p':>7} "
          f"{'old_dec':>12} {'new_dec':>12} {'change':>8}")
    print("-" * 80)
    changes = 0
    for r in sorted(rows, key=lambda x: x["qid"]):
        old_d = r["decision"]
        new_d = r["new_decision"]
        ch = "" if old_d == new_d else "<--"
        if ch:
            changes += 1
        print(f"{r['qid']:>5} {r['verdict']:>18} {r['old_presence']:>7.3f} "
              f"{r['new_presence']:>7.3f} {old_d:>12} {new_d:>12} {ch:>8}")
    print(f"\n  {changes} decisions changed out of {len(rows)}")


def print_metrics(rows, label):
    print(f"\n  === {label} ===")
    dec_counts = Counter(r["new_decision"] for r in rows)
    print(f"  Decisions: {dict(dec_counts)}")

    for dec in ["found", "uncertain", "not_found"]:
        subset = [r for r in rows if r["new_decision"] == dec]
        if not subset:
            continue
        n_correct = sum(1 for r in subset if r["label"] == 1)
        prec = n_correct / len(subset) if subset else 0
        print(f"    {dec:12s}: n={len(subset):3d}  correct={n_correct:3d}  "
              f"precision={prec:.3f}")

    n_correct_total = sum(1 for r in rows if r["label"] == 1)
    found_correct = sum(1 for r in rows
                       if r["new_decision"] == "found" and r["label"] == 1)
    recall = found_correct / n_correct_total if n_correct_total else 0
    found_wrong = sum(1 for r in rows
                     if r["new_decision"] == "found" and r["label"] == 0)
    n_found = sum(1 for r in rows if r["new_decision"] == "found")
    precision = (n_found - found_wrong) / n_found if n_found else 0
    print(f"    Recall (correct found/total correct): {recall:.3f}")
    print(f"    Precision (correct found/total found): {precision:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--C", type=float, default=1.0,
                   help="Regularization (lower=stronger). Default 1.0")
    p.add_argument("--tau-accept", type=float, default=0.50)
    p.add_argument("--tau-abstain", type=float, default=0.20)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write weights file, just simulate")
    args = p.parse_args()

    labels_path = Path(args.labels)
    run_dir = RESULTS_DIR / args.run_id
    qs = yaml.safe_load(QUERY_SET.read_text(encoding="utf-8"))

    rows = load_data(labels_path, run_dir, qs)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"Loaded {len(rows)} samples (dev={len(dev)}, test={len(test)})")
    print(f"  dev  pos={sum(r['label'] for r in dev)}  "
          f"neg={sum(1-r['label'] for r in dev)}")
    print(f"  test pos={sum(r['label'] for r in test)}  "
          f"neg={sum(1-r['label'] for r in test)}")

    # --- Baseline: hand-initialized weights ---
    baseline = PresenceModel(weights_path=None)
    print("\n" + "=" * 70)
    print("BASELINE (hand-initialized weights)")
    print("=" * 70)
    print(f"  weights: { {k: f'{v:.1f}' for k, v in baseline.w.items()} }")
    print(f"  bias: {baseline.b:.1f}")

    base_sim = simulate_decisions(rows, baseline, args.tau_accept, args.tau_abstain)
    print_metrics([r for r in base_sim if r["split"] == "dev"], "Dev baseline")
    print_metrics([r for r in base_sim if r["split"] == "test"], "Test baseline")

    # --- Fit new weights on DEV only ---
    weights_path = TEXTUAL_DIR / "configs" / "presence_weights.json"
    fitted = PresenceModel(weights_path=str(weights_path) if not args.dry_run else None)

    dev_vectors = [r["vector"] for r in dev]
    dev_labels = [r["label"] for r in dev]

    # Use balanced class weights since pos:neg ratio is ~4.6:1
    data = fitted.fit(dev_vectors, dev_labels, save=not args.dry_run,
                      C=args.C, class_weight="balanced")

    print("\n" + "=" * 70)
    print("FITTED WEIGHTS (logistic regression on dev split)")
    print("=" * 70)
    print(f"  weights:")
    for k in FEATURE_NAMES:
        old_w = baseline.w[k]
        new_w = fitted.w[k]
        arrow = ">>>" if abs(new_w - old_w) > 1.0 else ""
        print(f"    {k:8s}: {old_w:+7.2f} -> {new_w:+7.2f}  {arrow}")
    print(f"  bias: {baseline.b:+.2f} -> {fitted.b:+.2f}")

    fit_sim = simulate_decisions(rows, fitted, args.tau_accept, args.tau_abstain)
    print_metrics([r for r in fit_sim if r["split"] == "dev"], "Dev fitted")
    print_metrics([r for r in fit_sim if r["split"] == "test"], "Test fitted")

    # --- Sweep tau_accept to find best operating point ---
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (fitted weights, dev split)")
    print("=" * 70)
    print(f"  {'tau_acc':>7} {'tau_abs':>7} {'found':>6} {'unc':>6} {'nf':>6} "
          f"{'prec':>6} {'recall':>6} {'F1':>6}")
    best_f1 = 0
    best_tau = (0.5, 0.2)
    for tau_a_10 in range(25, 70, 5):
        tau_a = tau_a_10 / 100.0
        for tau_b_10 in range(10, int(tau_a_10), 5):
            tau_b = tau_b_10 / 100.0
            sim = simulate_decisions(dev, fitted, tau_a, tau_b)
            n_found = sum(1 for r in sim if r["new_decision"] == "found")
            n_unc = sum(1 for r in sim if r["new_decision"] == "uncertain")
            n_nf = sum(1 for r in sim if r["new_decision"] == "not_found")
            found_correct = sum(1 for r in sim
                               if r["new_decision"] == "found" and r["label"] == 1)
            total_correct = sum(r["label"] for r in sim)
            prec = found_correct / n_found if n_found else 0
            rec = found_correct / total_correct if total_correct else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_tau = (tau_a, tau_b)
                print(f"  {tau_a:>7.2f} {tau_b:>7.2f} {n_found:>6} {n_unc:>6} "
                      f"{n_nf:>6} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} *")

    print(f"\n  Best F1={best_f1:.3f} at tau_accept={best_tau[0]:.2f}, "
          f"tau_abstain={best_tau[1]:.2f}")

    # --- Per-query comparison at best thresholds ---
    print("\n" + "=" * 70)
    print(f"PER-QUERY COMPARISON (tau_accept={best_tau[0]:.2f}, "
          f"tau_abstain={best_tau[1]:.2f})")
    print("=" * 70)
    fit_best = simulate_decisions(rows, fitted, best_tau[0], best_tau[1])
    print_comparison(fit_best)

    if not args.dry_run:
        print(f"\nWeights saved to: {weights_path}")


if __name__ == "__main__":
    main()
