#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run compliance triage over a CSV of features (blind to gold labels),
then compare predictions vs. expected triage labels for evaluation.

Input CSV must include:
  - feature_name
  - feature_description
Optional:
  - feature_id           (if absent, ROW-<n> is assigned)
  - triage_label         (gold labels: requires_compliance | uncertain | no_compliance)

Outputs:
  - <out_csv>            (default: outputs/predictions.csv)
  - outputs/eval_summary.json  (metrics & confusion matrix)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- make sure we can import from project root ---
THIS_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = THIS_DIR.parent                      # repo root
sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------------------------------

try:
    # If your classifier is under compliance/simple_classifier.py (as in your project)
    from compliance.simple_classifier import SimpleComplianceClassifier
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import compliance.simple_classifier.\n"
        "Make sure you run this from the repo root and that compliance/ exists with __init__.py.\n"
        f"Details: {e}"
    )


# ----- Helpers ---------------------------------------------------------------

GOLD_TO_MODEL = {
    # map your triage_label -> model label space
    "requires_compliance": "yes",
    "no_compliance": "no",
    "uncertain": "uncertain",
}

MODEL_LABELS = ("yes", "no", "uncertain")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _map_gold_label(raw: str) -> str:
    key = _norm(raw).lower().replace("-", "_").replace(" ", "_")
    return GOLD_TO_MODEL.get(key, "")  # return empty if unknown


def _read_features(csv_path: Path) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Reads features from CSV. Returns:
      - features: list of dicts with feature_id, feature_name, feature_description
      - gold_map: feature_id -> gold label mapped into model space ('yes'/'no'/'uncertain') if present
    """
    rows: List[Dict] = []
    gold_map: Dict[str, str] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV appears to have no header row.")

        required = {"feature_name", "feature_description"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for i, row in enumerate(reader, start=1):
            fid = _norm(row.get("feature_id")) or f"ROW-{i}"
            name = _norm(row.get("feature_name"))
            desc = _norm(row.get("feature_description"))
            if not name and not desc:
                # skip empty lines
                continue

            rows.append({
                "feature_id": fid,
                "feature_name": name,
                "feature_description": desc,
            })

            gold_raw = row.get("triage_label")
            if gold_raw is not None:
                mapped = _map_gold_label(gold_raw)
                if mapped:
                    gold_map[fid] = mapped

    if not rows:
        raise ValueError("No usable rows found in CSV.")

    return rows, gold_map


def _best_passage(passages: List[Dict]) -> Dict:
    if not passages:
        return {}
    return max(passages, key=lambda p: p.get("relevance_score", 0.0))


def _write_predictions(out_path: Path, results, gold_map: Dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature_id", "feature_name",
            "predicted_label", "confidence", "risk_level",
            "expected_label", "match",
            "applicable_regulations", "reasoning",
            "top_jurisdiction", "top_law", "top_relevance"
        ])
        for r in results:
            top = _best_passage(r.retrieved_passages)
            expected = gold_map.get(r.feature_id, "")
            predicted = r.needs_compliance
            match = (expected.lower() == predicted.lower()) if expected else ""

            w.writerow([
                r.feature_id,
                r.title,
                predicted,
                f"{r.confidence_score:.2f}",
                r.risk_level,
                expected,
                match,
                "; ".join(r.applicable_regulations or []),
                (r.reasoning or "").replace("\n", " ").strip(),
                (top.get("jurisdiction", "") or ""),
                (top.get("law", "") or ""),
                f"{top.get('relevance_score', 0.0):.3f}" if top else ""
            ])


def _compute_metrics(preds: List[str], golds: List[str]) -> Dict:
    """
    Compute accuracy, per-class precision/recall/F1, and confusion matrix.
    Only includes rows where gold label is present & valid.
    """
    # Filter out rows without gold
    pairs = [(p, g) for p, g in zip(preds, golds) if g in MODEL_LABELS]
    if not pairs:
        return {"note": "No gold labels available; metrics not computed."}

    labels = MODEL_LABELS
    # Confusion counts: cm[gold][pred]
    cm = {g: {p: 0 for p in labels} for g in labels}

    for p, g in pairs:
        if p not in labels:
            # treat OOD predictions as 'uncertain' bucket, or just skip
            p = "uncertain"
        cm[g][p] += 1

    # Totals
    total = sum(sum(cm[g].values()) for g in labels)
    correct = sum(cm[l][l] for l in labels)
    accuracy = correct / total if total else 0.0

    # Per-class metrics
    per_class = {}
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[g][l] for g in labels if g != l)
        fn = sum(cm[l][p] for p in labels if p != l)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[l] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(cm[l].values()),
        }

    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)

    summary = {
        "total_evaluated": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    return summary


def _write_summary_json(summary_path: Path, summary_obj: Dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")


# ----- Main -----------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Run compliance triage on a CSV and evaluate against gold labels.")
    p.add_argument("--csv", required=True,
                   help="Path to CSV with columns: feature_name, feature_description (optional: feature_id, triage_label)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "predictions.csv"),
                   help="Where to write predictions CSV (default: outputs/predictions.csv)")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()
    summary_path = PROJECT_ROOT / "outputs" / "eval_summary.json"

    # 1) Load data (features + gold, but model won't see gold)
    features, gold_map = _read_features(csv_path)

    # 2) Run model (rule-based + retrieval)
    clf = SimpleComplianceClassifier()
    results = clf.analyze_batch(features)

    # 3) Write row-level predictions (with gold + match column)
    _write_predictions(out_path, results, gold_map)
    print(f"✅ wrote predictions to: {out_path}")

    # 4) Build metrics if gold is available
    preds = [r.needs_compliance for r in results]
    golds = [gold_map.get(r.feature_id, "") for r in results]
    summary = _compute_metrics(preds, golds)

    # 5) Print and save summary
    if "total_evaluated" in summary:
        print("\n=== Evaluation Summary ===")
        print(f"Total evaluated: {summary['total_evaluated']}")
        print(f"Accuracy:       {summary['accuracy']:.4f}")
        print(f"Macro F1:       {summary['macro_f1']:.4f}")
        print("Per-class:")
        for label, stats in summary["per_class"].items():
            print(f"  {label:9s}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  F1={stats['f1']:.3f}  (support={stats['support']})")
        print("\nConfusion matrix (gold rows x pred cols):")
        cm = summary["confusion_matrix"]
        hdr = "           " + "  ".join(f"{l:9s}" for l in MODEL_LABELS)
        print(hdr)
        for g in MODEL_LABELS:
            row = f"{g:9s}  " + "  ".join(f"{cm[g][p]:9d}" for p in MODEL_LABELS)
            print(row)
    else:
        print("\n(No gold labels found; skipped metrics.)")

    _write_summary_json(summary_path, summary)
    print(f"📄 wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
