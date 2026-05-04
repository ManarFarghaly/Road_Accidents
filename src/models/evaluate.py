from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


# StringIndexer frequencyDesc on Accident_Severity:
#   0 → Slight (~85 %),  1 → Serious (~14 %),  2 → Fatal (~1 %)
_LABEL_MAP = {0.0: "Slight", 1.0: "Serious", 2.0: "Fatal"}
_CLASS_LABELS = [_LABEL_MAP[k] for k in sorted(_LABEL_MAP)]


def _cohen_kappa(conf_arr: np.ndarray) -> float:
    """
    Compute Cohen's Kappa from a confusion matrix.
    κ = (p_o - p_e) / (1 - p_e)
    p_o = observed agreement (accuracy = diagonal sum / total)
    p_e = expected agreement by chance = Σ (row_i * col_i) / n²
    """
    n = conf_arr.sum()
    if n == 0:
        return 0.0
    p_o = np.diag(conf_arr).sum() / n
    p_e = (conf_arr.sum(axis=1) * conf_arr.sum(axis=0)).sum() / (n ** 2)
    return float((p_o - p_e) / (1.0 - p_e)) if p_e != 1.0 else 1.0


def evaluate_model(model_name: str, model, train_df, test_df) -> dict:
    """
    Compute accuracy, weighted F1, Cohen's Kappa, confusion matrix, and
    per-class precision/recall/F1 on both train and test splits.

    Returns a dict shaped:
        {
          "train": { "accuracy": …, "weighted_f1": …, "cohen_kappa": …,
                     "confusion_matrix": [[…], …], "class_labels": […],
                     "per_class": { "Slight": {…}, "Serious": {…}, "Fatal": {…} } },
          "test":  { … same structure … }
        }
    """
    split_results: dict = {}
    for split_name, df in [("train", train_df), ("test", test_df)]:
        preds = model.transform(df)
        pred_rdd = (
            preds.select("prediction", "label")
                 .rdd.map(lambda r: (float(r.prediction), float(r.label)))
        )
        mm       = MulticlassMetrics(pred_rdd)
        conf_arr = mm.confusionMatrix().toArray()
        conf     = conf_arr.tolist()

        per_class: dict = {}
        for idx in sorted(_LABEL_MAP):
            lbl = _LABEL_MAP[idx]
            try:
                per_class[lbl] = {
                    "precision": round(float(mm.precision(idx)), 4),
                    "recall":    round(float(mm.recall(idx)),    4),
                    "f1":        round(float(mm.fMeasure(idx)),  4),
                }
            except Exception:
                per_class[lbl] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        split_results[split_name] = {
            "accuracy":         round(float(mm.accuracy),           4),
            "weighted_f1":      round(float(mm.weightedFMeasure()),  4),
            "cohen_kappa":      round(_cohen_kappa(conf_arr),        4),
            "confusion_matrix": [[int(v) for v in row] for row in conf],
            "class_labels":     _CLASS_LABELS,
            "per_class":        per_class,
        }
        sr = split_results[split_name]
        print(
            f"  [{model_name}] {split_name}: "
            f"acc={sr['accuracy']:.4f}  f1={sr['weighted_f1']:.4f}  "
            f"kappa={sr['cohen_kappa']:.4f}"
        )

    return split_results


def save_model_metrics(model_name: str, split_results: dict, reports_dir: Path) -> None:
    """
    Save one model's metrics to reports/metrics_{model_name}.json.
    """
    safe_name = model_name.replace(" ", "_")
    path = reports_dir / f"metrics_{safe_name}.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name":    model_name,
        "generated_at":  datetime.now().isoformat(),
        "train":         split_results.get("train", {}),
        "test":          split_results.get("test",  {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[evaluate] {model_name} metrics saved → {path}")
