"""Classification metrics with *defective* as the positive class.

Terminology used throughout the project:

* **False negative (FN)** – a defective part predicted *normal*  -> escapes to the customer.
  This is normally the expensive error (returns, warranty, safety).
* **False positive (FP)** – a normal part predicted *defective* -> unnecessary rejection / manual
  re-inspection (scrap or labour cost).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> float:
    """ECE of the predicted-class confidence."""
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, p_def: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_def = np.asarray(p_def, dtype=float)
    y_pred = (p_def >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float((tp + tn) / len(y_true)),
        "balanced_accuracy": float((rec[0] + rec[1]) / 2),
        "precision": float(prec[1]),  # defective = positive class
        "recall": float(rec[1]),
        "f1": float(f1[1]),
        "specificity": float(rec[0]),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "per_class": {
            "normal": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]),
                       "support": int(support[0])},
            "defective": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]),
                          "support": int(support[1])},
        },
        "macro_f1": float(f1.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": float(roc_auc_score(y_true, p_def)) if both else None,
        "pr_auc": float(average_precision_score(y_true, p_def)) if both else None,
        "ece": expected_calibration_error(y_true, p_def),
    }


def select_threshold(y_true: np.ndarray, p_def: np.ndarray, target_recall: float | None) -> tuple[float, str]:
    """Choose the decision threshold on the *validation* set.

    Policy: the **highest** threshold whose recall on defective parts is >= ``target_recall``
    (i.e. the fewest false alarms subject to a defect-escape budget). Falls back to the
    F1-optimal threshold if ``target_recall`` is None.
    """
    y_true = np.asarray(y_true).astype(int)
    p_def = np.asarray(p_def, dtype=float)
    candidates = np.unique(np.concatenate([p_def, [0.5]]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        m = compute_metrics(y_true, p_def, t)
        if m["f1"] > best_f1:
            best_t, best_f1 = float(t), m["f1"]
    if target_recall is None:
        return best_t, "max_f1"
    ok = [float(t) for t in candidates if compute_metrics(y_true, p_def, t)["recall"] >= target_recall]
    if not ok:
        return best_t, "max_f1 (target recall unreachable)"
    # Put the threshold half-way to the next lower score to avoid sitting exactly on a sample.
    t = max(ok)
    lower = p_def[p_def < t]
    t_mid = float((t + lower.max()) / 2) if lower.size else t
    return t_mid, f"recall>={target_recall}"
