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


def select_threshold(y_true: np.ndarray, p_def: np.ndarray, target_recall: float | None = 0.99,
                     fn_cost: float = 5.0, fp_cost: float = 1.0) -> tuple[float, str]:
    """Choose the decision threshold on the *validation* set (never on test).

    Policy (cost-sensitive with a safety constraint):

    1. Candidate thresholds are the mid-points between consecutive distinct scores, so a threshold
       never sits exactly on a validation sample.
    2. Keep candidates whose defect recall >= ``target_recall`` (defect-escape budget); if none
       meets it, keep all.
    3. Pick the minimum expected cost ``fn_cost * FN + fp_cost * FP`` – a missed defect is assumed
       ``fn_cost/fp_cost`` times as expensive as a false alarm.
    4. Ties (common on small, well-separated validation sets) are broken by the *largest gap*
       between neighbouring scores, i.e. the threshold with the biggest safety margin on both sides.
    """
    y_true = np.asarray(y_true).astype(int)
    p_def = np.asarray(p_def, dtype=float)
    s = np.unique(np.concatenate([[0.0, 1.0], p_def]))
    mids = (s[:-1] + s[1:]) / 2
    gaps = s[1:] - s[:-1]
    n_pos = max(int(y_true.sum()), 1)
    rows = []
    for t, g in zip(mids, gaps):
        pred = p_def >= t
        fn = int(((~pred) & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        rows.append((t, g, 1 - fn / n_pos, fn_cost * fn + fp_cost * fp))
    feasible = [r for r in rows if target_recall is None or r[2] >= target_recall]
    policy = f"min cost (FN={fn_cost:g}, FP={fp_cost:g})"
    if target_recall is not None:
        policy += f", recall>={target_recall}"
    if not feasible:
        feasible, policy = rows, policy + " (recall target unreachable)"
    best_cost = min(r[3] for r in feasible)
    t, *_ = max((r for r in feasible if r[3] == best_cost), key=lambda r: r[1])
    return float(t), policy
