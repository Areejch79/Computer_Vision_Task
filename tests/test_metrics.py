import numpy as np
import pytest

pytest.importorskip("sklearn")

from defect_detection.evaluation.metrics import compute_metrics, select_threshold  # noqa: E402


def test_confusion_matrix_and_rates():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["false_negative_rate"] == pytest.approx(0.25)
    assert m["specificity"] == pytest.approx(0.75)


def test_threshold_selection_meets_recall_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)].astype(int)
    p = np.clip(np.r_[rng.normal(0.75, 0.15, 200), rng.normal(0.3, 0.15, 200)], 0, 1)
    t, policy = select_threshold(y, p, target_recall=0.99)
    assert compute_metrics(y, p, t)["recall"] >= 0.99
    assert policy.startswith("recall")
    t_f1, _ = select_threshold(y, p, None)
    assert t <= t_f1  # a recall constraint can only push the threshold down
