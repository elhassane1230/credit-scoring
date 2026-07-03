import numpy as np

from creditscore.evaluation.metrics import (
    best_threshold_for_f1, compute_metrics,
)


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, prob, threshold=0.5)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_counts():
    y = np.array([0, 1, 1, 0])
    prob = np.array([0.9, 0.9, 0.1, 0.1])  # 1 tp, 1 fn, 1 fp, 1 tn
    cm = compute_metrics(y, prob, 0.5)["confusion_matrix"]
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_ks_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = rng.uniform(0, 1, 200)
    ks = compute_metrics(y, prob, 0.5)["ks"]
    assert 0.0 <= ks <= 1.0


def test_best_threshold_improves_f1():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=500) < 0.2).astype(int)         # imbalanced
    prob = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
    t, f1 = best_threshold_for_f1(y, prob)
    assert 0.05 <= t <= 0.95
    f1_default = compute_metrics(y, prob, 0.5)["f1"]
    assert f1 >= f1_default  # tuned threshold is at least as good as 0.5
