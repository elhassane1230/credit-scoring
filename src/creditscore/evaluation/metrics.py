"""Evaluation metrics tailored to imbalanced credit scoring.

The report tracks three headline metrics, all computed here:
  * Recall (of the default class) — minimise false negatives (approving a loan
    that later defaults is the costly error).
  * F1 — precision/recall balance under class imbalance.
  * AUC-ROC — threshold-independent ranking/discrimination.

Also included: precision, average precision (PR-AUC, more informative than ROC
under heavy imbalance), the confusion matrix, and the Kolmogorov-Smirnov (KS)
statistic — a standard credit-risk separability measure.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All headline metrics at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": _ks_statistic(y_true, y_prob),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Max separation between cumulative distributions of good/bad scores."""
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    pos_total = y_sorted.sum()
    neg_total = len(y_sorted) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.0
    cum_pos = np.cumsum(y_sorted) / pos_total
    cum_neg = np.cumsum(1 - y_sorted) / neg_total
    return float(np.max(np.abs(cum_pos - cum_neg)))


def best_threshold_for_f1(y_true, y_prob) -> tuple[float, float]:
    """Grid-search the probability threshold that maximises F1 on given data.

    Useful because the default 0.5 is rarely optimal under class imbalance;
    the operating point should be chosen on validation data, not test.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)
