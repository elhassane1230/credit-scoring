"""Train, tune and select credit-scoring models.

For each model: run a cross-validated randomized hyperparameter search
(F1-scored, to respect class imbalance) on the training set, then choose the
decision threshold that maximises F1 on the *validation* set (not test), and
finally report all metrics on the held-out *test* set.

Returns a structured result per model plus the selected champion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ..config import TrainConfig
from ..data.etl import Dataset
from ..evaluation.metrics import best_threshold_for_f1, compute_metrics
from .pipeline import MODEL_NAMES, SEARCH_SPACES, build_model


@dataclass
class ModelResult:
    name: str
    estimator: object                       # fitted sklearn Pipeline
    best_params: dict
    threshold: float
    val_f1: float
    test_metrics: dict
    cv_best_score: float


@dataclass
class TrainingReport:
    results: dict[str, ModelResult] = field(default_factory=dict)
    champion: str = ""

    def leaderboard(self) -> list[dict]:
        rows = [{
            "model": r.name,
            "test_recall": round(r.test_metrics["recall"], 4),
            "test_f1": round(r.test_metrics["f1"], 4),
            "test_roc_auc": round(r.test_metrics["roc_auc"], 4),
            "test_pr_auc": round(r.test_metrics["pr_auc"], 4),
            "test_ks": round(r.test_metrics["ks"], 4),
            "threshold": round(r.threshold, 3),
        } for r in self.results.values()]
        return sorted(rows, key=lambda d: d["test_roc_auc"], reverse=True)


def _scale_pos_weight(y) -> float:
    y = np.asarray(y)
    pos = max(1, int(y.sum()))
    neg = int(len(y) - y.sum())
    return neg / pos


def train_one(name: str, ds: Dataset, cfg: TrainConfig,
              n_iter: int = 20) -> ModelResult:
    spw = _scale_pos_weight(ds.y_train) if name == "xgboost" else None
    model = build_model(name, class_weight="balanced", scale_pos_weight=spw,
                        random_state=cfg.random_state)

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                         random_state=cfg.random_state)
    search = RandomizedSearchCV(
        model, SEARCH_SPACES[name], n_iter=n_iter, scoring=cfg.scoring,
        cv=cv, n_jobs=cfg.n_jobs, random_state=cfg.random_state, refit=True,
    )
    search.fit(ds.X_train, ds.y_train)
    best = search.best_estimator_

    # Choose operating threshold on validation, evaluate on test.
    val_prob = best.predict_proba(ds.X_val)[:, 1]
    threshold, val_f1 = best_threshold_for_f1(ds.y_val, val_prob)
    test_prob = best.predict_proba(ds.X_test)[:, 1]
    test_metrics = compute_metrics(ds.y_test, test_prob, threshold)

    return ModelResult(
        name=name, estimator=best, best_params=search.best_params_,
        threshold=threshold, val_f1=val_f1, test_metrics=test_metrics,
        cv_best_score=float(search.best_score_),
    )


def train_all(ds: Dataset, cfg: TrainConfig | None = None,
              models: list[str] | None = None, n_iter: int = 20) -> TrainingReport:
    cfg = cfg or TrainConfig()
    models = models or MODEL_NAMES
    report = TrainingReport()
    for name in models:
        report.results[name] = train_one(name, ds, cfg, n_iter=n_iter)
    # Champion = best test ROC-AUC (discrimination), tie-broken by F1.
    report.champion = max(
        report.results,
        key=lambda k: (report.results[k].test_metrics["roc_auc"],
                       report.results[k].test_metrics["f1"]),
    )
    return report
