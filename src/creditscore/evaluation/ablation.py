"""Ablation studies for the credit-scoring model.

Three experiments, all reported in the paper/README:

  A. Model family — XGBoost vs Logistic Regression: does capturing non-linear
     structure yield a significant AUC gain?
  B. Feature ablation — drop the whole credit-history bundle and measure the F1
     drop, to confirm its predictive importance.
  C. Resampling strategy — class weights vs SMOTE-like oversampling vs none,
     on Recall/F1 under imbalance.

Plus a permutation-importance ranking of the champion model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from ..config import TrainConfig
from ..data.etl import Dataset, resample_smote_like
from ..evaluation.metrics import best_threshold_for_f1, compute_metrics
from ..schemas import ALL_FEATURES, CREDIT_HISTORY_FEATURES
from ..models.pipeline import build_model


def _fit_eval(model_name: str, ds: Dataset, features: list[str],
              cfg: TrainConfig, resample: str = "class_weight") -> dict:
    """Fit a model on a feature subset and evaluate on test (threshold from val)."""
    Xtr, Xval, Xte = ds.X_train[features], ds.X_val[features], ds.X_test[features]
    ytr = ds.y_train

    if resample == "smote_like":
        Xtr, ytr = resample_smote_like(Xtr, ytr, seed=cfg.random_state)
        cw, spw = None, 1.0
    elif resample == "none":
        cw, spw = None, 1.0
    else:  # class_weight / cost-sensitive
        cw = "balanced"
        neg, pos = int((ytr == 0).sum()), max(1, int((ytr == 1).sum()))
        spw = neg / pos

    # Build a model whose preprocessor only knows the given features.
    model = _model_for_features(model_name, features, class_weight=cw,
                                scale_pos_weight=spw,
                                random_state=cfg.random_state)
    model.fit(Xtr, ytr)
    val_prob = model.predict_proba(Xval)[:, 1]
    thr, _ = best_threshold_for_f1(ds.y_val, val_prob)
    test_prob = model.predict_proba(Xte)[:, 1]
    return compute_metrics(ds.y_test, test_prob, thr)


def _model_for_features(model_name, features, class_weight, random_state,
                        scale_pos_weight=1.0):
    """Rebuild a pipeline whose ColumnTransformer targets only ``features``."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from ..schemas import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    num = [c for c in features if c in NUMERIC_FEATURES]
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    scale = model_name == "logistic_regression"
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    pre = ColumnTransformer(transformers)

    base = build_model(model_name, class_weight=class_weight,
                       scale_pos_weight=scale_pos_weight,
                       random_state=random_state)
    est = base.named_steps["clf"]
    if model_name != "xgboost":
        est.set_params(class_weight=class_weight)
    return Pipeline([("pre", pre), ("clf", est)])


# --------------------------------------------------------------------------- #
def ablation_model_family(ds: Dataset, cfg: TrainConfig) -> dict:
    """XGBoost vs Logistic Regression on the full feature set."""
    xgb = _fit_eval("xgboost", ds, ALL_FEATURES, cfg)
    logreg = _fit_eval("logistic_regression", ds, ALL_FEATURES, cfg)
    return {
        "xgboost": xgb,
        "logistic_regression": logreg,
        "auc_gain_xgb_over_logreg": round(xgb["roc_auc"] - logreg["roc_auc"], 4),
        "f1_gain_xgb_over_logreg": round(xgb["f1"] - logreg["f1"], 4),
    }


def ablation_credit_history(ds: Dataset, cfg: TrainConfig,
                            model_name: str = "xgboost") -> dict:
    """Drop the credit-history bundle and measure the F1/AUC degradation."""
    full = _fit_eval(model_name, ds, ALL_FEATURES, cfg)
    kept = [f for f in ALL_FEATURES if f not in CREDIT_HISTORY_FEATURES]
    reduced = _fit_eval(model_name, ds, kept, cfg)
    return {
        "model": model_name,
        "removed_features": CREDIT_HISTORY_FEATURES,
        "with_credit_history": full,
        "without_credit_history": reduced,
        "f1_drop": round(full["f1"] - reduced["f1"], 4),
        "f1_drop_pct": round(100 * (full["f1"] - reduced["f1"]) / full["f1"], 1),
        "recall_drop": round(full["recall"] - reduced["recall"], 4),
        "auc_drop": round(full["roc_auc"] - reduced["roc_auc"], 4),
    }


def ablation_resampling(ds: Dataset, cfg: TrainConfig,
                        model_name: str = "xgboost") -> dict:
    out = {}
    for strat in ["none", "class_weight", "smote_like"]:
        out[strat] = _fit_eval(model_name, ds, ALL_FEATURES, cfg, resample=strat)
    return out


def permutation_feature_importance(estimator, ds: Dataset,
                                   n_repeats: int = 10,
                                   random_state: int = 42) -> pd.DataFrame:
    """Permutation importance of the champion pipeline on the test set (F1)."""
    from sklearn.metrics import f1_score, make_scorer

    result = permutation_importance(
        estimator, ds.X_test, ds.y_test, n_repeats=n_repeats,
        random_state=random_state, scoring=make_scorer(f1_score),
    )
    return (pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))
