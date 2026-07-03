"""ETL: turn the raw SQL extract into model-ready train/val/test matrices.

Steps:
  1. Clean   — drop duplicates, clip impossible values, coerce types.
  2. Split   — stratified train/val/test *before* any fitting, to prevent
               data leakage (imputation/scaling statistics are learned on
               train only, inside the sklearn Pipeline).
  3. Resample — handle class imbalance. Default strategy is cost-sensitive
               learning (class weights), which is leak-free and needs no
               synthetic rows; an optional SMOTE-like oversampler is provided
               for the ablation on resampling strategy.

Note: imputation, scaling and one-hot encoding are deliberately **not** applied
here. They live in the sklearn ``Pipeline`` (see models/train.py) so they are
fit on training folds only and travel with the saved model — the correct,
leak-free design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DataConfig
from ..schemas import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
)


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def summary(self) -> dict:
        return {
            "n_train": len(self.X_train),
            "n_val": len(self.X_val),
            "n_test": len(self.X_test),
            "default_rate_train": round(float(self.y_train.mean()), 4),
            "default_rate_test": round(float(self.y_test.mean()), 4),
            "n_features": self.X_train.shape[1],
        }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning that does not depend on any learned statistic."""
    df = df.drop_duplicates().copy()

    # Clip numeric features to their declared plausible ranges (cap outliers).
    for col, spec in NUMERIC_FEATURES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=spec["min"], upper=spec["max"])

    # Normalise categoricals to their allowed vocabulary; unknown -> "other"/mode.
    for col, allowed in CATEGORICAL_FEATURES.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(allowed), other=allowed[-1])

    return df


def split(df: pd.DataFrame, cfg: DataConfig | None = None) -> Dataset:
    """Stratified train/val/test split (leak-safe: done before fitting)."""
    cfg = cfg or DataConfig()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=cfg.val_size,
        stratify=y_tr_full, random_state=cfg.seed,
    )
    return Dataset(X_train, X_val, X_test, y_train, y_val, y_test)


def resample_smote_like(X: pd.DataFrame, y: pd.Series,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """A light, dependency-free SMOTE-style oversampler for the minority class.

    Interpolates new minority numeric rows between a sample and one of its
    random minority neighbours; categoricals are copied from the base sample.
    Used only for the resampling-strategy ablation (class weights are the
    default and usually preferred).
    """
    rng = np.random.default_rng(seed)
    minority = y.value_counts().idxmin()
    X_min = X[y == minority]
    n_needed = (y == y.value_counts().idxmax()).sum() - len(X_min)
    if n_needed <= 0 or len(X_min) < 2:
        return X, y

    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    base_idx = rng.integers(0, len(X_min), n_needed)
    partner_idx = rng.integers(0, len(X_min), n_needed)
    gap = rng.uniform(0, 1, (n_needed, len(num_cols)))

    base = X_min.iloc[base_idx].reset_index(drop=True)
    partner = X_min.iloc[partner_idx].reset_index(drop=True)
    synth = base.copy()
    synth[num_cols] = base[num_cols].values + gap * (
        partner[num_cols].values - base[num_cols].values
    )
    synth[cat_cols] = base[cat_cols].values

    X_res = pd.concat([X, synth], ignore_index=True)
    y_res = pd.concat([y, pd.Series([minority] * n_needed)], ignore_index=True)
    return X_res, y_res
