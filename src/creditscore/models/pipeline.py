"""Preprocessing + model pipeline factory.

Each estimator is wrapped in a single sklearn ``Pipeline`` that owns its
preprocessing (median imputation + scaling for numerics, most-frequent
imputation + one-hot for categoricals). Because preprocessing lives *inside*
the pipeline, it is fit on training folds only (no leakage) and is saved and
shipped together with the model — inference applies the exact same transforms.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schemas import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_cols = list(NUMERIC_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # trees don't need scaling; linear models do
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_model(name: str, class_weight: str | dict | None = "balanced",
                scale_pos_weight: float | None = None,
                random_state: int = 42) -> Pipeline:
    """Return an untrained Pipeline(preprocessor + estimator) for ``name``."""
    if name == "logistic_regression":
        est = LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=random_state,
        )
        pre = build_preprocessor(scale=True)

    elif name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, n_jobs=-1,
            random_state=random_state,
        )
        pre = build_preprocessor(scale=False)

    elif name == "xgboost":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight or 1.0,
            random_state=random_state, n_jobs=-1, tree_method="hist",
        )
        pre = build_preprocessor(scale=False)

    else:
        raise ValueError(f"Unknown model '{name}'. Options: {MODEL_NAMES}")

    return Pipeline([("pre", pre), ("clf", est)])


# Hyperparameter search spaces (keys are pipeline-qualified).
SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    "random_forest": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
}
