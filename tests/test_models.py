import numpy as np

from creditscore.config import DataConfig
from creditscore.data import clean, generate, split
from creditscore.models.pipeline import build_model
from creditscore.models.predict import risk_band


def _small_dataset():
    df = clean(generate(DataConfig(n_samples=1500, seed=7)))
    return split(df, DataConfig(n_samples=1500, seed=7))


def test_models_fit_and_predict_proba():
    ds = _small_dataset()
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = build_model(name)
        model.fit(ds.X_train, ds.y_train)
        prob = model.predict_proba(ds.X_test)[:, 1]
        assert prob.shape[0] == len(ds.X_test)
        assert ((prob >= 0) & (prob <= 1)).all()


def test_model_handles_missing_values_at_inference():
    ds = _small_dataset()
    model = build_model("random_forest").fit(ds.X_train, ds.y_train)
    x = ds.X_test.iloc[[0]].copy()
    x["monthly_income"] = np.nan          # imputer must handle this
    x["credit_utilization"] = np.nan
    prob = model.predict_proba(x)[:, 1]
    assert 0.0 <= prob[0] <= 1.0


def test_trees_beat_linear_on_auc():
    """By construction the DGP has non-linear structure trees should exploit."""
    from sklearn.metrics import roc_auc_score

    ds = _small_dataset()
    aucs = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        m = build_model(name).fit(ds.X_train, ds.y_train)
        aucs[name] = roc_auc_score(ds.y_test, m.predict_proba(ds.X_test)[:, 1])
    assert max(aucs["random_forest"], aucs["xgboost"]) >= aucs["logistic_regression"]


def test_risk_band_monotonic():
    bands = [risk_band(p) for p in [0.05, 0.2, 0.4, 0.6, 0.9]]
    assert bands == ["very_low", "low", "medium", "high", "very_high"]
