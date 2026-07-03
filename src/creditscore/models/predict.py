"""Real-time inference for a single applicant profile.

Loads the saved champion pipeline and returns a probability of default, an
eligibility decision at the model's chosen threshold, and a simple risk band.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import get_config
from ..schemas import ALL_FEATURES
from .registry import load_model


@lru_cache(maxsize=1)
def _get_bundle(model_file: str):
    return load_model(model_file)


def risk_band(prob: float) -> str:
    if prob < 0.10:
        return "very_low"
    if prob < 0.25:
        return "low"
    if prob < 0.50:
        return "medium"
    if prob < 0.75:
        return "high"
    return "very_high"


def predict_profile(profile: dict, model_file: str | Path | None = None) -> dict:
    """Score one applicant.

    ``profile`` maps feature names to values. Missing features are allowed —
    the pipeline's imputers fill them — but unknown extra keys are ignored.
    """
    model_file = str(model_file or get_config().paths.model_file)
    bundle = _get_bundle(model_file)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    row = {f: profile.get(f, None) for f in ALL_FEATURES}
    X = pd.DataFrame([row], columns=ALL_FEATURES)

    prob = float(pipeline.predict_proba(X)[:, 1][0])
    eligible = prob < threshold  # low default prob -> eligible
    return {
        "probability_default": round(prob, 4),
        "eligible": bool(eligible),
        "decision": "APPROVE" if eligible else "DECLINE",
        "risk_band": risk_band(prob),
        "threshold": round(float(threshold), 3),
    }
