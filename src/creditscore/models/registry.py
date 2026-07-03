"""Persist and load the champion model together with its metadata.

The saved artifact bundles the fitted Pipeline (preprocessing + estimator) and
the chosen decision threshold, so inference is a single, self-contained object.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ..schemas import feature_order


def save_model(estimator, threshold: float, metadata: dict,
               model_file: str | Path, metadata_file: str | Path) -> None:
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": estimator, "threshold": float(threshold),
                 "features": feature_order()}, model_file)

    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "features": feature_order(),
        **metadata,
    }
    Path(metadata_file).write_text(json.dumps(meta, indent=2))


def load_model(model_file: str | Path) -> dict:
    """Return {'pipeline', 'threshold', 'features'}."""
    return joblib.load(model_file)


def load_metadata(metadata_file: str | Path) -> dict:
    p = Path(metadata_file)
    return json.loads(p.read_text()) if p.exists() else {}
