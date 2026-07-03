"""Flask web app: real-time credit-eligibility scoring for analysts.

Serves an interactive form (HTML/CSS/JS) where an analyst enters an applicant
profile and instantly gets the default probability, an APPROVE/DECLINE
decision at the model's operating threshold, and a risk band. A JSON API
(``POST /api/score``) backs the UI and is usable directly by other services.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ..config import get_config
from ..models.predict import predict_profile
from ..models.registry import load_metadata
from ..schemas import CATEGORICAL_FEATURES, NUMERIC_FEATURES

HERE = Path(__file__).parent


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(HERE / "templates"),
                static_folder=str(HERE / "static"))
    cfg = get_config()

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            numeric=NUMERIC_FEATURES,
            categorical=CATEGORICAL_FEATURES,
        )

    @app.route("/health")
    def health():
        ok = cfg.paths.model_file.exists()
        return jsonify({"status": "ok" if ok else "no_model",
                        "model_present": ok})

    @app.route("/api/metadata")
    def metadata():
        meta = load_metadata(cfg.paths.metadata_file)
        return jsonify({
            "champion": meta.get("champion"),
            "threshold": meta.get("threshold"),
            "test_metrics": meta.get("test_metrics"),
        })

    @app.route("/api/score", methods=["POST"])
    def score():
        if not cfg.paths.model_file.exists():
            return jsonify({"error": "No trained model. Run scripts/run_pipeline.py."}), 503
        payload = request.get_json(force=True, silent=True) or {}
        profile = _coerce(payload)
        try:
            result = predict_profile(profile, cfg.paths.model_file)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        result["input"] = profile
        return jsonify(result)

    return app


def _coerce(payload: dict) -> dict:
    """Cast incoming form strings to numbers where appropriate."""
    profile: dict = {}
    for key in NUMERIC_FEATURES:
        if key in payload and payload[key] not in ("", None):
            try:
                profile[key] = float(payload[key])
            except (TypeError, ValueError):
                pass
    for key in CATEGORICAL_FEATURES:
        if key in payload and payload[key]:
            profile[key] = str(payload[key])
    return profile


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
