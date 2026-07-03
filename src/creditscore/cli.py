"""Console entrypoints (thin wrappers around the scripts)."""
from __future__ import annotations

import runpy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(script: str) -> None:
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")


def train_cmd() -> None:
    _run("run_pipeline.py")


def eda_cmd() -> None:
    _run("run_eda.py")


def ablation_cmd() -> None:
    _run("run_ablation.py")
