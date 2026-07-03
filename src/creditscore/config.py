"""Central configuration. Paths and hyperparameters live in one place so
scripts, the app, and tests all agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    db: Path = ROOT / "data" / "credit.db"
    raw_csv: Path = ROOT / "data" / "applicants_raw.csv"
    models: Path = ROOT / "models"
    model_file: Path = ROOT / "models" / "model.joblib"
    metadata_file: Path = ROOT / "models" / "metadata.json"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"

    def ensure(self) -> "Paths":
        for p in (self.data, self.models, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class DataConfig:
    n_samples: int = 12000
    default_rate: float = 0.18        # target class imbalance (minority = default)
    missing_rate: float = 0.06        # fraction of missing cells in some columns
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2             # of the train remainder


@dataclass
class TrainConfig:
    resample: str = "class_weight"    # "class_weight" | "smote_like" | "none"
    cv_folds: int = 4
    scoring: str = "f1"               # tuning objective (imbalance-aware)
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    cfg = Config()
    cfg.paths.ensure()
    return cfg
