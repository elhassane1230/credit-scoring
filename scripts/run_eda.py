"""Run EDA on the collected dataset and write figures + a JSON summary."""
from __future__ import annotations

import json

from creditscore.config import get_config
from creditscore.data import collect, generate, create_database
from creditscore.eda import run_eda


def main():
    cfg = get_config()
    if not cfg.paths.db.exists():
        create_database(generate(cfg.data), cfg.paths.db)
    df = collect(cfg.paths.db)
    summary = run_eda(df, cfg.paths.figures)
    (cfg.paths.reports / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== Solvency ranking (|corr| with default) ===")
    for row in summary["solvency_ranking"]:
        print(f"  {row['feature']:<24} r={row['corr_with_default']:+.3f}")
    print("\nFigures:", *summary["figures"].values(), sep="\n  ")


if __name__ == "__main__":
    main()
