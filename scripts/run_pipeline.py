"""End-to-end training pipeline.

  generate synthetic applicants
    -> load into SQLite  (the "collecte SQL" stage)
    -> collect via SQL query
    -> clean + stratified split (leak-safe)
    -> train & tune Logistic Regression / Random Forest / XGBoost
    -> select champion, evaluate on held-out test
    -> persist champion (+ metadata) and write the leaderboard

Run:  python scripts/run_pipeline.py [--n 12000] [--n-iter 20]
"""
from __future__ import annotations

import argparse
import json
import time

from creditscore.config import get_config
from creditscore.data import clean, collect, create_database, generate, split
from creditscore.models import save_model, train_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="override sample count")
    p.add_argument("--n-iter", type=int, default=20, help="hyperparam search iters")
    args = p.parse_args()

    cfg = get_config()
    if args.n:
        cfg.data.n_samples = args.n

    t0 = time.perf_counter()
    print(f"[1/6] Generating {cfg.data.n_samples} synthetic applicants…")
    df = generate(cfg.data)

    print(f"[2/6] Loading into SQLite → {cfg.paths.db}")
    create_database(df, cfg.paths.db)
    df.to_csv(cfg.paths.raw_csv, index=False)

    print("[3/6] Collecting via SQL + cleaning…")
    raw = collect(cfg.paths.db)
    cleaned = clean(raw)

    print("[4/6] Stratified train/val/test split…")
    ds = split(cleaned, cfg.data)
    print("      ", ds.summary())

    print(f"[5/6] Training & tuning 3 models (n_iter={args.n_iter})…")
    report = train_all(ds, cfg.train, n_iter=args.n_iter)

    print("\n=== Leaderboard (sorted by test ROC-AUC) ===")
    board = report.leaderboard()
    hdr = f"{'model':<22}{'recall':>8}{'F1':>8}{'ROC-AUC':>9}{'PR-AUC':>8}{'KS':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in board:
        print(f"{row['model']:<22}{row['test_recall']:>8.3f}{row['test_f1']:>8.3f}"
              f"{row['test_roc_auc']:>9.3f}{row['test_pr_auc']:>8.3f}{row['test_ks']:>7.3f}")
    print(f"\nChampion: {report.champion}")

    champ = report.results[report.champion]
    print(f"[6/6] Saving champion → {cfg.paths.model_file}")
    save_model(
        champ.estimator, champ.threshold,
        metadata={
            "champion": report.champion,
            "best_params": champ.best_params,
            "test_metrics": champ.test_metrics,
            "leaderboard": board,
            "data_summary": ds.summary(),
        },
        model_file=cfg.paths.model_file,
        metadata_file=cfg.paths.metadata_file,
    )
    (cfg.paths.reports / "leaderboard.json").write_text(json.dumps(board, indent=2))
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
