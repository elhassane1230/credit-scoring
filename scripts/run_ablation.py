"""Run all ablation studies and write results + a feature-importance figure.

Run:  python scripts/run_ablation.py
Outputs: reports/ablation_results.json, reports/figures/feature_importance.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from creditscore.config import get_config  # noqa: E402
from creditscore.data import clean, collect, create_database, generate, split  # noqa: E402
from creditscore.evaluation import (  # noqa: E402
    ablation_credit_history, ablation_model_family, ablation_resampling,
    permutation_feature_importance,
)
from creditscore.models import build_model  # noqa: E402


def main():
    cfg = get_config()
    if not cfg.paths.db.exists():
        create_database(generate(cfg.data), cfg.paths.db)
    ds = split(clean(collect(cfg.paths.db)), cfg.data)

    print("[A] Model family: XGBoost vs Logistic Regression…")
    family = ablation_model_family(ds, cfg.train)
    print(f"    XGB AUC={family['xgboost']['roc_auc']:.4f}  "
          f"LogReg AUC={family['logistic_regression']['roc_auc']:.4f}  "
          f"→ AUC gain = {family['auc_gain_xgb_over_logreg']:+.4f}")

    print("[B] Feature ablation: remove credit-history bundle…")
    ch = ablation_credit_history(ds, cfg.train, model_name="xgboost")
    print(f"    F1 {ch['with_credit_history']['f1']:.4f} → "
          f"{ch['without_credit_history']['f1']:.4f}  "
          f"(drop {ch['f1_drop']:+.4f}, {ch['f1_drop_pct']:.1f}%)")

    print("[C] Resampling strategy…")
    resamp = ablation_resampling(ds, cfg.train, model_name="xgboost")
    for k, m in resamp.items():
        print(f"    {k:<14} recall={m['recall']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")

    print("[D] Permutation feature importance (Random Forest)…")
    champ = build_model("random_forest").fit(ds.X_train, ds.y_train)
    imp = permutation_feature_importance(champ, ds, n_repeats=8)
    print(imp.head(6).to_string(index=False))

    # Feature-importance figure.
    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp.iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#4c72b0")
    ax.set_title("Permutation feature importance (Δ F1 when shuffled)")
    ax.set_xlabel("mean importance")
    fig.tight_layout()
    fig_path = cfg.paths.figures / "feature_importance.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    results = {
        "model_family": family,
        "credit_history": ch,
        "resampling": resamp,
        "feature_importance": imp.to_dict(orient="records"),
    }
    (cfg.paths.reports / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {cfg.paths.reports / 'ablation_results.json'}")
    print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
