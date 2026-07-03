"""Exploratory Data Analysis: identify solvency indicators.

Produces the figures and correlation tables that motivate the modelling:
correlation heatmap, target-vs-feature relationships, class balance, and a
ranked list of the numeric features most associated with default (point-biserial
correlation). Figures are saved under reports/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ..schemas import NUMERIC_FEATURES, TARGET  # noqa: E402

sns.set_theme(style="whitegrid")


def solvency_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric features by |correlation| with default (point-biserial)."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    corrs = {c: df[c].corr(df[TARGET]) for c in num}
    out = (pd.DataFrame({"feature": list(corrs), "corr_with_default": list(corrs.values())})
           .assign(abs_corr=lambda d: d["corr_with_default"].abs())
           .sort_values("abs_corr", ascending=False)
           .reset_index(drop=True))
    return out


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["good (0)", "default (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df):.0%})", ha="center", va="bottom")
    ax.set_title("Class balance (target)")
    ax.set_ylabel("count")
    path = out_dir / "class_balance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    num = [c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET]
    corr = df[num].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Numeric feature correlation matrix")
    path = out_dir / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_solvency_indicators(df: pd.DataFrame, out_dir: Path,
                             top_k: int = 6) -> Path:
    ranking = solvency_ranking(df).head(top_k)
    feats = ranking["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.ravel(), feats):
        for label, color in [(0, "#2a9d8f"), (1, "#e76f51")]:
            sns.kdeplot(df.loc[df[TARGET] == label, feat].dropna(),
                        ax=ax, fill=True, alpha=0.4, color=color,
                        label=("good" if label == 0 else "default"))
        r = ranking.loc[ranking.feature == feat, "corr_with_default"].iloc[0]
        ax.set_title(f"{feat}  (r={r:+.2f})", fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("Top solvency indicators: distribution by default status", y=1.02)
    path = out_dir / "solvency_indicators.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking = solvency_ranking(df)
    figures = {
        "class_balance": str(plot_class_balance(df, out_dir)),
        "correlation_heatmap": str(plot_correlation_heatmap(df, out_dir)),
        "solvency_indicators": str(plot_solvency_indicators(df, out_dir)),
    }
    return {
        "n_rows": len(df),
        "default_rate": round(float(df[TARGET].mean()), 4),
        "missing_by_column": {c: int(df[c].isna().sum())
                              for c in df.columns if df[c].isna().any()},
        "solvency_ranking": ranking.to_dict(orient="records"),
        "figures": figures,
    }
