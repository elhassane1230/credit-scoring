"""Synthetic credit-applicant generator with a *documented* data-generating
process (DGP).

Why synthetic? A realistic, openly-shareable credit dataset with a known
ground-truth risk mechanism lets the ablation studies be honest experiments
rather than assertions:

  * ``num_past_defaults``, ``credit_utilization`` and ``credit_history_length``
    are wired to be *strong* drivers of default — so removing the credit-history
    bundle must degrade F1 (the report's key ablation).
  * The DGP contains genuine *non-linear* effects and interactions (utilisation
    kink above 0.8; young-age × high loan-to-income interaction; DTI × income).
    A linear Logistic Regression cannot represent these without manual feature
    engineering, so tree models (RF/XGBoost) should win on AUC — the report's
    other ablation.
  * Class imbalance (~18% default), missing values, and a mix of numeric and
    categorical fields make the ETL, resampling and imputation steps meaningful.

The latent risk is a transparent function; default ~ Bernoulli(sigmoid(risk)).
Everything is seeded and reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..schemas import CATEGORICAL_FEATURES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: DataConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    # ---- draw raw applicant attributes ---------------------------------- #
    age = rng.integers(18, 76, n)
    # income depends mildly on age & employment (log-normal, floored).
    employment_length = np.clip(rng.gamma(2.0, 3.0, n), 0, 45).round(1)
    base_income = rng.lognormal(mean=7.8, sigma=0.45, size=n)  # ~2400 median
    monthly_income = np.clip(
        base_income * (1 + 0.02 * (age - 30) + 0.03 * employment_length), 400, 20000
    ).round(0)

    loan_amount = np.clip(rng.lognormal(mean=9.2, sigma=0.7, size=n), 500, 100000).round(0)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72, 84, 120], n,
                                  p=[.08, .16, .24, .18, .16, .1, .05, .03])

    credit_history_length = np.clip((age - 18) * rng.uniform(0.3, 0.95, n), 0, 50).round(1)
    # past defaults: most have 0, heavy right tail.
    num_past_defaults = rng.poisson(0.35, n)
    num_past_defaults = np.clip(num_past_defaults, 0, 8)

    credit_utilization = np.clip(rng.beta(2.0, 4.0, n) * 1.3, 0, 1.5).round(3)
    num_open_accounts = np.clip(rng.poisson(4.0, n), 0, 20)
    num_recent_inquiries = np.clip(rng.poisson(1.1, n), 0, 12)

    debt_to_income = np.clip(
        (loan_amount / loan_term_months) / (monthly_income + 1)
        + rng.normal(0.15, 0.1, n), 0, 1.5
    ).round(3)

    home_ownership = rng.choice(CATEGORICAL_FEATURES["home_ownership"], n,
                                p=[0.45, 0.25, 0.30])
    purpose = rng.choice(CATEGORICAL_FEATURES["purpose"], n,
                         p=[0.30, 0.18, 0.14, 0.10, 0.10, 0.08, 0.10])

    loan_to_income = loan_amount / (monthly_income * 12 + 1)

    # ---- latent risk (the ground-truth mechanism) ----------------------- #
    # DESIGN: the dominant drivers act through STEP FUNCTIONS and PURE
    # INTERACTIONS with only weak linear main effects. A Logistic Regression
    # (linear in the raw features) cannot represent step/interaction structure
    # without manual feature engineering, so tree ensembles (RF/XGBoost) fit
    # the true surface better and win on AUC — an honest, by-construction result.
    risk = np.full(n, -1.4)

    # Weak linear main effects (a small baseline both model families capture).
    risk += 0.12 * num_recent_inquiries
    risk += -0.010 * employment_length
    risk += -0.00002 * (monthly_income - 2500)

    # Threshold (step) effects — the linear model must approximate these with a
    # single slope and necessarily underfits them.
    risk += 2.4 * (credit_utilization > 0.75)
    risk += 1.9 * (num_past_defaults >= 2)
    risk += 1.2 * (debt_to_income > 0.45)

    # Pure interaction effects (near-zero marginal linear correlation).
    risk += 1.8 * ((debt_to_income > 0.45) & (credit_utilization > 0.6))
    risk += 1.6 * ((age < 28) & (loan_to_income > 0.35))
    risk += 1.4 * ((employment_length < 2) & (loan_amount > 25000))
    risk += 1.2 * ((home_ownership == "rent") & (debt_to_income > 0.5))
    risk += 1.1 * ((num_past_defaults >= 1) & (credit_utilization > 0.7))
    # Protective interaction: a long clean history strongly lowers risk.
    risk += -1.3 * ((credit_history_length > 15) & (num_past_defaults == 0))

    # Categorical main effects.
    risk += np.where(home_ownership == "own", -0.35, 0.0)
    risk += np.where(purpose == "business", 0.30, 0.0)

    # Irreducible noise (modest, so the non-linear signal stays exploitable).
    risk += rng.normal(0, 0.25, n)

    # ---- calibrate intercept to the target default rate (bisection) ------ #
    def _mean_rate(b: float) -> float:
        return float(_sigmoid(risk + b).mean())

    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _mean_rate(mid) < cfg.default_rate:
            lo = mid
        else:
            hi = mid
    prob = _sigmoid(risk + (lo + hi) / 2)
    default = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "monthly_income": monthly_income,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "debt_to_income": debt_to_income,
        "credit_history_length": credit_history_length,
        "num_past_defaults": num_past_defaults,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "num_recent_inquiries": num_recent_inquiries,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "default": default,
    })

    # ---- inject missing values (MAR) so ETL imputation matters ---------- #
    for col in ["monthly_income", "employment_length", "credit_utilization",
                "debt_to_income"]:
        mask = rng.uniform(0, 1, n) < cfg.missing_rate
        df.loc[mask, col] = np.nan

    # add a client id for the SQL layer
    df.insert(0, "client_id", np.arange(1, n + 1))
    return df
