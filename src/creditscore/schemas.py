"""Single source of truth for the feature schema.

Defining the columns, their types, and the target once here keeps the ETL,
the models, the Flask form, and the tests perfectly in sync. If a feature is
added, it is added here and everything downstream follows.
"""
from __future__ import annotations

TARGET = "default"  # 1 = will default (ineligible), 0 = good (eligible)

# Numeric features and a human-readable description + plausible UI range.
NUMERIC_FEATURES: dict[str, dict] = {
    "age":                    {"desc": "Applicant age (years)", "min": 18, "max": 80},
    "monthly_income":         {"desc": "Net monthly income (EUR)", "min": 400, "max": 20000},
    "employment_length":      {"desc": "Years at current employer", "min": 0, "max": 45},
    "loan_amount":            {"desc": "Requested loan amount (EUR)", "min": 500, "max": 100000},
    "loan_term_months":       {"desc": "Loan term (months)", "min": 6, "max": 120},
    "debt_to_income":         {"desc": "Debt-to-income ratio (0-1)", "min": 0.0, "max": 1.5},
    "credit_history_length":  {"desc": "Length of credit history (years)", "min": 0, "max": 50},
    "num_past_defaults":      {"desc": "Number of past defaults", "min": 0, "max": 8},
    "credit_utilization":     {"desc": "Revolving credit utilisation (0-1)", "min": 0.0, "max": 1.5},
    "num_open_accounts":      {"desc": "Number of open credit accounts", "min": 0, "max": 20},
    "num_recent_inquiries":   {"desc": "Credit inquiries in last 6 months", "min": 0, "max": 12},
}

# Categorical features and their allowed values.
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "home_ownership": ["rent", "own", "mortgage"],
    "purpose": ["debt_consolidation", "car", "home_improvement",
                "business", "education", "medical", "other"],
}

# The "credit history" bundle — used by the ablation that removes it wholesale
# to measure its predictive contribution.
CREDIT_HISTORY_FEATURES = [
    "credit_history_length", "num_past_defaults", "credit_utilization",
    "num_recent_inquiries",
]

ALL_FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


def feature_order() -> list[str]:
    """Deterministic column order used everywhere (training & inference)."""
    return ALL_FEATURES
