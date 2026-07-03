"""creditscore — predictive scoring engine for bank credit eligibility.

An end-to-end classical-ML system: SQL ETL -> EDA -> model training &
tuning (Logistic Regression / Random Forest / XGBoost) -> multi-criteria
evaluation (Recall, F1, AUC-ROC) -> ablation studies -> Flask inference app.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
