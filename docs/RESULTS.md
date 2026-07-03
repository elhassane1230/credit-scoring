# Results & methodology

Reproduce everything:

```bash
make all      # data -> train -> eda -> ablation
```

Artifacts: `models/model.joblib`, `models/metadata.json`,
`reports/leaderboard.json`, `reports/ablation_results.json`,
`reports/eda_summary.json`, and figures under `reports/figures/`.

## Setup
- 12,000 synthetic applicants, default rate ≈ 18% (calibrated by bisection on
  the intercept of the latent-risk model).
- Stratified split: 64% train / 16% validation / 20% test.
- Hyperparameters tuned with `RandomizedSearchCV` (4-fold `StratifiedKFold`,
  scored by F1). Decision threshold selected on validation.

## Model leaderboard (test)

| Model | Recall | F1 | ROC-AUC | PR-AUC | KS |
|-------|:------:|:--:|:-------:|:------:|:--:|
| Random Forest | 0.664 | 0.577 | 0.836 | 0.624 | 0.537 |
| XGBoost | 0.596 | 0.574 | 0.832 | 0.619 | 0.553 |
| Logistic Regression | 0.608 | 0.511 | 0.796 | 0.524 | 0.457 |

Both tree models beat the linear baseline by ~0.04 ROC-AUC and ~0.06 F1. Random
Forest and XGBoost are within noise of each other; Random Forest is selected as
champion by the highest test ROC-AUC (tie-broken by F1).

## Ablation A — model family (XGBoost vs Logistic Regression)

On the full feature set, evaluated identically:

| Model | ROC-AUC | F1 |
|-------|:-------:|:--:|
| XGBoost | 0.817 | 0.581 |
| Logistic Regression | 0.796 | 0.511 |
| **AUC gain** | **+0.021** | — |

The gain is the value of modelling the DGP's threshold and interaction effects,
which a linear model on the raw features cannot represent.

## Ablation B — remove the credit-history bundle

Removing `credit_history_length`, `num_past_defaults`, `credit_utilization`,
`num_recent_inquiries` and retraining:

| Feature set | F1 | Recall | ROC-AUC |
|-------------|:--:|:------:|:-------:|
| With credit history | 0.581 | — | — |
| Without credit history | 0.389 | — | — |
| **F1 drop** | **−0.192 (−33%)** | | |

A one-third collapse in F1 confirms credit history carries the dominant signal
— consistent with the permutation-importance ranking below.

## Ablation C — resampling strategy

| Strategy | Recall | F1 | ROC-AUC |
|----------|:------:|:--:|:-------:|
| none | 0.643 | 0.576 | 0.823 |
| class weights | 0.668 | 0.581 | 0.817 |
| SMOTE-like | 0.603 | 0.567 | 0.821 |

Cost-sensitive class weights give the best recall — the priority when a missed
default is the costly error — with essentially unchanged AUC. Synthetic
oversampling underperforms here, so class weights are the default.

## Permutation feature importance (champion)

| Feature | Importance (Δ F1 when shuffled) |
|---------|:-------------------------------:|
| credit_utilization | 0.215 |
| credit_history_length | 0.093 |
| debt_to_income | 0.085 |
| age | 0.079 |
| num_past_defaults | 0.076 |
| loan_amount | 0.050 |

Three of the top five are credit-history features, independently corroborating
Ablation B.

## Caveats
- The dataset is synthetic; absolute metric values reflect the generator's
  signal-to-noise ratio, not a specific real portfolio. The *relative* findings
  (trees > linear; credit history dominant; class weights improve recall) are
  the transferable results.
- No fairness/bias audit is included because the synthetic data has no protected
  attributes; see the roadmap for the intended approach on real data.
