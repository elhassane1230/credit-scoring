# Results & methodology

Reproduce everything:

```bash
make all      # data -> train -> eda -> ablation
```

Artifacts: `models/model.joblib`, `models/metadata.json`,
`reports/leaderboard.json`, `reports/ablation_results.json`,
`reports/eda_summary.json`, and figures under `reports/figures/`.

## Setup
- Real UCI "Default of Credit Card Clients" dataset (Taiwan), 30,000 records,
  default rate ≈ 22%.
- Stratified split: 64% train (19,200) / 16% validation (4,800) / 20% test (6,000).
- Hyperparameters tuned with `RandomizedSearchCV` (3-fold `StratifiedKFold`,
  scored by F1). Decision threshold selected on validation (champion: 0.55).

## Model leaderboard (test)

| Model | Recall | F1 | ROC-AUC | PR-AUC | KS |
|-------|:------:|:--:|:-------:|:------:|:--:|
| XGBoost | 0.569 | 0.545 | 0.778 | 0.556 | 0.432 |
| Random Forest | 0.524 | 0.537 | 0.776 | 0.548 | 0.422 |
| Logistic Regression | 0.491 | 0.496 | 0.710 | 0.492 | 0.363 |

XGBoost is selected as champion by the highest test ROC-AUC (tie-broken by F1);
Random Forest is a close second, and both tree models clearly beat the linear
baseline (~0.07 ROC-AUC).

## Ablation A — model family (XGBoost vs Logistic Regression)

On the full feature set, evaluated identically:

| Model | ROC-AUC |
|-------|:-------:|
| XGBoost | 0.775 |
| Logistic Regression | 0.711 |
| **AUC gain** | **+0.064** |

The gain is the value of modelling the non-linear/interaction structure of the
repayment-status features, which a linear model on the raw features cannot
represent.

## Ablation B — remove the repayment-history bundle

Removing `PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` and retraining:

| Feature set | F1 | ROC-AUC |
|-------------|:--:|:-------:|
| With payment history | 0.538 | 0.777 |
| Without payment history | 0.473 | 0.733 |
| **Drop** | **−0.065 (−12.1%)** | **−0.044** |

The F1 drop confirms payment history carries the dominant signal — consistent
with the permutation-importance ranking below.

## Ablation C — resampling strategy

| Strategy | Recall | F1 | ROC-AUC |
|----------|:------:|:--:|:-------:|
| none | 0.540 | 0.538 | 0.777 |
| class weights | 0.545 | 0.538 | 0.775 |
| SMOTE-like | 0.548 | 0.537 | 0.778 |

On this dataset resampling has little effect: the three strategies are within
noise of each other. Threshold tuning on validation already handles most of the
class imbalance, so cost-sensitive class weights (the default) are kept for their
marginal recall benefit without synthetic rows.

## Permutation feature importance (champion)

| Feature | Importance (Δ F1 when shuffled) |
|---------|:-------------------------------:|
| PAY_1 | 0.209 |
| PAY_2 | 0.020 |
| PAY_3 | 0.019 |
| LIMIT_BAL | 0.016 |
| PAY_AMT2 | 0.012 |

`PAY_1` (last-month repayment status) dominates by a wide margin; the top
features are repayment-history fields, independently corroborating Ablation B.

## Caveats
- Metrics reflect this specific portfolio (Taiwan, 2005); deploying to a new
  population would require re-training and re-calibration.
- No fairness/bias audit is included yet, though the dataset carries protected
  attributes (SEX, AGE, MARRIAGE, EDUCATION); see the roadmap for the intended
  approach (per-group performance, calibration, mitigation).
