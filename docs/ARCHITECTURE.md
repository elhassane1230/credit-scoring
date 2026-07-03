# Architecture

## Pipeline

```
┌─────────────┐   generate    ┌──────────────┐   SQL query   ┌──────────────┐
│ synthetic   │ ────────────▶ │  SQLite DB   │ ────────────▶ │  raw extract │
│ DGP         │               │ applicants   │   collect()   │  (DataFrame) │
└─────────────┘               └──────────────┘               └──────┬───────┘
                                                                     ▼
                                              ┌──────────────────────────────┐
                                              │ ETL                          │
                                              │  clean · clip · normalise    │
                                              │  stratified train/val/test   │
                                              └──────────────┬───────────────┘
                                                             ▼
                          ┌───────────────────────────────────────────────────┐
                          │ sklearn Pipeline (per model)                      │
                          │  ColumnTransformer:                               │
                          │    numeric  → median impute (+ scale for linear)  │
                          │    category → mode impute → one-hot               │
                          │  estimator → LogReg | RandomForest | XGBoost      │
                          └──────────────┬────────────────────────────────────┘
                                         ▼
            RandomizedSearchCV (StratifiedKFold, F1) → threshold on val → test
                                         ▼
                            champion (joblib) + metadata (json)
                                         ▼
                              Flask API + HTML/CSS/JS UI
```

## Why these choices

### Synthetic data with a documented DGP
A known ground-truth risk mechanism makes the ablations *causal* experiments.
The DGP deliberately mixes:
- **weak linear main effects** (income, inquiries) — both model families capture;
- **step/threshold effects** (utilisation > 0.75, DTI > 0.45) — a linear model
  can only approximate with a single slope, so it underfits;
- **pure interactions** (young × over-borrowed; renter × high-DTI; new-job ×
  large-loan) — near-zero marginal correlation, invisible to plain linear models.

This is why tree ensembles win on AUC and why removing the credit-history bundle
hurts F1 — both are properties of the generator, then *measured*, not assumed.

### Leak-free preprocessing
All fitting-based transforms (imputation, scaling, encoding) sit inside the
`Pipeline`. Cross-validation and the train/val/test split therefore never leak
test statistics into training, and the saved artifact reproduces the exact
transform chain at inference.

### Imbalance handling
Default is **cost-sensitive learning** (class weights for LR/RF,
`scale_pos_weight` for XGBoost): no synthetic rows, no leakage, and it directly
targets the minority (default) class. A dependency-free **SMOTE-like**
oversampler is provided for the resampling ablation.

### Threshold selection
Chosen on the validation set to maximise F1, because under ~18% prevalence the
naïve 0.5 cut-off is suboptimal and the business cost of a false negative
(approved-but-defaults) is high. The threshold ships inside the model bundle.

### Metrics
Recall, F1, ROC-AUC, PR-AUC and the KS statistic — the standard credit-risk
panel. PR-AUC and KS are emphasised because they are more informative than raw
accuracy under heavy class imbalance.

## Deployment
The `Dockerfile` installs the runtime, trains a model at build time, and serves
the Flask app via gunicorn with a health check. `docker-compose.yml` mounts
`models/` and `data/` so a pre-trained model can be persisted or swapped without
rebuilding.
