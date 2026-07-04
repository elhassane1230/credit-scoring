# Credit Scoring — Predictive Credit-Eligibility Engine

An end-to-end classical-ML system that automates credit-default scoring, from
**SQL ETL to a deployed Flask app**. It ingests client records from a database,
cleans them, trains and tunes three model families, evaluates them with
imbalance-aware metrics, runs ablation studies, and serves real-time decisions
through an interactive web UI.

```
 SQLite  ──▶  ETL (clean · impute · stratified split · rebalance)
                    │
                    ├──▶  EDA (correlations · risk indicators)
                    │
                    └──▶  Train & tune  ┬ Logistic Regression
                                        ├ Random Forest
                                        └ XGBoost
                                            │  (F1-tuned, threshold on val)
                              select champion · evaluate on test
                                            │
                              Flask app  ──▶  analyst enters a profile,
                                              gets APPROVE/DECLINE + risk band
```

**Dataset:** the real **UCI "Default of Credit Card Clients"** dataset (Taiwan,
30,000 anonymised client records, April–September 2005). Target:
`default.payment.next.month`. The repayment-history fields (`PAY_1…PAY_6`) are
the dominant predictive signal, which the ablation studies confirm empirically.

---

## Results (all computed by the scripts in this repo)

30,000 clients, ~22% default rate, stratified 64/16/20 train/val/test.
Reproduce with `make all`.

### Model leaderboard (held-out test set)

| Model | Recall | F1 | ROC-AUC | PR-AUC | KS |
|-------|:------:|:--:|:-------:|:------:|:--:|
| **XGBoost** (champion) | 0.569 | 0.545 | **0.778** | 0.556 | 0.432 |
| Random Forest | 0.524 | 0.537 | 0.776 | 0.548 | 0.422 |
| Logistic Regression | 0.491 | 0.496 | 0.710 | 0.492 | 0.363 |

![ROC curves](reports/figures/model_comparison.png)

Metrics are chosen for an imbalanced, cost-sensitive problem: **Recall**
(minimise false negatives — approving a client who defaults is the expensive
error), **F1** (precision/recall balance), and **ROC-AUC / KS** (ranking power).
The decision threshold is tuned on validation, not test.

### Ablation studies

**A · XGBoost vs Logistic Regression.** On identical features, XGBoost reaches
ROC-AUC **0.775** vs Logistic Regression **0.711** — a **+0.064 AUC** gain from
capturing the non-linear structure of the repayment-status features that a
linear model on the raw features cannot represent.

**B · Remove the repayment-history bundle.** Dropping `PAY_1 … PAY_6` and
retraining drops F1 from **0.538 → 0.473 (−12.1%)** and ROC-AUC by **0.044** —
confirming payment history is the dominant predictive signal.

**C · Resampling strategy.** On this dataset resampling has little effect:
cost-sensitive class weights give a marginal recall gain (0.540 → 0.545) with
essentially unchanged AUC, and SMOTE-like oversampling is comparable. Threshold
tuning on validation already handles most of the imbalance.

**Permutation feature importance** (champion): `PAY_1` (last-month repayment
status) dominates by a wide margin, followed by `PAY_2`, `PAY_3`, `LIMIT_BAL`
and `PAY_AMT2` — the repayment-history features, corroborating ablation B.

![Feature importance](reports/figures/feature_importance.png)

---

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .

make all        # data -> train -> eda -> ablation  (writes models/ + reports/)
make test       # unit tests
make app        # http://localhost:5000  (interactive scoring UI)
```

Individual stages:

```bash
make data       # load the real dataset into SQLite
make train      # ETL -> tune 3 models -> select & save champion
make eda        # correlation heatmap, risk indicators, class balance
make ablation   # the three ablation studies + feature importance
```

---

## The web app

`make app` serves an analyst-facing form. Enter a profile (or click *Fill
sample*) and get an instant decision. It is backed by a JSON API:

```bash
curl -X POST http://localhost:5000/api/score -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"AGE":24,"SEX":"male","EDUCATION":"high_school",
       "MARRIAGE":"single","PAY_1":3,"PAY_2":3,"PAY_3":2,"PAY_4":2,"PAY_5":2,
       "PAY_6":2,"BILL_AMT1":19000,"PAY_AMT1":0}'
# -> {"decision":"DECLINE","probability_default":0.92,"risk_band":"very_high", ...}
```

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Interactive scoring UI |
| `GET /health` | Liveness + model-present check |
| `GET /api/metadata` | Champion model + test metrics + threshold |
| `POST /api/score` | Score one client profile |

Missing fields are allowed — the pipeline's imputers fill them.

---

## Project layout

```
src/creditscore/
  schemas.py            single source of truth for features & target
  config.py             paths + data/train hyperparameters
  data/
    load.py             load & normalise the real UCI credit-card dataset
    database.py         SQLite create + SQL collection ("collecte SQL")
    etl.py              clean · stratified split · SMOTE-like resampler
  eda/explore.py        correlations, risk ranking, seaborn figures
  models/
    pipeline.py         preprocessing + model factory + search spaces
    train.py            CV tuning, threshold selection, champion pick
    registry.py         save/load champion + metadata
    predict.py          single-profile inference + risk banding
  evaluation/
    metrics.py          recall/precision/F1/ROC-AUC/PR-AUC/KS
    ablation.py         model-family · payment-history · resampling ablations
  app/                  Flask API + HTML/CSS/JS scoring UI
scripts/                run_pipeline · run_eda · run_ablation
tests/                  unit tests
```

---

## Design decisions

- **No data leakage.** Imputation, scaling and one-hot encoding live *inside*
  each sklearn `Pipeline`, so they are fit on training folds only and travel
  with the saved model — inference applies identical transforms.
- **Threshold chosen on validation.** Under ~22% prevalence the default 0.5 is
  rarely optimal; the operating point is selected to maximise validation F1.
- **Imbalance handled cost-sensitively** by default (class weights /
  `scale_pos_weight`), which needs no synthetic rows; a SMOTE-like oversampler
  is available and benchmarked in the ablation.
- **Honest evaluation.** Every headline number is produced by the scripts here;
  see [`docs/RESULTS.md`](docs/RESULTS.md) for methodology and
  [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) for the roadmap
  (SHAP explainability, fairness auditing, calibration, monitoring).

> This is a public research dataset. Absolute metrics reflect this portfolio;
> deploying to a new population would require re-training, calibration, and the
> fairness checks described in the roadmap.

## Tech stack

Python · Pandas · NumPy · scikit-learn · XGBoost · Random Forest ·
Logistic Regression · Matplotlib · Seaborn · Flask · SQL (SQLite) · HTML/CSS/JS.

## License

MIT. Dataset © UCI Machine Learning Repository (public use).
