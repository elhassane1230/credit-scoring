# Proposed improvements & roadmap

Concrete extensions beyond the delivered scope, ordered by impact-to-effort.

## 1. Explainability (SHAP) — highest impact for a credit product
Regulators and analysts need *reasons*, not just scores. Add SHAP values so
every decision comes with the top contributing factors ("declined mainly due to
95% credit utilisation and 3 past defaults"). This is often a compliance
requirement (adverse-action notices) and slots in behind the existing
`predict_profile` return value.

## 2. Fairness & bias auditing
On real data with protected attributes (age band, gender, region), measure
disparate impact, equal-opportunity difference, and calibration by group; add
reject-option or reweighing mitigations if gaps appear. The synthetic generator
can be extended with a protected attribute to demo the tooling end-to-end.

## 3. Probability calibration
Tree models are not naturally well-calibrated. Wrap the champion in
`CalibratedClassifierCV` (isotonic/Platt) so `probability_default` can be read
as a true probability — important when the score feeds pricing or expected-loss
calculations, not just a binary decision.

## 4. Business-cost-optimal threshold
Replace the F1-optimal threshold with one that minimises expected monetary loss,
using a cost matrix (cost of a false negative ≫ false positive). Expose the
cost ratio as a config so risk teams can move the operating point deliberately.

## 5. Real data + validation
Swap the synthetic source for a real portfolio (e.g. a warehouse table) — only
`data/database.py::COLLECT_QUERY` changes. Add temporal (out-of-time) validation
since credit performance drifts, and back-test on vintages.

## 6. Monitoring & drift detection
Log prediction distributions and input feature stats in production; alert on
population-stability-index (PSI) drift and on score/accuracy degradation.
Schedule periodic retraining with champion/challenger comparison.

## 7. Richer models & features
- Gradient-boosting alternatives (LightGBM, CatBoost — CatBoost handles
  categoricals natively) as additional challengers.
- Feature engineering: ratios and trends (utilisation trajectory, income
  stability), and bureau-style aggregations if available.
- Optional monotonic constraints in XGBoost (e.g. risk must not decrease with
  more past defaults) for regulator-friendly, sensible behaviour.

## 8. Experiment tracking & reproducibility
Log runs, params, metrics and artifacts to MLflow so the leaderboard and
ablations are versioned and comparable across the team; register the champion in
a model registry with stage transitions (staging → production).

## 9. Hardening the service
- Input validation with pydantic request models and typed error responses.
- Batch scoring endpoint for portfolio re-evaluation.
- AuthN/AuthZ, rate limiting, request logging/audit trail for every decision.
- Containerised model versioning with canary rollout.

## 10. Evaluation depth
- Confidence intervals on metrics via bootstrap.
- Gains/lift charts and a full reliability (calibration) curve in the report.
- Cost-curve analysis across thresholds for the risk committee.
