# GALIATSATOS Method / PLA-BN — Manuscript Algorithm Lock v1.0.0

**Lock date:** 21 August 2026  
**Status:** manuscript-frozen methodological implementation

## Frozen methodological object

The manuscript evaluates the **GALIATSATOS outcome-definition alignment method** with **PLA-BN (Partition-Lattice Aligned Bayesian Network)** as its first reference estimator.

The following elements are frozen for the primary manuscript analysis and must not be changed after inspection of the repeated-validation results without incrementing the software version and rerunning every affected experiment:

- canonical posterior formulation and transport rule;
- operator-level rank/identifiability gate;
- PLA-BN TAN structure learner;
- guarded multi-start Structural EM;
- probability smoothing and input fail-closed validation;
- NHANES predictor set and fixed BN discretization;
- HbA1c/FPG/OGTT threshold definitions;
- primary canonical research endpoint: maximum laboratory screening severity;
- 20% outer-training calibration subset for stochastic operator estimation;
- three disjoint training environments, one observed laboratory definition per environment;
- `smoothing=0.10`, `max_iter=120`, `tol=1e-5`, `n_init=3`, `init_jitter=0.05`;
- repeated 5x5 outer-validation design with base seed `20260821`;
- operator-aware logistic EM and oracle-label logistic baselines;
- canonical primary metrics: accuracy, balanced accuracy, macro-F1, log-loss, and multiclass Brier score;
- inference based on repeat-level summaries and correlated-CV-aware paired comparisons.

## Anti-post-hoc rule

No estimator, hyperparameter, endpoint, predictor, bin boundary, operator estimator, comparison method, metric, or split seed may be modified in v1.0.0 to improve the observed manuscript result. Any such modification creates a new version and requires a complete rerun with the change disclosed as prospective or exploratory.

## Leakage policy

For every outer evaluation:

1. operator estimation uses only the calibration subset of the outer-training data;
2. preprocessing is fitted only in outer-training data;
3. the proposed method never receives canonical labels during its fit;
4. outer-test rows are not used for operator estimation, preprocessing, tuning, or model fitting;
5. test predictions are generated only after the fold-specific training pipeline is complete.

## Statistical inference policy

The 25 fold estimates from repeated 5-fold CV are correlated and are **not** treated as 25 independent observations.

- Descriptive uncertainty is summarized from the five repeat-level means.
- Exact paired Wilcoxon tests use the five repeat-level method means.
- As a complementary correlated-CV analysis, the Nadeau-Bengio corrected resampled t-test uses the 25 paired fold differences with `n_test/n_train = 1/4`.
- Holm adjustment is applied across the 10 canonical metric/comparator tests.
- Non-significance is not interpreted as equivalence.

## Locked primary repeated-validation result

The exact files under `publication_results_diabetes_repeated_5x5/` constitute the frozen v1.0.0 primary-result set. See `MANUSCRIPT_RESULTS_v1.0.0.md` for the answer-first summary.
