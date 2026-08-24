# Changelog

## v1.0.0 — 21 August 2026

- Froze the manuscript estimator and primary NHANES protocol after prospectively specified repeated 5×5 validation.
- Added `nhanes_diabetes_repeated_validation.py` and complete 25-fold outputs.
- Added correlated repeated-CV inference (repeat-level exact Wilcoxon plus Nadeau-Bengio corrected resampled t-tests with Holm adjustment).
- Added `MANUSCRIPT_LOCK_v1.0.0.md` and `MANUSCRIPT_RESULTS_v1.0.0.md`.
- No estimator logic was modified after inspection of the repeated-validation results.

## v0.6.0 — 21 August 2026

- Executed the Cleveland/Hungary/Switzerland -> VA Long Beach external-cohort stress test and added external prediction/confusion-matrix outputs.
- Added and executed NHANES 2015-2016 natural-definition diabetes validation using real HbA1c, FPG, and 2-hour OGTT discordance.
- Added calibration-estimated stochastic definition operators inside each outer-training fold.
- Added OOF predictions, class-wise metrics, definition-transport metrics, fold diagnostics, paired tests, and operator audit files.
- Added maximum-severity primary endpoint, median/majority endpoint sensitivity analysis, and exclusion of self-reported known diabetes.
- Added 3 NHANES validation tests; package total is now 19 pytest tests.
- Added `VALIDATION_RESULTS_v0.6.0.md` and `DATA_PROVENANCE_v0.6.0.md`.

## v0.5.0 — 21 August 2026

- Added guarded multi-start Structural EM (`n_init`, `init_jitter`).
- Retains the start with the highest final observed-data likelihood.
- Added `termination_reason_`, `final_log_likelihood_`,
  `relative_last_improvement_`, `structure_changes_`, `best_start_`, and
  per-start diagnostics.
- Increased real-data iteration caps for optimization auditing and set the
  nested-validation multi-start count to two for computational practicality.
- Added `convergence_stress_test.py` and saved optimization audit results.
- Added a natural-cohort / semi-synthetic-definition external-population
  protocol for the four UCI Heart Disease cohorts.
- Generalized the method specification from an obligatorily ordered outcome to
  a finite canonical latent outcome space; ordered thresholds are a special
  case.
- Added a formal partition-lattice interpretation and narrowed the rank-gate
  claim to definition/operator-level identifiability.
- Added three new pytest checks for multi-start reproducibility, diagnostics,
  and invalid optimization controls. Total: 16 pytest tests.

## v0.4.1

- Rejected fractional, negative, and unseen discrete category codes.
- Strengthened operator validation and required positive smoothing.
## v1.1.0 — prospective secondary real-data validation
- Core estimator unchanged from manuscript-lock v1.0.0.
- Added prespecified repeated 5x5 NHANES hypertension validation using documented 130/80 and 140/90 threshold systems.
- Added prespecified repeated 5x5 UCI White Wine Quality validation with exact-duplicate group protection.
- Added full fold-level metrics, corrected paired comparisons, transport results, audit trails, protocol hash, and data checksums.
