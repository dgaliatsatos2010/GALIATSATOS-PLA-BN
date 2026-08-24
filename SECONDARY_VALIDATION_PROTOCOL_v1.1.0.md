# Prospective secondary real-data validation protocol — v1.1.0

Protocol frozen before inspection of predictive performance for the two added datasets.
Core estimator `plabn.py` and all PLA-BN hyperparameters are unchanged from manuscript-lock v1.0.0.

## A. NHANES 2015–2016 blood-pressure definition validation
- Population: adults age >=20 with at least two valid paired auscultatory SBP/DBP readings and the same non-BP predictor requirements used in the frozen diabetes benchmark.
- BP phenotype: mean SBP and mean DBP across all available valid BPX readings 1–4 within the examination visit. This is a methodological screening phenotype, not a clinical diagnosis.
- Canonical states: z=0 if mean SBP<130 and mean DBP<80; z=1 if >=130/80 but <140/90; z=2 if mean SBP>=140 or mean DBP>=90.
- Definition A (ACC/AHA-type 130/80 threshold): binary positive iff z>=1.
- Definition B (ESC/conventional 140/90 threshold): binary positive iff z>=2.
- Operators: deterministic and prespecified; vertically stacked rank must equal K=3.
- Predictors: age, sex, race/ethnicity, BMI, waist circumference, income-to-poverty ratio, education. No BP measurement or hypertension questionnaire/medication variable is used as a predictor.
- Evaluation: repeated stratified 5x5 outer CV. Within each outer-training set, development rows are randomly divided into two disjoint environments independent of outcome; one environment reveals only Definition A and the other only Definition B.
- PLA-BN: smoothing=0.10, max_iter=120, tol=1e-5, n_init=3, init_jitter=0.05.
- Comparators: operator-aware logistic EM and oracle-label multinomial logistic regression.
- Metrics: accuracy, balanced accuracy, macro-F1, log-loss, multiclass Brier; transported-definition metrics for both binary definitions.
- Statistical comparisons: Nadeau–Bengio corrected resampled t-tests over 25 paired folds plus exact Wilcoxon tests on five repeat-level means; Holm correction across canonical metric/comparator tests.

## B. UCI White Wine Quality validation
- Dataset: all 4,898 white Vinho Verde records and 11 physicochemical predictors from UCI Wine Quality.
- Canonical states: z=0 for quality <=5, z=1 for quality=6, z=2 for quality>=7.
- Definition A: binary quality>=6.
- Definition B: binary quality>=7.
- Operators: deterministic and prespecified; vertically stacked rank must equal K=3.
- Duplicate protection: exact predictor-vector duplicates are assigned to one outer fold through grouped stratified splitting.
- PLA-BN predictor preprocessing: five training-fold-only quantile bins per continuous predictor; cutpoints fitted only on outer training data.
- Logistic preprocessing: training-fold-only standardization of the original continuous predictors.
- Environment construction, PLA-BN hyperparameters, comparators, metrics, and inference are the same as in the hypertension experiment.
- This experiment is real-data / semi-synthetic-definition validation because the sensory score and physicochemical predictors are real but the two binary reporting definitions are imposed prospectively for the benchmark.

## Reporting rule
All results from both prespecified datasets will be retained and reported irrespective of whether PLA-BN outperforms the comparators. Dataset inclusion will not be reversed after performance is observed.
