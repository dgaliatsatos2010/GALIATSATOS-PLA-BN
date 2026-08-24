# GALIATSATOS / PLA-BN

Reproducibility package for the manuscript:

**Learning Across Heterogeneous Outcome Definitions: The GALIATSATOS Framework and Partition-Lattice Aligned Bayesian Networks**

Author: **Dimitrios Galiatsatos**  
Affiliation: Hellenic Open University, Patras, Greece  
ORCID: 0009-0005-5232-1425

## Overview

GALIATSATOS is an outcome-definition alignment framework for supervised learning when data sources use different documented outcome definitions. PLA-BN (Partition-Lattice Aligned Bayesian Network) is the first reference estimator. It represents each observed definition with a column-stochastic operator over a finite canonical latent state, applies a pre-fit rank diagnostic, estimates a canonical posterior through a guarded multi-start Structural EM procedure with a TAN or Naive Bayes structure, and transports the fitted posterior to documented target definitions.

The repository supports the controlled simulation and real-data validation analyses reported in the manuscript.

## Frozen core

The manuscript-facing runtime module is `plabn.py`.

SHA-256:

`c8e0a286213a87ecaf5a9ab03e473d76989c92ab5856520c60993a98bfb40751`

The separately supplied `plabn_frozen_v1.0.0.py` is byte-identical to `plabn.py` and should be added to the repository root as an archival frozen copy.

## Repository contents

- `plabn.py` — runtime PLA-BN implementation used by the validation scripts.
- `galiatsatos_method.py` — public import surface for the named GALIATSATOS method.
- `synthetic.py` — controlled synthetic data generator.
- `publication_experiments.py` — controlled simulation and comparison estimators.
- `nhanes_hypertension_repeated_validation.py` — repeated 5x5 NHANES blood-pressure threshold validation.
- `wine_quality_repeated_validation.py` — repeated 5x5 UCI White Wine Quality validation.
- `nhanes_diabetes_repeated_validation.py` — repeated 5x5 NHANES diabetes validation.
- `multicohort_heart_validation.py` — external UCI Heart Disease population-shift stress test.
- `verify_package.py` — dependency-light functional verification.
- `tests/` — automated unit and leakage-sensitive tests.
- `results/` — manuscript-supporting aggregate, fold-level, OOF, audit, and comparison outputs.
- `docs/` — method specification, frozen protocols, provenance, manuscript lock, and validation records.
- `figures/` — manuscript figures derived from the reported results.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

## Verification

Run the functional verification:

```bash
python verify_package.py
```

Run the automated tests:

```bash
python -m pytest -q
```

At manuscript freeze, the expected core test result was 22 passing pytest tests plus five independent functional checks.

## Reproducing the controlled simulation

The simulation analysis is implemented in `publication_experiments.py`. Final frozen-core 30-replicate outputs used for the manuscript are stored under:

`results/simulation/`

These outputs were regenerated with the same manuscript-locked `plabn.py` hash reported above.

## Reproducing the real-data validations

### NHANES blood-pressure threshold experiment

Place the following NHANES 2015-2016 CSV files in a local data directory:

- `DEMO_I.csv`
- `BMX_I.csv`
- `BPX_I.csv`

Then run:

```bash
python nhanes_hypertension_repeated_validation.py --data-dir <NHANES_DIR> --output-dir <OUT_DIR>
```

### NHANES diabetes experiment

The diabetes scripts use the NHANES 2015-2016 demographic, body-measures, glycohemoglobin, fasting-glucose, and oral-glucose-tolerance components documented in `docs/DATA_PROVENANCE_v0.6.0.md`.

Run:

```bash
python nhanes_diabetes_repeated_validation.py --data-dir <NHANES_DIR> --output-dir <OUT_DIR>
```

### UCI White Wine Quality

Download the public UCI Wine Quality white-wine data as documented in the provenance file, then run:

```bash
python wine_quality_repeated_validation.py --data-dir <WINE_DIR> --output-dir <OUT_DIR>
```

### UCI Heart Disease external stress test

Use the public UCI Heart Disease cohort files as documented in the provenance and validation protocol, then run the corresponding external-validation script.

## Data policy

Raw NHANES and UCI datasets are **not redistributed in this repository**. They should be downloaded from their official public sources. The repository contains code, provenance documentation, analysis outputs, audit records, and out-of-fold predictions required to reproduce and inspect the reported computational analyses.

The NHANES analyses are methodological predictive evaluations within the analytic samples. Survey weights, strata, and primary sampling units are not used to estimate nationally representative performance.

## Validation design

The manuscript uses a frozen, leakage-safe evaluation policy. Preprocessing, operator estimation where required, environment assignment, and model fitting occur within the relevant training partition. The principal real-data validations use five independent repeats of five-fold outer cross-validation. The blood-pressure and Wine Quality two-definition experiments use the locked approximately 2:1 environment allocation described in the protocol and manuscript.

The Heart Disease analysis is retained as a negative external population-shift stress test rather than evidence that definition alignment solves general domain shift.

## Results folders

- `results/simulation/` — final 30-replicate frozen-core simulation summaries and paired comparisons.
- `results/hypertension/` — 5x5 fold metrics, audit records, OOF predictions, paired tests, and transport summaries.
- `results/wine/` — 5x5 fold metrics, duplicate-group audit, OOF predictions, paired tests, and transport summaries.
- `results/diabetes/` — repeated 5x5 metrics, operator estimates, participant-level mean predictions, OOF predictions, audits, and paired tests.
- `results/heart/` — external cohort predictions, confusion matrices, cohort audit, metrics, and manifest.

## Reproducibility and versioning

The estimator and analysis settings were version-locked before interpretation of the confirmatory repeated-validation results. Any post-result modification of the estimator, hyperparameters, endpoint definitions, predictor specifications, operator estimation, comparator, metric, or split seed should be released under a new version and the affected analyses rerun.

See:

- `docs/MANUSCRIPT_LOCK_v1.0.0.md`
- `docs/SECONDARY_VALIDATION_PROTOCOL_v1.1.0.md`
- `docs/METHOD_SPECIFICATION.md`
- `docs/DATA_PROVENANCE_v0.6.0.md`
- `docs/CHANGELOG.md`

## Citation

Citation metadata are supplied in `CITATION.cff`. After creating the GitHub repository and archiving a release with Zenodo, add the GitHub URL and Zenodo DOI to this README and to the manuscript Code Availability statement.

## License

The code in this repository is released under the MIT License. Dataset licenses and terms remain those of the original NHANES and UCI data providers.
