# GALIATSATOS / PLA-BN

`galiatsatos-plabn` is the installable Python package for the **GALIATSATOS** outcome-definition alignment framework and its first Bayesian reference estimator, **PLA-BN (Partition-Lattice Aligned Bayesian Network)**.

GALIATSATOS is designed for supervised learning when different data sources use different documented definitions of the same underlying outcome. Each definition is encoded as a column-stochastic operator over a finite canonical state. Before model fitting, the stacked operators are checked by a rank/SVD gate. PLA-BN then estimates a canonical posterior using a discrete Naive Bayes or Tree-Augmented Naive Bayes structure with guarded multi-start coarsening-aware Structural EM, and the fitted posterior can be transported to another documented target definition.

## Installation

Install the latest public release from PyPI:

```bash
pip install galiatsatos-plabn
```

For local installation from this source directory:

```bash
pip install .
```

## Quick start

```python
import numpy as np
from galiatsatos_plabn import GaliatsatosMethod, make_ordered_threshold_operator

C1 = make_ordered_threshold_operator(3, threshold=1)
C2 = make_ordered_threshold_operator(3, threshold=2)

X1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]])
y1 = np.array([0, 1, 1, 1, 1, 1])
X2 = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [2, 1]])
y2 = np.array([0, 0, 0, 1, 1, 1])

model = GaliatsatosMethod(
    n_classes=3, structure="tan", smoothing=0.10,
    max_iter=120, tol=1e-5, n_init=3, init_jitter=0.05,
    random_state=0,
).fit([X1, X2], [y1, y2], [C1, C2])

canonical_proba = model.transform(X1[:2])
target_proba = model.transport_proba(X1[:2], C2)
```

## Public API

```python
from galiatsatos_plabn import (
    GaliatsatosMethod,
    PLABNClassifier,
    coarsening_rank,
    make_ordered_threshold_operator,
    sample_observed_labels,
)
```

- `GaliatsatosMethod` — named GALIATSATOS interface exposing canonical posterior embeddings and transport.
- `PLABNClassifier` — PLA-BN reference Bayesian estimator.
- `coarsening_rank` — rank/SVD definition-level identifiability diagnostic.
- `make_ordered_threshold_operator` — helper for deterministic ordered-threshold operators.
- `sample_observed_labels` — sampling helper for deterministic or stochastic operators.

## Frozen manuscript core

The packaged `src/galiatsatos_plabn/plabn.py` is a **byte-identical copy** of the manuscript-locked PLA-BN v1.0.0 core.

SHA-256:

```text
c8e0a286213a87ecaf5a9ab03e473d76989c92ab5856520c60993a98bfb40751
```

The root-level `plabn_frozen_v1.0.0.py` is included as an archival copy of the same frozen source.

## Reproducibility

Repository: https://github.com/dgaliatsatos2010/GALIATSATOS-PLA-BN

Archived v1.0.0 release: https://doi.org/10.5281/zenodo.22079293

The full research repository contains the validation pipelines, simulation code, fold-level outputs, out-of-fold predictions, audit records, tests, and provenance documentation supporting the manuscript.

## Citation

If you use this software, please cite the archived software release and the associated manuscript:

**Galiatsatos, D.** *Learning Across Heterogeneous Outcome Definitions: The GALIATSATOS Framework and Partition-Lattice Aligned Bayesian Networks.* Manuscript submitted to *Communications in Statistics – Simulation and Computation*.

Software archive: https://doi.org/10.5281/zenodo.22079293

## License

MIT License. See `LICENSE`.
