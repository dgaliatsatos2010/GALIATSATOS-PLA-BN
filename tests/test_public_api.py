import hashlib
from pathlib import Path
import numpy as np
from galiatsatos_plabn import (
    GaliatsatosMethod, PLABNClassifier, coarsening_rank, make_ordered_threshold_operator,
)
EXPECTED_CORE_SHA256 = "c8e0a286213a87ecaf5a9ab03e473d76989c92ab5856520c60993a98bfb40751"

def test_public_imports_exist():
    assert GaliatsatosMethod is not None
    assert PLABNClassifier is not None

def test_ordered_threshold_rank_gate():
    operators = [make_ordered_threshold_operator(3, 1), make_ordered_threshold_operator(3, 2)]
    result = coarsening_rank(operators)
    assert result["rank"] == 3
    assert result["full_column_rank"]

def test_basic_fit_canonical_posterior_and_transport():
    C1 = make_ordered_threshold_operator(3, 1)
    C2 = make_ordered_threshold_operator(3, 2)
    X1 = np.array([[0], [0], [1], [1], [2], [2]])
    y1 = np.array([0, 0, 1, 1, 1, 1])
    X2 = np.array([[0], [1], [1], [2], [2], [2]])
    y2 = np.array([0, 0, 0, 1, 1, 1])
    model = GaliatsatosMethod(3, structure="naive", smoothing=0.10, max_iter=30, tol=1e-5, n_init=2, random_state=0).fit([X1, X2], [y1, y2], [C1, C2])
    canonical = model.transform(np.array([[0], [1], [2]]))
    transported = model.transport_proba(np.array([[0], [1], [2]]), C2)
    assert canonical.shape == (3, 3)
    assert transported.shape == (3, 2)
    assert np.allclose(canonical.sum(axis=1), 1.0)
    assert np.allclose(transported.sum(axis=1), 1.0)

def test_packaged_core_hash_is_frozen():
    import galiatsatos_plabn.plabn as plabn
    digest = hashlib.sha256(Path(plabn.__file__).read_bytes()).hexdigest()
    assert digest == EXPECTED_CORE_SHA256
