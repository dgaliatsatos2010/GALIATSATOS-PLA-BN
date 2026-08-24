"""Functional verification for the GALIATSATOS/PLA-BN research package.

This script deliberately avoids pytest so that a reviewer can perform the core
checks after installing only the numerical and benchmark dependencies.
"""

from __future__ import annotations

import numpy as np

from plabn import (
    GaliatsatosMethod,
    PLABNClassifier,
    coarsening_rank,
    make_ordered_threshold_operator,
    sample_observed_labels,
)
from synthetic import simulate_ordered_tan


def check_rank_gate() -> None:
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    complete = coarsening_rank(operators)
    incomplete = coarsening_rank(operators[:2])
    assert complete["rank"] == 4 and complete["full_column_rank"]
    assert incomplete["rank"] == 3 and not incomplete["full_column_rank"]


def check_fit_and_probabilities() -> None:
    rng = np.random.default_rng(9)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    features_by_environment = []
    labels_by_environment = []
    for operator in operators:
        features, latent = simulate_ordered_tan(250, rng)
        features_by_environment.append(features)
        labels_by_environment.append(sample_observed_labels(latent, operator, rng))

    model = PLABNClassifier(4, max_iter=20).fit(
        features_by_environment,
        labels_by_environment,
        operators,
    )
    probabilities = model.predict_proba(features_by_environment[0][:30])
    assert probabilities.shape == (30, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(np.diff(model.log_likelihood_history_) >= -1e-7)


def check_fail_closed_behavior() -> None:
    rng = np.random.default_rng(11)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2)]
    features_by_environment = []
    labels_by_environment = []
    for operator in operators:
        features, latent = simulate_ordered_tan(100, rng)
        features_by_environment.append(features)
        labels_by_environment.append(sample_observed_labels(latent, operator, rng))

    try:
        PLABNClassifier(4).fit(
            features_by_environment,
            labels_by_environment,
            operators,
        )
    except ValueError as error:
        assert "rank deficient" in str(error)
    else:
        raise AssertionError("A rank-deficient design did not stop training.")


def check_named_method_interface() -> None:
    rng = np.random.default_rng(21)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    features_by_environment = []
    labels_by_environment = []
    for operator in operators:
        features, latent = simulate_ordered_tan(180, rng)
        features_by_environment.append(features)
        labels_by_environment.append(sample_observed_labels(latent, operator, rng))

    method = GaliatsatosMethod(4, max_iter=20).fit(
        features_by_environment,
        labels_by_environment,
        operators,
    )
    canonical = method.transform(features_by_environment[0][:25])
    target = method.transport_proba(features_by_environment[0][:25], operators[1])
    diagnostics = method.method_diagnostics()
    assert canonical.shape == (25, 4) and np.allclose(canonical.sum(axis=1), 1.0)
    assert target.shape == (25, 2) and np.allclose(target.sum(axis=1), 1.0)
    assert diagnostics["method"] == "GALIATSATOS"



def check_multistart_diagnostics() -> None:
    rng = np.random.default_rng(33)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    xs, ys = [], []
    for operator in operators:
        features, latent = simulate_ordered_tan(120, rng)
        xs.append(features)
        ys.append(sample_observed_labels(latent, operator, rng))
    method = GaliatsatosMethod(4, max_iter=20, n_init=3, random_state=44).fit(xs, ys, operators)
    assert len(method.start_diagnostics_) == 3
    assert method.final_log_likelihood_ >= method.start_diagnostics_[0]["final_log_likelihood"] - 1e-10
    assert method.termination_reason_ in {"tolerance", "max_iter", "likelihood_guard"}


def main() -> None:
    checks = [
        check_rank_gate,
        check_fit_and_probabilities,
        check_fail_closed_behavior,
        check_named_method_interface,
        check_multistart_diagnostics,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} GALIATSATOS/PLA-BN checks passed.")


if __name__ == "__main__":
    main()
