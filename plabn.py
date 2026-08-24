"""GALIATSATOS method with a PLA-BN reference estimator.

GALIATSATOS is a rank-gated machine-learning method for aligning heterogeneous
outcome definitions.  It maps records to a canonical posterior representation
and can transport that representation to a new documented definition.  PLA-BN
(Partition-Lattice Aligned Bayesian Network) is the first reference estimator:
it supports discrete predictors and a canonical latent class, and learns either a
Naive Bayes or TAN structure with coarsening-aware Structural EM. Ordered semantics
are supplied by ordered coarsening operators when the application requires them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp


Array = np.ndarray


def _as_integer_array(values: Array, *, ndim: int, name: str) -> Array:
    """Validate integer-coded discrete data without silently truncating floats."""
    raw = np.asarray(values)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D.")
    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric integer codes.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integer-coded values; fractional values are not allowed.")
    return numeric.astype(int)


def _project_simplex(values: Array) -> Array:
    """Euclidean projection onto the probability simplex."""
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    support = np.nonzero(ordered * np.arange(1, len(values) + 1) > cumulative - 1)[0]
    if support.size == 0:
        return np.full_like(values, 1.0 / len(values))
    rho = support[-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - theta, 0.0)


def coarsening_rank(coarsening_matrices: Sequence[Array], tolerance: float | None = None) -> dict:
    """Return the rank-based definition identifiability diagnostic.

    Matrices have shape (number of observed labels, number of latent classes)
    and encode P(observed label | latent class, environment).
    """
    matrices = [np.asarray(matrix, dtype=float) for matrix in coarsening_matrices]
    if not matrices:
        raise ValueError("At least one coarsening matrix is required.")
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("All coarsening matrices must be 2-D.")
    latent_classes = matrices[0].shape[1]
    if latent_classes < 2 or any(matrix.shape[1] != latent_classes for matrix in matrices):
        raise ValueError("All coarsening matrices must have the same number of latent-class columns.")
    for matrix in matrices:
        if matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
            raise ValueError("Coarsening matrices must be non-empty and finite.")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
            raise ValueError("Each coarsening matrix must be column-stochastic.")
    stacked = np.vstack(matrices)
    singular_values = np.linalg.svd(stacked, compute_uv=False)
    if tolerance is None:
        tolerance = max(stacked.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    positive = singular_values[singular_values > tolerance]
    condition_number = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return {
        "rank": rank,
        "latent_classes": latent_classes,
        "full_column_rank": rank == latent_classes,
        "singular_values": singular_values,
        "condition_number_on_identifiable_subspace": condition_number,
    }


@dataclass
class _Parameters:
    class_prior: Array
    feature_cpts: list[Array]


class PLABNClassifier:
    """Coarsening-aware discrete Naive Bayes or TAN classifier.

    Parameters
    ----------
    n_classes:
        Number of canonical latent outcome classes.
    structure:
        ``"tan"`` learns a feature tree conditional on the latent class;
        ``"naive"`` keeps all features conditionally independent.
    smoothing:
        Symmetric pseudo-count used for class priors and CPTs.
    max_iter, tol:
        Structural-EM stopping controls.
    require_identifiable:
        If true, training stops before optimization unless the stacked
        coarsening operators have full column rank.
    n_init:
        Number of Structural-EM starts. Start 0 uses the deterministic
        operator/prior initialization; additional starts perturb only
        compatible latent responsibilities and the best observed-data
        likelihood is retained.
    init_jitter:
        Log-scale standard deviation used for randomized compatible
        responsibility perturbations in starts after start 0.
    """

    def __init__(
        self,
        n_classes: int,
        structure: str = "tan",
        smoothing: float = 0.25,
        max_iter: int = 100,
        tol: float = 1e-6,
        require_identifiable: bool = True,
        random_state: int = 0,
        n_init: int = 3,
        init_jitter: float = 0.35,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if structure not in {"tan", "naive"}:
            raise ValueError("structure must be 'tan' or 'naive'.")
        if smoothing <= 0:
            raise ValueError("smoothing must be strictly positive for stable CPT estimation.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")
        if n_init < 1:
            raise ValueError("n_init must be at least 1.")
        if init_jitter < 0:
            raise ValueError("init_jitter must be non-negative.")
        self.n_classes = int(n_classes)
        self.structure = structure
        self.smoothing = float(smoothing)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.require_identifiable = bool(require_identifiable)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.init_jitter = float(init_jitter)

    @staticmethod
    def _validate_matrix(matrix: Array, n_classes: int) -> Array:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != n_classes:
            raise ValueError("A coarsening matrix has an incompatible shape.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Coarsening probabilities must be finite.")
        if np.any(matrix < 0):
            raise ValueError("Coarsening probabilities cannot be negative.")
        if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
            raise ValueError("Each coarsening-matrix column must sum to one.")
        return matrix

    def _estimate_prior(self, labels_by_environment: Sequence[Array], matrices: Sequence[Array]) -> Array:
        design_blocks = []
        target_blocks = []
        for labels, matrix in zip(labels_by_environment, matrices):
            proportions = np.bincount(labels, minlength=matrix.shape[0]).astype(float)
            proportions /= max(len(labels), 1)
            weight = np.sqrt(max(len(labels), 1))
            design_blocks.append(weight * matrix)
            target_blocks.append(weight * proportions)
        design = np.vstack(design_blocks)
        target = np.concatenate(target_blocks)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        projected = _project_simplex(solution)
        return np.maximum(projected, 1e-10) / np.maximum(projected, 1e-10).sum()

    def _initial_responsibilities(
        self,
        labels_by_environment: Sequence[Array],
        matrices: Sequence[Array],
        prior: Array,
    ) -> list[Array]:
        responsibilities = []
        for labels, matrix in zip(labels_by_environment, matrices):
            compatible = matrix[labels, :]
            weighted = compatible * prior[None, :]
            denominator = weighted.sum(axis=1, keepdims=True)
            if np.any(denominator <= 0):
                raise ValueError("At least one observed label is incompatible with every latent class.")
            responsibilities.append(weighted / denominator)
        return responsibilities

    def _fit_parameters(self, features: Array, responsibilities: Array, parents: Array) -> _Parameters:
        n_samples, n_features = features.shape
        class_counts = responsibilities.sum(axis=0)
        prior = (class_counts + self.smoothing) / (
            n_samples + self.smoothing * self.n_classes
        )
        cpts: list[Array] = []
        for feature_index in range(n_features):
            cardinality = self.cardinalities_[feature_index]
            parent_index = int(parents[feature_index])
            if parent_index < 0:
                counts = np.zeros((self.n_classes, cardinality), dtype=float)
                for latent_class in range(self.n_classes):
                    np.add.at(counts[latent_class], features[:, feature_index], responsibilities[:, latent_class])
                denominator = counts.sum(axis=1, keepdims=True)
                cpt = (counts + self.smoothing) / (
                    denominator + self.smoothing * cardinality
                )
            else:
                parent_cardinality = self.cardinalities_[parent_index]
                counts = np.zeros(
                    (self.n_classes, parent_cardinality, cardinality), dtype=float
                )
                for latent_class in range(self.n_classes):
                    np.add.at(
                        counts[latent_class],
                        (features[:, parent_index], features[:, feature_index]),
                        responsibilities[:, latent_class],
                    )
                denominator = counts.sum(axis=2, keepdims=True)
                cpt = (counts + self.smoothing) / (
                    denominator + self.smoothing * cardinality
                )
            cpts.append(cpt)
        return _Parameters(prior, cpts)

    def _log_joint(self, features: Array, parameters: _Parameters, parents: Array) -> Array:
        n_samples, n_features = features.shape
        log_joint = np.broadcast_to(np.log(parameters.class_prior), (n_samples, self.n_classes)).copy()
        for feature_index in range(n_features):
            cpt = parameters.feature_cpts[feature_index]
            parent_index = int(parents[feature_index])
            if parent_index < 0:
                probabilities = cpt[:, features[:, feature_index]].T
            else:
                probabilities = cpt[
                    :, features[:, parent_index], features[:, feature_index]
                ].T
            log_joint += np.log(np.maximum(probabilities, np.finfo(float).tiny))
        return log_joint

    def _e_step(
        self,
        features_by_environment: Sequence[Array],
        labels_by_environment: Sequence[Array],
        matrices: Sequence[Array],
        parameters: _Parameters,
        parents: Array,
    ) -> tuple[list[Array], float]:
        responsibilities = []
        observed_log_likelihood = 0.0
        for features, labels, matrix in zip(
            features_by_environment, labels_by_environment, matrices
        ):
            log_joint = self._log_joint(features, parameters, parents)
            log_compatibility = np.log(np.maximum(matrix[labels, :], np.finfo(float).tiny))
            impossible = matrix[labels, :] <= 0
            log_compatibility[impossible] = -np.inf
            unnormalized = log_joint + log_compatibility
            normalizer = logsumexp(unnormalized, axis=1)
            if np.any(~np.isfinite(normalizer)):
                raise FloatingPointError("A row has zero observed-data probability.")
            responsibilities.append(np.exp(unnormalized - normalizer[:, None]))
            observed_log_likelihood += float(normalizer.sum())
        return responsibilities, observed_log_likelihood

    def _conditional_mutual_information(self, features: Array, responsibilities: Array) -> Array:
        n_samples, n_features = features.shape
        weights = np.zeros((n_features, n_features), dtype=float)
        for left in range(n_features):
            for right in range(left + 1, n_features):
                value = 0.0
                left_cardinality = self.cardinalities_[left]
                right_cardinality = self.cardinalities_[right]
                for latent_class in range(self.n_classes):
                    class_weight = float(responsibilities[:, latent_class].sum())
                    if class_weight <= 0:
                        continue
                    counts = np.zeros((left_cardinality, right_cardinality), dtype=float)
                    np.add.at(
                        counts,
                        (features[:, left], features[:, right]),
                        responsibilities[:, latent_class],
                    )
                    joint = counts / class_weight
                    left_marginal = joint.sum(axis=1, keepdims=True)
                    right_marginal = joint.sum(axis=0, keepdims=True)
                    denominator = left_marginal @ right_marginal
                    mask = joint > 0
                    conditional_information = float(
                        np.sum(joint[mask] * np.log(joint[mask] / denominator[mask]))
                    )
                    value += (class_weight / n_samples) * conditional_information
                weights[left, right] = weights[right, left] = value
        return weights

    @staticmethod
    def _maximum_spanning_tree(weights: Array, root: int = 0) -> Array:
        n_features = weights.shape[0]
        parents = np.full(n_features, -2, dtype=int)
        parents[root] = -1
        selected = {root}
        while len(selected) < n_features:
            best_edge: tuple[int, int] | None = None
            best_weight = -np.inf
            for source in sorted(selected):
                for target in range(n_features):
                    if target in selected:
                        continue
                    candidate = weights[source, target]
                    if candidate > best_weight + 1e-15:
                        best_weight = candidate
                        best_edge = (source, target)
            if best_edge is None:
                raise RuntimeError("Unable to construct a spanning tree.")
            source, target = best_edge
            parents[target] = source
            selected.add(target)
        return parents

    def _learn_parents(self, features: Array, responsibilities: Array) -> Array:
        if self.structure == "naive" or features.shape[1] == 1:
            return np.full(features.shape[1], -1, dtype=int)
        information = self._conditional_mutual_information(features, responsibilities)
        return self._maximum_spanning_tree(information, root=0)

    def _perturb_responsibilities(
        self,
        base_responsibilities: Array,
        rng: np.random.Generator,
    ) -> Array:
        """Perturb compatible responsibilities without creating impossible states."""
        if self.init_jitter == 0:
            return base_responsibilities.copy()
        noise = rng.normal(0.0, self.init_jitter, size=base_responsibilities.shape)
        perturbed = base_responsibilities * np.exp(noise)
        denominator = perturbed.sum(axis=1, keepdims=True)
        if np.any(denominator <= 0):
            raise FloatingPointError("Randomized initialization produced an empty responsibility row.")
        return perturbed / denominator

    def _run_structural_em_start(
        self,
        features_list: Sequence[Array],
        labels_list: Sequence[Array],
        matrices: Sequence[Array],
        all_features: Array,
        initial_responsibilities: Array,
        start_index: int,
    ) -> dict:
        """Run one guarded Structural-EM start and return complete diagnostics."""
        responsibilities = initial_responsibilities
        parents = self._learn_parents(all_features, responsibilities)
        parameters = self._fit_parameters(all_features, responsibilities, parents)
        _, current_likelihood = self._e_step(
            features_list, labels_list, matrices, parameters, parents
        )
        likelihood_history = [current_likelihood]
        termination_reason = "max_iter"
        converged = False
        last_improvement = float("nan")
        structure_changes = 0

        for _iteration in range(1, self.max_iter + 1):
            responsibility_list, _ = self._e_step(
                features_list, labels_list, matrices, parameters, parents
            )
            responsibilities = np.vstack(responsibility_list)
            candidate_parents = self._learn_parents(all_features, responsibilities)
            if not np.array_equal(candidate_parents, parents):
                structure_changes += 1
            candidate_parameters = self._fit_parameters(
                all_features, responsibilities, candidate_parents
            )
            _, candidate_likelihood = self._e_step(
                features_list, labels_list, matrices, candidate_parameters, candidate_parents
            )

            # Smoothing and a structure move can lower the observed likelihood.
            # Retry the parameter update on the previous tree before rejecting the step.
            if candidate_likelihood < current_likelihood - 1e-8:
                fallback_parameters = self._fit_parameters(
                    all_features, responsibilities, parents
                )
                _, fallback_likelihood = self._e_step(
                    features_list, labels_list, matrices, fallback_parameters, parents
                )
                if fallback_likelihood < current_likelihood - 1e-8:
                    termination_reason = "likelihood_guard"
                    break
                candidate_parents = parents.copy()
                candidate_parameters = fallback_parameters
                candidate_likelihood = fallback_likelihood

            last_improvement = candidate_likelihood - current_likelihood
            parents = candidate_parents
            parameters = candidate_parameters
            current_likelihood = candidate_likelihood
            likelihood_history.append(current_likelihood)
            threshold = self.tol * (1.0 + abs(current_likelihood))
            if last_improvement <= threshold:
                converged = True
                termination_reason = "tolerance"
                break

        relative_improvement = (
            float(last_improvement / (1.0 + abs(current_likelihood)))
            if np.isfinite(last_improvement)
            else float("nan")
        )
        return {
            "start_index": int(start_index),
            "parents": parents.copy(),
            "parameters": parameters,
            "likelihood_history": np.asarray(likelihood_history, dtype=float),
            "final_log_likelihood": float(current_likelihood),
            "iterations": int(len(likelihood_history) - 1),
            "converged": bool(converged),
            "termination_reason": termination_reason,
            "last_improvement": float(last_improvement),
            "relative_last_improvement": relative_improvement,
            "structure_changes": int(structure_changes),
        }

    def fit(
        self,
        features_by_environment: Sequence[Array],
        labels_by_environment: Sequence[Array],
        coarsening_matrices: Sequence[Array],
    ) -> "PLABNClassifier":
        if not (
            len(features_by_environment)
            == len(labels_by_environment)
            == len(coarsening_matrices)
            > 0
        ):
            raise ValueError("Features, labels, and coarsening matrices must align by environment.")

        matrices = [self._validate_matrix(matrix, self.n_classes) for matrix in coarsening_matrices]
        features_list = [
            _as_integer_array(features, ndim=2, name=f"features_by_environment[{index}]")
            for index, features in enumerate(features_by_environment)
        ]
        labels_list = [
            _as_integer_array(labels, ndim=1, name=f"labels_by_environment[{index}]")
            for index, labels in enumerate(labels_by_environment)
        ]
        n_features = features_list[0].shape[1]
        if n_features < 1:
            raise ValueError("At least one predictor is required.")
        for features, labels, matrix in zip(features_list, labels_list, matrices):
            if features.shape[1] != n_features:
                raise ValueError("Every feature array must have the same number of columns.")
            if len(labels) != len(features):
                raise ValueError("Each label vector must align with its feature array.")
            if np.any(features < 0):
                raise ValueError("Discrete feature values must be non-negative integers.")
            if np.any(labels < 0) or np.any(labels >= matrix.shape[0]):
                raise ValueError("An observed label is outside its coarsening matrix.")

        self.identifiability_ = coarsening_rank(matrices)
        if self.require_identifiable and not self.identifiability_["full_column_rank"]:
            raise ValueError(
                "The stacked coarsening operators are rank deficient: the requested "
                "canonical label resolution is not identifiable from definitions alone."
            )

        all_features = np.vstack(features_list)
        self.cardinalities_ = np.max(all_features, axis=0).astype(int) + 1
        prior = self._estimate_prior(labels_list, matrices)
        base_list = self._initial_responsibilities(labels_list, matrices, prior)
        base_responsibilities = np.vstack(base_list)

        starts: list[dict] = []
        for start_index in range(self.n_init):
            if start_index == 0:
                initial = base_responsibilities.copy()
            else:
                rng = np.random.default_rng(
                    self.random_state + 104729 * start_index
                )
                initial = self._perturb_responsibilities(base_responsibilities, rng)
            starts.append(
                self._run_structural_em_start(
                    features_list, labels_list, matrices, all_features, initial, start_index
                )
            )

        # Start 0 is always present, so multi-start cannot perform worse in
        # observed-data likelihood than the deterministic initialization.
        best = max(starts, key=lambda item: item["final_log_likelihood"])
        parameters = best["parameters"]
        self.parents_ = best["parents"].copy()
        self.class_prior_ = parameters.class_prior.copy()
        self.feature_cpts_ = [cpt.copy() for cpt in parameters.feature_cpts]
        self.log_likelihood_history_ = best["likelihood_history"].copy()
        self.n_iter_ = int(best["iterations"])
        self.converged_ = bool(best["converged"])
        self.termination_reason_ = str(best["termination_reason"])
        self.final_log_likelihood_ = float(best["final_log_likelihood"])
        self.relative_last_improvement_ = float(best["relative_last_improvement"])
        self.structure_changes_ = int(best["structure_changes"])
        self.best_start_ = int(best["start_index"])
        self.start_diagnostics_ = [
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in item.items()
                if key not in {"parameters", "parents"}
            }
            for item in starts
        ]
        self.n_features_in_ = n_features
        return self

    def _parameters(self) -> _Parameters:
        if not hasattr(self, "class_prior_"):
            raise RuntimeError("The classifier is not fitted.")
        return _Parameters(self.class_prior_, self.feature_cpts_)

    def predict_proba(self, features: Array) -> Array:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("The classifier is not fitted.")
        features = _as_integer_array(features, ndim=2, name="features")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("features has an incompatible shape.")
        if np.any(features < 0):
            raise ValueError("Discrete feature values must be non-negative integers.")
        for feature_index, cardinality in enumerate(self.cardinalities_):
            if np.any(features[:, feature_index] >= cardinality):
                raise ValueError(
                    f"Feature {feature_index} contains a category not observed during fitting."
                )
        log_joint = self._log_joint(features, self._parameters(), self.parents_)
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def predict(self, features: Array) -> Array:
        return np.argmax(self.predict_proba(features), axis=1)

    def predict_definition_proba(self, features: Array, coarsening_matrix: Array) -> Array:
        matrix = self._validate_matrix(coarsening_matrix, self.n_classes)
        probabilities = self.predict_proba(features) @ matrix.T
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict_definition(self, features: Array, coarsening_matrix: Array) -> Array:
        return np.argmax(self.predict_definition_proba(features, coarsening_matrix), axis=1)

    def feature_edges(self) -> set[frozenset[int]]:
        if not hasattr(self, "parents_"):
            raise RuntimeError("The classifier is not fitted.")
        return {
            frozenset((feature, int(parent)))
            for feature, parent in enumerate(self.parents_)
            if parent >= 0
        }


class GaliatsatosMethod(PLABNClassifier):
    """Named interface for the GALIATSATOS outcome-alignment method.

    The estimator follows six fixed stages: declare a canonical outcome,
    encode every observed definition as a coarsening operator, apply the rank
    gate, initialize canonical responsibilities, learn the PLA-BN reference
    estimator, and expose a canonical posterior embedding that can be
    transported to another known definition.

    ``fit`` is inherited from :class:`PLABNClassifier`.  ``transform`` returns
    one canonical posterior feature per latent class, making the method usable
    as a supervised representation step.  Evaluation code must fit a fresh
    instance inside each training fold; fitting once before cross-validation
    would leak label-derived information into validation folds.
    """

    method_name = "GALIATSATOS"
    reference_estimator = "PLA-BN"

    def transform(self, features: Array) -> Array:
        """Map records to the K-dimensional canonical posterior embedding."""
        return self.predict_proba(features)

    def transport_proba(self, features: Array, target_operator: Array) -> Array:
        """Return probabilities in a documented target outcome definition."""
        return self.predict_definition_proba(features, target_operator)

    def transport(self, features: Array, target_operator: Array) -> Array:
        """Return hard predictions in a documented target outcome definition."""
        return self.predict_definition(features, target_operator)

    def method_diagnostics(self) -> dict:
        """Return the fitted method's definition and optimization diagnostics."""
        if not hasattr(self, "identifiability_"):
            raise RuntimeError("The method is not fitted.")
        return {
            "method": self.method_name,
            "reference_estimator": self.reference_estimator,
            "identifiability": self.identifiability_,
            "structure": self.structure,
            "converged": self.converged_,
            "termination_reason": self.termination_reason_,
            "iterations": self.n_iter_,
            "final_log_likelihood": self.final_log_likelihood_,
            "relative_last_improvement": self.relative_last_improvement_,
            "structure_changes": self.structure_changes_,
            "best_start": self.best_start_,
            "n_init": self.n_init,
            "start_diagnostics": self.start_diagnostics_,
            "log_likelihood_history": self.log_likelihood_history_.copy(),
        }


def make_ordered_threshold_operator(n_classes: int, threshold: int) -> Array:
    """Binary operator with label 1 iff latent class >= threshold."""
    if not 1 <= threshold < n_classes:
        raise ValueError("threshold must lie between 1 and n_classes - 1.")
    matrix = np.zeros((2, n_classes), dtype=float)
    matrix[0, :threshold] = 1.0
    matrix[1, threshold:] = 1.0
    return matrix


def sample_observed_labels(latent_labels: Array, coarsening_matrix: Array, rng: np.random.Generator) -> Array:
    """Sample observed labels from a deterministic or stochastic operator."""
    latent_labels = _as_integer_array(latent_labels, ndim=1, name="latent_labels")
    matrix = np.asarray(coarsening_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("coarsening_matrix must be 2-D with at least two latent classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("coarsening_matrix must be finite, non-negative, and column-stochastic.")
    if np.any(latent_labels < 0) or np.any(latent_labels >= matrix.shape[1]):
        raise ValueError("latent_labels contains a class outside coarsening_matrix.")
    output = np.empty(len(latent_labels), dtype=int)
    for index, latent_class in enumerate(latent_labels):
        output[index] = rng.choice(matrix.shape[0], p=matrix[:, latent_class])
    return output
