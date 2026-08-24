"""Leakage-safe nested validation of GALIATSATOS on real benchmark data.

The predictor matrices and canonical targets are real observations from the
Iris, Wine, and Optical Digits datasets bundled with scikit-learn.  To create
the heterogeneous-outcome problem required by GALIATSATOS, canonical labels in
each *outer training fold only* are converted to disjoint one-vs-rest outcome
definitions.  The outer test labels remain untouched and are used only after
all predictions have been produced.

This is therefore a real-data / semi-synthetic-definition benchmark, not a
claim of natural multi-study external validation.  Every learned operation is
fold-wise: imputation, feature selection, discretization, environment
construction, hyperparameter selection, and model fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from plabn import GaliatsatosMethod, coarsening_rank
from publication_experiments import OperatorAwareLogisticEM
from run_benchmark import expected_calibration_error, representative_labels


warnings.filterwarnings("ignore", category=ConvergenceWarning)


METHOD_ORDER = [
    "GALIATSATOS/PLA-BN (TAN)",
    "Operator-aware logistic EM",
    "Representative-label logistic",
    "Oracle-label TAN",
    "Oracle-label logistic",
]

PROPOSED_METHOD = METHOD_ORDER[0]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    loader: Callable
    max_features: int
    em_max_iter: int
    logistic_em_max_iter: int
    source_doi: str
    source_url: str


DATASET_SPECS = {
    "iris": DatasetSpec(
        key="iris",
        display_name="Iris",
        loader=load_iris,
        max_features=4,
        em_max_iter=80,
        logistic_em_max_iter=30,
        source_doi="10.24432/C56C76",
        source_url="https://archive.ics.uci.edu/dataset/53/iris",
    ),
    "wine": DatasetSpec(
        key="wine",
        display_name="Wine",
        loader=load_wine,
        max_features=13,
        em_max_iter=80,
        logistic_em_max_iter=30,
        source_doi="10.24432/C5PC7J",
        source_url="https://archive.ics.uci.edu/dataset/109/wine",
    ),
    "digits": DatasetSpec(
        key="digits",
        display_name="Optical Digits",
        loader=load_digits,
        max_features=12,
        em_max_iter=60,
        logistic_em_max_iter=18,
        source_doi="10.24432/C50P49",
        source_url=(
            "https://archive.ics.uci.edu/dataset/80/"
            "optical+recognition+of+handwritten+digits"
        ),
    ),
}


@dataclass(frozen=True)
class ValidationConfig:
    outer_splits: int = 5
    inner_splits: int = 3
    n_bins_candidates: tuple[int, ...] = (3, 4)
    smoothing: float = 0.25
    tolerance: float = 1e-5
    bootstrap_replicates: int = 2000
    random_seed: int = 20260820
    em_n_init: int = 2
    em_init_jitter: float = 0.35


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return sha256_array(np.sort(np.asarray(indices, dtype=np.int64)))


def exact_row_groups(features: np.ndarray) -> np.ndarray:
    """Assign identical feature rows to the same group."""
    _, inverse = np.unique(np.asarray(features), axis=0, return_inverse=True)
    return inverse.astype(int)


def one_vs_rest_operator(n_classes: int, positive_class: int) -> np.ndarray:
    if not 0 <= positive_class < n_classes:
        raise ValueError("positive_class is outside the canonical label space")
    operator = np.zeros((2, n_classes), dtype=float)
    operator[0, :] = 1.0
    operator[0, positive_class] = 0.0
    operator[1, positive_class] = 1.0
    return operator


def training_and_target_operators(n_classes: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Use K-1 one-vs-rest training definitions and hold out the final one."""
    training = [one_vs_rest_operator(n_classes, c) for c in range(n_classes - 1)]
    target = one_vs_rest_operator(n_classes, n_classes - 1)
    diagnostic = coarsening_rank(training)
    if not diagnostic["full_column_rank"]:
        raise RuntimeError("The one-vs-rest training operator stack should have full rank.")
    return training, target


def assign_environments(
    labels: np.ndarray,
    groups: np.ndarray,
    n_environments: int,
    random_seed: int,
) -> np.ndarray:
    """Allocate training-only duplicate groups evenly within canonical class.

    Canonical labels are used only to construct the semi-synthetic measurement
    design inside an outer training fold.  No outer-test row participates.
    """
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)
    rng = np.random.default_rng(random_seed)
    environment = np.full(len(labels), -1, dtype=int)
    offset = 0
    for canonical_class in np.unique(labels):
        class_groups = np.unique(groups[labels == canonical_class])
        rng.shuffle(class_groups)
        for position, group in enumerate(class_groups):
            environment[groups == group] = (position + offset) % n_environments
        offset = (offset + len(class_groups)) % n_environments
    if np.any(environment < 0):
        raise RuntimeError("Environment allocation left an unassigned row.")
    return environment


def coarsen_rows(
    canonical_labels: np.ndarray,
    environment_ids: np.ndarray,
    operators: Sequence[np.ndarray],
) -> np.ndarray:
    canonical_labels = np.asarray(canonical_labels, dtype=int)
    environment_ids = np.asarray(environment_ids, dtype=int)
    observed = np.empty(len(canonical_labels), dtype=int)
    for environment, operator in enumerate(operators):
        mask = environment_ids == environment
        observed[mask] = np.argmax(operator[:, canonical_labels[mask]], axis=0)
    return observed


class FoldwiseDiscretizer:
    """Training-only imputation, variance selection, and quantile binning."""

    def __init__(self, n_bins: int, max_features: int):
        self.n_bins = int(n_bins)
        self.max_features = int(max_features)

    def fit(self, features: np.ndarray) -> "FoldwiseDiscretizer":
        features = np.asarray(features, dtype=float)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(features)
        variances = np.var(imputed, axis=0)
        nonconstant = np.flatnonzero(variances > np.finfo(float).eps)
        if nonconstant.size == 0:
            raise ValueError("No nonconstant predictor remains in this training fold.")
        order = sorted(nonconstant.tolist(), key=lambda j: (-variances[j], j))
        self.selected_features_ = np.asarray(order[: self.max_features], dtype=int)
        self.training_variances_ = variances[self.selected_features_]
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
            subsample=None,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Bins whose width are too small", category=UserWarning
            )
            self.discretizer_.fit(imputed[:, self.selected_features_])
        self.effective_bins_ = np.asarray(self.discretizer_.n_bins_, dtype=int)
        self.n_features_in_ = features.shape[1]
        self.n_fit_rows_ = features.shape[0]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "discretizer_"):
            raise RuntimeError("The fold-wise discretizer is not fitted.")
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("features has an incompatible shape")
        imputed = self.imputer_.transform(features)
        transformed = self.discretizer_.transform(
            imputed[:, self.selected_features_]
        )
        return np.asarray(transformed, dtype=int)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def audit_record(self) -> dict:
        return {
            "n_fit_rows": int(self.n_fit_rows_),
            "n_input_features": int(self.n_features_in_),
            "selected_features": self.selected_features_.tolist(),
            "effective_bins": self.effective_bins_.tolist(),
        }


class FoldwiseContinuousPreprocessor:
    """Training-only median imputation and standardization for logistic models."""

    def fit(self, features: np.ndarray) -> "FoldwiseContinuousPreprocessor":
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = self.imputer_.fit_transform(np.asarray(features, dtype=float))
        self.scaler_ = StandardScaler().fit(imputed)
        self.n_fit_rows_ = len(imputed)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(
            self.imputer_.transform(np.asarray(features, dtype=float))
        )

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


def environment_lists(
    features: np.ndarray,
    observed_labels: np.ndarray,
    environment_ids: np.ndarray,
    n_environments: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    feature_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for environment in range(n_environments):
        mask = environment_ids == environment
        if not np.any(mask):
            raise ValueError(f"Environment {environment} is empty in this fold.")
        feature_list.append(features[mask])
        label_list.append(observed_labels[mask])
    return feature_list, label_list


def observed_log_loss(
    canonical_probabilities: np.ndarray,
    observed_labels: np.ndarray,
    environment_ids: np.ndarray,
    operators: Sequence[np.ndarray],
) -> float:
    row_probabilities = np.empty(len(observed_labels), dtype=float)
    for environment, operator in enumerate(operators):
        mask = environment_ids == environment
        mapped = canonical_probabilities[mask] @ np.asarray(operator).T
        row_probabilities[mask] = mapped[
            np.arange(np.sum(mask)), observed_labels[mask]
        ]
    return float(-np.mean(np.log(np.clip(row_probabilities, 1e-15, 1.0))))


def tune_bins_observed_only(
    features: np.ndarray,
    observed_labels: np.ndarray,
    environment_ids: np.ndarray,
    groups: np.ndarray,
    operators: Sequence[np.ndarray],
    n_classes: int,
    spec: DatasetSpec,
    config: ValidationConfig,
    outer_fold: int,
) -> tuple[int, list[dict]]:
    """Select bin count without using canonical labels or outer-test data."""
    strata = environment_ids * 2 + observed_labels
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=config.random_seed + 1000 + outer_fold,
    )
    splits = list(splitter.split(features, strata, groups))
    candidate_rows: list[dict] = []
    candidate_means: dict[int, float] = {}
    for n_bins in config.n_bins_candidates:
        scores = []
        for inner_fold, (inner_train, inner_validation) in enumerate(splits):
            preprocessor = FoldwiseDiscretizer(n_bins, spec.max_features)
            train_discrete = preprocessor.fit_transform(features[inner_train])
            validation_discrete = preprocessor.transform(features[inner_validation])
            x_env, y_env = environment_lists(
                train_discrete,
                observed_labels[inner_train],
                environment_ids[inner_train],
                len(operators),
            )
            model = GaliatsatosMethod(
                n_classes=n_classes,
                structure="tan",
                smoothing=config.smoothing,
                max_iter=spec.em_max_iter,
                tol=config.tolerance,
                require_identifiable=True,
                random_state=config.random_seed + outer_fold * 10 + inner_fold,
                n_init=config.em_n_init,
                init_jitter=config.em_init_jitter,
            ).fit(x_env, y_env, operators)
            canonical = model.predict_proba(validation_discrete)
            score = observed_log_loss(
                canonical,
                observed_labels[inner_validation],
                environment_ids[inner_validation],
                operators,
            )
            scores.append(score)
            candidate_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_n_bins": n_bins,
                    "inner_fold": inner_fold,
                    "observed_log_loss": score,
                    "inner_train_rows": len(inner_train),
                    "inner_validation_rows": len(inner_validation),
                    "canonical_labels_used_for_tuning": False,
                    "test_rows_used_for_tuning": False,
                }
            )
        candidate_means[n_bins] = float(np.mean(scores))
    selected = min(candidate_means, key=lambda value: (candidate_means[value], value))
    for row in candidate_rows:
        row["selected"] = row["candidate_n_bins"] == selected
        row["candidate_mean_observed_log_loss"] = candidate_means[
            row["candidate_n_bins"]
        ]
    return selected, candidate_rows


def aligned_logistic_probabilities(
    model: LogisticRegression, features: np.ndarray, n_classes: int
) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), n_classes), dtype=float)
    aligned[:, np.asarray(model.classes_, dtype=int)] = raw
    # Preserve an exact zero for classes never seen by the representative-label
    # baseline.  Adding tiny class-specific values can create a meaningless AUC
    # ordering from floating-point noise when every score should be tied.
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_outer_fold(
    train_features: np.ndarray,
    train_canonical_labels: np.ndarray,
    train_groups: np.ndarray,
    test_features: np.ndarray,
    spec: DatasetSpec,
    config: ValidationConfig,
    outer_fold: int,
) -> tuple[dict[str, np.ndarray], dict, list[dict]]:
    """Fit and predict one outer fold; outer-test labels are not an argument."""
    n_classes = int(np.max(train_canonical_labels)) + 1
    operators, target_operator = training_and_target_operators(n_classes)
    environment_ids = assign_environments(
        train_canonical_labels,
        train_groups,
        len(operators),
        config.random_seed + 100 * outer_fold,
    )
    observed_labels = coarsen_rows(
        train_canonical_labels, environment_ids, operators
    )
    selected_bins, tuning_rows = tune_bins_observed_only(
        train_features,
        observed_labels,
        environment_ids,
        train_groups,
        operators,
        n_classes,
        spec,
        config,
        outer_fold,
    )

    discrete = FoldwiseDiscretizer(selected_bins, spec.max_features)
    train_discrete = discrete.fit_transform(train_features)
    test_discrete = discrete.transform(test_features)
    x_env, y_env = environment_lists(
        train_discrete,
        observed_labels,
        environment_ids,
        len(operators),
    )

    proposed = GaliatsatosMethod(
        n_classes=n_classes,
        structure="tan",
        smoothing=config.smoothing,
        max_iter=spec.em_max_iter,
        tol=config.tolerance,
        require_identifiable=True,
        random_state=config.random_seed + outer_fold,
        n_init=config.em_n_init,
        init_jitter=config.em_init_jitter,
    ).fit(x_env, y_env, operators)
    predictions: dict[str, np.ndarray] = {
        PROPOSED_METHOD: proposed.predict_proba(test_discrete)
    }

    identity = np.eye(n_classes)
    oracle_tan = GaliatsatosMethod(
        n_classes=n_classes,
        structure="tan",
        smoothing=config.smoothing,
        max_iter=5,
        tol=config.tolerance,
        require_identifiable=True,
        random_state=config.random_seed + outer_fold,
        n_init=1,
    ).fit([train_discrete], [train_canonical_labels], [identity])
    predictions["Oracle-label TAN"] = oracle_tan.predict_proba(test_discrete)

    continuous = FoldwiseContinuousPreprocessor()
    train_continuous = continuous.fit_transform(train_features)
    test_continuous = continuous.transform(test_features)
    x_env_continuous, y_env_continuous = environment_lists(
        train_continuous,
        observed_labels,
        environment_ids,
        len(operators),
    )
    operator_logistic = OperatorAwareLogisticEM(
        n_classes=n_classes,
        max_iter=spec.logistic_em_max_iter,
        tol=1e-6,
        c_value=1.0,
    ).fit(x_env_continuous, y_env_continuous, operators)
    predictions["Operator-aware logistic EM"] = operator_logistic.predict_proba(
        test_continuous
    )

    pseudo_labels = np.concatenate(
        [
            representative_labels(labels, operator)
            for labels, operator in zip(y_env_continuous, operators)
        ]
    )
    pseudo_features = np.vstack(x_env_continuous)
    representative = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=config.random_seed
    ).fit(pseudo_features, pseudo_labels)
    predictions["Representative-label logistic"] = aligned_logistic_probabilities(
        representative, test_continuous, n_classes
    )

    oracle_logistic = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=config.random_seed
    ).fit(train_continuous, train_canonical_labels)
    predictions["Oracle-label logistic"] = aligned_logistic_probabilities(
        oracle_logistic, test_continuous, n_classes
    )

    for method, probability in predictions.items():
        if probability.shape != (len(test_features), n_classes):
            raise RuntimeError(f"{method} returned an incompatible probability matrix")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
            raise RuntimeError(f"{method} returned unnormalized probabilities")

    audit = {
        "outer_fold": outer_fold,
        "n_train": len(train_features),
        "n_test": len(test_features),
        "selected_n_bins": selected_bins,
        "preprocessor": discrete.audit_record(),
        "rank_diagnostic": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in coarsening_rank(operators).items()
        },
        "environment_counts": np.bincount(
            environment_ids, minlength=len(operators)
        ).tolist(),
        "observed_positive_counts": [
            int(np.sum(observed_labels[environment_ids == environment] == 1))
            for environment in range(len(operators))
        ],
        "proposed_converged": bool(proposed.converged_),
        "proposed_termination_reason": proposed.termination_reason_,
        "proposed_iterations": int(proposed.n_iter_),
        "proposed_final_log_likelihood": float(proposed.final_log_likelihood_),
        "proposed_relative_last_improvement": float(proposed.relative_last_improvement_),
        "proposed_structure_changes": int(proposed.structure_changes_),
        "proposed_best_start": int(proposed.best_start_),
        "proposed_n_init": int(proposed.n_init),
        "operator_logistic_iterations": int(operator_logistic.n_iter_),
        "outer_test_labels_passed_to_fit": False,
        "hyperparameter_selection_target": "observed-label log loss only",
        "target_definition_used_for_training": False,
        "target_operator": target_operator.tolist(),
    }
    return predictions, audit, tuning_rows


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "ece": float(expected_calibration_error(labels, probabilities)),
    }


def binary_target_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(2))),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
    }


def bootstrap_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric_function: Callable[[np.ndarray, np.ndarray], dict],
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, tuple[float, float, float]]:
    point = metric_function(labels, probabilities)
    rng = np.random.default_rng(random_seed)
    draws = {metric: [] for metric in point}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        metrics = metric_function(labels[indices], probabilities[indices])
        for metric, value in metrics.items():
            draws[metric].append(value)
    return {
        metric: (
            value,
            float(np.quantile(draws[metric], 0.025)),
            float(np.quantile(draws[metric], 0.975)),
        )
        for metric, value in point.items()
    }


def audit_dataset(spec: DatasetSpec, features: np.ndarray, labels: np.ndarray) -> dict:
    frame = pd.DataFrame(features)
    frame_with_label = frame.copy()
    frame_with_label["__target__"] = labels
    class_counts = np.bincount(labels)
    return {
        "dataset": spec.display_name,
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_classes": int(len(class_counts)),
        "minimum_class_count": int(class_counts.min()),
        "maximum_class_count": int(class_counts.max()),
        "class_counts": ";".join(map(str, class_counts.tolist())),
        "missing_cells": int(np.isnan(features).sum()),
        "nonfinite_cells": int((~np.isfinite(features)).sum()),
        "constant_features": int(np.sum(np.ptp(features, axis=0) == 0)),
        "rows_in_duplicate_feature_groups": int(frame.duplicated(keep=False).sum()),
        "rows_in_duplicate_feature_target_groups": int(
            frame_with_label.duplicated(keep=False).sum()
        ),
        "unique_feature_rows": int(len(frame.drop_duplicates())),
        "dataset_sha256": sha256_array(features, labels),
        "source_doi": spec.source_doi,
        "source_url": spec.source_url,
        "quality_status": "usable_with_reported_limitations",
    }


def run_dataset(
    spec: DatasetSpec,
    config: ValidationConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
    dict,
]:
    bunch = spec.loader()
    features = np.asarray(bunch.data, dtype=float)
    canonical_labels = np.asarray(bunch.target, dtype=int)
    n_classes = int(np.max(canonical_labels)) + 1
    groups = exact_row_groups(features)
    quality = audit_dataset(spec, features, canonical_labels)
    operators, target_operator = training_and_target_operators(n_classes)
    outer = StratifiedGroupKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.random_seed,
    )

    oof = {
        method: np.full((len(features), n_classes), np.nan, dtype=float)
        for method in METHOD_ORDER
    }
    fold_id = np.full(len(features), -1, dtype=int)
    prediction_count = np.zeros(len(features), dtype=int)
    fold_metrics: list[dict] = []
    tuning_rows: list[dict] = []
    fold_audits: list[dict] = []

    for outer_fold, (train_index, test_index) in enumerate(
        outer.split(features, canonical_labels, groups)
    ):
        if np.intersect1d(train_index, test_index).size:
            raise RuntimeError("Outer train/test index overlap detected.")
        predictions, audit, fold_tuning = fit_outer_fold(
            features[train_index],
            canonical_labels[train_index],
            groups[train_index],
            features[test_index],
            spec,
            config,
            outer_fold,
        )
        fold_id[test_index] = outer_fold
        prediction_count[test_index] += 1
        for method, probability in predictions.items():
            oof[method][test_index] = probability
            canonical = multiclass_metrics(canonical_labels[test_index], probability)
            target_labels = np.argmax(
                target_operator[:, canonical_labels[test_index]], axis=0
            )
            target_probability = probability @ target_operator.T
            target_probability = np.clip(target_probability, 0.0, 1.0)
            target_probability /= target_probability.sum(axis=1, keepdims=True)
            target = binary_target_metrics(target_labels, target_probability)
            for metric, value in canonical.items():
                fold_metrics.append(
                    {
                        "dataset": spec.display_name,
                        "outer_fold": outer_fold,
                        "method": method,
                        "outcome_space": "canonical",
                        "metric": metric,
                        "value": value,
                        "n_test": len(test_index),
                    }
                )
            for metric, value in target.items():
                fold_metrics.append(
                    {
                        "dataset": spec.display_name,
                        "outer_fold": outer_fold,
                        "method": method,
                        "outcome_space": "held_out_definition",
                        "metric": metric,
                        "value": value,
                        "n_test": len(test_index),
                    }
                )
        audit.update(
            {
                "dataset": spec.display_name,
                "train_index_sha256": sha256_indices(train_index),
                "test_index_sha256": sha256_indices(test_index),
                "index_overlap_count": 0,
                "duplicate_groups_split_across_train_test": int(
                    len(set(groups[train_index]) & set(groups[test_index]))
                ),
            }
        )
        if audit["duplicate_groups_split_across_train_test"] != 0:
            raise RuntimeError("An exact duplicate group crosses an outer fold boundary.")
        fold_audits.append(audit)
        for row in fold_tuning:
            tuning_rows.append({"dataset": spec.display_name, **row})

    if np.any(fold_id < 0) or not np.all(prediction_count == 1):
        raise RuntimeError("Every row must receive exactly one outer-fold prediction.")
    for method, probability in oof.items():
        if np.any(~np.isfinite(probability)):
            raise RuntimeError(f"Missing or nonfinite OOF predictions for {method}.")

    summary_rows: list[dict] = []
    target_rows: list[dict] = []
    canonical_prediction_rows: list[dict] = []
    target_prediction_rows: list[dict] = []
    target_labels_all = np.argmax(target_operator[:, canonical_labels], axis=0)
    for method_index, method in enumerate(METHOD_ORDER):
        probability = oof[method]
        canonical_summary = bootstrap_summary(
            canonical_labels,
            probability,
            multiclass_metrics,
            config.bootstrap_replicates,
            config.random_seed + 10000 + method_index,
        )
        for metric, (estimate, low, high) in canonical_summary.items():
            summary_rows.append(
                {
                    "dataset": spec.display_name,
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(features),
                    "interval": "paired-row percentile bootstrap",
                }
            )
        target_probability = probability @ target_operator.T
        target_probability = np.clip(target_probability, 0.0, 1.0)
        target_probability /= target_probability.sum(axis=1, keepdims=True)
        target_summary = bootstrap_summary(
            target_labels_all,
            target_probability,
            binary_target_metrics,
            config.bootstrap_replicates,
            config.random_seed + 20000 + method_index,
        )
        for metric, (estimate, low, high) in target_summary.items():
            target_rows.append(
                {
                    "dataset": spec.display_name,
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": len(features),
                    "interval": "paired-row percentile bootstrap",
                }
            )
        for row_index in range(len(features)):
            base = {
                "dataset": spec.display_name,
                "sample_index": row_index,
                "outer_fold": int(fold_id[row_index]),
                "method": method,
                "true_label": int(canonical_labels[row_index]),
                "predicted_label": int(np.argmax(probability[row_index])),
            }
            canonical_prediction_rows.append(
                {
                    **base,
                    **{
                        f"p_class_{canonical_class}": float(
                            probability[row_index, canonical_class]
                        )
                        for canonical_class in range(n_classes)
                    },
                }
            )
            target_prediction_rows.append(
                {
                    "dataset": spec.display_name,
                    "sample_index": row_index,
                    "outer_fold": int(fold_id[row_index]),
                    "method": method,
                    "true_target_label": int(target_labels_all[row_index]),
                    "predicted_target_label": int(
                        np.argmax(target_probability[row_index])
                    ),
                    "p_target_0": float(target_probability[row_index, 0]),
                    "p_target_1": float(target_probability[row_index, 1]),
                }
            )

    quality["outer_fold_count"] = config.outer_splits
    quality["all_rows_predicted_once"] = bool(np.all(prediction_count == 1))
    quality["operator_stack_rank"] = int(coarsening_rank(operators)["rank"])
    quality["operator_stack_full_column_rank"] = bool(
        coarsening_rank(operators)["full_column_rank"]
    )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(target_rows),
        pd.DataFrame(canonical_prediction_rows),
        pd.DataFrame(target_prediction_rows),
        fold_audits,
        quality,
        pd.DataFrame(fold_metrics),
        pd.DataFrame(tuning_rows),
    )


def plot_real_data_summary(
    canonical_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Two-panel dot-and-interval comparison for the manuscript."""
    available = set(canonical_summary["dataset"].unique())
    datasets = [
        DATASET_SPECS[key].display_name
        for key in DATASET_SPECS
        if DATASET_SPECS[key].display_name in available
    ]
    method_styles = {
        PROPOSED_METHOD: ("#2F5597", "o", "-"),
        "Operator-aware logistic EM": ("#D97706", "s", "--"),
        "Representative-label logistic": ("#777777", "^", ":"),
        "Oracle-label TAN": ("#222222", "D", "-."),
        "Oracle-label logistic": ("#999999", "P", "--"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=False)
    panels = [
        (canonical_summary, "macro_f1", "Canonical macro-F1"),
        (target_summary, "balanced_accuracy", "Held-out-definition balanced accuracy"),
    ]
    x = np.arange(len(datasets), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(METHOD_ORDER))
    for axis, (frame, metric, title) in zip(axes, panels):
        subset = frame[frame["metric"] == metric]
        for offset, method in zip(offsets, METHOD_ORDER):
            ordered = subset[subset["method"] == method].set_index("dataset").loc[
                datasets
            ]
            estimate = ordered["estimate"].to_numpy()
            lower = estimate - ordered["ci95_low"].to_numpy()
            upper = ordered["ci95_high"].to_numpy() - estimate
            color, marker, linestyle = method_styles[method]
            axis.errorbar(
                x + offset,
                estimate,
                yerr=np.vstack([lower, upper]),
                fmt=marker,
                color=color,
                linestyle="none",
                markersize=5.5,
                capsize=2.5,
                elinewidth=1.2,
                label=method,
            )
        axis.set_title(title, fontsize=10.5, color="#222222")
        axis.set_xticks(x, datasets)
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel("Estimate with 95% bootstrap interval")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "Real-data, semi-synthetic-definition nested validation",
        fontsize=12,
        fontweight="bold",
        color="#1F1F1F",
    )
    figure.text(
        0.5,
        0.91,
        "Five outer folds; three inner folds; all learned preprocessing refitted within fold",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.89))
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(
    dataset_keys: Iterable[str], output_dir: Path, config: ValidationConfig
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    target_summaries = []
    canonical_predictions = []
    target_predictions = []
    fold_metrics = []
    tuning = []
    audits = []
    data_quality = []
    for dataset_key in dataset_keys:
        spec = DATASET_SPECS[dataset_key]
        print(f"Running {spec.display_name} ...", flush=True)
        (
            summary,
            target_summary,
            canonical_oof,
            target_oof,
            fold_audit,
            quality,
            dataset_fold_metrics,
            dataset_tuning,
        ) = run_dataset(spec, config)
        summaries.append(summary)
        target_summaries.append(target_summary)
        canonical_predictions.append(canonical_oof)
        target_predictions.append(target_oof)
        fold_metrics.append(dataset_fold_metrics)
        tuning.append(dataset_tuning)
        audits.extend(fold_audit)
        data_quality.append(quality)

    summary_frame = pd.concat(summaries, ignore_index=True)
    target_summary_frame = pd.concat(target_summaries, ignore_index=True)
    canonical_prediction_frame = pd.concat(canonical_predictions, ignore_index=True)
    target_prediction_frame = pd.concat(target_predictions, ignore_index=True)
    fold_metric_frame = pd.concat(fold_metrics, ignore_index=True)
    tuning_frame = pd.concat(tuning, ignore_index=True)

    summary_frame.to_csv(output_dir / "real_data_canonical_summary.csv", index=False)
    target_summary_frame.to_csv(
        output_dir / "real_data_held_out_definition_summary.csv", index=False
    )
    canonical_prediction_frame.to_csv(
        output_dir / "real_data_canonical_oof_predictions.csv", index=False
    )
    target_prediction_frame.to_csv(
        output_dir / "real_data_held_out_definition_oof_predictions.csv", index=False
    )
    fold_metric_frame.to_csv(output_dir / "real_data_fold_metrics.csv", index=False)
    tuning_frame.to_csv(output_dir / "real_data_inner_tuning.csv", index=False)
    pd.DataFrame(data_quality).to_csv(
        output_dir / "real_data_quality_audit.csv", index=False
    )
    plot_real_data_summary(
        summary_frame,
        target_summary_frame,
        output_dir / "figure_real_data_nested_validation.png",
    )
    manifest = {
        "status": "completed",
        "validation_design": "nested stratified group cross-validation",
        "real_data_scope": (
            "Real predictors and canonical targets; training-fold-only "
            "semi-synthetic one-vs-rest outcome definitions"
        ),
        "config": asdict(config),
        "datasets": list(dataset_keys),
        "methods": METHOD_ORDER,
        "leakage_controls": {
            "outer_test_labels_passed_to_fit": False,
            "all_preprocessing_fit_inside_training_fold": True,
            "inner_selection_uses_canonical_labels": False,
            "inner_selection_uses_observed_labels": True,
            "exact_duplicate_groups_kept_within_outer_fold": True,
            "held_out_target_definition_used_for_training": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "fold_audits": audits,
    }
    (output_dir / "real_data_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Outputs written to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("publication_results")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smoke test with 3x2 folds and 100 bootstrap draws.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.outer_splits = 3
        args.inner_splits = 2
        args.bootstrap_replicates = 100
    config = ValidationConfig(
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        bootstrap_replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
    )
    write_outputs(args.datasets, args.output_dir, config)


if __name__ == "__main__":
    main()
