"""Reproducible synthetic benchmark for the PLA-BN prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from plabn import PLABNClassifier, coarsening_rank, make_ordered_threshold_operator, sample_observed_labels
from synthetic import TRUE_PARENTS, simulate_ordered_tan


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > left) & (confidence <= right)
        if np.any(mask):
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def edge_f1(estimated: set[frozenset[int]], truth: set[frozenset[int]]) -> float:
    if not estimated and not truth:
        return 1.0
    true_positive = len(estimated & truth)
    precision = true_positive / max(len(estimated), 1)
    recall = true_positive / max(len(truth), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def representative_labels(observed: np.ndarray, operator: np.ndarray) -> np.ndarray:
    representatives = []
    for observed_class in range(operator.shape[0]):
        compatible = np.flatnonzero(operator[observed_class] > 0)
        representatives.append(int(np.floor(compatible.mean() + 0.5)))
    return np.asarray(representatives, dtype=int)[observed]


def classification_row(
    replicate: int,
    method: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    model: PLABNClassifier,
    true_edges: set[frozenset[int]],
) -> dict:
    return {
        "replicate": replicate,
        "method": method,
        "canonical_accuracy": accuracy_score(labels, probabilities.argmax(axis=1)),
        "canonical_macro_f1": f1_score(labels, probabilities.argmax(axis=1), average="macro"),
        "canonical_log_loss": log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])),
        "canonical_ece": expected_calibration_error(labels, probabilities),
        "feature_tree_edge_f1": edge_f1(model.feature_edges(), true_edges),
        "em_iterations": model.n_iter_,
    }


def run_replicate(replicate: int, n_train: int, n_test: int, max_iter: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(20260820 + replicate)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    features_by_environment = []
    latent_by_environment = []
    observed_by_environment = []
    for operator in operators:
        features, latent = simulate_ordered_tan(n_train, rng)
        observed = sample_observed_labels(latent, operator, rng)
        features_by_environment.append(features)
        latent_by_environment.append(latent)
        observed_by_environment.append(observed)

    test_features, test_latent = simulate_ordered_tan(n_test, rng)
    all_train_features = np.vstack(features_by_environment)
    all_train_latent = np.concatenate(latent_by_environment)
    identity = np.eye(4)
    true_edges = {
        frozenset((feature, int(parent)))
        for feature, parent in enumerate(TRUE_PARENTS)
        if parent >= 0
    }

    models: dict[str, PLABNClassifier] = {}
    models["PLA-BN (TAN)"] = PLABNClassifier(
        4, structure="tan", max_iter=max_iter, tol=1e-7, smoothing=0.25
    ).fit(features_by_environment, observed_by_environment, operators)
    models["PLA-BN (Naive Bayes)"] = PLABNClassifier(
        4, structure="naive", max_iter=max_iter, tol=1e-7, smoothing=0.25
    ).fit(features_by_environment, observed_by_environment, operators)

    pseudo_labels = np.concatenate(
        [
            representative_labels(observed, operator)
            for observed, operator in zip(observed_by_environment, operators)
        ]
    )
    models["Representative-label TAN"] = PLABNClassifier(
        4, structure="tan", max_iter=5, tol=1e-8, smoothing=0.25
    ).fit([all_train_features], [pseudo_labels], [identity])
    models["Oracle-label TAN"] = PLABNClassifier(
        4, structure="tan", max_iter=5, tol=1e-8, smoothing=0.25
    ).fit([all_train_features], [all_train_latent], [identity])

    canonical_rows = []
    definition_rows = []
    novel_definition = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    novel_labels = np.argmax(novel_definition[:, test_latent], axis=0)
    for method, model in models.items():
        canonical_probabilities = model.predict_proba(test_features)
        canonical_rows.append(
            classification_row(
                replicate,
                method,
                test_latent,
                canonical_probabilities,
                model,
                true_edges,
            )
        )
        definition_probabilities = model.predict_definition_proba(
            test_features, novel_definition
        )
        definition_rows.append(
            {
                "replicate": replicate,
                "method": method,
                "new_definition_accuracy": accuracy_score(
                    novel_labels, definition_probabilities.argmax(axis=1)
                ),
                "new_definition_log_loss": log_loss(
                    novel_labels, definition_probabilities, labels=np.arange(3)
                ),
            }
        )
    return canonical_rows, definition_rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in frame.columns if column not in {"replicate", "method"}]
    means = frame.groupby("method")[numeric].mean().add_suffix("_mean")
    standard_deviations = frame.groupby("method")[numeric].std(ddof=1).add_suffix("_sd")
    return means.join(standard_deviations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    diagnostic = coarsening_rank(operators)
    diagnostic["singular_values"] = diagnostic["singular_values"].tolist()
    (args.output_dir / "identifiability.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )

    canonical_rows: list[dict] = []
    definition_rows: list[dict] = []
    for replicate in range(args.replicates):
        canonical, definitions = run_replicate(
            replicate, args.n_train, args.n_test, args.max_iter
        )
        canonical_rows.extend(canonical)
        definition_rows.extend(definitions)

    canonical_frame = pd.DataFrame(canonical_rows)
    definition_frame = pd.DataFrame(definition_rows)
    canonical_frame.to_csv(args.output_dir / "canonical_metrics.csv", index=False)
    definition_frame.to_csv(args.output_dir / "new_definition_metrics.csv", index=False)
    canonical_summary = summarize(canonical_frame)
    definition_summary = summarize(definition_frame)
    canonical_summary.to_csv(args.output_dir / "canonical_summary.csv", index=False)
    definition_summary.to_csv(args.output_dir / "new_definition_summary.csv", index=False)
    print(canonical_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(definition_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

