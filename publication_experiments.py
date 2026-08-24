"""Publication-oriented validation for the PLA-BN research prototype.

The script extends the original proof of concept with:
1. 30 paired synthetic replicates and stronger discriminative baselines;
2. an operator-misspecification sensitivity analysis; and
3. an optional semi-synthetic optical-digits benchmark with heterogeneous
   thresholds that is not reported in the current manuscript.

All outputs are written to ``publication_results`` by default.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t, ttest_rel
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from plabn import (
    PLABNClassifier,
    coarsening_rank,
    make_ordered_threshold_operator,
    sample_observed_labels,
)
from run_benchmark import expected_calibration_error, representative_labels, run_replicate
from synthetic import simulate_ordered_tan


warnings.filterwarnings("ignore", category=ConvergenceWarning)


METHOD_ORDER = [
    "PLA-BN (TAN)",
    "Operator-aware logistic EM",
    "Independent-threshold logistic",
    "PLA-BN (Naive Bayes)",
    "Representative-label TAN",
    "Oracle-label TAN",
]


def estimate_prior(labels_by_environment, operators, n_classes: int) -> np.ndarray:
    design_blocks = []
    target_blocks = []
    for labels, operator in zip(labels_by_environment, operators):
        proportions = np.bincount(labels, minlength=operator.shape[0]).astype(float)
        proportions /= max(len(labels), 1)
        weight = np.sqrt(max(len(labels), 1))
        design_blocks.append(weight * operator)
        target_blocks.append(weight * proportions)
    design = np.vstack(design_blocks)
    target = np.concatenate(target_blocks)
    estimate, *_ = np.linalg.lstsq(design, target, rcond=None)
    estimate = np.maximum(estimate, 1e-8)
    return estimate / estimate.sum()


@dataclass
class OperatorAwareLogisticEM:
    n_classes: int
    max_iter: int = 40
    tol: float = 1e-6
    c_value: float = 1.0

    def fit(self, features_by_environment, labels_by_environment, operators):
        features = np.vstack(features_by_environment).astype(float)
        prior = estimate_prior(labels_by_environment, operators, self.n_classes)
        responsibilities = []
        for labels, operator in zip(labels_by_environment, operators):
            weighted = operator[labels, :] * prior[None, :]
            responsibilities.append(weighted / weighted.sum(axis=1, keepdims=True))
        responsibilities = np.vstack(responsibilities)

        repeated_features = np.repeat(features, self.n_classes, axis=0)
        expanded_labels = np.tile(np.arange(self.n_classes), len(features))
        previous_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            model = LogisticRegression(
                C=self.c_value,
                solver="lbfgs",
                max_iter=500,
                random_state=0,
            )
            model.fit(
                repeated_features,
                expanded_labels,
                sample_weight=responsibilities.ravel(),
            )

            updated = []
            observed_likelihood = 0.0
            for x_env, labels, operator in zip(
                features_by_environment, labels_by_environment, operators
            ):
                canonical = model.predict_proba(np.asarray(x_env, dtype=float))
                compatibility = operator[labels, :]
                unnormalized = canonical * compatibility
                denominator = unnormalized.sum(axis=1, keepdims=True)
                updated.append(unnormalized / np.maximum(denominator, 1e-300))
                observed_likelihood += float(np.log(np.maximum(denominator[:, 0], 1e-300)).sum())
            responsibilities = np.vstack(updated)
            self.log_likelihood_history_.append(observed_likelihood)
            if observed_likelihood - previous_likelihood <= self.tol * (
                1.0 + abs(observed_likelihood)
            ):
                break
            previous_likelihood = observed_likelihood

        self.model_ = model
        self.n_iter_ = iteration + 1
        return self

    def predict_proba(self, features) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(features, dtype=float))

    def predict_definition_proba(self, features, operator) -> np.ndarray:
        probabilities = self.predict_proba(features) @ np.asarray(operator, dtype=float).T
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class IndependentThresholdLogistic:
    def __init__(self, n_classes: int):
        self.n_classes = int(n_classes)

    def fit(self, features_by_environment, labels_by_environment):
        if len(features_by_environment) != self.n_classes - 1:
            raise ValueError("One ordered-threshold environment is required per cut-point.")
        self.models_ = []
        for x_env, labels in zip(features_by_environment, labels_by_environment):
            model = LogisticRegression(
                solver="lbfgs", max_iter=500, random_state=0
            ).fit(np.asarray(x_env, dtype=float), labels)
            self.models_.append(model)
        return self

    def predict_proba(self, features) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        cumulative = np.column_stack(
            [model.predict_proba(features)[:, 1] for model in self.models_]
        )
        cumulative = np.sort(cumulative, axis=1)[:, ::-1]
        probabilities = np.column_stack(
            [
                1.0 - cumulative[:, 0],
                *[
                    cumulative[:, index - 1] - cumulative[:, index]
                    for index in range(1, self.n_classes - 1)
                ],
                cumulative[:, -1],
            ]
        )
        probabilities = np.clip(probabilities, 1e-12, None)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict_definition_proba(self, features, operator) -> np.ndarray:
        probabilities = self.predict_proba(features) @ np.asarray(operator, dtype=float).T
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def metric_row(replicate: int, method: str, labels, probabilities) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "replicate": replicate,
        "method": method,
        "canonical_accuracy": accuracy_score(labels, predictions),
        "canonical_macro_f1": f1_score(labels, predictions, average="macro"),
        "canonical_log_loss": log_loss(
            labels, probabilities, labels=np.arange(probabilities.shape[1])
        ),
        "canonical_ece": expected_calibration_error(labels, probabilities),
    }


def definition_row(replicate: int, method: str, labels, probabilities) -> dict:
    return {
        "replicate": replicate,
        "method": method,
        "new_definition_accuracy": accuracy_score(labels, probabilities.argmax(axis=1)),
        "new_definition_log_loss": log_loss(
            labels, probabilities, labels=np.arange(probabilities.shape[1])
        ),
    }


def synthetic_data(replicate: int, n_train: int, n_test: int):
    rng = np.random.default_rng(20260820 + replicate)
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    features_by_environment = []
    latent_by_environment = []
    labels_by_environment = []
    for operator in operators:
        features, latent = simulate_ordered_tan(n_train, rng)
        labels = sample_observed_labels(latent, operator, rng)
        features_by_environment.append(features)
        latent_by_environment.append(latent)
        labels_by_environment.append(labels)
    test_features, test_latent = simulate_ordered_tan(n_test, rng)
    return (
        features_by_environment,
        latent_by_environment,
        labels_by_environment,
        operators,
        test_features,
        test_latent,
    )


def run_core_experiment(replicates: int, n_train: int, n_test: int):
    canonical_rows = []
    definition_rows = []
    novel_operator = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    for replicate in range(replicates):
        base_canonical, base_definition = run_replicate(
            replicate, n_train=n_train, n_test=n_test, max_iter=80
        )
        for row in base_canonical:
            canonical_rows.append(
                {key: value for key, value in row.items() if key != "feature_tree_edge_f1" and key != "em_iterations"}
            )
        definition_rows.extend(base_definition)

        (
            features_by_environment,
            _,
            labels_by_environment,
            operators,
            test_features,
            test_latent,
        ) = synthetic_data(replicate, n_train, n_test)
        novel_labels = np.argmax(novel_operator[:, test_latent], axis=0)

        extra_models = {
            "Operator-aware logistic EM": OperatorAwareLogisticEM(4).fit(
                features_by_environment, labels_by_environment, operators
            ),
            "Independent-threshold logistic": IndependentThresholdLogistic(4).fit(
                features_by_environment, labels_by_environment
            ),
        }
        for method, model in extra_models.items():
            probabilities = model.predict_proba(test_features)
            canonical_rows.append(metric_row(replicate, method, test_latent, probabilities))
            target_probabilities = model.predict_definition_proba(
                test_features, novel_operator
            )
            definition_rows.append(
                definition_row(replicate, method, novel_labels, target_probabilities)
            )
    return pd.DataFrame(canonical_rows), pd.DataFrame(definition_rows)


def confidence_summary(frame: pd.DataFrame, id_columns=("replicate", "method")) -> pd.DataFrame:
    rows = []
    metrics = [column for column in frame.columns if column not in set(id_columns)]
    for method, group in frame.groupby("method"):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_tests(frame: pd.DataFrame, reference="PLA-BN (TAN)") -> pd.DataFrame:
    comparisons = []
    metrics = [column for column in frame.columns if column not in {"replicate", "method"}]
    methods = [method for method in frame["method"].unique() if method != reference]
    raw_p = []
    for metric in metrics:
        pivot = frame.pivot(index="replicate", columns="method", values=metric)
        for method in methods:
            paired = pivot[[reference, method]].dropna()
            difference = paired[reference] - paired[method]
            result = ttest_rel(paired[reference], paired[method])
            sd = float(difference.std(ddof=1))
            half = float(t.ppf(0.975, len(difference) - 1) * sd / np.sqrt(len(difference)))
            comparisons.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparator": method,
                    "n": len(difference),
                    "mean_difference": float(difference.mean()),
                    "ci95_low": float(difference.mean() - half),
                    "ci95_high": float(difference.mean() + half),
                    "p_raw": float(result.pvalue),
                }
            )
            raw_p.append(float(result.pvalue))
    adjusted = holm_adjust(raw_p)
    for row, value in zip(comparisons, adjusted):
        row["p_holm"] = value
    return pd.DataFrame(comparisons)


def soft_threshold_operator(n_classes: int, threshold: int, flip: float) -> np.ndarray:
    deterministic = make_ordered_threshold_operator(n_classes, threshold)
    return (1.0 - flip) * deterministic + flip * (1.0 - deterministic)


def run_operator_robustness(replicates: int, n_train=800, n_test=2500) -> pd.DataFrame:
    rows = []
    deterministic = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    for flip in (0.00, 0.05, 0.10, 0.20):
        correct_operators = [
            soft_threshold_operator(4, threshold, flip) for threshold in (1, 2, 3)
        ]
        for replicate in range(replicates):
            rng = np.random.default_rng(20260900 + replicate + int(flip * 1000))
            features_by_environment = []
            labels_by_environment = []
            for operator in correct_operators:
                features, latent = simulate_ordered_tan(n_train, rng)
                labels = sample_observed_labels(latent, operator, rng)
                features_by_environment.append(features)
                labels_by_environment.append(labels)
            test_features, test_latent = simulate_ordered_tan(n_test, rng)
            specifications = {
                "Correct operator": correct_operators,
                "Deterministic operator assumed": deterministic,
            }
            for specification, fitted_operators in specifications.items():
                model = PLABNClassifier(
                    4, structure="tan", max_iter=60, tol=1e-7, smoothing=0.25
                ).fit(features_by_environment, labels_by_environment, fitted_operators)
                probabilities = model.predict_proba(test_features)
                rows.append(
                    {
                        "replicate": replicate,
                        "flip_probability": flip,
                        "operator_specification": specification,
                        "canonical_accuracy": accuracy_score(
                            test_latent, probabilities.argmax(axis=1)
                        ),
                        "canonical_log_loss": log_loss(
                            test_latent, probabilities, labels=np.arange(4)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def digitize_pixels(train, test, n_features=8):
    variances = train.var(axis=0)
    selected = np.argsort(variances)[-n_features:]
    bins = np.array([0.0, 4.0, 8.0, 12.0])
    train_discrete = np.digitize(train[:, selected], bins=bins, right=True)
    test_discrete = np.digitize(test[:, selected], bins=bins, right=True)
    return train_discrete.astype(int), test_discrete.astype(int)


def digits_target_operator() -> np.ndarray:
    operator = np.zeros((3, 10), dtype=float)
    operator[0, 0:3] = 1.0
    operator[1, 3:7] = 1.0
    operator[2, 7:10] = 1.0
    return operator


def run_digits_benchmark(replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_digits()
    canonical_rows = []
    definition_rows = []
    operators = [make_ordered_threshold_operator(10, threshold) for threshold in range(1, 10)]
    target_operator = digits_target_operator()

    for replicate in range(replicates):
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.30,
            random_state=4100 + replicate,
            stratify=data.target,
        )
        x_train, x_test = digitize_pixels(x_train, x_test)
        splitter = StratifiedKFold(
            n_splits=9, shuffle=True, random_state=5100 + replicate
        )
        folds = [test_index for _, test_index in splitter.split(x_train, y_train)]
        features_by_environment = [x_train[index] for index in folds]
        latent_by_environment = [y_train[index] for index in folds]
        labels_by_environment = [
            np.argmax(operator[:, latent], axis=0)
            for operator, latent in zip(operators, latent_by_environment)
        ]
        all_features = np.vstack(features_by_environment)
        all_latent = np.concatenate(latent_by_environment)
        identity = np.eye(10)

        pseudo_labels = np.concatenate(
            [
                representative_labels(labels, operator)
                for labels, operator in zip(labels_by_environment, operators)
            ]
        )
        models = {
            "PLA-BN (TAN)": PLABNClassifier(
                10, structure="tan", max_iter=25, tol=1e-5, smoothing=0.5
            ).fit(features_by_environment, labels_by_environment, operators),
            "Operator-aware logistic EM": OperatorAwareLogisticEM(
                10, max_iter=50, tol=1e-6
            ).fit(features_by_environment, labels_by_environment, operators),
            "Independent-threshold logistic": IndependentThresholdLogistic(10).fit(
                features_by_environment, labels_by_environment
            ),
            "Representative-label TAN": PLABNClassifier(
                10, structure="tan", max_iter=5, tol=1e-7, smoothing=0.5
            ).fit([all_features], [pseudo_labels], [identity]),
            "Oracle-label TAN": PLABNClassifier(
                10, structure="tan", max_iter=5, tol=1e-7, smoothing=0.5
            ).fit([all_features], [all_latent], [identity]),
        }
        target_labels = np.argmax(target_operator[:, y_test], axis=0)
        for method, model in models.items():
            probabilities = model.predict_proba(x_test)
            canonical_rows.append(metric_row(replicate, method, y_test, probabilities))
            target_probabilities = model.predict_definition_proba(
                x_test, target_operator
            )
            definition_rows.append(
                definition_row(replicate, method, target_labels, target_probabilities)
            )
    return pd.DataFrame(canonical_rows), pd.DataFrame(definition_rows)


def plot_metric_panel(frame, metrics, output_path, method_order, titles):
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#4d4d4d"]
    for axis, metric, title in zip(axes, metrics, titles):
        groups = []
        labels = []
        colors = []
        for index, method in enumerate(method_order):
            values = frame.loc[frame["method"] == method, metric].dropna().to_numpy()
            if len(values):
                groups.append(values)
                labels.append(method)
                colors.append(palette[index % len(palette)])
        plot = axis.boxplot(groups, patch_artist=True, showfliers=False)
        for patch, color in zip(plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_robustness(frame, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for specification, color, marker in [
        ("Correct operator", "#1f77b4", "o"),
        ("Deterministic operator assumed", "#d62728", "s"),
    ]:
        subset = frame[frame["operator_specification"] == specification]
        for axis, metric, title in [
            (axes[0], "canonical_accuracy", "Canonical accuracy"),
            (axes[1], "canonical_log_loss", "Canonical log loss"),
        ]:
            summary = subset.groupby("flip_probability")[metric].agg(["mean", "sem"]).reset_index()
            axis.errorbar(
                summary["flip_probability"],
                summary["mean"],
                yerr=1.96 * summary["sem"],
                label=specification,
                color=color,
                marker=marker,
                capsize=3,
            )
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Label-flip probability")
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean (95% normal CI)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_frame(frame, path):
    frame.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--digits-replicates", type=int, default=10)
    parser.add_argument("--digits-only", action="store_true")
    parser.add_argument(
        "--skip-digits",
        action="store_true",
        help="Run only the synthetic experiments reported in the manuscript.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("publication_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.digits_only:
        digits_canonical, digits_definition = run_digits_benchmark(args.digits_replicates)
        save_frame(digits_canonical, args.output_dir / "digits_canonical_metrics.csv")
        save_frame(digits_definition, args.output_dir / "digits_new_definition_metrics.csv")
        save_frame(confidence_summary(digits_canonical), args.output_dir / "digits_canonical_summary.csv")
        save_frame(confidence_summary(digits_definition), args.output_dir / "digits_new_definition_summary.csv")
        plot_metric_panel(
            digits_canonical,
            ["canonical_accuracy", "canonical_macro_f1"],
            args.output_dir / "figure_digits_metrics.png",
            METHOD_ORDER,
            ["Digits canonical accuracy", "Digits canonical macro-F1"],
        )
        print("Digits canonical summary")
        print(confidence_summary(digits_canonical).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nOutputs: {args.output_dir.resolve()}")
        return

    core_canonical, core_definition = run_core_experiment(
        args.replicates, n_train=1500, n_test=4000
    )
    save_frame(core_canonical, args.output_dir / "core_canonical_metrics.csv")
    save_frame(core_definition, args.output_dir / "core_new_definition_metrics.csv")
    save_frame(confidence_summary(core_canonical), args.output_dir / "core_canonical_summary.csv")
    save_frame(confidence_summary(core_definition), args.output_dir / "core_new_definition_summary.csv")
    save_frame(paired_tests(core_canonical), args.output_dir / "core_canonical_paired_tests.csv")
    save_frame(paired_tests(core_definition), args.output_dir / "core_new_definition_paired_tests.csv")
    plot_metric_panel(
        core_canonical,
        ["canonical_accuracy", "canonical_macro_f1", "canonical_log_loss"],
        args.output_dir / "figure_core_metrics.png",
        METHOD_ORDER,
        ["Canonical accuracy", "Canonical macro-F1", "Canonical log loss"],
    )

    robustness = run_operator_robustness(args.replicates)
    save_frame(robustness, args.output_dir / "operator_robustness_metrics.csv")
    save_frame(
        confidence_summary(
            robustness.rename(
                columns={
                    "flip_probability": "condition",
                    "operator_specification": "method",
                }
            ),
            id_columns=("replicate", "method", "condition"),
        ),
        args.output_dir / "operator_robustness_summary_overall.csv",
    )
    grouped = (
        robustness.groupby(["flip_probability", "operator_specification"])[
            ["canonical_accuracy", "canonical_log_loss"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(filter(None, map(str, column))) for column in grouped.columns]
    save_frame(grouped, args.output_dir / "operator_robustness_summary.csv")
    plot_robustness(robustness, args.output_dir / "figure_operator_robustness.png")

    if not args.skip_digits:
        digits_canonical, digits_definition = run_digits_benchmark(args.digits_replicates)
        save_frame(digits_canonical, args.output_dir / "digits_canonical_metrics.csv")
        save_frame(digits_definition, args.output_dir / "digits_new_definition_metrics.csv")
        save_frame(confidence_summary(digits_canonical), args.output_dir / "digits_canonical_summary.csv")
        save_frame(confidence_summary(digits_definition), args.output_dir / "digits_new_definition_summary.csv")
        plot_metric_panel(
            digits_canonical,
            ["canonical_accuracy", "canonical_macro_f1"],
            args.output_dir / "figure_digits_metrics.png",
            METHOD_ORDER,
            ["Digits canonical accuracy", "Digits canonical macro-F1"],
        )

    diagnostic = {
        "synthetic_threshold_stack": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in coarsening_rank(
                [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
            ).items()
        },
        "synthetic_rank_deficient_stack": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in coarsening_rank(
                [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2)]
            ).items()
        },
        "digits_threshold_stack": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in coarsening_rank(
                [make_ordered_threshold_operator(10, threshold) for threshold in range(1, 10)]
            ).items()
        },
    }
    (args.output_dir / "identifiability_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )

    print("Core canonical summary")
    print(confidence_summary(core_canonical).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not args.skip_digits:
        print("\nDigits canonical summary")
        print(
            confidence_summary(digits_canonical).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
