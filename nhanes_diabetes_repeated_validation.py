"""Repeated 5x5 NHANES natural-definition validation for GALIATSATOS/PLA-BN.

This script freezes the manuscript-scale validation protocol introduced in v0.6.0:
5 independent repeats of stratified 5-fold outer validation (25 outer evaluations).
Within every outer-training fold, operator calibration, preprocessing fitting,
environment assignment, and all model fitting remain strictly training-only.

Inference notes
---------------
The 25 fold estimates are correlated because training sets overlap. Therefore this
script does NOT report naive paired tests treating 25 folds as independent. It
reports (i) exact Wilcoxon tests on the five repeat-level means and (ii) the
Nadeau-Bengio corrected resampled t-test over the 25 fold-level paired
differences using n_test/n_train = 1/4 for 5-fold outer CV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from nhanes_diabetes_validation import (
    K,
    LAB_DEFINITIONS,
    continuous_preprocessor,
    discrete_features,
    estimate_operator,
    load_nhanes,
    metrics,
    random_environment_split,
)
from plabn import GaliatsatosMethod, coarsening_rank
from publication_experiments import OperatorAwareLogisticEM

METHODS = [
    "GALIATSATOS/PLA-BN TAN",
    "Operator-aware logistic EM",
    "Oracle-label logistic",
]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier"]


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def corrected_resampled_t(diff: np.ndarray, test_train_ratio: float = 0.25) -> dict:
    """Nadeau-Bengio corrected resampled t-test for paired repeated-CV differences."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = float(np.mean(diff))
    if n < 2:
        return {"mean_difference": mean, "corrected_se": np.nan, "t": np.nan,
                "df": n - 1, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    variance = float(np.var(diff, ddof=1))
    corrected_se = float(np.sqrt((1.0 / n + test_train_ratio) * variance))
    if corrected_se == 0:
        t_stat = 0.0 if mean == 0 else np.sign(mean) * np.inf
        p = 1.0 if mean == 0 else 0.0
        ci_low = ci_high = mean
    else:
        t_stat = mean / corrected_se
        p = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
        crit = float(student_t.ppf(0.975, df=n - 1))
        ci_low = mean - crit * corrected_se
        ci_high = mean + crit * corrected_se
    return {
        "mean_difference": mean,
        "corrected_se": corrected_se,
        "t": float(t_stat),
        "df": n - 1,
        "p": p,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def run(
    data_dir: Path,
    output_dir: Path,
    seed: int = 20260821,
    n_repeats: int = 5,
    canonical_mode: str = "max",
    exclude_known_diabetes: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_nhanes(data_dir)
    if exclude_known_diabetes:
        frame = frame.loc[frame["DIQ010"] != 1].reset_index(drop=True)
    if canonical_mode not in {"max", "median"}:
        raise ValueError("canonical_mode must be 'max' or 'median'")

    canonical_col = "canonical_max" if canonical_mode == "max" else "canonical_median"
    y = frame[canonical_col].to_numpy(dtype=int)
    observed_all = frame[["a1c_cat", "fpg_cat", "ogtt_cat"]].to_numpy(dtype=int)
    X_disc_all = discrete_features(frame)
    predictor_cols = ["RIDAGEYR", "BMXBMI", "BMXWAIST", "INDFMPIR", "RIAGENDR", "RIDRETH3", "DMDEDUC2"]
    X_raw = frame[predictor_cols].copy()

    rows: list[dict] = []
    operator_rows: list[dict] = []
    pred_rows: list[dict] = []
    fold_audit: list[dict] = []

    for repeat in range(1, n_repeats + 1):
        repeat_seed = seed + (repeat - 1) * 10007
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)

        for fold, (train_idx, test_idx) in enumerate(outer.split(X_raw, y), start=1):
            split_id = f"R{repeat}F{fold}"
            train_y = y[train_idx]
            dev_rel, cal_rel = train_test_split(
                np.arange(len(train_idx)),
                test_size=0.20,
                random_state=repeat_seed + fold * 101,
                stratify=train_y,
            )
            dev_idx = train_idx[dev_rel]
            cal_idx = train_idx[cal_rel]

            operators = [
                estimate_operator(y[cal_idx], observed_all[cal_idx, j], alpha=0.5)
                for j in range(3)
            ]
            rank_diag = coarsening_rank(operators)
            if not rank_diag["full_column_rank"]:
                raise RuntimeError(f"{split_id}: estimated operators are rank deficient")

            rng = np.random.default_rng(repeat_seed + 1000 + fold)
            env_id = random_environment_split(len(dev_idx), rng)
            x_disc_env = [X_disc_all[dev_idx][env_id == j] for j in range(3)]
            y_obs_env = [observed_all[dev_idx][env_id == j, j] for j in range(3)]

            proposed = GaliatsatosMethod(
                n_classes=K,
                structure="tan",
                smoothing=0.10,
                max_iter=120,
                tol=1e-5,
                n_init=3,
                init_jitter=0.05,
                random_state=repeat_seed + fold,
            ).fit(x_disc_env, y_obs_env, operators)
            p_prop = proposed.predict_proba(X_disc_all[test_idx])

            prep = continuous_preprocessor()
            X_dev_cont = prep.fit_transform(X_raw.iloc[dev_idx])
            X_test_cont = prep.transform(X_raw.iloc[test_idx])
            x_cont_env = [X_dev_cont[env_id == j] for j in range(3)]

            op_log = OperatorAwareLogisticEM(n_classes=K, max_iter=80, tol=1e-6, c_value=1.0)
            op_log.fit(x_cont_env, y_obs_env, operators)
            p_oplog = op_log.predict_proba(X_test_cont)

            oracle = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=repeat_seed + fold)
            oracle.fit(X_dev_cont, y[dev_idx])
            raw_oracle = oracle.predict_proba(X_test_cont)
            p_oracle = np.full((len(test_idx), K), 1e-12)
            for cpos, cls in enumerate(oracle.classes_.astype(int)):
                p_oracle[:, cls] = raw_oracle[:, cpos]
            p_oracle = p_oracle / p_oracle.sum(axis=1, keepdims=True)

            method_probs = {
                METHODS[0]: p_prop,
                METHODS[1]: p_oplog,
                METHODS[2]: p_oracle,
            }
            for method, probs in method_probs.items():
                mr = metrics(y[test_idx], probs)
                rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "evaluation": f"canonical_{canonical_mode}_severity",
                    "method": method,
                    **mr,
                })
                for j, name in enumerate(LAB_DEFINITIONS):
                    p_def = probs @ operators[j].T
                    p_def = p_def / p_def.sum(axis=1, keepdims=True)
                    dm = metrics(observed_all[test_idx, j], p_def)
                    rows.append({
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "evaluation": f"transport_{name}",
                        "method": method,
                        **dm,
                    })

                preds = probs.argmax(axis=1)
                for local, idx in enumerate(test_idx):
                    pred_rows.append({
                        "SEQN": int(frame.iloc[idx]["SEQN"]),
                        "repeat": repeat,
                        "fold": fold,
                        "split_id": split_id,
                        "method": method,
                        "canonical_true": int(y[idx]),
                        "canonical_pred": int(preds[local]),
                        "p0": float(probs[local, 0]),
                        "p1": float(probs[local, 1]),
                        "p2": float(probs[local, 2]),
                        "a1c_cat": int(observed_all[idx, 0]),
                        "fpg_cat": int(observed_all[idx, 1]),
                        "ogtt_cat": int(observed_all[idx, 2]),
                    })

            for j, name in enumerate(LAB_DEFINITIONS):
                op = operators[j]
                for obs in range(K):
                    for z in range(K):
                        operator_rows.append({
                            "repeat": repeat,
                            "fold": fold,
                            "split_id": split_id,
                            "definition": name,
                            "observed_class": obs,
                            "canonical_class": z,
                            "probability": float(op[obs, z]),
                        })

            fold_audit.append({
                "repeat": repeat,
                "fold": fold,
                "split_id": split_id,
                "n_train_total": int(len(train_idx)),
                "n_calibration": int(len(cal_idx)),
                "n_development": int(len(dev_idx)),
                "n_test": int(len(test_idx)),
                "environment_sizes": json.dumps([int(np.sum(env_id == j)) for j in range(3)]),
                "operator_rank": int(rank_diag["rank"]),
                "operator_condition_number": float(rank_diag["condition_number_on_identifiable_subspace"]),
                "proposed_converged": bool(proposed.converged_),
                "proposed_termination_reason": proposed.termination_reason_,
                "proposed_iterations": int(proposed.n_iter_),
                "proposed_best_start": int(proposed.best_start_),
                "canonical_labels_used_for_proposed_fit": False,
                "outer_test_rows_used_for_operator_estimation": False,
                "outer_test_rows_used_for_preprocessing_fit": False,
            })
            print(
                f"{split_id}: PLA-BN acc={metrics(y[test_idx], p_prop)['accuracy']:.3f}, "
                f"macro-F1={metrics(y[test_idx], p_prop)['macro_f1']:.3f}, "
                f"term={proposed.termination_reason_}, iter={proposed.n_iter_}",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "nhanes_repeated_metrics_by_fold.csv", index=False)
    pd.DataFrame(operator_rows).to_csv(output_dir / "nhanes_repeated_operators.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(output_dir / "nhanes_repeated_oof_predictions.csv", index=False)
    audit_df = pd.DataFrame(fold_audit)
    audit_df.to_csv(output_dir / "nhanes_repeated_fold_audit.csv", index=False)

    canonical_eval = f"canonical_{canonical_mode}_severity"
    canon = metrics_df.loc[metrics_df["evaluation"] == canonical_eval].copy()

    fold_summary = canon.groupby("method", as_index=False).agg(
        n_outer_evaluations=("accuracy", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
    )
    fold_summary.to_csv(output_dir / "nhanes_repeated_canonical_summary.csv", index=False)

    repeat_means = canon.groupby(["repeat", "method"], as_index=False)[METRICS].mean()
    repeat_means.to_csv(output_dir / "nhanes_repeated_repeat_means.csv", index=False)

    # 95% confidence intervals based on the five repeat-level means.
    ci_rows = []
    for method, group in repeat_means.groupby("method"):
        for metric in METRICS:
            vals = group[metric].to_numpy(float)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            crit = float(student_t.ppf(0.975, df=len(vals) - 1))
            half = crit * sd / np.sqrt(len(vals))
            ci_rows.append({
                "method": method,
                "metric": metric,
                "repeat_mean": mean,
                "repeat_sd": sd,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "n_repeats": len(vals),
            })
    pd.DataFrame(ci_rows).to_csv(output_dir / "nhanes_repeated_repeat_level_ci.csv", index=False)

    # Pairwise inference: PLA-BN versus operator-aware and oracle baselines.
    test_rows = []
    for comparator in METHODS[1:]:
        for metric in METRICS:
            prop_fold = canon.loc[canon["method"] == METHODS[0], ["split_id", metric]].set_index("split_id")[metric]
            comp_fold = canon.loc[canon["method"] == comparator, ["split_id", metric]].set_index("split_id")[metric]
            common = prop_fold.index.intersection(comp_fold.index)
            diff_fold = (prop_fold.loc[common] - comp_fold.loc[common]).to_numpy(float)
            corrected = corrected_resampled_t(diff_fold, test_train_ratio=0.25)

            prop_rep = repeat_means.loc[repeat_means["method"] == METHODS[0], ["repeat", metric]].set_index("repeat")[metric]
            comp_rep = repeat_means.loc[repeat_means["method"] == comparator, ["repeat", metric]].set_index("repeat")[metric]
            rep_common = prop_rep.index.intersection(comp_rep.index)
            diff_rep = (prop_rep.loc[rep_common] - comp_rep.loc[rep_common]).to_numpy(float)
            try:
                w = wilcoxon(diff_rep, alternative="two-sided", method="exact")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, w_p = 0.0, 1.0

            test_rows.append({
                "comparison": f"{METHODS[0]} minus {comparator}",
                "metric": metric,
                "direction_note": "positive favors PLA-BN for accuracy/balanced_accuracy/macro_f1; negative favors PLA-BN for log_loss/brier",
                "mean_fold_difference": corrected["mean_difference"],
                "corrected_se": corrected["corrected_se"],
                "corrected_t": corrected["t"],
                "corrected_df": corrected["df"],
                "corrected_p": corrected["p"],
                "corrected_ci95_low": corrected["ci_low"],
                "corrected_ci95_high": corrected["ci_high"],
                "repeat_level_wilcoxon_stat": w_stat,
                "repeat_level_wilcoxon_p": w_p,
                "n_outer_evaluations": len(diff_fold),
                "n_repeat_means": len(diff_rep),
            })
    tests = pd.DataFrame(test_rows)
    tests["corrected_p_holm"] = holm_adjust(tests["corrected_p"].to_numpy(float))
    tests["repeat_level_wilcoxon_p_holm"] = holm_adjust(tests["repeat_level_wilcoxon_p"].to_numpy(float))
    tests.to_csv(output_dir / "nhanes_repeated_paired_tests.csv", index=False)

    # Participant-averaged repeated OOF probabilities for stable classwise audit.
    avg = pred_df.groupby(["SEQN", "method", "canonical_true"], as_index=False)[["p0", "p1", "p2"]].mean()
    avg["canonical_pred"] = np.argmax(avg[["p0", "p1", "p2"]].to_numpy(), axis=1)
    avg.to_csv(output_dir / "nhanes_repeated_participant_mean_predictions.csv", index=False)
    class_rows = []
    for method, group in avg.groupby("method"):
        report = classification_report(
            group["canonical_true"], group["canonical_pred"], labels=[0, 1, 2],
            output_dict=True, zero_division=0,
        )
        for cls in ["0", "1", "2"]:
            class_rows.append({"method": method, "class": int(cls), **report[cls]})
    pd.DataFrame(class_rows).to_csv(output_dir / "nhanes_repeated_classwise_metrics.csv", index=False)

    transport_summary = metrics_df.loc[metrics_df["evaluation"].str.startswith("transport_")].groupby(
        ["evaluation", "method"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        log_loss_mean=("log_loss", "mean"),
        brier_mean=("brier", "mean"),
    )
    transport_summary.to_csv(output_dir / "nhanes_repeated_transport_summary.csv", index=False)

    manifest = {
        "dataset": "NHANES 2015-2016 complete-case adults with HbA1c, FPG, and 2-h OGTT",
        "n_participants": int(len(frame)),
        "canonical_mode": canonical_mode,
        "exclude_known_diabetes": bool(exclude_known_diabetes),
        "n_repeats": int(n_repeats),
        "outer_folds_per_repeat": 5,
        "total_outer_evaluations": int(n_repeats * 5),
        "base_seed": int(seed),
        "inference": {
            "primary_descriptive_unit": "outer fold, summarized with repeat-level uncertainty",
            "paired_test_1": "exact Wilcoxon on five repeat-level means",
            "paired_test_2": "Nadeau-Bengio corrected resampled t-test on 25 paired fold differences",
            "multiple_testing": "Holm adjustment across 10 canonical metric-comparison tests",
            "naive_25_fold_independence_assumed": False,
        },
        "leakage_guards": {
            "operators_estimated_inside_outer_training_only": True,
            "preprocessing_fit_inside_outer_training_only": True,
            "canonical_labels_supplied_to_proposed_fit": False,
            "outer_test_used_for_tuning_or_operator_estimation": False,
        },
        "all_operator_ranks_full": bool((audit_df["operator_rank"] == K).all()),
        "all_proposed_tolerance_converged": bool((audit_df["proposed_termination_reason"] == "tolerance").all()),
        "termination_reason_counts": audit_df["proposed_termination_reason"].value_counts().to_dict(),
        "condition_number_range": [
            float(audit_df["operator_condition_number"].min()),
            float(audit_df["operator_condition_number"].max()),
        ],
    }
    with open(output_dir / "nhanes_repeated_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nCANONICAL SUMMARY")
    print(fold_summary.to_string(index=False))
    print("\nPAIRED TESTS")
    print(tests.to_string(index=False))
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("publication_results_diabetes_repeated_5x5"))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--canonical-mode", choices=["max", "median"], default="max")
    parser.add_argument("--exclude-known-diabetes", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.seed, args.n_repeats, args.canonical_mode, args.exclude_known_diabetes)


if __name__ == "__main__":
    main()
