"""External-cohort stress test on the four UCI Heart Disease cohorts.

Design
------
The four processed cohorts are naturally distinct data-collection environments:
Cleveland, Hungary, Switzerland, and VA Long Beach.  The native outcome ranges
from 0 to 4.  For this stress test it is mapped to a four-class canonical space
0, 1, 2, 3+ and *only in the three training cohorts* converted to three
complementary threshold definitions.  The VA cohort is held out completely and
used as an external population; its canonical labels are evaluation-only.

This is natural-cohort / semi-synthetic-definition validation.  It is NOT a
claim that the original sites actually used the imposed threshold definitions.
The purpose is to stress the shared P(X,Z) assumption under genuine cohort
shift while preserving a known canonical evaluation target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from plabn import GaliatsatosMethod, coarsening_rank, make_ordered_threshold_operator
from real_data_validation import FoldwiseContinuousPreprocessor, FoldwiseDiscretizer
from publication_experiments import OperatorAwareLogisticEM

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
TRAINING_SITES = ("Cleveland", "Hungary", "Switzerland")
TARGET_SITE = "VA Long Beach"


def load_cohort(site: str, data_dir: Path | None) -> pd.DataFrame:
    filename = FILES[site]
    source = (data_dir / filename) if data_dir is not None else f"{BASE_URL}{filename}"
    frame = pd.read_csv(source, header=None, names=COLUMNS, na_values="?")
    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["num"].notna()].copy()
    frame["site"] = site
    # Canonical four-class space: 0, 1, 2, and 3+.
    frame["canonical"] = np.minimum(frame["num"].astype(int), 3)
    return frame


def deterministic_labels(canonical: np.ndarray, operator: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical, dtype=int)
    return np.argmax(operator[:, canonical], axis=0).astype(int)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-15, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
    }


def aligned_logistic(model: LogisticRegression, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), n_classes), dtype=float)
    out[:, np.asarray(model.classes_, dtype=int)] = raw
    return out / out.sum(axis=1, keepdims=True)


def run(data_dir: Path | None, output_dir: Path) -> None:
    cohorts: Dict[str, pd.DataFrame] = {site: load_cohort(site, data_dir) for site in FILES}
    features = COLUMNS[:-1]
    operators = [make_ordered_threshold_operator(4, threshold) for threshold in (1, 2, 3)]
    rank = coarsening_rank(operators)
    if not rank["full_column_rank"]:
        raise RuntimeError("Training-site operator design is unexpectedly rank deficient.")

    x_train_parts, y_observed_parts, y_canonical_parts = [], [], []
    cohort_rows = []
    for site, operator in zip(TRAINING_SITES, operators):
        frame = cohorts[site]
        x = frame[features].to_numpy(float)
        z = frame["canonical"].to_numpy(int)
        y_obs = deterministic_labels(z, operator)
        x_train_parts.append(x)
        y_observed_parts.append(y_obs)
        y_canonical_parts.append(z)
        cohort_rows.append({
            "site": site,
            "role": "training",
            "n": len(frame),
            "missing_fraction": float(np.isnan(x).mean()),
            **{f"class_{k}_fraction": float(np.mean(z == k)) for k in range(4)},
        })

    target = cohorts[TARGET_SITE]
    x_target = target[features].to_numpy(float)
    z_target = target["canonical"].to_numpy(int)
    cohort_rows.append({
        "site": TARGET_SITE,
        "role": "external_target",
        "n": len(target),
        "missing_fraction": float(np.isnan(x_target).mean()),
        **{f"class_{k}_fraction": float(np.mean(z_target == k)) for k in range(4)},
    })

    # One preprocessor fit on training cohorts only; then split transformed rows
    # back into their natural environments.
    stacked_train = np.vstack(x_train_parts)
    sizes = [len(x) for x in x_train_parts]
    discrete = FoldwiseDiscretizer(n_bins=3, max_features=10)
    train_disc_all = discrete.fit_transform(stacked_train)
    target_disc = discrete.transform(x_target)
    cuts = np.cumsum(sizes)[:-1]
    x_env = list(np.split(train_disc_all, cuts))

    proposed = GaliatsatosMethod(
        n_classes=4,
        structure="tan",
        smoothing=0.10,
        max_iter=120,
        tol=1e-5,
        n_init=3,
        random_state=20260821,
        require_identifiable=True,
    ).fit(x_env, y_observed_parts, operators)
    p_proposed = proposed.predict_proba(target_disc)

    # Operator-aware logistic baseline uses the same training-only imputation
    # and scaling, with no canonical training labels.
    continuous = FoldwiseContinuousPreprocessor()
    train_cont_all = continuous.fit_transform(stacked_train)
    target_cont = continuous.transform(x_target)
    x_env_cont = list(np.split(train_cont_all, cuts))
    op_logistic = OperatorAwareLogisticEM(
        n_classes=4, max_iter=100, tol=1e-6, c_value=1.0
    ).fit(x_env_cont, y_observed_parts, operators)
    p_op_logistic = op_logistic.predict_proba(target_cont)

    # Oracle ceiling: identical training cohorts but with unavailable canonical
    # labels supplied to a conventional classifier.
    oracle_y = np.concatenate(y_canonical_parts)
    oracle = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=20260821)
    oracle.fit(train_cont_all, oracle_y)
    p_oracle = aligned_logistic(oracle, target_cont, 4)

    rows = []
    prediction_rows = []
    for name, p in [
        ("GALIATSATOS/PLA-BN (TAN)", p_proposed),
        ("Operator-aware logistic EM", p_op_logistic),
        ("Oracle-label logistic", p_oracle),
    ]:
        canonical = metrics(z_target, p)
        target_operator = make_ordered_threshold_operator(4, 1)  # any disease vs none
        y_binary = deterministic_labels(z_target, target_operator)
        p_binary = p @ target_operator.T
        binary = metrics(y_binary, p_binary)
        canonical_pred = p.argmax(axis=1)
        binary_pred = p_binary.argmax(axis=1)
        for i in range(len(z_target)):
            prediction_rows.append({
                "row": i, "method": name,
                "canonical_true": int(z_target[i]), "canonical_pred": int(canonical_pred[i]),
                "binary_true": int(y_binary[i]), "binary_pred": int(binary_pred[i]),
                **{f"p{k}": float(p[i, k]) for k in range(4)},
            })
        for metric, value in canonical.items():
            rows.append({"method": name, "outcome_space": "canonical_4_class", "metric": metric, "value": value})
        for metric, value in binary.items():
            rows.append({"method": name, "outcome_space": "external_any_disease_definition", "metric": metric, "value": value})

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "multicohort_heart_metrics.csv", index=False)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(output_dir / "multicohort_heart_external_predictions.csv", index=False)
    for method, group in pred_df.groupby("method"):
        slug = method.lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        pd.crosstab(group["canonical_true"], group["canonical_pred"], rownames=["true"], colnames=["pred"]).to_csv(
            output_dir / f"confusion_canonical_{slug}.csv"
        )
        pd.crosstab(group["binary_true"], group["binary_pred"], rownames=["true"], colnames=["pred"]).to_csv(
            output_dir / f"confusion_binary_{slug}.csv"
        )
    pd.DataFrame(cohort_rows).to_csv(output_dir / "multicohort_heart_cohort_audit.csv", index=False)
    diagnostics = proposed.method_diagnostics()
    serializable = {
        "design": "natural-cohort / semi-synthetic-definition external-population stress test",
        "training_sites": list(TRAINING_SITES),
        "target_site": TARGET_SITE,
        "canonical_mapping": "native num 0->0, 1->1, 2->2, 3/4->3",
        "training_thresholds": [1, 2, 3],
        "rank": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rank.items()},
        "selected_features": discrete.selected_features_.tolist(),
        "effective_bins": discrete.effective_bins_.tolist(),
        "proposed": {
            "converged": diagnostics["converged"],
            "termination_reason": diagnostics["termination_reason"],
            "iterations": diagnostics["iterations"],
            "final_log_likelihood": diagnostics["final_log_likelihood"],
            "best_start": diagnostics["best_start"],
            "n_init": diagnostics["n_init"],
        },
        "canonical_target_labels_used_for_model_fitting": False,
        "external_target_rows_used_for_preprocessing_fit": False,
    }
    (output_dir / "multicohort_heart_manifest.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).pivot_table(index=["outcome_space", "metric"], columns="method", values="value").round(4))
    print(f"Outputs written to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Optional directory containing the four processed UCI files. If omitted, pandas downloads them from UCI.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("publication_results_multicohort_heart"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
