"""Natural-definition diabetes validation for the GALIATSATOS/PLA-BN method.

NHANES 2015-2016 participants with HbA1c, fasting plasma glucose (FPG), and
2-hour OGTT are categorized using ADA-style glycemic thresholds. The three
laboratory definitions are genuinely discordant on the same participants.

Primary canonical endpoint
--------------------------
A research *screening severity* endpoint is defined as the maximum severity
across the three laboratory definitions (0 normal, 1 intermediate/prediabetes,
2 diabetes-range). This is an operational research endpoint, not a clinical
diagnosis and not a replacement for confirmatory testing.

Validation protocol
-------------------
Five outer stratified folds are used. Within each outer-training fold, a
stratified calibration subset estimates stochastic definition operators
P(test-category | canonical screening severity), with Dirichlet smoothing.
The remaining development rows are randomly divided into three disjoint
environments independent of outcome; each environment reveals only one real
laboratory definition (HbA1c, FPG, or OGTT). Canonical labels are not supplied
to GALIATSATOS/PLA-BN or the operator-aware logistic baseline. The outer test
fold remains untouched until evaluation.

This is therefore a natural-definition / semi-synthetic-environment validation:
the laboratory definitions and their disagreement are real, while assignment
of complete-case participants to definition environments is experimental.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from plabn import GaliatsatosMethod, coarsening_rank
from publication_experiments import OperatorAwareLogisticEM


LAB_DEFINITIONS = ["HbA1c", "FPG", "OGTT"]
K = 3


def categorize_a1c(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 6.5, 2, np.where(x >= 5.7, 1, 0)).astype(int)


def categorize_fpg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 126.0, 2, np.where(x >= 100.0, 1, 0)).astype(int)


def categorize_ogtt(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 200.0, 2, np.where(x >= 140.0, 1, 0)).astype(int)


def load_nhanes(data_dir: Path) -> pd.DataFrame:
    demo = pd.read_csv(data_dir / "DEMO_I.csv")
    bmx = pd.read_csv(data_dir / "BMX_I.csv")
    diq = pd.read_csv(data_dir / "DIQ_I.csv")
    ghb = pd.read_csv(data_dir / "GHB_I.csv")
    glu = pd.read_csv(data_dir / "GLU_I.csv")
    ogtt = pd.read_csv(data_dir / "OGTT_I.csv")

    frame = (
        demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "DMDEDUC2",
              "SDMVPSU", "SDMVSTRA", "WTMEC2YR"]]
        .merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="inner")
        .merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
        .merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
        .merge(glu[["SEQN", "WTSAF2YR", "LBXGLU"]], on="SEQN", how="left")
        .merge(ogtt[["SEQN", "WTSOG2YR", "LBXGLT", "GTDCODE"]], on="SEQN", how="left")
    )
    # Adult complete cases. Predictor completeness is deliberately modest;
    # INDFMPIR/education can be imputed downstream, but core labs and body
    # measures required for the benchmark must be observed.
    frame = frame.loc[frame["RIDAGEYR"] >= 20].copy()
    frame = frame.dropna(
        subset=["LBXGH", "LBXGLU", "LBXGLT", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
                "BMXBMI", "BMXWAIST"]
    ).reset_index(drop=True)

    frame["a1c_cat"] = categorize_a1c(frame["LBXGH"].to_numpy())
    frame["fpg_cat"] = categorize_fpg(frame["LBXGLU"].to_numpy())
    frame["ogtt_cat"] = categorize_ogtt(frame["LBXGLT"].to_numpy())
    labs = frame[["a1c_cat", "fpg_cat", "ogtt_cat"]].to_numpy(dtype=int)
    frame["canonical_max"] = labs.max(axis=1)
    frame["canonical_median"] = np.median(labs, axis=1).astype(int)
    frame["all_three_agree"] = (labs[:, 0] == labs[:, 1]) & (labs[:, 1] == labs[:, 2])
    return frame


def discrete_features(frame: pd.DataFrame) -> np.ndarray:
    """Fixed, interpretable bins for the discrete BN (no data-dependent cutpoints)."""
    age = frame["RIDAGEYR"].to_numpy(float)
    bmi = frame["BMXBMI"].to_numpy(float)
    waist = frame["BMXWAIST"].to_numpy(float)
    pir = frame["INDFMPIR"].to_numpy(float)
    sex = frame["RIAGENDR"].to_numpy(float)
    race = frame["RIDRETH3"].to_numpy(float)
    edu = frame["DMDEDUC2"].to_numpy(float)

    age_bin = np.digitize(age, [40, 60], right=False)  # 20-39,40-59,60+
    bmi_bin = np.digitize(bmi, [18.5, 25.0, 30.0], right=False)
    waist_bin = np.digitize(waist, [80.0, 94.0, 102.0], right=False)
    pir_clean = np.where(np.isfinite(pir), pir, -1.0)
    pir_bin = np.where(pir_clean < 0, 0, np.digitize(pir_clean, [1.0, 2.0, 4.0], right=False) + 1)
    sex_code = np.where(sex == 1, 0, 1).astype(int)
    race_levels = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    race_code = np.array([race_levels.get(int(v), 6) if np.isfinite(v) else 6 for v in race], dtype=int)
    edu_code = np.array([int(v) if np.isfinite(v) and 1 <= int(v) <= 5 else 0 for v in edu], dtype=int)
    return np.column_stack([age_bin, sex_code, race_code, bmi_bin, waist_bin, pir_bin, edu_code]).astype(int)


def continuous_preprocessor() -> ColumnTransformer:
    numeric = ["RIDAGEYR", "BMXBMI", "BMXWAIST", "INDFMPIR"]
    categorical = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )


def estimate_operator(canonical: np.ndarray, observed: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    counts = np.full((K, K), float(alpha), dtype=float)
    for z, y in zip(canonical, observed):
        counts[int(y), int(z)] += 1.0
    return counts / counts.sum(axis=0, keepdims=True)


def random_environment_split(n: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n)
    env = np.empty(n, dtype=int)
    env[order] = np.arange(n) % 3
    return env


def multiclass_brier(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = np.argmax(p, axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
        "brier": multiclass_brier(y, p),
    }


def run(data_dir: Path, output_dir: Path, seed: int = 20260821, canonical_mode: str = "max", exclude_known_diabetes: bool = False) -> None:
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

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    operator_rows = []
    pred_rows = []
    fold_audit = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X_raw, y), start=1):
        train_y = y[train_idx]
        dev_rel, cal_rel = train_test_split(
            np.arange(len(train_idx)), test_size=0.20, random_state=seed + fold,
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
            raise RuntimeError(f"Fold {fold}: estimated operators are rank deficient")

        rng = np.random.default_rng(seed + 1000 + fold)
        env_id = random_environment_split(len(dev_idx), rng)

        x_disc_env = [X_disc_all[dev_idx][env_id == j] for j in range(3)]
        y_obs_env = [observed_all[dev_idx][env_id == j, j] for j in range(3)]

        proposed = GaliatsatosMethod(
            n_classes=K, structure="tan", smoothing=0.10, max_iter=120,
            tol=1e-5, n_init=3, init_jitter=0.05, random_state=seed + fold,
        ).fit(x_disc_env, y_obs_env, operators)
        p_prop = proposed.predict_proba(X_disc_all[test_idx])

        prep = continuous_preprocessor()
        X_dev_cont = prep.fit_transform(X_raw.iloc[dev_idx])
        X_test_cont = prep.transform(X_raw.iloc[test_idx])
        x_cont_env = [X_dev_cont[env_id == j] for j in range(3)]

        op_log = OperatorAwareLogisticEM(n_classes=K, max_iter=80, tol=1e-6, c_value=1.0)
        op_log.fit(x_cont_env, y_obs_env, operators)
        p_oplog = op_log.predict_proba(X_test_cont)

        oracle = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed + fold)
        oracle.fit(X_dev_cont, y[dev_idx])
        p_oracle = oracle.predict_proba(X_test_cont)
        # Align columns defensively.
        aligned = np.full((len(test_idx), K), 1e-12)
        for cpos, cls in enumerate(oracle.classes_.astype(int)):
            aligned[:, cls] = p_oracle[:, cpos]
        p_oracle = aligned / aligned.sum(axis=1, keepdims=True)

        method_probs = {
            "GALIATSATOS/PLA-BN TAN": p_prop,
            "Operator-aware logistic EM": p_oplog,
            "Oracle-label logistic": p_oracle,
        }
        for method, probs in method_probs.items():
            mr = metrics(y[test_idx], probs)
            rows.append({"fold": fold, "evaluation": f"canonical_{canonical_mode}_severity", "method": method, **mr})
            for j, name in enumerate(LAB_DEFINITIONS):
                p_def = probs @ operators[j].T
                p_def = p_def / p_def.sum(axis=1, keepdims=True)
                dm = metrics(observed_all[test_idx, j], p_def)
                rows.append({"fold": fold, "evaluation": f"transport_{name}", "method": method, **dm})

            preds = probs.argmax(axis=1)
            for local, idx in enumerate(test_idx):
                pred_rows.append({
                    "SEQN": int(frame.iloc[idx]["SEQN"]), "fold": fold, "method": method,
                    "canonical_true": int(y[idx]), "canonical_pred": int(preds[local]),
                    "p0": float(probs[local, 0]), "p1": float(probs[local, 1]), "p2": float(probs[local, 2]),
                    "a1c_cat": int(observed_all[idx, 0]), "fpg_cat": int(observed_all[idx, 1]),
                    "ogtt_cat": int(observed_all[idx, 2]),
                })

        for j, name in enumerate(LAB_DEFINITIONS):
            op = operators[j]
            for obs in range(K):
                for z in range(K):
                    operator_rows.append({
                        "fold": fold, "definition": name, "observed_class": obs,
                        "canonical_class": z, "probability": float(op[obs, z]),
                    })

        fold_audit.append({
            "fold": fold,
            "n_train_total": int(len(train_idx)), "n_calibration": int(len(cal_idx)),
            "n_development": int(len(dev_idx)), "n_test": int(len(test_idx)),
            "environment_sizes": [int(np.sum(env_id == j)) for j in range(3)],
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

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "nhanes_diabetes_metrics_by_fold.csv", index=False)
    summary = metrics_df.groupby(["evaluation", "method"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), log_loss_mean=("log_loss", "mean"),
        brier_mean=("brier", "mean"),
    )
    summary.to_csv(output_dir / "nhanes_diabetes_metrics_summary.csv", index=False)
    pd.DataFrame(operator_rows).to_csv(output_dir / "nhanes_diabetes_operators.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(output_dir / "nhanes_diabetes_oof_predictions.csv", index=False)
    class_rows = []
    for method, group in pred_df.groupby("method"):
        slug = method.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        pd.crosstab(group["canonical_true"], group["canonical_pred"], rownames=["true"], colnames=["pred"]).to_csv(
            output_dir / f"confusion_canonical_{slug}.csv"
        )
        report = classification_report(
            group["canonical_true"], group["canonical_pred"], labels=[0, 1, 2],
            output_dict=True, zero_division=0
        )
        for cls in ["0", "1", "2"]:
            class_rows.append({"method": method, "class": int(cls), **report[cls]})
    pd.DataFrame(class_rows).to_csv(output_dir / "nhanes_diabetes_classwise_metrics.csv", index=False)
    pd.DataFrame(fold_audit).to_csv(output_dir / "nhanes_diabetes_fold_audit.csv", index=False)

    agreement = {
        "n_complete_adults": int(len(frame)),
        "all_three_agree_fraction": float(frame["all_three_agree"].mean()),
        "a1c_categories": frame["a1c_cat"].value_counts().sort_index().to_dict(),
        "fpg_categories": frame["fpg_cat"].value_counts().sort_index().to_dict(),
        "ogtt_categories": frame["ogtt_cat"].value_counts().sort_index().to_dict(),
        "canonical_max_categories": frame["canonical_max"].value_counts().sort_index().to_dict(),
        "canonical_median_categories": frame["canonical_median"].value_counts().sort_index().to_dict(),
        "canonical_mode_used": canonical_mode,
        "known_diabetes_self_report_count": int(np.sum(frame["DIQ010"] == 1)),
        "canonical_definition": ("maximum ADA-threshold severity across HbA1c, FPG, and 2-h OGTT" if canonical_mode == "max" else "median/majority ADA-threshold severity across HbA1c, FPG, and 2-h OGTT") + "; research endpoint, not a clinical diagnosis",
        "validation_design": "natural-definition / semi-synthetic-environment",
        "features": ["age", "sex", "race/ethnicity", "BMI", "waist circumference", "income-to-poverty ratio", "education"],
        "glycemic_laboratory_values_used_as_predictors": False,
        "exclude_self_reported_known_diabetes": bool(exclude_known_diabetes),
    }
    with open(output_dir / "nhanes_diabetes_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"agreement_audit": agreement, "folds": fold_audit}, fh, indent=2)

    # Pairwise definition confusion tables from the actual same-person labels.
    for a, b, ca, cb in [
        ("HbA1c", "FPG", "a1c_cat", "fpg_cat"),
        ("HbA1c", "OGTT", "a1c_cat", "ogtt_cat"),
        ("FPG", "OGTT", "fpg_cat", "ogtt_cat"),
    ]:
        table = pd.crosstab(frame[ca], frame[cb], rownames=[a], colnames=[b])
        table.to_csv(output_dir / f"agreement_{a.lower()}_{b.lower()}.csv")

    print(summary.to_string(index=False))
    print(json.dumps(agreement, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("publication_results_diabetes"))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--canonical-mode", choices=["max", "median"], default="max")
    parser.add_argument("--exclude-known-diabetes", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.seed, args.canonical_mode, args.exclude_known_diabetes)


if __name__ == "__main__":
    main()
