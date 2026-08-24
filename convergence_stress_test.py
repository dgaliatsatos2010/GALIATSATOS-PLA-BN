"""Targeted convergence audit for GALIATSATOS/PLA-BN v0.5.0.

This is not a performance benchmark. It isolates the proposed estimator on the
same real sklearn datasets/environments used by real_data_validation.py, uses
fixed fold-wise discretization (3 bins), and reports optimization diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from plabn import GaliatsatosMethod
from real_data_validation import (
    DATASET_SPECS,
    FoldwiseDiscretizer,
    assign_environments,
    coarsen_rows,
    environment_lists,
    exact_row_groups,
    training_and_target_operators,
)


def run_dataset(key: str, n_init: int = 2, n_bins: int = 3, random_seed: int = 20260820) -> list[dict]:
    spec = DATASET_SPECS[key]
    bunch = spec.loader()
    x = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)
    groups = exact_row_groups(x)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    rows: list[dict] = []
    for fold, (train_idx, _test_idx) in enumerate(splitter.split(x, y, groups)):
        y_train = y[train_idx]
        x_train = x[train_idx]
        g_train = groups[train_idx]
        n_classes = int(y_train.max()) + 1
        operators, _ = training_and_target_operators(n_classes)
        env = assign_environments(y_train, g_train, len(operators), random_seed + 100 * fold)
        observed = coarsen_rows(y_train, env, operators)
        discretizer = FoldwiseDiscretizer(n_bins, spec.max_features)
        discrete = discretizer.fit_transform(x_train)
        x_env, y_env = environment_lists(discrete, observed, env, len(operators))
        model = GaliatsatosMethod(
            n_classes=n_classes,
            structure="tan",
            smoothing=0.25,
            max_iter=spec.em_max_iter,
            tol=1e-5,
            random_state=random_seed + fold,
            n_init=n_init,
            init_jitter=0.35,
        ).fit(x_env, y_env, operators)
        rows.append({
            "dataset": spec.display_name,
            "fold": fold,
            "n_train": len(train_idx),
            "n_init": n_init,
            "max_iter": spec.em_max_iter,
            "converged": model.converged_,
            "termination_reason": model.termination_reason_,
            "iterations": model.n_iter_,
            "best_start": model.best_start_,
            "final_log_likelihood": model.final_log_likelihood_,
            "relative_last_improvement": model.relative_last_improvement_,
            "structure_changes": model.structure_changes_,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("publication_results/convergence_stress_v050.csv"))
    parser.add_argument("--n-init", type=int, default=2)
    args = parser.parse_args()
    rows: list[dict] = []
    for key in DATASET_SPECS:
        print(f"Auditing {DATASET_SPECS[key].display_name} ...")
        rows.extend(run_dataset(key, n_init=args.n_init))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("dataset").agg(
        folds=("fold", "count"),
        converged_folds=("converged", "sum"),
        median_iterations=("iterations", "median"),
        max_relative_last_improvement=("relative_last_improvement", "max"),
    )
    print(summary.to_string())
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
