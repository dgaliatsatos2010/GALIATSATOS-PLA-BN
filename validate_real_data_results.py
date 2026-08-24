"""Independent integrity checks for the saved nested real-data results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_validation import binary_target_metrics, multiclass_metrics


def probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))


def validate(results_dir: Path) -> list[str]:
    checks: list[str] = []
    canonical = pd.read_csv(results_dir / "real_data_canonical_oof_predictions.csv")
    target = pd.read_csv(
        results_dir / "real_data_held_out_definition_oof_predictions.csv"
    )
    canonical_summary = pd.read_csv(results_dir / "real_data_canonical_summary.csv")
    target_summary = pd.read_csv(
        results_dir / "real_data_held_out_definition_summary.csv"
    )
    tuning = pd.read_csv(results_dir / "real_data_inner_tuning.csv")
    quality = pd.read_csv(results_dir / "real_data_quality_audit.csv")
    manifest = json.loads(
        (results_dir / "real_data_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    key = ["dataset", "method", "sample_index"]
    assert not canonical.duplicated(key).any()
    assert not target.duplicated(key).any()
    assert len(canonical) == len(target)
    checks.append("one canonical and target OOF prediction per dataset/method/row")

    for (dataset, method), group in canonical.groupby(["dataset", "method"]):
        p_columns = [
            column
            for column in probability_columns(group, "p_class_")
            if group[column].notna().all()
        ]
        probabilities = group[p_columns].to_numpy(float)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        recomputed = multiclass_metrics(group["true_label"].to_numpy(), probabilities)
        reported = canonical_summary[
            (canonical_summary["dataset"] == dataset)
            & (canonical_summary["method"] == method)
        ].set_index("metric")
        for metric, value in recomputed.items():
            assert np.isclose(value, reported.loc[metric, "estimate"], atol=1e-12)
    checks.append("canonical metrics independently recomputed from OOF probabilities")

    for (dataset, method), group in target.groupby(["dataset", "method"]):
        probabilities = group[["p_target_0", "p_target_1"]].to_numpy(float)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        recomputed = binary_target_metrics(
            group["true_target_label"].to_numpy(), probabilities
        )
        reported = target_summary[
            (target_summary["dataset"] == dataset)
            & (target_summary["method"] == method)
        ].set_index("metric")
        for metric, value in recomputed.items():
            assert np.isclose(value, reported.loc[metric, "estimate"], atol=1e-12)
    checks.append("held-out-definition metrics independently recomputed")

    assert not tuning["canonical_labels_used_for_tuning"].astype(bool).any()
    assert not tuning["test_rows_used_for_tuning"].astype(bool).any()
    checks.append("inner tuning trace excludes canonical labels and outer-test rows")

    assert quality["all_rows_predicted_once"].astype(bool).all()
    assert quality["operator_stack_full_column_rank"].astype(bool).all()
    expected_folds = int(manifest["config"]["outer_splits"]) * len(
        manifest["datasets"]
    )
    assert len(manifest["fold_audits"]) == expected_folds
    assert all(
        fold["index_overlap_count"] == 0
        and fold["duplicate_groups_split_across_train_test"] == 0
        and not fold["outer_test_labels_passed_to_fit"]
        for fold in manifest["fold_audits"]
    )
    checks.append(
        f"{expected_folds} outer folds pass overlap, duplicate-group, and label-leakage gates"
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("publication_results")
    )
    args = parser.parse_args()
    checks = validate(args.results_dir)
    for check in checks:
        print(f"PASS: {check}")
    print(f"All {len(checks)} saved-result integrity checks passed.")


if __name__ == "__main__":
    main()
