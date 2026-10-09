"""Independent structural and numeric validation for the MASTER package."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    STRATEGIES,
    feature_columns,
    find_duplicate_columns,
    load_master_csv,
    sha256_file,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> dict:
    manifest_path = ROOT / "artifacts" / "master_preprocessing_manifest.json"
    quality_path = ROOT / "results" / "master_quality_checks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    raw_path = ROOT / manifest["raw_file"]
    clean_path = ROOT / manifest["canonical_file"]
    raw = load_master_csv(raw_path)
    clean = load_master_csv(clean_path)
    require(raw.shape == clean.shape == (2931, 353), "Unexpected raw/clean shape")
    require(list(raw.columns) == list(clean.columns), "Raw/clean schemas differ")
    require(raw[ID_COLUMN].equals(clean[ID_COLUMN]), "Raw/clean ID order differs")
    require(raw[TARGET_COLUMN].equals(clean[TARGET_COLUMN]), "Raw/clean labels differ")
    require(
        int(raw[feature_columns(raw)].isna().sum().sum())
        == int(clean[feature_columns(clean)].isna().sum().sum()),
        "Raw/clean missing counts differ after canonical loading",
    )
    require(
        sha256_file(raw_path) == quality["raw"]["raw_sha256"],
        "Raw file checksum changed",
    )

    outer = pd.read_csv(ROOT / manifest["outer_split"]["source"], dtype={ID_COLUMN: "string"})
    require(len(outer) == len(raw), "Outer split is not exhaustive")
    require(outer[ID_COLUMN].nunique() == len(raw), "Outer split IDs are not unique")
    train_ids = set(outer.loc[outer["partition"] == "train", ID_COLUMN])
    validation_ids = set(outer.loc[outer["partition"] == "validation", ID_COLUMN])
    require(not train_ids.intersection(validation_ids), "Outer partitions overlap")
    require(train_ids.union(validation_ids) == set(raw[ID_COLUMN]), "Outer IDs mismatch")

    fold_path = ROOT / manifest["repeated_cv"]["assignment_file"]
    folds = pd.read_csv(fold_path, dtype={ID_COLUMN: "string"})
    expected_fold_rows = len(train_ids) * len(manifest["repeated_cv"]["seeds"])
    require(len(folds) == expected_fold_rows, "Repeated fold row count is wrong")
    for seed in manifest["repeated_cv"]["seeds"]:
        seed_frame = folds.loc[folds["repeat_seed"] == seed]
        require(len(seed_frame) == len(train_ids), f"Seed {seed} coverage is wrong")
        require(seed_frame[ID_COLUMN].nunique() == len(train_ids), f"Seed {seed} repeats IDs")
        require(set(seed_frame[ID_COLUMN]) == train_ids, f"Seed {seed} IDs mismatch")
        require(set(seed_frame["fold"]) == set(range(5)), f"Seed {seed} fold labels mismatch")

    scenario_results = {}
    for strategy in STRATEGIES:
        entry = manifest["materialized_outer_split_datasets"][strategy]
        train = pd.read_csv(ROOT / entry["train"], low_memory=False)
        validation = pd.read_csv(ROOT / entry["validation"], low_memory=False)
        require(len(train) == len(train_ids), f"{strategy} train row count is wrong")
        require(len(validation) == len(validation_ids), f"{strategy} validation row count is wrong")
        require(set(train[ID_COLUMN]) == train_ids, f"{strategy} train IDs mismatch")
        require(
            set(validation[ID_COLUMN]) == validation_ids,
            f"{strategy} validation IDs mismatch",
        )
        require(list(train.columns) == list(validation.columns), f"{strategy} schema differs")
        require(len(train.columns) - 2 == entry["features"], f"{strategy} feature count differs")
        require(train.columns.is_unique, f"{strategy} has duplicate feature names")
        matrix = pd.concat(
            [train.drop(columns=[ID_COLUMN, TARGET_COLUMN]), validation.drop(columns=[ID_COLUMN, TARGET_COLUMN])],
            ignore_index=True,
        ).to_numpy(dtype=float)
        require(np.isfinite(matrix).all(), f"{strategy} contains NaN/inf")
        features = feature_columns(train)
        _, duplicate_of, _ = find_duplicate_columns(train, features, respect_kind=False)
        duplicate_count = len(duplicate_of)
        if strategy != "M1_native_clean":
            require(duplicate_count == 0, f"{strategy} contains exact duplicate outputs")
        scenario_results[strategy] = {
            "rows": len(train) + len(validation),
            "features": entry["features"],
            "exact_duplicate_outputs": duplicate_count,
            "finite": True,
        }

    fold_audit = pd.read_csv(ROOT / "results" / "master_fold_safety_audit.csv")
    require(len(fold_audit) == 100, "Expected 100 fold-safe preprocessing audits")
    require(fold_audit["status"].eq("passed").all(), "A fold audit failed")
    require(fold_audit["schema_equal"].all(), "A fold schema differs")
    require(fold_audit["finite_values"].all(), "A fold output is non-finite")

    row_flags = pd.read_csv(ROOT / "artifacts" / "master_row_quality_flags.csv")
    require(len(row_flags) == 116, "Expected 116 all-feature-missing rows")
    require(row_flags[ID_COLUMN].nunique() == 116, "Quality flag IDs are not unique")

    result = {
        "status": "passed",
        "validated_at_package_version": manifest["package_version"],
        "raw_shape": list(raw.shape),
        "outer_split": {"train": len(train_ids), "validation": len(validation_ids)},
        "repeated_fold_rows": len(folds),
        "fold_safe_fits": len(fold_audit),
        "all_feature_missing_rows_flagged": len(row_flags),
        "scenarios": scenario_results,
    }
    output = ROOT / "results" / "FINAL_PACKAGE_VALIDATION.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
