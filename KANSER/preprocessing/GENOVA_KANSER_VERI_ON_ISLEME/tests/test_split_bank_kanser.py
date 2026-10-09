from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genova.kanser.split_bank import group_stratified_folds, validate_folds


def _toy_data():
    return pd.DataFrame({
        "Variant_ID": [f"v{i}" for i in range(40)],
        "Label": [0] * 12 + [1] * 28,
        "duplicate_group_id": ["dup"] * 3 + [f"g{i}" for i in range(3, 40)],
    })


def test_folds_partition_all_rows_and_preserve_groups():
    df = _toy_data()
    folds = group_stratified_folds(df, n_splits=4, seed=42)
    report = validate_folds(df, folds, n_splits=4)
    assert report["ok"], report["violations"]


def test_split_is_deterministic_for_same_seed():
    df = _toy_data()
    assert group_stratified_folds(df, 4, 42) == group_stratified_folds(df, 4, 42)
