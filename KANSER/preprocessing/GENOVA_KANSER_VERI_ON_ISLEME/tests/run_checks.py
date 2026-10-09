"""Harici test koşucusu gerektirmeyen teslim kalite kontrolleri."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genova.kanser.preprocessing import KanserPreprocessor  # noqa: E402
from genova.kanser.schema import validate_schema  # noqa: E402


def check_data():
    raw = pd.read_csv(ROOT / "data/raw/YARISMA_TRAIN_KANSER.csv", low_memory=False)
    report = validate_schema(raw)
    assert report["ok"], report["issues"]
    meta = pd.read_csv(ROOT / "data/processed/kanser/00_KANSER_TARGET_AND_METADATA.csv")
    assert len(meta) == len(raw) == 388
    assert meta.Variant_ID.tolist() == raw.Variant_ID.tolist()
    assert meta.Label.tolist() == raw.Label.tolist()
    assert meta.row_index.tolist() == list(range(len(raw)))
    expected = {
        "01_KANSER_NATIVE_SAFE_FULL.csv": 351,
        "02_KANSER_COMPACT_NONREDUNDANT.csv": 280,
        "03_KANSER_ROBUST_COVERAGE85.csv": 262,
        "04_KANSER_SOURCE_RESISTANT_ENGINEERED_LEGACY.csv": 279,
        "A1_KANSER_SOURCE_COLUMNS_REMOVED.csv": 260,
        "A2_KANSER_MISSINGNESS_FEATURES_ONLY.csv": 276,
        "A3_KANSER_AA_FEATURES_ONLY.csv": 267,
        "A4_KANSER_SOURCE_RESISTANT_COMBINED.csv": 279,
    }
    for filename, n_features in expected.items():
        x = pd.read_csv(ROOT / "data/processed/kanser" / filename)
        assert x.shape == (388, n_features), (filename, x.shape)
        assert not {"Variant_ID", "Label", "row_index", "duplicate_group_id"} & set(x.columns)
        numeric = x.select_dtypes(include="number")
        assert not np.isinf(numeric.to_numpy()).any(), filename


def check_fold_fit():
    raw = pd.read_csv(ROOT / "data/raw/YARISMA_TRAIN_KANSER.csv", low_memory=False)
    train, validation = raw.iloc[:300].copy(), raw.iloc[300:].copy()
    for policy in ("native", "compact", "robust85", "source_removed", "missingness_only", "aa_only", "source_resistant_combined"):
        transformer = KanserPreprocessor(policy=policy).fit(train)
        x_train = transformer.transform(train)
        x_val = transformer.transform(validation)
        assert list(x_train.columns) == list(x_val.columns)
        assert len(x_train) == 300 and len(x_val) == 88
        assert not {"Variant_ID", "Label"} & set(x_train.columns)
    source_safe = KanserPreprocessor(policy="source_resistant_combined").fit(raw)
    assert "CAT_1" not in source_safe.learned_.base_features
    assert "CAT_2" not in source_safe.learned_.base_features
    assert "CAT_1" not in source_safe.learned_.missingness_source_features
    assert "CAT_2" not in source_safe.learned_.missingness_source_features


def check_splits():
    split_dir = ROOT / "data/splits/kanser"
    meta = pd.read_csv(ROOT / "data/processed/kanser/00_KANSER_TARGET_AND_METADATA.csv")
    group_lookup = meta.set_index("Variant_ID").duplicate_group_id.to_dict()
    outer_files = sorted(split_dir.glob("outer_fold_repeat*.json"))
    inner_files = sorted(split_dir.glob("inner_fold_repeat*.json"))
    assert len(outer_files) == 10
    assert len(inner_files) == 50
    all_ids = set(meta.Variant_ID)
    for path in outer_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seen = []
        for fold in payload["folds"]:
            train, test = set(fold["train_variant_ids"]), set(fold["test_variant_ids"])
            assert not train & test
            assert train | test == all_ids
            assert not ({group_lookup[x] for x in train} & {group_lookup[x] for x in test})
            block = meta[meta.Variant_ID.isin(test)]
            assert set(block.Label.unique()) == {0, 1}
            seen.extend(test)
        assert len(seen) == len(set(seen)) == len(all_ids)
        assert set(seen) == all_ids


def main():
    check_data()
    check_fold_fit()
    check_splits()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
