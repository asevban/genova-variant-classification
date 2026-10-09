from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genova.kanser.preprocessing import KanserPreprocessor
from genova.kanser.schema import semantic_missing, validate_schema


def test_semantic_missing_handles_dot_slash_dot():
    s = pd.Series(["A", "./.", None])
    assert semantic_missing(s).tolist() == [False, True, True]


def test_schema_rejects_duplicate_id():
    df = pd.DataFrame({"Variant_ID": ["a", "a"], "Label": [0, 1]})
    assert not validate_schema(df)["ok"]


def test_training_fold_controls_high_missing_drop():
    train = pd.DataFrame({
        "Variant_ID": [f"v{i}" for i in range(10)],
        "Label": [0, 1] * 5,
        "AL_1": [np.nan] * 9 + [1.0],
        "AA_1": ["A"] * 10,
        "AA_2": ["V"] * 10,
    })
    validation = train.iloc[:2].copy()
    validation["AL_1"] = [1.0, 2.0]
    prep = KanserPreprocessor("robust85").fit(train)
    assert "AL_1" in prep.learned_.high_missing_columns
    assert "AL_1" not in prep.transform(validation).columns


def test_source_resistant_excludes_source_from_values_and_missingness():
    raw = pd.read_csv(ROOT / "data/raw/YARISMA_TRAIN_KANSER.csv", low_memory=False)
    prep = KanserPreprocessor("source_resistant_combined").fit(raw)
    out = prep.transform(raw)
    assert "CAT_1" not in out.columns and "CAT_2" not in out.columns
    assert "CAT_1" not in prep.learned_.missingness_source_features
    assert "CAT_2" not in prep.learned_.missingness_source_features


def test_transform_columns_align_between_train_and_validation():
    raw = pd.read_csv(ROOT / "data/raw/YARISMA_TRAIN_KANSER.csv", low_memory=False)
    train, validation = raw.iloc[:300], raw.iloc[300:]
    prep = KanserPreprocessor("missingness_only").fit(train)
    assert list(prep.transform(train).columns) == list(prep.transform(validation).columns)
