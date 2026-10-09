"""KANSER ham veri şeması ve eksiklik doğrulama araçları."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
EXPECTED_GROUP_COUNTS = {"AL": 334, "CAT": 6, "EK": 9, "AA": 2}
SENTINEL_VALUES = (-999, -9999, 999, 9999)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COLUMN, TARGET_COLUMN}]


def feature_group(column: str) -> str:
    match = re.match(r"([A-Za-z]+)_", column)
    return match.group(1).upper() if match else "OTHER"


def categorical_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> list[str]:
    columns = list(columns) if columns is not None else feature_columns(df)
    return [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]


def semantic_missing(series: pd.Series) -> pd.Series:
    """NaN ve kategorik './.' gösterimini tek eksiklik tanımında birleştirir."""
    missing = series.isna()
    if not pd.api.types.is_numeric_dtype(series):
        missing = missing | series.astype("string").eq("./.").fillna(False)
    return missing


def validate_schema(df: pd.DataFrame, expected_group_counts: dict[str, int] | None = None) -> dict:
    expected_group_counts = expected_group_counts or EXPECTED_GROUP_COUNTS
    issues: list[str] = []
    for required in (ID_COLUMN, TARGET_COLUMN):
        if required not in df.columns:
            issues.append(f"eksik zorunlu kolon: {required}")
    if ID_COLUMN in df.columns:
        if df[ID_COLUMN].isna().any():
            issues.append("Variant_ID null içeriyor")
        if not df[ID_COLUMN].is_unique:
            issues.append("Variant_ID benzersiz değil")
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].isna().any():
            issues.append("Label null içeriyor")
        bad = set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
        if bad:
            issues.append(f"Label {{0,1}} dışında değer içeriyor: {sorted(bad)}")

    counts: dict[str, int] = {}
    for column in feature_columns(df):
        group = feature_group(column)
        counts[group] = counts.get(group, 0) + 1
    for group, expected in expected_group_counts.items():
        if counts.get(group, 0) != expected:
            issues.append(f"{group}_ grubu {counts.get(group, 0)} kolon; beklenen {expected}")

    numeric = df[feature_columns(df)].select_dtypes(include="number")
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float)).any():
        issues.append("sayısal özelliklerde inf/-inf bulundu")
    return {
        "ok": not issues,
        "issues": issues,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "group_counts": counts,
    }


def scan_sentinels(df: pd.DataFrame, sentinels=SENTINEL_VALUES) -> dict[str, dict[str, int]]:
    hits: dict[str, dict[str, int]] = {}
    for column in df.select_dtypes(include="number").columns:
        column_hits = {str(value): int((df[column] == value).sum()) for value in sentinels}
        column_hits = {value: count for value, count in column_hits.items() if count}
        if column_hits:
            hits[column] = column_hits
    return hits
