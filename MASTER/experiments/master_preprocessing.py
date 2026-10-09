"""Leakage-safe preprocessing primitives for the MASTER panel.

This module deliberately trains no predictive model.  Every statistic that can
learn from data (median, rare-category list, missingness threshold, duplicate
selection and output schema) is learned in ``fit`` and is therefore suitable
for fitting separately inside each cross-validation training fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd


ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
RESERVED_COLUMNS = {ID_COLUMN, TARGET_COLUMN, "Panel"}
KNOWN_MISSING_TOKENS = ["./."]
MISSING_CATEGORY = "__MISSING__"
RARE_CATEGORY = "__RARE__"
MISSING_SENTINEL = "<MASTER_MISSING>"


@dataclass(frozen=True)
class StrategyConfig:
    remove_source_duplicates: bool = False
    remove_uninformative_values: bool = False
    add_deduplicated_missing_indicators: bool = False
    remove_high_missing: bool = False
    high_missing_threshold: float = 0.80
    rare_min_count: int | None = None
    remove_duplicate_outputs: bool = False


STRATEGIES: dict[str, StrategyConfig] = {
    "M1_native_clean": StrategyConfig(),
    "M2_compact": StrategyConfig(
        remove_source_duplicates=True,
        remove_uninformative_values=True,
        rare_min_count=10,
        remove_duplicate_outputs=True,
    ),
    "M3_missing_aware_compact": StrategyConfig(
        remove_source_duplicates=True,
        remove_uninformative_values=True,
        add_deduplicated_missing_indicators=True,
        rare_min_count=10,
        remove_duplicate_outputs=True,
    ),
    "M4_high_missing_ablation": StrategyConfig(
        remove_source_duplicates=True,
        remove_uninformative_values=True,
        add_deduplicated_missing_indicators=True,
        remove_high_missing=True,
        high_missing_threshold=0.80,
        rare_min_count=10,
        remove_duplicate_outputs=True,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in RESERVED_COLUMNS]


def feature_kind(column: str) -> str:
    if column.startswith(("AA_", "CAT_")):
        return "categorical"
    if column.startswith(("AL_", "EK_")):
        return "numeric"
    raise ValueError(f"Unrecognised MASTER feature family: {column}")


def canonicalize_master_frame(
    frame: pd.DataFrame,
    require_label: bool = True,
) -> pd.DataFrame:
    """Return a canonical copy without changing the input object."""
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(f"Duplicate column names are not allowed: {duplicated_columns}")

    result = frame.copy()
    required = {ID_COLUMN}
    if require_label:
        required.add(TARGET_COLUMN)
    missing_required = required.difference(result.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    result[ID_COLUMN] = result[ID_COLUMN].astype("string").str.strip()
    if result[ID_COLUMN].isna().any() or result[ID_COLUMN].eq("").any():
        raise ValueError("Variant_ID contains an empty value")
    if result[ID_COLUMN].duplicated().any():
        raise ValueError("Variant_ID must be unique")

    if TARGET_COLUMN in result.columns:
        numeric_target = pd.to_numeric(result[TARGET_COLUMN], errors="raise")
        if numeric_target.isna().any() or not numeric_target.isin([0, 1]).all():
            raise ValueError("Label must contain only 0 and 1")
        result[TARGET_COLUMN] = numeric_target.astype("int8")

    for column in feature_columns(result):
        kind = feature_kind(column)
        series = result[column]
        if kind == "categorical":
            text = series.astype("string").str.strip()
            text = text.replace({token: pd.NA for token in KNOWN_MISSING_TOKENS})
            text = text.replace({"": pd.NA})
            result[column] = text
        else:
            if not pd.api.types.is_numeric_dtype(series):
                text = series.astype("string").str.strip()
                text = text.replace({token: pd.NA for token in KNOWN_MISSING_TOKENS})
                text = text.replace({"": pd.NA})
                parsed = pd.to_numeric(text, errors="coerce")
                invalid = text.notna() & parsed.isna()
                if invalid.any():
                    examples = sorted(text.loc[invalid].dropna().unique().tolist())[:5]
                    raise ValueError(f"Non-numeric values in {column}: {examples}")
                result[column] = parsed.astype("float64")
            else:
                result[column] = pd.to_numeric(series, errors="coerce").astype("float64")
    return result


def load_master_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        low_memory=False,
        keep_default_na=True,
        na_values=KNOWN_MISSING_TOKENS,
    )
    return canonicalize_master_frame(raw, require_label=True)


def _normalised_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_SENTINEL)


def _series_digest(series: pd.Series) -> str:
    normalised = _normalised_series(series)
    hashed = pd.util.hash_pandas_object(normalised, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _same_series(left: pd.Series, right: pd.Series) -> bool:
    return _normalised_series(left).reset_index(drop=True).equals(
        _normalised_series(right).reset_index(drop=True)
    )


def find_duplicate_columns(
    frame: pd.DataFrame,
    columns: list[str],
    respect_kind: bool = True,
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """Return representatives, duplicate->representative and complete groups."""
    representatives: list[str] = []
    duplicate_of: dict[str, str] = {}
    buckets: dict[tuple[str, str], list[str]] = {}
    groups: dict[str, list[str]] = {}

    for column in columns:
        kind = feature_kind(column) if respect_kind else "output"
        key = (kind, _series_digest(frame[column]))
        matched: str | None = None
        for candidate in buckets.get(key, []):
            if _same_series(frame[column], frame[candidate]):
                matched = candidate
                break
        if matched is None:
            representatives.append(column)
            buckets.setdefault(key, []).append(column)
            groups[column] = [column]
        else:
            duplicate_of[column] = matched
            groups[matched].append(column)
    return representatives, duplicate_of, groups


def create_repeated_stratified_folds(
    frame: pd.DataFrame,
    seeds: list[int],
    n_splits: int,
) -> pd.DataFrame:
    """Create deterministic repeated stratified fold assignments.

    A true group split is intentionally not claimed because the supplied data
    contains no patient, family or sample-group identifier.
    """
    rows: list[dict[str, Any]] = []
    labels = frame[TARGET_COLUMN].to_numpy()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        assignment = np.full(len(frame), -1, dtype=int)
        for label in sorted(np.unique(labels)):
            indices = np.flatnonzero(labels == label)
            indices = rng.permutation(indices)
            assignment[indices] = np.arange(len(indices)) % n_splits
        if (assignment < 0).any():
            raise RuntimeError("Fold assignment failed")
        for position, fold in enumerate(assignment):
            rows.append(
                {
                    ID_COLUMN: frame.iloc[position][ID_COLUMN],
                    TARGET_COLUMN: int(labels[position]),
                    "repeat_seed": int(seed),
                    "fold": int(fold),
                }
            )
    return pd.DataFrame(rows)


class MasterPreprocessor:
    """Pandas-only, fold-fit MASTER preprocessor."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "MasterPreprocessor":
        data = canonicalize_master_frame(frame, require_label=True)
        candidates = feature_columns(data)
        self.input_columns_ = list(candidates)
        self.missing_rate_ = data[candidates].isna().mean().to_dict()

        self.high_missing_dropped_: list[str] = []
        if self.config.remove_high_missing:
            self.high_missing_dropped_ = [
                column
                for column in candidates
                if self.missing_rate_[column] >= self.config.high_missing_threshold
            ]
        candidates = [c for c in candidates if c not in self.high_missing_dropped_]

        self.source_duplicate_of_: dict[str, str] = {}
        self.source_duplicate_groups_: dict[str, list[str]] = {
            column: [column] for column in candidates
        }
        if self.config.remove_source_duplicates:
            candidates, self.source_duplicate_of_, self.source_duplicate_groups_ = (
                find_duplicate_columns(data, candidates, respect_kind=True)
            )
        self.source_representatives_ = list(candidates)

        numeric_candidates = [c for c in candidates if feature_kind(c) == "numeric"]
        categorical_candidates = [c for c in candidates if feature_kind(c) == "categorical"]

        self.uninformative_value_dropped_: list[str] = []
        if self.config.remove_uninformative_values:
            numeric_value_columns = [
                c for c in numeric_candidates if data[c].nunique(dropna=True) > 1
            ]
            categorical_value_columns = []
            for column in categorical_candidates:
                effective_states = data[column].nunique(dropna=True) + int(data[column].isna().any())
                if effective_states > 1:
                    categorical_value_columns.append(column)
            kept_values = set(numeric_value_columns + categorical_value_columns)
            self.uninformative_value_dropped_ = [
                c for c in candidates if c not in kept_values
            ]
        else:
            numeric_value_columns = numeric_candidates
            categorical_value_columns = categorical_candidates

        self.numeric_value_columns_ = numeric_value_columns
        self.categorical_value_columns_ = categorical_value_columns
        self.medians_ = (
            data[self.numeric_value_columns_].median().fillna(0.0).astype(float).to_dict()
        )

        self.allowed_categories_: dict[str, list[str]] = {}
        self.encoded_categories_: dict[str, list[str]] = {}
        for column in self.categorical_value_columns_:
            values = data[column].fillna(MISSING_CATEGORY).astype(str)
            counts = values.value_counts(dropna=False)
            if self.config.rare_min_count is None:
                allowed = set(counts.index.tolist())
            else:
                allowed = set(counts[counts >= self.config.rare_min_count].index.tolist())
                if MISSING_CATEGORY in counts.index:
                    allowed.add(MISSING_CATEGORY)
            mapped = values.where(values.isin(allowed), RARE_CATEGORY)
            categories = sorted(mapped.unique().tolist())
            self.allowed_categories_[column] = sorted(allowed)
            self.encoded_categories_[column] = categories

        self.missing_indicator_groups_: dict[str, list[str]] = {}
        self.indicator_representatives_: list[str] = []
        if self.config.add_deduplicated_missing_indicators:
            indicator_candidates = [
                c
                for c in numeric_candidates
                if 0 < int(data[c].isna().sum()) < len(data)
            ]
            indicator_frame = pd.DataFrame(
                {column: data[column].isna().astype("int8") for column in indicator_candidates},
                index=data.index,
            )
            if indicator_candidates:
                (
                    self.indicator_representatives_,
                    _,
                    self.missing_indicator_groups_,
                ) = find_duplicate_columns(
                    indicator_frame,
                    indicator_candidates,
                    respect_kind=True,
                )

        base = self._transform_base(data)
        self.output_constant_dropped_: list[str] = []
        self.output_duplicate_of_: dict[str, str] = {}
        self.output_keep_columns_ = list(base.columns)
        if self.config.remove_duplicate_outputs and len(base.columns):
            self.output_constant_dropped_ = [
                c for c in base.columns if base[c].nunique(dropna=False) <= 1
            ]
            variable = [c for c in base.columns if c not in self.output_constant_dropped_]
            if variable:
                representatives, duplicate_of, _ = find_duplicate_columns(
                    base,
                    variable,
                    respect_kind=False,
                )
                self.output_keep_columns_ = representatives
                self.output_duplicate_of_ = duplicate_of
            else:
                self.output_keep_columns_ = []

        self.feature_names_ = list(self.output_keep_columns_)
        if not self.feature_names_:
            raise ValueError("Preprocessing removed every feature")
        if len(self.feature_names_) != len(set(self.feature_names_)):
            raise ValueError("Preprocessing produced duplicate feature names")
        self.is_fitted_ = True
        return self

    def _transform_base(self, data: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, pd.Series | np.ndarray] = {}
        for column in self.numeric_value_columns_:
            numeric = pd.to_numeric(data[column], errors="coerce")
            output[column] = numeric.fillna(self.medians_[column]).astype(float)

        for representative in self.indicator_representatives_:
            output[f"MISS__{representative}"] = data[representative].isna().astype(float)

        for column in self.categorical_value_columns_:
            values = data[column].fillna(MISSING_CATEGORY).astype(str)
            allowed = set(self.allowed_categories_[column])
            mapped = values.where(values.isin(allowed), RARE_CATEGORY)
            for category in self.encoded_categories_[column]:
                name = f"OHE__{column}=={category}"
                output[name] = mapped.eq(category).astype(float)

        return pd.DataFrame(output, index=data.index, dtype=float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        data = canonicalize_master_frame(frame, require_label=False)
        missing_input = set(self.input_columns_).difference(data.columns)
        if missing_input:
            raise ValueError(f"Transform input lacks columns: {sorted(missing_input)}")
        unexpected_input = set(feature_columns(data)).difference(self.input_columns_)
        if unexpected_input:
            raise ValueError(
                f"Transform input contains unexpected feature columns: {sorted(unexpected_input)}"
            )
        base = self._transform_base(data)
        missing_output = set(self.output_keep_columns_).difference(base.columns)
        if missing_output:
            raise ValueError(f"Transform schema mismatch: {sorted(missing_output)}")
        result = base.loc[:, self.output_keep_columns_].copy()
        values = result.to_numpy(dtype=float, copy=False)
        if values.size and not np.isfinite(values).all():
            raise ValueError("Non-finite value produced by preprocessing")
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def audit_dict(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before requesting the audit")
        return {
            "config": asdict(self.config),
            "input_feature_count": len(self.input_columns_),
            "high_missing_dropped": self.high_missing_dropped_,
            "source_duplicate_of": self.source_duplicate_of_,
            "source_duplicate_groups": self.source_duplicate_groups_,
            "uninformative_value_dropped": self.uninformative_value_dropped_,
            "numeric_value_columns": self.numeric_value_columns_,
            "categorical_value_columns": self.categorical_value_columns_,
            "missing_indicator_groups": self.missing_indicator_groups_,
            "indicator_representatives": self.indicator_representatives_,
            "output_constant_dropped": self.output_constant_dropped_,
            "output_duplicate_of": self.output_duplicate_of_,
            "output_feature_count": len(self.feature_names_),
            "output_features": self.feature_names_,
        }
