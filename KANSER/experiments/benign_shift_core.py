"""Reusable preprocessing and deployment helpers for the benign-prior model.

The public classes in this module deliberately avoid reading the target while
building features.  Keeping them outside the training script also makes the
final joblib artifact loadable from a fresh Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from kanser_v4_core import KanserV4Preprocessor


ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
SEMANTIC_MISSING_TOKEN = "./."
MISSING_CATEGORY = "__MISSING__"


def _features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {ID_COLUMN, TARGET_COLUMN}]


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _missing(series: pd.Series) -> pd.Series:
    result = series.isna()
    if _is_categorical(series):
        result = result | series.astype("string").eq(SEMANTIC_MISSING_TOKEN).fillna(False)
    return result.astype(bool)


def _normalise(series: pd.Series) -> pd.Series:
    if _is_categorical(series):
        values = series.astype("string")
        values = values.mask(values.eq(SEMANTIC_MISSING_TOKEN), pd.NA)
        return values.fillna(MISSING_CATEGORY).astype(str)
    return pd.to_numeric(series, errors="coerce")


def _series_key(series: pd.Series) -> bytes:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype("float64")
    else:
        values = _normalise(series)
    return pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64").tobytes()


def _mask_key(mask: pd.Series) -> bytes:
    return mask.to_numpy(dtype="uint8").tobytes()


class SourceAwareInfoPreprocessor:
    """A target-free A5-style transformer that intentionally retains CAT_1/2.

    This is the performance-oriented counterpart of the source-resistant A5
    policy.  Constant observed values are condensed to distinct missingness
    masks, exact duplicate values are removed, and non-redundant missingness
    summaries plus amino-acid interactions are added.
    """

    policy = "source_aware_info"

    def __init__(self):
        self.input_features_: list[str] | None = None
        self.base_features_: list[str] | None = None
        self.missing_representatives_: list[str] | None = None
        self.summary_sources_: list[str] | None = None
        self.output_features_: list[str] | None = None

    def _derived(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.missing_representatives_ is not None
        assert self.summary_sources_ is not None
        blocks: list[pd.DataFrame] = []
        if self.missing_representatives_:
            blocks.append(pd.DataFrame({
                f"DER_MISSREP_{c}": _missing(frame[c]).astype("int8")
                for c in self.missing_representatives_
            }, index=frame.index))

        if self.summary_sources_:
            miss = pd.DataFrame({
                c: _missing(frame[c]).astype("int8") for c in self.summary_sources_
            }, index=frame.index)
            summary = pd.DataFrame(index=frame.index)
            for prefix in ("AL_", "EK_", "CAT_", "AA_"):
                members = [c for c in self.summary_sources_ if c.startswith(prefix)]
                if members:
                    tag = prefix.rstrip("_")
                    summary[f"DER_MISS_{tag}_RATE"] = miss[members].mean(axis=1)
                    summary[f"DER_MISS_{tag}_ALL"] = miss[members].all(axis=1).astype("int8")
            summary["DER_MISS_TOTAL_RATE"] = miss.mean(axis=1)
            blocks.append(summary)

        if "AA_1" in frame and "AA_2" in frame:
            aa1 = _normalise(frame["AA_1"]).astype("string")
            aa2 = _normalise(frame["AA_2"]).astype("string")
            valid = aa1.ne(MISSING_CATEGORY) & aa2.ne(MISSING_CATEGORY)
            blocks.append(pd.DataFrame({
                "DER_AA_PAIR": (aa1 + ">" + aa2).where(valid, MISSING_CATEGORY).astype(str),
                "DER_AA_STOP_GAIN": (valid & aa2.eq("*") & aa1.ne("*")).astype("int8"),
                "DER_AA_STOP_LOSS": (valid & aa1.eq("*") & aa2.ne("*")).astype("int8"),
                "DER_AA_IS_SAME": (valid & aa1.eq(aa2)).astype("int8"),
                "DER_AA_NONSTANDARD": (
                    valid & ((aa1.str.len() != 1) | (aa2.str.len() != 1))
                ).astype("int8"),
            }, index=frame.index))
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=frame.index)

    def fit(self, frame: pd.DataFrame, y=None):
        del y
        inputs = _features(frame)
        fully_missing = [c for c in inputs if _missing(frame[c]).all()]
        working = [c for c in inputs if c not in fully_missing]

        constant_observed: list[str] = []
        missing_patterns: dict[bytes, list[str]] = {}
        for column in working:
            mask = _missing(frame[column])
            observed_unique = int(frame.loc[~mask, column].nunique(dropna=True))
            if observed_unique <= 1:
                constant_observed.append(column)
                if mask.any() and (~mask).any():
                    missing_patterns.setdefault(_mask_key(mask), []).append(column)
        representatives = [members[0] for members in missing_patterns.values()]

        nonconstant = [c for c in working if c not in constant_observed]
        seen: dict[bytes, str] = {}
        duplicate_drop: list[str] = []
        for column in nonconstant:
            key = _series_key(frame[column])
            if key in seen:
                duplicate_drop.append(column)
            else:
                seen[key] = column
        base = [c for c in nonconstant if c not in duplicate_drop]

        self.input_features_ = inputs
        self.base_features_ = base
        self.missing_representatives_ = representatives
        self.summary_sources_ = base + representatives

        clean = pd.DataFrame({c: _normalise(frame[c]) for c in base}, index=frame.index)
        complete = pd.concat([clean, self._derived(frame)], axis=1)
        keep: list[str] = []
        seen_complete: dict[bytes, str] = {}
        for column in complete.columns:
            if complete[column].nunique(dropna=False) <= 1:
                continue
            key = _series_key(complete[column])
            if key not in seen_complete:
                keep.append(column)
                seen_complete[key] = column
        self.output_features_ = keep
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.output_features_ is None or self.base_features_ is None:
            raise RuntimeError("fit must be called before transform")
        missing_columns = [c for c in self.input_features_ or [] if c not in frame]
        if missing_columns:
            raise ValueError(f"Missing raw input columns: {missing_columns[:10]}")
        clean = pd.DataFrame(
            {c: _normalise(frame[c]) for c in self.base_features_}, index=frame.index
        )
        complete = pd.concat([clean, self._derived(frame)], axis=1)
        return complete[self.output_features_].copy()

    def fit_transform(self, frame: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(frame, y=y).transform(frame)


class SourceFreeRobust85PlusPreprocessor:
    """Robust85 policy plus target-free amino-acid change descriptors.

    The base transformer still learns all removal/normalisation decisions on the
    training fold.  The extra descriptors use only AA_1 and AA_2 symbols, so they
    can be safely learned and applied inside CV without accessing ``Label``.
    """

    policy = "source_free_robust85_plus"

    _HYDRO = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    _VOLUME = {
        "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
        "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
        "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
        "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
    }
    _POLAR = {
        "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
        "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
        "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
        "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
    }
    _CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}

    def __init__(self, missing_threshold: float = 0.85):
        self.base = KanserV4Preprocessor(
            policy="source_free_robust85", missing_threshold=missing_threshold
        )
        self.output_features_: list[str] | None = None

    @staticmethod
    def _aa_values(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series("__MISSING__", index=frame.index, dtype="string")
        values = frame[column].astype("string")
        return values.mask(values.isna() | values.eq("./."), "__MISSING__")

    def _derived(self, frame: pd.DataFrame) -> pd.DataFrame:
        aa1 = self._aa_values(frame, "AA_1")
        aa2 = self._aa_values(frame, "AA_2")
        valid = aa1.str.len().eq(1) & aa2.str.len().eq(1)
        hyd1 = aa1.map(self._HYDRO).fillna(0.0).astype(float)
        hyd2 = aa2.map(self._HYDRO).fillna(0.0).astype(float)
        vol1 = aa1.map(self._VOLUME).fillna(0.0).astype(float)
        vol2 = aa2.map(self._VOLUME).fillna(0.0).astype(float)
        pol1 = aa1.map(self._POLAR).fillna(0.0).astype(float)
        pol2 = aa2.map(self._POLAR).fillna(0.0).astype(float)
        ch1 = aa1.map(self._CHARGE).fillna(0.0).astype(float)
        ch2 = aa2.map(self._CHARGE).fillna(0.0).astype(float)
        out = pd.DataFrame({
            "DER_AA_PAIR": (aa1 + ">" + aa2).where(valid, "__MISSING__").astype(str),
            "DER_AA_VALID": valid.astype("int8"),
            "DER_AA_HYDRO_DELTA": (hyd2 - hyd1).where(valid, 0.0),
            "DER_AA_VOLUME_DELTA": (vol2 - vol1).where(valid, 0.0),
            "DER_AA_POLAR_DELTA": (pol2 - pol1).where(valid, 0.0),
            "DER_AA_CHARGE_DELTA": (ch2 - ch1).where(valid, 0.0),
            "DER_AA_HYDRO_ABS_DELTA": (hyd2 - hyd1).abs().where(valid, 0.0),
            "DER_AA_VOLUME_ABS_DELTA": (vol2 - vol1).abs().where(valid, 0.0),
            "DER_AA_CHARGE_CHANGE": (ch1.ne(ch2) & valid).astype("int8"),
            "DER_AA_STOP_GAIN": (valid & aa2.eq("*") & aa1.ne("*")).astype("int8"),
            "DER_AA_STOP_LOSS": (valid & aa1.eq("*") & aa2.ne("*")).astype("int8"),
            "DER_AA_IS_SAME": (valid & aa1.eq(aa2)).astype("int8"),
        }, index=frame.index)
        return out

    @staticmethod
    def _deduplicate(frame: pd.DataFrame) -> list[str]:
        keep: list[str] = []
        seen: dict[bytes, str] = {}
        for column in frame.columns:
            if frame[column].nunique(dropna=False) <= 1:
                continue
            key = _series_key(frame[column])
            if key not in seen:
                keep.append(column)
                seen[key] = column
        return keep

    def fit(self, frame: pd.DataFrame, y=None):
        del y
        self.base.fit(frame)
        complete = pd.concat([self.base.transform(frame), self._derived(frame)], axis=1)
        self.output_features_ = self._deduplicate(complete)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.output_features_ is None:
            raise RuntimeError("fit must be called before transform")
        complete = pd.concat([self.base.transform(frame), self._derived(frame)], axis=1)
        return complete[self.output_features_].copy()

    def fit_transform(self, frame: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(frame, y=y).transform(frame)


def build_encoder(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numeric = [c for c in frame.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", add_indicator=False), numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class FittedComponent:
    name: str
    policy: str
    preprocessor: Any
    encoder: ColumnTransformer
    selector_indices: np.ndarray
    selected_feature_names: list[str]
    model: Any
    scaler: Any = None

    def predict_score(self, raw: pd.DataFrame) -> np.ndarray:
        processed = self.preprocessor.transform(raw)
        matrix = np.asarray(self.encoder.transform(processed), dtype=float)
        selected = matrix[:, self.selector_indices]
        if self.scaler is not None:
            selected = self.scaler.transform(selected)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(selected)[:, 1], dtype=float)
        return np.asarray(self.model.decision_function(selected), dtype=float)


@dataclass
class DeployableBenignShiftModel:
    """Serializable final model, including preprocessing and locked threshold."""

    components: list[FittedComponent]
    component_weights: list[float]
    combine_mode: str
    threshold: float
    target_benign: int = 3000
    target_pathogenic: int = 500
    positive_label: int = 1
    negative_label: int = 0
    rows_used: int = 388
    rows_dropped: int = 0

    def predict_score(self, raw: pd.DataFrame) -> np.ndarray:
        scores = [component.predict_score(raw) for component in self.components]
        matrix = np.column_stack(scores)
        weights = np.asarray(self.component_weights, dtype=float)
        weights = weights / weights.sum()
        if self.combine_mode == "soft":
            return matrix @ weights
        if self.combine_mode == "rank":
            ranked = np.column_stack([
                pd.Series(matrix[:, i]).rank(method="average", pct=True).to_numpy()
                for i in range(matrix.shape[1])
            ])
            return ranked @ weights
        raise ValueError(f"Unknown combine_mode={self.combine_mode!r}")

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_score(raw) >= self.threshold).astype(int)
