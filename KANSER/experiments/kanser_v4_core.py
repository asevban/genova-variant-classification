"""GENOVA KANSER V4: target-free, fold-safe preprocessing primitives.

The static CSV generated from the full training set is a reference/final-training
artifact. During cross-validation this transformer must be fitted again on each
training fold. No decision in this module reads ``Label``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
SOURCE_COLUMNS = ("CAT_1", "CAT_2")
SEMANTIC_MISSING_TOKEN = "./."
MISSING_CATEGORY = "__MISSING__"
FEATURE_PREFIXES = ("AL_", "EK_", "CAT_", "AA_")

POLICIES = {
    "source_free_compact": (
        "CAT_1/2 removed; redundant raw values removed; variable missingness in "
        "constant-observed columns retained as deduplicated binary indicators."
    ),
    "source_free_robust85": (
        "source_free_compact plus training-fold semantic missingness >=85% removal."
    ),
    "a4_corrected": (
        "Previous corrected A4: robust85, CAT_1/2 removed, source-excluding group "
        "missingness summaries and amino-acid interaction features."
    ),
    "a5_info_preserving": (
        "Recommended V4 policy: source-free compact base, high-missing raw values "
        "retained, deduplicated missingness representatives, non-redundant group "
        "missingness summaries and amino-acid interactions."
    ),
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {ID_COLUMN, TARGET_COLUMN}]


def is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def semantic_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if is_categorical(series):
        missing = missing | series.astype("string").eq(SEMANTIC_MISSING_TOKEN).fillna(False)
    return missing.astype(bool)


def normalized_series(series: pd.Series) -> pd.Series:
    if is_categorical(series):
        values = series.astype("string")
        values = values.mask(values.eq(SEMANTIC_MISSING_TOKEN), pd.NA)
        return values.fillna(MISSING_CATEGORY)
    return series.copy()


def normalized_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({c: normalized_series(df[c]) for c in columns}, index=df.index)


def _series_key(series: pd.Series) -> bytes:
    """Stable equality key after semantic missingness normalization."""
    if pd.api.types.is_numeric_dtype(series):
        # Numeric equality must not depend on int/float storage dtype.
        normalized = pd.to_numeric(series, errors="coerce").astype("float64")
    else:
        normalized = normalized_series(series)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashed.tobytes()


def _mask_key(mask: pd.Series) -> bytes:
    return mask.to_numpy(dtype="uint8").tobytes()


def _duplicate_groups(df: pd.DataFrame, columns: list[str]) -> list[list[str]]:
    buckets: dict[bytes, list[str]] = {}
    for column in columns:
        buckets.setdefault(_series_key(df[column]), []).append(column)
    return [members for members in buckets.values() if len(members) > 1]


def _constant_missing_groups(
    df: pd.DataFrame, columns: list[str]
) -> tuple[list[str], list[list[str]]]:
    """Find constant-observed columns and deduplicate their missingness masks."""
    constant_observed: list[str] = []
    patterns: dict[bytes, list[str]] = {}
    for column in columns:
        missing = semantic_missing(df[column])
        observed_unique = int(df.loc[~missing, column].nunique(dropna=True))
        if observed_unique <= 1:
            constant_observed.append(column)
            if missing.any() and (~missing).any():
                patterns.setdefault(_mask_key(missing), []).append(column)
    return constant_observed, list(patterns.values())


def _missingness_features(
    raw: pd.DataFrame,
    source_columns: list[str],
) -> pd.DataFrame:
    """Group summaries based on non-redundant source units only."""
    if not source_columns:
        return pd.DataFrame(index=raw.index)
    miss = pd.DataFrame(
        {c: semantic_missing(raw[c]).astype("int8") for c in source_columns},
        index=raw.index,
    )
    out = pd.DataFrame(index=raw.index)
    for prefix in FEATURE_PREFIXES:
        members = [c for c in source_columns if c.startswith(prefix)]
        if not members:
            continue
        tag = prefix.rstrip("_")
        out[f"DER_MISS_{tag}_RATE"] = miss[members].mean(axis=1)
        out[f"DER_MISS_{tag}_ALL"] = miss[members].all(axis=1).astype("int8")
    out["DER_MISS_TOTAL_RATE"] = miss.mean(axis=1)
    return out


def _aa_features(clean: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=clean.index)
    if "AA_1" not in clean or "AA_2" not in clean:
        return out
    aa1 = clean["AA_1"].astype("string")
    aa2 = clean["AA_2"].astype("string")
    valid = aa1.ne(MISSING_CATEGORY) & aa2.ne(MISSING_CATEGORY)
    out["DER_AA_PAIR"] = (aa1 + ">" + aa2).where(valid, MISSING_CATEGORY)
    out["DER_AA_STOP_GAIN"] = (valid & aa2.eq("*") & aa1.ne("*")).astype("int8")
    out["DER_AA_STOP_LOSS"] = (valid & aa1.eq("*") & aa2.ne("*")).astype("int8")
    out["DER_AA_IS_SAME"] = (valid & aa1.eq(aa2)).astype("int8")
    out["DER_AA_NONSTANDARD"] = (
        valid & ((aa1.str.len() != 1) | (aa2.str.len() != 1))
    ).astype("int8")
    return out


def _drop_constant_and_duplicate_derived(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    constants = [
        c for c in frame.columns
        if frame[c].nunique(dropna=False) <= 1
    ]
    kept = [c for c in frame.columns if c not in constants]
    duplicate_drop: list[str] = []
    seen: dict[bytes, str] = {}
    for column in kept:
        key = _series_key(frame[column])
        if key in seen:
            duplicate_drop.append(column)
        else:
            seen[key] = column
    final = frame[[c for c in kept if c not in duplicate_drop]].copy()
    return final, constants, duplicate_drop


@dataclass
class LearnedPolicy:
    policy: str
    description: str
    missing_threshold: float
    input_features: list[str]
    source_columns_removed: list[str]
    fully_missing_removed: list[str]
    constant_observed_removed: list[str]
    exact_duplicate_removed: list[str]
    high_missing_removed: list[str]
    base_features: list[str]
    constant_missingness_groups: list[list[str]]
    missingness_representatives: list[str]
    missingness_summary_sources: list[str]
    derived_constant_removed: list[str]
    derived_duplicate_removed: list[str]
    output_features: list[str]


class KanserV4Preprocessor:
    """Target-free DataFrame transformer used both statically and inside CV."""

    def __init__(self, policy: str = "a5_info_preserving", missing_threshold: float = 0.85):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy={policy!r}; expected one of {sorted(POLICIES)}")
        self.policy = policy
        self.missing_threshold = float(missing_threshold)
        self.learned_: LearnedPolicy | None = None

    def fit(self, df: pd.DataFrame, y=None):
        del y
        inputs = feature_columns(df)
        if not inputs:
            raise ValueError("No feature columns found")
        source_removed = [c for c in SOURCE_COLUMNS if c in inputs]
        allowed = [c for c in inputs if c not in source_removed]

        fully_missing = [c for c in allowed if semantic_missing(df[c]).all()]
        working = [c for c in allowed if c not in fully_missing]
        constant_observed, constant_groups = _constant_missing_groups(df, working)

        # Exact duplicate comparison is performed on non-constant columns; constant
        # columns are represented by deduplicated missingness indicators below.
        nonconstant = [c for c in working if c not in constant_observed]
        duplicate_groups = _duplicate_groups(df, nonconstant)
        exact_duplicate_removed = [c for group in duplicate_groups for c in group[1:]]
        compact = [c for c in nonconstant if c not in exact_duplicate_removed]

        high_missing_candidates = [
            c for c in compact if float(semantic_missing(df[c]).mean()) >= self.missing_threshold
        ]
        if self.policy in {"source_free_robust85", "a4_corrected"}:
            high_missing_removed = high_missing_candidates
        else:
            high_missing_removed = []
        base = [c for c in compact if c not in high_missing_removed]

        # A4 reproduces the corrected previous logic and therefore does not rescue
        # constant-observed missingness patterns. Other V4 policies do.
        use_missing_reps = self.policy in {
            "source_free_compact", "source_free_robust85", "a5_info_preserving"
        }
        representative_groups = constant_groups if use_missing_reps else []
        representatives = [group[0] for group in representative_groups]

        add_summaries = self.policy in {"a4_corrected", "a5_info_preserving"}
        add_aa = self.policy in {"a4_corrected", "a5_info_preserving"}
        if self.policy == "a4_corrected":
            # Mirrors the corrected legacy A4 for a fair comparator.
            summary_sources = [c for c in inputs if c not in SOURCE_COLUMNS]
        else:
            # Avoid counting exact duplicate columns dozens of times. Preserve one
            # source per variable missingness pattern that replaced raw constants.
            summary_sources = base + representatives

        # Fit-time derived schema is determined on training rows only.
        clean = normalized_frame(df, inputs)
        derived_blocks: list[pd.DataFrame] = []
        if use_missing_reps:
            derived_blocks.append(pd.DataFrame(
                {
                    f"DER_MISSREP_{rep}": semantic_missing(df[rep]).astype("int8")
                    for rep in representatives
                },
                index=df.index,
            ))
        if add_summaries:
            derived_blocks.append(_missingness_features(df, summary_sources))
        if add_aa:
            derived_blocks.append(_aa_features(clean))
        derived = pd.concat(derived_blocks, axis=1) if derived_blocks else pd.DataFrame(index=df.index)
        # Recheck the complete output, not only derived-with-derived equality. Base
        # features are ordered first, so a derived feature duplicating a base value
        # is removed while the original measurement is preserved.
        complete_fit_frame = pd.concat([clean[base], derived], axis=1)
        complete_fit_frame, complete_constants, complete_duplicates = _drop_constant_and_duplicate_derived(
            complete_fit_frame
        )
        derived_constants = [c for c in complete_constants if c not in base]
        derived_duplicates = [c for c in complete_duplicates if c not in base]

        self.learned_ = LearnedPolicy(
            policy=self.policy,
            description=POLICIES[self.policy],
            missing_threshold=self.missing_threshold,
            input_features=inputs,
            source_columns_removed=source_removed,
            fully_missing_removed=fully_missing,
            constant_observed_removed=constant_observed,
            exact_duplicate_removed=exact_duplicate_removed,
            high_missing_removed=high_missing_removed,
            base_features=base,
            constant_missingness_groups=representative_groups,
            missingness_representatives=representatives,
            missingness_summary_sources=summary_sources,
            derived_constant_removed=derived_constants,
            derived_duplicate_removed=derived_duplicates,
            output_features=complete_fit_frame.columns.tolist(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.learned_ is None:
            raise RuntimeError("fit must be called before transform")
        missing = [c for c in self.learned_.input_features if c not in df.columns]
        if missing:
            raise ValueError(f"Input schema is missing columns: {missing[:10]}")
        clean = normalized_frame(df, self.learned_.input_features)
        out = clean[self.learned_.base_features].copy()
        blocks = [out]
        if self.learned_.missingness_representatives:
            blocks.append(pd.DataFrame(
                {
                    f"DER_MISSREP_{rep}": semantic_missing(df[rep]).astype("int8")
                    for rep in self.learned_.missingness_representatives
                },
                index=df.index,
            ))
        if self.policy in {"a4_corrected", "a5_info_preserving"}:
            blocks.append(_missingness_features(df, self.learned_.missingness_summary_sources))
            blocks.append(_aa_features(clean))
        result = pd.concat(blocks, axis=1)
        missing_output = [c for c in self.learned_.output_features if c not in result.columns]
        if missing_output:
            raise AssertionError(f"Transform failed to create: {missing_output}")
        return result[self.learned_.output_features].copy()

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y=y).transform(df)

    def policy_dict(self) -> dict:
        if self.learned_ is None:
            raise RuntimeError("fit must be called before policy_dict")
        result = asdict(self.learned_)
        result["target_used"] = False
        result["fold_safe_note"] = (
            "Static full-training CSV is a reference/final-training artifact. "
            "For CV or validation this policy must be fitted on the training fold only."
        )
        return result


def validate_raw_schema(df: pd.DataFrame) -> dict:
    issues: list[str] = []
    if ID_COLUMN not in df:
        issues.append(f"Missing required ID column: {ID_COLUMN}")
    if TARGET_COLUMN not in df:
        issues.append(f"Missing required target column: {TARGET_COLUMN}")
    if ID_COLUMN in df:
        if df[ID_COLUMN].isna().any():
            issues.append("Variant_ID contains missing values")
        if not df[ID_COLUMN].is_unique:
            issues.append("Variant_ID is not unique")
    if TARGET_COLUMN in df:
        labels = set(df[TARGET_COLUMN].dropna().unique())
        if labels != {0, 1}:
            issues.append(f"Label values are {sorted(labels)}, expected [0, 1]")
    features = feature_columns(df)
    numeric = df[features].select_dtypes(include="number")
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float)).any():
        issues.append("Numeric features contain inf/-inf")
    groups = {prefix.rstrip("_"): sum(c.startswith(prefix) for c in features) for prefix in FEATURE_PREFIXES}
    expected = {"AL": 334, "EK": 9, "CAT": 6, "AA": 2}
    if groups != expected:
        issues.append(f"Feature-group counts {groups}, expected {expected}")
    return {
        "ok": not issues,
        "issues": issues,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features": len(features),
        "group_counts": groups,
        "label_counts": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()
        } if TARGET_COLUMN in df else {},
    }
