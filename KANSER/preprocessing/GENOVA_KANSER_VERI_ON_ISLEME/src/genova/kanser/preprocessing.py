"""KANSER paneli için fold-içi fit/transform ön işleme politikaları.

Statik CSV'ler inceleme ve deney iskeletidir. Resmî CV performansı için bu
sınıf her outer/inner training fold'unda yeniden fit edilmelidir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .schema import ID_COLUMN, TARGET_COLUMN, feature_columns, semantic_missing

SOURCE_COLUMNS = ("CAT_1", "CAT_2")

POLICIES = {
    "native": "Bütün özellikleri korur; yalnızca kategorik eksikliği standartlaştırır.",
    "compact": "Sabit ve training fold'unda birebir kopya sütunları çıkarır.",
    "robust85": "Compact + training fold'unda semantik eksikliği >=%85 sütunları çıkarır.",
    "legacy_combined": "Önceki birleşik dosya: robust85 - CAT_1/2 + tüm kaynaklardan missingness + AA.",
    "source_removed": "Robust85 tabanından yalnızca CAT_1 ve CAT_2 çıkarılır.",
    "missingness_only": "Robust85 tabanına yalnızca satır-temelli missingness özetleri eklenir.",
    "aa_only": "Robust85 tabanına yalnızca aminoasit etkileşimleri eklenir.",
    "source_resistant_combined": "Robust85 - CAT_1/2 + kaynak sütunlarını dışlayan missingness + AA.",
}


@dataclass
class LearnedPolicy:
    policy: str
    missing_threshold: float
    input_features: list[str]
    categorical_features: list[str]
    constant_columns: list[str]
    exact_duplicate_dropped: list[str]
    high_missing_columns: list[str]
    base_features: list[str]
    output_features: list[str]
    missingness_source_features: list[str]


class KanserPreprocessor:
    """DataFrame giriş/çıkışlı, hedef kullanmayan fold-safe transformer."""

    def __init__(self, policy: str = "compact", missing_threshold: float = 0.85):
        if policy not in POLICIES:
            raise ValueError(f"bilinmeyen policy={policy!r}; seçenekler={sorted(POLICIES)}")
        self.policy = policy
        self.missing_threshold = float(missing_threshold)
        self.learned_: LearnedPolicy | None = None

    @staticmethod
    def _normalize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        clean = df[columns].copy()
        for column in columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                values = df[column].astype("string").mask(
                    df[column].astype("string").eq("./."), pd.NA
                )
                clean[column] = values.fillna("__MISSING__")
        return clean

    @staticmethod
    def _exact_duplicate_drop(raw_features: pd.DataFrame) -> list[str]:
        """Training fold'unda ham değerleri ve NaN konumları birebir aynı sütunları bulur."""
        columns = raw_features.columns.tolist()
        dropped: set[str] = set()
        for i, reference in enumerate(columns):
            if reference in dropped:
                continue
            for candidate in columns[i + 1:]:
                if candidate not in dropped and raw_features[reference].equals(raw_features[candidate]):
                    dropped.add(candidate)
        return [c for c in columns if c in dropped]

    @staticmethod
    def _missing_features(raw_features: pd.DataFrame, allowed: list[str]) -> pd.DataFrame:
        miss = pd.DataFrame(
            {column: semantic_missing(raw_features[column]).astype("int8") for column in allowed},
            index=raw_features.index,
        )
        out = pd.DataFrame(index=raw_features.index)
        for prefix in ("AL_", "EK_", "CAT_", "AA_"):
            columns = [c for c in allowed if c.startswith(prefix)]
            tag = prefix.rstrip("_")
            if columns:
                out[f"DER_MISS_{tag}_COUNT"] = miss[columns].sum(axis=1).astype("int16")
                out[f"DER_MISS_{tag}_RATE"] = miss[columns].mean(axis=1)
                out[f"DER_MISS_{tag}_ALL"] = miss[columns].all(axis=1).astype("int8")
        out["DER_MISS_TOTAL_COUNT"] = miss.sum(axis=1).astype("int16")
        out["DER_MISS_TOTAL_RATE"] = miss.mean(axis=1)
        return out

    @staticmethod
    def _aa_features(clean: pd.DataFrame) -> pd.DataFrame:
        aa1 = clean["AA_1"].astype("string")
        aa2 = clean["AA_2"].astype("string")
        valid = aa1.ne("__MISSING__") & aa2.ne("__MISSING__")
        out = pd.DataFrame(index=clean.index)
        out["DER_AA_PAIR"] = (aa1 + ">" + aa2).where(valid, "__MISSING__")
        out["DER_AA_IS_SAME"] = (valid & aa1.eq(aa2)).astype("int8")
        out["DER_AA_STOP_GAIN"] = (valid & aa2.eq("*") & aa1.ne("*")).astype("int8")
        out["DER_AA_STOP_LOSS"] = (valid & aa1.eq("*") & aa2.ne("*")).astype("int8")
        out["DER_AA_NONSTANDARD"] = (
            valid & ((aa1.str.len() != 1) | (aa2.str.len() != 1))
        ).astype("int8")
        return out

    def fit(self, df: pd.DataFrame, y=None):
        del y  # Hedef hiçbir seçimde kullanılmaz.
        inputs = feature_columns(df)
        if not inputs:
            raise ValueError("özellik kolonu bulunamadı")
        categorical = [c for c in inputs if not pd.api.types.is_numeric_dtype(df[c])]
        clean = self._normalize_features(df, inputs)

        # Önceki teslimle aynı deterministik tanım: ham training fold'unda,
        # NaN hariç tek değerli ve ham Series.equals ile birebir aynı sütunlar.
        constant = [c for c in inputs if df[c].nunique(dropna=True) <= 1]
        duplicate_drop = self._exact_duplicate_drop(df[inputs])
        high_missing = [c for c in inputs if semantic_missing(df[c]).mean() >= self.missing_threshold]

        structural_drop: set[str] = set()
        if self.policy != "native":
            structural_drop.update(constant)
            structural_drop.update(duplicate_drop)
        if self.policy in {
            "robust85", "legacy_combined", "source_removed", "missingness_only",
            "aa_only", "source_resistant_combined",
        }:
            structural_drop.update(high_missing)
        if self.policy in {"legacy_combined", "source_removed", "source_resistant_combined"}:
            structural_drop.update(SOURCE_COLUMNS)

        base = [c for c in inputs if c not in structural_drop]
        missing_sources = inputs.copy()
        if self.policy == "source_resistant_combined":
            missing_sources = [c for c in inputs if c not in SOURCE_COLUMNS]

        derived: list[str] = []
        if self.policy in {"legacy_combined", "missingness_only", "source_resistant_combined"}:
            derived.extend(self._missing_features(df[inputs], missing_sources).columns.tolist())
        if self.policy in {"legacy_combined", "aa_only", "source_resistant_combined"}:
            derived.extend(self._aa_features(clean).columns.tolist())

        self.learned_ = LearnedPolicy(
            policy=self.policy,
            missing_threshold=self.missing_threshold,
            input_features=inputs,
            categorical_features=categorical,
            constant_columns=constant,
            exact_duplicate_dropped=duplicate_drop,
            high_missing_columns=high_missing,
            base_features=base,
            output_features=base + derived,
            missingness_source_features=missing_sources,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.learned_ is None:
            raise RuntimeError("transform öncesinde fit çağrılmalıdır")
        missing_input = [c for c in self.learned_.input_features if c not in df.columns]
        if missing_input:
            raise ValueError(f"girdide eksik özellikler: {missing_input[:10]}")
        clean = self._normalize_features(df, self.learned_.input_features)
        out = clean[self.learned_.base_features].copy()
        if self.policy in {"legacy_combined", "missingness_only", "source_resistant_combined"}:
            miss = self._missing_features(
                df[self.learned_.input_features], self.learned_.missingness_source_features
            )
            out = pd.concat([out, miss], axis=1)
        if self.policy in {"legacy_combined", "aa_only", "source_resistant_combined"}:
            out = pd.concat([out, self._aa_features(clean)], axis=1)
        return out[self.learned_.output_features]

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y=y).transform(df)

    def policy_dict(self) -> dict:
        if self.learned_ is None:
            raise RuntimeError("policy_dict öncesinde fit çağrılmalıdır")
        result = asdict(self.learned_)
        result["description"] = POLICIES[self.policy]
        result["leakage_note"] = (
            "Bu policy dosyası bütün etiketli veride üretilmiş referans artefaktıdır. "
            "CV performansında KanserPreprocessor her training fold'unda yeniden fit edilmelidir."
        )
        return result
