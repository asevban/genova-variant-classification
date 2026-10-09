"""KANSER EDA kanıt tablolarını ve hafif SVG grafiklerini üretir."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import KanserPreprocessor
from .schema import (
    categorical_columns,
    feature_columns,
    feature_group,
    scan_sentinels,
    semantic_missing,
    validate_schema,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "raw" / "YARISMA_TRAIN_KANSER.csv"
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"


def rank_auc(y: pd.Series, score: pd.Series) -> float | None:
    block = pd.DataFrame({"y": y, "score": score}).dropna()
    n1, n0 = int((block.y == 1).sum()), int((block.y == 0).sum())
    if n1 == 0 or n0 == 0 or block.score.nunique() < 2:
        return None
    ranks = block.score.rank(method="average")
    auc = (ranks[block.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def _write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], value_format="{:.1f}", color="#2563EB"):
    width = 960
    row_h = 34
    left = 260
    right = 100
    top = 70
    height = top + row_h * len(labels) + 35
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-12)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#17365D">{html.escape(title)}</text>',
    ]
    bar_max = width - left - right
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        bar_w = max(1, bar_max * value / max_value)
        pieces.append(f'<text x="{left-12}" y="{y+20}" text-anchor="end" font-family="Arial" font-size="14" fill="#263238">{html.escape(str(label))}</text>')
        pieces.append(f'<rect x="{left}" y="{y+4}" width="{bar_w:.1f}" height="22" rx="3" fill="{color}"/>')
        pieces.append(f'<text x="{left+bar_w+8:.1f}" y="{y+20}" font-family="Arial" font-size="13" fill="#263238">{html.escape(value_format.format(value))}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces), encoding="utf-8")


def build_all(raw: pd.DataFrame | None = None) -> dict:
    raw = pd.read_csv(RAW_PATH, low_memory=False) if raw is None else raw.copy()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    features = feature_columns(raw)
    cats = categorical_columns(raw, features)
    nums = [c for c in features if c not in cats]

    schema = validate_schema(raw)
    schema["categorical_features"] = len(cats)
    schema["numeric_features"] = len(nums)
    schema["label_counts"] = {str(k): int(v) for k, v in raw.Label.value_counts().sort_index().items()}
    schema["sentinel_hits"] = scan_sentinels(raw)
    (TABLE_DIR / "schema_validation.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    clean = KanserPreprocessor._normalize_features(raw, features)
    duplicate_drop = set(KanserPreprocessor._exact_duplicate_drop(raw[features]))
    constant = {c for c in features if raw[c].nunique(dropna=True) <= 1}

    missing_matrix = pd.DataFrame({c: semantic_missing(raw[c]).astype("int8") for c in features})
    missing_rates = missing_matrix.mean()
    audit = []
    for column in features:
        audit.append({
            "column": column,
            "group": feature_group(column),
            "dtype": str(raw[column].dtype),
            "semantic_missing_n": int(missing_matrix[column].sum()),
            "semantic_missing_rate": float(missing_rates[column]),
            "nonmissing_unique": int(raw[column].nunique(dropna=True)),
            "dominant_value_rate": float(clean[column].value_counts(dropna=False, normalize=True).iloc[0]),
            "constant_nonmissing": column in constant,
            "exact_duplicate_dropped": column in duplicate_drop,
            "high_missing_ge_85": bool(missing_rates[column] >= 0.85),
            "source_sensitive_candidate": column in {"CAT_1", "CAT_2"},
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(TABLE_DIR / "column_audit.csv", index=False, encoding="utf-8-sig")
    audit_df[audit_df.dominant_value_rate >= 0.95].to_csv(TABLE_DIR / "near_constant_scan.csv", index=False, encoding="utf-8-sig")

    row_missing = pd.DataFrame({
        "row_index": range(len(raw)),
        "Variant_ID": raw.Variant_ID,
        "Label": raw.Label,
        "missing_count": missing_matrix.sum(axis=1),
        "missing_rate": missing_matrix.mean(axis=1),
    })
    for prefix in ("AL_", "EK_", "CAT_", "AA_"):
        cols = [c for c in features if c.startswith(prefix)]
        row_missing[f"{prefix.rstrip('_')}_missing_rate"] = missing_matrix[cols].mean(axis=1)
    row_missing.to_csv(TABLE_DIR / "row_missingness.csv", index=False, encoding="utf-8-sig")

    missing_by_label = row_missing.groupby("Label").agg(
        n=("Variant_ID", "size"),
        mean_missing_rate=("missing_rate", "mean"),
        std_missing_rate=("missing_rate", "std"),
        min_missing_rate=("missing_rate", "min"),
        max_missing_rate=("missing_rate", "max"),
    ).reset_index()
    missing_by_label.to_csv(TABLE_DIR / "missingness_by_label.csv", index=False, encoding="utf-8-sig")

    source_rows = []
    for column in ("CAT_1", "CAT_2"):
        miss = semantic_missing(raw[column])
        for label in (0, 1):
            mask = raw.Label.eq(label)
            source_rows.append({
                "column": column,
                "Label": label,
                "n": int(mask.sum()),
                "missing_n": int((miss & mask).sum()),
                "missing_rate": float(miss[mask].mean()),
            })
    pd.DataFrame(source_rows).to_csv(TABLE_DIR / "source_missingness_by_label.csv", index=False, encoding="utf-8-sig")

    cat_rows = []
    for column in cats:
        for level, count in clean[column].value_counts(dropna=False).items():
            mask = clean[column].eq(level)
            cat_rows.append({
                "column": column,
                "level": str(level),
                "count": int(count),
                "rate": float(count / len(raw)),
                "label1_rate": float(raw.loc[mask, "Label"].mean()),
                "rare_lt5": bool(count < 5),
            })
    pd.DataFrame(cat_rows).to_csv(TABLE_DIR / "categorical_level_audit.csv", index=False, encoding="utf-8-sig")

    duplicate_rows = []
    for reference in features:
        matches = [c for c in features if c != reference and clean[reference].equals(clean[c])]
        if matches and not any(reference in row["columns"].split(";") for row in duplicate_rows):
            group = [reference] + matches
            duplicate_rows.append({"reference": reference, "n_columns": len(group), "columns": ";".join(group)})
    pd.DataFrame(duplicate_rows).to_csv(TABLE_DIR / "exact_duplicate_column_groups.csv", index=False, encoding="utf-8-sig")

    # Numerik tek-değişkenli ayırım taraması; yalnızca EDA, özellik seçimi değildir.
    auc_rows = []
    for column in nums:
        auc = rank_auc(raw.Label, raw[column])
        if auc is not None:
            auc_rows.append({
                "column": column,
                "n_nonmissing": int(raw[column].notna().sum()),
                "auc": auc,
                "auc_direction_free": max(auc, 1 - auc),
            })
    auc_df = pd.DataFrame(auc_rows).sort_values("auc_direction_free", ascending=False)
    auc_df.to_csv(TABLE_DIR / "univariate_numeric_auc_scan.csv", index=False, encoding="utf-8-sig")

    outlier_rows = []
    for column in nums:
        s = raw[column].dropna().astype(float)
        if len(s) < 10 or s.nunique() < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_rows.append({
            "column": column,
            "n_nonmissing": int(len(s)),
            "iqr_low": float(low),
            "iqr_high": float(high),
            "outlier_n": int(((s < low) | (s > high)).sum()),
            "outlier_rate": float(((s < low) | (s > high)).mean()),
        })
    pd.DataFrame(outlier_rows).sort_values("outlier_rate", ascending=False).to_csv(
        TABLE_DIR / "outlier_iqr_scan.csv", index=False, encoding="utf-8-sig"
    )

    provenance = {
        "row_missing_rate_auc": rank_auc(raw.Label, row_missing.missing_rate),
        "mean_missing_rate_label0": float(row_missing.loc[row_missing.Label == 0, "missing_rate"].mean()),
        "mean_missing_rate_label1": float(row_missing.loc[row_missing.Label == 1, "missing_rate"].mean()),
        "cat1_missing_rate_label0": float(semantic_missing(raw.CAT_1)[raw.Label == 0].mean()),
        "cat1_missing_rate_label1": float(semantic_missing(raw.CAT_1)[raw.Label == 1].mean()),
        "cat2_missing_rate_label0": float(semantic_missing(raw.CAT_2)[raw.Label == 0].mean()),
        "cat2_missing_rate_label1": float(semantic_missing(raw.CAT_2)[raw.Label == 1].mean()),
        "interpretation": "Missingness etiketi güçlü biçimde ayırıyor; katkı ve kaynak kestirmesi ayrı ablation ile test edilmelidir.",
    }
    (TABLE_DIR / "missingness_provenance_summary.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    group_counts = pd.Series({g: sum(c.startswith(g + "_") for c in features) for g in ("AL", "CAT", "EK", "AA")})
    group_missing = pd.Series({g: float(missing_matrix[[c for c in features if c.startswith(g + "_")]].to_numpy().mean()) for g in group_counts.index})
    _write_bar_svg(FIGURE_DIR / "class_distribution.svg", "KANSER sınıf dağılımı", ["Benign (0)", "Patojenik (1)"], [120, 268], "{:.0f}")
    _write_bar_svg(FIGURE_DIR / "feature_group_counts.svg", "Özellik grubu sütun sayıları", group_counts.index.tolist(), group_counts.astype(float).tolist(), "{:.0f}", "#0F766E")
    _write_bar_svg(FIGURE_DIR / "missing_rate_by_group.svg", "Grup bazında anlamsal eksiklik", group_missing.index.tolist(), (group_missing * 100).tolist(), "%{:.1f}", "#D97706")
    _write_bar_svg(FIGURE_DIR / "missing_rate_by_label.svg", "Sınıfa göre satır başına ortalama eksiklik", ["Benign (0)", "Patojenik (1)"], (missing_by_label.mean_missing_rate * 100).tolist(), "%{:.1f}", "#9333EA")
    top = missing_rates.sort_values(ascending=False).head(20)
    _write_bar_svg(FIGURE_DIR / "top20_missing_columns.svg", "Eksikliği en yüksek 20 sütun", top.index.tolist(), (top * 100).tolist(), "%{:.1f}", "#DC2626")
    version_path = TABLE_DIR / "dataset_versions.csv"
    if version_path.exists():
        versions = pd.read_csv(version_path)
        _write_bar_svg(FIGURE_DIR / "dataset_feature_counts.svg", "Referans veri seti özellik sayıları", versions.version.tolist(), versions.n_features.astype(float).tolist(), "{:.0f}", "#0369A1")

    summary = {
        "semantic_missing_cells": int(missing_matrix.to_numpy().sum()),
        "semantic_missing_rate": float(missing_matrix.to_numpy().mean()),
        "constant_columns": len(constant),
        "exact_duplicate_dropped_columns": len(duplicate_drop),
        "exact_duplicate_groups": len(duplicate_rows),
        "high_missing_ge_85_columns": int((missing_rates >= 0.85).sum()),
        "categorical_levels": len(cat_rows),
        **provenance,
    }
    (TABLE_DIR / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    print(json.dumps(build_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
