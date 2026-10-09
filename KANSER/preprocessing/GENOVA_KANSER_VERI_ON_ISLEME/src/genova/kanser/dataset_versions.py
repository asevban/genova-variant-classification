"""Ham KANSER CSV'sinden referans/ablation veri setlerini üretir."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .preprocessing import KanserPreprocessor, POLICIES
from .schema import feature_columns, validate_schema

ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "raw" / "YARISMA_TRAIN_KANSER.csv"
OUT_DIR = ROOT / "data" / "processed" / "kanser"
CONFIG_DIR = ROOT / "configs" / "kanser"
ARTIFACT_DIR = ROOT / "artifacts" / "preprocessors"

DATASETS = [
    ("01_KANSER_NATIVE_SAFE_FULL.csv", "native", "V1"),
    ("02_KANSER_COMPACT_NONREDUNDANT.csv", "compact", "V2"),
    ("03_KANSER_ROBUST_COVERAGE85.csv", "robust85", "V3"),
    ("04_KANSER_SOURCE_RESISTANT_ENGINEERED_LEGACY.csv", "legacy_combined", "V4_LEGACY"),
    ("A1_KANSER_SOURCE_COLUMNS_REMOVED.csv", "source_removed", "A1"),
    ("A2_KANSER_MISSINGNESS_FEATURES_ONLY.csv", "missingness_only", "A2"),
    ("A3_KANSER_AA_FEATURES_ONLY.csv", "aa_only", "A3"),
    ("A4_KANSER_SOURCE_RESISTANT_COMBINED.csv", "source_resistant_combined", "A4"),
]


def _row_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    features = feature_columns(raw)
    row_hash = pd.util.hash_pandas_object(raw[features], index=False).astype("uint64").astype(str)
    meta = raw[["Variant_ID", "Label"]].copy()
    meta.insert(0, "row_index", range(len(meta)))
    meta["duplicate_group_id"] = "DG_" + row_hash
    meta["duplicate_group_size"] = row_hash.map(row_hash.value_counts()).astype("int16")
    return meta


def build_all(raw: pd.DataFrame | None = None) -> dict:
    raw = pd.read_csv(RAW_PATH, low_memory=False) if raw is None else raw.copy()
    report = validate_schema(raw)
    if not report["ok"]:
        raise ValueError(f"şema doğrulama başarısız: {report['issues']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    meta = _row_metadata(raw)
    meta.to_csv(OUT_DIR / "00_KANSER_TARGET_AND_METADATA.csv", index=False, encoding="utf-8-sig")

    versions = []
    policies = {}
    for filename, policy_name, version_id in DATASETS:
        transformer = KanserPreprocessor(policy=policy_name, missing_threshold=0.85)
        x = transformer.fit_transform(raw)
        if {"Variant_ID", "Label"} & set(x.columns):
            raise AssertionError(f"yasak kolon özellik dosyasına girdi: {filename}")
        x.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig", na_rep="")
        policy = transformer.policy_dict()
        policies[policy_name] = policy
        (ARTIFACT_DIR / f"{policy_name}_learned_policy.json").write_text(
            json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        config = {
            "version": version_id,
            "policy": policy_name,
            "description": POLICIES[policy_name],
            "source_file": "data/raw/YARISMA_TRAIN_KANSER.csv",
            "target_file": "data/processed/kanser/00_KANSER_TARGET_AND_METADATA.csv",
            "output_file": f"data/processed/kanser/{filename}",
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "fit_inside_cv": True,
            "reference_file_caveat": (
                "Statik CSV bütün etiketli eğitim verisinde tanımlanmış referanstır; "
                "resmî CV için aynı policy her training fold'unda yeniden fit edilir."
            ),
        }
        (CONFIG_DIR / f"{version_id.lower()}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        versions.append(config)

    summary = {
        "source_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "source_features": len(feature_columns(raw)),
        "label_counts": {str(k): int(v) for k, v in raw["Label"].value_counts().sort_index().items()},
        "duplicate_rows": int((meta["duplicate_group_size"] > 1).sum()),
        "duplicate_groups": int(meta.loc[meta["duplicate_group_size"] > 1, "duplicate_group_id"].nunique()),
        "versions": versions,
    }
    (OUT_DIR / "DATASET_BUILD_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(versions).to_csv(ROOT / "reports" / "tables" / "dataset_versions.csv", index=False, encoding="utf-8-sig")
    return {"summary": summary, "metadata": meta, "policies": policies}


def main():
    result = build_all()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
