"""Build all MASTER preprocessing artefacts, datasets and reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    STRATEGIES,
    MasterPreprocessor,
    create_repeated_stratified_folds,
    feature_columns,
    feature_kind,
    find_duplicate_columns,
    load_master_csv,
    sha256_file,
)


RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
CLEAN_FILE = ROOT / "data" / "clean" / "MASTER_00_CANONICAL.csv"
OUTER_SPLIT_FILE = ROOT / "artifacts" / "master_outer_split.csv"
FOLD_FILE = ROOT / "artifacts" / "master_repeated_cv_folds.csv"
FEATURE_AUDIT_FILE = ROOT / "artifacts" / "master_feature_audit.csv"
ROW_FLAGS_FILE = ROOT / "artifacts" / "master_row_quality_flags.csv"
MANIFEST_FILE = ROOT / "artifacts" / "master_preprocessing_manifest.json"
FOLD_AUDIT_FILE = ROOT / "results" / "master_fold_safety_audit.csv"
QUALITY_FILE = ROOT / "results" / "master_quality_checks.json"
SEEDS = [42, 52, 62, 72, 82]
N_SPLITS = 5


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10g")


def _prepend_identifiers(processed: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    output = processed.reset_index(drop=True).copy()
    output.insert(0, TARGET_COLUMN, source[TARGET_COLUMN].to_numpy())
    output.insert(0, ID_COLUMN, source[ID_COLUMN].astype(str).to_numpy())
    return output


def _output_duplicate_count(frame: pd.DataFrame) -> int:
    columns = feature_columns(frame)
    if not columns:
        return 0
    _, duplicate_of, _ = find_duplicate_columns(frame, columns, respect_kind=False)
    return len(duplicate_of)


def _validate_outer_split(canonical: pd.DataFrame) -> pd.DataFrame:
    if not OUTER_SPLIT_FILE.exists():
        raise FileNotFoundError("Frozen outer split is required")
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    required = {ID_COLUMN, TARGET_COLUMN, "partition"}
    if not required.issubset(outer.columns):
        raise ValueError(f"Outer split lacks columns: {sorted(required - set(outer.columns))}")
    if outer[ID_COLUMN].duplicated().any():
        raise ValueError("Outer split contains duplicate Variant_ID values")
    if set(outer[ID_COLUMN]) != set(canonical[ID_COLUMN]):
        raise ValueError("Outer split is not exhaustive for the current raw file")
    if set(outer["partition"]) != {"train", "validation"}:
        raise ValueError("Outer split partitions must be train and validation")
    labels = canonical.set_index(ID_COLUMN)[TARGET_COLUMN]
    expected = outer[ID_COLUMN].map(labels).astype(int)
    if not expected.eq(pd.to_numeric(outer[TARGET_COLUMN], errors="raise").astype(int)).all():
        raise ValueError("Outer split labels do not match the raw data")
    return outer


def _build_feature_audit(
    raw_text: pd.DataFrame,
    full: pd.DataFrame,
    train: pd.DataFrame,
    processors: dict[str, MasterPreprocessor],
) -> pd.DataFrame:
    features = feature_columns(full)
    _, full_duplicate_of, _ = find_duplicate_columns(full, features, respect_kind=True)
    rows: list[dict[str, Any]] = []

    indicator_lookup: dict[str, dict[str, str]] = {}
    for strategy, processor in processors.items():
        lookup: dict[str, str] = {}
        for representative, group in processor.missing_indicator_groups_.items():
            for member in group:
                lookup[member] = representative
        indicator_lookup[strategy] = lookup

    for column in features:
        observed_unique = int(train[column].nunique(dropna=True))
        effective_states = observed_unique + int(train[column].isna().any())
        row: dict[str, Any] = {
            "feature": column,
            "family": column.split("_", 1)[0],
            "inferred_type": feature_kind(column),
            "full_missing_count": int(full[column].isna().sum()),
            "full_missing_rate": float(full[column].isna().mean()),
            "outer_train_missing_count": int(train[column].isna().sum()),
            "outer_train_missing_rate": float(train[column].isna().mean()),
            "outer_train_observed_unique": observed_unique,
            "outer_train_effective_states": effective_states,
            "special_dot_slash_dot_count": int(
                raw_text[column].astype("string").eq("./.").sum()
            ),
            "full_exact_duplicate_of": full_duplicate_of.get(column, ""),
            "outer_train_high_missing_80": bool(train[column].isna().mean() >= 0.80),
        }
        for strategy, processor in processors.items():
            row[f"{strategy}__value_used"] = bool(
                column in processor.numeric_value_columns_
                or column in processor.categorical_value_columns_
            )
            row[f"{strategy}__source_duplicate_of"] = processor.source_duplicate_of_.get(
                column, ""
            )
            row[f"{strategy}__missing_indicator_representative"] = indicator_lookup[
                strategy
            ].get(column, "")
            row[f"{strategy}__high_missing_dropped"] = bool(
                column in processor.high_missing_dropped_
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_fold_audit(train: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, config in STRATEGIES.items():
        for seed in SEEDS:
            seed_folds = folds.loc[folds["repeat_seed"] == seed]
            for fold in range(N_SPLITS):
                hold_ids = set(seed_folds.loc[seed_folds["fold"] == fold, ID_COLUMN])
                hold_mask = train[ID_COLUMN].isin(hold_ids)
                fit_frame = train.loc[~hold_mask].reset_index(drop=True)
                hold_frame = train.loc[hold_mask].reset_index(drop=True)
                processor = MasterPreprocessor(config).fit(fit_frame)
                fit_output = processor.transform(fit_frame)
                hold_output = processor.transform(hold_frame)
                schema_equal = list(fit_output.columns) == list(hold_output.columns)
                finite = bool(
                    np.isfinite(fit_output.to_numpy()).all()
                    and np.isfinite(hold_output.to_numpy()).all()
                )
                if not schema_equal or not finite:
                    raise ValueError(
                        f"Fold safety failed for {strategy}, seed={seed}, fold={fold}"
                    )
                rows.append(
                    {
                        "strategy": strategy,
                        "repeat_seed": seed,
                        "fold": fold,
                        "fit_n": len(fit_frame),
                        "holdout_n": len(hold_frame),
                        "output_features": len(fit_output.columns),
                        "high_missing_dropped": len(processor.high_missing_dropped_),
                        "source_duplicates_removed": len(processor.source_duplicate_of_),
                        "value_columns_removed_as_uninformative": len(
                            processor.uninformative_value_dropped_
                        ),
                        "unique_missing_indicators": len(
                            processor.indicator_representatives_
                        ),
                        "schema_equal": schema_equal,
                        "finite_values": finite,
                        "fit_scope": "fold_training_only",
                        "status": "passed",
                    }
                )
    return pd.DataFrame(rows)


def _write_reports(manifest: dict[str, Any], quality: dict[str, Any]) -> None:
    scenario = manifest["materialized_outer_split_datasets"]
    report = f"""# MASTER Paneli Güncel Veri Ön İşleme Raporu

## Sonuç

Ham veri korunmuş, bilinen `./.` eksik değerleri standartlaştırılmış ve dört karşılaştırılabilir ön işleme varyantı hazırlanmıştır. Model eğitimi yapılmamıştır; hangi varyantın kazanan olduğu aynı CV bölümlerinde model karşılaştırması yapılmadan ilan edilmemelidir.

## Veri özeti

- Satır: **{quality['raw']['rows']:,}**
- Ham özellik: **{quality['raw']['features']}**
- Sınıf dağılımı: **0={quality['raw']['label_counts']['0']:,}, 1={quality['raw']['label_counts']['1']:,}**
- Standartlaştırılan `./.` hücresi: **{quality['raw']['dot_slash_dot_cells']}**
- Kanonik eksik hücre oranı: **%{100 * quality['raw']['canonical_missing_rate']:.2f}**
- Tamamen özelliksiz satır: **{quality['raw']['all_feature_missing_rows']}**; silinmedi, kalite bayrağıyla raporlandı.
- `%80+` eksik ham özellik: **{quality['raw']['high_missing_80_features']}**
- Birebir aynı ham sütunlarda ilk kopya dışındaki sütun: **{quality['raw']['duplicate_source_columns_beyond_first']}**
- M1 referansında birebir aynı çıktı özelliği: **{quality['scenarios']['M1_native_clean']['exact_duplicate_output_features']}**; M2-M4'te **0**.

## Üretilen varyantlar

| Varyant | Dış-train özellik | Açıklama | Kullanım |
|---|---:|---|---|
| M1_native_clean | {scenario['M1_native_clean']['features']} | Eksik token temizliği, medyan doldurma ve kategorik one-hot temel çizgisi | Referans |
| M2_compact | {scenario['M2_compact']['features']} | M1 + birebir aynı kaynakların ve bilgi taşımayan değer sütunlarının azaltılması | Kompakt ablation |
| M3_missing_aware_compact | {scenario['M3_missing_aware_compact']['features']} | M2 + 285 adaydan 18 benzersiz sayısal eksiklik deseni | **İlk önerilen aday** |
| M4_high_missing_ablation | {scenario['M4_high_missing_ablation']['features']} | M3 + dış-train içinde `%80+` eksik kaynakların çıkarılması | Riskli eşik ablation'ı |

M3 ilk model karşılaştırması için önerilir; ancak M1-M4 aynı model ve aynı fold bankasıyla kıyaslanmalıdır. M4 otomatik olarak daha iyi kabul edilmemelidir, çünkü seyrek fakat yararlı bilgi kaybedebilir.

## Sızıntı önleme

- Medyanlar, nadir kategori listeleri, sütun seçimleri ve eksiklik göstergeleri yalnız ilgili eğitim fold'unda öğrenilir.
- Beş seed × beş fold = **25 değerlendirme** için toplam **{quality['fold_audit']['audited_fits']} fold-fit** kontrol edilmiştir.
- Dış validation preprocessing/model/threshold seçimi için kullanılmamalıdır.
- Hazır `train.csv` ve `validation.csv` dosyaları dış ayrımın incelenmesi içindir. CV sırasında bu dosyaları birleştirmek yerine `MasterPreprocessor` her fold'da yeniden fit edilmelidir.

## Önemli sınırlamalar

- Veri setinde hasta, aile veya örnek grup kimliği yoktur. Bu nedenle gerçek group-safe CV yapılamaz; kullanılan yöntem tekrarlı stratified 5-fold'dur.
- 116 satırın 351 özelliğinin tamamı eksiktir ve bu satırlarda iki etiket de vardır. Özelliklerden güvenilir biçimde ayrıştırılamazlar; keyfî silme yerine ayrıca raporlanmışlardır.
- Eksiklik örüntüsü hedefi veya veri kaynağını dolaylı kodlayabilir. Eksiklik göstergelerinin katkısı M2-M3 karşılaştırmasıyla ölçülmelidir.
"""
    (ROOT / "reports" / "MASTER_VERI_ONISLEME_RAPORU.md").write_text(
        report, encoding="utf-8"
    )

    changes = f"""# Kanser Paneline Göre MASTER Uyarlama Notları

| Değişiklik | Not | Gerekçe |
|---|---|---|
| Ham veriyi ayrı ve değişmeden koruma | İYİ | Bilimsel izlenebilirlik ve geri dönüş sağlar. |
| Ön işlemeyi her CV fold'unda yeniden fit etme | İYİ | Veri sızıntısını önler; Kanser panelindeki deney disiplinini taşır. |
| `./.` değerlerini gerçek eksik olarak tanıma | İYİ – MASTER'a özel | {quality['raw']['dot_slash_dot_cells']} hücrenin yanlışlıkla kategori/değer sayılmasını engeller. |
| Birebir aynı ham sütunları tek temsilciye indirme | İYİ | Gereksiz boyutu ve aynı bilginin tekrar ağırlık kazanmasını azaltır. |
| Her sütuna ayrı eksiklik göstergesi yerine benzersiz desenleri kullanma | İYİ | Önceki güncel V3'teki yinelenen göstergelerin yerine 285 aday deseni 18 temsilciye indirir. |
| Gözlenen değeri sabit olan sayısal sütunu azaltırken değişken eksiklik göstergesini koruma | İYİ | Değer bilgi taşımıyorsa atılır; eksiklik bilgi taşıyorsa kaybolmaz. |
| `%80+` eksik sütunları M3'te koruma | İYİ/temkinli | Sırf eksikliği yüksek diye olası önemli bilgi silinmez. |
| `%80+` eksik sütunları M4'te çıkarma | RİSKLİ | Yalnız ablation amacıyla vardır; CV kanıtı olmadan ana veri seti seçilmemelidir. |
| Tamamen boş {quality['raw']['all_feature_missing_rows']} satırı doğrudan silmeme, ayrı bayraklama | TEMKİNLİ | Bu satırlar bilgi taşımıyor; ancak silmek sınıf dağılımını değiştirebilir. Karar model ablation'ında verilmelidir. |
| AL/EK sütunlarını anlamsal doğrulama olmadan toplam/ortalama ile birleştirmeme | İYİ | Farklı biyolojik anlamları yanlışlıkla karıştırma riski önlenir. |
| Group-safe CV iddiasını kullanmama | SINIRLAMA | Gerçek grup kimliği verilmediği için yalnız stratified CV bilimsel olarak savunulabilir. |
| Information Gain, ikili/üçlü etkileşim ve voting'i bu pakete eklememe | İYİ – kapsam ayrımı | Bunlar sonraki seçim/modelleme aşamasında inner CV içinde denenmelidir. |

## Kısa karar

Kanser panelinden en faydalı aktarımlar fold-safe öğrenme, sabit split bankası, yinelenen özellikleri azaltma ve eksiklik bilgisini ayrı ablation olarak ölçmedir. MASTER için ek olarak `./.` tokenı düzeltilmiş ve tamamen boş satırlar görünür hâle getirilmiştir. Başlangıç adayı **M3**, karşılaştırma zorunluluğu ise **M1-M4'ün aynı model ve aynı fold'larla değerlendirilmesidir**.
"""
    (ROOT / "reports" / "KANSER_PANELINE_GORE_DEGISIKLIK_NOTLARI.md").write_text(
        changes, encoding="utf-8"
    )


def main() -> dict[str, Any]:
    for directory in [
        ROOT / "data" / "clean",
        ROOT / "data" / "scenarios",
        ROOT / "artifacts",
        ROOT / "reports",
        ROOT / "results",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_text = pd.read_csv(RAW_FILE, low_memory=False, keep_default_na=True)
    dot_slash_dot_cells = int(
        sum(raw_text[c].astype("string").eq("./.").sum() for c in raw_text.columns)
    )
    canonical = load_master_csv(RAW_FILE)
    _save_csv(canonical, CLEAN_FILE)

    outer = _validate_outer_split(canonical)
    partition = outer.set_index(ID_COLUMN)["partition"]
    train = canonical.loc[canonical[ID_COLUMN].map(partition).eq("train")].reset_index(drop=True)
    validation = canonical.loc[
        canonical[ID_COLUMN].map(partition).eq("validation")
    ].reset_index(drop=True)
    if len(train) != 2345 or len(validation) != 586:
        raise ValueError("Unexpected frozen outer split sizes")

    folds = create_repeated_stratified_folds(train, SEEDS, N_SPLITS)
    _save_csv(folds, FOLD_FILE)

    features = feature_columns(canonical)
    all_missing_mask = canonical[features].isna().all(axis=1)
    row_flags = canonical.loc[all_missing_mask, [ID_COLUMN, TARGET_COLUMN]].copy()
    row_flags["missing_feature_count"] = len(features)
    row_flags["missing_feature_rate"] = 1.0
    row_flags["quality_flag"] = "ALL_351_FEATURES_MISSING"
    _save_csv(row_flags, ROW_FLAGS_FILE)

    _, full_duplicate_of, full_duplicate_groups = find_duplicate_columns(
        canonical, features, respect_kind=True
    )
    processors: dict[str, MasterPreprocessor] = {}
    scenario_manifest: dict[str, Any] = {}
    scenario_quality: dict[str, Any] = {}

    for strategy, config in STRATEGIES.items():
        processor = MasterPreprocessor(config).fit(train)
        train_processed = processor.transform(train)
        validation_processed = processor.transform(validation)
        if list(train_processed.columns) != list(validation_processed.columns):
            raise ValueError(f"Outer schema mismatch for {strategy}")
        train_out = _prepend_identifiers(train_processed, train)
        validation_out = _prepend_identifiers(validation_processed, validation)
        strategy_dir = ROOT / "data" / "scenarios" / strategy
        train_path = strategy_dir / "train.csv"
        validation_path = strategy_dir / "validation.csv"
        _save_csv(train_out, train_path)
        _save_csv(validation_out, validation_path)
        processors[strategy] = processor

        duplicate_outputs = _output_duplicate_count(train_out)
        if strategy != "M1_native_clean" and duplicate_outputs:
            raise ValueError(f"{strategy} retains {duplicate_outputs} exact duplicate outputs")
        scenario_quality[strategy] = {
            "train_rows": len(train_out),
            "validation_rows": len(validation_out),
            "features": len(train_processed.columns),
            "exact_duplicate_output_features": duplicate_outputs,
            "nan_cells": int(train_processed.isna().sum().sum())
            + int(validation_processed.isna().sum().sum()),
            "infinite_cells": int(
                np.isinf(train_processed.to_numpy()).sum()
                + np.isinf(validation_processed.to_numpy()).sum()
            ),
        }
        scenario_manifest[strategy] = {
            "config": asdict(config),
            "train": str(train_path.relative_to(ROOT)).replace("\\", "/"),
            "validation": str(validation_path.relative_to(ROOT)).replace("\\", "/"),
            "features": len(train_processed.columns),
            "fit_scope": "outer_training_only",
            "cv_usage": "refit MasterPreprocessor inside every training fold",
            "processor_audit": processor.audit_dict(),
        }

    feature_audit = _build_feature_audit(raw_text, canonical, train, processors)
    _save_csv(feature_audit, FEATURE_AUDIT_FILE)

    fold_audit = _build_fold_audit(train, folds)
    _save_csv(fold_audit, FOLD_AUDIT_FILE)

    label_counts = {
        str(int(label)): int(count)
        for label, count in canonical[TARGET_COLUMN].value_counts().sort_index().items()
    }
    high_missing = canonical[features].isna().mean() >= 0.80
    quality = {
        "status": "passed",
        "raw": {
            "rows": len(canonical),
            "features": len(features),
            "label_counts": label_counts,
            "unique_variant_ids": int(canonical[ID_COLUMN].nunique()),
            "dot_slash_dot_cells": dot_slash_dot_cells,
            "canonical_missing_cells": int(canonical[features].isna().sum().sum()),
            "canonical_missing_rate": float(canonical[features].isna().mean().mean()),
            "all_feature_missing_rows": int(all_missing_mask.sum()),
            "all_feature_missing_label_counts": {
                str(int(label)): int(count)
                for label, count in canonical.loc[all_missing_mask, TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            },
            "high_missing_80_features": int(high_missing.sum()),
            "duplicate_source_groups": int(
                sum(len(group) > 1 for group in full_duplicate_groups.values())
            ),
            "duplicate_source_columns_beyond_first": len(full_duplicate_of),
            "raw_sha256": sha256_file(RAW_FILE),
            "clean_sha256": sha256_file(CLEAN_FILE),
        },
        "outer_split": {
            "train_n": len(train),
            "validation_n": len(validation),
            "train_label_counts": {
                str(int(label)): int(count)
                for label, count in train[TARGET_COLUMN].value_counts().sort_index().items()
            },
            "validation_label_counts": {
                str(int(label)): int(count)
                for label, count in validation[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "fold_audit": {
            "seeds": SEEDS,
            "folds_per_seed": N_SPLITS,
            "audited_fits": len(fold_audit),
            "all_passed": bool(fold_audit["status"].eq("passed").all()),
        },
        "scenarios": scenario_quality,
    }

    manifest = {
        "panel": "MASTER",
        "package_version": "2026-08-15",
        "model_training_performed": False,
        "raw_file": str(RAW_FILE.relative_to(ROOT)).replace("\\", "/"),
        "canonical_file": str(CLEAN_FILE.relative_to(ROOT)).replace("\\", "/"),
        "outer_split": {
            "source": str(OUTER_SPLIT_FILE.relative_to(ROOT)).replace("\\", "/"),
            "train_n": len(train),
            "validation_n": len(validation),
            "policy": "frozen current MASTER split retained for comparability",
        },
        "repeated_cv": {
            "method": "repeated stratified 5-fold",
            "seeds": SEEDS,
            "folds_per_seed": N_SPLITS,
            "assignment_file": str(FOLD_FILE.relative_to(ROOT)).replace("\\", "/"),
            "group_safe_claim": False,
            "reason": "No patient/family/sample group identifier was supplied.",
        },
        "recommended_first_candidate": "M3_missing_aware_compact",
        "selection_rule": "Compare M1-M4 with the same model and identical CV folds before choosing.",
        "materialized_outer_split_datasets": scenario_manifest,
        "leakage_policy": "Every learned preprocessing parameter must be fit on the current training fold only.",
        "high_missing_policy": "Retained in M3; removal exists only as the M4 ablation.",
        "empty_row_policy": "Retained and separately flagged; no silent deletion.",
    }
    _json_dump(MANIFEST_FILE, manifest)
    _json_dump(QUALITY_FILE, quality)
    _write_reports(manifest, quality)
    print(
        json.dumps(
            {
                "status": "built",
                "rows": len(canonical),
                "scenarios": {
                    key: value["features"] for key, value in scenario_manifest.items()
                },
                "fold_fits_audited": len(fold_audit),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return {"manifest": manifest, "quality": quality}


if __name__ == "__main__":
    main()
