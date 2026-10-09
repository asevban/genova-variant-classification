"""Ham CSV'den v1-v3 PAH deney veri seti versiyonlarını üretir; yalnızca
schema.py / missingness.py / encoding.py / transforms.py'deki fit/transform
yapı taşlarını kullanır. v4 (özellik-seçilmiş), kararlı özellik listesi
oluştuktan sonra Aşama D'de ayrıca üretilir.

ÖNEMLİ sızıntı notu: bu modülün yazdığı parquet dosyaları REFERANS/DENEY-
İSKELETİ çıktılarıdır -- buradaki imputasyon (EK_3 medyanı) ve kategorik
frekans kodlaması tam (dedup sonrası) veri setinde fit edilir, fold-bazında
değil. Bu, inceleme, hızlı doğruluk kontrolleri ve özellik uzayını
tanımlamak için sorun değil, ama Aşama D/E'nin gerçek nested-CV performans
rakamları bu dosyalardaki önceden-hesaplanmış kolonları olduğu gibi okumak
yerine AYNI transformer sınıflarını fold-içinde (yalnızca eğitim fold'unda)
yeniden fit ETMELİDİR. Bunun aşağı akışta nasıl gözetildiği için
reports/02_ON_ISLEME_KARARLARI_PAH.md ve reports/03_OZELLIK_SECIMI_PAH.md'ye
bakın.

Çalıştırma: python -m genova.pah.dataset_versions
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from genova.pah.schema import validate_schema
from genova.pah.missingness import (
    BlockMissingIndicator,
    ConstantFillImputer,
    MedianImputerWithIndicator,
)
from genova.pah.encoding import NominalOneHotEncoder, FrequencyEncoder, MultiValueFrequencyEncoder
from genova.pah.transforms import (
    Log1pTransformer,
    LogitTransformer,
    RankQuantileHarmonizer,
    classify_al_columns,
)

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "data" / "raw" / "YARISMA_TRAIN_PAH.csv"
OUT_DIR = ROOT / "data" / "processed" / "pah"
CONFIG_DIR = ROOT / "configs" / "pah"

NOMINAL_COLS = ["CAT_3", "CAT_4", "CAT_5", "AA_1", "AA_2"]
FREQ_ENCODE_COLS = ["CAT_1", "CAT_2"]
# CAT_1 bazi satirlarda "&" ile birlesik cok-degerli deger tasiyor (5 satir,
# bkz. reports/01_EDA_RAPORU_PAH.md); CAT_2'de bu desen yok. Frekans kodlama
# bu yuzden ikiye ayrilir -- CAT_1 MultiValueFrequencyEncoder, CAT_2 duz
# FrequencyEncoder kullanir.
MULTI_VALUE_FREQ_COLS = ["CAT_1"]
SINGLE_VALUE_FREQ_COLS = ["CAT_2"]
ALL_CAT_COLS = NOMINAL_COLS + FREQ_ENCODE_COLS
EK_BLOCK_COLS = ["EK_1", "EK_2", "EK_4", "EK_5", "EK_6", "EK_7", "EK_8", "EK_9"]  # EK_3 haric tum EK_
EK_UNBOUNDED_COLS = ["EK_1", "EK_2", "EK_7", "EK_8", "EK_9"]  # sinirsiz native olcek, bkz. EDA A.6
CAT_BLOCK_COLS = ["CAT_3", "CAT_4", "CAT_5"]
CONFLICT_GROUP_ID = "conflict_group_1"


def build_v1(raw_df):
    """Minimal temizlik: CAT_6'yı (%100 eksik) düşürür, tam-yinelenen
    satırları düşürür, GroupKFold için group_id atar, kategorikleri native
    dtype'a çevirir. Sayısal AL_/EK_ kolonları ham NaN korur (ağaç modeli
    dostu).
    """
    report = validate_schema(raw_df)
    if not report["ok"]:
        raise ValueError(f"schema dogrulama basarisiz: {report['issues']}")

    df = raw_df.drop(columns=["CAT_6"]).copy()

    feature_cols = [c for c in df.columns if c not in ("Variant_ID", "Label")]
    is_dupe = df.duplicated(subset=feature_cols + ["Label"], keep="first")
    dropped_ids = df.loc[is_dupe, "Variant_ID"].tolist()
    df = df.loc[~is_dupe].reset_index(drop=True)

    profile_str = df[feature_cols].fillna("__NA__").astype(str).agg("|".join, axis=1)
    conflict_hashes = profile_str[profile_str.duplicated(keep=False)]
    df["group_id"] = df["Variant_ID"]
    df.loc[conflict_hashes.index, "group_id"] = CONFLICT_GROUP_ID

    for col in ALL_CAT_COLS:
        df[col] = df[col].astype("category")

    return df, {"dropped_duplicate_variant_ids": dropped_ids, "n_rows": len(df)}


def build_v2(v1_df, al_columns):
    """v1 + blok-seviye eksiklik göstergeleri + EK_3 için gösterge-ile-
    medyan-doldurma; hem AL_ sıfır-dolu hem ham-NaN varyantı olarak (aynı
    kod yolu, farklı doldurma stratejisi).

    Not (Aşama D Düzeltme Turu): önceden burada ayrıca bir
    `AL_296_missing` göstergesi de ekleniyordu. Fisher's exact test ile
    doğrulandı (bkz. reports/03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md):
    bu göstergenin koşulsuz anlamlılığı (p=6.08e-06) tamamen
    `al_all_missing` ile örtüşmesinden geliyordu -- `al_all_missing=1`
    alt-kümesinde sabit (sıfır ek ayırt edicilik), `al_all_missing=0`
    alt-kümesinde anlamsız (p=0.184). Kaldırıldı.
    """
    al_types = classify_al_columns(v1_df, al_columns)
    al_numeric_cols = al_types["constant"] + al_types["ratio_type"] + al_types["frequency_type"]

    df = BlockMissingIndicator(columns=al_columns, name="al_all_missing").fit_transform(v1_df)
    df = BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing").fit_transform(df)
    df = MedianImputerWithIndicator(columns=["EK_3"]).fit_transform(df)

    df_zero = ConstantFillImputer(columns=al_numeric_cols, strategy="zero").fit_transform(df)
    df_raw_nan = ConstantFillImputer(columns=al_numeric_cols, strategy="none").fit_transform(df)

    return df_zero, df_raw_nan, {"al_types": al_types}


def build_v3(v2_zero_df, al_columns, al_corr_clusters, min_cluster_size=3):
    """v2 (sıfır-dolu) + log1p/logit AL_ dönüşümleri + EK_ rank/quantile
    harmonizasyonu + korelasyon-kümesi grup-özet özellikleri + tam kategorik
    kodlama. v1/v2 native kategoriklerle ağaç-modeli dostu kalırken, v3 bu
    kodlanmış/dönüştürülmüş tam-sayısal kolu temsil eder.

    ÖLÇEKLEME İÇERMEZ: önceden burada koşulsuz bir `RobustScaler` gömülüydü.
    Ampirik olarak (bkz. reports/02_ON_ISLEME_KARARLARI_PAH.md §11,
    reports/04_PREPROCESSING_MERDIVENI_PAH.md M-4) `RobustScaler`'ın etkisi
    model-ailesine göre çok farklı: ağaç modelleri için kayıtsız, ama LogReg
    için `StandardScaler`'a göre 15 puan daha kötü F1. Bu yüzden ölçekleme
    artık v3'e gömülü değil -- Aşama E'nin nested iç döngüsünde model-
    ailesine göre ayrı bir hiperparametre olarak seçilecek.

    `CAT_1` bazı satırlarda `&`-birleşik çok-değerli değer taşıdığı için
    (5 satır, bkz. reports/01_EDA_RAPORU_PAH.md) `MultiValueFrequencyEncoder`
    ile kodlanır; `CAT_2`'de bu desen yok, düz `FrequencyEncoder` kullanır.
    """
    al_types = classify_al_columns(v2_zero_df, al_columns)

    df = Log1pTransformer(columns=al_types["frequency_type"]).fit_transform(v2_zero_df)
    df = LogitTransformer(columns=al_types["ratio_type"]).fit_transform(df)
    df = RankQuantileHarmonizer(columns=EK_UNBOUNDED_COLS).fit_transform(df)

    cluster_sizes = al_corr_clusters["cluster_r05"].value_counts()
    usable_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
    for cluster_id in usable_clusters:
        cols = al_corr_clusters.loc[al_corr_clusters["cluster_r05"] == cluster_id, "column"].tolist()
        prefix = f"AL_cluster{cluster_id}"
        block = df[cols]
        df[f"{prefix}_min"] = block.min(axis=1)
        df[f"{prefix}_max"] = block.max(axis=1)
        df[f"{prefix}_median"] = block.median(axis=1)
        df[f"{prefix}_n_positive"] = (block > 0).sum(axis=1)

    df = MultiValueFrequencyEncoder(columns=MULTI_VALUE_FREQ_COLS).fit_transform(df)
    df = FrequencyEncoder(columns=SINGLE_VALUE_FREQ_COLS).fit_transform(df)
    df = NominalOneHotEncoder(columns=NOMINAL_COLS).fit_transform(df)

    return df, {"n_group_summary_clusters": len(usable_clusters)}


def build_v4(v2_zero_df, v3_df, stable_features):
    """v2 ve v3'ün, Aşama D.1'in kararlı özellik havuzuna (4 seçim
    yönteminden ≥2'sinde uzlaşan özellikler) indirgenmiş hâli. Havuz ne
    olursa olsun Variant_ID/Label/group_id her zaman korunur.
    """
    keep_meta = ["Variant_ID", "Label", "group_id"]
    v4_from_v2 = v2_zero_df[keep_meta + [c for c in stable_features if c in v2_zero_df.columns]]
    v4_from_v3 = v3_df[keep_meta + [c for c in stable_features if c in v3_df.columns]]
    return v4_from_v2, v4_from_v3


def _write_version(df, version_name, description, provenance, chain):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / f"{version_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    config = {
        "version": version_name,
        "description": description,
        "source_file": str(RAW_CSV.relative_to(ROOT)),
        "output_file": str(parquet_path.relative_to(ROOT)),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "function_chain": chain,
        "provenance": provenance,
        "leakage_caveat": (
            "Bu dosya referans/deney iskeleti amaclidir. Medyan doldurma ve "
            "frekans kodlamasi tam veri setinde (post-dedup, 369 satir) fit "
            "edilmistir. Asama D/E'deki resmi nested-CV performans olcumu "
            "icin bu degerler kullanilmaz; ayni fit/transform siniflari "
            "fold-icinde train-fold-only olarak yeniden fit edilir."
        ),
    }
    with open(CONFIG_DIR / f"{version_name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"{version_name}: {df.shape} -> {parquet_path}")


def main():
    raw_df = pd.read_csv(RAW_CSV)
    al_columns = [c for c in raw_df.columns if c.startswith("AL_")]

    v1_df, v1_meta = build_v1(raw_df)
    _write_version(
        v1_df, "v1", "Minimal temiz: CAT_6 dusuruldu, tam-yinelenen 3 satir dedup edildi, sema dogrulandi.",
        v1_meta, ["validate_schema", "CAT_6'yi dusur", "duplicated(keep='first') ile dedup", "group_id ata",
                  "CAT_1/2/3/4/5, AA_1/2 icin astype('category')"],
    )

    v2_zero_df, v2_raw_df, v2_meta = build_v2(v1_df, al_columns)
    _write_version(
        v2_zero_df, "v2", "Eksiklik-farkinda (AL_ sifir-doldurulmus): v1 + blok/secili eksiklik gostergeleri + AL_ sifir-doldurma.",
        v2_meta, ["BlockMissingIndicator(AL_)", "BlockMissingIndicator(EK_+CAT_ block)",
                  "MedianImputerWithIndicator(EK_3)",
                  "ConstantFillImputer(AL_ numeric, strategy=zero)"],
    )
    _write_version(
        v2_raw_df, "v2_raw_nan", "Eksiklik-farkinda (AL_ ham NaN alternatifi): v2 ile ayni gostergeler, AL_ NaN korunur.",
        v2_meta, ["BlockMissingIndicator(AL_)", "BlockMissingIndicator(EK_+CAT_ block)",
                  "MedianImputerWithIndicator(EK_3)",
                  "ConstantFillImputer(AL_ numeric, strategy=none)"],
    )

    al_corr_clusters = pd.read_csv(ROOT / "reports" / "tables" / "AL_correlation_clusters.csv")
    v3_df, v3_meta = build_v3(v2_zero_df, al_columns, al_corr_clusters)
    _write_version(
        v3_df, "v3", "Donusturulmus + grup-ozellikli, OLCEKSIZ: v2 + log1p/logit + EK_ harmonizasyon + korelasyon-kumesi grup-ozetleri + tam kategorik kodlama. Olcekleme Asama E'de model-ailesine gore nested olarak yapilir (v3'e artik gomulu degil).",
        v3_meta, ["Log1pTransformer(frekans-tipi AL_)", "LogitTransformer(oran-tipi AL_)",
                  "RankQuantileHarmonizer(sinirsiz EK_)", "grup-ozet ozellikleri (|r|>0.5 kumeleri, boyut>=3)",
                  "MultiValueFrequencyEncoder(CAT_1)", "FrequencyEncoder(CAT_2)",
                  "NominalOneHotEncoder(CAT_3/4/5, AA_1/2)"],
    )

    pool_path = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
    if pool_path.exists():
        pool = json.loads(pool_path.read_text())
        v4_from_v2, v4_from_v3 = build_v4(v2_zero_df, v3_df, pool["features"])
        _write_version(
            v4_from_v2, "v4_from_v2", "Ozellik-secilmis (v2 tabanli): Asama D.1 stabil havuzuna (>=2 yontemde uyum) indirgenmis v2.",
            {"stable_pool_rule": pool["rule"], "n_features": pool["n_features"]},
            ["build_v2()", f"kararli havuza indirge ({pool_path.relative_to(ROOT)})"],
        )
        _write_version(
            v4_from_v3, "v4_from_v3", "Ozellik-secilmis (v3 tabanli): Asama D.1 stabil havuzuna (>=2 yontemde uyum) indirgenmis v3.",
            {"stable_pool_rule": pool["rule"], "n_features": pool["n_features"]},
            ["build_v3()", f"kararli havuza indirge ({pool_path.relative_to(ROOT)})"],
        )
    else:
        print("v4 atlandi: Asama D.1 stabil ozellik havuzu (reports/tables/v4_final_feature_pool.json) henuz yok.")


if __name__ == "__main__":
    main()
