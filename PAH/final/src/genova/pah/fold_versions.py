"""Asama E2 icin fold-guvenli veri versiyonu insaacilari. Split bankasindaki
her outer/inner fold'da yalnizca egitim satirlarinda fit edilen,
dataset_versions.py ile ayni fit/transform yapi taslarini yeniden kullanan
v1 / v2 / v2_raw_nan / v4_from_v2 / v3 / v4_from_v3 esdegerleri.

`data/processed/pah/*.parquet` dosyalari DOGRUDAN OKUNMAZ -- onlarin
medyan/frekans/kume istatistikleri tum 369 satirda fit edilmis (bkz.
dataset_versions.py'nin sizinti notu), nested-CV'de kullanilamaz. Bu modul
yalnizca `v1.parquet`'i (ham NaN + native kategorik, hicbir istatistiksel
fit icermez) kaynak alir ve her seferinde ayni sinif/fonksiyonlari
(missingness.py, encoding.py, transforms.py) fold-ici yeniden fit eder.

Leakage notu (yalnizca v3/v4_from_v3 icin): AL_ korelasyon-kumesi grup-ozet
ozellikleri (yalnizca v3-tam kullanir, v4_from_v3 hic kullanmaz --
25-ozellik havuzunda hicbir AL_clusterN_* kolonu yok), `reports/tables/
AL_correlation_clusters.csv`'deki SABIT (tum 369 satirda bir kez
hesaplanmis) kume tanimini kullanir -- bu, Asama B/C'den devralinan,
istatistiksel bir "fit" degil, hangi AL_ kolonlarinin ayni grupta
sayilacagina dair sabit bir tasarim-zamani karari. Sayisal ozet degerleri
yine de yalnizca o fold'un fold-ici sifir-doldurulmus AL_ degerlerinden
hesaplanir. Bu, E2 raporunda acikca belirtilir.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from genova.pah.missingness import BlockMissingIndicator, ConstantFillImputer, MedianImputerWithIndicator
from genova.pah.encoding import NominalOneHotEncoder, FrequencyEncoder, MultiValueFrequencyEncoder
from genova.pah.transforms import Log1pTransformer, LogitTransformer, RankQuantileHarmonizer, classify_al_columns
from genova.pah.dataset_versions import (
    NOMINAL_COLS,
    MULTI_VALUE_FREQ_COLS,
    SINGLE_VALUE_FREQ_COLS,
    EK_BLOCK_COLS,
    EK_UNBOUNDED_COLS,
    CAT_BLOCK_COLS,
)

ROOT = Path(__file__).resolve().parents[3]
CAT_ALL_COLS = ["CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "AA_1", "AA_2"]
NON_FEATURE_COLS = ("Variant_ID", "Label", "group_id")


def _split(v1_df, train_ids, test_ids):
    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    test_df = v1_df[v1_df["Variant_ID"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def _refit_categoricals(train_df, test_df, columns):
    """Kategori sozlugunu yalnizca train_df'ten cikarir; test_df'te
    gorulmeyen bir kategori NaN'a eslenir -- kategori sozlugu de bir
    sizinti kaynagi olmasin diye.
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    for col in columns:
        train_values = train_df[col].astype(object)
        categories = sorted(v for v in train_values.dropna().unique())
        train_df[col] = pd.Categorical(train_values, categories=categories)
        test_values = test_df[col].astype(object).where(test_df[col].astype(object).isin(categories), other=np.nan)
        test_df[col] = pd.Categorical(test_values, categories=categories)
    return train_df, test_df


def _al_ek_cols(v1_df):
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    ek_cols = [c for c in v1_df.columns if c.startswith("EK_")]
    return al_cols, ek_cols


def build_v1(v1_df, train_ids, test_ids):
    """v1: ham NaN (AL_/EK_), native kategorik (CAT_/AA_)."""
    al_cols, ek_cols = _al_ek_cols(v1_df)
    feature_cols = al_cols + ek_cols + CAT_ALL_COLS
    train_df, test_df = _split(v1_df, train_ids, test_ids)
    train_df, test_df = _refit_categoricals(train_df, test_df, CAT_ALL_COLS)
    return train_df[feature_cols], train_df["Label"], test_df[feature_cols], test_df["Label"]


def _v2_steps(al_cols, al_fill_strategy):
    return [
        BlockMissingIndicator(columns=al_cols, name="al_all_missing"),
        BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing"),
        MedianImputerWithIndicator(columns=["EK_3"]),
        ConstantFillImputer(columns=al_cols, strategy=al_fill_strategy),
    ]


def _build_v2_like(v1_df, train_ids, test_ids, al_fill_strategy):
    al_cols, ek_cols = _al_ek_cols(v1_df)
    train_df, test_df = _split(v1_df, train_ids, test_ids)
    for step in _v2_steps(al_cols, al_fill_strategy):
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)
    train_df, test_df = _refit_categoricals(train_df, test_df, CAT_ALL_COLS)
    feature_cols = al_cols + ek_cols + CAT_ALL_COLS + ["al_all_missing", "ek_cat_block_missing", "EK_3_missing"]
    return train_df[feature_cols], train_df["Label"], test_df[feature_cols], test_df["Label"]


def build_v2(v1_df, train_ids, test_ids):
    """v2: v1 + blok eksiklik gostergeleri + EK_3 medyan-doldurma+gosterge
    + AL_ sifir-doldurma (dataset_versions.build_v2 ile ayni davranis)."""
    return _build_v2_like(v1_df, train_ids, test_ids, al_fill_strategy="zero")


def build_v2_raw_nan(v1_df, train_ids, test_ids):
    """v2_raw_nan: v2 ile ayni gostergeler, AL_ ham NaN korunur."""
    return _build_v2_like(v1_df, train_ids, test_ids, al_fill_strategy="none")


def build_v4_from_v2(v1_df, train_ids, test_ids, pool):
    """v4_from_v2: v2'nin Asama D.1 kararli 25-ozellik havuzuna indirgenmis hali."""
    X_train, y_train, X_test, y_test = build_v2(v1_df, train_ids, test_ids)
    cols = [c for c in pool if c in X_train.columns]
    return X_train[cols], y_train, X_test[cols], y_test


def build_v3_full(v1_df, train_ids, test_ids, al_corr_clusters, min_cluster_size=3):
    """v3-tam: v2(sifir-dolu) + log1p/logit AL_ donusumleri + EK_ rank/
    quantile harmonizasyonu + korelasyon-kumesi grup-ozetleri (SABIT kume
    tanimi, bkz. modul-basi leakage notu) + tam kategorik kodlama
    (dataset_versions.build_v3 ile ayni davranis, fold-ici yeniden fit)."""
    al_cols, ek_cols = _al_ek_cols(v1_df)
    train_df, test_df = _split(v1_df, train_ids, test_ids)
    for step in _v2_steps(al_cols, "zero"):
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    al_types = classify_al_columns(train_df, al_cols)
    for step in [
        Log1pTransformer(columns=al_types["frequency_type"]),
        LogitTransformer(columns=al_types["ratio_type"]),
        RankQuantileHarmonizer(columns=EK_UNBOUNDED_COLS),
    ]:
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    # RankQuantileHarmonizer NaN'lari donusturmeden birakir (yalnizca
    # notna() maskesini islerken); Elastic-Net NaN kabul etmedigi icin
    # kalan EK_ blok-eksik satirlarini medyanla dolduruyoruz. Kendi
    # kolon-bazli gostergeleri ek_cat_block_missing ile ayni 8 satiri
    # tekrar ettigi icin dusuruluyor (fold_features.py'deki ayni desen).
    ek_median_step = MedianImputerWithIndicator(columns=EK_BLOCK_COLS)
    ek_median_step.fit(train_df)
    train_df = ek_median_step.transform(train_df)
    test_df = ek_median_step.transform(test_df)
    redundant_indicators = [f"{c}_missing" for c in EK_BLOCK_COLS]
    train_df = train_df.drop(columns=redundant_indicators)
    test_df = test_df.drop(columns=redundant_indicators)

    cluster_sizes = al_corr_clusters["cluster_r05"].value_counts()
    usable_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
    for cluster_id in usable_clusters:
        cols = al_corr_clusters.loc[al_corr_clusters["cluster_r05"] == cluster_id, "column"].tolist()
        prefix = f"AL_cluster{cluster_id}"
        for df in (train_df, test_df):
            block = df[cols]
            df[f"{prefix}_min"] = block.min(axis=1)
            df[f"{prefix}_max"] = block.max(axis=1)
            df[f"{prefix}_median"] = block.median(axis=1)
            df[f"{prefix}_n_positive"] = (block > 0).sum(axis=1)

    for step in [
        MultiValueFrequencyEncoder(columns=MULTI_VALUE_FREQ_COLS),
        FrequencyEncoder(columns=SINGLE_VALUE_FREQ_COLS),
        NominalOneHotEncoder(columns=NOMINAL_COLS),
    ]:
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X_train = train_df[feature_cols].astype(float)
    X_test = test_df.reindex(columns=feature_cols, fill_value=0).astype(float)
    return X_train, train_df["Label"], X_test, test_df["Label"]


def build_v4_from_v3(v1_df, train_ids, test_ids, pool):
    """v4_from_v3: v3'un 25-ozellik havuzuna indirgenmis hali. Havuzda hic
    AL_clusterN_* veya one-hot kolonu olmadigi icin (bkz.
    reports/tables/v4_final_feature_pool.json) kume/one-hot adimlarina hic
    gerek yok -- tamamen fold-ici, sabit-kume leakage notu bu fonksiyonu
    etkilemez. Havuzdaki EK_ kolonlarinin (EK_1/2/5/6/7/8/9) 8'er satirlik
    blok-eksikligi, medyanla dolduruluyor (indikator eklenmiyor -- havuzda
    zaten olmayan bir kolon icin sizintisiz ama sessiz bir doldurma;
    fold_features.py'nin ayni EK_ blok gostergelerini havuz-disi bulup
    dusurme deseniyle tutarli).
    """
    al_cols, ek_cols = _al_ek_cols(v1_df)
    train_df, test_df = _split(v1_df, train_ids, test_ids)
    for step in _v2_steps(al_cols, "zero"):
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    al_types = classify_al_columns(train_df, al_cols)
    ek_pool_cols = [c for c in pool if c in EK_BLOCK_COLS]
    for step in [
        Log1pTransformer(columns=al_types["frequency_type"]),
        LogitTransformer(columns=al_types["ratio_type"]),
        RankQuantileHarmonizer(columns=[c for c in EK_UNBOUNDED_COLS if c in pool]),
        MedianImputerWithIndicator(columns=ek_pool_cols),
        MultiValueFrequencyEncoder(columns=[c for c in MULTI_VALUE_FREQ_COLS if c in pool]),
    ]:
        if not step.columns:
            continue
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    cols = [c for c in pool if c in train_df.columns]
    X_train = train_df[cols].astype(float)
    X_test = test_df.reindex(columns=cols, fill_value=0).astype(float)
    return X_train, train_df["Label"], X_test, test_df["Label"]
