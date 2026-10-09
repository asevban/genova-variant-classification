"""Aşama D için fold-güvenli özellik matrisi üretimi. Aşama B'nin tıpatıp
aynı fit/transform sınıflarını yeniden kullanır; her adımı YALNIZCA fold'un
eğitim satırlarında fit eder ve tutulan (held-out) satırlara yalnızca
transform uygular -- bu, dataset_versions.build_v2'nin (yalnızca referans/
inceleme amacıyla tam dedup-sonrası veri setinde fit eden) sızıntısız
karşılığıdır.
"""
import pandas as pd

from genova.pah.missingness import (
    BlockMissingIndicator,
    ConstantFillImputer,
    MedianImputerWithIndicator,
)
from genova.pah.encoding import NominalOneHotEncoder, FrequencyEncoder, MultiValueFrequencyEncoder
from genova.pah.transforms import classify_al_columns
from genova.pah.dataset_versions import (
    NOMINAL_COLS,
    MULTI_VALUE_FREQ_COLS,
    SINGLE_VALUE_FREQ_COLS,
    EK_BLOCK_COLS,
    CAT_BLOCK_COLS,
)

NON_FEATURE_COLS = ("Variant_ID", "Label", "group_id")


def build_fold_features(v1_df, al_columns, train_ids, test_ids):
    """Bir train/test bölmesi için v2-eşdeğeri (blok eksiklik göstergeleri
    + EK_3 medyan-doldurma + AL_ sıfır-doldurma + CAT_1/2 frekans +
    CAT_3/4/5,AA_1/2 one-hot) sayısal özellik matrisleri üretir; durumsal
    her adım yalnızca eğitim satırlarında fit edilir.

    (X_train, y_train, X_test, y_test)'i sayısal DataFrame/Series olarak döner.

    NOT (Aşama D'ye özgü): v2'nin aksine (Aşama B'nin "yalnızca EK_3 özel
    medyan doldurma" politikası gereği ~8 satırlık EK_/CAT_ blok-eksik
    satırları kendi blok göstergesinin arkasında ham NaN bırakır), bu
    fonksiyon kalan EK_ blok kolonlarını da AYRICA medyanla dolduruyor.
    Mutual information ve elastic-net seçici NaN kabul edemiyor; bu ek
    işlem yalnızca Aşama D'nin seçim-amaçlı yardımcı modelleriyle sınırlı
    ve reports/02_ON_ISLEME_KARARLARI_PAH.md'de belgelenen v1-v3
    politikasını değiştirmiyor.
    """
    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    test_df = v1_df[v1_df["Variant_ID"].isin(test_ids)].reset_index(drop=True)

    al_types = classify_al_columns(train_df, al_columns)
    al_numeric_cols = al_types["constant"] + al_types["ratio_type"] + al_types["frequency_type"]

    steps = [
        BlockMissingIndicator(columns=al_columns, name="al_all_missing"),
        BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing"),
        MedianImputerWithIndicator(columns=["EK_3"]),
        MedianImputerWithIndicator(columns=EK_BLOCK_COLS),
        ConstantFillImputer(columns=al_numeric_cols, strategy="zero"),
        MultiValueFrequencyEncoder(columns=MULTI_VALUE_FREQ_COLS),
        FrequencyEncoder(columns=SINGLE_VALUE_FREQ_COLS),
        NominalOneHotEncoder(columns=NOMINAL_COLS),
    ]

    for step in steps:
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    # EK_ blok doldurucusunun kendi kolon-bazli "_missing" gostergeleri
    # "ek_cat_block_missing" ile ayni bilgiyi tekrar ediyor (ayni 8 satir
    # tum blogu paylasiyor) -- secicilere ayni sinyalin gereksiz kopyalarini
    # vermemek icin bunlari dusuruyoruz.
    redundant_indicators = [f"{c}_missing" for c in EK_BLOCK_COLS]
    train_df = train_df.drop(columns=redundant_indicators)
    test_df = test_df.drop(columns=redundant_indicators)

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    # NominalOneHotEncoder kategorileri yalnizca eğitim fold'undan fit
    # ediyor, bu yuzden test_df'in one-hot kolonlari zaten tasarim geregi
    # train_df'inkiyle hizali (handle_unknown="ignore"); yine de savunmaci
    # olarak reindex ediyoruz.
    X_train = train_df[feature_cols].astype(float)
    X_test = test_df.reindex(columns=feature_cols, fill_value=0).astype(float)
    return X_train, train_df["Label"], X_test, test_df["Label"]


GROUP_PREFIXES = {
    "AL": lambda col: col.startswith("AL_"),
    "EK": lambda col: col.startswith("EK_") or col == "ek_cat_block_missing",
    "AA": lambda col: col.startswith("AA_"),
    "CAT": lambda col: col.startswith("CAT_"),
}


def columns_for_group(feature_cols, group):
    return [c for c in feature_cols if GROUP_PREFIXES[group](c)]
