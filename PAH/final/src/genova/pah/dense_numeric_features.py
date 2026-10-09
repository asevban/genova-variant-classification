"""Asama E2-EK: Random Forest ve KNN, XGBoost/CatBoost/LightGBM'in aksine
ham NaN ya da native `category` dtype kabul etmez -- tam sayisal, NaN-siz
bir matris ister. Bu modul, `fold_versions.py`'nin urettigi (native NaN +
native kategorik) ciktiyi, yalnizca train fold'undan fit edilerek boyle
bir matrise donusturur:

  1. Mevcut olan kategorik kolonlar (CAT_1/2 frekans, CAT_3/4/5+AA_1/2
     one-hot) `encoding.py`'nin zaten test edilmis sinifiklariyla kodlanir.
  2. Kalan sayisal kolonlardaki NaN, kolon-bazli TRAIN medyaniyla doldurulur.

Hangi kategorik kolonlarin gercekten mevcut oldugu versiyona gore degisir
(orn. `v4_from_v2` yalnizca `CAT_1` icerir) -- fonksiyon bunu otomatik
tespit eder.
"""
import pandas as pd

from genova.pah.encoding import NominalOneHotEncoder, FrequencyEncoder, MultiValueFrequencyEncoder
from genova.pah.fold_versions import CAT_ALL_COLS
from genova.pah.dataset_versions import NOMINAL_COLS, MULTI_VALUE_FREQ_COLS, SINGLE_VALUE_FREQ_COLS


def to_dense_numeric(X_train, X_test):
    """RF/KNN icin: tam sayisal, NaN-siz (X_train, X_test) cifti doner.
    Her adim yalnizca X_train'den fit edilir -- sizintisiz.
    """
    train_df, test_df = X_train.copy(), X_test.copy()
    cat_cols_present = [c for c in CAT_ALL_COLS if c in train_df.columns]

    encoders = [
        (MultiValueFrequencyEncoder, [c for c in MULTI_VALUE_FREQ_COLS if c in cat_cols_present]),
        (FrequencyEncoder, [c for c in SINGLE_VALUE_FREQ_COLS if c in cat_cols_present]),
        (NominalOneHotEncoder, [c for c in NOMINAL_COLS if c in cat_cols_present]),
    ]
    for cls, cols in encoders:
        if not cols:
            continue
        step = cls(columns=cols).fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    numeric_cols = [c for c in train_df.columns if str(train_df[c].dtype) != "category"]
    for col in numeric_cols:
        if train_df[col].isna().any() or test_df[col].isna().any():
            train_median = train_df[col].median()
            train_df[col] = train_df[col].fillna(train_median)
            test_df[col] = test_df[col].fillna(train_median)

    X_train_dense = train_df.astype(float)
    X_test_dense = test_df.reindex(columns=train_df.columns, fill_value=0).astype(float)
    return X_train_dense, X_test_dense


def to_dense_numeric_al_zero_fill(X_train, X_test):
    """P2 madde 17: `to_dense_numeric` ile AYNI kategorik kodlama -- ama
    `AL_` kolonlarindaki NaN, TRAIN medyani yerine SIFIR ile doldurulur
    (`v2`'nin `ConstantFillImputer(strategy="zero")` stratejisiyle ayni).
    Gerekce: CatBoost/XGBoost/LightGBM native NaN kullandigi icin AL_'de
    hicbir doldurma gormuyor; RF/KNN NaN kabul etmedigi icin MUTLAKA bir
    doldurma gerekiyor -- ama bu doldurmanin `v1`/`v2_raw_nan` medyaniyle
    mi yoksa `v2`'nin sifir-doldurmasiyla mi yapildigi, RF'in agac
    modelleriyle adil karsilastirilip karsilastirilmadigini etkiliyor.
    `EK_` kolonlari (ve diger sayisal kolonlar) yine TRAIN medyaniyla
    doldurulur -- yalnizca `AL_` degisir.
    """
    train_df, test_df = X_train.copy(), X_test.copy()
    cat_cols_present = [c for c in CAT_ALL_COLS if c in train_df.columns]

    encoders = [
        (MultiValueFrequencyEncoder, [c for c in MULTI_VALUE_FREQ_COLS if c in cat_cols_present]),
        (FrequencyEncoder, [c for c in SINGLE_VALUE_FREQ_COLS if c in cat_cols_present]),
        (NominalOneHotEncoder, [c for c in NOMINAL_COLS if c in cat_cols_present]),
    ]
    for cls, cols in encoders:
        if not cols:
            continue
        step = cls(columns=cols).fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    numeric_cols = [c for c in train_df.columns if str(train_df[c].dtype) != "category"]
    al_cols = [c for c in numeric_cols if c.startswith("AL_")]
    other_cols = [c for c in numeric_cols if not c.startswith("AL_")]

    for col in al_cols:
        if train_df[col].isna().any() or test_df[col].isna().any():
            train_df[col] = train_df[col].fillna(0.0)
            test_df[col] = test_df[col].fillna(0.0)

    for col in other_cols:
        if train_df[col].isna().any() or test_df[col].isna().any():
            train_median = train_df[col].median()
            train_df[col] = train_df[col].fillna(train_median)
            test_df[col] = test_df[col].fillna(train_median)

    X_train_dense = train_df.astype(float)
    X_test_dense = test_df.reindex(columns=train_df.columns, fill_value=0).astype(float)
    return X_train_dense, X_test_dense
