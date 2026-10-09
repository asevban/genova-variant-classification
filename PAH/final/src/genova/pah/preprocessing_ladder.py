"""Tamamlayıcı Deney Turu — Görev 5 (preprocessing merdiveni) ve Görev 4/6'nın
ortak alt yapısı. `build_step`, v1'den başlayarak, her çağrıda tek bir ön
işleme bileşeni ekleyen 0-5 arası "basamakları" fold-güvenli biçimde inşa
eder -- Aşama D.1'in `fold_features.py`'siyle aynı ilke (fit yalnızca
eğitim fold'unda), ama basamak sayısı parametrik.

Basamaklar:
  0 - v1 (minimal temiz, ham NaN)
  1 - + blok eksiklik göstergeleri (al_all_missing, EK_3_missing) -- henüz doldurma yok
  2 - + AL_ sıfır-doldurma + EK_3 medyan-doldurma (v2 dengi)
  3 - + log1p/logit + EK_ harmonizasyonu (v3'ün ölçeklemesiz hâli)
  4 - + ölçekleme (v3 dengi)
  5 - + Aşama D'nin 24-özellik havuzuna indirgeme (v4 dengi)

Not: sabit diagnostic modeller (LogReg/ExtraTrees) ham NaN kabul etmez.
Basamak 0/1'de kalan ham NaN, ön işleme kararı SAYILMAYAN, yalnızca model
girişini mümkün kılan nötr bir medyan doldurmayla (`SimpleImputer`)
kapatılır -- bu doldurma her iki basamakta da AYNI şekilde uygulanır, bu
yüzden basamaklar arası karşılaştırmayı bozmaz.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

from genova.pah.missingness import BlockMissingIndicator, SelectedMissingIndicators, MedianImputerWithIndicator, ConstantFillImputer
from genova.pah.encoding import FrequencyEncoder, MultiValueFrequencyEncoder, NominalOneHotEncoder
from genova.pah.transforms import classify_al_columns, Log1pTransformer, LogitTransformer, RankQuantileHarmonizer, ColumnwiseScaler
from genova.pah.dataset_versions import NOMINAL_COLS, MULTI_VALUE_FREQ_COLS, SINGLE_VALUE_FREQ_COLS, EK_BLOCK_COLS, CAT_BLOCK_COLS

ROOT = Path(__file__).resolve().parents[3]
EK_UNBOUNDED_COLS = ["EK_1", "EK_2", "EK_7", "EK_8", "EK_9"]
NON_FEATURE_COLS = ("Variant_ID", "Label", "group_id")


def _load_stable_pool():
    pool_path = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
    return json.loads(pool_path.read_text())["features"]


def build_step(v1_df, al_columns, train_ids, test_ids, step, scaler_method="robust", ek_columns=None):
    """Basamak `step` (0-5) için fold-güvenli (X_train, y_train, X_test, y_test) üretir.

    `al_columns`/`ek_columns` verilirse (Görev 6'nın yüksek-eksiklik filtresi
    gibi), yalnızca bu alt-kümeler kullanılır -- filtre kararının kendisi
    çağıran tarafından, train-fold-only olarak verilmelidir.
    """
    ek_block_cols = ek_columns if ek_columns is not None else EK_BLOCK_COLS
    ek_unbounded_cols = [c for c in EK_UNBOUNDED_COLS if c in ek_block_cols or c == "EK_3"]

    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    test_df = v1_df[v1_df["Variant_ID"].isin(test_ids)].reset_index(drop=True)

    al_types = classify_al_columns(train_df, al_columns)
    al_numeric_cols = al_types["constant"] + al_types["ratio_type"] + al_types["frequency_type"]

    chain = []
    if step == 1:
        chain += [
            BlockMissingIndicator(columns=al_columns, name="al_all_missing"),
            SelectedMissingIndicators(columns=["EK_3"]),
        ]
    elif step >= 2:
        chain += [
            BlockMissingIndicator(columns=al_columns, name="al_all_missing"),
            MedianImputerWithIndicator(columns=["EK_3"]),
            ConstantFillImputer(columns=al_numeric_cols, strategy="zero"),
        ]
    if step >= 3:
        chain += [
            Log1pTransformer(columns=al_types["frequency_type"]),
            LogitTransformer(columns=al_types["ratio_type"]),
            RankQuantileHarmonizer(columns=[c for c in ek_unbounded_cols if c != "EK_3"]),
        ]
    # kategorik kodlama her basamakta sabit -- ladder'in test ettigi degisken degil,
    # sklearn modellerinin sayisal girdi zorunlulugu.
    chain += [
        MultiValueFrequencyEncoder(columns=MULTI_VALUE_FREQ_COLS),
        FrequencyEncoder(columns=SINGLE_VALUE_FREQ_COLS),
        NominalOneHotEncoder(columns=NOMINAL_COLS),
    ]

    for tr in chain:
        tr.fit(train_df)
        train_df = tr.transform(train_df)
        test_df = tr.transform(test_df)

    num_cols = [c for c in train_df.select_dtypes(include="number").columns if c != "Label"]
    if train_df[num_cols].isna().any().any():
        neutral = SimpleImputer(strategy="median")
        neutral.fit(train_df[num_cols])
        train_df[num_cols] = neutral.transform(train_df[num_cols])
        test_df[num_cols] = neutral.transform(test_df.reindex(columns=num_cols))

    if step >= 4:
        continuous_cols = [c for c in num_cols if not c.endswith("_missing") and c != "al_all_missing"]
        if scaler_method != "none":
            scaler = ColumnwiseScaler(columns=continuous_cols, method=scaler_method)
            scaler.fit(train_df)
            train_df = scaler.transform(train_df)
            test_df = scaler.transform(test_df)

    if step >= 5:
        pool = _load_stable_pool()
        keep = [c for c in pool if c in train_df.columns]
        feature_cols = list(dict.fromkeys(keep))
    else:
        feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

    X_train = train_df[feature_cols].astype(float)
    X_test = test_df.reindex(columns=feature_cols, fill_value=0).astype(float)
    return X_train, train_df["Label"], X_test, test_df["Label"]


def train_fold_missing_rate(v1_df, al_columns, train_ids):
    """Görev 6 için: yalnızca eğitim fold'undaki eksiklik oranı (fold-güvenli)."""
    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)]
    return train_df[al_columns].isna().mean()
