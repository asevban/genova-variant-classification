"""Asama E2b: literatur taramasindan gelen iki deneysel aday ozellik seti
(ZORT_* ve AL_ ozet-istatistikleri). Ikisi de fold-guvenli (istatistikleri
yalnizca outer/inner train'den ogrenir) POST-PROCESSING adimlaridir --
`fold_versions.py`'nin resmi versiyonlarinin UZERINE eklenir, resmi
v4_final_feature_pool.json'i degistirmez. Yalnizca en iyi cikan model
ailesiyle (Asama 2a sonucuna gore CatBoost) denenir.

ZORT_* mantigi `experiments/tam_16_adim_deneme/scripts/step11_12_transform_
candidates.py::add_zscore_and_averaged_features`'in birebir ayni --
o script hicbir hesaplama yapmiyor, yalnizca bu fonksiyonu TANIMLIYOR ve
"Adim 13 tarafindan import edilir" diyor; burada production tarafinda ayni
mantik yeniden yazildi (deney klasorune E-F kesin kisitlari geregi yazma
yapilamiyor/import edilemiyor, yalnizca referans olarak okunuyor).
"""
import pandas as pd

from genova.pah.transforms import classify_al_columns

ZORT_PAIRS = [
    ("AL_88", "AL_121", "ZORT_AL88_AL121"),
    ("EK_7", "EK_9", "ZORT_EK7_EK9"),
    ("AL_23", "AL_283", "ZORT_AL23_AL283"),
    ("AL_7", "AL_103", "ZORT_AL7_AL103"),
]


def add_zort_features(X_train, X_test, pairs=ZORT_PAIRS):
    """ZORT_X_Y = ortalama(Z(X), Z(Y)); Z-skor istatistikleri (ortalama/std)
    yalnizca X_train'den ogrenilir, X_test'e yalnizca uygulanir."""
    X_train_ext, X_test_ext = X_train.copy(), X_test.copy()
    for c1, c2, name in pairs:
        if c1 not in X_train.columns or c2 not in X_train.columns:
            continue
        mean1, std1 = X_train[c1].mean(), X_train[c1].std()
        mean2, std2 = X_train[c2].mean(), X_train[c2].std()
        std1 = std1 if std1 > 1e-9 else 1.0
        std2 = std2 if std2 > 1e-9 else 1.0
        X_train_ext[name] = ((X_train[c1] - mean1) / std1 + (X_train[c2] - mean2) / std2) / 2
        X_test_ext[name] = ((X_test[c1] - mean1) / std1 + (X_test[c2] - mean2) / std2) / 2
    return X_train_ext, X_test_ext


def add_al_summary_features(X_train, X_test, al_cols):
    """Satir-bazli AL_ ozet istatistikleri: max/min/ortalama allel frekansi
    + gozlenen (dolu) AL_ kolon sayisi -- ACMG BA1/BS1/PM2 kriterlerinin
    dogrudan sayisal karsiligi (bkz. 07_LITERATUR_TARAMASI raporu).

    `al_cols` yalnizca gercekten [0,1] frekans-olcekli kolonlarla
    sinirlanmalidir (`classify_al_columns`'in ratio_type + frequency_type
    birlesimi) -- 334 AL_ kolonunun 90'i "constant" (bazilari 0/1
    frekansindan tamamen farkli, ornegin ~264690 gibi buyuk bir sabit
    tasiyabilir) ve bunlari dahil etmek satir-bazli max/min/ortalamayi o
    sabitin domine etmesine yol acar -- bu yuzden cagiran taraf
    (`build_v1_plus_al_summary`) al_cols'u yalnizca train fold'unda
    `classify_al_columns` ile fit edip filtreler.

    Tum AL_ blogu eksik olan satirlarda (al_all_missing=1) max/min/ortalama
    NaN cikar; bu artik-NaN'lar YALNIZCA train fold medyaniyla doldurulur.
    """
    X_train_ext, X_test_ext = X_train.copy(), X_test.copy()
    al_train, al_test = X_train[al_cols], X_test[al_cols]

    X_train_ext["AL_summary_max"] = al_train.max(axis=1, skipna=True)
    X_train_ext["AL_summary_min"] = al_train.min(axis=1, skipna=True)
    X_train_ext["AL_summary_mean"] = al_train.mean(axis=1, skipna=True)
    X_train_ext["AL_summary_n_observed"] = al_train.notna().sum(axis=1)

    X_test_ext["AL_summary_max"] = al_test.max(axis=1, skipna=True)
    X_test_ext["AL_summary_min"] = al_test.min(axis=1, skipna=True)
    X_test_ext["AL_summary_mean"] = al_test.mean(axis=1, skipna=True)
    X_test_ext["AL_summary_n_observed"] = al_test.notna().sum(axis=1)

    for col in ("AL_summary_max", "AL_summary_min", "AL_summary_mean"):
        train_median = X_train_ext[col].median()
        X_train_ext[col] = X_train_ext[col].fillna(train_median)
        X_test_ext[col] = X_test_ext[col].fillna(train_median)
    return X_train_ext, X_test_ext


def build_v1_plus_zort(v1_df, train_ids, test_ids, base_builder):
    X_train, y_train, X_test, y_test = base_builder(v1_df, train_ids, test_ids)
    X_train, X_test = add_zort_features(X_train, X_test)
    return X_train, y_train, X_test, y_test


def build_v1_plus_al_summary(v1_df, train_ids, test_ids, base_builder, al_cols):
    """`al_cols` -- tam 334 AL_ kolon listesi (ham). Ozet-uygun alt-kume
    (ratio_type + frequency_type, "constant" haric) yalnizca train fold'unda
    `classify_al_columns` ile belirlenir -- fold-guvenli."""
    X_train, y_train, X_test, y_test = base_builder(v1_df, train_ids, test_ids)
    al_types = classify_al_columns(X_train, al_cols)
    summary_input_cols = al_types["ratio_type"] + al_types["frequency_type"]
    X_train, X_test = add_al_summary_features(X_train, X_test, summary_input_cols)
    return X_train, y_train, X_test, y_test
