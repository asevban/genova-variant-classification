"""Asama E1 -- PDR'nin XGBoost(MCC-ayarli) PAH modelini, PDR'de belgelenen
SABIT hiperparametrelerle, `data/splits/pah/` split bankasinin her dis
fold'unda yeniden olcer. Hicbir hiperparametre aramasi yapilmaz (ic fold
kullanilmaz) -- bu bir baseline olcumudur, optimizasyon degildir.

Orijinal PDR model dosyasi/agirliklari elde degil, bu yuzden burada
uretilen model "yeniden egitilmis tarihsel referans"tir, "orijinal PDR
modeli" degildir.

Calistirma: python -m genova.pah.e1_baseline
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah.missingness import BlockMissingIndicator, MedianImputerWithIndicator, ConstantFillImputer
from genova.pah.dataset_versions import EK_BLOCK_COLS, CAT_BLOCK_COLS

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
MODELS_DIR = ROOT / "models" / "pah"

PDR_THRESHOLD = 0.359
CAT_COLS = ["CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "AA_1", "AA_2"]
N_REPEATS = 10
MINORITY_WEIGHT = 2.0

# PDR'de belgelenen sabit hiperparametreler -- bu adimda aranmaz, degistirilmez.
FIXED_HYPERPARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5,
    random_state=42,
    tree_method="hist",
    enable_categorical=True,
)


def _refit_categoricals(train_df, test_df, columns):
    """Kategori sozlugunu yalnizca train_df'ten cikarir; test_df'te
    gorulmeyen bir kategori NaN'a (XGBoost'un native eksik-deger yolu)
    eslenir -- boylece kategori sozlugu de sizinti kaynagi olmaz.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in columns:
        train_values = train_df[col].astype(object)
        categories = sorted(v for v in train_values.dropna().unique())
        train_df[col] = pd.Categorical(train_values, categories=categories)
        test_values = test_df[col].astype(object)
        test_values = test_values.where(test_values.isin(categories), other=np.nan)
        test_df[col] = pd.Categorical(test_values, categories=categories)
    return train_df, test_df


def build_v1_features(v1_df, train_ids, test_ids):
    """v1 spesifikasyonu: ham NaN (AL_/EK_), native kategorik (CAT_/AA_),
    eksiklik-farkinda gosterge yok. Kategori sozlugu train fold'undan.
    """
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    ek_cols = [c for c in v1_df.columns if c.startswith("EK_")]
    feature_cols = al_cols + ek_cols + CAT_COLS

    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    test_df = v1_df[v1_df["Variant_ID"].isin(test_ids)].reset_index(drop=True)
    train_df, test_df = _refit_categoricals(train_df, test_df, CAT_COLS)
    return train_df[feature_cols], train_df["Label"], test_df[feature_cols], test_df["Label"], feature_cols


def build_v2_features(v1_df, train_ids, test_ids):
    """v2 spesifikasyonu: v1 + al_all_missing + ek_cat_block_missing
    blok gostergeleri + EK_3 medyan-doldurma+gosterge + AL_ sifir-doldurma;
    diger EK_ kolonlari ham NaN kalir (dataset_versions.build_v2 ile ayni
    davranis). Tum adimlar yalnizca train fold'unda fit edilir.
    """
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    ek_cols = [c for c in v1_df.columns if c.startswith("EK_")]

    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    test_df = v1_df[v1_df["Variant_ID"].isin(test_ids)].reset_index(drop=True)

    steps = [
        BlockMissingIndicator(columns=al_cols, name="al_all_missing"),
        BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing"),
        MedianImputerWithIndicator(columns=["EK_3"]),
        ConstantFillImputer(columns=al_cols, strategy="zero"),
    ]
    for step in steps:
        step.fit(train_df)
        train_df = step.transform(train_df)
        test_df = step.transform(test_df)

    train_df, test_df = _refit_categoricals(train_df, test_df, CAT_COLS)
    feature_cols = al_cols + ek_cols + CAT_COLS + ["al_all_missing", "ek_cat_block_missing", "EK_3_missing"]
    return train_df[feature_cols], train_df["Label"], test_df[feature_cols], test_df["Label"], feature_cols


def _minority_sample_weight(y_train, minority_weight=MINORITY_WEIGHT):
    """"Azinlik sinif agirligi x2.0": train fold'undaki azinlik sinifinin
    (PAH'ta beklenen: Label=0/benign) her satirina 2.0, cogunluk sinifina
    1.0 agirlik verir. XGBoost'un scale_pos_weight'i kullanilmaz çünkü o
    pozitif (Label=1) sinifi olceklendirir -- burada azinlik cogunlukla
    Label=0, yani anlami farkli olurdu.
    """
    minority_class = y_train.value_counts().idxmin()
    return np.where(y_train == minority_class, minority_weight, 1.0), minority_class


def evaluate_outer_folds(v1_df, feature_builder, label):
    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for fold in outer["folds"]:
            X_train, y_train, X_test, y_test, _ = feature_builder(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
            weights, minority_class = _minority_sample_weight(y_train)
            model = XGBClassifier(**FIXED_HYPERPARAMS)
            model.fit(X_train, y_train, sample_weight=weights)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= PDR_THRESHOLD).astype(int)
            rows.append({
                "data_version": label,
                "repeat": repeat_idx,
                "outer_fold": fold["fold"],
                "n_train": len(y_train),
                "n_test": len(y_test),
                "minority_class": int(minority_class),
                "f1": f1_binary_positive(y_test, y_pred),
                "mcc": matthews_correlation_coefficient(y_test, y_pred),
                "specificity": specificity(y_test, y_pred),
            })
    return pd.DataFrame(rows)


def summarize(results_df):
    summary = {}
    for metric in ("f1", "mcc", "specificity"):
        summary[metric] = {
            "mean": float(results_df[metric].mean()),
            "std": float(results_df[metric].std()),
            "min": float(results_df[metric].min()),
            "max": float(results_df[metric].max()),
        }
    return summary


def fit_full_model(v1_df, feature_builder):
    """Secilen recete ile TUM 369 satirda (dis test bolmesi yok) fit
    edilen dondurulmus referans model -- E2+'da karsilastirma icin."""
    all_ids = v1_df["Variant_ID"].tolist()
    X_full, y_full, _, _, feature_cols = feature_builder(v1_df, all_ids, all_ids)
    weights, minority_class = _minority_sample_weight(y_full)
    model = XGBClassifier(**FIXED_HYPERPARAMS)
    model.fit(X_full, y_full, sample_weight=weights)
    return model, feature_cols, int(minority_class)


def main():
    v1_df = pd.read_parquet(V1_PATH)

    results_v1 = evaluate_outer_folds(v1_df, build_v1_features, "v1")
    results_v2 = evaluate_outer_folds(v1_df, build_v2_features, "v2")

    print("=== v1 (ham NaN, native kategorik) ===")
    print(summarize(results_v1))
    print("=== v2 (eksiklik-farkinda, AL_ sifir-dolu) ===")
    print(summarize(results_v2))

    chosen_label = "v1"
    chosen_builder = build_v1_features
    model, feature_cols, minority_class = fit_full_model(v1_df, chosen_builder)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "data_version": chosen_label,
        "feature_columns": feature_cols,
        "categorical_columns": CAT_COLS,
        "threshold": PDR_THRESHOLD,
        "hyperparameters": FIXED_HYPERPARAMS,
        "minority_class": minority_class,
        "minority_weight": MINORITY_WEIGHT,
        "outer_cv_results_v1": results_v1,
        "outer_cv_results_v2": results_v2,
        "summary_v1": summarize(results_v1),
        "summary_v2": summarize(results_v2),
        "label": "yeniden egitilmis tarihsel referans (orijinal PDR model dosyasi degil)",
    }
    with open(MODELS_DIR / "baseline_frozen.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"kaydedildi: {MODELS_DIR / 'baseline_frozen.pkl'} (secilen veri versiyonu: {chosen_label})")


if __name__ == "__main__":
    main()
