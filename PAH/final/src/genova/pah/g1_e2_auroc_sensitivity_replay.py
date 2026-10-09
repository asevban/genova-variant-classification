"""Revize turu, Adim 1: `e2_model_comparison.csv`'nin "cekirdek matris"
16 kombinasyonu (06_MODEL_SECIM_RAPORU_PAH.md'nin E2 karsilastirma
tablosundaki 16 satir -- weighting_variant="B_fixed_minority_2x",
RF/KNN/A-C agirliklandirma genisletmeleri (E2-EK/madde 9) DAHIL DEGIL)
icin AUROC + Sensitivity ekler.

HICBIR hiperparametre yeniden ARANMIYOR -- her (model, data_version,
repeat, outer_fold) satiri icin CSV'de zaten kayitli `best_params`
JSON'u okunup AYNEN kullanilarak tek bir fit/predict yapiliyor (REPLAY,
arama degil) -- `nested_cv_evaluate`'in ic-fold arama adiminin AYNISI
DEGIL, yalnizca dis-train/dis-test'in son adimi. Ayni seed (RANDOM_
STATE=42) + ayni best_params + ayni ozellik matrisi -> f1/mcc/auprc/
specificity BIT-BIT AYNI cikmasi beklenir (regresyon testiyle
dogrulaniyor); auroc + sensitivity ise sabit esik=0,5 rejiminde (E2'nin
DECISION_THRESHOLD'uyla ayni) YENI kolonlar olarak ekleniyor.

`e2_model_comparison.csv`'ye YALNIZCA `auroc`/`sensitivity` kolonlari
eklenir -- mevcut f1/mcc/auprc/specificity/best_params sayilari
DEGISTIRILMEZ. Kapsam disi (RF/KNN/A-C agirlik) satirlar icin bu iki
kolon NaN kalir.

Calistirma: python -m genova.pah.g1_e2_auroc_sensitivity_replay
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import sensitivity
from genova.pah import fold_versions as fv
from genova.pah import e2_candidate_features as cf
from genova.pah import models as m

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
AL_CLUSTERS_PATH = ROOT / "reports" / "tables" / "AL_correlation_clusters.csv"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"

DECISION_THRESHOLD = 0.5  # E2'nin sabit esik rejimiyle ayni.

# 16 "cekirdek matris" kombinasyonu -- 06_MODEL_SECIM_RAPORU_PAH.md'nin
# "Tam karsilastirma tablosu" bolumundeki satirlarla BIREBIR ayni.
CORE_COMBOS = [
    ("catboost", "v1"), ("catboost", "v1_plus_zort"), ("catboost", "v1_plus_al_summary"),
    ("catboost", "v2_raw_nan"), ("catboost", "v4_from_v2"), ("catboost", "v2"),
    ("lightgbm", "v1"), ("xgboost", "v1"),
    ("lightgbm", "v2_raw_nan"), ("lightgbm", "v4_from_v2"),
    ("xgboost", "v2_raw_nan"), ("lightgbm", "v2"),
    ("xgboost", "v4_from_v2"), ("xgboost", "v2"),
    ("elasticnet", "v4_from_v3"), ("elasticnet", "v3"),
]


def _load_outer_fold(repeat_idx, fold_idx):
    outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
    return outer["folds"][fold_idx]


def main():
    v1_df = pd.read_parquet(V1_PATH)
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    pool = json.loads(POOL_PATH.read_text())["features"]
    al_clusters = pd.read_csv(AL_CLUSTERS_PATH)

    builders = {
        "v1": lambda df, tr, te: fv.build_v1(df, tr, te),
        "v2": lambda df, tr, te: fv.build_v2(df, tr, te),
        "v2_raw_nan": lambda df, tr, te: fv.build_v2_raw_nan(df, tr, te),
        "v4_from_v2": lambda df, tr, te: fv.build_v4_from_v2(df, tr, te, pool),
        "v3": lambda df, tr, te: fv.build_v3_full(df, tr, te, al_clusters),
        "v4_from_v3": lambda df, tr, te: fv.build_v4_from_v3(df, tr, te, pool),
        "v1_plus_zort": lambda df, tr, te: cf.build_v1_plus_zort(df, tr, te, fv.build_v1),
        "v1_plus_al_summary": lambda df, tr, te: cf.build_v1_plus_al_summary(df, tr, te, fv.build_v1, al_cols),
    }
    fit_predict_fns = {
        "catboost": m.fit_predict_catboost, "xgboost": m.fit_predict_xgb,
        "lightgbm": m.fit_predict_lgbm, "elasticnet": m.fit_predict_elasticnet,
    }

    comparison_df = pd.read_csv(COMPARISON_CSV)
    original_snapshot = comparison_df[["model", "data_version", "weighting_variant", "repeat", "outer_fold",
                                        "f1", "mcc", "auprc", "specificity"]].copy()

    new_cols = {}
    n_done, n_total = 0, sum(
        len(comparison_df[(comparison_df.model == mn) & (comparison_df.data_version == dv)
                           & (comparison_df.weighting_variant == "B_fixed_minority_2x")])
        for mn, dv in CORE_COMBOS
    )
    for model_name, version_name in CORE_COMBOS:
        rows = comparison_df[(comparison_df.model == model_name) & (comparison_df.data_version == version_name)
                              & (comparison_df.weighting_variant == "B_fixed_minority_2x")]
        builder = builders[version_name]
        fit_predict_fn = fit_predict_fns[model_name]
        for idx, row in rows.iterrows():
            repeat_idx, outer_fold_idx = int(row["repeat"]), int(row["outer_fold"])
            best_params = json.loads(row["best_params"])
            outer_fold = _load_outer_fold(repeat_idx, outer_fold_idx)
            X_tr, y_tr, X_te, y_te = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            proba = fit_predict_fn(X_tr, y_tr, X_te, best_params)
            y_pred = (proba >= DECISION_THRESHOLD).astype(int)
            new_cols[idx] = {"auroc": roc_auc_score(y_te, proba), "sensitivity": sensitivity(y_te, y_pred)}
            n_done += 1
            print(f"  [{n_done}/{n_total}] {model_name}/{version_name} repeat={repeat_idx} outer={outer_fold_idx} "
                  f"auroc={new_cols[idx]['auroc']:.4f} sens={new_cols[idx]['sensitivity']:.4f}", flush=True)

    comparison_df["auroc"] = pd.Series({i: v["auroc"] for i, v in new_cols.items()})
    comparison_df["sensitivity"] = pd.Series({i: v["sensitivity"] for i, v in new_cols.items()})

    unchanged = comparison_df.loc[original_snapshot.index,
                                   ["model", "data_version", "weighting_variant", "repeat", "outer_fold",
                                    "f1", "mcc", "auprc", "specificity"]]
    pd.testing.assert_frame_equal(unchanged, original_snapshot)
    print("\nGUVENLIK KONTROLU GECTI: mevcut f1/mcc/auprc/specificity degerleri REPLAY sonrasi degismedi.")

    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"kaydedildi: {COMPARISON_CSV} ({n_done} satira auroc/sensitivity eklendi, kapsam disi satirlar NaN)")


if __name__ == "__main__":
    main()
