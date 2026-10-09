"""Caraz-panel karsilastirmasi Eksen 3: CFTR'nin "mutlak esik-alti fold
sayma" tarzini PAH'a getirir. HICBIR yeni tahmin/egitim/arama yapmaz --
yalnizca mevcut fold-duzeyi sayilarin (`e5_threshold_selection.csv`'nin
mcc_chosen/sensitivity_chosen/specificity_chosen kolonlari, Revize
turunda eklendi) SAYILMASI + F0 final modelin kendi 5-fold ic OOF'unun
(f0_final_model.py::cross_fit_oof ile AYNI, deterministik, split
bankasindan BAGIMSIZ 5-fold) fold-bazinda kirilimi.

Iki eşik seti:
  1. CFTR'nin kendi esikleri (birebir karsilastirma icin): MCC<0,40,
     Sensitivity<0,85, Specificity<0,70.
  2. PAH-kalibreli esikler: E5'in 4 final adayinin havuzlanmis (4x50=200
     gozlem) fold-duzeyi dagilimindan HER METRIK icin AYRI 25. persentil
     -- "bu adaylarin kendi tipik performansina gore kotu sayilan alt-
     ceyrek" anlamina gelir. F0 de AYNI (E5'ten turetilen) esiklere karsi
     olculur -- kendi esigini kendi belirlemiyor.

Calistirma: python -m genova.pah.g2_fold_fragility_counting
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import matthews_correlation_coefficient, sensitivity, specificity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.f0_final_model import F0_SEED, V1_PATH, fit_predict_catboost

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models" / "pah"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"
OUT_CSV = TAB_DIR / "14_FOLD_KIRILGANLIK_SAYIMI.csv"

CFTR_THRESHOLDS = {"mcc": 0.40, "sensitivity": 0.85, "specificity": 0.70}

E5_CANDIDATES = [
    ("catboost", "v1", "CatBoost/v1 (E5)"),
    ("catboost", "v4_from_v2", "CatBoost/v4_from_v2 (E5, final aday)"),
    ("lightgbm", "v1", "LightGBM/v1 (E5)"),
    ("random_forest", "v4_from_v2", "Random Forest/v4_from_v2 (E5-EK)"),
]


def _f0_per_fold_scores():
    """f0_final_model.py::cross_fit_oof ile AYNI 5-fold yeniden uretilir
    (deterministik, split bankasindan bagimsiz, F0_SEED+1) -- yalnizca
    fold uyeligi de saklanarak. Bundle'in KENDI fit edilmis kalibratoru/
    w1/w0/esigi (tum-OOF uzerinde fit edilmis) her fold'un ham
    olasiligina uygulanir -- hicbir yeniden fit yok."""
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(MODEL_DIR / "final_model_bundle_v2.pkl")
    pool, best_params = bundle["pool"], bundle["model_best_params"]

    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        raw_proba = fit_predict_catboost(X_tr, y_tr, X_va, best_params)
        calibrated = bundle["calibrator"].transform(raw_proba)
        sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
        pred = (sld >= bundle["threshold"]).astype(int)
        y_va_arr = y_va.to_numpy() if hasattr(y_va, "to_numpy") else np.asarray(y_va)
        rows.append({
            "fold": fold["fold"],
            "mcc": matthews_correlation_coefficient(y_va_arr, pred),
            "sensitivity": sensitivity(y_va_arr, pred),
            "specificity": specificity(y_va_arr, pred),
        })
    return pd.DataFrame(rows)


def _pah_calibrated_thresholds(e5_df):
    pooled = e5_df[e5_df.apply(lambda r: (r["model"], r["data_version"]) in
                                [(m, v) for m, v, _ in E5_CANDIDATES], axis=1)]
    return {
        "mcc": pooled["mcc_chosen"].quantile(0.25),
        "sensitivity": pooled["sensitivity_chosen"].quantile(0.25),
        "specificity": pooled["specificity_chosen"].quantile(0.25),
    }


def _count_below(values, threshold):
    return int((values < threshold).sum())


def main():
    e5_df = pd.read_csv(E5_CSV)
    pah_thresholds = _pah_calibrated_thresholds(e5_df)

    print("=== Esik setleri ===")
    print(f"CFTR: {CFTR_THRESHOLDS}")
    print(f"PAH-kalibreli (E5'in 4 adayinin havuzlanmis 200-fold dagiliminin 25. persentili): "
          f"{ {k: round(v, 4) for k, v in pah_thresholds.items()} }")

    rows = []
    for model_name, version_name, label in E5_CANDIDATES:
        sub = e5_df[(e5_df.model == model_name) & (e5_df.data_version == version_name)]
        assert len(sub) == 50, f"{label}: 50 fold bekleniyordu, {len(sub)} bulundu"
        for threshold_set_name, thresholds in (("CFTR", CFTR_THRESHOLDS), ("PAH-kalibreli", pah_thresholds)):
            rows.append({
                "Aday": label, "Esik_seti": threshold_set_name, "n_fold": len(sub),
                "MCC_alti_fold": _count_below(sub["mcc_chosen"], thresholds["mcc"]),
                "Sens_alti_fold": _count_below(sub["sensitivity_chosen"], thresholds["sensitivity"]),
                "Spec_alti_fold": _count_below(sub["specificity_chosen"], thresholds["specificity"]),
            })

    print("\n=== F0 final modelin kendi 5-fold ic OOF'u yeniden uretiliyor (deterministik) ===")
    f0_scores = _f0_per_fold_scores()
    print(f0_scores.to_string(index=False))
    for threshold_set_name, thresholds in (("CFTR", CFTR_THRESHOLDS), ("PAH-kalibreli", pah_thresholds)):
        rows.append({
            "Aday": "[FINAL] CatBoost (F0)", "Esik_seti": threshold_set_name, "n_fold": len(f0_scores),
            "MCC_alti_fold": _count_below(f0_scores["mcc"], thresholds["mcc"]),
            "Sens_alti_fold": _count_below(f0_scores["sensitivity"], thresholds["sensitivity"]),
            "Spec_alti_fold": _count_below(f0_scores["specificity"], thresholds["specificity"]),
        })

    result = pd.DataFrame(rows, columns=["Aday", "Esik_seti", "n_fold", "MCC_alti_fold", "Sens_alti_fold", "Spec_alti_fold"])
    result["Toplam_kirilgan_fold_isareti"] = result["MCC_alti_fold"] + result["Sens_alti_fold"] + result["Spec_alti_fold"]
    print("\n=== Sayim tablosu ===")
    print(result.to_string(index=False))
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
