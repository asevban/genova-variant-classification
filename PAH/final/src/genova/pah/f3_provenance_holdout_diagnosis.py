"""Saf teshis turu: F3'un `CAT_1` bos->dolu gecis testinin (raw AUC
0,831->0,573) ne kadar GENEL oldugunu UC farkli acidan olcer:

  1. `CAT_2` leave-one-category-out (kaynak-bazli, 8 kategori: 7
     AllofUs alt-grubu + "EMPTY"/AllofUs-disi).
  2. `AL_` eksiklik-yogunlugu tertile holdout (dusuk/orta/yuksek, 3 tur).
  3. `CAT_1` capraz degerlendirme (iki yon -- F3'un zaten yaptigi testin
     TEKRARI degil, DOGRULAMASI: ayni referans/yontemle, bu modulun
     kendi kod yolundan gectigini teyit eder).

HICBIR MODEL/ESIK/DOSYA DEGISTIRILMEZ -- `final_model_bundle_v2.pkl`/
`predict.py`/split bankasi dokunulmaz, karar kurali YOK, bu SAF teshis.

`f0_final_model.py::cross_fit_oof`/`select_calibrator`/`select_prior_
and_threshold`, `f3_robustness_stress_tests.py::_score`, `f0_weighting_
experiment.py::final_projection_metrics` DEGISTIRILMEDI -- yalnizca
import edilip yeniden kullanildi. `fit_and_score_on_split`in KENDISI
kullanilmadi -- o yalnizca ozet metrik donduruyor, bu tur agirlikli-F1
icin ham proba/pred'e ihtiyac duyuyor; bunun yerine AYNI ic adimlari
(`build_v4_from_v2` -> `fit_predict_catboost` -> `calibrator.
transform` -> `sld_correct`) kucuk, bilincli bir orkestrasyonla tekrar
cagiran `_fit_score_full` yazildi -- mevcut hicbir fonksiyon
degistirilmedi.

Her holdout senaryosu icin (tutulan grubun kalibrasyona SIZMAMASI icin):
  1. Kalan veri UZERINDE `cross_fit_oof` (degismedi, 5-fold) ->
     kalibrator+esik BU kalan-veri OOF'undan secilir.
  2. Kalan verinin TAMAMIYLA TEK bir model fit edilir, tutulan grupta
     test edilir (raw AUC + agirlikli-F1 + final-projeksiyon F1/MCC,
     100P/250B).

Calistirma: python -m genova.pah.f3_provenance_holdout_diagnosis
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive_weighted
from genova.pah import fold_versions as fv
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, select_calibrator, select_prior_and_threshold, cross_fit_oof
from genova.pah.f3_robustness_stress_tests import BUNDLE_PATH, _score
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import _prior_weights
from genova.pah.f0_weighting_experiment import final_projection_metrics

OUT_CSV = TAB_DIR / "f3_provenance_holdout_diagnosis.csv"
LOW_POWER_N = 15


def _fit_score_full(v1_df, pool, best_params, calibrator, threshold, train_ids, test_ids):
    """`f3_robustness_stress_tests.py::fit_and_score_on_split` ile AYNI
    ic adimlar -- yalnizca agirlikli-F1 + final-projeksiyon F1/MCC de
    hesaplaniyor (bkz. modul docstring'i)."""
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    proba = fit_predict_catboost(X_tr, y_tr, X_te, best_params)
    calibrated = calibrator.transform(proba)
    sld = sld_correct(calibrated)
    pred = (sld >= threshold).astype(int)
    y_te_arr = y_te.to_numpy()

    row = {"n_train": len(y_tr)}
    row.update(_score(y_te_arr, pred))
    row["raw_auc"] = roc_auc_score(y_te_arr, proba) if len(set(y_te_arr)) > 1 else float("nan")

    weights = _prior_weights(y_te_arr, FINAL_PATHOGENIC_PRIOR)
    row["weighted_f1"] = f1_binary_positive_weighted(y_te_arr, pred, weights)
    proj = final_projection_metrics(row["sensitivity"], row["specificity"])
    row["proj_f1"], row["proj_mcc"] = proj["f1"], proj["mcc"]
    return row


def evaluate_holdout(v1_df, pool, best_params, train_ids, test_ids):
    reduced_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].reset_index(drop=True)
    oof_proba, y_full = cross_fit_oof(reduced_df, pool, best_params, train_ids)
    winner_name, winner_cal, _ = select_calibrator(oof_proba, y_full)
    calibrated = winner_cal.transform(oof_proba)
    threshold, _ = select_prior_and_threshold(calibrated, y_full)
    row = _fit_score_full(v1_df, pool, best_params, winner_cal, threshold, train_ids, test_ids)
    row["calibrator"], row["threshold"] = winner_name, threshold
    row["low_power"] = row["n_test"] < LOW_POWER_N
    return row


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    print(f"final bundle tarifi: {len(pool)} ozellik, {best_params}", flush=True)

    print("\n=== Referans: normal CV (bundle'in kendi cross_fit_oof'u) ===", flush=True)
    ref_oof, ref_y = cross_fit_oof(v1_df, pool, best_params, all_ids)
    ref_calibrated = bundle["calibrator"].transform(ref_oof)
    ref_sld = sld_correct(ref_calibrated, w1=bundle["w1"], w0=bundle["w0"])
    ref_pred = (ref_sld >= bundle["threshold"]).astype(int)
    ref = _score(ref_y, ref_pred)
    ref["raw_auc"] = roc_auc_score(ref_y, ref_oof)
    ref_weights = _prior_weights(ref_y, FINAL_PATHOGENIC_PRIOR)
    ref["weighted_f1"] = f1_binary_positive_weighted(ref_y, ref_pred, ref_weights)
    print(f"  raw_AUC={ref['raw_auc']:.4f} weighted_F1={ref['weighted_f1']:.4f} "
          f"specificity={ref['specificity']:.4f} sensitivity={ref['sensitivity']:.4f} "
          f"(F3'un kayitli referansi: raw_AUC=0,8309)", flush=True)

    all_rows = []

    print("\n=== Adim 1: CAT_2 leave-one-category-out (kaynak-bazli, 8 kategori) ===", flush=True)
    cat2_group = v1_df["CAT_2"].astype(object).fillna("EMPTY")
    for category in sorted(cat2_group.unique()):
        test_ids = v1_df.loc[cat2_group == category, "Variant_ID"].tolist()
        train_ids = v1_df.loc[cat2_group != category, "Variant_ID"].tolist()
        row = evaluate_holdout(v1_df, pool, best_params, train_ids, test_ids)
        row.update({"test_type": "cat2_leave_one_out", "test_group": category, "test": "CAT_2 (kaynak) leave-one-out"})
        all_rows.append(row)
        power_note = " [DUSUK GUC, n<15]" if row["low_power"] else ""
        print(f"  {category} (n_test={row['n_test']}, benign={row['n_test_benign']}): "
              f"raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f}{power_note}", flush=True)

    print("\n=== Adim 2: AL_ eksiklik-yogunlugu tertile holdout (3 tur, tam veride tanimli tertil) ===", flush=True)
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    al_rate = v1_df[al_cols].isna().mean(axis=1)
    tertile = pd.qcut(al_rate, 3, labels=["dusuk", "orta", "yuksek"])
    for level in ["dusuk", "orta", "yuksek"]:
        test_ids = v1_df.loc[tertile == level, "Variant_ID"].tolist()
        train_ids = v1_df.loc[tertile != level, "Variant_ID"].tolist()
        row = evaluate_holdout(v1_df, pool, best_params, train_ids, test_ids)
        row.update({"test_type": "al_missingness_tertile_holdout", "test_group": level, "test": "AL_ eksiklik tertile holdout"})
        all_rows.append(row)
        print(f"  {level} (n_test={row['n_test']}, benign={row['n_test_benign']}): "
              f"raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f}", flush=True)

    print("\n=== Adim 3: CAT_1 capraz degerlendirme (iki yon) ===", flush=True)
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()

    row = evaluate_holdout(v1_df, pool, best_params, cat1_filled_ids, cat1_empty_ids)
    row.update({"test_type": "cat1_cross", "test_group": "dolu->bos", "test": "CAT_1 capraz (train=dolu, test=bos)"})
    all_rows.append(row)
    print(f"  train=CAT_1-dolu -> test=CAT_1-bos (n_test={row['n_test']}): "
          f"raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f}", flush=True)

    row = evaluate_holdout(v1_df, pool, best_params, cat1_empty_ids, cat1_filled_ids)
    row.update({"test_type": "cat1_cross", "test_group": "bos->dolu (KRITIK)", "test": "CAT_1 capraz (train=bos, test=dolu, KRITIK)"})
    all_rows.append(row)
    print(f"  train=CAT_1-bos -> test=CAT_1-dolu (KRITIK, n_test={row['n_test']}): "
          f"raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f} "
          f"(F3'un kayitli referansi: raw_AUC=0,5727)", flush=True)

    result = pd.DataFrame(all_rows)
    result["ref_raw_auc"] = ref["raw_auc"]
    result["ref_weighted_f1"] = ref["weighted_f1"]
    result["delta_raw_auc"] = result["raw_auc"] - ref["raw_auc"]
    result["delta_weighted_f1"] = result["weighted_f1"] - ref["weighted_f1"]
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== Adim 4: kirilganlik haritasi (ozet) ===", flush=True)
    summary = result[["test", "test_group", "n_test", "raw_auc", "delta_raw_auc", "delta_weighted_f1", "low_power"]]
    print(summary.to_string(index=False), flush=True)

    worst = result.loc[result["delta_raw_auc"].idxmin()]
    print(f"\nEn buyuk raw_AUC dususu: {worst['test']} / {worst['test_group']} "
          f"(delta={worst['delta_raw_auc']:+.4f}, n_test={worst['n_test']}, "
          f"dusuk_guc={worst['low_power']})", flush=True)


if __name__ == "__main__":
    main()
