"""Asama E6-EK: RF/v4_from_v2 eklenmis genisletilmis aday havuzuyla
ensemble yeniden degerlendirmesi. Adim 1 -- ucu de (CatBoost/v4_from_v2,
LightGBM/v1, RF/v4_from_v2) SLD-duzeltilmis dis-test olasiliklarindan
3x3 OOF korelasyon matrisi (orijinal E6'nin 2 uyeli matrisinden farkli,
RF dahil). Adim 2 -- yalnizca en dusuk korele ikili icin (bulgu: LightGBM/
v1 + RF/v4_from_v2) ayni 3 blend yontemi + nested E4(SLD)+E5(esik).
Adim 3 -- en iyi blend'i en iyi solo modele karsi paired karsilastirir.

`process_ensemble_fold`, orijinal `e6_ensemble.py`'nin ayni fonksiyonunun
GENELLESTIRILMIS hali -- `fit_predict_a`/`fit_predict_b` artik sabit
(catboost/lgbm) degil, parametre. Orijinal `e6_ensemble.py` DEGISTIRILMEDI
(hala CatBoost+LightGBM icin calisir durumda, kendi ciktilari korunuyor);
bu modul E6-EK'in kendi (yeni, ayri) ciktilarini uretir.

Calistirma: python -m genova.pah.e6ek_ensemble
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, f1_binary_positive_weighted, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah import e2ek_models as em
from genova.pah.calibration import BetaCalibrator
from genova.pah.e3_calibration_run import cross_fit_probabilities, _lookup_best_params
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.e5_threshold_selection import select_threshold, _prior_weights, FINAL_PATHOGENIC_PRIOR
from genova.pah.e6_ensemble import _normalized_rank, _combine, compute_oof_correlation_matrix

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
E3_OOF_CSV = ROOT / "reports" / "tables" / "e3_calibrated_oof_predictions.csv"
E4_CSV = ROOT / "reports" / "tables" / "e4_prior_corrected_probabilities.csv"
CORR_OUT = ROOT / "reports" / "tables" / "e6ek_oof_correlation_matrix.csv"
ENSEMBLE_OUT = ROOT / "reports" / "tables" / "e6ek_ensemble_metrics.csv"

N_REPEATS = 10
ALPHA_GRID = np.round(np.arange(0.0, 1.001, 0.1), 2)
WEIGHTING_VARIANT = "B_fixed_minority_2x"

MEMBER_A = ("lightgbm", "v1")
MEMBER_B = ("random_forest", "v4_from_v2")


def _score_combo(combo_scores, y):
    sld_scores = sld_correct(combo_scores)
    threshold, inner_best_wf1 = select_threshold(sld_scores, y)
    return threshold, inner_best_wf1, sld_scores


def process_ensemble_fold(v1_df, comparison_df, e3_beta_a, e3_beta_b, builder_a, builder_b,
                           fit_predict_a, fit_predict_b, repeat_idx, outer_fold_idx):
    """`e6_ensemble.py::process_ensemble_fold`'un genellestirilmis hali --
    ayni mantik, `fit_predict_a`/`fit_predict_b` artik parametre."""
    best_params_a = _lookup_best_params(comparison_df, *MEMBER_A, repeat_idx, outer_fold_idx)
    best_params_b = _lookup_best_params(comparison_df, *MEMBER_B, repeat_idx, outer_fold_idx)

    proba_a, y_by_vid_a = cross_fit_probabilities(v1_df, builder_a, fit_predict_a, best_params_a, repeat_idx, outer_fold_idx)
    proba_b, y_by_vid_b = cross_fit_probabilities(v1_df, builder_b, fit_predict_b, best_params_b, repeat_idx, outer_fold_idx)
    assert set(proba_a.keys()) == set(proba_b.keys()), "iki uyenin dis-train Variant_ID kumeleri hizali degil"

    vids = list(proba_a.keys())
    y_inner = np.array([y_by_vid_a[v] for v in vids])
    raw_inner_a = np.array([proba_a[v] for v in vids])
    raw_inner_b = np.array([proba_b[v] for v in vids])

    cal_a = BetaCalibrator().fit(raw_inner_a, y_inner)
    cal_b = BetaCalibrator().fit(raw_inner_b, y_inner)
    calibrated_inner_a = cal_a.transform(raw_inner_a)
    calibrated_inner_b = cal_b.transform(raw_inner_b)

    test_a = e3_beta_a[(e3_beta_a.repeat == repeat_idx) & (e3_beta_a.outer_fold == outer_fold_idx)].set_index("variant_id")
    test_b = e3_beta_b[(e3_beta_b.repeat == repeat_idx) & (e3_beta_b.outer_fold == outer_fold_idx)].set_index("variant_id")
    test_vids = test_a.index.tolist()
    assert test_vids == test_b.index.tolist() or set(test_vids) == set(test_b.index), "dis-test Variant_ID kumeleri hizali degil"
    test_b = test_b.loc[test_vids]
    y_test = test_a["y_true"].to_numpy()
    calibrated_test_a = test_a["proba"].to_numpy()
    calibrated_test_b = test_b["proba"].to_numpy()

    rows = []
    for method in ("simple_avg", "rank_avg", "weighted"):
        if method == "weighted":
            best_alpha, best_inner_wf1, best_thr = None, -1, None
            for alpha in ALPHA_GRID:
                combo_inner = _combine(calibrated_inner_a, calibrated_inner_b, "weighted", alpha)
                thr, inner_wf1, _ = _score_combo(combo_inner, y_inner)
                if inner_wf1 > best_inner_wf1:
                    best_inner_wf1, best_alpha, best_thr = inner_wf1, alpha, thr
            alpha, threshold, inner_best_wf1 = best_alpha, best_thr, best_inner_wf1
        else:
            combo_inner = _combine(calibrated_inner_a, calibrated_inner_b, method)
            threshold, inner_best_wf1, _ = _score_combo(combo_inner, y_inner)
            alpha = np.nan

        combo_test = _combine(calibrated_test_a, calibrated_test_b, method, alpha if method == "weighted" else None)
        sld_test = sld_correct(combo_test)
        pred = (sld_test >= threshold).astype(int)
        target_weights = _prior_weights(y_test, FINAL_PATHOGENIC_PRIOR)

        rows.append({
            "blend_method": method, "alpha": alpha,
            "repeat": repeat_idx, "outer_fold": outer_fold_idx,
            "chosen_threshold": threshold, "inner_best_weighted_f1": inner_best_wf1,
            "f1": f1_binary_positive(y_test, pred),
            "mcc": matthews_correlation_coefficient(y_test, pred),
            "specificity": specificity(y_test, pred),
            "weighted_f1": f1_binary_positive_weighted(y_test, pred, target_weights),
            "n_dis_train": len(vids), "n_dis_test": len(y_test),
        })
    return rows


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    comparison_b_only = comparison_df[comparison_df.weighting_variant == WEIGHTING_VARIANT]
    e4_df = pd.read_csv(E4_CSV)
    e3_oof = pd.read_csv(E3_OOF_CSV)

    print("=== Adim 1: 3x3 OOF korelasyon matrisi (CatBoost/v4_from_v2, LightGBM/v1, RF/v4_from_v2) ===")
    corr = compute_oof_correlation_matrix(e4_df)
    CORR_OUT.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(CORR_OUT)
    print(corr.round(4))
    print(f"kaydedildi: {CORR_OUT}")

    e3_beta_a = e3_oof[(e3_oof.model == MEMBER_A[0]) & (e3_oof.data_version == MEMBER_A[1]) & (e3_oof.method == "beta")]
    e3_beta_b = e3_oof[(e3_oof.model == MEMBER_B[0]) & (e3_oof.data_version == MEMBER_B[1]) & (e3_oof.method == "beta")]

    pool = json.loads(POOL_PATH.read_text())["features"]

    def builder_a(df, tr, te):
        return fv.build_v1(df, tr, te)

    def builder_b(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    print(f"\n=== Adim 2: ensemble ({MEMBER_A} + {MEMBER_B}), 50 dis fold, 3 blend yontemi ===")
    all_rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            rows = process_ensemble_fold(
                v1_df, comparison_b_only, e3_beta_a, e3_beta_b, builder_a, builder_b,
                m.fit_predict_lgbm, em.fit_predict_rf,
                repeat_idx, outer_fold["fold"],
            )
            all_rows.extend(rows)
            print(f"  repeat={repeat_idx} outer={outer_fold['fold']} done", flush=True)

    result = pd.DataFrame(all_rows)
    result.to_csv(ENSEMBLE_OUT, index=False)
    print(f"\nkaydedildi: {ENSEMBLE_OUT}")
    print(result.groupby("blend_method")[["weighted_f1", "f1", "chosen_threshold"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
