"""Ortusme teshisinin (`f3_cat1_al_overlap_diagnosis.py`) bulgusuna
dayanan deneme: `CAT_1`-bos kirilganligi ile `AL_`-eksiklik-yogunlugu
YAPISAL olarak neredeyse ayni eksen oldugu icin (0/132 kesisim), bu tur
dogrudan `AL_` eksiklik-desenini HEDEFLEYEN bir missingness-augmentation
dener: egitim sirasinda `AL_` degerleri/bloklari kontrollu bicimde
maskelenerek modelin belirli bir eksiklik-deseni ile etiket arasinda
kisayol kurmasi zorlastirilir. Final tarifin (26 ozellik, `depth=5,
lr=0,05`, Beta, SLD, uyarlanabilir esik) KUCUK bir varyasyonu -- yeni
bir model ailesi degil.

n=19'luk ikincil, aciklanamayan sinyal (AllofUs-bagimsiz, tamami
`CAT_1`-dolu) BU TURUN KAPSAMI DISI -- ayri bir acik soru olarak
kaliyor.

`f0_final_model.py::cross_fit_oof`/`select_calibrator`/`select_prior_
and_threshold`, `f3_robustness_stress_tests.py::_score`,
`models.py::fit_predict_catboost` DEGISTIRILMEDI -- yalnizca import
edilip yeniden kullanildi. Maskeleme SAF BIR VERI DONUSUMU (egitim
X'inin AL_ kolonlarina uygulanir, X_va/X_te HICBIR ZAMAN augment
edilmez) -- `fit_predict_catboost`'un KENDISI degistirilmeden,
maskelenmemis X_va ile birlikte dogrudan cagrilir.

Calistirma: python -m genova.pah.f0_missingness_augmentation
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive_weighted, specificity, sensitivity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import fit_predict_catboost
from genova.pah.e3_calibration_run import _variant_ids_in_row_order
from genova.pah.f0_final_model import (
    V1_PATH, TAB_DIR, F0_SEED,
    cross_fit_oof, select_calibrator, select_prior_and_threshold,
)
from genova.pah.f3_robustness_stress_tests import BUNDLE_PATH, _score
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR, W1, W0
from genova.pah.e5_threshold_selection import _prior_weights
from genova.pah.f0_weighting_experiment import final_projection_metrics
from genova.statistics import nadeau_bengio_corrected_ttest

CANDIDATES = [
    {"name": "mask=0,15 bagimsiz-sutun", "mask_rate": 0.15, "block_mode": False},
    {"name": "mask=0,15 blok", "mask_rate": 0.15, "block_mode": True},
    {"name": "mask=0,30 bagimsiz-sutun", "mask_rate": 0.30, "block_mode": False},
    {"name": "mask=0,30 blok", "mask_rate": 0.30, "block_mode": True},
]

OUT_GRID_CSV = TAB_DIR / "f0_missingness_augmentation_grid.csv"
CAT1_SHIFT_CRITICAL_DELTA_THRESHOLD = 0.10
NOISE_BAND = 0.02


def augment_al_columns(X, al_cols, mask_rate, block_mode, rng):
    """Egitim X'inin AL_ kolonlarina kontrollu maskeleme -- etiketten
    BAGIMSIZ (hem patojenik hem benign satirlara orantili uygulanir,
    y_tr'a hic bakilmaz). `block_mode=False`: her AL_ kolonu bagimsiz
    olarak `mask_rate` olasilikla NaN. `block_mode=True`: satirin TUM
    AL_ kolonlari birlikte `mask_rate` olasilikla NaN (gercek CAT_1-bos
    satirlarin "hepsi birden eksik" desenini taklit eder)."""
    if mask_rate <= 0 or not al_cols:
        return X
    X = X.copy()
    n = len(X)
    if block_mode:
        row_mask = rng.random(n) < mask_rate
        X.loc[X.index[row_mask], al_cols] = np.nan
    else:
        arr = X[al_cols].to_numpy(dtype=float)
        cell_mask = rng.random(arr.shape) < mask_rate
        arr[cell_mask] = np.nan
        X[al_cols] = arr
    return X


def cross_fit_oof_augmented(v1_df, pool, best_params, outer_folds, mask_rate, block_mode, seed):
    rng = np.random.RandomState(seed)
    proba_by_vid, y_by_vid = {}, {}
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        al_cols = [c for c in X_tr.columns if c.startswith("AL_")]
        X_tr_aug = augment_al_columns(X_tr, al_cols, mask_rate, block_mode, rng)
        proba = fit_predict_catboost(X_tr_aug, y_tr, X_va, best_params)
        val_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    ordered_ids = v1_df["Variant_ID"].tolist()
    oof_proba = np.array([proba_by_vid[v] for v in ordered_ids])
    y_full = np.array([y_by_vid[v] for v in ordered_ids])
    return oof_proba, y_full


def fit_and_score_augmented_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold,
                                      train_ids, test_ids, mask_rate, block_mode, seed):
    """`f3_robustness_stress_tests.py::fit_and_score_on_split` ile AYNI
    ic adimlar -- yalnizca egitim X'ine augmentasyon uygulanir, test X'i
    HIC dokunulmaz."""
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    al_cols = [c for c in X_tr.columns if c.startswith("AL_")]
    rng = np.random.RandomState(seed)
    X_tr_aug = augment_al_columns(X_tr, al_cols, mask_rate, block_mode, rng)
    proba = fit_predict_catboost(X_tr_aug, y_tr, X_te, best_params)
    calibrated = calibrator.transform(proba)
    sld = sld_correct(calibrated, w1=w1, w0=w0)
    pred = (sld >= threshold).astype(int)
    row = {"n_train": len(y_tr)}
    row.update(_score(y_te.to_numpy(), pred))
    row["raw_auc"] = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return row


def _fold_membership(v1_df):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    fold_by_vid, fold_sizes = {}, {}
    for fold in outer_folds:
        test_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid in test_vids:
            fold_by_vid[vid] = fold["fold"]
        fold_sizes[fold["fold"]] = {"n_train": len(fold["train_variant_ids"]), "n_test": len(test_vids)}
    return outer_folds, fold_by_vid, fold_sizes


def evaluate_oof(v1_df, oof_proba, y_full, fold_by_vid):
    winner_name, winner_cal, _ = select_calibrator(oof_proba, y_full)
    calibrated = winner_cal.transform(oof_proba)
    threshold, _ = select_prior_and_threshold(calibrated, y_full)
    sld = sld_correct(calibrated)
    pred = (sld >= threshold).astype(int)
    weights = _prior_weights(y_full, FINAL_PATHOGENIC_PRIOR)

    ordered_ids = v1_df["Variant_ID"].tolist()
    fold_idx = np.array([fold_by_vid[v] for v in ordered_ids])
    per_fold = []
    for k in sorted(set(fold_idx)):
        mask = fold_idx == k
        per_fold.append({"fold": k, "weighted_f1": f1_binary_positive_weighted(y_full[mask], pred[mask], weights[mask])})

    return {
        "calibrator_name": winner_name, "calibrator": winner_cal, "threshold": threshold,
        "sensitivity": sensitivity(y_full, pred), "specificity": specificity(y_full, pred),
        "weighted_f1_overall": f1_binary_positive_weighted(y_full, pred, weights),
        "raw_auc_overall": roc_auc_score(y_full, oof_proba),
        "per_fold_weighted_f1": pd.DataFrame(per_fold).sort_values("fold")["weighted_f1"].to_numpy(),
    }


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    print(f"final bundle tarifi: {len(pool)} ozellik, {best_params}", flush=True)

    outer_folds, fold_by_vid, fold_sizes = _fold_membership(v1_df)
    fold_ids_sorted = sorted(fold_sizes.keys())
    n_train_arr = np.array([fold_sizes[k]["n_train"] for k in fold_ids_sorted])
    n_test_arr = np.array([fold_sizes[k]["n_test"] for k in fold_ids_sorted])

    print("\n=== Referans: MEVCUT uretim modeli (cross_fit_oof, degismedi -- import) ===", flush=True)
    ref_oof, ref_y = cross_fit_oof(v1_df, pool, best_params, all_ids)
    ref = evaluate_oof(v1_df, ref_oof, ref_y, fold_by_vid)
    ref_proj = final_projection_metrics(ref["sensitivity"], ref["specificity"])
    print(f"  esik={ref['threshold']:.2f} sens={ref['sensitivity']:.4f} spec={ref['specificity']:.4f} "
          f"weighted_F1(OOF)={ref['weighted_f1_overall']:.4f}", flush=True)
    print(f"  final-projeksiyon (100P/250B): F1={ref_proj['f1']:.4f} MCC={ref_proj['mcc']:.4f} FP={ref_proj['fp']:.2f}", flush=True)
    print(f"  worst-fold weighted_F1={ref['per_fold_weighted_f1'].min():.4f}", flush=True)

    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    ref_cat1_r = fit_and_score_augmented_on_split(v1_df, pool, best_params, bundle["calibrator"], bundle["w1"], bundle["w0"],
                                                   bundle["threshold"], cat1_empty_ids, cat1_filled_ids, 0.0, False, 0)
    print(f"  CAT_1 kritik-yon raw_AUC (bundle'in kendi kalibratoru)={ref_cat1_r['raw_auc']:.4f} "
          f"(F3 kayitli referans: 0,5727)", flush=True)

    print(f"\n=== Izgara: {len(CANDIDATES)} augmentasyon adayi ===", flush=True)
    grid_rows = []
    for cand_spec in CANDIDATES:
        mask_rate, block_mode = cand_spec["mask_rate"], cand_spec["block_mode"]
        oof_proba, y_full = cross_fit_oof_augmented(v1_df, pool, best_params, outer_folds, mask_rate, block_mode, seed=1)
        cand = evaluate_oof(v1_df, oof_proba, y_full, fold_by_vid)
        proj = final_projection_metrics(cand["sensitivity"], cand["specificity"])

        nb = nadeau_bengio_corrected_ttest(cand["per_fold_weighted_f1"], ref["per_fold_weighted_f1"], n_train_arr, n_test_arr)

        cat1_r = fit_and_score_augmented_on_split(v1_df, pool, best_params, cand["calibrator"], W1, W0, cand["threshold"],
                                                   cat1_empty_ids, cat1_filled_ids, mask_rate, block_mode, seed=1)

        row = {
            "name": cand_spec["name"], "mask_rate": mask_rate, "block_mode": block_mode,
            "threshold": cand["threshold"], "calibrator": cand["calibrator_name"],
            "sensitivity": cand["sensitivity"], "specificity": cand["specificity"],
            "proj_f1": proj["f1"], "proj_mcc": proj["mcc"], "proj_fp": proj["fp"],
            "weighted_f1_oof": cand["weighted_f1_overall"],
            "nb_mean_diff_vs_ref": nb["mean_diff"], "nb_p_value": nb["p_value"],
            "cat1_critical_raw_auc": cat1_r["raw_auc"],
            "worst_fold_weighted_f1": cand["per_fold_weighted_f1"].min(),
        }
        grid_rows.append(row)
        print(f"  {cand_spec['name']}: proj_F1={proj['f1']:.4f} spec={cand['specificity']:.4f} "
              f"NBp={nb['p_value']:.4f} diff={nb['mean_diff']:+.4f} CAT1_AUC={cat1_r['raw_auc']:.4f} "
              f"worst_fold={row['worst_fold_weighted_f1']:.4f}", flush=True)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_GRID_CSV, index=False)
    print(f"\nkaydedildi: {OUT_GRID_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== Adim 3: karar kurali (onceden sabitlenmis) ===", flush=True)
    ref_worst = ref["per_fold_weighted_f1"].min()
    ref_cat1_auc = ref_cat1_r["raw_auc"]

    def passes(row):
        c1 = (row["nb_p_value"] >= 0.05) or (row["nb_mean_diff_vs_ref"] > 0)
        c2 = (row["cat1_critical_raw_auc"] - ref_cat1_auc) >= CAT1_SHIFT_CRITICAL_DELTA_THRESHOLD
        c3 = (row["worst_fold_weighted_f1"] - ref_worst) > -0.03
        return c1 and c2 and c3

    grid_df["passes_all_criteria"] = grid_df.apply(passes, axis=1)
    print(grid_df[["name", "nb_p_value", "nb_mean_diff_vs_ref", "cat1_critical_raw_auc",
                    "worst_fold_weighted_f1", "passes_all_criteria"]].to_string(index=False), flush=True)

    winners = grid_df[grid_df["passes_all_criteria"]]
    if len(winners) == 0:
        best = grid_df.loc[(grid_df["cat1_critical_raw_auc"]).idxmax()]
        print(f"\nSONUC: HICBIR aday uc kriteri BIRDEN saglamiyor. En yakin aday: {best['name']} "
              f"(NBp={best['nb_p_value']:.4f}, cat1_delta={best['cat1_critical_raw_auc']-ref_cat1_auc:+.4f}, "
              f"worst_fold_delta={best['worst_fold_weighted_f1']-ref_worst:+.4f}). "
              f"KARAR: hicbiri benimsenmiyor, DUR.", flush=True)
    else:
        winners_sorted = winners.sort_values(["cat1_critical_raw_auc", "nb_p_value"], ascending=[False, False])
        top = winners_sorted.iloc[0]
        print(f"\nSONUC: {len(winners)} aday KRITERI SAGLIYOR. En iyi: {top['name']} "
              f"(NBp={top['nb_p_value']:.4f}, cat1_delta={top['cat1_critical_raw_auc']-ref_cat1_auc:+.4f}, "
              f"worst_fold_delta={top['worst_fold_weighted_f1']-ref_worst:+.4f}). "
              f"Adim 4 (tekrarli-bolme dogrulamasi) GEREKLI.", flush=True)


if __name__ == "__main__":
    main()
