"""Dis denetim onerisi: seed-bagged CatBoost denemesi. Final tarifin
(26 ozellik, `depth=5,lr=0,05`, Beta, SLD, uyarlanabilir esik) KUCUK bir
varyasyonu -- her ic fold'da AYNI ozellik/hiperparametre ile 5 FARKLI
seed'le (11,23,42,71,101) ayri ayri CatBoost egitilip olasilik
ortalamasi alinir ("seed bagging"). YENI bir model ailesi DEGIL.

`f0_final_model.py`/`f3_robustness_stress_tests.py`/`models.py`
DEGISTIRILMEDI -- yalnizca import edilip yeniden kullanildi.
`models.py::fit_predict_catboost` seed'i parametrelestirmedigi icin
(`RANDOM_STATE=42` modul-sabiti) burada KUCUK, BILINCLI bir kod tekrari
var -- `f0_final_model.py::_fit_catboost_final`'in ayni gerekcesiyle:
CatBoost cagrisinin mimarisi/hiperparametreleri birebir ayni, yalnizca
`random_seed` disaridan parametre.

KESIN KISIT: `final_model_bundle_v2.pkl`/`predict.py`/split bankasi bu
script tarafindan OTOMATIK DEGISTIRILMEZ -- kazanc bulunsa bile
yalnizca DURDUR/bildir, hicbir dosya uzerine yazilmaz.

Metodolojik seffaflik notu: Adim 3.1/3.2'nin fold-bazli karsilastirmasi,
`f0_final_model.py`'nin KENDI (kalibratoru/esigi TUM 5-fold OOF'ta BIR
KEZ fit eden) prosedurunu iki aday icin de SIMETRIK olarak uygular --
bu, E5'in per-outer-fold nested kalibrasyonundan daha basit bir
yaklasimdir (f0_final_model.py'nin kendi "final bundle uretme"
adimiyla AYNI basitlestirme), ama iki aday da AYNI sekilde
degerlendirildigi icin karsilastirma taraf tutmuyor.

Calistirma: python -m genova.pah.f0_seed_bagging_experiment
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive, f1_binary_positive_weighted, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import _minority_sample_weight, _category_columns
from genova.pah.e2ek_models import _stringify_categoricals
from genova.pah.e3_calibration_run import _variant_ids_in_row_order
from genova.pah.f0_final_model import (
    V1_PATH, TAB_DIR, MODEL_DIR, F0_SEED,
    cross_fit_oof, select_calibrator, select_prior_and_threshold,
)
from genova.pah.f3_robustness_stress_tests import fit_and_score_on_split, _score, BUNDLE_PATH
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import _prior_weights
from genova.statistics import nadeau_bengio_corrected_ttest

import joblib

BAGGING_SEEDS = (11, 23, 42, 71, 101)
OUT_FOLD_CSV = TAB_DIR / "f0_seed_bagging_per_fold.csv"
OUT_SHIFT_CSV = TAB_DIR / "f0_seed_bagging_cat1_shift.csv"


def fit_predict_catboost_seeded(X_train, y_train, X_test, params, seed):
    """`models.py::fit_predict_catboost` ile BIREBIR AYNI mimari/HP --
    yalnizca `random_seed` disaridan parametre (bkz. modul docstring'i)."""
    weights = _minority_sample_weight(y_train)
    cat_cols = _category_columns(X_train)
    model = CatBoostClassifier(
        iterations=100, cat_features=cat_cols, random_seed=seed,
        verbose=False, thread_count=8, allow_writing_files=False, **params,
    )
    model.fit(_stringify_categoricals(X_train, cat_cols), y_train, sample_weight=weights)
    return model.predict_proba(_stringify_categoricals(X_test, cat_cols))[:, 1]


def fit_predict_catboost_bagged(X_train, y_train, X_test, params, seeds=BAGGING_SEEDS):
    probas = [fit_predict_catboost_seeded(X_train, y_train, X_test, params, s) for s in seeds]
    return np.mean(probas, axis=0)


def _fold_membership(v1_df):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    fold_by_vid, fold_sizes = {}, {}
    for fold in outer_folds:
        test_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid in test_vids:
            fold_by_vid[vid] = fold["fold"]
        fold_sizes[fold["fold"]] = {"n_train": len(fold["train_variant_ids"]), "n_test": len(test_vids)}
    return outer_folds, fold_by_vid, fold_sizes


def cross_fit_oof_bagged(v1_df, pool, best_params, all_ids, seeds=BAGGING_SEEDS):
    """`f0_final_model.py::cross_fit_oof` ile AYNI split (repeat_idx=0,
    seed=F0_SEED+1, 5-fold) -- yalnizca her fold'da TEK model yerine
    `seeds` icindeki HER seed icin ayri model egitilip olasiliklar
    ortalaniyor."""
    outer_folds, _, _ = _fold_membership(v1_df)
    proba_by_vid, y_by_vid = {}, {}
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        proba = fit_predict_catboost_bagged(X_tr, y_tr, X_va, best_params, seeds)
        val_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    ordered_ids = v1_df["Variant_ID"].tolist()
    oof_proba = np.array([proba_by_vid[v] for v in ordered_ids])
    y_full = np.array([y_by_vid[v] for v in ordered_ids])
    return oof_proba, y_full


def evaluate_candidate(v1_df, oof_proba, y_full, fold_by_vid):
    """f0_final_model.py'nin AYNI zinciri: kalibrator secimi (Beta/Platt,
    Brier'e gore) -> SLD onsel duzeltmesi -> uyarlanabilir esik -- TUM
    OOF'ta BIR KEZ (f0'in kendi prosedurü). Sonra fold-bazli agirlikli-F1
    kirilimi cikarilir (Adim 3.1/3.2 icin)."""
    winner_name, winner_cal, cal_scores = select_calibrator(oof_proba, y_full)
    calibrated = winner_cal.transform(oof_proba)
    threshold, inner_best_wf1 = select_prior_and_threshold(calibrated, y_full)
    sld = sld_correct(calibrated)
    pred = (sld >= threshold).astype(int)
    weights = _prior_weights(y_full, FINAL_PATHOGENIC_PRIOR)

    ordered_ids = v1_df["Variant_ID"].tolist()
    fold_idx = np.array([fold_by_vid[v] for v in ordered_ids])

    per_fold = []
    for k in sorted(set(fold_idx)):
        mask = fold_idx == k
        per_fold.append({
            "fold": k,
            "weighted_f1": f1_binary_positive_weighted(y_full[mask], pred[mask], weights[mask]),
            "f1": f1_binary_positive(y_full[mask], pred[mask]),
        })

    return {
        "calibrator_name": winner_name, "threshold": threshold,
        "weighted_f1_overall": f1_binary_positive_weighted(y_full, pred, weights),
        "f1_overall": f1_binary_positive(y_full, pred),
        "mcc_overall": matthews_correlation_coefficient(y_full, pred),
        "specificity_overall": specificity(y_full, pred),
        "sensitivity_overall": sensitivity(y_full, pred),
        "raw_auc_overall": roc_auc_score(y_full, oof_proba),
        "per_fold": pd.DataFrame(per_fold),
        "calibrator": winner_cal, "sld_w1_w0_default": True,
    }


def fit_and_score_bagged_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, train_ids, test_ids, seeds=BAGGING_SEEDS):
    """`f3_robustness_stress_tests.py::fit_and_score_on_split` ile AYNI
    mantik -- yalnizca `fit_predict_catboost` yerine bagged versiyonu."""
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    proba = fit_predict_catboost_bagged(X_tr, y_tr, X_te, best_params, seeds)
    calibrated = calibrator.transform(proba)
    sld = sld_correct(calibrated, w1=w1, w0=w0)
    pred = (sld >= threshold).astype(int)
    row = {"n_train": len(y_tr)}
    row.update(_score(y_te.to_numpy(), pred))
    row["raw_auc"] = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return row


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    print(f"final bundle tarifi: {len(pool)} ozellik, {best_params}, seed'ler={BAGGING_SEEDS}", flush=True)

    outer_folds, fold_by_vid, fold_sizes = _fold_membership(v1_df)

    print("\n=== Adim 1a: mevcut TEK-seed cross-fit OOF (referans, degismedi -- cross_fit_oof import) ===", flush=True)
    t0 = time.time()
    oof_single, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print("\n=== Adim 1b: seed-bagged cross-fit OOF (5 seed x 5 fold = 25 fit) ===", flush=True)
    t0 = time.time()
    oof_bagged, y_full_b = cross_fit_oof_bagged(v1_df, pool, best_params, all_ids, BAGGING_SEEDS)
    assert np.array_equal(y_full, y_full_b)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print("\n=== Adim 2: kalibrasyon+SLD+esik (ayni zincir, iki aday icin ayri ayri) ===", flush=True)
    single = evaluate_candidate(v1_df, oof_single, y_full, fold_by_vid)
    bagged = evaluate_candidate(v1_df, oof_bagged, y_full, fold_by_vid)
    print(f"  TEK-seed : kalibrator={single['calibrator_name']} esik={single['threshold']:.2f} "
          f"weighted_F1={single['weighted_f1_overall']:.4f} raw_AUC={single['raw_auc_overall']:.4f}", flush=True)
    print(f"  BAGGED   : kalibrator={bagged['calibrator_name']} esik={bagged['threshold']:.2f} "
          f"weighted_F1={bagged['weighted_f1_overall']:.4f} raw_AUC={bagged['raw_auc_overall']:.4f}", flush=True)

    print("\n=== Adim 3.1: NB testi (weighted-F1, fold-bazli, n=5) ===", flush=True)
    fold_ids_sorted = sorted(fold_sizes.keys())
    n_train = np.array([fold_sizes[k]["n_train"] for k in fold_ids_sorted])
    n_test = np.array([fold_sizes[k]["n_test"] for k in fold_ids_sorted])
    single_wf1 = single["per_fold"].sort_values("fold")["weighted_f1"].to_numpy()
    bagged_wf1 = bagged["per_fold"].sort_values("fold")["weighted_f1"].to_numpy()
    nb = nadeau_bengio_corrected_ttest(bagged_wf1, single_wf1, n_train, n_test)
    print(f"  fold-bazli weighted_F1: TEK-seed={list(np.round(single_wf1,4))}", flush=True)
    print(f"  fold-bazli weighted_F1: BAGGED  ={list(np.round(bagged_wf1,4))}", flush=True)
    print(f"  mean_diff(bagged-single)={nb['mean_diff']:+.4f}  NB p={nb['p_value']:.4f}", flush=True)

    print("\n=== Adim 3.2: varyans/worst-fold karsilastirmasi ===", flush=True)
    print(f"  TEK-seed : std={single_wf1.std():.4f}  worst-fold={single_wf1.min():.4f}", flush=True)
    print(f"  BAGGED   : std={bagged_wf1.std():.4f}  worst-fold={bagged_wf1.min():.4f}", flush=True)

    per_fold_out = single["per_fold"].rename(columns={"weighted_f1": "weighted_f1_single", "f1": "f1_single"}).merge(
        bagged["per_fold"].rename(columns={"weighted_f1": "weighted_f1_bagged", "f1": "f1_bagged"}), on="fold",
    )
    per_fold_out.to_csv(OUT_FOLD_CSV, index=False)
    print(f"kaydedildi: {OUT_FOLD_CSV}", flush=True)

    print("\n=== Adim 3.3: CAT_1 doluluk-gecisi testi -- BAGGED modelle (F3'un EN KRITIK testi) ===", flush=True)
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    b_cal, b_thr = bagged["calibrator"], bagged["threshold"]
    from genova.pah.e4_prior_correction import W1, W0

    shift_rows = []
    r = fit_and_score_bagged_on_split(v1_df, pool, best_params, b_cal, W1, W0, b_thr, cat1_filled_ids, cat1_empty_ids)
    r["direction"] = "train=CAT_1_dolu -> test=CAT_1_bos"
    print(f"  [BAGGED] {r['direction']}: F1={r['f1']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    r = fit_and_score_bagged_on_split(v1_df, pool, best_params, b_cal, W1, W0, b_thr, cat1_empty_ids, cat1_filled_ids)
    r["direction"] = "train=CAT_1_bos -> test=CAT_1_dolu (KRITIK)"
    print(f"  [BAGGED] {r['direction']}: F1={r['f1']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    print("\n  --- Karsilastirma icin: MEVCUT (tek-seed) bundle'in AYNI test uzerindeki sonucu ---", flush=True)
    s_cal, s_w1, s_w0, s_thr = bundle["calibrator"], bundle["w1"], bundle["w0"], bundle["threshold"]
    r_ref = fit_and_score_on_split(v1_df, pool, best_params, s_cal, s_w1, s_w0, s_thr, cat1_empty_ids, cat1_filled_ids)
    print(f"  [TEK-seed, bundle] train=CAT_1_bos -> test=CAT_1_dolu (KRITIK): "
          f"F1={r_ref['f1']:.4f} raw_AUC={r_ref['raw_auc']:.4f}  "
          f"(F3 raporunda kayitli referans: F1=0,0588 raw_AUC=0,5727)", flush=True)

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(OUT_SHIFT_CSV, index=False)
    print(f"kaydedildi: {OUT_SHIFT_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== Adim 4: karar kurali ===", flush=True)
    critical_auc_bagged = shift_rows[1]["raw_auc"]
    critical_auc_single_ref = r_ref["raw_auc"]
    print(f"  NB p-degeri (ortalama performans)={nb['p_value']:.4f} (esik: >=0,05 -> anlamli fark yok)", flush=True)
    print(f"  CAT_1 kritik yon raw_AUC: TEK-seed(referans)={critical_auc_single_ref:.4f} -> BAGGED={critical_auc_bagged:.4f} "
          f"(esik: >0,65 -> belirgin iyilesme)", flush=True)
    if nb["p_value"] >= 0.05 and critical_auc_bagged <= 0.65:
        print("  KARAR: Bagging'in ANLAMLI GETIRISI YOK -- 5x hesaplama maliyetini karsilamiyor. "
              "Mevcut model KORUNUYOR. DUR.", flush=True)
    elif nb["p_value"] >= 0.05 and critical_auc_bagged > 0.65:
        print("  KARAR: Ortalama performans korunuyor VE CAT_1 cokusu belirgin hafifliyor -- "
              "GERCEK BIR KAZANC. DURDURULDU, kullanici onayi bekleniyor.", flush=True)
    elif nb["p_value"] < 0.05 and nb["mean_diff"] < 0:
        print("  KARAR: Bagging ortalama performansta ANLAMLI DUSUS gosteriyor -- REDDEDILDI. DUR.", flush=True)
    else:
        print("  KARAR: Beklenmedik kombinasyon -- elle degerlendirilmeli.", flush=True)


if __name__ == "__main__":
    main()
