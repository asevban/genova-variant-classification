"""P1 madde 7: `f0_final_model.py`'nin Adim 2'sinin (hiperparametre secimi)
ic secim olcutunu duzeltir -- eski olcut "esik=0,5 + egitim-onseli altinda
ham F1" idi, ama final sistemin GERCEK hedefi kalibrasyon -> onsel
duzeltmesi -> uyarlanabilir esik -> onsel-agirlikli F1. Bu iki olcut ayni
siralamayi vermek ZORUNDA degil (denetimin "objective mismatch" bulgusu).

Bilincli sadelestirme (kapsam karari): kalibratoru bu ic dongude ATLAR --
369 satirda 4-fold ic CV'nin icine bir de kalibrasyon bolmesi eklemek veri
acligina yol acar (~277 train/~92 val zaten kucuk). Bunun yerine EN BUYUK
mismatch kaynagi duzeltiliyor: ham olasiliklara dogrudan kapali-form Bayes
onsel duzeltmesi (sabit w1/w0) + uyarlanabilir esik uygulanip, o esigin
maksimize ettigi onsel-agirlikli F1 aday skoru yapiliyor.

`f0_final_model.py` DEGISTIRILMEDI -- bu modul onun fonksiyonlarini
(`select_hyperparameters` [eski olcut, referans/karsilastirma icin],
`cross_fit_oof`, `fit_frozen_v2_preprocessing`, `_fit_catboost_final`,
`E5_FOLD_THRESHOLD_RANGE`) aynen ice aktarir. `e5_threshold_selection.py::
select_threshold` (0,01 cozunurluk, ayni onsel-agirlikli F1 tarifi)
DOGRUDAN yeniden kullanilir -- yeni bir esik-arama mantigi icat edilmedi.

Akis: mevcut resmi bundle'in (`final_model_bundle_v2.pkl`, P1 madde 11)
havuzunu/hiperparametresini okur -> yeni olcutle hiperparametre secer ->
AYNIYSA sadece dogrulayip DURUR (yeniden fit yok) -> FARKLIYSA yeni
hiperparametreyle Adim 3-6'yi tekrarlayip (kalibrator ARANMIYOR, Beta
sabit tutuluyor -- yukaridaki sadelestirme geregi) yeni bir bundle
(`final_model_bundle_v3.pkl`) uretir, madde 10/11'in araclariyla eskiyle
karsilastirir.

Calistirma: python -m genova.pah.f0_final_model_v3
"""
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import CATBOOST_GRID, fit_predict_catboost
from genova.pah.f0_final_model import (
    V1_PATH, MODEL_DIR, F0_SEED, E5_FOLD_THRESHOLD_RANGE,
    select_hyperparameters, cross_fit_oof, fit_frozen_v2_preprocessing, _fit_catboost_final,
)
from genova.pah.calibration import BetaCalibrator
from genova.pah.e4_prior_correction import sld_correct, W1, W0, FINAL_PATHOGENIC_PRIOR, TRAIN_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import select_threshold
from genova.statistics import sample_level_bootstrap_ci, nadeau_bengio_corrected_ttest, monte_carlo_final_f1_simulation

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
OUT_CSV = TAB_DIR / "f0_objective_fix_comparison.csv"

CURRENT_BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"  # P1 madde 11'in mevcut resmi finali
BUNDLE_ARCHIVED = MODEL_DIR / "final_model_bundle_ARCHIVED_pre_objective_fix.pkl"
BUNDLE_V3_OUT = MODEL_DIR / "final_model_bundle_v3.pkl"
N_BOOT = 2000
N_SIMULATIONS = 2000


def prior_weighted_inner_score(raw_proba, y_true, w1=W1, w0=W0):
    """P1 madde 7'nin CEKIRDEK duzeltmesi: bir ic-val'in HAM olasiliklarindan
    (esik=0,5 uygulanmadan once), kapali-form Bayes onsel duzeltmesi +
    onsel-agirlikli-F1'i maksimize eden esik arayarak TEK bir skor uretir.
    Eski "esik=0,5 ham F1" olcutunun yerini alir -- `select_threshold`
    (e5_threshold_selection.py, 0,01 cozunurluk) DOGRUDAN yeniden kullanilir.
    """
    sld = sld_correct(np.asarray(raw_proba, dtype=float), w1=w1, w0=w0)
    _, best_weighted_f1 = select_threshold(sld, np.asarray(y_true))
    return best_weighted_f1


def select_hyperparameters_prior_weighted(v1_df, pool, all_ids):
    """`f0_final_model.py::select_hyperparameters` ile AYNI 4-fold ic CV
    bolmesini (ayni `F0_SEED`) kullanir -- tek fark, her adayin skorunun
    `prior_weighted_inner_score` ile hesaplanmasi (eski: esik=0,5 ham F1).
    Bu, iki olcutun secimlerinin DOGRUDAN karsilastirilabilir olmasini
    saglar (ayni fold'lar, ayni veri, farkli yalnizca skor formulu).
    """
    inner_folds = sb.build_inner_folds(v1_df, all_ids, seed=F0_SEED)
    inner_mean_scores = []
    for params in CATBOOST_GRID:
        fold_scores = []
        for fold in inner_folds:
            X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
                v1_df, fold["train_variant_ids"], fold["val_variant_ids"], pool,
            )
            raw_proba = fit_predict_catboost(X_tr, y_tr, X_va, params)
            fold_scores.append(prior_weighted_inner_score(raw_proba, y_va))
        inner_mean_scores.append(float(np.mean(fold_scores)))
    best_idx = int(np.argmax(inner_mean_scores))
    return CATBOOST_GRID[best_idx], inner_mean_scores[best_idx], inner_mean_scores


def _oof_predictions_for_bundle(v1_df, bundle, all_ids):
    oof_proba, y_full = cross_fit_oof(v1_df, bundle["pool"], bundle["model_best_params"], all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= bundle["threshold"]).astype(int)
    return sld, pred, y_full


def _per_fold_scores_for_bundle(v1_df, bundle):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
            v1_df, fold["train_variant_ids"], fold["test_variant_ids"], bundle["pool"],
        )
        proba = fit_predict_catboost(X_tr, y_tr, X_va, bundle["model_best_params"])
        calibrated = bundle["calibrator"].transform(proba)
        sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
        pred = (sld >= bundle["threshold"]).astype(int)
        rows.append({
            "fold": fold["fold"], "n_train": len(y_tr), "n_test": len(y_va),
            "f1": f1_binary_positive(y_va, pred),
            "mcc": matthews_correlation_coefficient(y_va, pred),
            "specificity": specificity(y_va, pred),
        })
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    current_bundle = joblib.load(CURRENT_BUNDLE_PATH)
    pool = current_bundle["pool"]
    old_params = current_bundle["model_best_params"]
    print(f"mevcut resmi model (v2): {old_params}, esik={current_bundle['threshold']}, "
          f"kalibrator={current_bundle['calibrator_name']}, havuz={len(pool)} ozellik", flush=True)

    print("\n=== Adim 1: yeni ic secim olcutu (onsel-agirlikli F1, esik=0,5 DEGIL) ===", flush=True)
    new_params, new_score, all_new_scores = select_hyperparameters_prior_weighted(v1_df, pool, all_ids)
    print(f"yeni olcut: secilen={new_params} (ic-ortalama onsel-agirlikli F1={new_score:.4f}, "
          f"tum adaylar={all_new_scores})", flush=True)

    print("\n=== Adim 2: eski olcutle referans (ayni havuz, ayni fold'lar) ===", flush=True)
    old_metric_params, old_metric_f1, old_metric_all = select_hyperparameters(v1_df, pool, all_ids)
    print(f"eski olcut (esik=0,5 ham F1): secilen={old_metric_params} "
          f"(ic-ortalama F1={old_metric_f1:.4f}, tum adaylar={old_metric_all})", flush=True)
    if old_metric_params != old_params:
        print(f"UYARI: eski olcutun bu havuzdaki secimi ({old_metric_params}) mevcut bundle'in "
              f"kaydettiginden ({old_params}) farkli -- havuz degismis olabilir, dikkatli yorumlanmali.", flush=True)

    if new_params == old_params:
        print("\n=== SONUC: AYNI hiperparametre secildi -- final_model_bundle_v2.pkl dogru "
              "olcutle de DOGRULANIYOR. Yeniden fit YOK, DURULUYOR. ===", flush=True)
        pd.DataFrame([{
            "old_metric_params": str(old_metric_params), "old_metric_score": old_metric_f1,
            "new_metric_params": str(new_params), "new_metric_score": new_score,
            "same_hyperparameter_selected": True, "refit_performed": False,
        }]).to_csv(OUT_CSV, index=False)
        print(f"kaydedildi: {OUT_CSV}")
        return

    print(f"\n=== SONUC: FARKLI hiperparametre secildi ({old_params} -> {new_params}) -- "
          f"Adim 3-6'ya geciliyor. ===", flush=True)

    print("\n=== Adim 3: cross-fit OOF (5-fold ic CV, yeni hiperparametreyle) ===", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, new_params, all_ids)

    print("\n=== Adim 4: kalibrator ARANMIYOR (bilincli sadelestirme) -- Beta sabit tutuluyor ===", flush=True)
    calibrator = BetaCalibrator().fit(oof_proba, y_full)
    calibrated_oof = calibrator.transform(oof_proba)

    print("\n=== Adim 5: onsel duzeltmesi + tek esik ===", flush=True)
    sld_oof = sld_correct(calibrated_oof)
    threshold, inner_best_wf1 = select_threshold(sld_oof, y_full)
    lo, hi = E5_FOLD_THRESHOLD_RANGE
    print(f"esik={threshold:.2f} (onsel-agirlikli ic-F1={inner_best_wf1:.4f})", flush=True)
    if not (lo <= threshold <= hi):
        raise RuntimeError(f"Saglamlik kontrolu basarisiz: esik={threshold:.2f}, [{lo},{hi}] disinda -- DURDURULDU.")
    print(f"saglamlik kontrolu OK: esik E5 araligi [{lo},{hi}] icinde.", flush=True)

    print("\n=== Adim 6: final fit + serilestirme ===", flush=True)
    X_all, y_all, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    model, model_cat_features = _fit_catboost_final(X_all, y_all, new_params)
    steps, cat_categories, al_cols, ek_cols, v2_feature_cols = fit_frozen_v2_preprocessing(v1_df, all_ids)

    new_bundle = {
        "v2_steps": steps, "cat_categories": cat_categories,
        "al_cols": al_cols, "ek_cols": ek_cols, "v2_feature_cols": v2_feature_cols,
        "pool": pool, "model": model, "model_cat_features": model_cat_features,
        "model_best_params": new_params,
        "calibrator_name": "beta", "calibrator": calibrator,
        "w1": W1, "w0": W0, "threshold": threshold,
        "final_pathogenic_prior": FINAL_PATHOGENIC_PRIOR, "train_pathogenic_prior": TRAIN_PATHOGENIC_PRIOR,
        "metadata": {
            "model_name": "catboost", "data_version": "v4_from_v2_final_v3_objective_fix",
            "weighting_variant": "B_fixed_minority_2x",
            "n_train_rows": len(v1_df), "n_pool_features": len(pool),
            "calibrator": "beta", "threshold": threshold,
            "inner_selection_metric": "prior_weighted_f1_after_sld (P1 madde 7)",
        },
    }

    print("\n=== eski/yeni karsilastirma (madde 10 araclari) ===", flush=True)
    variant_ids = v1_df["Variant_ID"].tolist()
    old_sld, old_pred, old_y = _oof_predictions_for_bundle(v1_df, current_bundle, all_ids)
    new_sld_full = sld_oof  # Adim 5'te zaten hesaplandi (calibrated_oof -> sld_correct) -- yeniden hesaplanmadi
    new_pred = (new_sld_full >= threshold).astype(int)
    assert np.array_equal(old_y, y_full)

    rows = []
    ci_overlap = {}
    for metric_name, metric_fn in (
        ("f1", f1_binary_positive), ("mcc", matthews_correlation_coefficient), ("specificity", specificity),
    ):
        old_ci = sample_level_bootstrap_ci(variant_ids, old_y, old_pred, metric_fn, n_boot=N_BOOT, seed=1)
        new_ci = sample_level_bootstrap_ci(variant_ids, y_full, new_pred, metric_fn, n_boot=N_BOOT, seed=1)
        overlap = not (new_ci["ci_high"] < old_ci["ci_low"] or old_ci["ci_high"] < new_ci["ci_low"])
        ci_overlap[metric_name] = overlap
        print(f"  {metric_name}: eski={old_ci['point_estimate']:.4f} [{old_ci['ci_low']:.4f},{old_ci['ci_high']:.4f}]  "
              f"yeni={new_ci['point_estimate']:.4f} [{new_ci['ci_low']:.4f},{new_ci['ci_high']:.4f}]  ortusuyor={overlap}", flush=True)
        rows.append({"analysis": "sample_level_bootstrap", "metric": metric_name,
                      "old_point": old_ci["point_estimate"], "new_point": new_ci["point_estimate"],
                      "ci_overlap": overlap})

    old_folds = _per_fold_scores_for_bundle(v1_df, current_bundle)
    new_folds = _per_fold_scores_for_bundle(v1_df, new_bundle)
    n_train, n_test = old_folds["n_train"].to_numpy(), old_folds["n_test"].to_numpy()
    for metric in ("f1", "mcc", "specificity"):
        nb = nadeau_bengio_corrected_ttest(old_folds[metric].to_numpy(), new_folds[metric].to_numpy(), n_train, n_test)
        print(f"  NB {metric}: mean_diff(eski-yeni)={nb['mean_diff']:+.4f}  p={nb['p_value']:.4f}", flush=True)
        rows.append({"analysis": "nadeau_bengio", "metric": metric,
                      "old_point": np.nan, "new_point": nb["mean_diff"], "nb_p_value": nb["p_value"]})

    old_mc = monte_carlo_final_f1_simulation(variant_ids, old_y, old_sld, threshold=current_bundle["threshold"], n_simulations=N_SIMULATIONS)
    new_mc = monte_carlo_final_f1_simulation(variant_ids, y_full, new_sld_full, threshold=threshold, n_simulations=N_SIMULATIONS)
    print(f"  Monte Carlo: eski ort={old_mc['mean']:.4f}  yeni ort={new_mc['mean']:.4f}", flush=True)
    rows.append({"analysis": "monte_carlo_100_250", "metric": "f1",
                  "old_point": old_mc["mean"], "new_point": new_mc["mean"]})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}", flush=True)

    all_overlap = all(ci_overlap.values())
    if not all_overlap:
        print("\n=== KARAR: buyuk sapma tespit edildi (bazi metriklerde CI ortusmuyor) -- DURDURULDU, "
              "yeni bundle KAYDEDILDI ama RESMI YAPILMADI, onay bekleniyor. ===", flush=True)
        joblib.dump(new_bundle, BUNDLE_V3_OUT)
        print(f"(incelemek icin kaydedildi: {BUNDLE_V3_OUT}, ama predict.py GUNCELLENMEDI)", flush=True)
        return

    print("\n=== KARAR: buyuk sapma yok -- final_model_bundle_v3.pkl yeni resmi final yapiliyor. ===", flush=True)
    if not BUNDLE_ARCHIVED.exists():
        shutil.copy2(CURRENT_BUNDLE_PATH, BUNDLE_ARCHIVED)
        print(f"arsivlendi: {CURRENT_BUNDLE_PATH} -> {BUNDLE_ARCHIVED} (orijinal SILINMEDI)", flush=True)
    joblib.dump(new_bundle, BUNDLE_V3_OUT)
    print(f"kaydedildi: {BUNDLE_V3_OUT}", flush=True)


if __name__ == "__main__":
    main()
