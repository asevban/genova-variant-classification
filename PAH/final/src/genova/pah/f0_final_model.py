"""P0-5: final model egitim hatti + serilestirme, 369 satirin TAMAMINDA
BIR KEZ. E2-E6'nin 50-dis-fold karsilastirma mantigini TEKRARLAMIYOR --
o zaten "hangi model/versiyon/kalibrator daha iyi" sorusunu cevapladi
(CatBoost + v4_from_v2 + Beta/Platt-yakin + kapali-form Bayes onsel
duzeltmesi + uyarlanabilir esik, ağırlıklandırma B). Bu modul o karari
tek, dondurulmus bir prosedurle uyguluyor.

Kendi ic CV'sini kurar -- `split_bank.py`'nin `build_outer_folds`/
`build_inner_folds` SAF fonksiyonlarini (data/splits/pah/ dosyalarina
hicbir okuma/yazma yapmadan) yeni bir tohumla (F0_SEED) yeniden kullanir.
Split bankasinin KENDISI bu adimda ne okunuyor ne de degistiriliyor.

Adim 1 -- Ozellik secimi (369'un tamami, `compute_fold_local_pool` kendi
          ic train/val bolmesini kendisi yapiyor).
Adim 2 -- Hiperparametre secimi (369 -> 4-fold ic CV, CATBOOST_GRID).
Adim 3 -- Cross-fit OOF olasiliklar (369 -> 5-fold ic CV).
Adim 4 -- Kalibrator secimi (Platt vs Beta, OOF uzerinde, Brier'e gore).
Adim 5 -- Onsel duzeltmesi (sabit w1/w0) + tek esik (OOF uzerinde).
Adim 6 -- Final fit (369'un tamami) + tek `joblib` paketi.

Calistirma: python -m genova.pah.f0_final_model
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from genova.metrics import f1_binary_positive
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import CATBOOST_GRID, fit_predict_catboost, _minority_sample_weight, _category_columns, RANDOM_STATE
from genova.pah.e2ek_models import _stringify_categoricals
from genova.pah.e3_calibration_run import _variant_ids_in_row_order
from genova.pah.feature_selection import compute_fold_local_pool
from genova.pah.fold_features import build_fold_features
from genova.pah.calibration import PlattCalibrator, BetaCalibrator, evaluate_probabilities
from genova.pah.e4_prior_correction import sld_correct, W1, W0, FINAL_PATHOGENIC_PRIOR, TRAIN_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import select_threshold

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models" / "pah"
POOL_OUT = TAB_DIR / "f0_final_feature_pool.json"
BUNDLE_OUT = MODEL_DIR / "final_model_bundle.pkl"

F0_SEED = 42  # split bankasindan BAGIMSIZ -- yalnizca bu final-fit'in kendi ic CV'si icin
N_HP_FOLDS = 4
N_OOF_FOLDS = 5
E5_FOLD_THRESHOLD_RANGE = (0.13, 0.50)  # e5_threshold_selection.csv'den DOGRUDAN dogrulandi (catboost/v4_from_v2, n=50)


def select_final_pool(v1_df):
    al_columns = [c for c in v1_df.columns if c.startswith("AL_")]
    all_ids = v1_df["Variant_ID"].tolist()
    X_all, y_all, _, _ = build_fold_features(v1_df, al_columns, all_ids, all_ids)
    pool, stability = compute_fold_local_pool(X_all, y_all, seed=F0_SEED)
    return sorted(pool), stability


def select_hyperparameters(v1_df, pool, all_ids):
    inner_folds = sb.build_inner_folds(v1_df, all_ids, seed=F0_SEED)
    inner_mean_f1 = []
    for params in CATBOOST_GRID:
        f1s = []
        for fold in inner_folds:
            X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
                v1_df, fold["train_variant_ids"], fold["val_variant_ids"], pool,
            )
            proba = fit_predict_catboost(X_tr, y_tr, X_va, params)
            f1s.append(f1_binary_positive(y_va, (proba >= 0.5).astype(int)))
        inner_mean_f1.append(float(np.mean(f1s)))
    best_idx = int(np.argmax(inner_mean_f1))
    return CATBOOST_GRID[best_idx], inner_mean_f1[best_idx], inner_mean_f1


def cross_fit_oof(v1_df, pool, best_params, all_ids):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    proba_by_vid, y_by_vid = {}, {}
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
            v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool,
        )
        proba = fit_predict_catboost(X_tr, y_tr, X_va, best_params)
        val_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    assert len(proba_by_vid) == len(all_ids), "OOF butun 369 satiri kapsamadi"

    ordered_ids = v1_df["Variant_ID"].tolist()
    oof_proba = np.array([proba_by_vid[v] for v in ordered_ids])
    y_full = np.array([y_by_vid[v] for v in ordered_ids])
    return oof_proba, y_full


def select_calibrator(oof_proba, y_full):
    scores = {}
    fitted = {}
    for name, cls in (("platt", PlattCalibrator), ("beta", BetaCalibrator)):
        cal = cls().fit(oof_proba, y_full)
        calibrated = cal.transform(oof_proba)
        scores[name] = evaluate_probabilities(y_full, calibrated)
        fitted[name] = cal
    winner = min(scores, key=lambda k: scores[k]["brier"])
    return winner, fitted[winner], scores


def select_prior_and_threshold(calibrated_oof, y_full):
    sld_oof = sld_correct(calibrated_oof)
    threshold, inner_best_wf1 = select_threshold(sld_oof, y_full)
    return threshold, inner_best_wf1


def _fit_catboost_final(X_all, y_all, best_params):
    """`models.py::fit_predict_catboost` ile AYNI mimari/hiperparametreler
    -- yalnizca `predict_proba` yerine FITTED model nesnesini dondurur
    (serilestirme icin gerekli). E2-E6'nin kendi dosyalarina dokunmamak
    icin kucuk, bilincli bir kod tekrari (E2-E6 bu turda degistirilmiyor)."""
    weights = _minority_sample_weight(y_all)
    cat_cols = _category_columns(X_all)
    model = CatBoostClassifier(
        iterations=100, cat_features=cat_cols, random_seed=RANDOM_STATE,
        verbose=False, thread_count=8, allow_writing_files=False, **best_params,
    )
    model.fit(_stringify_categoricals(X_all, cat_cols), y_all, sample_weight=weights)
    return model, cat_cols


def fit_frozen_v2_preprocessing(v1_df, all_ids):
    """v2-stili on isleme adimlarini (`fold_versions.py::_v2_steps` +
    `_refit_categoricals`) 369 satirin TAMAMINDA fit eder ve FITTED nesneleri
    dondurur -- `build_v2`'nin aksine (her cagrida yeni nesne yaratip atar),
    bu fonksiyon nesneleri `predict.py`'nin daha sonra `.transform()` ile
    yeniden kullanabilmesi icin saklar."""
    al_cols, ek_cols = fv._al_ek_cols(v1_df)
    train_df, _ = fv._split(v1_df, all_ids, all_ids)
    steps = fv._v2_steps(al_cols, al_fill_strategy="zero")
    for step in steps:
        step.fit(train_df)
        train_df = step.transform(train_df)
    train_df, _ = fv._refit_categoricals(train_df, train_df, fv.CAT_ALL_COLS)
    cat_categories = {c: train_df[c].cat.categories.tolist() for c in fv.CAT_ALL_COLS}
    v2_feature_cols = al_cols + ek_cols + fv.CAT_ALL_COLS + ["al_all_missing", "ek_cat_block_missing", "EK_3_missing"]
    return steps, cat_categories, al_cols, ek_cols, v2_feature_cols


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()

    print("=== Adim 1: ozellik secimi (369 satirin tamami) ===", flush=True)
    pool, stability = select_final_pool(v1_df)
    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "rule": "n_methods_stable >= 2 (P0-duzeltmeli compute_fold_local_pool, 369 satirin tamaminda BIR fit)",
            "n_features": len(pool), "features": pool,
        }, f, indent=2, ensure_ascii=False)
    print(f"final havuz: {len(pool)} ozellik -> {POOL_OUT}")

    print("\n=== Adim 2: hiperparametre secimi (4-fold ic CV) ===", flush=True)
    best_params, best_inner_f1, all_inner_f1 = select_hyperparameters(v1_df, pool, all_ids)
    print(f"secilen: {best_params} (ic-ortalama F1={best_inner_f1:.4f}, tum adaylar={all_inner_f1})")

    print("\n=== Adim 3: cross-fit OOF (5-fold ic CV) ===", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    print(f"OOF vektoru: {len(oof_proba)} satir")

    print("\n=== Adim 4: kalibrator secimi (Platt vs Beta) ===", flush=True)
    winner_name, winner_cal, cal_scores = select_calibrator(oof_proba, y_full)
    print(f"platt: {cal_scores['platt']}  beta: {cal_scores['beta']}  -> secilen: {winner_name}")
    calibrated_oof = winner_cal.transform(oof_proba)

    print("\n=== Adim 5: onsel duzeltmesi + tek esik ===", flush=True)
    threshold, inner_best_wf1 = select_prior_and_threshold(calibrated_oof, y_full)
    lo, hi = E5_FOLD_THRESHOLD_RANGE
    print(f"w1={W1:.6f} w0={W0:.6f} -> esik={threshold:.2f} (onsel-agirlikli ic-F1={inner_best_wf1:.4f})")
    if not (lo <= threshold <= hi):
        raise RuntimeError(
            f"Saglamlik kontrolu basarisiz: esik={threshold:.2f}, E5'in gozlenen fold-esik "
            f"araligi [{lo},{hi}] disinda -- DURDURULDU, ilerlenmiyor."
        )
    print(f"saglamlik kontrolu OK: esik E5 araligi [{lo},{hi}] icinde.")

    print("\n=== Adim 6: final fit (369 satirin tamami) + serilestirme ===", flush=True)
    X_all, y_all, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    model, model_cat_features = _fit_catboost_final(X_all, y_all, best_params)

    steps, cat_categories, al_cols, ek_cols, v2_feature_cols = fit_frozen_v2_preprocessing(v1_df, all_ids)

    bundle = {
        "v2_steps": steps,
        "cat_categories": cat_categories,
        "al_cols": al_cols, "ek_cols": ek_cols,
        "v2_feature_cols": v2_feature_cols,
        "pool": pool,
        "model": model,
        "model_cat_features": model_cat_features,
        "model_best_params": best_params,
        "calibrator_name": winner_name,
        "calibrator": winner_cal,
        "w1": W1, "w0": W0,
        "threshold": threshold,
        "final_pathogenic_prior": FINAL_PATHOGENIC_PRIOR,
        "train_pathogenic_prior": TRAIN_PATHOGENIC_PRIOR,
        "metadata": {
            "model_name": "catboost", "data_version": "v4_from_v2_final",
            "weighting_variant": "B_fixed_minority_2x",
            "n_train_rows": len(v1_df), "n_pool_features": len(pool),
            "calibrator": winner_name, "threshold": threshold,
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_OUT)
    print(f"kaydedildi: {BUNDLE_OUT}")


if __name__ == "__main__":
    main()
