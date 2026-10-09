"""F3 sonrasi karar matrisi, Adim 1: aday B (23 ozellik -- `AL_26`,
`AL_12`, `AL_7` cikarildi, `AL_49` TUTULDU). `f1_ablation_candidate.py`
ile AYNI desen (madde 11 disipliniyle): Adim 2-6'yi (hiperparametre ->
cross-fit OOF -> kalibrator -> onsel duzeltme+esik -> final fit ->
serilestirme) 23-ozellik havuzuyla tekrarlar, `f0_final_model.py`
DEGISTIRILMEDEN fonksiyonlari yeniden kullanir.

Bu SADECE bir ADAY uretir -- `final_model_bundle_v2.pkl`'e DOKUNMAZ,
`predict.py` GUNCELLENMEZ.

Calistirma: python -m genova.pah.f3_ablation_candidate_b
"""
from pathlib import Path

import joblib
import json
import pandas as pd

from genova.pah.f0_final_model import (
    V1_PATH, MODEL_DIR, E5_FOLD_THRESHOLD_RANGE,
    select_hyperparameters, cross_fit_oof, select_calibrator,
    select_prior_and_threshold, fit_frozen_v2_preprocessing, _fit_catboost_final,
)
from genova.pah import fold_versions as fv
from genova.pah.e4_prior_correction import W1, W0, FINAL_PATHOGENIC_PRIOR, TRAIN_PATHOGENIC_PRIOR

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
POOL_PATH = TAB_DIR / "f3_final_feature_pool_b23.json"
BUNDLE_OUT = MODEL_DIR / "final_model_bundle_f3_candidate_b23.pkl"


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))["features"]
    print(f"havuz (aday B): {len(pool)} ozellik <- {POOL_PATH.name}", flush=True)

    print("\n=== Adim 2: hiperparametre secimi (4-fold ic CV) ===", flush=True)
    best_params, best_inner_f1, all_inner_f1 = select_hyperparameters(v1_df, pool, all_ids)
    print(f"secilen: {best_params} (ic-ortalama F1={best_inner_f1:.4f}, tum adaylar={all_inner_f1})", flush=True)

    print("\n=== Adim 3: cross-fit OOF (5-fold ic CV) ===", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    print(f"OOF vektoru: {len(oof_proba)} satir", flush=True)

    print("\n=== Adim 4: kalibrator secimi (Platt vs Beta) ===", flush=True)
    winner_name, winner_cal, cal_scores = select_calibrator(oof_proba, y_full)
    print(f"platt: {cal_scores['platt']}  beta: {cal_scores['beta']}  -> secilen: {winner_name}", flush=True)
    calibrated_oof = winner_cal.transform(oof_proba)

    print("\n=== Adim 5: onsel duzeltmesi + tek esik ===", flush=True)
    threshold, inner_best_wf1 = select_prior_and_threshold(calibrated_oof, y_full)
    lo, hi = E5_FOLD_THRESHOLD_RANGE
    print(f"esik={threshold:.2f} (onsel-agirlikli ic-F1={inner_best_wf1:.4f})", flush=True)
    if not (lo <= threshold <= hi):
        raise RuntimeError(f"Saglamlik kontrolu basarisiz: esik={threshold:.2f} [{lo},{hi}] disinda -- DURDURULDU.")
    print(f"saglamlik kontrolu OK: esik E5 araligi [{lo},{hi}] icinde.", flush=True)

    print("\n=== Adim 6: final fit (369 satirin tamami) + serilestirme ===", flush=True)
    X_all, y_all, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    model, model_cat_features = _fit_catboost_final(X_all, y_all, best_params)
    steps, cat_categories, al_cols, ek_cols, v2_feature_cols = fit_frozen_v2_preprocessing(v1_df, all_ids)

    bundle = {
        "v2_steps": steps, "cat_categories": cat_categories,
        "al_cols": al_cols, "ek_cols": ek_cols, "v2_feature_cols": v2_feature_cols,
        "pool": pool, "model": model, "model_cat_features": model_cat_features,
        "model_best_params": best_params,
        "calibrator_name": winner_name, "calibrator": winner_cal,
        "w1": W1, "w0": W0, "threshold": threshold,
        "final_pathogenic_prior": FINAL_PATHOGENIC_PRIOR, "train_pathogenic_prior": TRAIN_PATHOGENIC_PRIOR,
        "metadata": {
            "model_name": "catboost", "data_version": "v4_from_v2_f3_candidate_b23",
            "weighting_variant": "B_fixed_minority_2x",
            "n_train_rows": len(v1_df), "n_pool_features": len(pool),
            "calibrator": winner_name, "threshold": threshold,
            "removed_for_provenance_risk": ["AL_26", "AL_12", "AL_7"],
            "kept_despite_weaker_provenance_signal": ["AL_49"],
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_OUT)
    print(f"kaydedildi (ADAY, resmi DEGIL): {BUNDLE_OUT}", flush=True)


if __name__ == "__main__":
    main()
