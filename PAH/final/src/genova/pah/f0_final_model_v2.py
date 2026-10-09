"""P1 madde 11: `al_all_missing`/`CAT_1` provenance-riski nedeniyle final
havuzdan elle cikarildiktan sonra (28->26 ozellik, bkz. reports/tables/
f0_final_feature_pool_v2.json), `f0_final_model.py`'nin Adim 2-6'sini
(hiperparametre secimi -> cross-fit OOF -> kalibrator secimi -> onsel
duzeltme+esik -> final fit -> serilestirme) AYNI (degistirilmemis)
fonksiyonlarla, YALNIZCA farkli bir havuzla tekrarlar. Adim 1 (ozellik
secimi) bu turda TEKRAR CALISTIRILMIYOR -- havuz elle duzenlendi.

`f0_final_model.py` DEGISTIRILMEDI -- bu modul onun fonksiyonlarini
(`select_hyperparameters`, `cross_fit_oof`, `select_calibrator`,
`select_prior_and_threshold`, `fit_frozen_v2_preprocessing`,
`_fit_catboost_final`) aynen ice aktarip yeniden kullanir.

Cikti: `models/pah/final_model_bundle_v2.pkl`. Eski `final_model_bundle.
pkl` bu betik calistirilirken ARSIVLENIR (`final_model_bundle_ARCHIVED_
with_provenance_features.pkl`), SILINMEZ.

Calistirma: python -m genova.pah.f0_final_model_v2
"""
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from genova.pah.f0_final_model import (
    V1_PATH, MODEL_DIR, BUNDLE_OUT, E5_FOLD_THRESHOLD_RANGE,
    select_hyperparameters, cross_fit_oof, select_calibrator,
    select_prior_and_threshold, fit_frozen_v2_preprocessing, _fit_catboost_final,
)
from genova.pah import fold_versions as fv
from genova.pah.e4_prior_correction import W1, W0, FINAL_PATHOGENIC_PRIOR, TRAIN_PATHOGENIC_PRIOR

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
POOL_V2_PATH = TAB_DIR / "f0_final_feature_pool_v2.json"
BUNDLE_ARCHIVED = MODEL_DIR / "final_model_bundle_ARCHIVED_with_provenance_features.pkl"
BUNDLE_V2_OUT = MODEL_DIR / "final_model_bundle_v2.pkl"


def main():
    if BUNDLE_OUT.exists() and not BUNDLE_ARCHIVED.exists():
        shutil.copy2(BUNDLE_OUT, BUNDLE_ARCHIVED)
        print(f"arsivlendi: {BUNDLE_OUT} -> {BUNDLE_ARCHIVED} (orijinal SILINMEDI, yerinde kaldi)")

    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    pool = json.loads(POOL_V2_PATH.read_text(encoding="utf-8"))["features"]
    print(f"havuz (Adim 1 TEKRAR CALISTIRILMADI, elle duzenlenmis): {len(pool)} ozellik <- {POOL_V2_PATH.name}")

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
            "model_name": "catboost", "data_version": "v4_from_v2_final_v2_no_provenance",
            "weighting_variant": "B_fixed_minority_2x",
            "n_train_rows": len(v1_df), "n_pool_features": len(pool),
            "calibrator": winner_name, "threshold": threshold,
            "removed_for_provenance_risk": ["al_all_missing", "CAT_1"],
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_V2_OUT)
    print(f"kaydedildi: {BUNDLE_V2_OUT}")


if __name__ == "__main__":
    main()
