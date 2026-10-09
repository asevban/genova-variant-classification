"""P0-4 (bkz. reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md): duzeltilmis
fold-lokal `v4_from_v2` havuzuyla (feature_selection.py::compute_fold_local_
pool, `v4_fold_local_pools.json`) CatBoost/RandomForest/LightGBM/XGBoost'u
50 dis fold'da yeniden olcer.

Hicbir hiperparametre yeniden aranmiyor -- `e2_model_comparison.csv`'nin
v4_from_v2 icin ZATEN kaydettigi `best_params` aynen yeniden kullanilir
(E3'un kalibrasyon kosumuyla ayni disiplin). Bu, havuz-duzeltmesini TEK
degisen degisken olarak izole eder: ayni model ailesi, ayni hiperparametre,
ayni agirliklandirma varyanti -- yalnizca ozellik havuzunun kaynagi
(global sabit -> fold-lokal) degisiyor.

Agirliklandirma varyantlari, `e2_model_comparison.csv`'de v4_from_v2 icin
FIILEN mevcut olanlarla birebir ayni: catboost/random_forest -> A/B/C,
lightgbm/xgboost -> yalnizca B (bu ikisinin fit_predict fonksiyonlari
agirligi zaten sabit x2.0 ile hard-code ediyor, hicbir A/C varyanti hic
uretilmedi -- bkz. models.py::fit_predict_lgbm/fit_predict_xgb).

Cikti: reports/tables/e2_model_comparison_v4fixed.csv (YENI dosya,
e2_model_comparison.csv UZERINE YAZILMAZ).

Calistirma: python -m genova.pah.e2_rescoring_v4fixed
"""
import json
import time
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah import e2ek_models as em
from genova.pah.weighting import weights_no_weight, weights_data_driven_spw
from genova.pah.feature_selection import compute_fold_local_pool
from genova.pah.fold_features import build_fold_features
from genova.pah.models import _score_row

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
POOLS_JSON = ROOT / "reports" / "tables" / "v4_fold_local_pools.json"
OUT_CSV = ROOT / "reports" / "tables" / "e2_model_comparison_v4fixed.csv"

N_REPEATS = 10
SEED = 42

# (model_name, weighting_variant, fit_predict_fn) -- e2_model_comparison.csv'de
# v4_from_v2 icin FIILEN var olan (model, weighting_variant) kombinasyonlariyla
# birebir ayni.
COMBOS = [
    ("catboost", "A_no_weight", em.make_fit_predict_catboost(weights_no_weight)),
    ("catboost", "B_fixed_minority_2x", m.fit_predict_catboost),
    ("catboost", "C_data_driven_spw", em.make_fit_predict_catboost(weights_data_driven_spw)),
    ("random_forest", "A_no_weight", em.make_fit_predict_rf(weights_no_weight)),
    ("random_forest", "B_fixed_minority_2x", em.fit_predict_rf),
    ("random_forest", "C_data_driven_spw", em.make_fit_predict_rf(weights_data_driven_spw)),
    ("lightgbm", "B_fixed_minority_2x", m.fit_predict_lgbm),
    ("xgboost", "B_fixed_minority_2x", m.fit_predict_xgb),
]


def _lookup_best_params(comparison_df, model_name, weighting_variant, version_name, repeat_idx, outer_fold_idx):
    mask = (
        (comparison_df.model == model_name) & (comparison_df.data_version == version_name)
        & (comparison_df.weighting_variant == weighting_variant)
        & (comparison_df.repeat == repeat_idx) & (comparison_df.outer_fold == outer_fold_idx)
    )
    rows = comparison_df.loc[mask, "best_params"]
    assert len(rows) == 1, (
        f"{model_name}/{weighting_variant}/{version_name} repeat={repeat_idx} "
        f"outer={outer_fold_idx}: {len(rows)} satir bulundu"
    )
    return json.loads(rows.iloc[0])


def main():
    v1_df = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1_df.columns if c.startswith("AL_")]
    comparison_df = pd.read_csv(COMPARISON_CSV)
    pools = json.loads(POOLS_JSON.read_text())["pools"]

    all_rows = []
    n_folds_run = 0
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            fold_idx = outer_fold["fold"]
            fold_key = f"repeat{repeat_idx:02d}_fold{fold_idx}"
            pool = pools[fold_key]["features"]
            train_ids, test_ids = outer_fold["train_variant_ids"], outer_fold["test_variant_ids"]

            X_train, y_train, X_test, y_test = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)

            for model_name, weighting_variant, fit_predict_fn in COMBOS:
                t0 = time.time()
                best_params = _lookup_best_params(
                    comparison_df, model_name, weighting_variant, "v4_from_v2", repeat_idx, fold_idx,
                )
                proba = fit_predict_fn(X_train, y_train, X_test, best_params)
                scores = _score_row(y_test, proba)
                all_rows.append({
                    "model": model_name, "data_version": "v4_from_v2",
                    "weighting_variant": weighting_variant,
                    "repeat": repeat_idx, "outer_fold": fold_idx,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "n_pool_features": len(pool),
                    "best_params": json.dumps(best_params),
                    "seconds": time.time() - t0, **scores,
                })

            n_folds_run += 1
            print(f"repeat {repeat_idx} fold {fold_idx} done ({n_folds_run}/{N_REPEATS * 5}) "
                  f"-- pool={len(pool)} ozellik", flush=True)

    result = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")
    print(result.groupby(["model", "weighting_variant"])[["f1", "mcc", "specificity"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
