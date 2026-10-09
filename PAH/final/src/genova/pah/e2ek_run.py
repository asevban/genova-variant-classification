"""Asama E2-EK: Tier 1 (Random Forest x {v1,v2,v2_raw_nan,v4_from_v2},
KNN x v4_from_v2) + Tier 2 (A/B/C agirliklandirma karsilastirmasi, yeni
RF'in en iyi versiyonu + CatBoost/v4_from_v2). Sonuclar AYNI `reports/
tables/e2_model_comparison.csv`'ye eklenir -- mevcut satirlar SILINMEZ,
yalnizca geriye-donuk uyumluluk icin `weighting_variant` kolonu eklenir
(hepsi zaten Strateji B ile uretildigi icin `B_fixed_minority_2x` olarak
etiketlenir).

Calistirma: python -m genova.pah.e2ek_run
"""
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import e2ek_models as em
from genova.pah import models as m
from genova.pah.weighting import WEIGHTING_STRATEGIES
from genova.pah.idempotent_io import upsert_csv

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"

B_LABEL = "B_fixed_minority_2x"
NA_LABEL = "not_applicable"


def migrate_add_weighting_column():
    df = pd.read_csv(COMPARISON_CSV)
    if "weighting_variant" not in df.columns:
        df["weighting_variant"] = B_LABEL
        df.to_csv(COMPARISON_CSV, index=False)
        print(f"{COMPARISON_CSV}: 'weighting_variant' kolonu eklendi (mevcut satirlar '{B_LABEL}' olarak etiketlendi)")
    else:
        print("'weighting_variant' kolonu zaten mevcut, migrasyon atlandi")


def append_with_variant(results_df, weighting_variant):
    """P2 madde 19: idempotent yazar -- ayni (model,data_version,
    weighting_variant,repeat,outer_fold) icin script iki kez calistirilirsa
    satir COGALMAZ."""
    results_df = results_df.copy()
    results_df["weighting_variant"] = weighting_variant
    upsert_csv(results_df, COMPARISON_CSV, ["model", "data_version", "weighting_variant", "repeat", "outer_fold"])


def main():
    migrate_add_weighting_column()

    v1_df = pd.read_parquet(V1_PATH)
    pool = json.loads(POOL_PATH.read_text())["features"]

    builders = {
        "v1": lambda df, tr, te: fv.build_v1(df, tr, te),
        "v2": lambda df, tr, te: fv.build_v2(df, tr, te),
        "v2_raw_nan": lambda df, tr, te: fv.build_v2_raw_nan(df, tr, te),
        "v4_from_v2": lambda df, tr, te: fv.build_v4_from_v2(df, tr, te, pool),
    }

    # ---------------------------------------------------------- Tier 1
    print("=== Tier 1: Random Forest (4 versiyon, Strateji B) ===", flush=True)
    rf_results = {}
    for version_name, builder in builders.items():
        print(f"--- random_forest x {version_name} ---", flush=True)
        results = m.nested_cv_evaluate(v1_df, builder, em.fit_predict_rf, em.RF_GRID, "random_forest", version_name)
        append_with_variant(results, B_LABEL)
        rf_results[version_name] = results
        print(f"    f1={results['f1'].mean():.4f}+/-{results['f1'].std():.4f}", flush=True)

    best_rf_version = max(rf_results, key=lambda v: rf_results[v]["f1"].mean())
    print(f"Tier 1: en iyi RF veri versiyonu = {best_rf_version} "
          f"(f1={rf_results[best_rf_version]['f1'].mean():.4f})", flush=True)

    print("=== Tier 1: KNN (yalnizca v4_from_v2) ===", flush=True)
    knn_results = m.nested_cv_evaluate(v1_df, builders["v4_from_v2"], em.fit_predict_knn, em.KNN_GRID, "knn", "v4_from_v2")
    append_with_variant(knn_results, NA_LABEL)
    print(f"    f1={knn_results['f1'].mean():.4f}+/-{knn_results['f1'].std():.4f}", flush=True)

    # ---------------------------------------------------------- Tier 2
    print(f"=== Tier 2: A/C agirliklandirma -- RF/{best_rf_version} ===", flush=True)
    for label in ("A_no_weight", "C_data_driven_spw"):
        weight_fn = WEIGHTING_STRATEGIES[label]
        print(f"--- random_forest x {best_rf_version} x {label} ---", flush=True)
        results = m.nested_cv_evaluate(
            v1_df, builders[best_rf_version], em.make_fit_predict_rf(weight_fn), em.RF_GRID,
            "random_forest", best_rf_version,
        )
        append_with_variant(results, label)
        print(f"    f1={results['f1'].mean():.4f}+/-{results['f1'].std():.4f} "
              f"spec={results['specificity'].mean():.4f}", flush=True)

    print("=== Tier 2: A/C agirliklandirma -- catboost/v4_from_v2 ===", flush=True)
    for label in ("A_no_weight", "C_data_driven_spw"):
        weight_fn = WEIGHTING_STRATEGIES[label]
        print(f"--- catboost x v4_from_v2 x {label} ---", flush=True)
        results = m.nested_cv_evaluate(
            v1_df, builders["v4_from_v2"], em.make_fit_predict_catboost(weight_fn), m.CATBOOST_GRID,
            "catboost", "v4_from_v2",
        )
        append_with_variant(results, label)
        print(f"    f1={results['f1'].mean():.4f}+/-{results['f1'].std():.4f} "
              f"spec={results['specificity'].mean():.4f}", flush=True)

    print("Tamamlandi.", flush=True)


if __name__ == "__main__":
    main()
