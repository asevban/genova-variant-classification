"""Asama E3-EK: RF/v4_from_v2 (Strateji B) icin nested kalibrasyon --
E3'un orijinal `run_model_calibration` cercevesini (cok-fit -> Beta
kalibrasyon -> yalnizca transform dis-teste) aynen yeniden kullanir.
CatBoost/v4_from_v2 ve LightGBM/v1 YENIDEN KALIBRE EDILMEZ -- mevcut
`e3_calibration_metrics.csv`/`e3_calibrated_oof_predictions.csv`
satirlari korunur, yalnizca RF icin yeni satirlar EKLENIR.

`e2_model_comparison.csv` artik random_forest/v4_from_v2 icin 3
agirliklandirma varyanti (A/B/C) icerdigi icin, `best_params` araması
yalnizca Strateji B satirlarina filtrelenerek yapilir (E3 Adim 0'in
sectigi varyant).

Calistirma: python -m genova.pah.e3ek_rf_calibration
"""
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import e2ek_models as em
from genova.pah.e3_calibration_run import run_model_calibration
from genova.pah.idempotent_io import upsert_csv

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
METRICS_OUT = ROOT / "reports" / "tables" / "e3_calibration_metrics.csv"
OOF_OUT = ROOT / "reports" / "tables" / "e3_calibrated_oof_predictions.csv"

WEIGHTING_VARIANT = "B_fixed_minority_2x"


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    comparison_b_only = comparison_df[comparison_df.weighting_variant == WEIGHTING_VARIANT]
    pool = json.loads(POOL_PATH.read_text())["features"]

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    print("=== random_forest x v4_from_v2 (Strateji B, 50 dis fold, capraz-fit kalibrasyon) ===", flush=True)
    metrics_df, oof_df = run_model_calibration(
        v1_df, comparison_b_only, "random_forest", "v4_from_v2", v4v2_builder, em.fit_predict_rf,
    )

    # P2 madde 19: idempotent yazar -- ayni anahtar icin script iki kez
    # calistirilirsa satir COGALMAZ.
    upsert_csv(metrics_df, METRICS_OUT, ["model", "data_version", "method", "repeat", "outer_fold"])
    upsert_csv(oof_df, OOF_OUT, ["model", "data_version", "method", "repeat", "outer_fold", "variant_id"])

    summary = metrics_df.groupby("method")[["brier", "logloss", "f1", "n_unique_proba"]].agg(["mean", "std"])
    print(summary, flush=True)
    print(f"eklendi: {METRICS_OUT}, {OOF_OUT}")


if __name__ == "__main__":
    main()
