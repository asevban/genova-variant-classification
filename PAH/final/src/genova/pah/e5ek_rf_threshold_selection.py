"""Asama E5-EK: RF/v4_from_v2 (Strateji B) icin nested esik secimi --
E5'in orijinal `process_fold`/`select_threshold` cercevesini (E5_threshold_
selection.py) birebir ayni disiplinle yeniden kullanir. CatBoost/v4_from_v2
ve LightGBM/v1 YENIDEN HESAPLANMAZ -- mevcut `e5_threshold_selection.csv`
satirlari (orijinal E5'ten, ayni Strateji B ile) dogrudan referans olarak
kullanilir, bu script yalnizca RF icin yeni satirlar EKLER.

`e2_model_comparison.csv` artik random_forest/v4_from_v2 icin 3
agirliklandirma varyanti icerdigi icin, `best_params` aramasi yalnizca
Strateji B satirlarina filtrelenerek yapilir (E3-EK/E4-EK ile ayni
desen).

Calistirma: python -m genova.pah.e5ek_rf_threshold_selection
"""
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import e2ek_models as em
from genova.pah.e5_threshold_selection import process_fold, N_REPEATS
from genova.pah.idempotent_io import upsert_csv

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
E4_CSV = ROOT / "reports" / "tables" / "e4_prior_corrected_probabilities.csv"
OUT_CSV = ROOT / "reports" / "tables" / "e5_threshold_selection.csv"

WEIGHTING_VARIANT = "B_fixed_minority_2x"
MODEL_NAME = "random_forest"
VERSION_NAME = "v4_from_v2"


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    comparison_b_only = comparison_df[comparison_df.weighting_variant == WEIGHTING_VARIANT]
    e4_df = pd.read_csv(E4_CSV)
    e4_sub = e4_df[(e4_df.model == MODEL_NAME) & (e4_df.data_version == VERSION_NAME)]
    pool = json.loads(POOL_PATH.read_text())["features"]

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            row = process_fold(
                v1_df, comparison_b_only, e4_sub, MODEL_NAME, VERSION_NAME, v4v2_builder, em.fit_predict_rf,
                repeat_idx, outer_fold["fold"],
            )
            rows.append(row)
            print(f"  [{MODEL_NAME}/{VERSION_NAME}] repeat={repeat_idx} outer={outer_fold['fold']} "
                  f"thr={row['chosen_threshold']:.2f} f1={row['f1_chosen']:.4f}", flush=True)

    result = pd.DataFrame(rows)
    # P2 madde 19: idempotent yazar -- ayni anahtar icin script iki kez
    # calistirilirsa satir COGALMAZ.
    upsert_csv(result, OUT_CSV, ["model", "data_version", "repeat", "outer_fold"])
    print(f"yazildi (idempotent): {OUT_CSV} ({len(result)} satir)")
    print(result[["chosen_threshold", "f1_chosen", "weighted_f1_chosen", "weighted_f1_fixed05"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
