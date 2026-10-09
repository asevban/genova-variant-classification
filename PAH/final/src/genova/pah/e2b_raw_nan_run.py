"""Asama E2b calistiricisi: uc agac modelinin (CatBoost/XGBoost/LightGBM)
`v2_raw_nan` (AL_ sifir-dolgu yerine ham NaN) versiyonundaki performansi.
`v2` (sifir-dolgu) ile karsilastirma icin.

Calistirma: python -m genova.pah.e2b_raw_nan_run
"""
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import models as m

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"


def main():
    v1_df = pd.read_parquet(V1_PATH)

    def v2_raw_nan_builder(df, tr, te):
        return fv.build_v2_raw_nan(df, tr, te)

    combos = [
        ("catboost", "v2_raw_nan", v2_raw_nan_builder, m.fit_predict_catboost, m.CATBOOST_GRID),
        ("xgboost", "v2_raw_nan", v2_raw_nan_builder, m.fit_predict_xgb, m.XGB_GRID),
        ("lightgbm", "v2_raw_nan", v2_raw_nan_builder, m.fit_predict_lgbm, m.LGBM_GRID),
    ]

    for model_name, version_name, builder, fit_predict_fn, grid in combos:
        print(f"=== {model_name} x {version_name} ({len(grid)} aday, 50 dis fold) ===", flush=True)
        results = m.nested_cv_evaluate(v1_df, builder, fit_predict_fn, grid, model_name, version_name)
        m.append_results(results)
        print(f"--- {model_name} x {version_name} bitti: {m.summarize(results)}", flush=True)


if __name__ == "__main__":
    main()
