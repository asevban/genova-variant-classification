"""Asama E2b calistiricisi: ZORT_* ve AL_-ozet aday ozellik setlerini,
Asama 2a'nin en iyi cikan model ailesiyle (CatBoost) v1 uzerine ekleyerek
dener. `e2_model_comparison.csv`'ye "v1_plus_zort" / "v1_plus_al_summary"
data_version etiketiyle eklenir.

Calistirma: python -m genova.pah.e2b_run
"""
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import e2_candidate_features as cf
from genova.pah import models as m

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"


def main():
    v1_df = pd.read_parquet(V1_PATH)
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]

    def zort_builder(df, tr, te):
        return cf.build_v1_plus_zort(df, tr, te, fv.build_v1)

    def al_summary_builder(df, tr, te):
        return cf.build_v1_plus_al_summary(df, tr, te, fv.build_v1, al_cols)

    combos = [
        ("catboost", "v1_plus_zort", zort_builder, m.fit_predict_catboost, m.CATBOOST_GRID),
        ("catboost", "v1_plus_al_summary", al_summary_builder, m.fit_predict_catboost, m.CATBOOST_GRID),
    ]

    for model_name, version_name, builder, fit_predict_fn, grid in combos:
        print(f"=== {model_name} x {version_name} ({len(grid)} aday, 50 dis fold) ===", flush=True)
        results = m.nested_cv_evaluate(v1_df, builder, fit_predict_fn, grid, model_name, version_name)
        m.append_results(results)
        print(f"--- {model_name} x {version_name} bitti: {m.summarize(results)}", flush=True)


if __name__ == "__main__":
    main()
