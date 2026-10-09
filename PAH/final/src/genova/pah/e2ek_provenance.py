"""Asama E2-EK Tier 3: provenance alt-grup performans kirilimi
(`PAH_Literatur_ve_Modelleme_Asamasi.md` SS38). Bu bir ABLASYON DEGIL --
model degistirilmiyor, hicbir hiperparametre yeniden aranmiyor. `e2_model_
comparison.csv`'nin zaten sectigi `best_params` aynen yeniden kullanilir
(deterministik, yalnizca satir-bazli tahmini yakalamak icin yeniden
hesaplanir -- E2-sonrasi/E3-E6'nin kurdugu ayni desen). Mevcut dis-fold
tahminleri `al_all_missing=0` / `al_all_missing=1` alt-gruplarina bolunup
AYRI AYRI skorlanir.

Calistirma: python -m genova.pah.e2ek_provenance
"""
import json
from pathlib import Path

import pandas as pd

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
OUT_CSV = ROOT / "reports" / "tables" / "e2ek_provenance_subgroup_predictions.csv"

N_REPEATS = 10


def _al_all_missing_by_variant(v1_df):
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    return v1_df.set_index("Variant_ID")[al_cols].isna().all(axis=1).astype(int)


def collect_predictions(v1_df, comparison_df, model_name, version_name, weighting_variant, builder, fit_predict_fn):
    comparison_sub = comparison_df[
        (comparison_df.model == model_name) & (comparison_df.data_version == version_name)
        & (comparison_df.weighting_variant == weighting_variant)
    ]
    al_all_missing = _al_all_missing_by_variant(v1_df)

    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for fold in outer["folds"]:
            row_match = comparison_sub[(comparison_sub.repeat == repeat_idx) & (comparison_sub.outer_fold == fold["fold"])]
            assert len(row_match) == 1, f"repeat={repeat_idx} outer={fold['fold']}: {len(row_match)} best_params satiri bulundu"
            best_params = json.loads(row_match["best_params"].iloc[0])

            X_train, y_train, X_test, y_test = builder(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
            proba = fit_predict_fn(X_train, y_train, X_test, best_params)
            pred = (proba >= 0.5).astype(int)
            test_vids = v1_df[v1_df["Variant_ID"].isin(fold["test_variant_ids"])]["Variant_ID"].reset_index(drop=True)

            for vid, p, yt in zip(test_vids, pred, y_test):
                rows.append({
                    "repeat": repeat_idx, "outer_fold": fold["fold"], "variant_id": vid,
                    "y_true": int(yt), "y_pred": int(p), "al_all_missing": int(al_all_missing[vid]),
                })
            print(f"  repeat={repeat_idx} outer={fold['fold']} done", flush=True)
    return pd.DataFrame(rows)


def score_subgroups(pred_df):
    results = {}
    for group in (0, 1):
        sub = pred_df[pred_df.al_all_missing == group]
        y_true, y_pred = sub["y_true"], sub["y_pred"]
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        results[group] = {
            "n": len(sub),
            "n_benign": int((y_true == 0).sum()),
            "n_pathogenic": int((y_true == 1).sum()),
            "f1": f1_binary_positive(y_true, y_pred),
            "mcc": matthews_correlation_coefficient(y_true, y_pred),
            "specificity": specificity(y_true, y_pred),
            "fp": fp,
            "fn": fn,
        }
    return results


def main(model_name, version_name, weighting_variant, builder, fit_predict_fn):
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)

    pred_df = collect_predictions(v1_df, comparison_df, model_name, version_name, weighting_variant, builder, fit_predict_fn)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}")

    results = score_subgroups(pred_df)
    print(f"al_all_missing=0 (n={results[0]['n']}): {results[0]}")
    print(f"al_all_missing=1 (n={results[1]['n']}): {results[1]}")
    return results


if __name__ == "__main__":
    import json as _json
    from genova.pah import fold_versions as fv

    pool = _json.loads((ROOT / "reports" / "tables" / "v4_final_feature_pool.json").read_text())["features"]

    def catboost_v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    from genova.pah import models as m
    main("catboost", "v4_from_v2", "B_fixed_minority_2x", catboost_v4v2_builder, m.fit_predict_catboost)
