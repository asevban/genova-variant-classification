"""Asama E3: uc secilmis aday (CatBoost/v1, CatBoost/v4_from_v2,
LightGBM/v1) icin nested kalibrasyon. Hicbir hiperparametre yeniden
aranmiyor -- E2'nin `nested_cv_evaluate`'inin HER dis fold icin zaten
sectigi `best_params` (`e2_model_comparison.csv`) aynen yeniden kullanilir.

Sizinti-kritik adim (capraz-fit): kalibrator hicbir zaman taban modelin
kendi egitim verisiyle uretilen (in-sample) olasiliklarla fit edilmez.
Bunun yerine her dis fold icin:
  1. dis-train, split bankasinin kendi 4 ic fold'una gore capraz-fit
     edilir -- her ic fold'da taban model (ayni best_params ile) ic-train'de
     fit edilir, ic-val'de olasilik uretilir; 4 ic fold'un val kumeleri
     dis-train'in TAMAMINI sizintisiz kaplar (split_bank.py'nin
     StratifiedGroupKFold(ic-train uzerinde) tasarimi geregi).
  2. Kalibrator bu capraz-fit olasiliklarla + dis-train etiketleriyle fit
     edilir.
  3. Taban model TUM dis-train'de (E2'de oldugu gibi) yeniden fit edilir,
     dis-test'te ham olasilik uretilir, kalibrator bu ham olasiliga
     yalnizca TRANSFORM uygular.

Calistirma: python -m genova.pah.e3_calibration_run
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive
from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah.calibration import CALIBRATORS, BETACAL_AVAILABLE, evaluate_probabilities

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
METRICS_OUT = ROOT / "reports" / "tables" / "e3_calibration_metrics.csv"
OOF_OUT = ROOT / "reports" / "tables" / "e3_calibrated_oof_predictions.csv"

N_REPEATS = 10
DECISION_THRESHOLD = 0.5


def _variant_ids_in_row_order(v1_df, ids):
    """fold_versions._split()'in ayni filtre+reset_index mantigi -- X_*'in
    satir sirasiyla birebir hizali Variant_ID listesi doner."""
    return v1_df[v1_df["Variant_ID"].isin(ids)]["Variant_ID"].reset_index(drop=True)


def cross_fit_probabilities(v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx):
    """dis-train'in TAMAMI icin, sizintisiz capraz-fit olasiliklar.
    (variant_id -> proba) ve (variant_id -> y) sozlukleri doner."""
    inner_folds = json.loads(
        (SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{outer_fold_idx}.json").read_text()
    )["folds"]
    proba_by_vid, y_by_vid = {}, {}
    for inner_fold in inner_folds:
        X_tr, y_tr, X_va, y_va = builder(v1_df, inner_fold["train_variant_ids"], inner_fold["val_variant_ids"])
        proba = fit_predict_fn(X_tr, y_tr, X_va, best_params)
        val_vids = _variant_ids_in_row_order(v1_df, inner_fold["val_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    return proba_by_vid, y_by_vid


def _lookup_best_params(comparison_df, model_name, version_name, repeat_idx, outer_fold_idx):
    mask = (
        (comparison_df.model == model_name) & (comparison_df.data_version == version_name)
        & (comparison_df.repeat == repeat_idx) & (comparison_df.outer_fold == outer_fold_idx)
    )
    rows = comparison_df.loc[mask, "best_params"]
    assert len(rows) == 1, f"{model_name}/{version_name} repeat={repeat_idx} outer={outer_fold_idx}: {len(rows)} satir bulundu"
    return json.loads(rows.iloc[0])


def run_model_calibration(v1_df, comparison_df, model_name, version_name, builder, fit_predict_fn, n_repeats=N_REPEATS):
    metric_rows, oof_rows = [], []
    methods = {k: v for k, v in CALIBRATORS.items() if k != "beta" or BETACAL_AVAILABLE}

    for repeat_idx in range(n_repeats):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            outer_fold_idx = outer_fold["fold"]
            best_params = _lookup_best_params(comparison_df, model_name, version_name, repeat_idx, outer_fold_idx)

            proba_by_vid, y_by_vid = cross_fit_probabilities(
                v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx,
            )
            train_vids = list(proba_by_vid.keys())
            assert len(train_vids) == len(outer_fold["train_variant_ids"]), "capraz-fit dis-train'in tamamini kaplamadi"
            cross_fit_scores = np.array([proba_by_vid[v] for v in train_vids])
            y_dis_train = np.array([y_by_vid[v] for v in train_vids])

            X_train, y_train, X_test, y_test = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            raw_proba = fit_predict_fn(X_train, y_train, X_test, best_params)
            test_vids = _variant_ids_in_row_order(v1_df, outer_fold["test_variant_ids"])

            def _record(method_name, proba):
                metrics = evaluate_probabilities(y_test, proba)
                y_pred = (proba >= DECISION_THRESHOLD).astype(int)
                metric_rows.append({
                    "model": model_name, "data_version": version_name, "method": method_name,
                    "repeat": repeat_idx, "outer_fold": outer_fold_idx,
                    "n_dis_train": len(train_vids), "n_dis_test": len(test_vids),
                    "f1": f1_binary_positive(y_test, y_pred),
                    "n_unique_proba": len(np.unique(np.round(proba, 6))),
                    **metrics,
                })
                for vid, p, y in zip(test_vids, proba, y_test):
                    oof_rows.append({
                        "model": model_name, "data_version": version_name, "method": method_name,
                        "repeat": repeat_idx, "outer_fold": outer_fold_idx,
                        "variant_id": vid, "y_true": int(y), "proba": float(p),
                    })

            _record("raw", raw_proba)
            for method_name, cls in methods.items():
                calibrator = cls().fit(cross_fit_scores, y_dis_train)
                calibrated_proba = calibrator.transform(raw_proba)
                _record(method_name, calibrated_proba)

            print(f"  [{model_name}/{version_name}] repeat={repeat_idx} outer={outer_fold_idx} done", flush=True)

    return pd.DataFrame(metric_rows), pd.DataFrame(oof_rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    pool = json.loads((ROOT / "reports" / "tables" / "v4_final_feature_pool.json").read_text())["features"]

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    combos = [
        ("catboost", "v1", v1_builder, m.fit_predict_catboost),
        ("catboost", "v4_from_v2", v4v2_builder, m.fit_predict_catboost),
        ("lightgbm", "v1", v1_builder, m.fit_predict_lgbm),
    ]

    all_metrics, all_oof = [], []
    for model_name, version_name, builder, fit_predict_fn in combos:
        print(f"=== {model_name} x {version_name} (50 dis fold, capraz-fit kalibrasyon) ===", flush=True)
        metrics_df, oof_df = run_model_calibration(v1_df, comparison_df, model_name, version_name, builder, fit_predict_fn)
        all_metrics.append(metrics_df)
        all_oof.append(oof_df)
        summary = metrics_df.groupby("method")[["brier", "logloss", "f1"]].agg(["mean", "std"])
        print(summary, flush=True)

    pd.concat(all_metrics, ignore_index=True).to_csv(METRICS_OUT, index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(OOF_OUT, index=False)
    print(f"kaydedildi: {METRICS_OUT}, {OOF_OUT}")


if __name__ == "__main__":
    main()
