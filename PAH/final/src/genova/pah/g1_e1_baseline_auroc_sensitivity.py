"""Revize turu, Adim 4'un E1 satiri icin girdi: E1 baseline'in (v1,
XGBoost, SABIT PDR hiperparametreleri -- `models/pah/baseline_frozen.
pkl` ile AYNI FIXED_HYPERPARAMS/esik, hicbir yeniden arama yok) 50 dis
fold'unda AUROC + Sensitivity. `baseline_frozen.pkl` DEGISTIRILMIYOR --
zaten kaydedilmis f1/mcc/specificity'nin YANINA, yeni bir CSV'ye ayrica
yaziliyor (P1 madde 19 idempotent deseniyle degil, bu tek seferlik bir
turdur).

Calistirma: python -m genova.pah.g1_e1_baseline_auroc_sensitivity
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah.e1_baseline import (
    V1_PATH, SPLITS_DIR, FIXED_HYPERPARAMS, PDR_THRESHOLD,
    N_REPEATS, build_v1_features, _minority_sample_weight,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_CSV = ROOT / "reports" / "tables" / "g1_e1_baseline_auroc_sensitivity.csv"


def main():
    v1_df = pd.read_parquet(V1_PATH)
    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for fold in outer["folds"]:
            X_train, y_train, X_test, y_test, _ = build_v1_features(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
            weights, _ = _minority_sample_weight(y_train)
            model = XGBClassifier(**FIXED_HYPERPARAMS)
            model.fit(X_train, y_train, sample_weight=weights)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= PDR_THRESHOLD).astype(int)
            rows.append({
                "repeat": repeat_idx, "outer_fold": fold["fold"],
                "f1": f1_binary_positive(y_test, y_pred),
                "mcc": matthews_correlation_coefficient(y_test, y_pred),
                "specificity": specificity(y_test, y_pred),
                "sensitivity": sensitivity(y_test, y_pred),
                "auroc": roc_auc_score(y_test, proba),
            })
            print(f"  repeat={repeat_idx} outer={fold['fold']} f1={rows[-1]['f1']:.4f} auroc={rows[-1]['auroc']:.4f}", flush=True)

    result = pd.DataFrame(rows)
    print("\n=== ozet (50 dis fold) ===")
    print(result[["f1", "mcc", "specificity", "sensitivity", "auroc"]].agg(["mean", "std"]))
    result.to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
