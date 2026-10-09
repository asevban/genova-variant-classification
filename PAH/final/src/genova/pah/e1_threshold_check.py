"""E2-sonrasi dogrulama, Kisim 1: "genis specificity iyilesmesi kismen
esik-artefakti" hipotezini olcer. E1'in AYNI sabit-hiperparametreli
modelini (yeniden aranmiyor, `e1_baseline.py`'den ithal edilen ayni
tarif/ayni 50 dis fold) yeniden calistirir -- tek fark, bu kez olasilik
tahminleri saklanir, boylece ayni model AYNI olasiliklardan iki farkli
esikle (0,359 PDR-esigi vs 0,5 E2-esigi) skorlanabilir. Model yeniden
ARANMIYOR/DEGISMIYOR -- yalnizca ikilileme esigi degisiyor.

Calistirma: python -m genova.pah.e1_threshold_check
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah.e1_baseline import (
    V1_PATH, SPLITS_DIR, MODELS_DIR, N_REPEATS,
    FIXED_HYPERPARAMS, PDR_THRESHOLD,
    build_v1_features, _minority_sample_weight,
)

THRESHOLDS_TO_COMPARE = [PDR_THRESHOLD, 0.5]
OUT_PATH = MODELS_DIR / "e1_threshold_check_proba.pkl"


def run():
    v1_df = pd.read_parquet(V1_PATH)
    fold_records = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for fold in outer["folds"]:
            X_train, y_train, X_test, y_test, _ = build_v1_features(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
            weights, _ = _minority_sample_weight(y_train)
            model = XGBClassifier(**FIXED_HYPERPARAMS)
            model.fit(X_train, y_train, sample_weight=weights)
            proba = model.predict_proba(X_test)[:, 1]
            fold_records.append({
                "repeat": repeat_idx, "outer_fold": fold["fold"],
                "y_test": y_test.to_numpy(), "proba": proba,
            })
            print(f"repeat={repeat_idx} outer={fold['fold']} done", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(fold_records, f)
    print(f"kaydedildi: {OUT_PATH}")
    return fold_records


def summarize(fold_records, thresholds=THRESHOLDS_TO_COMPARE):
    rows = []
    for rec in fold_records:
        y_test, proba = rec["y_test"], rec["proba"]
        row = {"repeat": rec["repeat"], "outer_fold": rec["outer_fold"]}
        for thr in thresholds:
            y_pred = (proba >= thr).astype(int)
            row[f"f1@{thr}"] = f1_binary_positive(y_test, y_pred)
            row[f"mcc@{thr}"] = matthews_correlation_coefficient(y_test, y_pred)
            row[f"spec@{thr}"] = specificity(y_test, y_pred)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    records = run()
    df = summarize(records)
    for thr in THRESHOLDS_TO_COMPARE:
        print(f"esik={thr}: F1={df[f'f1@{thr}'].mean():.4f}+/-{df[f'f1@{thr}'].std():.4f}, "
              f"specificity={df[f'spec@{thr}'].mean():.4f}+/-{df[f'spec@{thr}'].std():.4f}")
