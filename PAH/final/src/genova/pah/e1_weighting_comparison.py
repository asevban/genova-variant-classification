"""Asama E1-EK: E1'in AYNI sabit-hiperparametreli XGBoost/v1 modelini
(yeniden aranmiyor, `e1_baseline.py`'den `build_v1_features`/`FIXED_
HYPERPARAMS`/`PDR_THRESHOLD` aynen ice aktarilir), ayni 50 dis fold'da,
uc agirliklandirma stratejisiyle yeniden olcer:

  A - agirliksiz
  B - sabit azinlik x2.0 (E1'in orijinal politikasi -- burada YENIDEN
      CALISTIRILMAZ, `models/pah/baseline_frozen.pkl`'deki mevcut sonuc
      okunur)
  C - veri-gudumlu scale_pos_weight (her fold kendi train'inden)

Tek degisken agirliklandirma -- hiperparametre, esik (0,359, E1'le ayni),
veri versiyonu (v1) hepsi sabit tutulur.

Calistirma: python -m genova.pah.e1_weighting_comparison
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah.e1_baseline import V1_PATH, SPLITS_DIR, MODELS_DIR, N_REPEATS, FIXED_HYPERPARAMS, PDR_THRESHOLD, build_v1_features
from genova.pah.weighting import weights_no_weight, weights_data_driven_spw

OUT_CSV = Path(__file__).resolve().parents[3] / "reports" / "tables" / "e1ek_weighting_comparison.csv"

VARIANTS = {
    "A_no_weight": weights_no_weight,
    "C_data_driven_spw": weights_data_driven_spw,
}


def evaluate_variant(v1_df, weight_fn, label):
    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for fold in outer["folds"]:
            X_train, y_train, X_test, y_test, _ = build_v1_features(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
            weights = weight_fn(y_train)
            model = XGBClassifier(**FIXED_HYPERPARAMS)
            model.fit(X_train, y_train, sample_weight=weights)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= PDR_THRESHOLD).astype(int)
            rows.append({
                "variant": label,
                "repeat": repeat_idx,
                "outer_fold": fold["fold"],
                "n_train": len(y_train),
                "n_test": len(y_test),
                "f1": f1_binary_positive(y_test, y_pred),
                "mcc": matthews_correlation_coefficient(y_test, y_pred),
                "specificity": specificity(y_test, y_pred),
            })
            print(f"  [{label}] repeat={repeat_idx} outer={fold['fold']} done", flush=True)
    return pd.DataFrame(rows)


def load_variant_b():
    """B: E1'in orijinal sonucu -- yeniden calistirilmiyor, `baseline_
    frozen.pkl`'den okunuyor."""
    with open(MODELS_DIR / "baseline_frozen.pkl", "rb") as f:
        bundle = pickle.load(f)
    df = bundle["outer_cv_results_v1"][["repeat", "outer_fold", "n_train", "n_test", "f1", "mcc", "specificity"]].copy()
    df["variant"] = "B_fixed_minority_2x"
    return df


def summarize(df):
    return df.groupby("variant")[["f1", "mcc", "specificity"]].agg(["mean", "std"])


def main():
    v1_df = pd.read_parquet(V1_PATH)

    results = [load_variant_b()]
    for label, weight_fn in VARIANTS.items():
        print(f"=== {label} (50 dis fold) ===", flush=True)
        results.append(evaluate_variant(v1_df, weight_fn, label))

    all_results = pd.concat(results, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(OUT_CSV, index=False)
    print(summarize(all_results))
    print(f"kaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
