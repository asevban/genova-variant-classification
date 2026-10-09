"""Revize turu, Adim 4'un birlesik tablosu icin ek girdi: `random_forest/
v4_from_v2` (Strateji B, `weighting_variant="B_fixed_minority_2x"`)
E2-EK kombinasyonu, "cekirdek 16" disinda oldugu icin `g1_e2_auroc_
sensitivity_replay.py`'nin kapsaminda DEGILDI -- ama Osman Hoca
formatindaki birlesik tablo bu satiri istiyor. AYNI REPLAY mantigi
(kayitli `best_params`, yeniden arama yok), yalnizca bu TEK kombinasyon
icin, `e2_model_comparison.csv`'YE YAZILMIYOR (kapsam siniri acik
kalsin diye) -- ayri bir CSV'ye.

Calistirma: python -m genova.pah.g1_rf_v4fromv2_auroc_sensitivity
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import sensitivity
from genova.pah import fold_versions as fv
from genova.pah.e2ek_models import fit_predict_rf

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
OUT_CSV = ROOT / "reports" / "tables" / "g1_rf_v4fromv2_auroc_sensitivity.csv"

DECISION_THRESHOLD = 0.5


def main():
    v1_df = pd.read_parquet(V1_PATH)
    pool = json.loads(POOL_PATH.read_text())["features"]
    comparison_df = pd.read_csv(COMPARISON_CSV)
    rf_rows = comparison_df[(comparison_df.model == "random_forest") & (comparison_df.data_version == "v4_from_v2")
                             & (comparison_df.weighting_variant == "B_fixed_minority_2x")]
    assert len(rf_rows) == 50, f"50 dis-fold bekleniyordu, {len(rf_rows)} bulundu"

    rows = []
    for _, row in rf_rows.iterrows():
        repeat_idx, outer_fold_idx = int(row["repeat"]), int(row["outer_fold"])
        best_params = json.loads(row["best_params"])
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        outer_fold = outer["folds"][outer_fold_idx]
        X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"], pool)
        proba = fit_predict_rf(X_tr, y_tr, X_te, best_params)
        y_pred = (proba >= DECISION_THRESHOLD).astype(int)
        rows.append({
            "repeat": repeat_idx, "outer_fold": outer_fold_idx,
            "f1_replay": row["f1"], "auroc": roc_auc_score(y_te, proba), "sensitivity": sensitivity(y_te, y_pred),
        })
        print(f"  repeat={repeat_idx} outer={outer_fold_idx} auroc={rows[-1]['auroc']:.4f} "
              f"sens={rows[-1]['sensitivity']:.4f} (kayitli f1={row['f1']:.4f})", flush=True)

    result = pd.DataFrame(rows)
    print("\n=== ozet ===")
    print(result[["auroc", "sensitivity"]].agg(["mean", "std"]))
    result.to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
