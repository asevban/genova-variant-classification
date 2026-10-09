"""E2-sonrasi dogrulama, Kisim 2: ucuncu E3 adayi (LightGBM mi Elastic-Net
mi) icin varsayima degil olcume dayali karar. CatBoost/v1, LightGBM/v1 ve
Elastic-Net/v3'un out-of-fold (OOF) olasilik tahminlerini uretir --
repeat=0'in 5 dis fold'u tam bir 369-satirlik partisyon oldugu icin
(bkz. split bankasi, her satir tam olarak bir kez test'te) bu, sizintisiz
tam bir OOF vektoru verir.

Hiperparametre YENIDEN ARANMIYOR: her fold icin, E2'nin nested_cv_
evaluate'inin O FOLD'da zaten sectigi `best_params`
(`reports/tables/e2_model_comparison.csv`) aynen yeniden kullanilir --
E2 sirasinda olasiliklar saklanmadigi icin (yalnizca esiklenmis skorlar),
bu script yalnizca AYNI konfigurasyonla yeniden fit edip proba'yi
yakalıyor.

Calistirma: python -m genova.pah.e2_oof_correlation
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from genova.pah import fold_versions as fv
from genova.pah import models as m

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
AL_CLUSTERS_PATH = ROOT / "reports" / "tables" / "AL_correlation_clusters.csv"
REPEAT_FOR_OOF = 0


def _variant_ids_in_row_order(v1_df, ids):
    """fold_versions._split()'in ayni filtre+reset_index mantigi -- X_*'in
    satir sirasiyla birebir hizali Variant_ID listesi doner."""
    return v1_df[v1_df["Variant_ID"].isin(ids)]["Variant_ID"].reset_index(drop=True)


def compute_oof(v1_df, best_params_df, model_name, version_name, builder, fit_predict_fn):
    fold_rows = best_params_df[
        (best_params_df.model == model_name) & (best_params_df.data_version == version_name)
        & (best_params_df.repeat == REPEAT_FOR_OOF)
    ].sort_values("outer_fold")
    assert len(fold_rows) == 5, f"{model_name}/{version_name}: repeat={REPEAT_FOR_OOF} icin 5 dis fold bekleniyor, {len(fold_rows)} bulundu"

    outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{REPEAT_FOR_OOF:02d}.json").read_text())
    oof = {}
    for _, row in fold_rows.iterrows():
        fold = outer["folds"][int(row["outer_fold"])]
        best_params = json.loads(row["best_params"])
        X_train, y_train, X_test, y_test = builder(v1_df, fold["train_variant_ids"], fold["test_variant_ids"])
        proba = fit_predict_fn(X_train, y_train, X_test, best_params)
        test_variant_ids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p in zip(test_variant_ids, proba):
            oof[vid] = p
    return pd.Series(oof, name=f"{model_name}_{version_name}")


def main():
    v1_df = pd.read_parquet(V1_PATH)
    al_clusters = pd.read_csv(AL_CLUSTERS_PATH)
    best_params_df = pd.read_csv(COMPARISON_CSV)

    def elasticnet_builder(df, tr, te):
        return fv.build_v3_full(df, tr, te, al_clusters)

    oof_catboost = compute_oof(v1_df, best_params_df, "catboost", "v1", fv.build_v1, m.fit_predict_catboost)
    print("catboost/v1 OOF hazir, n=", len(oof_catboost), flush=True)
    oof_lightgbm = compute_oof(v1_df, best_params_df, "lightgbm", "v1", fv.build_v1, m.fit_predict_lgbm)
    print("lightgbm/v1 OOF hazir, n=", len(oof_lightgbm), flush=True)
    oof_elasticnet = compute_oof(v1_df, best_params_df, "elasticnet", "v3", elasticnet_builder, m.fit_predict_elasticnet)
    print("elasticnet/v3 OOF hazir, n=", len(oof_elasticnet), flush=True)

    oof_df = pd.DataFrame({
        "catboost_v1": oof_catboost, "lightgbm_v1": oof_lightgbm, "elasticnet_v3": oof_elasticnet,
    }).dropna()
    assert len(oof_df) == 369, f"beklenen 369 satir, bulunan {len(oof_df)}"

    OUT_CSV = ROOT / "reports" / "tables" / "e2_oof_predictions_repeat0.csv"
    oof_df.to_csv(OUT_CSV)
    print(f"kaydedildi: {OUT_CSV}")

    r_cb_lgbm, p_cb_lgbm = pearsonr(oof_df["catboost_v1"], oof_df["lightgbm_v1"])
    r_cb_en, p_cb_en = pearsonr(oof_df["catboost_v1"], oof_df["elasticnet_v3"])
    print(f"Pearson(CatBoost/v1, LightGBM/v1)   = {r_cb_lgbm:.4f} (p={p_cb_lgbm:.2e})")
    print(f"Pearson(CatBoost/v1, ElasticNet/v3) = {r_cb_en:.4f} (p={p_cb_en:.2e})")


if __name__ == "__main__":
    main()
