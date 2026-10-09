"""Asama F1: adversarial validation -- final 26-ozellik havuzunda (P1
madde 11'in `al_all_missing`/`CAT_1`'i cikarmasindan SONRA) baska bir
provenance/kaynak-kisayolu sinyali kaldi mi?

Metodolojik not: bu bir tanisal/kesif taramasidir (final modelin nested
performans tahmini DEGIL) -- Asama A'nin `univariate_auc_scan.csv`/
`03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md` ile ayni ruhta. Ozellik
matrisi (`build_v4_from_v2`) TUM 369 satirda BIR KEZ fit edilir (fold-ici
yeniden-fit degil) -- imputasyon sabitleri (AL_ sifir, EK_3 medyani)
kaynak-etiketiyle iliskili olmadigi icin bu basitlestirme dusuk risklidir,
ama seffaflik icin belirtiliyor. Adversarial siniflandirici, `feature_
selection.py::_lgb_model` ile AYNI "yardimci model" felsefesini paylasan
kucuk bir LightGBM (final model AILESI DEGIL, yalnizca taniyici arac).

Kaynak-1 proxy notu: veri setinde acik bir "ClinVar" bayragi YOK. `CAT_1`
(gnomAD alt-populasyon etiketi) dolu olmasi "gnomAD'den frekans/populasyon
kanitiyla geldi" anlamina gelir; `CAT_1` bos olmasi ise byte-bazli bir
ClinVar-KESIN kaniti degil, yalnizca "gnomAD populasyon-frekans kanitina
sahip degil" (ClinVar-agirlikli bir alt-kume icin makul bir proxy)
anlamina gelir. Bu proxy niteligi raporda acikca belirtiliyor.

Calistirma: python -m genova.pah.f1_adversarial_validation
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from genova.pah import fold_versions as fv

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
MODEL_DIR = ROOT / "models" / "pah"
BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
TAB_DIR = ROOT / "reports" / "tables"
SOURCE_CV_OUT = TAB_DIR / "f1_adversarial_source_cv_auc.csv"
FEATURE_AUC_OUT = TAB_DIR / "f1_adversarial_per_feature_auc.csv"

SEED = 42
N_SPLITS = 5
FEATURE_AUC_THRESHOLD = 0.75


def _lgb_helper(seed):
    return lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.1,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )


def adversarial_cv_auc(X, y, groups, seed=SEED, n_splits=N_SPLITS):
    """Grup-farkinda CV ile adversarial AUC -- yardimci LightGBM, final
    model DEGIL."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y, groups):
        model = _lgb_helper(seed).fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return np.array(aucs)


def build_pool_features_full(v1_df, pool):
    all_ids = v1_df["Variant_ID"].tolist()
    X, y, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    return X, y


def main():
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    pool = sorted(bundle["pool"])
    print(f"final havuz: {len(pool)} ozellik (madde 11 sonrasi -- al_all_missing/CAT_1 zaten yok)", flush=True)

    X_all, _ = build_pool_features_full(v1_df, pool)
    groups_all = v1_df["group_id"].to_numpy()
    benign_mask = (v1_df["Label"] == 0).to_numpy()
    X_benign = X_all[benign_mask].reset_index(drop=True)
    groups_benign = groups_all[benign_mask]

    print(f"\n=== Adim 1: Kaynak-1 proxy (CAT_1 dolu [gnomAD] vs bos [ClinVar-agirlikli proxy], yalniz benign) ===", flush=True)
    y_source1 = v1_df.loc[benign_mask, "CAT_1"].notna().astype(int).to_numpy()
    print(f"n={len(y_source1)}, CAT_1-dolu={y_source1.sum()}, CAT_1-bos={len(y_source1) - y_source1.sum()}", flush=True)
    aucs1 = adversarial_cv_auc(X_benign, y_source1, groups_benign)
    print(f"AUC (5-fold, grup-farkinda): ort={aucs1.mean():.4f} std={aucs1.std():.4f}", flush=True)

    print(f"\n=== Adim 2: Kaynak-2 (CAT_2 dolu [AllofUs] vs bos, yalniz benign) ===", flush=True)
    y_source2 = v1_df.loc[benign_mask, "CAT_2"].notna().astype(int).to_numpy()
    print(f"n={len(y_source2)}, CAT_2-dolu={y_source2.sum()}, CAT_2-bos={len(y_source2) - y_source2.sum()}", flush=True)
    print(f"not: CAT_2 havuzda GIRDI olarak yok ({'CAT_2' in pool}) -- yalnizca DIŞLANMIŞ hedef degisken olarak kullaniliyor.", flush=True)
    aucs2 = adversarial_cv_auc(X_benign, y_source2, groups_benign)
    print(f"AUC (5-fold, grup-farkinda): ort={aucs2.mean():.4f} std={aucs2.std():.4f}", flush=True)

    pd.DataFrame([
        {"test": "source1_clinvar_proxy_vs_gnomad", "n": len(y_source1), "n_positive": int(y_source1.sum()),
         "auc_mean": aucs1.mean(), "auc_std": aucs1.std(), "auc_per_fold": list(aucs1)},
        {"test": "source2_allofus_vs_not", "n": len(y_source2), "n_positive": int(y_source2.sum()),
         "auc_mean": aucs2.mean(), "auc_std": aucs2.std(), "auc_per_fold": list(aucs2)},
    ]).to_csv(SOURCE_CV_OUT, index=False)
    print(f"\nkaydedildi: {SOURCE_CV_OUT}", flush=True)

    print(f"\n=== Adim 3: ozellik-bazli tek-degiskenli adversarial AUC (26 ozellik x 2 hedef) ===", flush=True)
    rows = []
    for col in X_all.columns:
        vals = X_benign[col].to_numpy()
        finite = ~np.isnan(vals)
        auc1 = roc_auc_score(y_source1[finite], vals[finite])
        auc1 = max(auc1, 1 - auc1)
        auc2 = roc_auc_score(y_source2[finite], vals[finite])
        auc2 = max(auc2, 1 - auc2)
        rows.append({
            "feature": col, "n_finite": int(finite.sum()),
            "auc_source1_clinvar_proxy_vs_gnomad": auc1,
            "auc_source2_allofus_vs_not": auc2, "max_auc": max(auc1, auc2),
        })
    result = pd.DataFrame(rows).sort_values("max_auc", ascending=False)
    print(result.to_string(index=False), flush=True)
    result.to_csv(FEATURE_AUC_OUT, index=False)
    print(f"\nkaydedildi: {FEATURE_AUC_OUT}", flush=True)

    exceeds = result[result["max_auc"] > FEATURE_AUC_THRESHOLD]
    print(f"\n=== SONUC: >{FEATURE_AUC_THRESHOLD} esigini gecen ozellik sayisi: {len(exceeds)} ===", flush=True)
    if len(exceeds) > 0:
        print(exceeds.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
