"""Inceleme raporu Bolum 34: persistent-hard-benign hata analizi. SAF
TESHIS -- hicbir model/esik/dosya degistirilmez, karar kurali yok.

61 benign satirin her biri, mevcut 50-outer-fold split bankasinin
(10 tekrar x 5 dis-fold) KENDI dis-test'ine dustugu her fold'da nasil
skorlandi -- bu, `e4_prior_corrected_probabilities.csv`'nin (E3->E4
hattinin zaten urettigi, model=catboost/data_version=v4_from_v2/
method=beta_sld) satirlarindan DOGRUDAN okunuyor, hicbir model yeniden
FIT EDILMIYOR. Sabit final esik (bundle'dan okunur, 0,35) her fold'un
SLD-duzeltilmis olasiligina uygulanip dogru/yanlis sayiliyor.

SHAP (Adim 3): `f4_shap_explainability.py` ile AYNI mekanizma (`shap.
TreeExplainer` + `fold_versions.build_v4_from_v2`, final fit modeli
DEGISTIRILMEDEN) -- F4 yalnizca AGREGE mean|SHAP| CSV'sini persist
etmisti, per-satir ham SHAP degerlerini DEGIL; bu yuzden ayni deterministik
aciklama COK UCUZ bir turda (369 satir, saniyeler) yeniden calistirilip
yalnizca persistent-hard satirlarin kendi katkilari cikariliyor -- bu
yeniden EGITIM degil, F4'un zaten yaptigi aciklamanin tekrari.

Calistirma: python -m genova.pah.f_persistent_hard_benign_analysis
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from genova.pah import fold_versions as fv
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, MODEL_DIR

BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
E4_CSV = TAB_DIR / "e4_prior_corrected_probabilities.csv"
OUT_CSV = TAB_DIR / "f_persistent_hard_benign_error_stats.csv"

EASY_MAX_ERROR_RATE = 0.10
HARD_MIN_ERROR_RATE = 0.70
N_SHAP_TOP = 5


def per_row_error_stats(threshold):
    """Adim 1: e4'un 50-outer-fold SLD-olasiliklarindan, 61 benign satirin
    her biri icin n_folds_seen/n_folds_wrong/error_rate/mean_P/std_P."""
    e4 = pd.read_csv(E4_CSV)
    sub = e4[(e4.model == "catboost") & (e4.data_version == "v4_from_v2")
              & (e4.method == "beta_sld") & (e4.y_true == 0)].copy()
    sub["wrong"] = (sub["proba"] >= threshold).astype(int)  # y_true=0 icin: yanlis <=> pred=1

    stats = sub.groupby("variant_id").agg(
        n_folds_seen=("wrong", "size"), n_folds_wrong=("wrong", "sum"),
        mean_P=("proba", "mean"), std_P=("proba", "std"),
    ).reset_index()
    stats["error_rate"] = stats["n_folds_wrong"] / stats["n_folds_seen"]
    return stats


def categorize(stats):
    conditions = [stats["error_rate"] <= EASY_MAX_ERROR_RATE, stats["error_rate"] >= HARD_MIN_ERROR_RATE]
    stats = stats.copy()
    stats["category"] = np.select(conditions, ["easy", "persistent_hard"], default="unstable")
    return stats


def attach_structural_features(stats, v1_df):
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    al_rate = v1_df[al_cols].isna().mean(axis=1)
    tertile = pd.Series(pd.qcut(al_rate, 3, labels=["dusuk", "orta", "yuksek"]), index=v1_df.index)
    struct = pd.DataFrame({
        "variant_id": v1_df["Variant_ID"],
        "cat1_status": v1_df["CAT_1"].notna().map({True: "dolu", False: "bos"}).to_numpy(),
        "cat2": v1_df["CAT_2"].astype(object).fillna("EMPTY").to_numpy(),
        "al_tertile": tertile.to_numpy(),
        "al_missing_rate": al_rate.to_numpy(),
    })
    return stats.merge(struct, on="variant_id", how="left")


def compute_shap_top_features(v1_df, persistent_hard_ids):
    """F4 ile ayni mekanizma (TreeExplainer, ayni final-fit modeli,
    degistirilmedi) -- yalnizca persistent-hard satirlar icin top-N
    |SHAP| katkisi cikarilir."""
    if len(persistent_hard_ids) == 0:
        return {}
    bundle = joblib.load(BUNDLE_PATH)
    pool = sorted(bundle["pool"])
    all_ids = v1_df["Variant_ID"].tolist()
    X_all, _, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    X_all = X_all[pool]

    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(X_all)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_by_id = {vid: i for i, vid in enumerate(v1_df["Variant_ID"])}
    top_features = {}
    for vid in persistent_hard_ids:
        i = row_by_id[vid]
        contrib = pd.Series(shap_values[i], index=pool).sort_values(key=np.abs, ascending=False)
        top_features[vid] = contrib.head(N_SHAP_TOP)
    return top_features


def main():
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    threshold = bundle["threshold"]
    print(f"final esik (bundle'dan okundu, degistirilmedi): {threshold}", flush=True)

    print("\n=== Adim 1: 61 benign satirin per-fold hata istatistikleri (50 dis-fold, e4 CSV'sinden) ===", flush=True)
    stats = per_row_error_stats(threshold)
    assert len(stats) == 61, f"61 benzersiz benign bekleniyordu, {len(stats)} bulundu"
    assert (stats["n_folds_seen"] == 10).all(), "her satir tam 10 repeat'te (kendi dis-test'inde) gorulmeli"

    print("\n=== Adim 2: kategorizasyon ===", flush=True)
    stats = categorize(stats)
    counts = stats["category"].value_counts()
    print(counts.to_string(), flush=True)
    print(f"  easy: error_rate<={EASY_MAX_ERROR_RATE}, persistent_hard: error_rate>={HARD_MIN_ERROR_RATE}, "
          f"unstable: arada", flush=True)

    print("\n=== Adim 3: yapisal ozellikler (CAT_1/AL_/CAT_2) ===", flush=True)
    stats = attach_structural_features(stats, v1_df)

    hard = stats[stats.category == "persistent_hard"].sort_values("error_rate", ascending=False)
    print(f"\npersistent-hard grubu (n={len(hard)}):", flush=True)
    print(hard[["variant_id", "error_rate", "mean_P", "std_P", "cat1_status", "cat2", "al_tertile"]].to_string(index=False), flush=True)

    print("\n=== Adim 3 (devam): SHAP top-{} katkilari (persistent-hard satirlar) ===".format(N_SHAP_TOP), flush=True)
    shap_top = compute_shap_top_features(v1_df, hard["variant_id"].tolist())
    for vid, contrib in shap_top.items():
        print(f"  {vid}:", flush=True)
        for feat, val in contrib.items():
            print(f"      {feat}: {val:+.4f}", flush=True)

    unstable = stats[stats.category == "unstable"]
    print(f"\n=== Adim 4: unstable grubu (n={len(unstable)}) CAT_1/AL_ dagilimi ===", flush=True)
    print(pd.crosstab(unstable["cat1_status"], unstable["al_tertile"]).to_string(), flush=True)
    print("\ntum satirlarla (61) karsilastirma icin CAT_1/AL_ dagilimi:", flush=True)
    print(pd.crosstab(stats["cat1_status"], stats["al_tertile"]).to_string(), flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
