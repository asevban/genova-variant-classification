"""Asama F4: SHAP aciklanabilirlik. Final model (`final_model_bundle_v2.
pkl`) HICBIR SEKILDE DEGISTIRILMEZ/YENIDEN EGITILMEZ -- yalnizca kendi
CatBoost modeli `shap.TreeExplainer` ile 369 satirin tamaminda (final
fit'in kendi egitim verisi) aciklaniyor. Saf aciklama/gorsellestirme
turu, bir model karari degil.

F1/F3'un buldugu `AL_26/12/7/49` kaynak-korelasyonu ve `CAT_1`-gecis
kirilganligina SHAP'in nasil bir aciklanabilirlik katmani ekledigine
ozellikle odaklanir (Adim 2).

Calistirma: python -m genova.pah.f4_shap_explainability
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from genova.pah import fold_versions as fv
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR
from genova.pah.e4_prior_correction import sld_correct

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
STABILITY_CSV = TAB_DIR / "feature_selection_stability_fold_local.csv"
SHAP_IMPORTANCE_OUT = TAB_DIR / "f4_shap_feature_importance.csv"
COMPARISON_OUT = TAB_DIR / "f4_shap_vs_stage_d_comparison.csv"

RISKY_FEATURES = ["AL_26", "AL_12", "AL_7", "AL_49"]


def main():
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    pool = sorted(bundle["pool"])
    all_ids = v1_df["Variant_ID"].tolist()

    X_all, y_all, _, _ = fv.build_v4_from_v2(v1_df, all_ids, all_ids, pool)
    X_all = X_all[pool]
    model = bundle["model"]

    print(f"final model: {len(pool)} ozellik, {len(X_all)} satir", flush=True)

    # ---------------------------------------------------------------- Adim 1
    print("\n=== Adim 1: global SHAP ozeti ===", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else float(np.atleast_1d(expected_value)[0])

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=pool).sort_values(ascending=False)
    print(mean_abs_shap.head(10).to_string(), flush=True)
    mean_abs_shap.to_csv(SHAP_IMPORTANCE_OUT, header=["mean_abs_shap"])
    print(f"kaydedildi: {SHAP_IMPORTANCE_OUT}", flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_all, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "f4_shap_summary_beeswarm.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_all, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "f4_shap_summary_bar.png", dpi=150)
    plt.close(fig)
    print(f"kaydedildi: {FIG_DIR / 'f4_shap_summary_beeswarm.png'}, {FIG_DIR / 'f4_shap_summary_bar.png'}", flush=True)

    # ---------------------------------------------------------------- Adim 2
    print("\n=== Adim 2: riskli 4 ozelligin SHAP davranisi ===", flush=True)
    rank = mean_abs_shap.rank(ascending=False, method="min")
    for f in RISKY_FEATURES:
        print(f"  {f}: mean|SHAP|={mean_abs_shap[f]:.4f}  sira={int(rank[f])}/{len(pool)}", flush=True)

    cat1_filled = v1_df["CAT_1"].notna().to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, feat in zip(axes.ravel(), RISKY_FEATURES):
        idx = pool.index(feat)
        x_vals = X_all[feat].to_numpy()
        y_vals = shap_values[:, idx]
        ax.scatter(x_vals[~cat1_filled], y_vals[~cat1_filled], c="tab:red", alpha=0.6, label="CAT_1 bos", s=18)
        ax.scatter(x_vals[cat1_filled], y_vals[cat1_filled], c="tab:blue", alpha=0.6, label="CAT_1 dolu", s=18)
        ax.set_xlabel(f"{feat} (deger)")
        ax.set_ylabel("SHAP degeri")
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle("Riskli 4 ozelligin SHAP dependence'i, CAT_1 doluluguna gore renklendirilmis (F1/F3 baglantisi)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "f4_shap_risky_features_cat1_dependence.png", dpi=150)
    plt.close(fig)
    print(f"kaydedildi: {FIG_DIR / 'f4_shap_risky_features_cat1_dependence.png'}", flush=True)

    for feat in RISKY_FEATURES:
        idx = pool.index(feat)
        y_vals = shap_values[:, idx]
        print(f"  {feat}: CAT_1-bos ortalama |SHAP|={np.abs(y_vals[~cat1_filled]).mean():.4f}  "
              f"CAT_1-dolu ortalama |SHAP|={np.abs(y_vals[cat1_filled]).mean():.4f}", flush=True)

    # ---------------------------------------------------------------- Adim 3
    print("\n=== Adim 3: yerel aciklama ornekleri ===", flush=True)
    raw_proba = model.predict_proba(X_all)[:, 1]
    calibrated = bundle["calibrator"].transform(raw_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= bundle["threshold"]).astype(int)
    y_np = y_all.to_numpy()
    correct_mask = pred == y_np
    n_correct, n_wrong = int(correct_mask.sum()), int((~correct_mask).sum())
    print(f"in-sample (egitim verisinin kendisi -- SHAP'in standart calisma sekli): "
          f"dogru={n_correct} yanlis={n_wrong}", flush=True)

    rng = np.random.RandomState(42)
    correct_idx = rng.choice(np.flatnonzero(correct_mask), size=min(3, n_correct), replace=False)
    wrong_idx = np.flatnonzero(~correct_mask)[:2] if n_wrong > 0 else np.array([], dtype=int)

    explanation = shap.Explanation(
        values=shap_values, base_values=np.full(len(X_all), expected_value),
        data=X_all.values, feature_names=pool,
    )
    for label, idx_arr in (("dogru", correct_idx), ("yanlis", wrong_idx)):
        for i in idx_arr:
            vid = v1_df.iloc[i]["Variant_ID"]
            fig = plt.figure(figsize=(9, 6))
            shap.plots.waterfall(explanation[int(i)], show=False)
            plt.tight_layout()
            fname = FIG_DIR / f"f4_shap_waterfall_{label}_{vid}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  [{label}] {vid}: y_true={y_np[i]} pred={pred[i]} sld_proba={sld[i]:.3f} -> {fname.name}", flush=True)
    if n_wrong == 0:
        print("  UYARI: in-sample'da hic yanlis siniflandirma yok (final fit kendi egitim verisinde beklenen sekilde "
              "cok basarili) -- 'yanlis' orneği uretilemedi, yalnizca dogru orneklerle devam edildi.", flush=True)

    # ---------------------------------------------------------------- Adim 4
    print("\n=== Adim 4: Asama D ile tutarlilik ===", flush=True)
    stab = pd.read_csv(STABILITY_CSV)
    stab_pool = stab[stab["feature"].isin(pool)]
    mean_stability = stab_pool.groupby("feature")["n_methods_stable"].mean().reindex(pool)
    comparison = pd.DataFrame({
        "shap_mean_abs": mean_abs_shap.reindex(pool),
        "shap_rank": rank.reindex(pool),
        "stage_d_mean_n_methods_stable": mean_stability,
        "stage_d_rank": mean_stability.rank(ascending=False, method="min"),
    }).sort_values("shap_rank")
    spearman = comparison["shap_rank"].corr(comparison["stage_d_rank"], method="spearman")
    print(comparison.to_string(), flush=True)
    print(f"\nSpearman sira-korelasyonu (SHAP vs Asama D stabilite): {spearman:.4f}", flush=True)
    comparison.to_csv(COMPARISON_OUT)
    print(f"kaydedildi: {COMPARISON_OUT}", flush=True)


if __name__ == "__main__":
    main()
