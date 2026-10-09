"""P1 madde 10, Adim 2: final modelin (CatBoost/v4_from_v2, `models/pah/
final_model_bundle_v2.pkl` -- RESMI final bundle) gercek belirsizligini
olcer -- HICBIR MODEL YENIDEN EGITILMIYOR.

DUZELTME (Revize turu, bkz. `reports/10_BOOTSTRAP_GUVEN_ARALIKLARI_
PAH.md`'nin basindaki DUZELTME NOTU): bu script daha once yanlislikla
`f0_final_model.py::BUNDLE_OUT` (`final_model_bundle.pkl` -- P1 madde 11
ONCESI, 28 ozellikli, provenance-riskli `al_all_missing`/`CAT_1` DAHIL,
artik resmi OLMAYAN tarihsel bundle) yukluyordu. `BUNDLE_PATH`
(`final_model_bundle_v2.pkl`) olarak duzeltildi -- dosyanin baska hicbir
mantigi degistirilmedi.

`f0_final_model.py` DEGISTIRILMEDI; Adim 3'un (cross-fit OOF) urettigi 369
satirlik olasilik vektoru bu turda hic persist edilmemisti, bu yuzden ayni
(deterministik, ayni seed'li) `cross_fit_oof` fonksiyonu buradan ithal
edilip, bundle'in KENDI dondurulmus `pool`/`model_best_params`'iyla
yeniden CAGRILIYOR -- bu "yeniden egitim" degil, zaten bundle'i
uretmis olan ayni 5-fold cross-fit'in ayni girdilerle tekrar calistirilmasi
(bit-bit ayni sonucu verir, cunku hem seed hem veri hem hiperparametre
sabit). Kalibrasyon/onsel-duzeltme/esik ise bundle'in KENDI fit edilmis
nesnelerinden (`calibrator.transform()`, sabit `w1`/`w0`/`threshold`)
okunuyor -- hicbiri burada yeniden fit edilmiyor.

Adim 1: sample_level_bootstrap_ci ile F1/MCC/specificity/sensitivity/AUROC icin
        ornek-duzeyi %95 CI (369 satirin SATIR-bazli yeniden orneklemesi).
        AUROC (Revize turu) esik-bagimsizdir, `sld` (kalibre+onsel-duzeltilmis)
        olasiligindan hesaplanir -- Beta kalibrasyonu ve SLD ikisi de
        MONOTONIK oldugu icin sonuc `oof_proba` (ham) ile hesaplansaydi da
        birebir ayni cikardi.
Adim 2: mevcut fold-duzeyi CI (e5_threshold_selection.csv'nin catboost/
        v4_from_v2 satirlarindaki f1_chosen'in FOLD-bazli bootstrap'i) ile
        yan yana karsilastirma.
Adim 3: monte_carlo_final_f1_simulation ile sartnamenin gercek final test
        kompozisyonunda (100 patojenik/250 benign) F1 dagilimi.
Adim 4 (istege bagli, P1 madde 10 Adim 3): en tartismali gecmis karar --
        CatBoost/v4_from_v2 vs RandomForest/v4_from_v2 (E5) -- icin
        `nadeau_bengio_corrected_ttest` ile duzeltilmis p-degeri. Bu karari
        DEGISTIRMIYOR (CatBoost zaten secilmisti), yalnizca istatistiksel
        ifadeyi netlestiriyor. `e5_threshold_selection.csv` YALNIZCA
        OKUNUYOR, yeniden hesaplanmiyor.

Calistirma: python -m genova.pah.f0_uncertainty_analysis
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah.f0_final_model import cross_fit_oof, V1_PATH, MODEL_DIR
from genova.pah.e4_prior_correction import sld_correct
from genova.statistics import sample_level_bootstrap_ci, monte_carlo_final_f1_simulation, nadeau_bengio_corrected_ttest

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"
OUT_CSV = TAB_DIR / "f0_uncertainty_analysis.csv"
BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"  # RESMI final bundle (duzeltildi -- eskiden BUNDLE_OUT/final_model_bundle.pkl idi)

N_BOOT = 2000
N_SIMULATIONS = 2000
FOLD_LEVEL_SEED = 42


def _fold_level_bootstrap_ci(fold_scores, n_boot=N_BOOT, alpha=0.05, seed=FOLD_LEVEL_SEED):
    """YALNIZCA karsilastirma amacli: 50 dis-fold'un ZATEN hesaplanmis
    F1 degerlerini (`e5_threshold_selection.csv`) FOLD bazinda (satir
    bazinda DEGIL) yeniden orneklemek -- denetimin "dar/iyimser" dedigi
    yontemin ta kendisi. Hicbir E5 sonucu yeniden hesaplanmiyor, yalnizca
    zaten var olan 50 sayi bootstrap'a tabi tutuluyor."""
    fold_scores = np.asarray(fold_scores, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(fold_scores)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = fold_scores[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"point_estimate": float(fold_scores.mean()), "ci_low": float(lo), "ci_high": float(hi), "ci_width": float(hi - lo)}


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool = bundle["pool"]
    best_params = bundle["model_best_params"]
    threshold = bundle["threshold"]

    print("=== bundle'in kendi 5-fold cross-fit OOF'unu yeniden uretiyor (deterministik, ayni sonuc) ===", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= threshold).astype(int)
    variant_ids = v1_df["Variant_ID"].tolist()

    print("\n=== Adim 1: ornek-duzeyi (satir-bazli) bootstrap CI ===", flush=True)
    rows = []
    for metric_name, metric_fn in (
        ("f1", f1_binary_positive), ("mcc", matthews_correlation_coefficient),
        ("specificity", specificity), ("sensitivity", sensitivity),
    ):
        sample_ci = sample_level_bootstrap_ci(variant_ids, y_full, pred, metric_fn, n_boot=N_BOOT)
        print(f"  {metric_name}: nokta={sample_ci['point_estimate']:.4f} "
              f"CI=[{sample_ci['ci_low']:.4f}, {sample_ci['ci_high']:.4f}] genislik={sample_ci['ci_width']:.4f}", flush=True)
        rows.append({
            "analysis": "sample_level_bootstrap", "metric": metric_name,
            "point_estimate": sample_ci["point_estimate"], "ci_low": sample_ci["ci_low"],
            "ci_high": sample_ci["ci_high"], "ci_width": sample_ci["ci_width"], "n_boot": N_BOOT,
        })

    print("\n=== Revize turu: AUROC (esik-bagimsiz, `sld` olasiligindan -- monotonik "
          "donusum oldugu icin `oof_proba` ile birebir ayni sonucu verir) ===", flush=True)
    auroc_raw_check = roc_auc_score(y_full, oof_proba)
    auroc_sld_check = roc_auc_score(y_full, sld)
    assert abs(auroc_raw_check - auroc_sld_check) < 1e-9, (
        f"monotoniklik varsayimi bozuldu: ham AUROC={auroc_raw_check:.6f}, sld AUROC={auroc_sld_check:.6f}"
    )
    print(f"  monotoniklik dogrulandi: ham AUROC={auroc_raw_check:.4f} == sld AUROC={auroc_sld_check:.4f}", flush=True)
    auroc_ci = sample_level_bootstrap_ci(variant_ids, y_full, sld, roc_auc_score, n_boot=N_BOOT)
    print(f"  auroc: nokta={auroc_ci['point_estimate']:.4f} "
          f"CI=[{auroc_ci['ci_low']:.4f}, {auroc_ci['ci_high']:.4f}] genislik={auroc_ci['ci_width']:.4f}", flush=True)
    rows.append({
        "analysis": "sample_level_bootstrap", "metric": "auroc",
        "point_estimate": auroc_ci["point_estimate"], "ci_low": auroc_ci["ci_low"],
        "ci_high": auroc_ci["ci_high"], "ci_width": auroc_ci["ci_width"], "n_boot": N_BOOT,
    })

    print("\n=== Adim 2: mevcut fold-duzeyi CI ile karsilastirma (yalnizca F1, catboost/v4_from_v2) ===", flush=True)
    e5 = pd.read_csv(E5_CSV)
    fold_f1 = e5[(e5.model == "catboost") & (e5.data_version == "v4_from_v2")]["f1_chosen"].to_numpy()
    assert len(fold_f1) == 50, f"beklenen 50 dis-fold, bulunan {len(fold_f1)}"
    fold_ci = _fold_level_bootstrap_ci(fold_f1)
    print(f"  fold-duzeyi (n=50, yanlislikla bagimsiz varsayilan): nokta={fold_ci['point_estimate']:.4f} "
          f"CI=[{fold_ci['ci_low']:.4f}, {fold_ci['ci_high']:.4f}] genislik={fold_ci['ci_width']:.4f}", flush=True)
    sample_f1_ci = next(r for r in rows if r["metric"] == "f1")
    print(f"  ornek-duzeyi (n=369, dogru birim): genislik={sample_f1_ci['ci_width']:.4f}", flush=True)
    print(f"  genislik orani (ornek/fold): {sample_f1_ci['ci_width'] / fold_ci['ci_width']:.2f}x", flush=True)
    rows.append({
        "analysis": "fold_level_bootstrap_comparison", "metric": "f1",
        "point_estimate": fold_ci["point_estimate"], "ci_low": fold_ci["ci_low"],
        "ci_high": fold_ci["ci_high"], "ci_width": fold_ci["ci_width"], "n_boot": N_BOOT,
    })

    print(f"\n=== Adim 3: Monte Carlo final-F1 simulasyonu (100 patojenik/250 benign, esik={threshold}) ===", flush=True)
    mc = monte_carlo_final_f1_simulation(
        variant_ids, y_full, sld, n_pathogenic=100, n_benign=250,
        n_simulations=N_SIMULATIONS, threshold=threshold,
    )
    print(f"  ort={mc['mean']:.4f} medyan={mc['median']:.4f} "
          f"%95 aralik=[{mc['ci_low']:.4f}, {mc['ci_high']:.4f}]", flush=True)
    rows.append({
        "analysis": "monte_carlo_100_250", "metric": "f1",
        "point_estimate": mc["mean"], "ci_low": mc["ci_low"], "ci_high": mc["ci_high"],
        "ci_width": mc["ci_high"] - mc["ci_low"], "n_boot": N_SIMULATIONS,
    })

    print("\n=== Adim 4 (istege bagli): Nadeau-Bengio -- CatBoost/v4_from_v2 vs RF/v4_from_v2 (E5) ===", flush=True)
    cb = e5[(e5.model == "catboost") & (e5.data_version == "v4_from_v2")].sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
    rf = e5[(e5.model == "random_forest") & (e5.data_version == "v4_from_v2")].sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
    assert len(cb) == len(rf) == 50
    assert (cb["repeat"].to_numpy() == rf["repeat"].to_numpy()).all()
    assert (cb["outer_fold"].to_numpy() == rf["outer_fold"].to_numpy()).all()
    n_train_pair, n_test_pair = cb["n_dis_train"].to_numpy(), cb["n_dis_test"].to_numpy()

    for metric_col, label in (("weighted_f1_chosen", "weighted_f1 (E5'in resmi karar metrigi)"), ("f1_chosen", "f1 (ham, bilgi amacli)")):
        naive_t, naive_p = scipy_stats.ttest_rel(cb[metric_col], rf[metric_col])
        nb = nadeau_bengio_corrected_ttest(cb[metric_col].to_numpy(), rf[metric_col].to_numpy(), n_train_pair, n_test_pair)
        print(f"  {label}: mean_diff(cb-rf)={nb['mean_diff']:+.4f}  naif p={naive_p:.4f}  NB-duzeltilmis p={nb['p_value']:.4f}", flush=True)
        rows.append({
            "analysis": f"nadeau_bengio_catboost_vs_rf_{metric_col}", "metric": metric_col,
            "point_estimate": nb["mean_diff"], "ci_low": np.nan, "ci_high": np.nan,
            "ci_width": np.nan, "n_boot": np.nan,
            "naive_p_value": naive_p, "nb_t_stat": nb["t_stat"], "nb_p_value": nb["p_value"],
        })

    result = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
