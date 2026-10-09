"""P0-3: Coklu-prevalans esik saglamligi (final oncesi son P0 deneyi).

Sartnamedeki 100P/250B sayisi "yaklasik" -- gercek final test seti farkli
bir oranla gelebilir. Bu script frozen final bundle'in (`final_model_
bundle_v2.pkl`, esik=0,35 SABIT) 5 farkli prevalans senaryosunda (P(Pathogenic)
in {0,20; 0,25; 0,286; 0,33; 0,38}, sabit N=350) ne kadar iyi/kotu
performans gosterecegini olcer -- HICBIR MODEL/ESIK YENIDEN FIT EDILMIYOR.

Mekanizma: bundle'in kendi deterministik `cross_fit_oof`'u (f0_final_model.py,
degistirilmedi) yeniden uretilir, ayni sld (kalibre + onsel-duzeltilmis)
olasiliklar uzerinde `f0_final_performance_card.py::monte_carlo_full_metrics`
(degistirilmedi, zaten n_pathogenic/n_benign parametreli, genel-amacli)
her prevalans icin farkli n_pathogenic/n_benign ile CAGRILIR -- bu
fonksiyonun zaten var olan, degistirilmeden yeniden kullanilabilir tasarimi
bu turun tum agir hesaplama ihtiyacini karsiliyor.

Adim 3 (senaryo-ozel optimal esik, YALNIZCA KARSILASTIRMA -- gercek esik
degismiyor): `e5_threshold_selection.py::_prior_weights`/`THRESHOLD_GRID`
(degistirilmedi) ile, OOF'un sld olasiliklari uzerinde her prevalans icin
onsel-agirlikli F1'i maksimize eden esik aranir -- E5'in kendi "sistem
sabit kalirsa gercek onsel kayarsa ne olur" sorusunun tersi ("gercek onsel
onceden bilinseydi esik ne olurdu").

Calistirma: python -m genova.pah.f0_multi_prevalence_threshold_robustness
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive_weighted
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, MODEL_DIR, cross_fit_oof
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import _prior_weights, THRESHOLD_GRID
from genova.pah.f0_final_performance_card import monte_carlo_full_metrics

BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
OUT_CSV = TAB_DIR / "f0_multi_prevalence_threshold_robustness.csv"

N_TOTAL = 350
PREVALENCE_GRID = [0.20, 0.25, FINAL_PATHOGENIC_PRIOR, 0.33, 0.38]
N_SIMULATIONS = 2000  # Adim 2'nin >=1000 sartini asiyor
FINAL_BENIGN_STRESS_BAND = (0.62, 0.64)  # 06 raporu, 100P/250B Monte Carlo referansi


def _scenario_sample_sizes(prevalence, n_total=N_TOTAL):
    n_pathogenic = int(round(n_total * prevalence))
    return n_pathogenic, n_total - n_pathogenic


def _scenario_optimal_threshold(y_full, sld, prevalence):
    """YALNIZCA karsilastirma amacli: bu prevalansi onceden bilseydik,
    OOF'un kendi sld olasiliklari uzerinde onsel-agirlikli F1'i maksimize
    eden esik ne olurdu? Gercek bundle esigi (0,35) degismiyor."""
    weights = _prior_weights(y_full, prevalence)
    scores = np.array([
        f1_binary_positive_weighted(y_full, (sld >= thr).astype(int), weights)
        for thr in THRESHOLD_GRID
    ])
    best = scores.max()
    tied = THRESHOLD_GRID[np.isclose(scores, best, atol=1e-9)]
    return float(tied.mean()), float(best)


def run_grid(variant_ids, y_full, sld, threshold):
    rows, mc_cache = [], {}
    for prevalence in PREVALENCE_GRID:
        n_pathogenic, n_benign = _scenario_sample_sizes(prevalence)
        block, scores = monte_carlo_full_metrics(
            variant_ids, y_full, sld, threshold,
            n_pathogenic=n_pathogenic, n_benign=n_benign, n_simulations=N_SIMULATIONS,
        )
        mc_cache[prevalence] = (block, scores)
        fp_per_sim = n_benign * (1 - scores["specificity"])
        opt_thr, opt_weighted_f1 = _scenario_optimal_threshold(y_full, sld, prevalence)

        row = {"prevalence": prevalence, "n_pathogenic": n_pathogenic, "n_benign": n_benign,
               "frozen_threshold": threshold, "scenario_optimal_threshold": opt_thr,
               "threshold_gap": abs(opt_thr - threshold), "scenario_optimal_weighted_f1": opt_weighted_f1,
               "fp_mean": float(fp_per_sim.mean()),
               "fp_ci_low": float(np.percentile(fp_per_sim, 2.5)),
               "fp_ci_high": float(np.percentile(fp_per_sim, 97.5))}
        for metric in ("f1", "mcc", "specificity", "sensitivity", "precision"):
            m = block[block.metric == metric].iloc[0]
            row[f"{metric}_mean"] = m["point_estimate"]
            row[f"{metric}_ci_low"] = m["ci_low"]
            row[f"{metric}_ci_high"] = m["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows), mc_cache


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params, threshold = bundle["pool"], bundle["model_best_params"], bundle["threshold"]

    print(f"final bundle: {len(pool)} ozellik, esik={threshold} (SABIT, bu turda degismiyor)", flush=True)
    print("bundle'in kendi 5-fold cross-fit OOF'u yeniden uretiliyor (deterministik, yeniden egitim degil)...", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    variant_ids = v1_df["Variant_ID"].tolist()

    print(f"\n=== P0-3: {len(PREVALENCE_GRID)} prevalans senaryosu, N={N_TOTAL}, "
          f"esik={threshold} sabit, n_simulations={N_SIMULATIONS} ===", flush=True)
    result, _ = run_grid(variant_ids, y_full, sld, threshold)
    print(result.to_string(index=False), flush=True)

    mean_f1 = result["f1_mean"].mean()
    worst_row = result.loc[result["f1_mean"].idxmin()]
    max_gap = result["threshold_gap"].max()
    print(f"\nortalama F1={mean_f1:.4f}  worst-case F1={worst_row['f1_mean']:.4f} "
          f"(prevalans={worst_row['prevalence']})  max esik-farki={max_gap:.4f}", flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
