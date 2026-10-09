"""P0-3 Takip: minimax saglam esik arastirmasi. Test verisine HIC
bakilmiyor -- yalnizca mevcut egitim OOF olasiliklari + genisletilmis
prevalans izgarasi uzerinde, tum makul prevalans araligindaki worst-case
F1'i maksimize eden tek bir esik arayisi (minimax prensibi). Bu bir
"test gormeden once donan karar" -- final gunu kuralini ihlal etmiyor.

HICBIR MODEL/ESIK OTOMATIK DEGISTIRILMIYOR -- bu script yalnizca arastirma/
oneri uretir. `final_model_bundle_v2.pkl`'nin esigi (0,35) bu turda
degismedi.

Mekanizma: `f0_multi_prevalence_threshold_robustness.py`'nin ayni bundle/
OOF/sld kurulumu ve `f0_final_performance_card.py::monte_carlo_full_
metrics` (degistirilmedi, zaten threshold parametreli) yeniden kullanilir.
Kaba 8x9 izgara HIZLI bir n_simulations ile taranir (rank-siralama icin
yeterli), sonra yalnizca secilen aday(lar) P0-3 ile AYNI hassasiyette
(n_simulations=2000) yeniden hesaplanip resmi tabloya o sayilar yazilir --
kaba taramanin gurultulu sayilari resmi rapora yazilmaz (surec kurali).

Calistirma: python -m genova.pah.f0_minimax_threshold_search
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.pah.f0_final_model import V1_PATH, TAB_DIR, MODEL_DIR, cross_fit_oof
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.f0_final_performance_card import monte_carlo_full_metrics
from genova.pah.f0_multi_prevalence_threshold_robustness import _scenario_sample_sizes

BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
OUT_CSV = TAB_DIR / "f0_minimax_threshold_search.csv"
OUT_CANDIDATE_CSV = TAB_DIR / "f0_minimax_threshold_candidate_comparison.csv"

PREVALENCE_GRID = [0.15, 0.18, 0.20, 0.22, 0.25, 0.286, 0.33, 0.38]
THRESHOLD_GRID = [0.30, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45, 0.48, 0.50]
SPEC_PREVALENCE = 0.286
CURRENT_THRESHOLD = 0.35
N_SIM_COARSE = 400   # kaba tarama -- yalnizca siralama icin
N_SIM_FINAL = 2000   # secilen aday(lar) icin, P0-3 ile ayni hassasiyet
BALANCED_MIN_WORST_CASE_GAIN = 0.03
BALANCED_MAX_SPEC_LOSS = 0.02


def sweep(variant_ids, y_full, sld, thresholds, prevalences, n_simulations):
    rows = []
    for prevalence in prevalences:
        n_pathogenic, n_benign = _scenario_sample_sizes(prevalence)
        for threshold in thresholds:
            block, _ = monte_carlo_full_metrics(
                variant_ids, y_full, sld, threshold,
                n_pathogenic=n_pathogenic, n_benign=n_benign, n_simulations=n_simulations,
            )
            row = {"prevalence": prevalence, "threshold": threshold}
            for metric in ("f1", "mcc", "specificity", "sensitivity", "precision"):
                row[metric] = block[block.metric == metric]["point_estimate"].iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_worst_case(sweep_df):
    """Her esik icin, prevalans izgarasinin en kotu (worst-case) F1'i +
    sartname prevalansindaki (0,286) F1."""
    rows = []
    for threshold in sorted(sweep_df["threshold"].unique()):
        sub = sweep_df[sweep_df.threshold == threshold]
        worst = sub.loc[sub["f1"].idxmin()]
        spec_row = sub[np.isclose(sub.prevalence, SPEC_PREVALENCE)].iloc[0]
        rows.append({
            "threshold": threshold, "worst_case_f1": worst["f1"], "worst_case_prevalence": worst["prevalence"],
            "f1_at_spec_prevalence": spec_row["f1"], "mcc_at_spec_prevalence": spec_row["mcc"],
        })
    return pd.DataFrame(rows)


def pick_minimax_candidates(summary_df):
    current = summary_df[np.isclose(summary_df.threshold, CURRENT_THRESHOLD)].iloc[0]
    pure = summary_df.loc[summary_df["worst_case_f1"].idxmax()]

    balanced_pool = summary_df[
        (summary_df["worst_case_f1"] - current["worst_case_f1"] >= BALANCED_MIN_WORST_CASE_GAIN)
        & (current["f1_at_spec_prevalence"] - summary_df["f1_at_spec_prevalence"] <= BALANCED_MAX_SPEC_LOSS)
    ]
    balanced = None if balanced_pool.empty else balanced_pool.loc[balanced_pool["worst_case_f1"].idxmax()]
    return current, pure, balanced


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]

    print(f"final bundle: {len(pool)} ozellik, mevcut esik={bundle['threshold']} "
          f"(bu turda DEGISTIRILMIYOR -- yalnizca arastirma)", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    variant_ids = v1_df["Variant_ID"].tolist()

    print(f"\n=== Adim 1-2: {len(PREVALENCE_GRID)} prevalans x {len(THRESHOLD_GRID)} esik = "
          f"{len(PREVALENCE_GRID) * len(THRESHOLD_GRID)} hucre, kaba tarama (n_sim={N_SIM_COARSE}) ===", flush=True)
    coarse = sweep(variant_ids, y_full, sld, THRESHOLD_GRID, PREVALENCE_GRID, N_SIM_COARSE)
    summary = summarize_worst_case(coarse)
    print(summary.to_string(index=False), flush=True)

    current, pure, balanced = pick_minimax_candidates(summary)
    print(f"\nsaf minimax (kaba): esik={pure['threshold']} worst-case F1={pure['worst_case_f1']:.4f}", flush=True)
    if balanced is not None:
        print(f"dengeli minimax (kaba): esik={balanced['threshold']} worst-case F1={balanced['worst_case_f1']:.4f} "
              f"0,286'daki F1={balanced['f1_at_spec_prevalence']:.4f}", flush=True)
    else:
        print("dengeli minimax kriterini karsilayan aday yok (kaba tarama)", flush=True)

    candidate_thresholds = sorted({CURRENT_THRESHOLD, float(pure["threshold"])} | (
        {float(balanced["threshold"])} if balanced is not None else set()))
    print(f"\n=== Adim 3 dogrulama: aday esik(ler) {candidate_thresholds} icin TAM hassasiyet "
          f"(n_sim={N_SIM_FINAL}) yeniden hesaplaniyor ===", flush=True)
    fine = sweep(variant_ids, y_full, sld, candidate_thresholds, PREVALENCE_GRID, N_SIM_FINAL)
    fine_summary = summarize_worst_case(fine)
    print(fine_summary.to_string(index=False), flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(OUT_CSV, index=False)
    fine.to_csv(OUT_CANDIDATE_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}, {OUT_CANDIDATE_CSV}")


if __name__ == "__main__":
    main()
