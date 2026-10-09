"""Asama F1, Adim 4.2: mevcut resmi model (`final_model_bundle_v2.pkl`,
26 ozellik) vs F1 adversarial-validation ablasyon adayi (`final_model_
bundle_f1_ablation_candidate.pkl`, 22 ozellik -- AL_26/AL_12/AL_7/AL_49
cikarilmis) -- madde 10/11'in AYNI araclariyla (`sample_level_bootstrap_
ci`, `nadeau_bengio_corrected_ttest`, `monte_carlo_final_f1_simulation`)
karsilastirir. HICBIR MODEL YENIDEN EGITILMIYOR -- her iki bundle'in
KENDI dondurulmus pool/best_params/calibrator/threshold'u kullanilarak
ayni deterministik cross-fit tekrar calistiriliyor.

Bu script hicbir karar VERMEZ/UYGULAMAZ -- yalnizca olcer ve raporlar.
final_model_bundle_v2.pkl'e/predict.py'ye dokunmaz.

Calistirma: python -m genova.pah.f1_ablation_comparison
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR, F0_SEED
from genova.pah.f0_final_model_v3 import _oof_predictions_for_bundle
from genova.pah.e4_prior_correction import sld_correct
from genova.statistics import sample_level_bootstrap_ci, nadeau_bengio_corrected_ttest, monte_carlo_final_f1_simulation

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
OUT_CSV = TAB_DIR / "f1_ablation_comparison.csv"

OLD_BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
NEW_BUNDLE_PATH = MODEL_DIR / "final_model_bundle_f1_ablation_candidate.pkl"
N_BOOT = 2000
N_SIMULATIONS = 2000


def _per_fold_scores(v1_df, bundle):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
            v1_df, fold["train_variant_ids"], fold["test_variant_ids"], bundle["pool"],
        )
        proba = fit_predict_catboost(X_tr, y_tr, X_va, bundle["model_best_params"])
        calibrated = bundle["calibrator"].transform(proba)
        sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
        pred = (sld >= bundle["threshold"]).astype(int)
        rows.append({
            "fold": fold["fold"], "n_train": len(y_tr), "n_test": len(y_va),
            "f1": f1_binary_positive(y_va, pred),
            "mcc": matthews_correlation_coefficient(y_va, pred),
            "specificity": specificity(y_va, pred),
        })
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    variant_ids = v1_df["Variant_ID"].tolist()

    old_bundle = joblib.load(OLD_BUNDLE_PATH)
    new_bundle = joblib.load(NEW_BUNDLE_PATH)
    print(f"eski (resmi, 26 ozellik): {old_bundle['model_best_params']}, esik={old_bundle['threshold']}", flush=True)
    print(f"yeni (F1 ablasyon adayi, 22 ozellik): {new_bundle['model_best_params']}, esik={new_bundle['threshold']}", flush=True)

    old_sld, old_pred, old_y = _oof_predictions_for_bundle(v1_df, old_bundle, all_ids)
    new_sld, new_pred, new_y = _oof_predictions_for_bundle(v1_df, new_bundle, all_ids)
    assert np.array_equal(old_y, new_y)

    print("\n=== Ornek-duzeyi CI karsilastirmasi ===", flush=True)
    rows = []
    ci_overlap = {}
    for metric_name, metric_fn in (
        ("f1", f1_binary_positive), ("mcc", matthews_correlation_coefficient), ("specificity", specificity),
    ):
        old_ci = sample_level_bootstrap_ci(variant_ids, old_y, old_pred, metric_fn, n_boot=N_BOOT, seed=1)
        new_ci = sample_level_bootstrap_ci(variant_ids, new_y, new_pred, metric_fn, n_boot=N_BOOT, seed=1)
        overlap = not (new_ci["ci_high"] < old_ci["ci_low"] or old_ci["ci_high"] < new_ci["ci_low"])
        ci_overlap[metric_name] = overlap
        print(f"  {metric_name}: eski={old_ci['point_estimate']:.4f} [{old_ci['ci_low']:.4f},{old_ci['ci_high']:.4f}]  "
              f"yeni={new_ci['point_estimate']:.4f} [{new_ci['ci_low']:.4f},{new_ci['ci_high']:.4f}]  ortusuyor={overlap}", flush=True)
        rows.append({"analysis": "sample_level_bootstrap", "metric": metric_name,
                      "old_point": old_ci["point_estimate"], "new_point": new_ci["point_estimate"], "ci_overlap": overlap})

    print("\n=== Nadeau-Bengio (ayni 5-fold ic CV yapisi, k=5) ===", flush=True)
    old_folds = _per_fold_scores(v1_df, old_bundle)
    new_folds = _per_fold_scores(v1_df, new_bundle)
    n_train, n_test = old_folds["n_train"].to_numpy(), old_folds["n_test"].to_numpy()
    for metric in ("f1", "mcc", "specificity"):
        nb = nadeau_bengio_corrected_ttest(old_folds[metric].to_numpy(), new_folds[metric].to_numpy(), n_train, n_test)
        print(f"  {metric}: mean_diff(eski-yeni)={nb['mean_diff']:+.4f}  p={nb['p_value']:.4f}", flush=True)
        rows.append({"analysis": "nadeau_bengio", "metric": metric,
                      "old_point": np.nan, "new_point": nb["mean_diff"], "nb_p_value": nb["p_value"]})

    print("\n=== Monte Carlo final-F1 (100 patojenik/250 benign) ===", flush=True)
    old_mc = monte_carlo_final_f1_simulation(variant_ids, old_y, old_sld, threshold=old_bundle["threshold"], n_simulations=N_SIMULATIONS)
    new_mc = monte_carlo_final_f1_simulation(variant_ids, new_y, new_sld, threshold=new_bundle["threshold"], n_simulations=N_SIMULATIONS)
    print(f"  eski: ort={old_mc['mean']:.4f} [{old_mc['ci_low']:.4f},{old_mc['ci_high']:.4f}]", flush=True)
    print(f"  yeni: ort={new_mc['mean']:.4f} [{new_mc['ci_low']:.4f},{new_mc['ci_high']:.4f}]", flush=True)
    rows.append({"analysis": "monte_carlo_100_250", "metric": "f1", "old_point": old_mc["mean"], "new_point": new_mc["mean"]})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}", flush=True)

    all_overlap = all(ci_overlap.values())
    print(f"\n=== tum metriklerde CI ortusuyor mu: {all_overlap} === (KARAR VERILMEDI -- yalnizca olcum)", flush=True)


if __name__ == "__main__":
    main()
