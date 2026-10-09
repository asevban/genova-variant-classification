"""P1 madde 11, Adim 3: eski final model (28 ozellik, `al_all_missing`/
`CAT_1` dahil) vs yeni final model (26 ozellik, provenance-riskli ikisi
cikarilmis) -- madde 10'un istatistik araclariyla (`sample_level_
bootstrap_ci`, `nadeau_bengio_corrected_ttest`, `monte_carlo_final_f1_
simulation`) karsilastirilir. HICBIR MODEL YENIDEN EGITILMIYOR -- her iki
bundle'in KENDI dondurulmus `pool`/`best_params`/`calibrator`/`threshold`'u
kullanilarak, ayni deterministik 5-fold cross-fit (`sb.build_outer_folds`,
F0_SEED+1 -- iki bundle de AYNI split'i kullaniyor, cunku split yalnizca
`v1_df` + sabit seed'e bagli, `pool`'a degil) yeniden calistiriliyor.

Calistirma: python -m genova.pah.f0_provenance_ablation_comparison
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR, F0_SEED, cross_fit_oof
from genova.pah.e4_prior_correction import sld_correct
from genova.statistics import sample_level_bootstrap_ci, nadeau_bengio_corrected_ttest, monte_carlo_final_f1_simulation

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
OUT_CSV = TAB_DIR / "f0_provenance_ablation_comparison.csv"

OLD_BUNDLE_PATH = MODEL_DIR / "final_model_bundle.pkl"
NEW_BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
N_BOOT = 2000
N_SIMULATIONS = 2000


def _oof_predictions(v1_df, bundle, all_ids):
    """Bundle'in KENDI pool/best_params'iyla, `f0_final_model.py::cross_
    fit_oof`'un ayni deterministik cagrisi -- yeniden egitim degil."""
    oof_proba, y_full = cross_fit_oof(v1_df, bundle["pool"], bundle["model_best_params"], all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= bundle["threshold"]).astype(int)
    return sld, pred, y_full


def _per_fold_scores(v1_df, bundle):
    """`cross_fit_oof` ile AYNI 5-fold split (F0_SEED+1, yalnizca v1_df'e
    bagli -- pool'dan bagimsiz, iki bundle de bu yuzden AYNI fold'lari
    kullanir), ama fold-BAZLI skorlari da doner (Nadeau-Bengio icin
    gerekli -- cross_fit_oof yalnizca havuzlanmis OOF vektoru donuyor)."""
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
    print(f"eski: {old_bundle['metadata']['n_pool_features']} ozellik, esik={old_bundle['threshold']}")
    print(f"yeni: {new_bundle['metadata']['n_pool_features']} ozellik, esik={new_bundle['threshold']}")

    old_sld, old_pred, old_y = _oof_predictions(v1_df, old_bundle, all_ids)
    new_sld, new_pred, new_y = _oof_predictions(v1_df, new_bundle, all_ids)
    assert np.array_equal(old_y, new_y), "iki bundle'in OOF etiket sirasi hizali degil"

    print("\n=== Adim 3.1: ornek-duzeyi CI karsilastirmasi (eski vs yeni) ===", flush=True)
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
        for label, ci in (("old", old_ci), ("new", new_ci)):
            rows.append({
                "analysis": "sample_level_bootstrap", "variant": label, "metric": metric_name,
                "point_estimate": ci["point_estimate"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "ci_width": ci["ci_width"], "ci_overlap_with_other_variant": overlap,
            })

    print("\n=== Adim 3.2: Nadeau-Bengio (ayni 5-fold ic CV yapisi, k=5) ===", flush=True)
    old_folds = _per_fold_scores(v1_df, old_bundle)
    new_folds = _per_fold_scores(v1_df, new_bundle)
    assert (old_folds["fold"].to_numpy() == new_folds["fold"].to_numpy()).all()
    n_train, n_test = old_folds["n_train"].to_numpy(), old_folds["n_test"].to_numpy()
    for metric in ("f1", "mcc", "specificity"):
        nb = nadeau_bengio_corrected_ttest(old_folds[metric].to_numpy(), new_folds[metric].to_numpy(), n_train, n_test)
        print(f"  {metric}: mean_diff(eski-yeni)={nb['mean_diff']:+.4f}  NB p={nb['p_value']:.4f}  (k=5, kucuk-k uyarisi)", flush=True)
        rows.append({
            "analysis": "nadeau_bengio_old_vs_new", "variant": "diff", "metric": metric,
            "point_estimate": nb["mean_diff"], "ci_low": np.nan, "ci_high": np.nan, "ci_width": np.nan,
            "nb_t_stat": nb["t_stat"], "nb_p_value": nb["p_value"], "nb_k": nb["k"],
        })

    print(f"\n=== Adim 3.3: Monte Carlo final-F1 (100 patojenik/250 benign) karsilastirmasi ===", flush=True)
    old_mc = monte_carlo_final_f1_simulation(variant_ids, old_y, old_sld, n_pathogenic=100, n_benign=250,
                                              n_simulations=N_SIMULATIONS, threshold=old_bundle["threshold"])
    new_mc = monte_carlo_final_f1_simulation(variant_ids, new_y, new_sld, n_pathogenic=100, n_benign=250,
                                              n_simulations=N_SIMULATIONS, threshold=new_bundle["threshold"])
    print(f"  eski: ort={old_mc['mean']:.4f} [{old_mc['ci_low']:.4f},{old_mc['ci_high']:.4f}]", flush=True)
    print(f"  yeni: ort={new_mc['mean']:.4f} [{new_mc['ci_low']:.4f},{new_mc['ci_high']:.4f}]", flush=True)
    for label, mc in (("old", old_mc), ("new", new_mc)):
        rows.append({
            "analysis": "monte_carlo_100_250", "variant": label, "metric": "f1",
            "point_estimate": mc["mean"], "ci_low": mc["ci_low"], "ci_high": mc["ci_high"],
            "ci_width": mc["ci_high"] - mc["ci_low"],
        })

    result = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")

    all_overlap = all(ci_overlap.values())
    print(f"\n=== SONUC: tum metriklerde CI ortusuyor mu: {all_overlap} ===")


if __name__ == "__main__":
    main()
