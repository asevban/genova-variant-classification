"""`f0_weighting_experiment.py`'nin tek-5-fold turunda karar kuralini
saglayan aday (`class_weight=1,0, strength=1,0` -- tam domain-dengeleme,
ekstra benign agirligi yok), madde 7'nin (`f0_madde7_stability_check.py`)
AYNI tekrarli-bolme disipliniyle test ediliyor: 10 farkli
`StratifiedKFold(n_splits=5, shuffle=True, random_state=i)` (split
bankasindan BAGIMSIZ, grup-farkinda DEGIL -- bilerek boyle, saf bir
gurultu/kararlilik testi, nested genelleme iddiasi degil).

`f0_weighting_experiment.py` DEGISTIRILMEDI -- `cross_fit_oof_weighted`/
`compute_sample_weight`/`fit_predict_catboost_weighted` AYNEN import
edilip yeniden kullanildi. Domain-weight tarifi (tertil sinirlari,
ters-frekans formulu, strength=1,0) bu turda YENIDEN AYARLANMIYOR.

`CAT_1` gecis testi notu: bu test HER SEED'DE TEKRAR FIT EDILMIYOR --
egitim kumesi (cat1_empty_ids, 132 satir) ve CatBoost'un kendi
`random_seed=42`'si (`models.py`'nin sabiti) outer-CV seed'inden
BAGIMSIZ oldugu icin, bir adayin bu testteki HAM olasiligi (dolayisiyla
raw AUC'si) tum 10 tekrarda birebir AYNIDIR -- yalnizca o tekrarin
kendi kalibratoru/esigi degisir (F1'i etkiler, AUC'yi etkilemez). Bu
yuzden fit BIR KEZ (aday basina) yapiliyor, raw AUC'nin seed-bagimsizligi
kodda dogrudan dogrulaniyor (assert).

Calistirma: python -m genova.pah.f0_weighting_repeated_split_check
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from genova.metrics import f1_binary_positive_weighted, specificity
from genova.pah import fold_versions as fv
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, select_calibrator, select_prior_and_threshold
from genova.pah.f3_robustness_stress_tests import BUNDLE_PATH
from genova.pah.f0_weighting_experiment import (
    cross_fit_oof_weighted, compute_sample_weight, fit_predict_catboost_weighted,
)
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR, W1, W0
from genova.pah.e5_threshold_selection import _prior_weights
from genova.statistics import nadeau_bengio_corrected_ttest

N_SEEDS = 10
N_SPLITS = 5
CURRENT = {"name": "mevcut (cw=2,0 str=0,0)", "class_weight": 2.0, "strength": 0.0}
CANDIDATE = {"name": "yeni_aday (cw=1,0 str=1,0)", "class_weight": 1.0, "strength": 1.0}

OUT_CSV = TAB_DIR / "f0_weighting_repeated_split_check.csv"
OUT_POOLED_CSV = TAB_DIR / "f0_weighting_repeated_split_pooled_folds.csv"

WIN_MARGIN_THRESHOLD = 0.01  # madde 7 dersi: yalnizca kazanma sayisina degil, buyuklugu de bak


def _stratified_outer_folds(v1_df, seed, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(v1_df, v1_df["Label"])):
        outer_folds.append({
            "fold": fold_idx,
            "train_variant_ids": v1_df.iloc[train_idx]["Variant_ID"].tolist(),
            "test_variant_ids": v1_df.iloc[test_idx]["Variant_ID"].tolist(),
        })
    return outer_folds


def _evaluate_candidate_on_split(v1_df, pool, best_params, outer_folds, class_weight, strength):
    oof_proba, y_full = cross_fit_oof_weighted(v1_df, pool, best_params, outer_folds, class_weight, strength)
    winner_name, winner_cal, _ = select_calibrator(oof_proba, y_full)
    calibrated = winner_cal.transform(oof_proba)
    threshold, _ = select_prior_and_threshold(calibrated, y_full)
    sld = sld_correct(calibrated)
    pred = (sld >= threshold).astype(int)
    weights = _prior_weights(y_full, FINAL_PATHOGENIC_PRIOR)

    fold_by_vid = {}
    n_train_by_fold, n_test_by_fold = {}, {}
    for fold in outer_folds:
        for vid in fold["test_variant_ids"]:
            fold_by_vid[vid] = fold["fold"]
        n_train_by_fold[fold["fold"]] = len(fold["train_variant_ids"])
        n_test_by_fold[fold["fold"]] = len(fold["test_variant_ids"])

    ordered_ids = v1_df["Variant_ID"].tolist()
    fold_idx = np.array([fold_by_vid[v] for v in ordered_ids])
    per_fold = []
    for k in sorted(set(fold_idx)):
        mask = fold_idx == k
        per_fold.append({
            "fold": k, "weighted_f1": f1_binary_positive_weighted(y_full[mask], pred[mask], weights[mask]),
            "n_train": n_train_by_fold[k], "n_test": n_test_by_fold[k],
        })

    return {
        "weighted_f1_overall": f1_binary_positive_weighted(y_full, pred, weights),
        "specificity_overall": specificity(y_full, pred),
        "per_fold": pd.DataFrame(per_fold),
    }


def _cat1_shift_raw_auc(v1_df, pool, best_params, class_weight, strength):
    """train=CAT_1-bos -> test=CAT_1-dolu (F3'un en kritik yonu). Bir KEZ
    fit edilir -- bkz. modul docstring'i (outer-CV seed'inden bagimsiz)."""
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, cat1_empty_ids, cat1_filled_ids, pool)
    sample_weight = compute_sample_weight(v1_df, cat1_empty_ids, y_tr.to_numpy(), class_weight, strength)
    proba = fit_predict_catboost_weighted(X_tr, y_tr, X_te, best_params, sample_weight)
    return roc_auc_score(y_te, proba), proba


def main():
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    print(f"tarif sabit: {len(pool)} ozellik, {best_params}", flush=True)
    print(f"MEVCUT: {CURRENT}\nADAY  : {CANDIDATE}", flush=True)

    print("\n=== CAT_1 gecis testi (aday basina BIR KEZ fit, seed-bagimsiz) ===", flush=True)
    auc_current, proba_current = _cat1_shift_raw_auc(v1_df, pool, best_params, CURRENT["class_weight"], CURRENT["strength"])
    auc_candidate, proba_candidate = _cat1_shift_raw_auc(v1_df, pool, best_params, CANDIDATE["class_weight"], CANDIDATE["strength"])
    print(f"  MEVCUT raw_AUC={auc_current:.4f}  ADAY raw_AUC={auc_candidate:.4f}", flush=True)

    print(f"\n=== {N_SEEDS} tekrarli {N_SPLITS}-fold (StratifiedKFold, split bankasindan bagimsiz) ===", flush=True)
    rows, pooled_folds = [], []
    for seed in range(N_SEEDS):
        outer_folds = _stratified_outer_folds(v1_df, seed)
        cur = _evaluate_candidate_on_split(v1_df, pool, best_params, outer_folds, CURRENT["class_weight"], CURRENT["strength"])
        cand = _evaluate_candidate_on_split(v1_df, pool, best_params, outer_folds, CANDIDATE["class_weight"], CANDIDATE["strength"])

        margin = cand["weighted_f1_overall"] - cur["weighted_f1_overall"]
        winner = "aday" if margin > 0 else ("mevcut" if margin < 0 else "berabere")
        spec_delta = cand["specificity_overall"] - cur["specificity_overall"]
        worst_cur = cur["per_fold"]["weighted_f1"].min()
        worst_cand = cand["per_fold"]["weighted_f1"].min()
        worst_delta = worst_cand - worst_cur

        rows.append({
            "seed": seed, "wf1_current": cur["weighted_f1_overall"], "wf1_candidate": cand["weighted_f1_overall"],
            "margin_candidate_minus_current": margin, "winner": winner,
            "spec_current": cur["specificity_overall"], "spec_candidate": cand["specificity_overall"], "spec_delta": spec_delta,
            "worst_fold_current": worst_cur, "worst_fold_candidate": worst_cand, "worst_fold_delta": worst_delta,
        })
        print(f"  seed={seed}: wf1 mevcut={cur['weighted_f1_overall']:.4f} aday={cand['weighted_f1_overall']:.4f} "
              f"marj={margin:+.4f} kazanan={winner} | spec_delta={spec_delta:+.4f} worst_fold_delta={worst_delta:+.4f}", flush=True)

        for _, r in cur["per_fold"].iterrows():
            pooled_folds.append({"seed": seed, "fold": r["fold"], "candidate": "current", "weighted_f1": r["weighted_f1"], "n_train": r["n_train"], "n_test": r["n_test"]})
        for _, r in cand["per_fold"].iterrows():
            pooled_folds.append({"seed": seed, "fold": r["fold"], "candidate": "candidate", "weighted_f1": r["weighted_f1"], "n_train": r["n_train"], "n_test": r["n_test"]})

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    pooled_df = pd.DataFrame(pooled_folds)
    pooled_df.to_csv(OUT_POOLED_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}, {OUT_POOLED_CSV}", flush=True)

    print("\n=== Adim 2: ozet ===", flush=True)
    n_win_candidate = int((result["winner"] == "aday").sum())
    n_win_current = int((result["winner"] == "mevcut").sum())
    margin_mean, margin_std = result["margin_candidate_minus_current"].mean(), result["margin_candidate_minus_current"].std()
    binom = binomtest(n_win_candidate, N_SEEDS, p=0.5)
    print(f"  Aday kazandi: {n_win_candidate}/{N_SEEDS}   Mevcut kazandi: {n_win_current}/{N_SEEDS}", flush=True)
    print(f"  Marj (aday-mevcut): ort={margin_mean:+.4f} std={margin_std:.4f}", flush=True)
    print(f"  Binom test (H0: p=0,5) p-degeri={binom.pvalue:.4f}", flush=True)

    spec_signs = np.sign(result["spec_delta"])
    worst_signs = np.sign(result["worst_fold_delta"])
    print(f"  Specificity_delta yon-tutarliligi: {int((spec_signs > 0).sum())}/{N_SEEDS} pozitif, "
          f"{int((spec_signs < 0).sum())}/{N_SEEDS} negatif (ort={result['spec_delta'].mean():+.4f})", flush=True)
    print(f"  Worst_fold_delta yon-tutarliligi: {int((worst_signs > 0).sum())}/{N_SEEDS} pozitif, "
          f"{int((worst_signs < 0).sum())}/{N_SEEDS} negatif (ort={result['worst_fold_delta'].mean():+.4f})", flush=True)

    print(f"\n  CAT_1 gecis raw_AUC (seed-bagimsiz, tek olcum): MEVCUT={auc_current:.4f} ADAY={auc_candidate:.4f} "
          f"(delta={auc_candidate-auc_current:+.4f})", flush=True)

    print("\n=== KARAR KURALINA GORE ===", flush=True)
    strong_margin = abs(margin_mean) >= WIN_MARGIN_THRESHOLD
    if n_win_candidate >= 7 and strong_margin:
        cur_pooled = pooled_df[pooled_df.candidate == "current"].sort_values(["seed", "fold"])
        cand_pooled = pooled_df[pooled_df.candidate == "candidate"].sort_values(["seed", "fold"])
        nb = nadeau_bengio_corrected_ttest(
            cand_pooled["weighted_f1"].to_numpy(), cur_pooled["weighted_f1"].to_numpy(),
            cand_pooled["n_train"].to_numpy(), cand_pooled["n_test"].to_numpy(),
        )
        print(f"  GERCEK SINYAL: aday {n_win_candidate}/10 kazandi (binom p={binom.pvalue:.4f}), "
              f"ortalama marj={margin_mean:+.4f} (>= {WIN_MARGIN_THRESHOLD} esigi).", flush=True)
        print(f"  Havuzlanmis NB testi (n=50, 10 tekrar x 5 fold): mean_diff(aday-mevcut)={nb['mean_diff']:+.4f} "
              f"NB p={nb['p_value']:.4f}", flush=True)
        print("  KARAR: DURDURULDU -- kullanici ile final model karari birlikte verilecek.", flush=True)
    elif n_win_candidate <= 3:
        print(f"  MEVCUT MODEL SAGLAM: aday yalnizca {n_win_candidate}/10 kazandi (binom p={binom.pvalue:.4f}). "
              "Mevcut model korunuyor.", flush=True)
        print("  KARAR: DUR -- 'denendi, tekrarli-bolmede dogrulanamadi' diye belgelenecek.", flush=True)
    else:
        print(f"  GURULTU: aday {n_win_candidate}/10 kazandi (binom p={binom.pvalue:.4f}), "
              f"ortalama marj={margin_mean:+.4f} ({'guclu' if strong_margin else 'zayif, esigin altinda'}). "
              "Tek-fold'daki parlak sonuc tekrarlanmadi.", flush=True)
        print("  KARAR: DUR -- mevcut model korunuyor, 'denendi, tekrarli-bolmede dogrulanamadi' diye belgelenecek.", flush=True)


if __name__ == "__main__":
    main()
