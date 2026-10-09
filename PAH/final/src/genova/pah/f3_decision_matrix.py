"""F3 sonrasi karar matrisi, Adim 2-4: aday B (23 ozellik, `AL_49`
tutuldu) ve C (22 ozellik, F1 Ek'in adayi) icin F3'un CAT_1 doluluk-
gecisi coku testini tekrarlar + madde 11/F1-Ek'in AYNI iki-katmanli
disipliniyle (10-tekrarli bolme win-count/marj + sabit 5-fold NB testi,
weighted-F1 uzerinde) A'ya (mevcut resmi, 26 ozellik) karsi ortalama
maliyeti olcer. HICBIR KARAR VERMEZ/UYGULAMAZ -- yalnizca olcer ve
karar matrisini raporlar. `final_model_bundle_v2.pkl`/`predict.py`'ye
dokunmaz.

Calistirma: python -m genova.pah.f3_decision_matrix
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from genova.metrics import f1_binary_positive_weighted
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import CATBOOST_GRID, fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR, F0_SEED
from genova.pah.f0_final_model_v3 import prior_weighted_inner_score
from genova.pah.f3_robustness_stress_tests import fit_and_score_on_split
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.e5_threshold_selection import _prior_weights, FINAL_PATHOGENIC_PRIOR
from genova.statistics import nadeau_bengio_corrected_ttest

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
COLLAPSE_OUT = TAB_DIR / "f3_decision_collapse_test.csv"
STABILITY_OUT = TAB_DIR / "f3_decision_stability_check.csv"
NB_OUT = TAB_DIR / "f3_decision_nb_test.csv"

BUNDLE_A_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
BUNDLE_B_PATH = MODEL_DIR / "final_model_bundle_f3_candidate_b23.pkl"
BUNDLE_C_PATH = MODEL_DIR / "final_model_bundle_f1_ablation_candidate.pkl"
EXISTING_C_STABILITY_CSV = TAB_DIR / "f1_madde_stability_check.csv"

N_STABILITY_SEEDS = 10
N_HP_SPLITS = 4
N_OOF_SPLITS = 5


# ---------------------------------------------------------- Adim 2: cokus testi

def collapse_test(v1_df, bundle, label):
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    calibrator, w1, w0, threshold = bundle["calibrator"], bundle["w1"], bundle["w0"], bundle["threshold"]

    rows = []
    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat1_filled_ids, cat1_empty_ids)
    r["candidate"], r["direction"] = label, "train=CAT_1_dolu -> test=CAT_1_bos"
    rows.append(r)
    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat1_empty_ids, cat1_filled_ids)
    r["candidate"], r["direction"] = label, "train=CAT_1_bos -> test=CAT_1_dolu (KRITIK)"
    rows.append(r)
    return rows


# ------------------------------------------------- Adim 3a: tekrarli-bolme testi

def select_hp(v1_df, pool, seed):
    skf = StratifiedKFold(n_splits=N_HP_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for params in CATBOOST_GRID:
        fold_scores = []
        for train_idx, val_idx in skf.split(v1_df, v1_df["Label"]):
            train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
            val_ids = v1_df.iloc[val_idx]["Variant_ID"].tolist()
            X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, train_ids, val_ids, pool)
            raw_proba = fit_predict_catboost(X_tr, y_tr, X_va, params)
            fold_scores.append(prior_weighted_inner_score(raw_proba, y_va))
        scores.append(float(np.mean(fold_scores)))
    return CATBOOST_GRID[int(np.argmax(scores))]


def cross_fit_oof_plain(v1_df, pool, best_params, seed):
    skf = StratifiedKFold(n_splits=N_OOF_SPLITS, shuffle=True, random_state=seed)
    proba_by_pos, y_by_pos = {}, {}
    for train_idx, test_idx in skf.split(v1_df, v1_df["Label"]):
        train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
        test_ids = v1_df.iloc[test_idx]["Variant_ID"].tolist()
        X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
        proba = fit_predict_catboost(X_tr, y_tr, X_te, best_params)
        for idx, p, y in zip(test_idx, proba, y_te):
            proba_by_pos[idx] = p
            y_by_pos[idx] = y
    ordered = sorted(proba_by_pos.keys())
    return np.array([proba_by_pos[i] for i in ordered]), np.array([y_by_pos[i] for i in ordered])


def pool_score(v1_df, pool, seed):
    from genova.pah.calibration import BetaCalibrator
    from genova.pah.e5_threshold_selection import select_threshold
    best_params = select_hp(v1_df, pool, seed)
    oof_proba, y_full = cross_fit_oof_plain(v1_df, pool, best_params, seed)
    cal = BetaCalibrator().fit(oof_proba, y_full)
    sld = sld_correct(cal.transform(oof_proba))
    _, best_wf1 = select_threshold(sld, y_full)
    return best_wf1


def stability_check(v1_df, pool_a, pool_new, label, n_seeds=N_STABILITY_SEEDS):
    rows = []
    for seed in range(n_seeds):
        score_a = pool_score(v1_df, pool_a, seed)
        score_new = pool_score(v1_df, pool_new, seed)
        margin_signed = score_new - score_a
        rows.append({"candidate": label, "seed": seed, "score_a": score_a, "score_new": score_new,
                      "winner": label if score_new > score_a else "A", "margin_signed_new_minus_a": margin_signed})
        print(f"  [{label}] seed={seed}: A->{score_a:.4f}  {label}->{score_new:.4f}  "
              f"marj({label}-A)={margin_signed:+.4f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Adim 3b: NB testi

def per_fold_weighted_f1(v1_df, bundle):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    calibrator, w1, w0, threshold = bundle["calibrator"], bundle["w1"], bundle["w0"], bundle["threshold"]
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        proba = fit_predict_catboost(X_tr, y_tr, X_va, best_params)
        calibrated = calibrator.transform(proba)
        sld = sld_correct(calibrated, w1=w1, w0=w0)
        pred = (sld >= threshold).astype(int)
        weights = _prior_weights(y_va, FINAL_PATHOGENIC_PRIOR)
        rows.append({"fold": fold["fold"], "n_train": len(y_tr), "n_test": len(y_va),
                      "weighted_f1": f1_binary_positive_weighted(y_va, pred, weights)})
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    bundle_a = joblib.load(BUNDLE_A_PATH)
    bundle_b = joblib.load(BUNDLE_B_PATH)
    bundle_c = joblib.load(BUNDLE_C_PATH)
    pool_a, pool_b, pool_c = bundle_a["pool"], bundle_b["pool"], bundle_c["pool"]
    print(f"A: {len(pool_a)} ozellik  B: {len(pool_b)} ozellik  C: {len(pool_c)} ozellik", flush=True)

    print("\n=== Adim 2: CAT_1 coku testi (B ve C) ===", flush=True)
    collapse_rows = []
    for label, bundle in (("B_23", bundle_b), ("C_22", bundle_c)):
        rows = collapse_test(v1_df, bundle, label)
        for r in rows:
            print(f"  [{label}] {r['direction']}: F1={r['f1']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
        collapse_rows.extend(rows)
    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(COLLAPSE_OUT, index=False)
    print(f"kaydedildi: {COLLAPSE_OUT}", flush=True)

    print("\n=== Adim 3a: tekrarli-bolme testi (B vs A, 10 tekrar) ===", flush=True)
    stab_b = stability_check(v1_df, pool_a, pool_b, "B_23")
    stab_b.to_csv(STABILITY_OUT, index=False)
    print(f"kaydedildi: {STABILITY_OUT}", flush=True)

    if EXISTING_C_STABILITY_CSV.exists():
        print(f"\n=== C (22-ozellik) icin 10-tekrarli test ZATEN mevcut (F1 Ek turu) -- yeniden hesaplanmiyor, okunuyor ===", flush=True)
        stab_c = pd.read_csv(EXISTING_C_STABILITY_CSV)
        # eski dosyanin kendi kolon adlari: winner in {"22","26"}, margin_signed_22_minus_26
        n_c_wins = int((stab_c["winner"] == "22").sum())
        mean_margin_c = stab_c["margin_signed_22_minus_26"].mean()
        std_margin_c = stab_c["margin_signed_22_minus_26"].std()
    else:
        stab_c = stability_check(v1_df, pool_a, pool_c, "C_22")
        n_c_wins = int((stab_c["winner"] == "C_22").sum())
        mean_margin_c = stab_c["margin_signed_new_minus_a"].mean()
        std_margin_c = stab_c["margin_signed_new_minus_a"].std()

    n_b_wins = int((stab_b["winner"] == "B_23").sum())
    mean_margin_b = stab_b["margin_signed_new_minus_a"].mean()
    std_margin_b = stab_b["margin_signed_new_minus_a"].std()
    print(f"\nB kazandi: {n_b_wins}/10, marj={mean_margin_b:+.4f}±{std_margin_b:.4f}", flush=True)
    print(f"C kazandi: {n_c_wins}/10, marj={mean_margin_c:+.4f}±{std_margin_c:.4f}", flush=True)

    print("\n=== Adim 3b: NB testi (weighted-F1, sabit 5-fold ic CV, A vs B, A vs C) ===", flush=True)
    wf1_a = per_fold_weighted_f1(v1_df, bundle_a)
    wf1_b = per_fold_weighted_f1(v1_df, bundle_b)
    wf1_c = per_fold_weighted_f1(v1_df, bundle_c)
    n_train, n_test = wf1_a["n_train"].to_numpy(), wf1_a["n_test"].to_numpy()

    nb_b = nadeau_bengio_corrected_ttest(wf1_b["weighted_f1"].to_numpy(), wf1_a["weighted_f1"].to_numpy(), n_train, n_test)
    nb_c = nadeau_bengio_corrected_ttest(wf1_c["weighted_f1"].to_numpy(), wf1_a["weighted_f1"].to_numpy(), n_train, n_test)
    print(f"  B-A: mean_diff(B-A)={nb_b['mean_diff']:+.4f} p={nb_b['p_value']:.4f}", flush=True)
    print(f"  C-A: mean_diff(C-A)={nb_c['mean_diff']:+.4f} p={nb_c['p_value']:.4f}", flush=True)

    pd.DataFrame([
        {"candidate": "B_23", "nb_mean_diff": nb_b["mean_diff"], "nb_p_value": nb_b["p_value"]},
        {"candidate": "C_22", "nb_mean_diff": nb_c["mean_diff"], "nb_p_value": nb_c["p_value"]},
    ]).to_csv(NB_OUT, index=False)
    print(f"kaydedildi: {NB_OUT}", flush=True)

    print("\n=== Adim 4: KARAR MATRISI (yalnizca rapor, otomatik secim yok) ===", flush=True)
    b_critical = collapse_df[(collapse_df.candidate == "B_23") & (collapse_df.direction.str.contains("KRITIK"))].iloc[0]
    c_critical = collapse_df[(collapse_df.candidate == "C_22") & (collapse_df.direction.str.contains("KRITIK"))].iloc[0]
    print(f"| Aday | Kritik yon AUC | 10-tekrar kazanma | Marj | NB p |")
    print(f"| A (26) | 0.573 (F3'ten) | -- | -- | -- |")
    print(f"| B (23, AL_49 tutuldu) | {b_critical['raw_auc']:.4f} | {n_b_wins}/10 | {mean_margin_b:+.4f} | {nb_b['p_value']:.4f} |")
    print(f"| C (22, hepsi cikarildi) | {c_critical['raw_auc']:.4f} | {n_c_wins}/10 | {mean_margin_c:+.4f} | {nb_c['p_value']:.4f} |")


if __name__ == "__main__":
    main()
