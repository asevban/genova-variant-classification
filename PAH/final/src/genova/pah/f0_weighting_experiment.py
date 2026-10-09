"""Kullanicinin kendi hesabiyla dogruladigi uc bulguya dayanan deneme:
final tarifin (26 ozellik, `depth=5,lr=0,05`, Beta, SLD, uyarlanabilir
esik) ayni kalip, yalnizca EGITIM sample_weight'inin degistirilmesi
("benign-agirlikli + domain-dengeli") ortalama performansi bozmadan
specificity/`CAT_1` dayanikligini iyilestirir mi?

    sample_weight(row) = class_weight(Label) x domain_weight(row)^strength

- class_weight: benign (Label=0) satirlara {1,0; 1,25; 1,5; 2,0},
  patojenik (Label=1) satirlara HER ZAMAN 1,0.
  NOT: mevcut uretim modeli (`models.py::_minority_sample_weight`,
  MINORITY_WEIGHT=2,0) zaten benign'e x2,0 uyguluyor -- yani bu izgarada
  class_weight=2,0/strength=0,0 hucresi MEVCUT MODELIN TA KENDISI
  (capraz-dogrulama icin asagida ayrica kontrol edilir).
- domain_weight: `CAT_1` dolu/bos x `AL_` eksiklik seviyesi (dusuk/orta/
  yuksek, egitim-fold'unun KENDI tertilleriyle -- fold-guvenli) olmak
  uzere 6 alt-grup; az temsil edilen grup 1/grup_orani agirlik alir,
  `strength` ussuyle yumusatilir (strength=0 -> etkisiz/1,0; strength=1
  -> tam ters-frekans).

`f0_final_model.py::cross_fit_oof` VE `f3_robustness_stress_tests.py`
DEGISTIRILMEDI -- yalnizca import edilip yeniden kullanildi (referans
icin); agirlikli varyant icin KUCUK, BILINCLI bir kod tekrari var
(`models.py::fit_predict_catboost` sample_weight'i disaridan almiyor).

KESIN KISIT: `final_model_bundle_v2.pkl`/`predict.py`/split bankasi bu
script tarafindan OTOMATIK DEGISTIRILMEZ -- bir aday kazansa bile
yalnizca DURDUR/bildir.

Calistirma: python -m genova.pah.f0_weighting_experiment
"""
import itertools
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive, f1_binary_positive_weighted, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import _category_columns
from genova.pah.e2ek_models import _stringify_categoricals
from genova.pah.e3_calibration_run import _variant_ids_in_row_order
from genova.pah.f0_final_model import (
    V1_PATH, TAB_DIR, F0_SEED,
    cross_fit_oof, select_calibrator, select_prior_and_threshold,
)
from genova.pah.f3_robustness_stress_tests import fit_and_score_on_split, _score, BUNDLE_PATH
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR, W1, W0
from genova.pah.e5_threshold_selection import _prior_weights
from genova.statistics import nadeau_bengio_corrected_ttest

RANDOM_STATE = 42
CLASS_WEIGHTS_BENIGN = (1.0, 1.25, 1.5, 2.0)
STRENGTHS = (0.0, 0.5, 1.0)
N_PATHOGENIC_FINAL, N_BENIGN_FINAL = 100, 250

OUT_GRID_CSV = TAB_DIR / "f0_weighting_experiment_grid.csv"
OUT_FOLD_CSV = TAB_DIR / "f0_weighting_experiment_per_fold.csv"


# ------------------------------------------------------------- domain_weight

def _al_missing_rate(v1_df, variant_ids):
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    sub = v1_df[v1_df["Variant_ID"].isin(variant_ids)]
    return sub.set_index("Variant_ID")[al_cols].isna().mean(axis=1)


def compute_domain_weights(v1_df, train_ids, strength):
    """Yalnizca `train_ids` (fold-guvenli) uzerinden 6 alt-grubun
    (CAT_1 dolu/bos x AL_ eksiklik tertili) payini hesaplayip ters-
    frekans agirligi doner -- {variant_id: domain_weight}."""
    train_df = v1_df[v1_df["Variant_ID"].isin(train_ids)].copy()
    al_rate = _al_missing_rate(v1_df, train_ids)
    # duplicates="drop" nadir kucuk alt-orneklemlerde (orn. CAT_1 stres testinin
    # train kismi) 3'ten az sinir uretebilir -- bu yuzden sabit "low/medium/high"
    # etiketi yerine qcut'un kendi (hashable) interval kategorilerini kullaniyoruz.
    al_level = pd.qcut(al_rate, 3, duplicates="drop")
    cat1_status = train_df.set_index("Variant_ID")["CAT_1"].notna().map({True: "filled", False: "empty"})
    group = cat1_status.astype(str) + "_" + al_level.astype(str)

    group_share = group.value_counts(normalize=True)
    if strength == 0.0:
        domain_weight = pd.Series(1.0, index=group.index)
    else:
        domain_weight_raw = group.map(lambda g: 1.0 / group_share[g])
        domain_weight = domain_weight_raw ** strength
    return domain_weight.to_dict()


def compute_sample_weight(v1_df, train_ids, y_train_ordered, class_weight_benign, strength):
    domain_w = compute_domain_weights(v1_df, train_ids, strength)
    ordered_vids = _variant_ids_in_row_order(v1_df, train_ids)
    dom = np.array([domain_w[v] for v in ordered_vids])
    cls = np.where(np.asarray(y_train_ordered) == 0, class_weight_benign, 1.0)
    return cls * dom


# ---------------------------------------------------------- fit/predict (agirlikli)

def fit_predict_catboost_weighted(X_train, y_train, X_test, params, sample_weight):
    """`models.py::fit_predict_catboost` ile BIREBIR AYNI mimari/HP --
    yalnizca sample_weight disaridan (bkz. modul docstring'i)."""
    cat_cols = _category_columns(X_train)
    model = CatBoostClassifier(
        iterations=100, cat_features=cat_cols, random_seed=RANDOM_STATE,
        verbose=False, thread_count=8, allow_writing_files=False, **params,
    )
    model.fit(_stringify_categoricals(X_train, cat_cols), y_train, sample_weight=sample_weight)
    return model.predict_proba(_stringify_categoricals(X_test, cat_cols))[:, 1]


def _fold_membership(v1_df):
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    fold_by_vid, fold_sizes = {}, {}
    for fold in outer_folds:
        test_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid in test_vids:
            fold_by_vid[vid] = fold["fold"]
        fold_sizes[fold["fold"]] = {"n_train": len(fold["train_variant_ids"]), "n_test": len(test_vids)}
    return outer_folds, fold_by_vid, fold_sizes


def cross_fit_oof_weighted(v1_df, pool, best_params, outer_folds, class_weight_benign, strength):
    proba_by_vid, y_by_vid = {}, {}
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        y_tr_ordered = _variant_ids_in_row_order(v1_df, fold["train_variant_ids"])
        sample_weight = compute_sample_weight(v1_df, fold["train_variant_ids"], y_tr.to_numpy(), class_weight_benign, strength)
        proba = fit_predict_catboost_weighted(X_tr, y_tr, X_va, best_params, sample_weight)
        val_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    ordered_ids = v1_df["Variant_ID"].tolist()
    oof_proba = np.array([proba_by_vid[v] for v in ordered_ids])
    y_full = np.array([y_by_vid[v] for v in ordered_ids])
    return oof_proba, y_full


def fit_and_score_weighted_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, train_ids, test_ids, class_weight_benign, strength):
    """`f3_robustness_stress_tests.py::fit_and_score_on_split` ile AYNI
    mantik -- yalnizca sample_weight'in class+domain agirlikli olmasi."""
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    sample_weight = compute_sample_weight(v1_df, train_ids, y_tr.to_numpy(), class_weight_benign, strength)
    proba = fit_predict_catboost_weighted(X_tr, y_tr, X_te, best_params, sample_weight)
    calibrated = calibrator.transform(proba)
    sld = sld_correct(calibrated, w1=w1, w0=w0)
    pred = (sld >= threshold).astype(int)
    row = {"n_train": len(y_tr)}
    row.update(_score(y_te.to_numpy(), pred))
    row["raw_auc"] = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return row


# --------------------------------------------------------------- degerlendirme

def evaluate_oof(v1_df, oof_proba, y_full, fold_by_vid):
    """f0_final_model.py'nin AYNI zinciri: kalibrator (Beta/Platt) ->
    SLD -> uyarlanabilir esik, TUM OOF'ta BIR KEZ -- sonra fold-bazli
    agirlikli-F1 kirilimi (NB testi/worst-fold icin)."""
    winner_name, winner_cal, cal_scores = select_calibrator(oof_proba, y_full)
    calibrated = winner_cal.transform(oof_proba)
    threshold, inner_best_wf1 = select_prior_and_threshold(calibrated, y_full)
    sld = sld_correct(calibrated)
    pred = (sld >= threshold).astype(int)
    weights = _prior_weights(y_full, FINAL_PATHOGENIC_PRIOR)

    ordered_ids = v1_df["Variant_ID"].tolist()
    fold_idx = np.array([fold_by_vid[v] for v in ordered_ids])
    per_fold = []
    for k in sorted(set(fold_idx)):
        mask = fold_idx == k
        per_fold.append({"fold": k, "weighted_f1": f1_binary_positive_weighted(y_full[mask], pred[mask], weights[mask])})

    return {
        "calibrator_name": winner_name, "calibrator": winner_cal, "threshold": threshold,
        "sensitivity": sensitivity(y_full, pred), "specificity": specificity(y_full, pred),
        "weighted_f1_overall": f1_binary_positive_weighted(y_full, pred, weights),
        "raw_auc_overall": roc_auc_score(y_full, oof_proba),
        "per_fold_weighted_f1": pd.DataFrame(per_fold).sort_values("fold")["weighted_f1"].to_numpy(),
    }


def final_projection_metrics(sens, spec, n_pathogenic=N_PATHOGENIC_FINAL, n_benign=N_BENIGN_FINAL):
    """Final sartname kompozisyonunda (100P/250B) BEKLENEN confusion
    matrix'ten analitik F1/MCC -- kullanicinin kendi dogruladigi yontem
    (Deltaspec=+0,10 -> DeltaF1=+0,072 iliskisi bununla uretiliyor)."""
    tp, fn = n_pathogenic * sens, n_pathogenic * (1 - sens)
    tn, fp = n_benign * spec, n_benign * (1 - spec)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else float("nan")
    return {"f1": f1, "mcc": mcc, "specificity": spec, "sensitivity": sens, "fp": fp}


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    print(f"final bundle tarifi: {len(pool)} ozellik, {best_params}", flush=True)

    outer_folds, fold_by_vid, fold_sizes = _fold_membership(v1_df)
    fold_ids_sorted = sorted(fold_sizes.keys())
    n_train_arr = np.array([fold_sizes[k]["n_train"] for k in fold_ids_sorted])
    n_test_arr = np.array([fold_sizes[k]["n_test"] for k in fold_ids_sorted])

    print("\n=== Referans: MEVCUT uretim modeli (cross_fit_oof, degismedi -- import) ===", flush=True)
    ref_oof, ref_y = cross_fit_oof(v1_df, pool, best_params, all_ids)
    ref = evaluate_oof(v1_df, ref_oof, ref_y, fold_by_vid)
    ref_proj = final_projection_metrics(ref["sensitivity"], ref["specificity"])
    print(f"  esik={ref['threshold']:.2f} sens={ref['sensitivity']:.4f} spec={ref['specificity']:.4f} "
          f"weighted_F1(OOF)={ref['weighted_f1_overall']:.4f}", flush=True)
    print(f"  final-projeksiyon (100P/250B): F1={ref_proj['f1']:.4f} MCC={ref_proj['mcc']:.4f} FP={ref_proj['fp']:.2f}", flush=True)
    print(f"  worst-fold weighted_F1={ref['per_fold_weighted_f1'].min():.4f}", flush=True)

    ref_cat1_r = fit_and_score_on_split(v1_df, pool, best_params, bundle["calibrator"], bundle["w1"], bundle["w0"],
                                         bundle["threshold"],
                                         v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist(),
                                         v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist())
    print(f"  CAT_1 kritik-yon raw_AUC (bundle'in kendi kalibratoru)={ref_cat1_r['raw_auc']:.4f} "
          f"(F3 kayitli referans: 0,5727)", flush=True)

    grid_rows, fold_rows = [], []
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()

    print(f"\n=== Izgara: {len(CLASS_WEIGHTS_BENIGN)}x{len(STRENGTHS)}={len(CLASS_WEIGHTS_BENIGN)*len(STRENGTHS)} kombinasyon ===", flush=True)
    for class_w, strength in itertools.product(CLASS_WEIGHTS_BENIGN, STRENGTHS):
        oof_proba, y_full = cross_fit_oof_weighted(v1_df, pool, best_params, outer_folds, class_w, strength)
        cand = evaluate_oof(v1_df, oof_proba, y_full, fold_by_vid)
        proj = final_projection_metrics(cand["sensitivity"], cand["specificity"])

        nb = nadeau_bengio_corrected_ttest(cand["per_fold_weighted_f1"], ref["per_fold_weighted_f1"], n_train_arr, n_test_arr)

        cat1_r = fit_and_score_weighted_on_split(v1_df, pool, best_params, cand["calibrator"], W1, W0, cand["threshold"],
                                                  cat1_empty_ids, cat1_filled_ids, class_w, strength)

        row = {
            "class_weight_benign": class_w, "strength": strength,
            "threshold": cand["threshold"], "calibrator": cand["calibrator_name"],
            "sensitivity": cand["sensitivity"], "specificity": cand["specificity"],
            "proj_f1": proj["f1"], "proj_mcc": proj["mcc"], "proj_fp": proj["fp"],
            "weighted_f1_oof": cand["weighted_f1_overall"],
            "nb_mean_diff_vs_ref": nb["mean_diff"], "nb_p_value": nb["p_value"],
            "cat1_critical_raw_auc": cat1_r["raw_auc"],
            "worst_fold_weighted_f1": cand["per_fold_weighted_f1"].min(),
        }
        grid_rows.append(row)
        for k, wf1 in zip(fold_ids_sorted, cand["per_fold_weighted_f1"]):
            fold_rows.append({"class_weight_benign": class_w, "strength": strength, "fold": k, "weighted_f1": wf1})

        print(f"  cw={class_w} str={strength}: proj_F1={proj['f1']:.4f} spec={cand['specificity']:.4f} "
              f"NBp={nb['p_value']:.4f} diff={nb['mean_diff']:+.4f} CAT1_AUC={cat1_r['raw_auc']:.4f} "
              f"worst_fold={row['worst_fold_weighted_f1']:.4f}", flush=True)

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_GRID_CSV, index=False)
    pd.DataFrame(fold_rows).to_csv(OUT_FOLD_CSV, index=False)
    print(f"\nkaydedildi: {OUT_GRID_CSV}, {OUT_FOLD_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== Adim 4: karar kurali (onceden sabitlenmis) ===", flush=True)
    ref_spec, ref_worst = ref["specificity"], ref["per_fold_weighted_f1"].min()
    ref_cat1_auc = ref_cat1_r["raw_auc"]

    def passes(row):
        c1 = (row["nb_p_value"] >= 0.05) or (row["nb_mean_diff_vs_ref"] > 0)
        c2 = ((row["specificity"] - ref_spec) >= 0.05) or ((row["cat1_critical_raw_auc"] - ref_cat1_auc) >= 0.10)
        c3 = (row["worst_fold_weighted_f1"] - ref_worst) > -0.03
        return c1 and c2 and c3

    grid_df["passes_all_criteria"] = grid_df.apply(passes, axis=1)
    winners = grid_df[grid_df["passes_all_criteria"]]
    print(grid_df[["class_weight_benign", "strength", "specificity", "nb_p_value", "nb_mean_diff_vs_ref",
                    "cat1_critical_raw_auc", "worst_fold_weighted_f1", "passes_all_criteria"]].to_string(index=False))

    if len(winners) == 0:
        best = grid_df.loc[(grid_df["nb_mean_diff_vs_ref"]).idxmax()]
        print(f"\nSONUC: HICBIR aday uc kriteri BIRDEN saglamiyor. En yakin aday: "
              f"cw={best['class_weight_benign']} str={best['strength']} "
              f"(NBp={best['nb_p_value']:.4f}, diff={best['nb_mean_diff_vs_ref']:+.4f}, "
              f"spec_delta={best['specificity']-ref_spec:+.4f}, cat1_delta={best['cat1_critical_raw_auc']-ref_cat1_auc:+.4f}). "
              f"KARAR: hicbiri benimsenmiyor, DUR.", flush=True)
    else:
        winners_sorted = winners.sort_values(["nb_p_value", "cat1_critical_raw_auc"], ascending=[False, False])
        top = winners_sorted.iloc[0]
        print(f"\nSONUC: {len(winners)} aday KRITERI SAGLIYOR. En iyi: "
              f"cw={top['class_weight_benign']} str={top['strength']} "
              f"(NBp={top['nb_p_value']:.4f}, spec_delta={top['specificity']-ref_spec:+.4f}, "
              f"cat1_delta={top['cat1_critical_raw_auc']-ref_cat1_auc:+.4f}). "
              f"KARAR: DURDURULDU -- kullanici onayi bekleniyor, otomatik degisiklik yapilmadi.", flush=True)


if __name__ == "__main__":
    main()
