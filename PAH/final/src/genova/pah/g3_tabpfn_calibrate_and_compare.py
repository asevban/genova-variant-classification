"""TabPFN v2 denemesi -- Adim 4: izole ortamin urettigi HAM TabPFN
tahminlerini (`calib_v4fromv2_predictions.csv`, hem outer hem inner
bolmeler) alip, E3/E4/E5'in AYNI (degistirilmemis) fonksiyonlariyla --
`BetaCalibrator`, `sld_correct`, `select_threshold` -- her dis fold icin
kendi ic capraz-fit orneklemine gore bir esik secer, bu esigi yalnizca o
fold'un dis-test'ine uygular (E5'in `process_fold`'uyla BIREBIR AYNI
disiplin, kod kopyalanmadi -- ayni fonksiyonlar cagriliyor).

Sonucu final adayin (CatBoost/v4_from_v2, `e5_threshold_selection.csv`)
weighted_f1_chosen'ina karsi `nadeau_bengio_corrected_ttest` ve
`sample_level_bootstrap_ci` ile karsilastirir.

Calistirma: python -m genova.pah.g3_tabpfn_calibrate_and_compare --predictions-csv <csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, f1_binary_positive_weighted, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah.calibration import BetaCalibrator
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import select_threshold, _prior_weights
from genova.statistics import nadeau_bengio_corrected_ttest, sample_level_bootstrap_ci

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"


def process_fold(preds, repeat_idx, fold_idx):
    inner = preds[(preds.kind == "inner") & (preds.repeat == repeat_idx) & (preds.fold == fold_idx)]
    outer = preds[(preds.kind == "outer") & (preds.repeat == repeat_idx) & (preds.fold == fold_idx)]

    raw_inner = inner["proba"].to_numpy()
    y_inner = inner["y_true"].to_numpy()
    beta_cal = BetaCalibrator().fit(raw_inner, y_inner)
    calibrated_inner = beta_cal.transform(raw_inner)
    sld_inner = sld_correct(calibrated_inner)
    chosen_threshold, inner_best_wf1 = select_threshold(sld_inner, y_inner)

    raw_outer = outer["proba"].to_numpy()
    y_outer = outer["y_true"].to_numpy()
    calibrated_outer = beta_cal.transform(raw_outer)
    sld_outer = sld_correct(calibrated_outer)
    pred_chosen = (sld_outer >= chosen_threshold).astype(int)
    target_weights = _prior_weights(y_outer, FINAL_PATHOGENIC_PRIOR)

    return {
        "repeat": repeat_idx, "outer_fold": fold_idx,
        "n_dis_train": len(raw_inner), "n_dis_test": len(y_outer),
        "chosen_threshold": chosen_threshold, "inner_best_weighted_f1": inner_best_wf1,
        "f1_chosen": f1_binary_positive(y_outer, pred_chosen),
        "mcc_chosen": matthews_correlation_coefficient(y_outer, pred_chosen),
        "specificity_chosen": specificity(y_outer, pred_chosen),
        "sensitivity_chosen": sensitivity(y_outer, pred_chosen),
        "weighted_f1_chosen": f1_binary_positive_weighted(y_outer, pred_chosen, target_weights),
        "variant_id": outer["variant_id"].tolist(), "y_outer": y_outer.tolist(),
        "sld_outer": sld_outer.tolist(), "pred_chosen_list": pred_chosen.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions_csv)
    combos = preds[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    assert len(combos) == 50, f"50 dis fold bekleniyordu, {len(combos)} bulundu"

    rows, pooled_repeat0 = [], []
    for _, c in combos.iterrows():
        repeat_idx, fold_idx = int(c["repeat"]), int(c["fold"])
        detail = process_fold(preds, repeat_idx, fold_idx)
        if repeat_idx == 0:
            pooled_repeat0.append(pd.DataFrame({
                "variant_id": detail.pop("variant_id"), "y_true": detail.pop("y_outer"),
                "sld_proba": detail.pop("sld_outer"), "pred_chosen": detail.pop("pred_chosen_list"),
            }))
        else:
            for k in ("variant_id", "y_outer", "sld_outer", "pred_chosen_list"):
                detail.pop(k)
        rows.append(detail)
        print(f"  repeat={detail['repeat']} fold={detail['outer_fold']} thr={detail['chosen_threshold']:.2f} "
              f"f1={detail['f1_chosen']:.4f} wf1={detail['weighted_f1_chosen']:.4f}", flush=True)

    result = pd.DataFrame(rows)
    from sklearn.metrics import roc_auc_score
    auroc_per_fold = []
    for _, c in combos.iterrows():
        outer = preds[(preds.kind == "outer") & (preds.repeat == c["repeat"]) & (preds.fold == c["fold"])]
        auroc_per_fold.append(roc_auc_score(outer["y_true"], outer["proba"]))
    result["auroc"] = auroc_per_fold

    out_csv = TAB_DIR / "g3_tabpfn_v4fromv2_e5_style.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nkaydedildi: {out_csv}")

    print("\n=== TabPFN/v4_from_v2 (E5-tarzi) ozet (50 dis fold) ===")
    print(result[["f1_chosen", "mcc_chosen", "sensitivity_chosen", "specificity_chosen", "weighted_f1_chosen", "auroc"]].agg(["mean", "std"]))

    print("\n=== NB testi: TabPFN/v4_from_v2 vs CatBoost/v4_from_v2 (final aday, E5) ===")
    e5 = pd.read_csv(E5_CSV)
    cb = e5[(e5.model == "catboost") & (e5.data_version == "v4_from_v2")].sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
    tp = result.sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
    assert len(cb) == len(tp) == 50
    assert (cb["repeat"].to_numpy() == tp["repeat"].to_numpy()).all()
    assert (cb["outer_fold"].to_numpy() == tp["outer_fold"].to_numpy()).all()

    for metric_col, label in (("weighted_f1_chosen", "weighted_f1 (resmi karsilastirma metrigi)"), ("f1_chosen", "f1 (ham)")):
        nb = nadeau_bengio_corrected_ttest(
            tp[metric_col].to_numpy(), cb[metric_col].to_numpy(),
            cb["n_dis_train"].to_numpy(), cb["n_dis_test"].to_numpy(),
        )
        print(f"  {label}: mean_diff(tabpfn-catboost)={nb['mean_diff']:+.4f}  NB p={nb['p_value']:.4f}")

    print("\n=== TabPFN/v4_from_v2'nin kendi ornek-duzeyi bootstrap CI'si ===")
    print("(repeat=0'in 5 dis-fold'u 369 satirin sizintisiz tam partisyonu -- F0/F2'nin ayni deseni)")
    pooled = pd.concat(pooled_repeat0, ignore_index=True)
    assert len(pooled) == 369, f"repeat=0 icin 369 satir bekleniyordu, {len(pooled)} bulundu"
    variant_ids = pooled["variant_id"].tolist()
    y_true = pooled["y_true"].to_numpy()
    pred_chosen = pooled["pred_chosen"].to_numpy()
    sld_proba = pooled["sld_proba"].to_numpy()

    for metric_name, metric_fn, values in (
        ("f1", f1_binary_positive, pred_chosen), ("mcc", matthews_correlation_coefficient, pred_chosen),
        ("specificity", specificity, pred_chosen), ("sensitivity", sensitivity, pred_chosen),
        ("auroc", roc_auc_score, sld_proba),
    ):
        ci = sample_level_bootstrap_ci(variant_ids, y_true, values, metric_fn, n_boot=2000)
        print(f"  {metric_name}: nokta={ci['point_estimate']:.4f} CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    pooled.to_csv(TAB_DIR / "g3_tabpfn_v4fromv2_repeat0_oof.csv", index=False)


if __name__ == "__main__":
    main()
