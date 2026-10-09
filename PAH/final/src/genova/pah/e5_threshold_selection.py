"""Asama E5: nested esik secimi. Esik, DIS-TEST'e HICBIR SEKILDE
BAKILMADAN, yalnizca dis-train'in kendi ic capraz-fit orneklemi uzerinde
secilir -- E3'un sizinti disiplini burada da aynen gecerli.

Her dis fold icin:
  1. E3'un `cross_fit_probabilities`'i (ayni best_params, yeniden arama
     yok) yeniden calistirilir -- dis-train'in TAMAMI icin sizintisiz ic
     capraz-fit HAM olasiliklar.
  2. Bu ic olasiliklar, o fold icin YENIDEN fit edilen (E3'unkiyle
     matematiksel olarak birebir ayni, deterministik) Beta kalibratoruyle
     donusturulur, sonra E4'un SLD formuluyle duzeltilir.
  3. Onsel-agirlikli (w1/w0) F1'i (`f1_binary_positive_weighted`) ince bir
     esik izgarasinda (0,05-0,95, adim 0,01) maksimize eden esik secilir
     (birden fazla esik ayni maksimumu veriyorsa, kararlilik icin bu
     esik-platosunun ORTASI secilir).
  4. Bu esik, YALNIZCA bu fold'un E4-ciktisi (dis-test'in SLD-duzeltilmis
     olasiliklari, `e4_prior_corrected_probabilities.csv`'den okunur,
     yeniden hesaplanmiyor) uzerinde bir KEZ uygulanir.

Duyarlilik egrisi: ayni (sabit) secilmis esik + SLD-duzeltilmis dis-test
olasiliklari kullanilarak, yalnizca DEGERLENDIRME agirliklandirmasinda
("gercek" onsel farkli cikarsa ne olur") 5 farkli varsayilan final onsel
degeriyle (0,20 - 0,38 arasi) agirlikli-F1 yeniden hesaplanir -- SLD
duzeltmesi veya esik YENIDEN OPTIMIZE EDILMEZ, yalnizca "sistem sabit
kalirsa gercek onsel kayarsa ne olur" sorusuna cevap verir.

Calistirma: python -m genova.pah.e5_threshold_selection
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, f1_binary_positive_weighted, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah.calibration import BetaCalibrator
from genova.pah.e3_calibration_run import cross_fit_probabilities, _lookup_best_params
from genova.pah.e4_prior_correction import sld_correct, W1, W0, FINAL_PATHOGENIC_PRIOR

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
E4_CSV = ROOT / "reports" / "tables" / "e4_prior_corrected_probabilities.csv"
OUT_CSV = ROOT / "reports" / "tables" / "e5_threshold_selection.csv"

N_REPEATS = 10
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
SENSITIVITY_PRIORS = [0.20, 0.25, FINAL_PATHOGENIC_PRIOR, 0.33, 0.38]
FIXED_ALTERNATIVE_THRESHOLD = 0.5  # SLD-sonrasi olceklerde teorik olarak "dogal" basit esik


def _prior_weights(y_true, prior):
    w1 = prior / 0.833
    w0 = (1 - prior) / (1 - 0.833)
    return np.where(np.asarray(y_true) == 1, w1, w0)


def select_threshold(sld_inner, y_dis_train):
    weights = np.where(y_dis_train == 1, W1, W0)
    scores = []
    for thr in THRESHOLD_GRID:
        pred = (sld_inner >= thr).astype(int)
        scores.append(f1_binary_positive_weighted(y_dis_train, pred, weights))
    scores = np.array(scores)
    best = scores.max()
    tied = THRESHOLD_GRID[np.isclose(scores, best, atol=1e-9)]
    return float(tied.mean()), float(best)


def process_fold(v1_df, comparison_df, e4_sub, model_name, version_name, builder, fit_predict_fn, repeat_idx, outer_fold_idx):
    """Tek bir dis fold icin E5 satiri. Esik SECIMI (`chosen_threshold`)
    yalnizca `cross_fit_probabilities`/`select_threshold`'a bagli -- ikisi
    de `e4_sub` (dis-test) parametresini GORMEZ; `e4_sub` yalnizca esik
    SECILDIKTEN SONRA, o esigi bir kez uygulamak icin okunur. Bu ayrim
    `tests/test_e5_threshold_selection_pah.py`'de dogrulaniyor."""
    best_params = _lookup_best_params(comparison_df, model_name, version_name, repeat_idx, outer_fold_idx)

    cross_fit_proba, y_by_vid = cross_fit_probabilities(
        v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx,
    )
    vids = list(cross_fit_proba.keys())
    raw_inner = np.array([cross_fit_proba[v] for v in vids])
    y_dis_train = np.array([y_by_vid[v] for v in vids])

    beta_cal = BetaCalibrator().fit(raw_inner, y_dis_train)
    calibrated_inner = beta_cal.transform(raw_inner)
    sld_inner = sld_correct(calibrated_inner)

    chosen_threshold, inner_best_weighted_f1 = select_threshold(sld_inner, y_dis_train)

    test_rows = e4_sub[(e4_sub.repeat == repeat_idx) & (e4_sub.outer_fold == outer_fold_idx)]
    y_test = test_rows["y_true"].to_numpy()
    sld_test = test_rows["proba"].to_numpy()

    pred_chosen = (sld_test >= chosen_threshold).astype(int)
    pred_fixed = (sld_test >= FIXED_ALTERNATIVE_THRESHOLD).astype(int)
    target_weights_test = _prior_weights(y_test, FINAL_PATHOGENIC_PRIOR)

    row = {
        "model": model_name, "data_version": version_name,
        "repeat": repeat_idx, "outer_fold": outer_fold_idx,
        "n_dis_train": len(vids), "n_dis_test": len(y_test),
        "chosen_threshold": chosen_threshold, "inner_best_weighted_f1": inner_best_weighted_f1,
        "f1_chosen": f1_binary_positive(y_test, pred_chosen),
        "mcc_chosen": matthews_correlation_coefficient(y_test, pred_chosen),
        "specificity_chosen": specificity(y_test, pred_chosen),
        "weighted_f1_chosen": f1_binary_positive_weighted(y_test, pred_chosen, target_weights_test),
        "f1_fixed05": f1_binary_positive(y_test, pred_fixed),
        "mcc_fixed05": matthews_correlation_coefficient(y_test, pred_fixed),
        "specificity_fixed05": specificity(y_test, pred_fixed),
        "weighted_f1_fixed05": f1_binary_positive_weighted(y_test, pred_fixed, target_weights_test),
    }
    for prior in SENSITIVITY_PRIORS:
        w_test_prior = _prior_weights(y_test, prior)
        row[f"weighted_f1_chosen_prior{prior}"] = f1_binary_positive_weighted(y_test, pred_chosen, w_test_prior)
    return row


def run_candidate(v1_df, comparison_df, e4_df, model_name, version_name, builder, fit_predict_fn):
    e4_sub = e4_df[(e4_df.model == model_name) & (e4_df.data_version == version_name)]
    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            row = process_fold(
                v1_df, comparison_df, e4_sub, model_name, version_name, builder, fit_predict_fn,
                repeat_idx, outer_fold["fold"],
            )
            rows.append(row)
            print(f"  [{model_name}/{version_name}] repeat={repeat_idx} outer={outer_fold['fold']} "
                  f"thr={row['chosen_threshold']:.2f} f1={row['f1_chosen']:.4f}", flush=True)
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    e4_df = pd.read_csv(E4_CSV)
    pool = json.loads((ROOT / "reports" / "tables" / "v4_final_feature_pool.json").read_text())["features"]

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    combos = [
        ("catboost", "v1", v1_builder, m.fit_predict_catboost),
        ("catboost", "v4_from_v2", v4v2_builder, m.fit_predict_catboost),
        ("lightgbm", "v1", v1_builder, m.fit_predict_lgbm),
    ]

    all_rows = []
    for model_name, version_name, builder, fit_predict_fn in combos:
        print(f"=== {model_name} x {version_name} ===", flush=True)
        df = run_candidate(v1_df, comparison_df, e4_df, model_name, version_name, builder, fit_predict_fn)
        all_rows.append(df)
        print(df[["chosen_threshold", "f1_chosen", "weighted_f1_chosen", "weighted_f1_fixed05"]].agg(["mean", "std"]), flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
