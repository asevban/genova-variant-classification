"""P1 madde 8: E3'un kalibrator YONTEM SECIMINI (Platt/Beta/Isotonic
arasindan hangisinin kullanilacagi) global'den fold-ici'ne tasir.

Eski (E3) davranis: uc aday model icin de 50 dis-fold'un ORTALAMA Brier'i
karsilastirilip "Beta kazandi" denilip bu TEK karar butun fold'lara
uygulaniyordu -- yontem SECIMI teknik olarak dis-test'e bakilarak
yapiliyordu (kalibratorun FIT'i capraz-fit/sizintisizdi, ama HANGI
yontemin kullanilacagi kararı degildi).

Yeni davranis: HER dis-fold icin, o fold'un kendi ic capraz-fit
orneklemi (`cross_fit_probabilities` -- E3'un ZATEN kurdugu, dis-train'in
ic 4-fold'undan capraz-fit HAM olasiliklar, dis-test'e hic erismiyor)
uzerinde Platt ve Beta AYRI AYRI fit+degerlendirilir (Isotonic atlaniyor,
E3'te asiri-uyum defalarca dogrulanmisti), dusuk Brier veren o FOLD icin
secilir. Fold'un dis-test tahmini SONRA bu secilen yontemle (yalnizca
transform, E3'un kendi disiplini) uretilir.

Bu, E3'un `e3_calibration_metrics.csv`/`e3_calibrated_oof_predictions.
csv`'sini DEGISTIRMEZ (uzerine yazilmaz) -- final model (F0) hattini da
ETKILEMEZ (`f0_final_model.py` zaten kendi OOF'unda Platt/Beta'yi
BAGIMSIZ karsilastirip seciyor). Bu yalnizca E3'un development kaydini
duzeltilmis bir yontemle YENIDEN uretir, ayri dosyalara yazar.

Calistirma: python -m genova.pah.e3_fold_local_calibrator
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah import e2ek_models as em
from genova.pah.calibration import PlattCalibrator, BetaCalibrator, evaluate_probabilities
from genova.pah.e3_calibration_run import cross_fit_probabilities, _lookup_best_params, _variant_ids_in_row_order

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
OLD_POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"  # E3/E3-EK'in FIILEN kullandigi (P0-oncesi) havuz -- bu turda degistirilmiyor
TAB_DIR = ROOT / "reports" / "tables"
FIG_DIR = ROOT / "reports" / "figures"
CHOICE_OUT = TAB_DIR / "e3_fold_local_calibrator_choice.csv"
RELIABILITY_FIG = FIG_DIR / "reliability_diagram_fold_local_calibration.png"

N_REPEATS = 10
WEIGHTING_VARIANT = "B_fixed_minority_2x"


def select_fold_local_calibrator(cross_fit_scores, y_dis_train):
    """P1 madde 8'in cekirdek duzeltmesi: dis-test'e HIC bakmadan, yalnizca
    bu fold'un kendi ic capraz-fit orneklemi (`cross_fit_scores`/`y_dis_
    train` -- ikisi de yalnizca dis-TRAIN'den turer) uzerinde Platt/Beta
    AYRI AYRI fit+degerlendirilir, dusuk Brier veren kazanir. Isotonic
    atlaniyor (E3'te asiri-uyumu defalarca dogrulandi)."""
    scores = {}
    fitted = {}
    for name, cls in (("platt", PlattCalibrator), ("beta", BetaCalibrator)):
        cal = cls().fit(cross_fit_scores, y_dis_train)
        calibrated = cal.transform(cross_fit_scores)
        scores[name] = evaluate_probabilities(y_dis_train, calibrated)["brier"]
        fitted[name] = cal
    winner = min(scores, key=scores.get)
    return winner, fitted[winner], scores


def expected_calibration_error(y_true, proba, n_bins=10):
    y_true, proba = np.asarray(y_true, dtype=float), np.asarray(proba, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (proba >= lo) & (proba <= hi if hi == 1 else proba < hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        bin_accuracy = y_true[in_bin].mean()
        bin_confidence = proba[in_bin].mean()
        ece += (n_in_bin / len(proba)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _reliability_curve(y_true, proba, n_bins=10):
    y_true, proba = np.asarray(y_true, dtype=float), np.asarray(proba, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accuracies = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (proba >= lo) & (proba <= hi if hi == 1 else proba < hi)
        if in_bin.sum() == 0:
            continue
        centers.append(proba[in_bin].mean())
        accuracies.append(y_true[in_bin].mean())
    return np.array(centers), np.array(accuracies)


def process_fold_calibrator_choice(v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx):
    """Tek bir dis-fold icin fold-ici kalibrator secimi + dis-test'e
    (yalnizca TRANSFORM icin, secimde DEGIL) uygulanmasi. `select_fold_
    local_calibrator`'a yalnizca `cross_fit_probabilities`'in dis-TRAIN'den
    urettigi orneklem giriyor -- dis-test (`X_test`/`y_test`) yalnizca
    SECIM YAPILDIKTAN SONRA, secilen kalibratoru bir kez uygulamak icin
    okunuyor (`tests/test_e3_fold_local_calibrator_pah.py`'de dogrulaniyor,
    E5/E6'nin ayni deseniyle)."""
    proba_by_vid, y_by_vid = cross_fit_probabilities(
        v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx,
    )
    vids = list(proba_by_vid.keys())
    cross_fit_scores = np.array([proba_by_vid[v] for v in vids])
    y_dis_train = np.array([y_by_vid[v] for v in vids])

    winner, winner_cal, scores = select_fold_local_calibrator(cross_fit_scores, y_dis_train)
    return winner, winner_cal, scores, vids, y_dis_train


def run_candidate(v1_df, comparison_df, model_name, version_name, builder, fit_predict_fn):
    fold_rows = []
    raw_all, cal_all, y_all = [], [], []

    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            outer_fold_idx = outer_fold["fold"]
            best_params = _lookup_best_params(comparison_df, model_name, version_name, repeat_idx, outer_fold_idx)

            winner, winner_cal, scores, vids, y_dis_train = process_fold_calibrator_choice(
                v1_df, builder, fit_predict_fn, best_params, repeat_idx, outer_fold_idx,
            )

            X_train, y_train, X_test, y_test = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            raw_proba = fit_predict_fn(X_train, y_train, X_test, best_params)
            calibrated_test = winner_cal.transform(raw_proba)

            fold_rows.append({
                "model": model_name, "data_version": version_name,
                "repeat": repeat_idx, "outer_fold": outer_fold_idx,
                "chosen_method": winner, "platt_brier_innersample": scores["platt"], "beta_brier_innersample": scores["beta"],
                "n_dis_train": len(vids), "n_dis_test": len(y_test),
            })
            raw_all.extend(raw_proba.tolist())
            cal_all.extend(calibrated_test.tolist())
            y_all.extend([int(y) for y in y_test])

            print(f"  [{model_name}/{version_name}] repeat={repeat_idx} outer={outer_fold_idx} secilen={winner}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    y_all, raw_all, cal_all = np.array(y_all), np.array(raw_all), np.array(cal_all)
    ece_raw = expected_calibration_error(y_all, raw_all)
    ece_cal = expected_calibration_error(y_all, cal_all)
    return fold_df, ece_raw, ece_cal, y_all, raw_all, cal_all


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    comparison_b_only = comparison_df[comparison_df.weighting_variant == WEIGHTING_VARIANT]
    pool = json.loads(OLD_POOL_PATH.read_text())["features"]

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    # e2_model_comparison.csv artik her model icin birden fazla agirliklandirma
    # varyanti (A/B/C) icerebiliyor (E2-EK, bu script'ten SONRA eklendi) --
    # E3'un orijinal (agirliklandirma sweep'inden ONCEKI) tasarimiyla tutarli
    # olmak icin tum kombinasyonlar Strateji B'ye filtreleniyor (E3-EK'in de
    # yaptigi gibi).
    combos = [
        ("catboost", "v1", v1_builder, m.fit_predict_catboost, comparison_b_only),
        ("catboost", "v4_from_v2", v4v2_builder, m.fit_predict_catboost, comparison_b_only),
        ("lightgbm", "v1", v1_builder, m.fit_predict_lgbm, comparison_b_only),
        ("random_forest", "v4_from_v2", v4v2_builder, em.fit_predict_rf, comparison_b_only),
    ]

    all_fold_rows = []
    ece_summary = []
    reliability_data = {}
    for model_name, version_name, builder, fit_predict_fn, cdf in combos:
        print(f"\n=== {model_name} x {version_name} (fold-ici kalibrator secimi, 50 dis fold) ===", flush=True)
        fold_df, ece_raw, ece_cal, y_all, raw_all, cal_all = run_candidate(v1_df, cdf, model_name, version_name, builder, fit_predict_fn)
        all_fold_rows.append(fold_df)

        n_platt = int((fold_df["chosen_method"] == "platt").sum())
        n_beta = int((fold_df["chosen_method"] == "beta").sum())
        print(f"  {model_name}/{version_name}: platt={n_platt}/50 beta={n_beta}/50  ECE ham={ece_raw:.4f} ECE kalibre={ece_cal:.4f}", flush=True)

        ece_summary.append({
            "model": model_name, "data_version": version_name,
            "n_platt_chosen": n_platt, "n_beta_chosen": n_beta,
            "ece_raw": ece_raw, "ece_calibrated_fold_local": ece_cal,
        })
        reliability_data[f"{model_name}/{version_name}"] = _reliability_curve(y_all, cal_all)

    choice_df = pd.concat(all_fold_rows, ignore_index=True)
    ece_df = pd.DataFrame(ece_summary)
    merged = choice_df.merge(ece_df, on=["model", "data_version"], how="left")
    CHOICE_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CHOICE_OUT, index=False)
    print(f"\nkaydedildi: {CHOICE_OUT}")

    print("\n=== ECE ozeti ===")
    print(ece_df.to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="mukemmel kalibrasyon")
    for label, (centers, accuracies) in reliability_data.items():
        ax.plot(centers, accuracies, marker="o", label=label)
    ax.set_xlabel("ortalama tahmin edilen olasilik (bin)")
    ax.set_ylabel("gozlenen oran (bin)")
    ax.set_title("Reliability Diagram -- fold-ici secilmis kalibratorle (P1 madde 8)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RELIABILITY_FIG, dpi=150)
    plt.close(fig)
    print(f"kaydedildi: {RELIABILITY_FIG}")


if __name__ == "__main__":
    main()
