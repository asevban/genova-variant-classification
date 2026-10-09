"""Juri icin final performans karti (13_FINAL_PERFORMANCE_CARD_PAH.md)
icin TUM sayilari hesaplar. HICBIR model/esik/dosya DEGISTIRILMEZ --
yalnizca mevcut final bundle'in KENDI deterministik OOF'u (`f0_final_
model.py::cross_fit_oof`, degismedi) yeniden uretilir (yeniden EGITIM
degil) ve bunun uzerine iki senaryo (Blok A: dogal 369-satir dagilimi,
Blok B: sartnamenin 100P/250B final-projeksiyonu) icin metrik+CI
hesaplanir.

Blok A = `f0_uncertainty_analysis.py`'nin "sample_level_bootstrap"
analiziyle AYNI (F1/MCC/spec/sens/AUROC zaten orada var, burada
DOGRULANIYOR + AUPRC/Precision/Balanced-Accuracy/Brier/LogLoss
EKLENIYOR).
Blok B = `f0_uncertainty_analysis.py`'nin "monte_carlo_100_250"
analiziyle AYNI kaynak/yontem (`genova.statistics.monte_carlo_final_
f1_simulation` ile AYNI resample semasi -- ayni seed, ayni rng.choice
sirasi, F1 sonucu BIT-BIT ayni cikmasi beklenir, capraz-dogrulama icin
kontrol edilir), ama TUM metrikler icin genisletilmis (o fonksiyon
yalnizca F1 donduruyordu, degistirilmedi -- burada AYNI resampling
semasini tekrar eden YENI bir fonksiyon yazildi, kucuk bilincli kod
tekrari, tum onceki turlardaki ayni gerekce).

Calistirma: python -m genova.pah.f0_final_performance_card
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    log_loss, precision_score, roc_auc_score,
)

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, MODEL_DIR, cross_fit_oof
from genova.pah.e4_prior_correction import sld_correct
from genova.statistics import sample_level_bootstrap_ci

BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
OUT_CSV = TAB_DIR / "f0_final_performance_card.csv"
N_BOOT = 1000
N_SIMULATIONS = 2000
N_PATHOGENIC_FINAL, N_BENIGN_FINAL = 100, 250
MC_SEED = 42


def _balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


def _precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def _auprc(y_true, proba):
    return average_precision_score(y_true, proba)


def _brier(y_true, proba):
    return brier_score_loss(y_true, proba)


def _logloss(y_true, proba):
    return log_loss(y_true, proba, labels=[0, 1])


def block_a_metrics(variant_ids, y_full, pred, sld):
    """Blok A -- 369 satirlik OOF'un DOGAL (egitim) kompozisyonu, bundle'in
    kendi esigi (0,35) + SLD-duzeltilmis olasiligi. `f0_uncertainty_
    analysis.py`'nin sample_level_bootstrap analiziyle AYNI mekanizma."""
    rows = []
    hard_label_metrics = [
        ("f1", f1_binary_positive), ("mcc", matthews_correlation_coefficient),
        ("specificity", specificity), ("sensitivity", sensitivity),
        ("balanced_accuracy", _balanced_accuracy), ("precision", _precision),
    ]
    for name, fn in hard_label_metrics:
        ci = sample_level_bootstrap_ci(variant_ids, y_full, pred, fn, n_boot=N_BOOT)
        rows.append({"metric": name, **ci})

    proba_metrics = [("auroc", roc_auc_score), ("auprc", _auprc), ("brier", _brier), ("logloss", _logloss)]
    for name, fn in proba_metrics:
        ci = sample_level_bootstrap_ci(variant_ids, y_full, sld, fn, n_boot=N_BOOT)
        rows.append({"metric": name, **ci})

    return pd.DataFrame(rows)


def monte_carlo_full_metrics(variant_ids, y_true, proba, threshold, n_pathogenic=N_PATHOGENIC_FINAL,
                              n_benign=N_BENIGN_FINAL, n_simulations=N_SIMULATIONS, seed=MC_SEED):
    """`genova.statistics.monte_carlo_final_f1_simulation` ile AYNI
    resampling semasi (ayni seed, ayni rng.choice cagri sirasi -- F1
    capraz-dogrulama icin bit-bit ayni cikmasi beklenir), TUM metrikler
    icin genisletilmis."""
    variant_ids = list(variant_ids)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pathogenic_idx = np.flatnonzero(y_true == 1)
    benign_idx = np.flatnonzero(y_true == 0)

    rng = np.random.RandomState(seed)
    metric_names = ["f1", "mcc", "specificity", "sensitivity", "balanced_accuracy", "precision",
                     "auroc", "auprc", "brier", "logloss"]
    scores = {m: np.empty(n_simulations) for m in metric_names}

    for i in range(n_simulations):
        sim_idx = np.concatenate([
            rng.choice(pathogenic_idx, size=n_pathogenic, replace=True),
            rng.choice(benign_idx, size=n_benign, replace=True),
        ])
        y_sim = y_true[sim_idx]
        proba_sim = proba[sim_idx]
        pred_sim = (proba_sim >= threshold).astype(int)

        scores["f1"][i] = f1_binary_positive(y_sim, pred_sim)
        scores["mcc"][i] = matthews_correlation_coefficient(y_sim, pred_sim)
        scores["specificity"][i] = specificity(y_sim, pred_sim)
        scores["sensitivity"][i] = sensitivity(y_sim, pred_sim)
        scores["balanced_accuracy"][i] = _balanced_accuracy(y_sim, pred_sim)
        scores["precision"][i] = _precision(y_sim, pred_sim)
        scores["auroc"][i] = roc_auc_score(y_sim, proba_sim)
        scores["auprc"][i] = _auprc(y_sim, proba_sim)
        scores["brier"][i] = _brier(y_sim, proba_sim)
        scores["logloss"][i] = _logloss(y_sim, proba_sim)

    rows = []
    for m in metric_names:
        vals = scores[m]
        rows.append({
            "metric": m, "point_estimate": float(vals.mean()),
            "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)),
            "n_boot": n_simulations,
        })
    return pd.DataFrame(rows), scores


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params, threshold = bundle["pool"], bundle["model_best_params"], bundle["threshold"]

    print(f"final bundle: {len(pool)} ozellik, esik={threshold}, kalibrator={bundle['calibrator_name']}", flush=True)
    print("bundle'in kendi 5-fold cross-fit OOF'u yeniden uretiliyor (deterministik, yeniden egitim degil)...", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= threshold).astype(int)
    variant_ids = v1_df["Variant_ID"].tolist()

    n_pathogenic_native = int((y_full == 1).sum())
    n_benign_native = int((y_full == 0).sum())
    print(f"dogal (egitim) kompozisyon: n={len(y_full)}, patojenik={n_pathogenic_native} "
          f"({n_pathogenic_native/len(y_full)*100:.1f}%), benign={n_benign_native} "
          f"({n_benign_native/len(y_full)*100:.1f}%)", flush=True)

    print("\n=== Blok A: dogal (369 satir) kompozisyon, ornek-duzeyi bootstrap CI ===", flush=True)
    block_a = block_a_metrics(variant_ids, y_full, pred, sld)
    print(block_a.to_string(index=False), flush=True)

    print("\n=== Blok B: final-projeksiyon (100P/250B) Monte Carlo, TUM metrikler ===", flush=True)
    block_b, mc_scores = monte_carlo_full_metrics(variant_ids, y_full, sld, threshold)
    print(block_b.to_string(index=False), flush=True)

    print("\n=== Capraz-dogrulama: F1 Monte Carlo, mevcut fonksiyonla AYNI mi? ===", flush=True)
    from genova.statistics import monte_carlo_final_f1_simulation
    ref_mc = monte_carlo_final_f1_simulation(variant_ids, y_full, sld, n_pathogenic=N_PATHOGENIC_FINAL,
                                              n_benign=N_BENIGN_FINAL, n_simulations=N_SIMULATIONS, threshold=threshold)
    my_f1_mean = block_b[block_b.metric == "f1"]["point_estimate"].iloc[0]
    print(f"  mevcut fonksiyon F1 ort={ref_mc['mean']:.6f}  bu script F1 ort={my_f1_mean:.6f}  "
          f"fark={abs(ref_mc['mean']-my_f1_mean):.2e}", flush=True)
    assert abs(ref_mc["mean"] - my_f1_mean) < 1e-9, "Monte Carlo resampling semasi tutarsiz!"
    print("  BIT-BIT AYNI -- resampling semasi dogrulandi.", flush=True)

    block_a["block"] = "A_dogal_kompozisyon"
    block_b["block"] = "B_final_projeksiyon_100P_250B"
    block_b = block_b.rename(columns={"point_estimate": "point_estimate"})
    combined = pd.concat([
        block_a[["block", "metric", "point_estimate", "ci_low", "ci_high", "n_boot"]],
        block_b[["block", "metric", "point_estimate", "ci_low", "ci_high", "n_boot"]],
    ], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
