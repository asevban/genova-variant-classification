"""Asama F3: final modelin (`final_model_bundle_v2.pkl`, degistirilmeden)
kaynak/eksiklik dagilimi kayarsa gercekten cokme riski tasiyip tasimadigini
olcer -- F1'in "AL_26/12/7/49 provenance-riskli ama cikarmanin maliyeti
buyuk, risk bilinclidir kabul edildi" kararinin izlenmesi.

HICBIR KALICI MODEL URETILMIYOR -- Adim 2-4'un "yeniden fit" adimlari,
bundle'in KENDI hiperparametre/havuzuyla, yalnizca bu testin GECICI,
bellekte kalan modelleri. Bundle'in kendi kalibratoru/onsel-duzeltmesi/
esigi (`.transform()` ile) tum stres senaryolarinda TUTARLI sekilde
uygulanir -- yeniden kalibre edilmez (kucuk alt-orneklemlerde yeniden
kalibrasyon gurultulu olurdu; asil soru "ham ayirt edicilik dagilim
kaymasinda ne kadar bozuluyor", zaten dondurulmus karar kuralinin
kendisi degil).

Referans ("normal/karisik CV performansi"): bundle'in kendi 5-fold
cross-fit OOF'u (`f0_final_model.py::cross_fit_oof`, deterministik,
yeniden eğitim degil) -- Adim 1-4'un hepsi bu referansa karsi olculur.

Calistirma: python -m genova.pah.f3_robustness_stress_tests
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity, sensitivity
from genova.pah import fold_versions as fv
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR, cross_fit_oof
from genova.pah.e4_prior_correction import sld_correct
from genova.statistics import monte_carlo_final_f1_simulation

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
BUNDLE_PATH = MODEL_DIR / "final_model_bundle_v2.pkl"
BENIGN_SCAN_OUT = TAB_DIR / "f3_benign_weighted_scan.csv"
SHIFT_TESTS_OUT = TAB_DIR / "f3_source_shift_tests.csv"

TOTAL_N = 350  # sartnamenin gercek final test toplami (100 patojenik + 250 benign)
BENIGN_FRACTIONS = [0.50, 0.60, 0.70, 250 / 350, 0.80]
N_SIMULATIONS = 2000
COLLAPSE_THRESHOLD = 0.15  # mutlak F1 dususu -- "ani buyuk cokme" esigi


def _score(y_true, pred):
    return {
        "n_test": len(y_true),
        "n_test_benign": int((y_true == 0).sum()), "n_test_pathogenic": int((y_true == 1).sum()),
        "f1": f1_binary_positive(y_true, pred), "mcc": matthews_correlation_coefficient(y_true, pred),
        "specificity": specificity(y_true, pred), "sensitivity": sensitivity(y_true, pred),
    }


def fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, train_ids, test_ids):
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    proba = fit_predict_catboost(X_tr, y_tr, X_te, best_params)
    calibrated = calibrator.transform(proba)
    sld = sld_correct(calibrated, w1=w1, w0=w0)
    pred = (sld >= threshold).astype(int)
    row = {"n_train": len(y_tr)}
    row.update(_score(y_te.to_numpy(), pred))
    # esiksiz/kalibrasyondan bagimsiz AUC -- donmus kalibrator/esigin
    # (orijinal model icin ayarlanmis) bu YENIDEN FIT edilen modele
    # uygulanmasindan kaynaklanan bir uyumsuzluk mu, yoksa gercek bir
    # ayirt edicilik cokusu mu oldugunu ayirt etmek icin.
    row["raw_auc"] = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan")
    return row


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]
    calibrator, w1, w0, threshold = bundle["calibrator"], bundle["w1"], bundle["w0"], bundle["threshold"]
    print(f"final bundle: {len(pool)} ozellik, {best_params}, esik={threshold}", flush=True)

    print("\n=== Referans: bundle'in kendi 5-fold cross-fit OOF'u (normal/karisik CV performansi) ===", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated_oof = calibrator.transform(oof_proba)
    sld_oof = sld_correct(calibrated_oof, w1=w1, w0=w0)
    pred_oof = (sld_oof >= threshold).astype(int)
    baseline = _score(y_full, pred_oof)
    baseline["raw_auc"] = roc_auc_score(y_full, oof_proba)
    print(f"  normal CV: F1={baseline['f1']:.4f} MCC={baseline['mcc']:.4f} "
          f"specificity={baseline['specificity']:.4f} sensitivity={baseline['sensitivity']:.4f} "
          f"raw_AUC={baseline['raw_auc']:.4f}", flush=True)

    print(f"\n=== Adim 1: benign-agirlikli stres taramasi ({len(BENIGN_FRACTIONS)} oran, toplam n={TOTAL_N}) ===", flush=True)
    scan_rows = []
    for frac in BENIGN_FRACTIONS:
        n_benign = round(TOTAL_N * frac)
        n_pathogenic = TOTAL_N - n_benign
        mc = monte_carlo_final_f1_simulation(
            all_ids, y_full, sld_oof, n_pathogenic=n_pathogenic, n_benign=n_benign,
            n_simulations=N_SIMULATIONS, threshold=threshold,
        )
        print(f"  benign%={frac*100:.1f} (patojenik={n_pathogenic}, benign={n_benign}): "
              f"F1 ort={mc['mean']:.4f} [{mc['ci_low']:.4f},{mc['ci_high']:.4f}]", flush=True)
        scan_rows.append({
            "benign_fraction": frac, "n_pathogenic": n_pathogenic, "n_benign": n_benign,
            "f1_mean": mc["mean"], "f1_median": mc["median"], "f1_ci_low": mc["ci_low"], "f1_ci_high": mc["ci_high"],
        })
    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_csv(BENIGN_SCAN_OUT, index=False)
    print(f"kaydedildi: {BENIGN_SCAN_OUT}", flush=True)

    shift_rows = []

    print("\n=== Adim 2: CAT_2 (AllofUs) kaynak-gecisi testi ===", flush=True)
    cat2_filled_ids = v1_df.loc[v1_df["CAT_2"].notna(), "Variant_ID"].tolist()
    cat2_empty_ids = v1_df.loc[v1_df["CAT_2"].isna(), "Variant_ID"].tolist()
    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat2_empty_ids, cat2_filled_ids)
    r["test_name"], r["direction"] = "cat2_source_shift", "train=not_AllofUs -> test=AllofUs"
    print(f"  train=CAT_2 bos({len(cat2_empty_ids)}) -> test=CAT_2 dolu({r['n_test']}, benign={r['n_test_benign']}): "
          f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} specificity={r['specificity']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat2_filled_ids, cat2_empty_ids)
    r["test_name"], r["direction"] = "cat2_source_shift", "train=AllofUs -> test=not_AllofUs"
    print(f"  train=CAT_2 dolu({len(cat2_filled_ids)}) -> test=CAT_2 bos({r['n_test']}, benign={r['n_test_benign']}): "
          f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} specificity={r['specificity']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    print("\n=== Adim 3: eksiklik-deseni (al_all_missing=1) holdout testi ===", flush=True)
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    al_all_missing_mask = v1_df[al_cols].isna().all(axis=1)
    train_ids_am = v1_df.loc[~al_all_missing_mask, "Variant_ID"].tolist()
    test_ids_am = v1_df.loc[al_all_missing_mask, "Variant_ID"].tolist()
    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, train_ids_am, test_ids_am)
    r["test_name"], r["direction"] = "al_all_missing_holdout", "train=al_all_missing=0 -> test=al_all_missing=1"
    print(f"  train=al_all_missing=0({len(train_ids_am)}) -> test=al_all_missing=1({r['n_test']}, "
          f"benign={r['n_test_benign']}, KUCUK ORNEKLEM UYARISI): "
          f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} specificity={r['specificity']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    print("\n=== Adim 4: CAT_1 doluluk-gecisi testi (EN KRITIK -- F1'in bulgusunu dogrudan sinar) ===", flush=True)
    cat1_filled_ids = v1_df.loc[v1_df["CAT_1"].notna(), "Variant_ID"].tolist()
    cat1_empty_ids = v1_df.loc[v1_df["CAT_1"].isna(), "Variant_ID"].tolist()
    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat1_filled_ids, cat1_empty_ids)
    r["test_name"], r["direction"] = "cat1_fill_shift", "train=CAT_1_dolu -> test=CAT_1_bos"
    print(f"  train=CAT_1 dolu({len(cat1_filled_ids)}) -> test=CAT_1 bos({r['n_test']}, benign={r['n_test_benign']}): "
          f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} specificity={r['specificity']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    r = fit_and_score_on_split(v1_df, pool, best_params, calibrator, w1, w0, threshold, cat1_empty_ids, cat1_filled_ids)
    r["test_name"], r["direction"] = "cat1_fill_shift", "train=CAT_1_bos -> test=CAT_1_dolu"
    print(f"  train=CAT_1 bos({len(cat1_empty_ids)}) -> test=CAT_1 dolu({r['n_test']}, benign={r['n_test_benign']}): "
          f"F1={r['f1']:.4f} MCC={r['mcc']:.4f} specificity={r['specificity']:.4f} raw_AUC={r['raw_auc']:.4f}", flush=True)
    shift_rows.append(r)

    shift_df = pd.DataFrame(shift_rows)
    for col, val in baseline.items():
        shift_df[f"baseline_{col}"] = val
    shift_df["f1_drop_vs_baseline"] = baseline["f1"] - shift_df["f1"]
    shift_df.to_csv(SHIFT_TESTS_OUT, index=False)
    print(f"\nkaydedildi: {SHIFT_TESTS_OUT}", flush=True)

    print("\n=== Adim 5: sentez ===", flush=True)
    max_drop = shift_df["f1_drop_vs_baseline"].max()
    worst_row = shift_df.loc[shift_df["f1_drop_vs_baseline"].idxmax()]
    print(f"normal CV F1={baseline['f1']:.4f} (raw_AUC={baseline['raw_auc']:.4f}); en buyuk dusus: "
          f"{worst_row['test_name']} ({worst_row['direction']}) -> F1={worst_row['f1']:.4f} "
          f"(raw_AUC={worst_row['raw_auc']:.4f}), dusus={max_drop:.4f}", flush=True)
    if max_drop > COLLAPSE_THRESHOLD:
        print(f"\nSONUC: >{COLLAPSE_THRESHOLD} mutlak F1 dususu VAR -- model KIRILGAN, F1'in karari "
              f"YENIDEN GOZDEN GECIRILMELI.", flush=True)
    else:
        print(f"\nSONUC: hicbir testte >{COLLAPSE_THRESHOLD} mutlak F1 dususu yok -- model SAGLAM, "
              f"F1'in karari (riski kabul et) DOGRULANDI.", flush=True)


if __name__ == "__main__":
    main()
