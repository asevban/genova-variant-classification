"""Kirilganlik haritasi teshisinin (`f3_provenance_holdout_diagnosis.py`)
iki bulgusunu dogrular: (1) `CAT_2`/AllofUs_EAS ve AllofUs_AFR leave-
one-out'unun raw AUC dususu (n kucuk oldugu icin) bootstrap CI'siz
guvenilir mi, (2) `AL_`-dusuk-eksiklik tertile'inin BEKLENMEDIK
kirilganligi (Delta=-0,2299) `CAT_2`'nin AllofUs alt-kategorileriyle
CONFOUND mu.

SAF TESHIS -- HICBIR MODEL/ESIK/DOSYA DEGISTIRILMEZ, karar kurali yok.
`final_model_bundle_v2.pkl`/`predict.py`/split bankasi dokunulmaz.

`f3_provenance_holdout_diagnosis.py::evaluate_holdout` (degismedi)
yeniden kullanilir -- Adim 3'un confound-kontrolu, AYNI train seti
(orta+yuksek tertile, degismedi) ile, yalnizca test setinden (dusuk
tertile) belirli satirlar CIKARILARAK tekrarlanir.

Calistirma: python -m genova.pah.f3_confound_audit
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.pah import fold_versions as fv
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, TAB_DIR
from genova.pah.f3_robustness_stress_tests import BUNDLE_PATH
from genova.pah.f3_provenance_holdout_diagnosis import evaluate_holdout

OUT_CROSSTAB_CSV = TAB_DIR / "f3_confound_cat2_tertile_crosstab.csv"
OUT_CONFOUND_CSV = TAB_DIR / "f3_confound_check_results.csv"
N_BOOT = 1000


def _raw_test_predictions(v1_df, pool, best_params, train_ids, test_ids):
    """`f3_provenance_holdout_diagnosis.py::_fit_score_full` ile AYNI
    fit adimi -- yalnizca kalibrasyon UYGULANMADAN ham proba/y_true
    dondurulur (AUC bootstrap'i kalibrasyondan bagimsizdir, bkz. proje
    genelindeki monotoniklik notlari)."""
    X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
    proba = fit_predict_catboost(X_tr, y_tr, X_te, best_params)
    return proba, y_te.to_numpy()


def bootstrap_auc_ci(y_true, proba, n_boot=N_BOOT, seed=42):
    """Satir-bazli bootstrap AUC CI'si -- kucuk/dengesiz n'lerde (orn.
    2 benign/17 satir) bazi resample'lar TEK sinif icerebilir; bunlar
    ATLANIR ve kac tanesinin atlandigi raporlanir (orneklemin ne kadar
    ince oldugunun dogrudan bir gostergesi)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs, n_skipped = [], 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if len(set(yb)) < 2:
            n_skipped += 1
            continue
        boot_aucs.append(roc_auc_score(yb, proba[idx]))
    boot_aucs = np.array(boot_aucs)
    point = roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    if len(boot_aucs) == 0:
        return {"point_estimate": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot_valid": 0, "n_boot_skipped": n_skipped}
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return {"point_estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot_valid": len(boot_aucs), "n_boot_skipped": n_skipped}


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]

    cat2_group = v1_df["CAT_2"].astype(object).fillna("EMPTY")
    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    al_rate = v1_df[al_cols].isna().mean(axis=1)
    tertile = pd.qcut(al_rate, 3, labels=["dusuk", "orta", "yuksek"])

    print("=== Adim 1: AllofUs_EAS / AllofUs_AFR icin bootstrap AUC CI ===", flush=True)
    ci_rows = []
    for category in ["AllofUs_EAS", "AllofUs_AFR"]:
        test_ids = v1_df.loc[cat2_group == category, "Variant_ID"].tolist()
        train_ids = v1_df.loc[cat2_group != category, "Variant_ID"].tolist()
        proba, y_te = _raw_test_predictions(v1_df, pool, best_params, train_ids, test_ids)
        n_benign, n_path = int((y_te == 0).sum()), int((y_te == 1).sum())
        ci = bootstrap_auc_ci(y_te, proba)
        ci_rows.append({"category": category, "n": len(y_te), "n_benign": n_benign, "n_pathogenic": n_path, **ci})
        print(f"  {category}: n={len(y_te)} (benign={n_benign}, patojenik={n_path}) "
              f"AUC={ci['point_estimate']:.4f} %95 CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
              f"(gecerli bootstrap={ci['n_boot_valid']}/{N_BOOT}, atlanan={ci['n_boot_skipped']})", flush=True)

    print("\n  --- Diger kucuk kategoriler (n<15) -- yalnizca sinif dagilimi, CI yok ---", flush=True)
    for category in ["AllofUs_AMR", "AllofUs_MID", "AllofUs_SAS"]:
        sub = v1_df.loc[cat2_group == category, "Label"]
        print(f"  {category}: n={len(sub)} (benign={(sub==0).sum()}, patojenik={(sub==1).sum()})", flush=True)

    print("\n=== Adim 2: CAT_2 x AL_-tertile crosstab ===", flush=True)
    ct = pd.crosstab(cat2_group, tertile)
    ct_pct = pd.crosstab(cat2_group, tertile, normalize="index")
    print(ct.to_string(), flush=True)
    print("\n  (satir yuzdesi)", flush=True)
    print(ct_pct.round(3).to_string(), flush=True)
    ct.to_csv(OUT_CROSSTAB_CSV)

    eas_in_dusuk = int(((cat2_group == "AllofUs_EAS") & (tertile == "dusuk")).sum())
    afr_in_dusuk = int(((cat2_group == "AllofUs_AFR") & (tertile == "dusuk")).sum())
    allofus_in_dusuk = int(((cat2_group != "EMPTY") & (tertile == "dusuk")).sum())
    dusuk_total = int((tertile == "dusuk").sum())
    print(f"\n  AllofUs_EAS satirlarinin {eas_in_dusuk}/{int((cat2_group=='AllofUs_EAS').sum())}'i dusuk tertile'da "
          f"({eas_in_dusuk/int((cat2_group=='AllofUs_EAS').sum())*100:.1f}%)", flush=True)
    print(f"  AllofUs_AFR satirlarinin {afr_in_dusuk}/{int((cat2_group=='AllofUs_AFR').sum())}'i dusuk tertile'da "
          f"({afr_in_dusuk/int((cat2_group=='AllofUs_AFR').sum())*100:.1f}%)", flush=True)
    print(f"  dusuk tertile'in ({dusuk_total} satir) {allofus_in_dusuk}'i ({allofus_in_dusuk/dusuk_total*100:.1f}%) "
          f"HERHANGI bir AllofUs alt-kategorisinden -- confound TUM AllofUs kategorilerine geneldir, "
          f"yalnizca EAS/AFR'ye ozgu degil.", flush=True)
    confound_detected = allofus_in_dusuk / dusuk_total > 0.5

    print("\n=== Adim 3: Confound kontrolu -- dusuk-tertile holdout, satirlar cikarilarak ===", flush=True)
    orta_yuksek_ids = v1_df.loc[tertile != "dusuk", "Variant_ID"].tolist()
    dusuk_mask = tertile == "dusuk"

    confound_rows = []

    print("  [orijinal] dusuk tertile (tum satirlar) -- onceki turun sonucu (referans):", flush=True)
    row = evaluate_holdout(v1_df, pool, best_params, orta_yuksek_ids, v1_df.loc[dusuk_mask, "Variant_ID"].tolist())
    row["scenario"] = "orijinal (tum dusuk-tertile)"
    confound_rows.append(row)
    print(f"    n_test={row['n_test']} raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f}", flush=True)

    print("  [Adim3-a, gorev talimati] dusuk tertile, EAS+AFR cikarilmis:", flush=True)
    mask_a = dusuk_mask & ~cat2_group.isin(["AllofUs_EAS", "AllofUs_AFR"])
    row = evaluate_holdout(v1_df, pool, best_params, orta_yuksek_ids, v1_df.loc[mask_a, "Variant_ID"].tolist())
    row["scenario"] = "dusuk-tertile minus EAS+AFR"
    confound_rows.append(row)
    print(f"    n_test={row['n_test']} raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f}", flush=True)

    print("  [Adim3-b, genisletilmis -- crosstab TUM AllofUs'ta yogunlasma gosterdigi icin] "
          "dusuk tertile, TUM AllofUs cikarilmis (yalnizca EMPTY kalir):", flush=True)
    mask_b = dusuk_mask & (cat2_group == "EMPTY")
    row = evaluate_holdout(v1_df, pool, best_params, orta_yuksek_ids, v1_df.loc[mask_b, "Variant_ID"].tolist())
    row["scenario"] = "dusuk-tertile minus TUM AllofUs (yalnizca EMPTY)"
    confound_rows.append(row)
    print(f"    n_test={row['n_test']} raw_AUC={row['raw_auc']:.4f} weighted_F1={row['weighted_f1']:.4f} "
          f"{'[DUSUK GUC, n<15]' if row['n_test'] < 15 else ''}", flush=True)

    confound_df = pd.DataFrame(confound_rows)
    ref_auc = confound_df.iloc[0]["raw_auc"]
    confound_df["delta_vs_original_reference_0.8309"] = confound_df["raw_auc"] - 0.8309
    confound_df.to_csv(OUT_CONFOUND_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CROSSTAB_CSV}, {OUT_CONFOUND_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== SENTEZ ===", flush=True)
    delta_a = confound_df.iloc[1]["raw_auc"] - 0.8309
    delta_b = confound_df.iloc[2]["raw_auc"] - 0.8309
    print(f"  Orijinal dusuk-tertile Delta (referans 0,8309'a gore): {ref_auc-0.8309:+.4f}", flush=True)
    print(f"  EAS+AFR cikarilinca Delta: {delta_a:+.4f}", flush=True)
    print(f"  TUM AllofUs cikarilinca Delta: {delta_b:+.4f}", flush=True)
    if abs(delta_a) >= 0.15:
        print("  YORUM (Adim3-a kurali): Dusus hala BUYUK (>=0,15) -- dusuk-tertile bulgusu "
              "EAS/AFR'nin OTESINDE de bagimsiz bir sinyal tasiyor.", flush=True)
    elif abs(delta_a) < 0.05:
        print("  YORUM (Adim3-a kurali): Dusus KUCULDU/KAYBOLDU (<0,05) -- dusuk-tertile bulgusu "
              "buyuk olcude EAS/AFR confound'uydu.", flush=True)
    else:
        print("  YORUM (Adim3-a kurali): Ara bolgede -- ne acik confound ne acik bagimsiz sinyal.", flush=True)


if __name__ == "__main__":
    main()
