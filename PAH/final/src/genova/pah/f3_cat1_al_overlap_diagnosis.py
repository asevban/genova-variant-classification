"""SAF TESHIS: `CAT_1` capraz bulgusu (Delta=-0,2582, n=237) ile `AL_`-
dusuk-eksiklik/AllofUs-bagimsiz bulgusu (confound audit'ten, Delta=
-0,2476, n=19) AYNI mekanizmanin iki gorunumu mu, yoksa iki YARI-
BAGIMSIZ mekanizma mi?

HICBIR MODEL/ESIK/DOSYA DEGISTIRILMEZ, karar kurali yok. `final_model_
bundle_v2.pkl`/`predict.py`/split bankasi dokunulmaz.

Adim 1 (crosstab) SAF PANDAS -- hicbir model egitimi yok.
Adim 2, referans modelin (bundle'in kendi `cross_fit_oof`u, TAM veriyle
"egitilmis" -- degismedi, TEK CAGRI, 5 fit) OOF tahminlerini 2x2
gruplara BOLEREK degerlendirir -- YENIDEN FIT/holdout YOK, yalnizca
MEVCUT capraz-dogrulanmis tahminlerin alt-kume analizi.

Calistirma: python -m genova.pah.f3_cat1_al_overlap_diagnosis
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import f1_binary_positive_weighted, specificity, sensitivity
from genova.pah.f0_final_model import V1_PATH, TAB_DIR, cross_fit_oof
from genova.pah.f3_robustness_stress_tests import BUNDLE_PATH
from genova.pah.e4_prior_correction import sld_correct, FINAL_PATHOGENIC_PRIOR
from genova.pah.e5_threshold_selection import _prior_weights

OUT_CROSSTAB_CSV = TAB_DIR / "f3_cat1_al_tertile_crosstab.csv"
OUT_CELLS_CSV = TAB_DIR / "f3_cat1_al_2x2_cells.csv"
LOW_POWER_N = 15


def _cell_metrics(y_true, proba, pred):
    row = {"n": len(y_true), "n_benign": int((y_true == 0).sum()), "n_pathogenic": int((y_true == 1).sum())}
    row["low_power"] = row["n"] < LOW_POWER_N or min(row["n_benign"], row["n_pathogenic"]) < 3
    if len(set(y_true)) > 1:
        row["raw_auc"] = roc_auc_score(y_true, proba)
    else:
        row["raw_auc"] = float("nan")
    weights = _prior_weights(y_true, FINAL_PATHOGENIC_PRIOR)
    row["weighted_f1"] = f1_binary_positive_weighted(y_true, pred, weights)
    row["specificity"] = specificity(y_true, pred) if row["n_benign"] > 0 else float("nan")
    row["sensitivity"] = sensitivity(y_true, pred) if row["n_pathogenic"] > 0 else float("nan")
    return row


def main():
    t_start = time.time()
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    bundle = joblib.load(BUNDLE_PATH)
    pool, best_params = bundle["pool"], bundle["model_best_params"]

    al_cols = [c for c in v1_df.columns if c.startswith("AL_")]
    al_rate = v1_df[al_cols].isna().mean(axis=1)
    tertile = pd.Series(pd.qcut(al_rate, 3, labels=["dusuk", "orta", "yuksek"]), index=v1_df.index)
    cat1_status = v1_df["CAT_1"].notna().map({True: "dolu", False: "bos"})

    print("=== Adim 1: CAT_1 x AL_-tertile crosstab ===", flush=True)
    ct = pd.crosstab(cat1_status, tertile)
    ct_pct = pd.crosstab(cat1_status, tertile, normalize="index")
    print(ct.to_string(), flush=True)
    print("\n  (satir yuzdesi)", flush=True)
    print(ct_pct.round(3).to_string(), flush=True)
    ct.to_csv(OUT_CROSSTAB_CSV)

    n_bos_dusuk = int(((cat1_status == "bos") & (tertile == "dusuk")).sum())
    n_bos_total = int((cat1_status == "bos").sum())
    print(f"\n  CAT_1-bos satirlarin AL_-dusuk tertile'da payi: {n_bos_dusuk}/{n_bos_total} "
          f"({n_bos_dusuk/n_bos_total*100:.1f}%)", flush=True)

    cat2_group = v1_df["CAT_2"].astype(object).fillna("EMPTY")
    empty_dusuk_mask = (tertile == "dusuk") & (cat2_group == "EMPTY")
    n_empty_dusuk = int(empty_dusuk_mask.sum())
    n_empty_dusuk_bos = int((empty_dusuk_mask & (cat1_status == "bos")).sum())
    print(f"  Onceki confound-audit'in 'EMPTY-only-dusuk-tertile' grubu (n={n_empty_dusuk}): "
          f"bunlarin {n_empty_dusuk_bos}/{n_empty_dusuk}'i CAT_1-bos "
          f"({'HEPSI CAT_1-dolu -- bu grup CAT_1-bos bulgusuyla ORTUSMUYOR' if n_empty_dusuk_bos == 0 else ''})", flush=True)

    if n_bos_dusuk == 0:
        print("\n  YAPISAL BULGU: CAT_1-bos satirlarin HICBIRI AL_-dusuk-eksiklik tertile'inda DEGIL "
              "(0 satir) -- bu iki grup ALMOST-PERFECTLY AYRIK, temiz bir 2x2 tasarim bu hucreyle KURULAMAZ.", flush=True)

    print("\n=== Adim 2: Referans modelin OOF'u -- 2x2 (+ orta-tertile ek karsilastirma) hucre analizi ===", flush=True)
    print("  (yeniden fit YOK -- bundle'in kendi cross_fit_oof'u TEK KEZ calistirilip alt-kumelere bolunuyor)", flush=True)
    oof_proba, y_full = cross_fit_oof(v1_df, pool, best_params, all_ids)
    calibrated = bundle["calibrator"].transform(oof_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= bundle["threshold"]).astype(int)

    ordered_ids = v1_df["Variant_ID"].tolist()
    id_to_idx = {vid: i for i, vid in enumerate(ordered_ids)}
    cat1_by_id = cat1_status.reindex(v1_df.index)
    cat1_by_id.index = v1_df["Variant_ID"]
    tertile_by_id = tertile.reindex(v1_df.index)
    tertile_by_id.index = v1_df["Variant_ID"]

    cells = []
    for c1 in ["dolu", "bos"]:
        for al in ["dusuk", "orta", "yuksek"]:
            vids = v1_df.loc[(cat1_status.to_numpy() == c1) & (tertile.to_numpy() == al), "Variant_ID"].tolist()
            if len(vids) == 0:
                cells.append({"CAT_1": c1, "AL_tertile": al, "n": 0, "n_benign": 0, "n_pathogenic": 0,
                              "low_power": True, "raw_auc": float("nan"), "weighted_f1": float("nan"),
                              "specificity": float("nan"), "sensitivity": float("nan")})
                print(f"  CAT_1={c1} AL_={al}: n=0 (BOS HUCRE)", flush=True)
                continue
            idx = [id_to_idx[v] for v in vids]
            metrics = _cell_metrics(y_full[idx], oof_proba[idx], pred[idx])
            metrics["CAT_1"], metrics["AL_tertile"] = c1, al
            cells.append(metrics)
            power_note = " [DUSUK GUC]" if metrics["low_power"] else ""
            print(f"  CAT_1={c1} AL_={al}: n={metrics['n']} (benign={metrics['n_benign']}, "
                  f"patojenik={metrics['n_pathogenic']}) raw_AUC={metrics['raw_auc']:.4f} "
                  f"weighted_F1={metrics['weighted_f1']:.4f} spec={metrics['specificity']:.4f} "
                  f"sens={metrics['sensitivity']:.4f}{power_note}", flush=True)

    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(OUT_CELLS_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CROSSTAB_CSV}, {OUT_CELLS_CSV}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Toplam hesaplama suresi: {elapsed:.0f}s ({elapsed/60:.1f} dk) ===", flush=True)

    print("\n=== SENTEZ ===", flush=True)
    dolu_dusuk = cells_df[(cells_df.CAT_1 == "dolu") & (cells_df.AL_tertile == "dusuk")].iloc[0]
    dolu_orta = cells_df[(cells_df.CAT_1 == "dolu") & (cells_df.AL_tertile == "orta")].iloc[0]
    dolu_yuksek = cells_df[(cells_df.CAT_1 == "dolu") & (cells_df.AL_tertile == "yuksek")].iloc[0]
    bos_orta = cells_df[(cells_df.CAT_1 == "bos") & (cells_df.AL_tertile == "orta")].iloc[0]
    bos_yuksek = cells_df[(cells_df.CAT_1 == "bos") & (cells_df.AL_tertile == "yuksek")].iloc[0]

    print(f"  CAT_1 SABIT (dolu), AL_ degisirken (dusuk->yuksek): "
          f"AUC {dolu_dusuk['raw_auc']:.4f} -> {dolu_yuksek['raw_auc']:.4f} "
          f"(delta={dolu_yuksek['raw_auc']-dolu_dusuk['raw_auc']:+.4f}, yuksek hucre n={dolu_yuksek['n']} DUSUK GUC)", flush=True)
    print(f"  AL_ SABIT (orta), CAT_1 degisirken (dolu->bos): "
          f"AUC {dolu_orta['raw_auc']:.4f} -> {bos_orta['raw_auc']:.4f} "
          f"(delta={bos_orta['raw_auc']-dolu_orta['raw_auc']:+.4f}, n={dolu_orta['n']}/{bos_orta['n']})", flush=True)
    print(f"  AL_ SABIT (yuksek), CAT_1 degisirken (dolu->bos): "
          f"AUC {dolu_yuksek['raw_auc']:.4f} -> {bos_yuksek['raw_auc']:.4f} "
          f"(delta={bos_yuksek['raw_auc']-dolu_yuksek['raw_auc']:+.4f}, dolu hucre n={dolu_yuksek['n']} DUSUK GUC)", flush=True)


if __name__ == "__main__":
    main()
