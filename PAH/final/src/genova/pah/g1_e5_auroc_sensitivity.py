"""Revize turu, Adim 2: E5'in 4 final adayina (catboost/v1, catboost/
v4_from_v2, lightgbm/v1, random_forest/v4_from_v2) AUROC + sensitivity
ekler -- HICBIR MODEL YENIDEN FIT EDILMIYOR, hatta "replay" bile degil,
saf aritmetik:

- AUROC: `e3_calibrated_oof_predictions.csv`'nin method="raw" satirlari
  (E3'un HAM, kalibrasyonsuz dis-test olasiliklari, zaten diskte) her
  (model, data_version, repeat, outer_fold) icin okunup roc_auc_score
  hesaplaniyor. AUROC esik-bagimsizdir VE Beta kalibrasyonu + SLD onsel
  duzeltmesi ikisi de olasiligin MONOTONIK (siralama-koruyan) donusumleri
  oldugu icin, bu sayi E4/E5'in kalibre+SLD-duzeltilmis olasiliklarindan
  hesaplansaydi da BIREBIR AYNI cikardi -- hangi asamadaki olasiliktan
  hesaplandigi onemli degil, sadece dogrulama amacli ham (en az islenmis)
  olan tercih edildi.
- sensitivity_chosen: `e4_prior_corrected_probabilities.csv`'nin
  (SLD-duzeltilmis dis-test olasiliklari) `e5_threshold_selection.csv`'de
  ZATEN KAYITLI `chosen_threshold`'u ile esiklenerek hesaplaniyor --
  f1_chosen/specificity_chosen'in ayni sekilde hesaplanmis olmasiyla
  birebir tutarli, yalnizca TPR eklenmis oluyor.

`e5_threshold_selection.csv`'ye YALNIZCA `auroc`/`sensitivity_chosen`
kolonlari eklenir, mevcut hicbir kolon degistirilmez (testle dogrulanir).

Calistirma: python -m genova.pah.g1_e5_auroc_sensitivity
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from genova.metrics import sensitivity

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
E3_CSV = TAB_DIR / "e3_calibrated_oof_predictions.csv"
E4_CSV = TAB_DIR / "e4_prior_corrected_probabilities.csv"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"


def main():
    e3 = pd.read_csv(E3_CSV)
    e4 = pd.read_csv(E4_CSV)
    e5 = pd.read_csv(E5_CSV)
    original_snapshot = e5.copy()

    e3_raw = e3[e3.method == "raw"]

    auroc_col, sens_col = [], []
    for _, row in e5.iterrows():
        model_name, version_name = row["model"], row["data_version"]
        repeat_idx, outer_fold_idx = int(row["repeat"]), int(row["outer_fold"])
        threshold = row["chosen_threshold"]

        fold_raw = e3_raw[(e3_raw.model == model_name) & (e3_raw.data_version == version_name)
                           & (e3_raw.repeat == repeat_idx) & (e3_raw.outer_fold == outer_fold_idx)]
        auroc_col.append(roc_auc_score(fold_raw["y_true"], fold_raw["proba"]))

        fold_sld = e4[(e4.model == model_name) & (e4.data_version == version_name)
                       & (e4.repeat == repeat_idx) & (e4.outer_fold == outer_fold_idx)]
        pred = (fold_sld["proba"] >= threshold).astype(int)
        sens_col.append(sensitivity(fold_sld["y_true"], pred))

    e5["auroc"] = auroc_col
    e5["sensitivity_chosen"] = sens_col

    unchanged = e5[original_snapshot.columns]
    pd.testing.assert_frame_equal(unchanged, original_snapshot)
    print("GUVENLIK KONTROLU GECTI: e5_threshold_selection.csv'nin mevcut kolonlari degismedi.")

    print(e5.groupby(["model", "data_version"])[["auroc", "sensitivity_chosen"]].agg(["mean", "std"]))
    e5.to_csv(E5_CSV, index=False)
    print(f"kaydedildi: {E5_CSV}")


if __name__ == "__main__":
    main()
