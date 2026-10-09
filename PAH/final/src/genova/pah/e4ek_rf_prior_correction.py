"""Asama E4-EK: RF/v4_from_v2 (Strateji B) icin SLD onsel duzeltmesi.
Mevcut `e4_prior_corrected_probabilities.csv`'deki CatBoost/v4_from_v2 ve
LightGBM/v1 satirlari DEGISTIRILMEZ -- yalnizca RF/v4_from_v2'nin Beta-
kalibre dis-test olasiliklari (E3-EK'in `e3_calibrated_oof_predictions.
csv`'ye eklendigi satirlar) okunup ayni `sld_correct` formuluyle
donusturulur ve EKLENIR. Model yeniden egitilmez.

Calistirma: python -m genova.pah.e4ek_rf_prior_correction
"""
from pathlib import Path

import pandas as pd

from genova.pah.e4_prior_correction import sld_correct, W1, W0
from genova.pah.idempotent_io import upsert_csv

ROOT = Path(__file__).resolve().parents[3]
E3_OOF_PATH = ROOT / "reports" / "tables" / "e3_calibrated_oof_predictions.csv"
OUT_PATH = ROOT / "reports" / "tables" / "e4_prior_corrected_probabilities.csv"


def main():
    oof = pd.read_csv(E3_OOF_PATH)
    rf_beta = oof[(oof.model == "random_forest") & (oof.data_version == "v4_from_v2") & (oof.method == "beta")].copy()
    assert len(rf_beta) > 0, "e3_calibrated_oof_predictions.csv'de random_forest/v4_from_v2/beta satiri bulunamadi"

    before_mean = rf_beta["proba"].mean()
    rf_beta["proba"] = sld_correct(rf_beta["proba"])
    rf_beta["method"] = "beta_sld"
    after_mean = rf_beta["proba"].mean()

    # P2 madde 19: idempotent yazar -- ayni anahtar icin script iki kez
    # calistirilirsa satir COGALMAZ.
    upsert_csv(rf_beta, OUT_PATH, ["model", "data_version", "method", "repeat", "outer_fold", "variant_id"])

    print(f"w1={W1:.6f} w0={W0:.6f}")
    print(f"random_forest/v4_from_v2: ort. proba SLD-oncesi={before_mean:.4f} -> SLD-sonrasi={after_mean:.4f}")
    print(f"eklendi: {OUT_PATH} ({len(rf_beta)} satir)")


if __name__ == "__main__":
    main()
