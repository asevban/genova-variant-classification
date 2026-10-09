"""Asama E4: kapali-form Bayes onsel duzeltmesi (P1 madde 13 -- bu, GERCEK
SLD/Saerens-Latinne-Decaestecker DEGILDIR: SLD, hedef onsel BILINMEDIGINDE
EM ile iteratif tahmin eder; burada hedef onsel (%28,6, sartnameden)
BILINIYOR, bu yuzden kullanilan yontem SLD'nin basitlestirilmis kapali-form
Bayes/Elkan-Noto tarzi karsiligidir -- kod tabaninda ve raporlarda "SLD"
adi tarihsel tutarlilik icin hala geciyor olabilir, ama dogru terim budur).
Egitim onseli (%83,3 patojenik) -> final onseli (%28,6 patojenik) icin,
E3'un zaten hesapladigi Beta-kalibre DIS-TEST olasiliklarina
(`e3_calibrated_oof_predictions.csv`, method="beta") uygulanir. Model
YENIDEN EGITILMEZ -- yalnizca olasilik olcegi donusturulur, siralama/AUC
degismez.

    p_yeni = w1*p / (w1*p + w0*(1-p))
    w1 = FINAL_PATHOGENIC_PRIOR / TRAIN_PATHOGENIC_PRIOR (asagidaki
         sabitlere bak -- onceki bir surumde bu satir ters yazilmisti,
         kodun kendisi hep dogruydu, yalnizca bu yorum satiri duzeltildi)

Calistirma: python -m genova.pah.e4_prior_correction
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
E3_OOF_PATH = ROOT / "reports" / "tables" / "e3_calibrated_oof_predictions.csv"
OUT_PATH = ROOT / "reports" / "tables" / "e4_prior_corrected_probabilities.csv"

FINAL_PATHOGENIC_PRIOR = 0.286
TRAIN_PATHOGENIC_PRIOR = 0.833
W1 = FINAL_PATHOGENIC_PRIOR / TRAIN_PATHOGENIC_PRIOR
W0 = (1 - FINAL_PATHOGENIC_PRIOR) / (1 - TRAIN_PATHOGENIC_PRIOR)


def sld_correct(proba, w1=W1, w0=W0):
    """Kapali-form Bayes onsel duzeltmesi (SLD DEGIL -- bkz. modul
    docstring'i), vektorize. Fonksiyon adi ("sld_correct") kod tabaninda
    zaten cok sayida dosyada kullanildigi icin (e5/e6/e6ek/f0_final_model)
    bu turda DEGISTIRILMEDI -- yalnizca terminoloji/dokumantasyon
    duzeltmesi, formul/sabitler/cagiran taraflar etkilenmedi."""
    proba = np.asarray(proba, dtype=float)
    return (w1 * proba) / (w1 * proba + w0 * (1 - proba))


def main():
    oof = pd.read_csv(E3_OOF_PATH)
    beta = oof[oof.method == "beta"].copy()
    before_mean = beta.groupby(["model", "data_version"])["proba"].mean()
    beta["proba"] = sld_correct(beta["proba"])
    beta["method"] = "beta_sld"
    after_mean = beta.groupby(["model", "data_version"])["proba"].mean()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    beta.to_csv(OUT_PATH, index=False)
    print(f"w1={W1:.6f} w0={W0:.6f}")
    print(f"kaydedildi: {OUT_PATH} ({len(beta)} satir)")
    for key in before_mean.index:
        print(f"{key}: ort. proba SLD-oncesi={before_mean[key]:.4f} -> SLD-sonrasi={after_mean[key]:.4f}")


if __name__ == "__main__":
    main()
