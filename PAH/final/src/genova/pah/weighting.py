"""Asama E1-EK: yeniden kullanilabilir ornek-agirliklandirma stratejileri.
Ucu de yalnizca `y_train` (bir dis/ic fold'un EGITIM etiketleri) gorur,
hicbir zaman test/val etiketlerine erismez -- fold-ici cagrildiginda
sizintisiz. Donen `sample_weight` dizisi, herhangi bir sklearn-uyumlu
`.fit(X, y, sample_weight=...)` cagrisina (XGBoost/LightGBM/CatBoost'un
sklearn API'si, sklearn LogisticRegression) dogrudan verilebilir -- E2'nin
tum model ailelerinde ayni arayuzle kullanilmasi icin boyle tasarlandi.

Uc strateji:
  A - agirliksiz: tum ornekler esit (1.0).
  B - sabit azinlik-sinif carpani: PDR/E1'in orijinal politikasi (azinlik
      sinifina x2.0). Burada yalnizca arayuz tutarliligi icin tanimli --
      E1'in kendi sonucu zaten `models/pah/baseline_frozen.pkl`'de mevcut,
      bu fonksiyon E1-EK deneyinde YENIDEN calistirilmiyor.
  C - veri-gudumlu scale_pos_weight: SPW = N_negatif_train / N_pozitif_train,
      literatur onerisi (Berra dort-panel raporu, modelleme_literatur_1.md,
      PAH_Literatur_ve_Modelleme_Asamasi.md) -- her fold kendi train
      kisminda yeniden hesaplanir.
"""
import numpy as np


def weights_no_weight(y_train):
    """A -- agirliksiz: her ornek 1.0."""
    return np.ones(len(y_train))


def weights_fixed_minority(y_train, minority_weight=2.0):
    """B -- azinlik sinifina sabit carpan, cogunluga 1.0."""
    y_train = np.asarray(y_train)
    values, counts = np.unique(y_train, return_counts=True)
    minority_class = values[np.argmin(counts)]
    return np.where(y_train == minority_class, minority_weight, 1.0)


def compute_data_driven_spw(y_train, positive_label=1):
    """C'nin SPW skaleri: N_negatif_train / N_pozitif_train, yalnizca
    train fold'undan. PAH'ta pozitif (Label=1) sinif zaten cogunlukta
    oldugu icin SPW<1 beklenir (~0.2 civari) -- bu bir hata degil,
    pozitif sinifi goreceli olarak asagi cekmenin dogru yonu.
    """
    y_train = np.asarray(y_train)
    n_pos = int(np.sum(y_train == positive_label))
    n_neg = int(np.sum(y_train != positive_label))
    if n_pos == 0:
        raise ValueError("train fold'unda pozitif sinif ornegi yok, SPW tanimsiz")
    return n_neg / n_pos


def weights_data_driven_spw(y_train, positive_label=1):
    """C -- veri-gudumlu scale_pos_weight, sample_weight karsiligi:
    pozitif sinif satirlarina SPW, negatif sinif satirlarina 1.0.
    XGBoost'un native `scale_pos_weight=SPW` parametresiyle matematiksel
    olarak esdegerdir (ikisi de pozitif-sinif gradyan/hessian katkisini
    ayni oranda olceklendirir).
    """
    spw = compute_data_driven_spw(y_train, positive_label)
    y_train = np.asarray(y_train)
    return np.where(y_train == positive_label, spw, 1.0)


WEIGHTING_STRATEGIES = {
    "A_no_weight": weights_no_weight,
    "B_fixed_minority_2x": weights_fixed_minority,
    "C_data_driven_spw": weights_data_driven_spw,
}
