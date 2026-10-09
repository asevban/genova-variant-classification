"""Nested oy-esigi (vote-threshold) aramasi.

21-agaclik bir ensemble icin, oy-esigini (11/21 .. 19/21 arasi) YALNIZCA ic
(inner) fold'larda arar -- dis (outer) test fold'u bu aramaya HICBIR sekilde
girmez (bkz. gorev metninin "kural 2"si). `run_experiment.py`:
  1. Her dis fold'un 4 ic fold'unda ayri birer ensemble egitir (ic-egitim),
     ic-val'de her esik icin F1 hesaplar (bu modul).
  2. 4 ic fold'un F1'lerini esik-basina ortalar, argmax esigi secer.
  3. Dis-egitimin TAMAMinda YENI bir ensemble egitir, dis-test'i YALNIZCA
     bu sabit esikle, tek seferde skorlar.

Bu modul hicbir dosyaya yazmaz -- yalnizca F1 hesaplayan saf fonksiyonlardir.
"""
import numpy as np
from sklearn.metrics import f1_score

THRESHOLD_GRID = list(range(11, 20))  # 11/21 .. 19/21 (dahil)


def f1_by_threshold(vote_counts, y_true):
    """Her esik icin F1 (patojenik=1 sinifi) doner: {esik: f1}."""
    y_true = np.asarray(y_true)
    return {t: f1_score(y_true, (vote_counts >= t).astype(int), pos_label=1, zero_division=0)
            for t in THRESHOLD_GRID}


def best_threshold_from_inner_folds(per_inner_fold_f1_dicts):
    """4 ic fold'un {esik: f1} sozluklerini esik-basina ortalar, en yuksek
    ortalama F1'e sahip esigi (ve tam esik->ortalama-F1 tablosunu) dondurur.
    """
    mean_f1_by_threshold = {}
    for t in THRESHOLD_GRID:
        values = [d[t] for d in per_inner_fold_f1_dicts]
        mean_f1_by_threshold[t] = float(np.mean(values))
    best_t = max(mean_f1_by_threshold, key=mean_f1_by_threshold.get)
    return best_t, mean_f1_by_threshold
