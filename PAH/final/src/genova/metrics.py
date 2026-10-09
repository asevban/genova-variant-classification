"""Resmi TEKNOFEST 2026 sıralama metriği: pozitif sınıf (Label=1) üzerinden
ikili F1, TP/FP/FN'den doğrudan hesaplanır. MCC ikincil/izleme metriğidir
(şartname Bölüm 7.3) -- sıralama kararı vermez.

Hiçbir yerde sklearn'ün varsayılan `average` parametresine güvenilmez;
`f1_binary_positive` her zaman TP/FP/FN sayımından hesaplar.
"""
import numpy as np


def _validate_and_coerce(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.dtype.kind in "OUS" or y_pred.dtype.kind in "OUS":
        raise TypeError("Label degerleri string/int karisik olamaz: yalnizca {0, 1} int/bool bekleniyor")
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if not set(np.unique(y_true)) <= {0, 1} or not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("Label degerleri {0, 1} disinda deger iceriyor")
    return y_true, y_pred


def f1_binary_positive(y_true, y_pred):
    """Pozitif sinif (Label=1) icin ikili F1 = 2*TP / (2*TP + FP + FN).

    >>> f1_binary_positive([1, 1, 0, 0], [1, 1, 0, 0])
    1.0
    >>> f1_binary_positive([1, 1, 0, 0], [0, 0, 0, 0])
    0.0
    """
    y_true, y_pred = _validate_and_coerce(y_true, y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def f1_binary_positive_weighted(y_true, y_pred, sample_weight):
    """Ornek-agirlikli pozitif-sinif F1 -- E5'in onsel-agirlikli esik
    aramasi icin: TP/FP/FN sayimlari, sample_weight ile agirliklandirilarak
    toplanir (`w1`/`w0` importance-agirliklariyla, egitim orneklemini final
    onsel dagilimini yansitacak sekilde yeniden tartmak icin). sample_
    weight=1 iken `f1_binary_positive` ile birebir ayni sonucu verir.

    >>> f1_binary_positive_weighted([1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1])
    1.0
    """
    y_true, y_pred = _validate_and_coerce(y_true, y_pred)
    w = np.asarray(sample_weight, dtype=float)
    tp = float(np.sum(w[(y_true == 1) & (y_pred == 1)]))
    fp = float(np.sum(w[(y_true == 0) & (y_pred == 1)]))
    fn = float(np.sum(w[(y_true == 1) & (y_pred == 0)]))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def matthews_correlation_coefficient(y_true, y_pred):
    """MCC -- ikincil/izleme metriği (şartname Bölüm 7.3), sıralama kararı
    vermez. Payda sıfırsa (dejenere karışıklık matrisi) 0.0 döner.

    >>> matthews_correlation_coefficient([1, 1, 0, 0], [1, 1, 0, 0])
    1.0
    """
    y_true, y_pred = _validate_and_coerce(y_true, y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom


def specificity(y_true, y_pred):
    """Benign (Label=0) uzerinde dogru negatif orani = TN / (TN + FP).
    Payda sifirsa (fold'da hic benign yoksa) 0.0 doner.
    """
    y_true, y_pred = _validate_and_coerce(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def sensitivity(y_true, y_pred):
    """Patojenik (Label=1) uzerinde dogru pozitif orani (recall/TPR) =
    TP / (TP + FN). Payda sifirsa (fold'da hic patojenik yoksa) 0.0 doner.

    >>> sensitivity([1, 1, 0, 0], [1, 0, 0, 0])
    0.5
    """
    y_true, y_pred = _validate_and_coerce(y_true, y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)
