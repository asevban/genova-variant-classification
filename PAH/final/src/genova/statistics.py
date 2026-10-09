"""P1 madde 10: örnek-düzeyi (fold-bağımsız) belirsizlik ölçümü için üç
genel-amaçlı istatistik yardımcısı.

Bu modül herhangi bir panele/modele özel değildir -- yalnızca sayısal
girdiler (etiket, tahmin/olasılık, skor dizileri) alır. Fold-düzeyi
bootstrap CI'nin (50 dış fold'u bağımsızmış gibi ele alan) neden dar
gösterdiğini düzeltmek için `sample_level_bootstrap_ci`, aynı fold'larda
ölçülen iki modeli karşılaştırırken bağımlılığı hesaba katmak için
`nadeau_bengio_corrected_ttest`, ve şartnamenin gerçek final test
kompozisyonunu (100 patojenik/250 benign) taklit etmek için
`monte_carlo_final_f1_simulation` sağlanır.
"""
import numpy as np
from scipy import stats

from genova.metrics import f1_binary_positive


def sample_level_bootstrap_ci(variant_ids, y_true, y_pred_or_proba, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    """369 satırın (ya da herhangi bir OOF vektörünün) SATIR bazında
    (fold bazında DEĞİL) bootstrap yeniden örneklemesiyle `metric_fn`'in
    %(1-alpha) güven aralığını döner -- fold-düzeyi CI'nin (yalnızca 50
    dış-fold'u yeniden örnekleyen) yaydığı yanlış bağımsızlık varsayımını
    (369 satırın gerçek örneklem birimleri olduğunu) düzeltir.

    `metric_fn(y_true, y_pred_or_proba) -> float` imzasında olmalı (ör.
    `genova.metrics.f1_binary_positive` -- `y_pred_or_proba` zaten
    0/1'e eşiklenmiş tahmin olmalı, ya da ROC-AUC gibi eşiksiz bir
    metrik icin ham olasilik).

    >>> ids = list(range(4))
    >>> y = [1, 1, 0, 0]
    >>> pred = [1, 1, 0, 0]
    >>> result = sample_level_bootstrap_ci(ids, y, pred, lambda yt, yp: (yt == yp).mean(), n_boot=100)
    >>> result["point_estimate"]
    1.0
    >>> result["ci_low"], result["ci_high"]
    (1.0, 1.0)
    """
    variant_ids = list(variant_ids)
    y_true = np.asarray(y_true)
    y_pred_or_proba = np.asarray(y_pred_or_proba)
    if not (len(variant_ids) == len(y_true) == len(y_pred_or_proba)):
        raise ValueError("variant_ids/y_true/y_pred_or_proba uzunluklari uyusmuyor")

    n = len(y_true)
    rng = np.random.RandomState(seed)
    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_scores[b] = metric_fn(y_true[idx], y_pred_or_proba[idx])

    point_estimate = metric_fn(y_true, y_pred_or_proba)
    ci_low, ci_high = np.percentile(boot_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "ci_width": float(ci_high - ci_low),
        "n_boot": n_boot, "n_samples": n,
        "boot_scores": boot_scores,
    }


def nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau & Bengio (2003) varyans-duzeltmeli paired t-testi -- k adet
    dis-fold'un BAGIMSIZ olmadigini (ayni 369 satirin farkli bolmelerini
    tekrar tekrar kullandiklarini) hesaba katar. Standart paired t-testin
    varyansina `n_test/n_train` terimi eklenir -- bu, standart testin
    (`1/k` varyans terimi) SISTEMATIK OLARAK dar/iyimser oldugunu, bu
    fonksiyonun her zaman standart teste esit ya da daha BUYUK bir
    p-degeri uretecegini garanti eder.

    `n_train`/`n_test`: skaler (tum fold'lar ayni boyuttaysa) ya da
    `scores_a` ile ayni uzunlukta dizi (fold-bazli boyut farkliliklarinda
    -- oran `n_test/n_train`'in ortalamasi alinir, formulun kendisi tek
    bir sabit oran varsayar).

    >>> a = [0.90, 0.88, 0.92, 0.89, 0.91]
    >>> b = [0.85, 0.86, 0.84, 0.87, 0.83]
    >>> result = nadeau_bengio_corrected_ttest(a, b, n_train=300, n_test=75)
    >>> result["mean_diff"] > 0
    True
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a/scores_b uzunluklari uyusmuyor")
    k = len(scores_a)
    if k < 2:
        raise ValueError("en az 2 fold/karsilastirma gerekli")

    diff = scores_a - scores_b
    mean_diff = float(diff.mean())
    var_diff = float(diff.var(ddof=1))

    ratio = float(np.mean(np.asarray(n_test, dtype=float) / np.asarray(n_train, dtype=float)))
    corrected_var = (1.0 / k + ratio) * var_diff
    if corrected_var <= 0:
        t_stat = 0.0 if mean_diff == 0 else np.sign(mean_diff) * np.inf
    else:
        t_stat = mean_diff / np.sqrt(corrected_var)
    df = k - 1
    p_value = float(2 * stats.t.sf(np.abs(t_stat), df)) if np.isfinite(t_stat) else 0.0

    return {
        "mean_diff": mean_diff, "var_diff": var_diff, "ratio_n_test_n_train": ratio,
        "t_stat": float(t_stat), "df": df, "p_value": p_value, "k": k,
    }


def monte_carlo_final_f1_simulation(variant_ids, y_true, proba, n_pathogenic=100, n_benign=250,
                                     n_simulations=2000, threshold=None, seed=42):
    """OOF olasiliklarindan, sartnamenin gercek final test kompozisyonunu
    (varsayilan 100 patojenik/250 benign) taklit eden alt-orneklemler
    ceker -- patojenik etiketli satirlardan `n_pathogenic`, benign
    etiketlilerden `n_benign` tanesi, HER ZAMAN YERINE KOYARAK (elimizde
    yalnizca 369 satir var, ozellikle 61 benign'den 250 cekmek yerine
    koymadan imkansiz). Her simulasyonda F1'i `threshold`'da hesaplar,
    dagilimi doner.

    >>> ids = list(range(6))
    >>> y = [1, 1, 1, 0, 0, 0]
    >>> proba = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    >>> result = monte_carlo_final_f1_simulation(ids, y, proba, n_pathogenic=2, n_benign=2, n_simulations=50, threshold=0.5)
    >>> result["mean"]
    1.0
    """
    if threshold is None:
        raise ValueError("threshold acikca verilmeli (F1 icin karar esigi gerekli)")
    variant_ids = list(variant_ids)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if not (len(variant_ids) == len(y_true) == len(proba)):
        raise ValueError("variant_ids/y_true/proba uzunluklari uyusmuyor")

    pathogenic_idx = np.flatnonzero(y_true == 1)
    benign_idx = np.flatnonzero(y_true == 0)
    if len(pathogenic_idx) == 0 or len(benign_idx) == 0:
        raise ValueError("hem patojenik hem benign etiketli en az bir satir gerekli")

    rng = np.random.RandomState(seed)
    f1_scores = np.empty(n_simulations)
    for i in range(n_simulations):
        sim_idx = np.concatenate([
            rng.choice(pathogenic_idx, size=n_pathogenic, replace=True),
            rng.choice(benign_idx, size=n_benign, replace=True),
        ])
        y_sim = y_true[sim_idx]
        pred_sim = (proba[sim_idx] >= threshold).astype(int)
        f1_scores[i] = f1_binary_positive(y_sim, pred_sim)

    return {
        "f1_scores": f1_scores,
        "mean": float(f1_scores.mean()), "median": float(np.median(f1_scores)),
        "ci_low": float(np.percentile(f1_scores, 2.5)), "ci_high": float(np.percentile(f1_scores, 97.5)),
        "n_simulations": n_simulations, "n_pathogenic": n_pathogenic, "n_benign": n_benign,
        "threshold": threshold,
    }
