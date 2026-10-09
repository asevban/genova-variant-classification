"""Faz 3'un tum adim script'leri icin ortak yukleme/yardimci fonksiyonlar.
Bu modul hicbir dosyaya YAZMAZ -- yalnizca okuma ve hesaplama fonksiyonlari.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
PHASE1_SCRIPTS = EXPERIMENT_DIR.parents[0] / "entropy_tree_ensemble_deneme" / "scripts"
PHASE2_SCRIPTS = EXPERIMENT_DIR.parents[0] / "outlier_partial_corr_analizi" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PHASE1_SCRIPTS))
sys.path.insert(0, str(PHASE2_SCRIPTS))

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "YARISMA_TRAIN_PAH.csv"
V1_PATH = PROJECT_ROOT / "data" / "processed" / "pah" / "v1.parquet"
V3_PATH = PROJECT_ROOT / "data" / "processed" / "pah" / "v3.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"

SEED = 42


def safe_print(text):
    """Windows konsolunun cp1254 gibi dar kod sayfalarinda bazi unicode
    karakterler (ok isaretleri vb.) print() ile cokebiliyor -- dosya yazimi
    (write_text(..., encoding='utf-8')) bundan ETKILENMEZ, yalnizca terminal
    gorunumu icin bu guvenli fallback kullanilir."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def load_raw():
    return pd.read_csv(RAW_CSV)


def load_v1():
    return pd.read_parquet(V1_PATH)


def load_v3():
    return pd.read_parquet(V3_PATH)


def load_pool_features():
    import json
    return json.loads(POOL_PATH.read_text())["features"]


def al_columns(df):
    return [c for c in df.columns if c.startswith("AL_")]


def numeric_columns_raw(raw_df):
    """Ham veride sayisal (AL_+EK_) kolonlar -- CAT_/AA_/Variant_ID/Label haric."""
    return [c for c in raw_df.columns if c.startswith("AL_") or c.startswith("EK_")]


def categorical_columns_raw(raw_df):
    """Ham veride kategorik (CAT_+AA_) kolonlar -- CAT_6 (%100 eksik) dahil,
    cagiran taraf isterse elesin."""
    return [c for c in raw_df.columns if c.startswith("CAT_") or c.startswith("AA_")]


def entropy(labels):
    """Shannon entropisi (log2), bos/tek-sinifli girdi icin 0 doner."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def information_gain_numeric(x, y):
    """Sayisal bir X icin ID3-tarzi IG: tum benzersiz ara-noktalari esik
    adayi olarak dener, en yuksek IG'yi (ve o esigi) dondurur. Eksik (NaN)
    degerler ayri bir 'eksik' dali olarak ele alinir (H(Label|X) hesabina
    eksiklik-dali da katilir) -- boylece eksiklik/doluluk ayriminin
    bilgilendirici olup olmadigi da yakalanir.

    Doner: (best_ig, best_threshold, gain_ratio, missing_branch_informative)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    h_total = entropy(y)

    missing_mask = np.isnan(x)
    present_mask = ~missing_mask
    if present_mask.sum() < 2:
        return 0.0, None, 0.0, False

    x_present, y_present = x[present_mask], y[present_mask]
    uniq = np.unique(x_present)
    if len(uniq) < 2:
        # tek deger -- esik aranamaz, yalnizca eksiklik-dali bilgilendirici olabilir
        best_ig_present, best_t = 0.0, None
    else:
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        best_ig_present, best_t = -1.0, None
        n_present = len(x_present)
        for t in thresholds:
            left = y_present[x_present <= t]
            right = y_present[x_present > t]
            if len(left) == 0 or len(right) == 0:
                continue
            h_split = (len(left) / n_present) * entropy(left) + (len(right) / n_present) * entropy(right)
            ig = entropy(y_present) - h_split
            if ig > best_ig_present:
                best_ig_present, best_t = ig, float(t)
        best_ig_present = max(best_ig_present, 0.0)

    # eksiklik dahil toplam IG: (dolu vs eksik) ayrimini da bir dal olarak ekle
    n = len(x)
    h_missing_split = (present_mask.sum() / n) * entropy(y_present) + (missing_mask.sum() / n) * entropy(y[missing_mask])
    ig_missing_branch = h_total - h_missing_split
    missing_informative = bool(missing_mask.any() and ig_missing_branch > 0.01)

    # best_t, yalnizca dolu alt-kumede hesaplanan esik -- toplam IG'yi
    # (eksiklik + esik) birlikte veren bilesik bir agac dalinin IG'si:
    if best_t is not None:
        h_combined = (missing_mask.sum() / n) * entropy(y[missing_mask]) if missing_mask.any() else 0.0
        left = y_present[x_present <= best_t]
        right = y_present[x_present > best_t]
        h_combined += (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
        ig_combined = h_total - h_combined
    else:
        ig_combined = ig_missing_branch

    ig_combined = max(ig_combined, 0.0)
    # split information (Gain Ratio paydasi) -- dal buyukluklerinin entropisi
    branch_sizes = []
    if missing_mask.any():
        branch_sizes.append(missing_mask.sum())
    if best_t is not None:
        branch_sizes += [len(left), len(right)]
    elif present_mask.any():
        branch_sizes.append(present_mask.sum())
    split_info = entropy(np.concatenate([[i] * s for i, s in enumerate(branch_sizes)])) if branch_sizes else 0.0
    gain_ratio = ig_combined / split_info if split_info > 1e-9 else 0.0

    return ig_combined, best_t, gain_ratio, missing_informative


def information_gain_categorical(x, y):
    """Kategorik X icin IG: her benzersiz kategori (NaN dahil, ayri kategori
    olarak) kendi dali. Gain Ratio ile birlikte doner."""
    x = pd.Series(x).astype(object).where(pd.notna(x), "__MISSING__")
    y = np.asarray(y)
    h_total = entropy(y)
    n = len(x)
    h_split = 0.0
    branch_sizes = []
    for cat in x.unique():
        mask = (x == cat).values
        h_split += (mask.sum() / n) * entropy(y[mask])
        branch_sizes.append(mask.sum())
    ig = max(h_total - h_split, 0.0)
    split_info = entropy(np.concatenate([[i] * s for i, s in enumerate(branch_sizes)]))
    gain_ratio = ig / split_info if split_info > 1e-9 else 0.0
    return ig, gain_ratio
