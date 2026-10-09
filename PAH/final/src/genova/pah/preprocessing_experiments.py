"""Tamamlayıcı Deney Turu — Görev 4 (ölçekleme karşılaştırması), Görev 5
(preprocessing merdiveni) ve Görev 6 (yüksek-eksiklik filtreleme denemesi).

Üçü de aynı sabit split bankasını (`data/splits/pah/`, 50 dış fold) ve aynı
3 sabit diagnostic modeli (`diagnostics.fixed_models`) kullanır; tek
değişken ön işleme kararıdır. Hiçbir model burada tune edilmez, hiçbir
sonuç final performans olarak raporlanmaz.

Çalıştırma: python -m genova.pah.preprocessing_experiments
"""
import json
from pathlib import Path

import pandas as pd

from genova.pah.diagnostics import evaluate_variant, summarize
from genova.pah.preprocessing_ladder import build_step, train_fold_missing_rate

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
TAB_DIR = ROOT / "reports" / "tables"


def _split_files():
    return sorted(SPLITS_DIR.glob("outer_fold_repeat*.json"))


def run_scaling_comparison(v1, al_columns, split_files):
    """Görev 4: basamak 3 (dönüştürülmüş, ölçeklemesiz) üzerine üç varyant."""
    variants = {
        "olceklemesiz": lambda tr, te: build_step(v1, al_columns, tr, te, step=3),
        "StandardScaler": lambda tr, te: build_step(v1, al_columns, tr, te, step=4, scaler_method="standard"),
        "RobustScaler": lambda tr, te: build_step(v1, al_columns, tr, te, step=4, scaler_method="robust"),
    }
    raw = pd.concat([evaluate_variant(fn, split_files, name) for name, fn in variants.items()], ignore_index=True)
    raw.to_csv(TAB_DIR / "scaling_comparison_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(TAB_DIR / "scaling_comparison.csv", index=False)
    print("\n=== Görev 4: Ölçekleme Karşılaştırması ===")
    print(summary.to_string(index=False))
    return summary


def run_ladder(v1, al_columns, split_files, winning_scaler):
    """Görev 5: basamak 0-5, kazanan ölçekleyiciyle (Görev 4'ten)."""
    variants = {f"basamak_{s}": (lambda tr, te, s=s: build_step(v1, al_columns, tr, te, step=s, scaler_method=winning_scaler)) for s in range(6)}
    raw = pd.concat([evaluate_variant(fn, split_files, name) for name, fn in variants.items()], ignore_index=True)
    raw.to_csv(TAB_DIR / "preprocessing_ladder_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(TAB_DIR / "preprocessing_ladder.csv", index=False)
    print("\n=== Görev 5: Preprocessing Merdiveni ===")
    print(summary.to_string(index=False))
    return summary


def run_missingness_filter(v1, al_columns, split_files, base_step=2):
    """Görev 6: train-fold-only eksiklik oranına göre AL_ kolon filtrelemesi,
    basamak 2 üzerine uygulanır (imputasyondan önce, ham eksiklik oranına göre).
    """
    thresholds = [0.70, 0.80, 0.90, 0.95]
    n_dropped_by_thresh = {}

    def make_variant(thresh):
        def fn(train_ids, test_ids):
            rates = train_fold_missing_rate(v1, al_columns, train_ids)
            keep_cols = rates[rates <= thresh].index.tolist()
            n_dropped_by_thresh.setdefault(thresh, []).append(len(al_columns) - len(keep_cols))
            return build_step(v1, keep_cols, train_ids, test_ids, step=base_step)
        return fn

    variants = {f"esik_{int(t*100)}pct": make_variant(t) for t in thresholds}
    variants["filtresiz_basamak2"] = lambda tr, te: build_step(v1, al_columns, tr, te, step=base_step)

    raw = pd.concat([evaluate_variant(fn, split_files, name) for name, fn in variants.items()], ignore_index=True)
    raw.to_csv(TAB_DIR / "missingness_filter_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(TAB_DIR / "missingness_filter_experiment.csv", index=False)

    drop_summary = {f"esik_{int(t*100)}pct": {"ort_dusen_kolon": sum(v) / len(v), "min": min(v), "max": max(v)}
                     for t, v in n_dropped_by_thresh.items()}
    with open(TAB_DIR / "missingness_filter_dropped_columns.json", "w") as f:
        json.dump(drop_summary, f, indent=2)

    print("\n=== Görev 6: Yüksek-Eksiklik Filtreleme Denemesi ===")
    print(summary.to_string(index=False))
    print("\nDüşen kolon sayısı (fold-bazlı ortalama):")
    print(json.dumps(drop_summary, indent=2))
    return summary, drop_summary


def main():
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    split_files = _split_files()
    print(f"split bankasi: {len(split_files)} dis tekrar dosyasi bulundu")

    scaling_summary = run_scaling_comparison(v1, al_columns, split_files)

    # kazanan olcekleyiciyi F1 (patojenik) ortalamasina gore sec (LogReg + ExtraTrees uzerinden, Dummy haric)
    non_dummy = scaling_summary[scaling_summary["model"] != "Dummy"]
    scaled_variants = non_dummy[non_dummy["variant"] != "olceklemesiz"]
    best_row = scaled_variants.loc[scaled_variants["f1_pathogenic_mean"].idxmax()]
    winning_scaler = "standard" if best_row["variant"] == "StandardScaler" else "robust"
    print(f"\nGörev 4 sonucuna göre kazanan ölçekleyici: {winning_scaler} ({best_row['variant']})")

    run_ladder(v1, al_columns, split_files, winning_scaler)
    run_missingness_filter(v1, al_columns, split_files)

    print("\nTüm deneyler tamamlandı.")


if __name__ == "__main__":
    main()
