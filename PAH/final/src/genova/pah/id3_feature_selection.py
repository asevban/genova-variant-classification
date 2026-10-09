"""Aşama D.1'e eklenen 5. özellik seçimi yöntemi: entropy-criterion
decision tree.

Terminoloji notu (rapora da taşınır): `sklearn.tree.DecisionTreeClassifier
(criterion="entropy")` tarihsel ID3 algoritmasının kendisi DEĞİLDİR -- ID3
yalnızca kategorik özelliklerle ve çok-yollu (multi-way) bölünmeyle çalışır,
budama yapmaz. Burada kullanılan, Information Gain kriterinin modern CART
(ikili bölünme, sürekli değişken desteği, isteğe bağlı derinlik sınırı)
çatısı altındaki karşılığıdır. Doğru ifade: "ID3'ün information-gain
mantığını CART çatısında uyguladık" -- "saf ID3 kullandık" değil.

Bu modül mevcut 4 yönteme (feature_selection.py) hiçbir şekilde dokunmaz;
yalnızca import edip okur. AYNI split bankasını (data/splits/pah/), AYNI
aday özellik evrenini (fold_features.build_fold_features) ve AYNI stabilite
formatını (%60 eşiği, 50 dış fold = 5x10) kullanır -- ki 5 yöntem doğrudan
karşılaştırılabilir olsun.

Çalıştırma: python -m genova.pah.id3_feature_selection
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from genova.pah.fold_features import build_fold_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
TAB_DIR = ROOT / "reports" / "tables"

EXISTING_STABILITY_CSV = TAB_DIR / "feature_selection_stability.csv"

N_REPEATS = 10
N_OUTER_SPLITS = 5
STABILITY_THRESHOLD = 0.60
SEED = 42

# Duyarlılık taraması bu adaylar üzerinde çalışır; 369 satırlık küçük bir
# veri setinde derin bir ağaç ezberler (aşırı öğrenir) ve stabilite ölçümü
# fold'lar arasında gürültüye döner -- bkz. reports/03c için gerekçe.
DEPTH_CANDIDATES = [3, 4, 5, 6, 8]
CHOSEN_MAX_DEPTH = 5


def id3_selected(X_train, y_train, seed, max_depth):
    """Bir dış-eğitim fold'unda entropy-criterion ağaç fit eder; ağaçta en
    az bir bölünmede kullanılan (feature_importances_ > 0) kolonları
    "seçilmiş" sayar.
    """
    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=max_depth, class_weight="balanced",
        random_state=seed,
    ).fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    selected = set(importances[importances > 0].index)
    return selected, importances


def _iter_folds(v1, al_columns):
    for repeat_idx in range(N_REPEATS):
        outer = json.load(open(SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json"))
        for fold in outer["folds"]:
            seed = SEED + repeat_idx * 100 + fold["fold"]
            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, fold["train_variant_ids"], fold["test_variant_ids"],
            )
            yield repeat_idx, fold["fold"], seed, X_train, y_train, X_test, y_test


def run_depth_sensitivity(v1, al_columns):
    """Her max_depth adayı için 50 dış fold boyunca ortalama seçilen
    özellik sayısını ve o derinlikteki stabil-havuz büyüklüğünü ölçer.
    Fold özellik matrisleri fold başına BİR KEZ inşa edilir, tüm derinlik
    adayları aynı matrisi paylaşır (gereksiz yeniden hesaplamayı önler).
    """
    hits_by_depth = {d: {} for d in DEPTH_CANDIDATES}
    n_selected_by_depth = {d: [] for d in DEPTH_CANDIDATES}
    n_folds = 0

    for repeat_idx, fold_idx, seed, X_train, y_train, _, _ in _iter_folds(v1, al_columns):
        for max_depth in DEPTH_CANDIDATES:
            sel, _ = id3_selected(X_train, y_train, seed, max_depth)
            n_selected_by_depth[max_depth].append(len(sel))
            for c in sel:
                hits_by_depth[max_depth][c] = hits_by_depth[max_depth].get(c, 0) + 1
        n_folds += 1
        print(f"[duyarlilik] repeat {repeat_idx} fold {fold_idx} done ({n_folds}/{N_REPEATS * N_OUTER_SPLITS})")

    rows = []
    for max_depth in DEPTH_CANDIDATES:
        rates = pd.Series(hits_by_depth[max_depth], dtype=float) / n_folds
        stable_pool_size = int((rates >= STABILITY_THRESHOLD).sum())
        rows.append({
            "max_depth": max_depth,
            "mean_features_selected_per_fold": float(np.mean(n_selected_by_depth[max_depth])),
            "std_features_selected_per_fold": float(np.std(n_selected_by_depth[max_depth])),
            "stable_pool_size_at_60pct": stable_pool_size,
        })
    return pd.DataFrame(rows)


def run_stability(v1, al_columns, max_depth):
    """Seçilen (chosen) max_depth ile 50 dış fold üzerinde tam stabilite
    tablosunu üretir -- mevcut 4 yöntemle (feature_selection.py) birebir
    aynı format ve eşik.
    """
    hits = {}
    all_features_seen = set()
    n_folds = 0
    for repeat_idx, fold_idx, seed, X_train, y_train, X_test, y_test in _iter_folds(v1, al_columns):
        all_features_seen.update(X_train.columns)
        sel, _ = id3_selected(X_train, y_train, seed, max_depth)
        for c in sel:
            hits[c] = hits.get(c, 0) + 1
        n_folds += 1
        print(f"[stabilite] repeat {repeat_idx} fold {fold_idx} done ({n_folds}/{N_REPEATS * N_OUTER_SPLITS})")

    all_features = sorted(all_features_seen)
    rate = (pd.Series(hits, dtype=float) / n_folds).reindex(all_features, fill_value=0.0)
    stability = pd.DataFrame({"id3_entropy_stability": rate})
    stability["stable_at_60pct"] = stability["id3_entropy_stability"] >= STABILITY_THRESHOLD
    stability = stability.sort_values("id3_entropy_stability", ascending=False)
    return stability, n_folds


def build_comparison_table():
    """Mevcut 4-yöntemli stabilite tablosunu (salt okunur) ID3 sonucuyla
    birleştirir; hem tam karşılaştırma tablosunu hem de "ID3 eklenseydi
    havuz nasıl değişirdi" simülasyonunu (bilgi amaçlı, uygulanmaz) üretir.
    """
    existing = pd.read_csv(EXISTING_STABILITY_CSV, index_col=0)
    id3 = pd.read_csv(TAB_DIR / "id3_feature_selection_stability.csv", index_col=0)

    all_features = sorted(set(existing.index) | set(id3.index))
    comparison = pd.DataFrame(index=all_features)
    comparison["mutual_information"] = existing["mutual_information"].reindex(all_features, fill_value=0.0)
    comparison["gbdt_importance"] = existing["gbdt_importance"].reindex(all_features, fill_value=0.0)
    comparison["elasticnet_stability"] = existing["elasticnet_stability"].reindex(all_features, fill_value=0.0)
    comparison["permutation_importance"] = existing["permutation_importance"].reindex(all_features, fill_value=0.0)
    comparison["id3_entropy_stability"] = id3["id3_entropy_stability"].reindex(all_features, fill_value=0.0)

    method_cols = ["mutual_information", "gbdt_importance", "elasticnet_stability",
                   "permutation_importance", "id3_entropy_stability"]
    for col in method_cols:
        comparison[f"{col}_stable"] = comparison[col] >= STABILITY_THRESHOLD

    stable_cols = [f"{c}_stable" for c in method_cols]
    comparison["n_methods_stable_with_id3"] = comparison[stable_cols].sum(axis=1)

    existing_stable_cols = [f"{c}_stable" for c in method_cols if c != "id3_entropy_stability"]
    comparison["n_methods_stable_without_id3"] = comparison[existing_stable_cols].sum(axis=1)

    comparison["in_current_pool_ge2_of_4"] = comparison["n_methods_stable_without_id3"] >= 2
    comparison["in_simulated_pool_ge2_of_5"] = comparison["n_methods_stable_with_id3"] >= 2

    comparison = comparison.sort_values("n_methods_stable_with_id3", ascending=False)
    comparison.to_csv(TAB_DIR / "five_method_stability_comparison.csv")

    current_pool = set(comparison[comparison["in_current_pool_ge2_of_4"]].index)
    simulated_pool = set(comparison[comparison["in_simulated_pool_ge2_of_5"]].index)
    summary = {
        "current_pool_size_4_methods": len(current_pool),
        "simulated_pool_size_5_methods": len(simulated_pool),
        "entered_pool_with_id3": sorted(simulated_pool - current_pool),
        "left_pool_with_id3": sorted(current_pool - simulated_pool),
        "unchanged": sorted(current_pool & simulated_pool),
        "note": ("Bu yalnizca bilgi amacli bir simulasyondur; v4 dosyalari veya "
                 "final ozellik havuzu bu gorevde DEGISTIRILMEDI."),
    }
    with open(TAB_DIR / "five_method_pool_simulation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return comparison, summary


def main():
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]

    print("=== max_depth duyarlilik kontrolu (5 aday x 50 dis fold) ===")
    sensitivity = run_depth_sensitivity(v1, al_columns)
    sensitivity.to_csv(TAB_DIR / "id3_max_depth_sensitivity.csv", index=False)
    print(sensitivity.to_string(index=False))

    print(f"\n=== secilen max_depth={CHOSEN_MAX_DEPTH} ile tam stabilite kosumu ===")
    stability, n_folds = run_stability(v1, al_columns, CHOSEN_MAX_DEPTH)
    stability.to_csv(TAB_DIR / "id3_feature_selection_stability.csv")

    stable_pool = stability[stability["stable_at_60pct"]].index.tolist()
    with open(TAB_DIR / "id3_feature_selection_stable_pool.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_total_folds": n_folds, "stability_threshold": STABILITY_THRESHOLD,
            "chosen_max_depth": CHOSEN_MAX_DEPTH, "depth_candidates": DEPTH_CANDIDATES,
            "n_stable_features": len(stable_pool), "stable_features": stable_pool,
        }, f, indent=2, ensure_ascii=False)
    print(f"ID3 (entropy, max_depth={CHOSEN_MAX_DEPTH}) stabil havuz buyuklugu: {len(stable_pool)}")

    print("\n=== 5-yontem karsilastirma tablosu + havuz simulasyonu ===")
    _, summary = build_comparison_table()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
