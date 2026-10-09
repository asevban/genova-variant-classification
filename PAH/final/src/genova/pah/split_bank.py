"""Sabit PAH CV split bankasını üretir ve yazar: dış 5-fold x 10-tekrar
(Repeated Stratified), iç 4-fold, ikisi de grup-farkında olduğu için
çelişkili-profil çifti (group_id="conflict_group_1", bkz.
dataset_versions.build_v1) hiçbir zaman fold sınırını geçmez. data/splits/
pah/ altına bir kez yazılır ve bir daha üretilmez -- sonraki her deney
(özellik seçimi, modelleme) kendi rastgele bölmesini çıkarmak yerine bu
dosyaları okur.

Çalıştırma: python -m genova.pah.split_bank
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"

N_OUTER_SPLITS = 5
N_REPEATS = 10
N_INNER_SPLITS = 4
BASE_SEED = 42


def build_outer_folds(df, repeat_idx, seed):
    """Tam veri seti üzerinde grup-farkında stratified 5-fold bölmenin bir
    tekrarı. {fold, train_variant_ids, test_variant_ids} listesi döner.
    """
    splitter = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, df["Label"], df["group_id"])):
        folds.append({
            "fold": fold_idx,
            "train_variant_ids": df.iloc[train_idx]["Variant_ID"].tolist(),
            "test_variant_ids": df.iloc[test_idx]["Variant_ID"].tolist(),
        })
    return folds


def build_inner_folds(df, outer_train_variant_ids, seed):
    """Yalnızca bir dış fold'un eğitim satırlarıyla sınırlı, grup-farkında
    stratified 4-fold bölme. {fold, train_variant_ids, val_variant_ids}
    listesi döner.
    """
    subset = df[df["Variant_ID"].isin(outer_train_variant_ids)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(subset, subset["Label"], subset["group_id"])):
        folds.append({
            "fold": fold_idx,
            "train_variant_ids": subset.iloc[train_idx]["Variant_ID"].tolist(),
            "val_variant_ids": subset.iloc[val_idx]["Variant_ID"].tolist(),
        })
    return folds


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(V1_PATH, columns=["Variant_ID", "Label", "group_id"])

    conflict_group_sizes = df.groupby("group_id").size()
    assert (conflict_group_sizes <= 2).all(), "beklenmeyen buyuk group_id kumesi bulundu"

    for repeat_idx in range(N_REPEATS):
        outer_seed = BASE_SEED + repeat_idx
        outer_folds = build_outer_folds(df, repeat_idx, outer_seed)

        with open(SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json", "w") as f:
            json.dump({
                "repeat": repeat_idx, "seed": outer_seed, "n_splits": N_OUTER_SPLITS,
                "library": "sklearn.model_selection.StratifiedGroupKFold",
                "folds": outer_folds,
            }, f, indent=2)

        for outer_fold in outer_folds:
            inner_seed = outer_seed * 1000 + outer_fold["fold"]
            inner_folds = build_inner_folds(df, outer_fold["train_variant_ids"], inner_seed)
            with open(SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{outer_fold['fold']}.json", "w") as f:
                json.dump({
                    "repeat": repeat_idx, "outer_fold": outer_fold["fold"], "seed": inner_seed,
                    "n_splits": N_INNER_SPLITS,
                    "library": "sklearn.model_selection.StratifiedGroupKFold",
                    "folds": inner_folds,
                }, f, indent=2)

    manifest = {
        "design": f"dis {N_OUTER_SPLITS}-fold x {N_REPEATS} tekrar (Repeated Stratified), ic {N_INNER_SPLITS}-fold",
        "group_awareness": (
            "group_id (v1.parquet'ten) celiskili-profil ciftini "
            "(VAR_003238/VAR_003234, group_id='conflict_group_1') her dis ve ic "
            "bolmede ayni tarafta tutar; diger tum satirlar tekil (singleton) gruptur."
        ),
        "base_seed": BASE_SEED,
        "library": "sklearn.model_selection.StratifiedGroupKFold",
        "source_dataset": str(V1_PATH.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "outer_files": f"outer_fold_repeat00.json .. outer_fold_repeat{N_REPEATS - 1:02d}.json",
        "inner_files": f"inner_fold_repeat{{RR}}_outer{{F}}.json for RR in 00..{N_REPEATS - 1:02d}, F in 0..{N_OUTER_SPLITS - 1}",
        "immutable": True,
    }
    with open(SPLITS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"split bankasi yazildi: {SPLITS_DIR}")
    print(f"{N_REPEATS} outer dosyasi + {N_REPEATS * N_OUTER_SPLITS} inner dosyasi + manifest.json")


if __name__ == "__main__":
    main()
