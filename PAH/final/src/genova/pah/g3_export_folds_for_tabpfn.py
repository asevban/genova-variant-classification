"""TabPFN v2 denemesi icin gecici cikti: split bankasinin dis fold'larini
(v2/v4_from_v2 temsilleriyle, `fold_versions.py`'nin AYNI fold-guvenli
insaacilariyla) izole 3.11/TabPFN ortamina tasimak icin parquet olarak
disari yazar. TabPFN'in kendisi bu ortamda calismiyor -- yalnizca ozellik
matrisleri, iki ortam arasinda parquet ile tasiniyor.

BU BIR REPO-KALICI SCRIPT DEGIL -- yalnizca gecici deney-degisim
dosyalari (proje disi bir scratch dizinine) uretir, split bankasina/
data/processed'a hicbir yazma yapmaz.

Calistirma: python -m genova.pah.g3_export_folds_for_tabpfn --out-dir <dir> [--n-folds 5|50]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"

VERSIONS = {
    "v2": lambda df, tr, te, pool: fv.build_v2(df, tr, te),
    "v4_from_v2": lambda df, tr, te, pool: fv.build_v4_from_v2(df, tr, te, pool),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-folds-per-repeat", type=int, default=5)
    parser.add_argument("--limit-folds", type=int, default=None, help="ilk N (repeat,fold) cifti ile pilot")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_parquet(V1_PATH)
    pool = json.loads(POOL_PATH.read_text())["features"]

    manifest = []
    n_written = 0
    for repeat_idx in range(args.n_repeats):
        # split_bank.py::main()'in kendi konvansiyonuyla AYNI: outer_seed = BASE_SEED(42) + repeat_idx.
        # repeat_idx'in kendisi splitter'a hic gitmiyor, yalnizca seed farklilastiriyor -- bunu
        # sabit seed=42 ile cagirmak butun "tekrarlari" ayni tek-5-fold'a indirger (bulunup duzeltildi).
        outer_folds = sb.build_outer_folds(v1_df, repeat_idx=repeat_idx, seed=42 + repeat_idx)
        for outer_fold in outer_folds:
            if args.limit_folds is not None and n_written >= args.limit_folds:
                break
            fold_idx = outer_fold["fold"]
            for version_name, builder in VERSIONS.items():
                X_tr, y_tr, X_te, y_te = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"], pool)
                fname_prefix = f"repeat{repeat_idx:02d}_fold{fold_idx}_{version_name}"
                X_tr.to_parquet(out_dir / f"{fname_prefix}_Xtr.parquet")
                X_te.to_parquet(out_dir / f"{fname_prefix}_Xte.parquet")
                pd.DataFrame({"y": y_tr}).to_parquet(out_dir / f"{fname_prefix}_ytr.parquet")
                pd.DataFrame({"y": y_te}).to_parquet(out_dir / f"{fname_prefix}_yte.parquet")
                manifest.append({"repeat": repeat_idx, "fold": fold_idx, "version": version_name, "prefix": fname_prefix})
            n_written += 1
        if args.limit_folds is not None and n_written >= args.limit_folds:
            break

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{n_written} fold, {len(manifest)} (fold,version) kombinasyonu yazildi -> {out_dir}")


if __name__ == "__main__":
    main()
