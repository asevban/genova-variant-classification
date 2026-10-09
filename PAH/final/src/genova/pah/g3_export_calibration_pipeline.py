"""TabPFN v2 denemesi -- Adim 4 (nested kalibrasyon+SLD+esik ile
weighted-F1 hesaplamak icin) icin gecici veri disari-aktarimi.
Yalnizca `v4_from_v2` (final adayin kendi ozellik temsili) icin: her dis
fold'un hem OUTER (train,test) hem 4 INNER (train,val) bolmelerini,
`Variant_ID` korunarak, izole TabPFN ortamina parquet olarak yazar --
E3/E5'in `cross_fit_probabilities`/`process_fold` deseniyle AYNI split
yapisi (split bankasindan, degistirilmeden okunur).

BU BIR REPO-KALICI SCRIPT DEGIL -- yalnizca gecici scratch cikti
uretir, split bankasina/data/processed'a hicbir yazma yapmaz.

Calistirma: python -m genova.pah.g3_export_calibration_pipeline --out-dir <dir> [--n-repeats 10]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.e3_calibration_run import _variant_ids_in_row_order

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"


def _save(X_tr, y_tr, X_te, y_te, test_vids, out_dir, prefix):
    X_tr.to_parquet(out_dir / f"{prefix}_Xtr.parquet")
    X_te.to_parquet(out_dir / f"{prefix}_Xte.parquet")
    pd.DataFrame({"y": y_tr}).to_parquet(out_dir / f"{prefix}_ytr.parquet")
    pd.DataFrame({"y": y_te, "variant_id": test_vids}).to_parquet(out_dir / f"{prefix}_yte.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-repeats", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_parquet(V1_PATH)
    pool = json.loads(POOL_PATH.read_text())["features"]

    def builder(tr, te):
        return fv.build_v4_from_v2(v1_df, tr, te, pool)

    manifest = []
    for repeat_idx in range(args.n_repeats):
        # split_bank.py::main() ile AYNI: outer_seed = BASE_SEED(42) + repeat_idx (sabit seed=42
        # DEGIL -- ilk tur bunu sabit birakip 10 "tekrari" ayni tek-5-fold'a indirgemisti, duzeltildi).
        outer_folds = sb.build_outer_folds(v1_df, repeat_idx=repeat_idx, seed=42 + repeat_idx)
        for outer_fold in outer_folds:
            fold_idx = outer_fold["fold"]

            X_tr, y_tr, X_te, y_te = builder(outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            test_vids = _variant_ids_in_row_order(v1_df, outer_fold["test_variant_ids"])
            prefix = f"repeat{repeat_idx:02d}_fold{fold_idx}_outer"
            _save(X_tr, y_tr, X_te, y_te, test_vids, out_dir, prefix)
            manifest.append({"repeat": repeat_idx, "fold": fold_idx, "kind": "outer", "inner_idx": -1, "prefix": prefix})

            inner_folds = json.loads((SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json").read_text())["folds"]
            for inner_fold in inner_folds:
                Xi_tr, yi_tr, Xi_va, yi_va = builder(inner_fold["train_variant_ids"], inner_fold["val_variant_ids"])
                val_vids = _variant_ids_in_row_order(v1_df, inner_fold["val_variant_ids"])
                iprefix = f"repeat{repeat_idx:02d}_fold{fold_idx}_inner{inner_fold['fold']}"
                _save(Xi_tr, yi_tr, Xi_va, yi_va, val_vids, out_dir, iprefix)
                manifest.append({"repeat": repeat_idx, "fold": fold_idx, "kind": "inner", "inner_idx": inner_fold["fold"], "prefix": iprefix})

            print(f"repeat={repeat_idx} fold={fold_idx}: 1 outer + {len(inner_folds)} inner yazildi", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"toplam {len(manifest)} (outer+inner) bolme yazildi -> {out_dir}")


if __name__ == "__main__":
    main()
