"""PRECHECK -- Faz 3 (orijinal 16 adimlik prompt, tam kapsam) icin, herhangi
bir analiz baslamadan ONCE calistirilir. Faz 1/2 ile ayni format. Herhangi
bir madde FAIL ise script hata ile durur. Sonuc results/precheck_log.txt'e
yazilir. Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina
yazar; data/, configs/, reports/, src/genova/pah/*.py -- hicbirine yazmaz.
"""
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "YARISMA_TRAIN_PAH.csv"

N_EXPECTED_OUTER = 50
N_EXPECTED_INNER_PER_OUTER = 4
EXPECTED_RAW_SHAPE = (372, 353)
FORBIDDEN_DIRS = ["data/", "configs/", "reports/", "src/genova/pah/"]


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {label}" + (f" -- {detail}" if detail else "")
    print(line)
    return status, line


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    all_pass = True

    def record(label, condition, detail=""):
        nonlocal all_pass
        status, line = check(label, condition, detail)
        log_lines.append(line)
        if status == "FAIL":
            all_pass = False
        return condition

    splits_exist = SPLITS_DIR.exists()
    outer_files = sorted(SPLITS_DIR.glob("outer_fold_repeat*.json")) if splits_exist else []
    total_outer = 0
    outer_keys = []
    for f in outer_files:
        d = json.loads(f.read_text())
        for fold in d["folds"]:
            total_outer += 1
            outer_keys.append((d["repeat"], fold["fold"]))
    record("split bank bulundu, outer fold sayisi = 50",
           splits_exist and total_outer == N_EXPECTED_OUTER, f"bulunan: {total_outer}")

    inner_ok = True
    for repeat_idx, fold_idx in outer_keys:
        p = SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json"
        if not p.exists() or len(json.loads(p.read_text())["folds"]) != N_EXPECTED_INNER_PER_OUTER:
            inner_ok = False
    record("her outer fold icin inner fold sayisi = 4", inner_ok)

    try:
        from genova.pah.fold_features import build_fold_features
        import genova.pah.fold_features as ff_module
        is_original = Path(ff_module.__file__).resolve() == (SRC_DIR / "genova" / "pah" / "fold_features.py").resolve()
        record("build_fold_features import edildi", is_original, str(Path(ff_module.__file__).resolve()))
    except Exception as exc:
        record("build_fold_features import edildi", False, str(exc))

    pool_features = []
    if POOL_PATH.exists():
        pool_features = json.loads(POOL_PATH.read_text())["features"]
    record("v4_final_feature_pool.json'dan taze okunan liste = 25 ozellik",
           len(pool_features) == 25, f"bulunan: {len(pool_features)}")

    raw_shape = None
    if RAW_CSV.exists():
        raw_df = pd.read_csv(RAW_CSV)
        raw_shape = raw_df.shape
    record("ham veri (YARISMA_TRAIN_PAH.csv) satir/kolon sayisi = 372/353",
           raw_shape == EXPECTED_RAW_SHAPE, f"bulunan: {raw_shape}")

    record("cikti dizini yalnizca experiments/tam_16_adim_deneme/",
           str(RESULTS_DIR.resolve()).startswith(str(EXPERIMENT_DIR.resolve())), str(RESULTS_DIR.resolve()))

    write_calls_found = []
    for py_file in SCRIPT_DIR.glob("*.py"):
        if py_file.resolve() == Path(__file__).resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        for marker in ["open(", "to_csv(", "to_parquet(", "to_json(", "json.dump(", "write_text(", "savefig("]:
            idx = text.find(marker)
            while idx != -1:
                window = text[max(0, idx - 200):idx + 50]
                if any(t in window for t in FORBIDDEN_DIRS):
                    write_calls_found.append(f"{py_file.name}: '{marker}' yakininda yasakli yol")
                idx = text.find(marker, idx + 1)
    record("official dosyalara yazma modu KAPALI (kaynak denetimi)",
           len(write_calls_found) == 0, "; ".join(write_calls_found) if write_calls_found else "temiz")

    log_path = RESULTS_DIR / "precheck_log.txt"
    log_path.write_text("\n".join(log_lines) + f"\n\nGENEL SONUC: {'PASS' if all_pass else 'FAIL'}\n", encoding="utf-8")
    print(f"\nGENEL SONUC: {'PASS' if all_pass else 'FAIL'}")
    print(f"Kaydedildi: {log_path}")
    if not all_pass:
        print("\nEN AZ BIR MADDE FAIL -- analiz BASLATILMAYACAK.")
        sys.exit(1)
    return pool_features


if __name__ == "__main__":
    main()
