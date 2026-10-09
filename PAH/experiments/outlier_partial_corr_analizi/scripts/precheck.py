"""PRECHECK -- Faz 2 (aykiri gozlem + kismi korelasyon/kosullu IG) icin,
modelleme/analiz baslamadan ONCE calistirilir. Herhangi bir madde FAIL ise
script hata ile durur. Sonuc results/precheck_log.txt'e yazilir.

Faz 1'deki (experiments/entropy_tree_ensemble_deneme/scripts/precheck.py)
formatla aynidir, artı AL_correlation_summary.json kontrolu eklendi. Bu
script YALNIZCA experiments/outlier_partial_corr_analizi/results/ altina
yazar; data/, configs/, reports/ (tumu -- tablolar dahil), src/genova/pah/*.py
-- hicbirine yazmaz.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
AL_CORR_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "AL_correlation_summary.json"

N_EXPECTED_OUTER = 50
N_EXPECTED_INNER_PER_OUTER = 4
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

    # split bank + fold sayilari
    splits_exist = SPLITS_DIR.exists()
    record("split bank bulundu (data/splits/pah/)", splits_exist)

    outer_files = sorted(SPLITS_DIR.glob("outer_fold_repeat*.json")) if splits_exist else []
    total_outer = 0
    outer_keys = []
    for f in outer_files:
        d = json.loads(f.read_text())
        for fold in d["folds"]:
            total_outer += 1
            outer_keys.append((d["repeat"], fold["fold"]))
    record("outer fold sayisi = 50", total_outer == N_EXPECTED_OUTER, f"bulunan: {total_outer}")

    inner_ok = True
    for repeat_idx, fold_idx in outer_keys:
        p = SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json"
        if not p.exists() or len(json.loads(p.read_text())["folds"]) != N_EXPECTED_INNER_PER_OUTER:
            inner_ok = False
    record("her outer fold icin inner fold sayisi = 4", inner_ok)

    # build_fold_features import
    try:
        from genova.pah.fold_features import build_fold_features
        import genova.pah.fold_features as ff_module
        is_original = Path(ff_module.__file__).resolve() == (SRC_DIR / "genova" / "pah" / "fold_features.py").resolve()
        record("build_fold_features import edildi", is_original, str(Path(ff_module.__file__).resolve()))
    except Exception as exc:
        record("build_fold_features import edildi", False, str(exc))

    # pool
    pool_features = []
    if POOL_PATH.exists():
        pool_features = json.loads(POOL_PATH.read_text())["features"]
    record("v4_final_feature_pool.json'dan taze okunan liste = 25 ozellik",
           len(pool_features) == 25, f"bulunan: {len(pool_features)}")

    # AL_correlation_summary.json |r|>=0.90 kontrolu
    n_gt1_at_090 = None
    if AL_CORR_SUMMARY_PATH.exists():
        summary = json.loads(AL_CORR_SUMMARY_PATH.read_text())
        for row in summary["multi_threshold_summary"]:
            if abs(row["abs_r_threshold"] - 0.90) < 1e-9:
                n_gt1_at_090 = row["n_clusters_size_gt1"]
    record("AL_correlation_summary.json okundu, |r|>=0.90 cift sayisi = 0 (beklenen)",
           n_gt1_at_090 == 0, f"bulunan n_clusters_size_gt1: {n_gt1_at_090}")

    # cikti dizini
    record("cikti dizini yalnizca experiments/outlier_partial_corr_analizi/",
           str(RESULTS_DIR.resolve()).startswith(str(EXPERIMENT_DIR.resolve())), str(RESULTS_DIR.resolve()))

    # kaynak taramasi (precheck.py haric)
    write_calls_found = []
    for py_file in SCRIPT_DIR.glob("*.py"):
        if py_file.resolve() == Path(__file__).resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        for marker in ["open(", "to_csv(", "to_parquet(", "to_json(", "json.dump(", "write_text("]:
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


if __name__ == "__main__":
    main()
