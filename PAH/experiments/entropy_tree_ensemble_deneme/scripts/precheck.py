"""PRECHECK -- bagimsiz entropy-tree ensemble deneyi, modelleme baslamadan
ONCE calistirilir. Herhangi bir madde FAIL ise script hata ile durur; hicbir
model egitilmez. Sonuc results/precheck_log.txt'e yazilir.

Bu script YALNIZCA experiments/entropy_tree_ensemble_deneme/results/ altina
yazar. data/, configs/, reports/tables/, src/genova/pah/*.py -- hicbirine
yazma girisiminde bulunmaz, yalnizca import/okuma yapar.

Calistirma: python precheck.py (experiments/entropy_tree_ensemble_deneme/scripts/ icinden)
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]  # .../experiments/<bu>/  -> proje koku
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"

EXPECTED_25 = sorted([
    "AL_12", "AL_171", "AL_22", "AL_26", "AL_277", "AL_298", "AL_300", "AL_301",
    "AL_306", "AL_317", "AL_318", "AL_329", "AL_330", "AL_334", "AL_49", "AL_8",
    "CAT_1", "EK_1", "EK_2", "EK_5", "EK_6", "EK_7", "EK_8", "EK_9", "al_all_missing",
])

N_EXPECTED_OUTER = 50
N_EXPECTED_INNER_PER_OUTER = 4


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

    # 1) split bank bulundu
    splits_exist = SPLITS_DIR.exists() and SPLITS_DIR.is_dir()
    record("split bank bulundu (data/splits/pah/)", splits_exist, str(SPLITS_DIR))

    # 2) outer fold sayisi = 50 (10 repeat dosyasinin toplam fold sayisi)
    outer_files = sorted(SPLITS_DIR.glob("outer_fold_repeat*.json")) if splits_exist else []
    total_outer_folds = 0
    outer_fold_keys = []  # (repeat_idx, fold_idx)
    for f in outer_files:
        data = json.loads(f.read_text())
        repeat_idx = data["repeat"]
        for fold in data["folds"]:
            total_outer_folds += 1
            outer_fold_keys.append((repeat_idx, fold["fold"]))
    record("outer fold sayisi = 50", total_outer_folds == N_EXPECTED_OUTER,
           f"bulunan: {total_outer_folds} ({len(outer_files)} repeat dosyasi)")

    # 3) her outer fold icin inner fold sayisi = 4
    inner_ok = True
    inner_missing = []
    for repeat_idx, fold_idx in outer_fold_keys:
        inner_path = SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json"
        if not inner_path.exists():
            inner_ok = False
            inner_missing.append(str(inner_path.name))
            continue
        inner_data = json.loads(inner_path.read_text())
        if len(inner_data["folds"]) != N_EXPECTED_INNER_PER_OUTER:
            inner_ok = False
            inner_missing.append(f"{inner_path.name} (n_inner={len(inner_data['folds'])})")
    record("her outer fold icin inner fold sayisi = 4", inner_ok,
           "hepsi tamam" if inner_ok else f"sorunlu: {inner_missing[:5]}")

    # 4) build_fold_features import edildi (yeniden yazilmadi)
    try:
        from genova.pah.fold_features import build_fold_features, columns_for_group
        import genova.pah.fold_features as ff_module
        ff_source_path = Path(ff_module.__file__).resolve()
        is_original = ff_source_path == (SRC_DIR / "genova" / "pah" / "fold_features.py").resolve()
        record("build_fold_features import edildi (yeniden yazilmadi)", is_original,
               str(ff_source_path))
    except Exception as exc:
        record("build_fold_features import edildi (yeniden yazilmadi)", False, str(exc))
        build_fold_features = None

    # 5) v4_final_feature_pool.json'dan taze okunan liste uzunlugu = 25
    pool_features = []
    if POOL_PATH.exists():
        pool = json.loads(POOL_PATH.read_text())
        pool_features = pool["features"]
    record("v4_final_feature_pool.json'dan taze okunan liste uzunlugu = 25",
           len(pool_features) == 25, f"bulunan: {len(pool_features)}")

    # 6) liste bu gorevdeki sabit listeyle eslesiyor mu -- eslesmiyorsa json esas alinir, DURDURMA sebebi degil
    fresh_sorted = sorted(pool_features)
    lists_match = fresh_sorted == EXPECTED_25
    log_lines.append(
        f"[INFO] json listesi gorev listesiyle {'birebir eslesiyor' if lists_match else 'FARKLI -- json esas alinacak'}"
    )
    print(log_lines[-1])
    if not lists_match:
        log_lines.append(f"[INFO] json'daki liste: {fresh_sorted}")
        log_lines.append(f"[INFO] gorevdeki liste : {EXPECTED_25}")
        print(log_lines[-2])
        print(log_lines[-1])

    # 7) build_fold_features() ciktisinin kolon sayisi ~405, Label/Variant_ID yok
    if build_fold_features is not None and splits_exist and total_outer_folds > 0:
        v1_path = PROJECT_ROOT / "data" / "processed" / "pah" / "v1.parquet"
        import pandas as pd
        v1 = pd.read_parquet(v1_path)
        al_columns = [c for c in v1.columns if c.startswith("AL_")]
        first_outer = json.loads(outer_files[0].read_text())["folds"][0]
        X_train, y_train, X_test, y_test = build_fold_features(
            v1, al_columns, first_outer["train_variant_ids"], first_outer["test_variant_ids"],
        )
        n_cols = X_train.shape[1]
        cols_ok = 395 <= n_cols <= 415  # "+-birkac", aday evren tanimina gore
        no_leak_cols = ("Label" not in X_train.columns) and ("Variant_ID" not in X_train.columns)
        record("build_fold_features() cikti kolon sayisi ~405 (+-birkac)", cols_ok, f"bulunan: {n_cols}")
        record("Label ve Variant_ID aday evrende YOK", no_leak_cols, f"kolonlarda: {no_leak_cols}")
        record("25-ozellik havuzunun tamami aday evrende mevcut",
               set(pool_features).issubset(set(X_train.columns)),
               f"eksik: {set(pool_features) - set(X_train.columns)}")
    else:
        record("build_fold_features() cikti kolon sayisi ~405 (+-birkac)", False, "onceki adim basarisiz, atlandi")
        record("Label ve Variant_ID aday evrende YOK", False, "onceki adim basarisiz, atlandi")
        record("25-ozellik havuzunun tamami aday evrende mevcut", False, "onceki adim basarisiz, atlandi")

    # 8) cikti dizini yalnizca experiments/entropy_tree_ensemble_deneme/
    record("cikti dizini yalnizca experiments/entropy_tree_ensemble_deneme/",
           RESULTS_DIR.resolve().is_relative_to(EXPERIMENT_DIR.resolve()) if hasattr(Path, "is_relative_to")
           else str(RESULTS_DIR.resolve()).startswith(str(EXPERIMENT_DIR.resolve())),
           str(RESULTS_DIR.resolve()))

    # 9) data/, configs/, reports/tables/, src/genova/pah/*.py yazilabilir MODDA DEGIL
    #    (bu scriptin/paket kodunun bu yollara acikca write cagirmadigini kaynak-denetimiyle dogrula)
    forbidden_targets = ["data/", "configs/", "reports/tables/", "src/genova/pah/"]
    write_calls_found = []
    # precheck.py kendi kaynagini taramaz -- bu dosyanin KENDI marker/forbidden-
    # targets liste tanimlari (asagida), metinsel pencere taramasinda kendi
    # kendine yanlis-pozitif uretiyordu (ornegin bu satirin kendisi "open("
    # icin bir string literal iceriyor). precheck.py'nin tek yazma cagrisi
    # `log_path.write_text(...)` -- log_path = RESULTS_DIR/precheck_log.txt,
    # kod okunarak dogrulanabilir. Diger tum script'ler (gercek write cagrilari
    # icerecekler) bu taramaya dahil.
    for py_file in (SCRIPT_DIR).glob("*.py"):
        if py_file.resolve() == Path(__file__).resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        for marker in ["open(", "to_csv(", "to_parquet(", "to_json(", "json.dump(", "write_text("]:
            idx = text.find(marker)
            while idx != -1:
                window = text[max(0, idx - 200):idx + 50]
                if any(t in window for t in forbidden_targets):
                    write_calls_found.append(f"{py_file.name}: '{marker}' cagrisi yakininda yasakli yol")
                idx = text.find(marker, idx + 1)
    record("data/, configs/, reports/tables/, src/genova/pah/*.py'a yazma girisimi YOK (kaynak denetimi)",
           len(write_calls_found) == 0, "; ".join(write_calls_found) if write_calls_found else "temiz")

    log_path = RESULTS_DIR / "precheck_log.txt"
    log_path.write_text("\n".join(log_lines) + f"\n\nGENEL SONUC: {'PASS' if all_pass else 'FAIL'}\n", encoding="utf-8")
    print(f"\nGENEL SONUC: {'PASS' if all_pass else 'FAIL'}")
    print(f"Kaydedildi: {log_path}")

    if not all_pass:
        print("\nEN AZ BIR MADDE FAIL -- modelleme BASLATILMAYACAK.")
        sys.exit(1)
    return pool_features


if __name__ == "__main__":
    main()
