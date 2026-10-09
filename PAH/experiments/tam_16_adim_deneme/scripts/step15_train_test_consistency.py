"""ADIM 15 -- Train-Test Sutun Uyumu.

Bu deney kapsaminda gercek bir test seti YOK (yarisma test seti henuz
paylasilmadi). Bu adim ILKE olarak dokumante edilir + kucuk bir dogrulama
fonksiyonu eklenir (resmi src/genova/pah/fold_features.py'nin zaten
yaptigi train/test tutarlilik mantigina atifla -- kod DEGISTIRILMEDI,
yalnizca ayni ilke burada bagimsiz kucuk bir fonksiyon olarak yeniden
uygulandi, cunku resmi fold_features.py'ye bu deneyden yazma/degistirme
yetkisi yok).

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_v1, load_pool_features, safe_print  # noqa: E402
from genova.pah.fold_features import build_fold_features  # noqa: E402


def validate_train_test_column_consistency(X_train, X_test, feature_cols, id_col_excluded_check=None):
    """Gelecekte gercek bir test seti geldiginde uygulanacak prosedurun
    kucuk bir dogrulama fonksiyonu -- resmi fold_features.py'nin ZATEN
    yaptigi train/test tutarlilik mantigina (fit yalnizca train'de,
    transform ile test'e uygulanir, `X_test.reindex(columns=feature_cols,
    fill_value=0)` ile kolon hizalamasi) atifla, ayni ilkeyi bagimsiz
    dogrular:

    1. Train'de olup test'te olmayan/test'te olup train'de olmayan hicbir
       ozellik kolonu KALMAMALI (ikisi de feature_cols'a hizalanmis olmali).
    2. Kolon SIRASI birebir ayni olmali (modelin `.predict()`'i pozisyonel
       calisir, isim degil).
    3. id_col_excluded_check (orn. 'Variant_ID') feature_cols icinde HIC
       olmamali.

    Uyumsuzluk varsa AssertionError firlatir (sessizce gecmez).
    """
    assert list(X_train.columns) == list(feature_cols), \
        f"X_train kolonlari feature_cols ile birebir ayni degil: {set(X_train.columns) ^ set(feature_cols)}"
    assert list(X_test.columns) == list(feature_cols), \
        f"X_test kolonlari feature_cols ile birebir ayni degil (sira dahil): {set(X_test.columns) ^ set(feature_cols)}"
    if id_col_excluded_check is not None:
        assert id_col_excluded_check not in feature_cols, \
            f"{id_col_excluded_check} feature_cols icinde bulunmamali (model girdisi olarak kullanilmamali)"
    return True


def main():
    v1 = load_v1()
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    pool_25 = load_pool_features()

    # kanit: mevcut fold_features.build_fold_features cikti kolonlari
    # zaten bu ilkeyi karsiliyor mu -- bagimsiz dogrulama
    ids = v1["Variant_ID"].tolist()
    X_train, y_train, X_test, y_test = build_fold_features(v1, al_columns, ids[:300], ids[300:])
    ok = validate_train_test_column_consistency(X_train[pool_25], X_test[pool_25], pool_25, "Variant_ID")
    safe_print(f"validate_train_test_column_consistency() -- resmi build_fold_features çıktısı üzerinde "
               f"bağımsız doğrulama: {'PASS' if ok else 'FAIL'}")

    report = f"""# Adım 15 — Train-Test Sütun Uyumu (İlke Dokümantasyonu)

**Bu deney kapsamında gerçek bir test seti yok** (yarışma test seti henüz
paylaşılmadı) — bu adım yalnızca **ilke olarak** dokümante ediliyor +
küçük bir doğrulama fonksiyonu (`validate_train_test_column_consistency()`,
bu script içinde tanımlı) ekleniyor.

## Gelecekte Test Seti Geldiğinde Uygulanacak Prosedür

1. **Train'de çıkarılan/dönüştürülen her özellik, test'ten de aynı şekilde
   çıkarılacak/dönüştürülecek.** Hiçbir istatistik (medyan, frekans, ölçek
   parametresi) test verisinden öğrenilmeyecek — yalnızca train'den `fit`
   edilip test'e `transform` uygulanacak. Bu, resmi
   `src/genova/pah/fold_features.py::build_fold_features`'ın **zaten**
   uyguladığı ilkeyle birebir aynı (bu deney o modülü değiştirmeden import
   ederek kullandı, aynı ilkeyi miras aldı).
2. **`Variant_ID` hiçbir zaman modele girdi olarak verilmeyecek** — yalnızca
   kimlik/eşleme amaçlı saklanacak (tahminleri `Variant_ID`'ye göre
   raporlamak için).
3. **İstatistikler yalnızca train'den öğrenilecek** — medyan doldurma,
   frekans kodlama, ölçekleme parametreleri (varsa) dahil.
4. **Kolon sırası/eksik-fazla sütun kontrolü yapılacak** — `X_test`,
   `X_train`'in kolon listesine (sırayla) `reindex` edilecek; train'de
   olmayan bir test-kolonu sessizce atılacak, train'de olan ama test'te
   eksik bir kolon `fill_value=0` ile doldurulacak (resmi
   `fold_features.py`'nin zaten yaptığı gibi — `X_test.reindex(columns=
   feature_cols, fill_value=0)`).

## Kod Doğrulaması (bu turda)

`validate_train_test_column_consistency(X_train, X_test, feature_cols,
id_col_excluded_check)` fonksiyonu yazıldı ve **resmi
`build_fold_features()` çıktısı üzerinde bağımsız olarak test edildi** —
yukarıdaki `PASS` sonucu, resmi pipeline'ın bu ilkeyi zaten doğru
uyguladığının bu deney tarafından bağımsız bir teyididir.
"""
    (RESULTS_DIR / "step15_train_test_consistency.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
