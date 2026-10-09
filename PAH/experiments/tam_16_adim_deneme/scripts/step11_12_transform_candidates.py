"""ADIM 11 -- Normalizasyon ve ortalama-ozellik denemesi (aday tanimlama).
ADIM 12 -- Ustel gorunen ciftlerde log donusumu (aday tanimlama).

Bu script HESAPLAMA YAPMAZ (nested olmayan bir CV'de normalizasyon/ortalama
ozellik "denemek" sizinti riski tasir -- z-skor/min-max istatistikleri
yalnizca outer-train'den ogrenilmelidir). Bunun yerine:
  (a) Adim 13'un nested prosedurunde kullanilacak ortalama-ozellik ve
      normalizasyon fonksiyonlarini TANIMLAR (fold-guvenli, outer-train
      fit edilecek sekilde tasarlanir),
  (b) Adim 12'nin kapsamini dogru sekilde raporlar: Adim 10, test edilen 4
      guclu ciftin HICBIRINDE ustel iliski BULAMADI (lineer her zaman
      kazandi) -- bu yuzden Adim 12'nin "spesifik cift" kosulu bu turda
      BOS. Bunun yerine, GENOVA'nin M-3 basamagindaki GENEL log1p bulgusuyla
      celisip celismedigini kontrol etmek icin log1p-donusumlu (v3 kaynakli)
      bir varyant Adim 13'e eklenir.

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, safe_print  # noqa: E402


AVERAGED_FEATURE_CANDIDATES = [
    ("AL_88", "AL_121", "ZORT_AL88_AL121"),
    ("EK_7", "EK_9", "ZORT_EK7_EK9"),
    ("AL_23", "AL_283", "ZORT_AL23_AL283"),
    ("AL_7", "AL_103", "ZORT_AL7_AL103"),
]


def add_zscore_and_averaged_features(X_train, X_test, pairs=AVERAGED_FEATURE_CANDIDATES):
    """FOLD-GUVENLI: z-skor istatistikleri (mean/std) YALNIZCA X_train'den
    ogrenilir, X_test'e yalnizca uygulanir (transform). Adim 13 tarafindan
    import edilip cagrilir -- bu script'te CALISTIRILMAZ (nested disi
    calistirmak sizinti olurdu).

    Doner: (X_train_ext, X_test_ext) -- her cift icin ZORT_X_Y eklenmis.
    """
    X_train_ext, X_test_ext = X_train.copy(), X_test.copy()
    for c1, c2, name in pairs:
        if c1 not in X_train.columns or c2 not in X_train.columns:
            continue
        mean1, std1 = X_train[c1].mean(), X_train[c1].std()
        mean2, std2 = X_train[c2].mean(), X_train[c2].std()
        std1 = std1 if std1 > 1e-9 else 1.0
        std2 = std2 if std2 > 1e-9 else 1.0
        z1_train = (X_train[c1] - mean1) / std1
        z2_train = (X_train[c2] - mean2) / std2
        X_train_ext[name] = (z1_train + z2_train) / 2
        z1_test = (X_test[c1] - mean1) / std1  # TRAIN istatistikleriyle
        z2_test = (X_test[c2] - mean2) / std2
        X_test_ext[name] = (z1_test + z2_test) / 2
    return X_train_ext, X_test_ext


def main():
    report = f"""# Adım 11-12 — Normalizasyon/Ortalama-Özellik ve Log Dönüşüm Adayları

## Adım 11 — Ortalama-Özellik Adayları (Adım 13'te fold-güvenli test edilecek)

Adım 10'un 4 güçlü çiftinin tamamı ("Pearson+Spearman birlikte yüksek,
ilişki lineer, ortak-n yeterli" kriterlerini karşılıyorlar — Adım 10'da
doğrulandı) `ZORT_X_Y = (Z(X)+Z(Y))/2` adayı olarak işaretlendi:

{chr(10).join(f"- `{name}` = ortalama(Z({c1}), Z({c2}))" for c1, c2, name in AVERAGED_FEATURE_CANDIDATES)}

**Fold-güvenlik:** Z-skor istatistikleri (ortalama/std) yalnızca outer-
train'den öğrenilir (`add_zscore_and_averaged_features()` fonksiyonu,
bu script'te tanımlı, Adım 13 tarafından import edilir) — bu script'in
kendisi hiçbir hesaplama/deneme YAPMAZ, çünkü split bankasının dışında
normalizasyon "denemek" sızıntı riski taşırdı.

**Nihai karar kuralı (görev metninden):** Bu adaylar yalnızca Adım 13'ün
nested CV'sinde performans **iyileşmesi** gösterirse ana modele aday
gösterilir — otomatik olarak resmi havuza eklenmez, yalnızca bu deney
kapsamında.

## Adım 12 — Log Dönüşüm: Kapsam Boş (dürüstçe raporlanıyor)

Adım 10, test edilen 4 güçlü çiftin **hiçbirinde üstel ilişki bulmadı**
(lineer form her çiftte açık farkla kazandı — R² 0.71-0.98, üstel/kuvvet
formları bazı çiftlerde katastrofik negatif R² verdi). **Bu yüzden Adım
12'nin "Adım 10'da üstel ilişki bulunan spesifik çiftler" koşulu bu turda
karşılanmıyor — kapsam boş, zorla bir dönüşüm uygulanmadı.**

Bunun yerine, GENOVA'nın `04_PREPROCESSING_MERDIVENI_PAH.md`'sindeki GENEL
log1p bulgusuyla (M-3 basamağı) çelişip çelişmediğini kontrol etmek için,
Adım 13'e **log1p-dönüşümlü bir varyant** (v3 kaynaklı, `log1p`/`logit`
zaten uygulanmış) ek karşılaştırma kolu olarak eklendi — bu, spesifik bir
çift için değil, GENEL dönüşüm etkisini bu model ailesinde de doğrulamak
içindir.
"""
    (RESULTS_DIR / "step11_12_transform_candidates_report.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
