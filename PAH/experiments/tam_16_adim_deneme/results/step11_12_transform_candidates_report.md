# Adım 11-12 — Normalizasyon/Ortalama-Özellik ve Log Dönüşüm Adayları

## Adım 11 — Ortalama-Özellik Adayları (Adım 13'te fold-güvenli test edilecek)

Adım 10'un 4 güçlü çiftinin tamamı ("Pearson+Spearman birlikte yüksek,
ilişki lineer, ortak-n yeterli" kriterlerini karşılıyorlar — Adım 10'da
doğrulandı) `ZORT_X_Y = (Z(X)+Z(Y))/2` adayı olarak işaretlendi:

- `ZORT_AL88_AL121` = ortalama(Z(AL_88), Z(AL_121))
- `ZORT_EK7_EK9` = ortalama(Z(EK_7), Z(EK_9))
- `ZORT_AL23_AL283` = ortalama(Z(AL_23), Z(AL_283))
- `ZORT_AL7_AL103` = ortalama(Z(AL_7), Z(AL_103))

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
