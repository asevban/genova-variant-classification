# Yeni Model Geliştirme Serisi — Sonuçlar

Kısıtlar: Yeni özellik eklenmedi, TabPFN kullanılmadı, 319 özellik ve veri etiketleri değiştirilmedi.

## Referans

`ID3 + CatBoost + Random Forest`, eşit soft voting: MCC `0.5225`, projeksiyon F1 `0.6798`, FP/100 benign `8.57`.

## 1. Varyant hata denetimi

- 64 varyant 5/5 doğru.
- 24 varyant aralıklı hata verdi.
- 11 varyant 3/5 veya 4/5 hata verdi.
- 12 patojenik varyant 5/5 FN oldu.
- 5/5 FP olan benign varyant yoktur.

Bu kayıtlar silinmedi; etiket ve annotation kaynağı için manuel doğrulama adaylarıdır.

## 2. Leave-One-Variant-Out etki analizi

- Yalnız `VAR_001930` çıkarıldığında ana–yedek F1 sıralaması tersine döndü.
- Bu varyant çıkarıldığında ana model projeksiyon F1'i `+0.0611` değişti.
- Sonuç küçük benign örneklemin tek varyanta duyarlı olduğunu gösterdi; varyant silinmedi.

## 3. Cross-fitted bagging

Her varyant için onu eğitimde görmeyen beş dış-fold modelinin tahminleri ortalandı.

| Yöntem | MCC | Proj. F1 | Recall | Specificity | FP/100 |
|---|---:|---:|---:|---:|---:|
| Pooled referans 5×5 | 0.5225 | 0.6798 | 0.7356 | 0.9143 | 8.57 |
| Olasılık ortalaması | **0.5717** | **0.7580** | **0.7556** | **0.9524** | **4.76** |
| 3/5 çoğunluk oyu | **0.5717** | **0.7580** | **0.7556** | **0.9524** | **4.76** |
| Muhafazakâr 4/5 oy | 0.5171 | 0.7224 | 0.7000 | **0.9524** | **4.76** |

Bu seri içindeki tek açık performans iyileşmesi cross-fitted bagging oldu. Pooled sonuç ile tek-varyant bagged sonuç farklı toplama biçimleri olduğundan sonuç güçlü aday kanıtıdır; bağımsız test garantisi değildir. Yarışma tahmininde uygulanması için fold modellerinin saklanıp test olasılıklarının ortalanması gerekir.

## 4. Kararlılık cezalı eşik

| Ceza | MCC | Proj. F1 | FP/100 |
|---|---:|---:|---:|
| λ=0.25 | 0.5000 | 0.6201 | 13.33 |
| λ=0.50 | 0.4883 | 0.6391 | 10.48 |
| λ=1.00 | 0.4767 | 0.6405 | 9.52 |

Referansı geçmediği için reddedildi.

## 5. Özellik kararlılığı

- 14 özellik en az 20/25 fold'da ilk 20'ye girdi.
- 34 özellik en az 20/25 fold'da ilk 50'ye girdi.
- 205 özellik hiçbir fold'da ilk 50'ye girmedi.
- En kararlı üç özellik: `CAT_1`, `AA_2`, `AA_1`.

Bu analiz risk denetimidir. Etkileşimli katkılar tek-değişkenli IG'de görünmeyebileceği için kolon silinmedi.

## Güncel karar

Model bileşenleri değişmedi: `ID3 + CatBoost + Random Forest`. Yeni güçlü aday, bu bileşenlerin farklı fold eğitimlerinden gelen tahminlerini ortalayan **cross-fitted bagging tahmin stratejisidir**. Etiketler, 319 özellik ve baseline güvenlik modeli korunmuştur.

