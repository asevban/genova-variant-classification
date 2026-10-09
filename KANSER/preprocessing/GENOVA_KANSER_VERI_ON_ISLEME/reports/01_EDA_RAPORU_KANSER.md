# 01 — KANSER EDA Raporu

## Yapı

- 388 satır, 353 sütun: `Variant_ID`, `Label` ve 351 model özelliği.
- Özellik grupları: 334 `AL`, 6 `CAT`, 9 `EK`, 2 `AA`.
- Sınıflar: Label 0 = 120, Label 1 = 268.
- 343 sayısal, 8 kategorik özellik; sonsuz değer veya tanımlı sentinel sayı görülmedi.

## Veri kalitesi

- Anlamsal eksik hücre oranı `%57,56`.
- Ham veride 69 sabit sütun.
- Beş birebir kopya grubunda referans dışındaki 61 sütun yineleniyor.
- 19 sütunda eksiklik oranı en az `%85`.
- Üç satır aynı özellik değerlerini paylaşan tek bir kopya grubunda.

Tam listeler `reports/tables/column_audit.csv`, `exact_duplicate_column_groups.csv`, `near_constant_scan.csv` ve `outlier_iqr_scan.csv` dosyalarındadır.

## Eksiklik ve provenance

Label 0 satırlarında ortalama eksiklik `%32,99`, Label 1'de `%68,57` bulundu. `CAT_1` eksikliği iki sınıfta sırasıyla `%8,33` ve `%58,21`; `CAT_2` için `%35,83` ve `%76,87` düzeyindedir. Satır eksiklik oranının yön düzeltilmiş tek değişkenli ROC-AUC değeri `0,768`'dir.

Bu fark, eksiklik bilgisinin tahmine katkı sağlayabileceğini gösterir; aynı zamanda laboratuvar, kaynak veya veri toplama sürecini kestiren istenmeyen bir kısa yol riski oluşturur. Bu nedenle missingness özellikleri yalnızca A2/A4 ablation deneylerinde değerlendirilir.

## EDA kararı

EDA aşamasında satır/sütun silme, imputasyon, outlier kırpma veya hedefe göre seçim yapılmadı. Yapısal azaltmalar V2/V3 politikaları içinde tanımlandı; resmî CV'de training fold'una taşındı.
