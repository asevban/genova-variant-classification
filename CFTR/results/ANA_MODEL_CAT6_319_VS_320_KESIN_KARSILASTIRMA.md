# Ana Modelde CAT_6 Kontrolü: 319 ve 320 Özellik

## Deney tasarımı

`CAT_6` kolonunun ana modele etkisini izole etmek için iki tarafta da aynı model kullanılmıştır:

`ID3 + CatBoost + Random Forest — eşit ağırlıklı soft voting`

- Aynı 5-fold × 5 tekrar dış çapraz doğrulama bölünmeleri kullanıldı.
- Her dış fold için karar eşiği yalnız 4-fold iç CV tahminlerinden seçildi.
- Hedef, 100 benign + 20 pathogenic test dağılımındaki projeksiyon F1'i yükseltmek; eşitlikte FP'yi azaltmaktı.
- 320 özellikte `CAT_6` vardır; 319 özellikte yalnız `CAT_6` çıkarılmıştır.

## Sonuçlar

| Ölçüt | 319 özellik (`CAT_6` yok) | 320 özellik (`CAT_6` var) | 319 − 320 |
|---|---:|---:|---:|
| Accuracy | **0.7694** | 0.7387 | +0.0306 |
| Specificity | **0.9143** | 0.8952 | +0.0190 |
| Recall | **0.7356** | 0.7022 | +0.0333 |
| Macro-F1 | **0.7190** | 0.6890 | +0.0300 |
| MCC | **0.5225** | 0.4756 | +0.0468 |
| Projeksiyon F1 | **0.6798** | 0.6309 | +0.0489 |
| FP / 100 benign | **8.57** | 10.48 | −1.90 |
| Ortalama eşik | 0.7224 | 0.7216 | +0.0008 |

Birleştirilmiş 25 dış fold karmaşıklık sayıları:

- 319 özellik: TN=96, FP=9, FN=119, TP=331
- 320 özellik: TN=94, FP=11, FN=134, TP=316

## Karar

Aynı ana model ve aynı doğrulama düzeninde `CAT_6` çıkarılmış 319 özellikli yapı bütün temel performans ölçütlerinde daha iyidir. `CAT_6` tutulduğunda FP artmış, recall düşmüş ve MCC/projeksiyon F1 gerilemiştir. Bu nedenle ana `ID3 + CatBoost + Random Forest` ensemble için **319 özellikli, CAT_6 içermeyen veri hattı korunmalıdır**.

TabPFN'nin 320 özellikte daha iyi sonuç vermesi model-özel bir etkidir ve ana ensemble kararına taşınmamalıdır.

## Kanıt dosyaları

- `main_ensemble_319_vs_320_scores.csv`
- `main_ensemble_319_vs_320_folds.csv`
- `main_ensemble_319_vs_320_predictions.csv`
- `randomforest_320_nested_score.csv`
- `randomforest_320_nested_folds.csv`
- `randomforest_320_nested_predictions.csv`
- `randomforest_320_nested_inner_predictions.csv`
- `randomforest_320_nested_choices.csv`

