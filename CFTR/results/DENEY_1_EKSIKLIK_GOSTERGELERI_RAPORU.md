# Deney 1 — Fold-İçi Eksiklik Göstergeleri

## Amaç

319 özellikli ana veri hattında bir değerin eksik olmasının ayrıca bilgi taşıyıp taşımadığı test edildi. Ana kolonlar korunarak uygun kolonlar için `MISS_<kolon>` biçiminde ikili göstergeler üretildi.

## Veri sızıntısını önleme

- Göstergeler her eğitim fold'unda yeniden belirlendi.
- Bir göstergenin üretilebilmesi için eğitim fold'unda en az 5 eksik ve 5 dolu gözlem şartı uygulandı.
- Aynı eksiklik desenine sahip kolonlardan yalnız bir gösterge tutuldu.
- Dış test fold'u gösterge seçiminde, ön işlemede veya eşik seçiminde kullanılmadı.
- Model ve eşik değerlendirmesi 5-fold × 5 tekrar nested CV ile yapıldı.

Tam veri üzerinde bu kuralla 16 benzersiz gösterge oluştu; şema 319'dan 335 özelliğe çıktı. Fold içindeki uygun gösterge sayısı yalnız ilgili eğitim verisine göre değişebilir.

## Tek model sonuçları

| Model | Durum | Macro-F1 | MCC | Proj. F1 | Recall | Specificity | FP/100 benign |
|---|---|---:|---:|---:|---:|---:|---:|
| ID3 | Baseline | **0.6459** | **0.4260** | **0.6034** | 0.6378 | **0.9048** | **9.52** |
| ID3 | Eksiklik göstergeli | 0.6389 | 0.3843 | 0.5231 | **0.6578** | 0.8286 | 17.14 |
| CatBoost | Baseline | **0.6539** | **0.4074** | **0.5437** | 0.6756 | **0.8381** | **16.19** |
| CatBoost | Eksiklik göstergeli | 0.6445 | 0.3801 | 0.5062 | **0.6778** | 0.8000 | 20.00 |
| Random Forest | Baseline | **0.6791** | **0.4404** | 0.5642 | **0.7111** | 0.8381 | 16.19 |
| Random Forest | Eksiklik göstergeli | 0.6609 | 0.4361 | **0.5957** | 0.6667 | **0.8857** | **11.43** |

Random Forest'ta FP azalmış ve projeksiyon F1 yükselmiştir; ancak recall ve Macro-F1 düşmüştür. ID3 ve CatBoost'ta göstergeler genel olarak zararlı olmuştur.

## Ana ensemble karşılaştırması

Her iki durumda da aynı yapı kullanıldı:

`ID3 + CatBoost + Random Forest — eşit ağırlıklı soft voting`

| Ölçüt | Baseline 319 | Eksiklik göstergeli 319 | Değişim |
|---|---:|---:|---:|
| Accuracy | **0.7694** | 0.7405 | −0.0288 |
| Specificity | **0.9143** | 0.8762 | −0.0381 |
| Recall | **0.7356** | 0.7089 | −0.0267 |
| Macro-F1 | **0.7190** | 0.6884 | −0.0306 |
| MCC | **0.5225** | 0.4674 | −0.0550 |
| Projeksiyon F1 | **0.6798** | 0.6090 | −0.0707 |
| FP / 100 benign | **8.57** | 12.38 | +3.81 |

## Karar

Eksiklik göstergeleri ana ensemble'ın bütün temel başarı ölçütlerini düşürmüş ve 100 benign başına FP'yi 8.57'den 12.38'e yükseltmiştir. Bu nedenle deney **başarısız** kabul edilmiş ve göstergeler ana 319 özellikli veri hattına eklenmemiştir.

Random Forest'taki tek-model FP azalması ilginç olsa da üçlü ensemble güvenlik kapısını geçmemiştir. Güncel ana model ve veri seti değişmemiştir.

## Kanıt dosyaları

- `missing_indicator_ensemble_scores.csv`
- `missing_indicator_ensemble_folds.csv`
- `missing_indicator_ensemble_predictions.csv`
- `id3_benign_weighted_322_vs_320_scores_missing_indicators.csv`
- `catboost_benign_weighted_finalists_scores_missing_indicators.csv`
- `randomforest_319_missing_indicators_nested_score.csv`

