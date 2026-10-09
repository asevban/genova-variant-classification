# Takım Arkadaşı Sunumundan Yöntem Aktarımı

Kaynak sunum: `CFTR_Modelleme_Sunumu.pptx` (18 slayt). Sunumdaki skorlar ve veri dönüşümleri bizim sonuçlarımız gibi kabul edilmemiş; yalnız yöntem fikirleri çıkarılmıştır. Bütün yeni deneylerde bizim 319 özellikli veri hattımız ve mevcut nested CV split düzenimiz korunmaktadır.

## Sunumda görülen yöntemler

- Aynı split bank ile tekrarlı stratified CV
- Fold-local preprocessing ve nested threshold
- Final patojenik prior'ına göre F1/MCC okuması
- Düzenlileştirilmiş Logistic Regression, RandomForest, XGBoost ve LightGBM
- Native eksik değer yaklaşımı
- OOF skor korelasyonu ve hibrit reçeteler
- Bootstrap güven aralıkları, fold fragility ve threshold sensitivity
- Basit guardrail model ile daha karmaşık ana modelin birlikte saklanması

## Bizde zaten uygulanmış olanlar

Nested threshold, yaklaşık `20/120` final prior projeksiyonu, bootstrap güven aralıkları, aynı fold karşılaştırması, CatBoost/ExtraTrees/AdaBoost, hard/soft/weighted voting, iki aşamalı hibrit, stacking ve çeşitli stabilite kontrolleri daha önce uygulanmıştır. Bunlar sunumdan yeniden kopyalanmamıştır.

## Yeni deney 1 - Düzenlileştirilmiş Logistic Regression

319 özellik sabit tutularak L1/L2 düzenlileştirme, `C` değerleri ve sınıf ağırlıkları dış eğitim fold'undaki iç CV ile seçildi. Robust ölçekleme, imputasyon ve kategorik one-hot encoding yalnız ilgili eğitim fold'unda öğrenildi.

| Model | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6757 | 0.2857 | 0.7667 | 0.5216 | 0.0478 | 0.2872 | 71.4 |
| Mevcut ensemble | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |

**Karar:** Logistic Regression bizim veri setinde benign sınıfı ayıramadı ve reddedildi. Takım arkadaşının veri setindeki LR başarısı bizim farklı özellik/gözlem yapımıza genellenmemektedir.

Ham sonuçlar: `logistic_319_nested_score.csv`, `logistic_319_nested_folds.csv`, `logistic_319_nested_predictions.csv`, `logistic_319_nested_inner_predictions.csv` ve `logistic_319_nested_choices.csv`.

## Yeni deney 2 - Düzenlileştirilmiş RandomForest

Sunumdaki regularized tree yaklaşımından hareketle `max_depth`, `min_samples_leaf` ve `max_features` için beş kontrollü yapı denendi. `class_weight=balanced_subsample` kullanıldı. Parametre ve karar eşiği yalnız iç CV'de seçildi; tüm yapılarda aynı 319 özellik kullanıldı.

| RandomForest | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tek model | 0.7351 | 0.8381 | 0.7111 | 0.6791 | 0.4404 | 0.5642 | 16.2 |

Tek başına nihai ensemble'ı geçmedi; ancak farklı hata profili nedeniyle küçük hibrit katkısı ayrıca denendi.

## Yeni deney 3 - RandomForest destekli hibrit

| Ensemble | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eski: ID3 + CatBoost + AdaBoost | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |
| Yeni: ID3 + CatBoost + RandomForest | **0.7694** | **0.9143** | **0.7356** | **0.7190** | **0.5225** | **0.6798** | **8.6** |

RandomForest, AdaBoost'un yerine eşit ağırlıkla kullanıldığında FP değişmeden recall, F1 ve MCC yükseldi. Dört-model birleşimi ve küçük sabit RF ağırlıkları daha kötü sonuç verdi.

## Stage 7 benzeri stabilite denetimi

Yeni ve eski model aynı orijinal gözlemlerle 4.000 paired cluster bootstrap örnekleminde karşılaştırıldı:

| Fark (Yeni - Eski) | Ortalama | %95 GA | Yeni modelin üstünlük olasılığı |
|---|---:|---:|---:|
| MCC | +0.0266 | -0.0231 – +0.0745 | %86.4 |
| Proj. F1 | +0.0156 | -0.0723 – +0.0933 | %65.5 |
| Recall | +0.0271 | -0.0001 – +0.0556 | %96.7 |
| FP / 100 benign | +0.0024 | -4.7619 – +5.7143 | - |

Güven aralıkları MCC ve F1 için sıfırı içerdiğinden üstünlük kesinleşmiş değildir. Fold kırılganlığında MCC<0.40 fold sayısı `5 → 4`, minimum projekte F1 `0.3317 → 0.3846`, minimum recall `0.4444 → 0.5000` ve minimum specificity `0.25 → 0.50` iyileşti; minimum MCC ise `0.2609 → 0.2357` ile hafif kötüleşti.

Eşik hassasiyetinde fold-içi seçilen eşiklerin tam noktası yeni modelde F1 `0.6798` ve FP `8.6` verdi. Dış sonuçlarda eşiklere `+0.02` eklenmesi F1 `0.6859`, FP `5.7` üretse de bu düzeltme dış-fold sonuçları görülerek fark edildiği için tarafsız nihai seçim olarak kullanılamaz; yalnız bağımsız test için önceden kaydedilmiş ikincil politika olabilir.

**Karar:** `ID3 + CatBoost + RandomForest` eşit soft voting yeni ana adaydır. Eski `ID3 + CatBoost + AdaBoost` ensemble daha önce kilitlenmiş guardrail/yedek olarak korunur. Bağımsız test olmadan yeni adayın kesin üstün olduğu iddia edilmez.

Ham sonuçlar: `randomforest_319_nested_*`, `randomforest_319_hybrid_*`, `rf_hybrid_319_paired_bootstrap*`, `rf_hybrid_319_fold_fragility.csv` ve `rf_hybrid_319_threshold_sensitivity.csv`.

## Sunumdan alınıp henüz uygulanmayanlar

- LightGBM ve XGBoost paketleri mevcut ortamda kurulu değildir. Sunumda tek başlarına specificity/MCC zayıf kaldıkları, LightGBM yalnız tamamlayıcı hibrit sinyal verdiği için yeni bağımlılık eklemeden önce RandomForest deneyi önceliklendirilmiştir.
- Native-NaN compact preprocessing takım arkadaşının farklı kolon yapısına aittir; bizim veri hattımıza doğrudan kopyalanmamıştır.
- Takım arkadaşının PREP-05/PDRR-08 kolon reçeteleri bizim veri setine uygulanmamış ve 319 özellikli dosya değiştirilmemiştir.
