# 319 Özellikli Modeli Geliştirme — Sıralı Deney Günlüğü

Bu raporda her değişiklik, **aynı 319 giriş özelliğini kullanan ID3 + CatBoost + AdaBoost eşit ağırlıklı soft-voting ensemble** üzerinde ve dış test fold'una bilgi sızdırmadan değerlendirilmiştir. Veri 5-fold stratified çapraz doğrulama × 5 tekrar ile ölçülmüş; kararlar yalnız ilgili dış fold'un eğitim bölümündeki 4-fold iç CV sonuçlarından verilmiştir.

## Başlangıç modeli

| Özellik | Değer |
|---|---:|
| Özellik sayısı | 319 |
| Accuracy | 0.7477 |
| Specificity | 0.9143 |
| Recall | 0.7089 |
| Macro-F1 | 0.6992 |
| MCC | 0.4964 |
| Benign-ağırlıklı projekte F1 | 0.6633 |
| 100 benign içindeki tahmini FP | 8.6 |

## 1. Sağlamlık ve belirsizlik kontrolü

Amaç, tek bir ortalama skor yerine küçük veri setinin oluşturduğu belirsizliği görmekti. Aynı hastaya/gözleme ait beş tekrarlı tahminler birlikte tutularak gözlem düzeyinde 2.000 kez stratified cluster bootstrap yapıldı.

| Ölçüt | Nokta tahmini | %95 güven aralığı | 5 tekrar aralığı |
|---|---:|---:|---:|
| Accuracy | 0.7477 | 0.6847–0.8090 | 0.7117–0.7838 |
| Specificity | 0.9143 | 0.8095–0.9810 | 0.8571–0.9524 |
| Recall | 0.7089 | 0.6311–0.7844 | 0.6667–0.7667 |
| Macro-F1 | 0.6992 | 0.6382–0.7602 | 0.6608–0.7366 |
| MCC | 0.4964 | 0.3915–0.5932 | 0.4240–0.5602 |
| Proj. F1 | 0.6633 | 0.5185–0.8058 | 0.5667–0.7510 |
| FP / 100 benign | 8.6 | 1.9–19.0 | 4.8–14.3 |

**Karar:** Ortalama performans kullanılabilir düzeydedir; fakat yalnız 111 bağımsız gözlem bulunduğu için güven aralıkları geniştir. Bu, tek başına overfitting kanıtı değildir; sonucun örnekleme duyarlı olduğunu gösterir.

## 2. Fold-içi olasılık kalibrasyonu

Platt ve izotonik kalibrasyon yalnız dış eğitim fold'undaki iç-CV tahminleriyle öğrenildi. Dış test fold'u kalibrasyon veya eşik seçiminde kullanılmadı.

| Yöntem | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kalibrasyonsuz | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |
| Platt | 0.7658 | 0.8667 | 0.7422 | 0.7102 | 0.4931 | 0.6162 | 13.3 |
| İzotonik | 0.6847 | 0.8667 | 0.6422 | 0.6387 | 0.4003 | 0.5563 | 13.3 |

**Karar:** Platt accuracy ve recall'u yükseltse de benign yanlış alarmını artırdı; MCC ve benign-ağırlıklı F1'i iyileştirmedi. İzotonik de geriledi. Bu nedenle **kalibrasyonsuz ensemble olasılıkları korunmuştur**.

## 3. Karar eşiği stratejileri

Her eşik dış eğitim fold'undaki iç-CV tahminlerinden seçildi. Böylece dış test sonuçlarına bakarak eşik ayarlanmadı.

| Eşik hedefi | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Benign-ağırlıklı projekte F1 | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |
| Specificity ≥ 0.90 | 0.6793 | 0.9143 | 0.6244 | 0.6392 | 0.4224 | 0.6083 | 8.6 |
| Specificity ≥ 0.95 | 0.5495 | 0.9714 | 0.4511 | 0.5341 | 0.3425 | 0.5660 | 2.9 |
| En yüksek MCC | 0.8378 | 0.6762 | 0.8756 | 0.7548 | 0.5144 | 0.5011 | 32.4 |
| En yüksek Macro-F1 | 0.8360 | 0.6762 | 0.8733 | 0.7528 | 0.5109 | 0.5001 | 32.4 |
| Sabit 0.50 | 0.8649 | 0.4667 | 0.9578 | 0.7432 | 0.5070 | 0.4142 | 53.3 |

**Karar:** MCC odaklı eşik MCC'yi 0.4964'ten 0.5144'e çıkarsa da 100 benign içindeki FP'yi 8.6'dan 32.4'e yükseltmektedir. Test verisinin benign ağırlıklı olacağı bilgisi nedeniyle **projekte-F1 eşiği korunmuştur**. Çok düşük FP istenirse specificity ≥ 0.95 seçilebilir; bunun bedeli recall'un 0.4511'e düşmesidir.

## Güncel sonuç

İlk üç kontrollü deney sonunda başlangıç modelini değiştirecek, ana hedefte güvenilir bir iyileşme bulunmamıştır. Güncel tercih hâlâ 319 özellikli ID3 + CatBoost + AdaBoost eşit soft voting, kalibrasyonsuz olasılık ve fold-içi benign-ağırlıklı eşiktir: **MCC 0.4964, projekte F1 0.6633 ve FP/100 benign 8.6**.

Ham sonuç dosyaları: `final_319_robustness_summary.csv`, `final_319_calibration_scores.csv` ve `final_319_threshold_strategy_scores.csv`.

## 4. Fold-içi Information Gain ile ortak özellik sayısı seçimi

Amaç 319 özelliği daha da azaltırken dış test fold'una bilgi sızdırmamaktı. Her dış eğitim fold'unda özellikler yalnız eğitim verisinden hesaplanan Information Gain ile sıralandı. `100, 150, 200, 250, 319` adayları 4-fold iç CV'de benign-ağırlıklı projekte F1'e göre karşılaştırıldı. Seçilen kolonlar dış test fold'una uygulanmadan önce yalnız dış eğitim verisiyle yeniden sıralandı.

Bu aşamada önce bir güvenlik kapısı olarak CatBoost kullanıldı. Seçim CatBoost'un kendi bağımsız dış-fold sonucunu dahi iyileştirmezse aynı zayıf seçimi ID3 ve AdaBoost'a zorla uygulamak doğru olmayacaktı.

| CatBoost deneyi | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fold-içi IG ile değişken özellik sayısı | 0.7135 | 0.8476 | 0.6822 | 0.6612 | 0.4207 | 0.5583 | 15.2 |

İç CV'nin seçtiği özellik sayıları 25 dış fold içinde kararlı değildi: 100 özellik 5 kez, 150 özellik 5 kez, 200 özellik 6 kez, 250 özellik 4 kez ve 319 özellik 5 kez seçildi. Ortalama 201.8 özellik kullanılmasına rağmen güvenilir tek bir azaltma noktası oluşmadı.

**Karar:** IG tabanlı ek azaltma CatBoost'un dış-fold performansını düşürdü ve FP'yi artırdı. Bu nedenle bu kolon listeleri üç modele aktarılmadı. Üç modelin tamamında aynı **319 özellik** korunmuştur. Bu, başarısız deney sonucunu saklamak değil; bağımsız dış-fold güvenlik kapısında elenen dönüşümü üretim adayına taşımamaktır.

Ham sonuçlar: `nested_ig_feature_count_catboost_score.csv`, `nested_ig_feature_count_catboost_folds.csv`, `nested_ig_feature_count_catboost_predictions.csv` ve `nested_ig_common_feature_selections.json`.

## 5. Hiperparametre ayarı — CatBoost

319 ortak özellik sabit tutuldu. Baseline ile birlikte daha sığ, daha güçlü düzenlileştirilmiş ve daha yavaş öğrenen toplam beş kontrollü yapı her dış eğitim fold'undaki 4-fold iç CV'de karşılaştırıldı. Dış test fold'u parametre veya eşik seçiminde kullanılmadı.

| CatBoost | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sabit baseline | 0.7063 | 0.8381 | 0.6756 | 0.6539 | 0.4074 | 0.5437 | 16.2 |
| Nested CV ile ayarlanmış | 0.6919 | 0.8762 | 0.6489 | 0.6459 | 0.4133 | 0.5722 | 12.4 |

**Yorum:** Ayar accuracy ve recall'u bir miktar düşürdü; buna karşılık specificity'yi 0.8381'den 0.8762'ye çıkardı, FP/100 benign değerini 16.2'den 12.4'e indirdi ve MCC'yi 0.4074'ten 0.4133'e hafif yükseltti. Benign-ağırlıklı hedef bakımından küçük ama olumlu bir değişimdir.

Seçimler tam kararlı değildir: `depth4_more` 9, `shallow_reg` 8, `depth5_reg` 4, `depth4_slow` 3 ve baseline 1 dış fold'da seçildi. Bu nedenle sonuç güçlü bir tek-parametre kazancı değil, fold'a göre düzenlileştirme tercihidir.

Ham sonuçlar: `catboost_319_tuned_nested_score.csv`, `catboost_319_tuned_nested_folds.csv`, `catboost_319_tuned_nested_predictions.csv`, `catboost_319_tuned_nested_inner_predictions.csv` ve `catboost_319_tuned_nested_choices.csv`.

### AdaBoost

319 ortak özellik korunarak `n_estimators` ve `learning_rate` için beş kontrollü yapı nested CV içinde karşılaştırıldı.

| AdaBoost | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sabit baseline | 0.6270 | 0.7905 | 0.5889 | 0.5821 | 0.2973 | 0.4467 | 21.0 |
| Nested CV ile ayarlanmış | 0.6414 | 0.8190 | 0.6000 | 0.5972 | 0.3285 | 0.4791 | 18.1 |

**Karar:** Ayarlanmış AdaBoost kendi baseline'ına göre bütün temel ölçütlerde iyileşti. FP/100 benign 21.0'dan 18.1'e düştüğü, MCC 0.2973'ten 0.3285'e ve projekte F1 0.4467'den 0.4791'e çıktığı için ayarlı sürüm sonraki ensemble denemesine aday olarak kabul edildi. Tek başına ensemble'dan zayıf olması, ensemble içindeki tamamlayıcı katkısının ayrıca ölçülmesi gerektiği anlamına gelir.

Parametre seçimi yine tam kararlı değildir: baseline 8, `lr050_n050` 7, `lr010_n200` 5, `lr050_n100` 3 ve `lr025_n150` 2 fold'da seçildi.

Ham sonuçlar: `adaboost_319_tuned_nested_score.csv`, `adaboost_319_tuned_nested_folds.csv`, `adaboost_319_tuned_nested_predictions.csv`, `adaboost_319_tuned_nested_inner_predictions.csv` ve `adaboost_319_tuned_nested_choices.csv`.

### ID3 / 21 ağaçlı hard voting

Özel ID3 uygulaması korunarak baseline, daha sığ, daha derin-düzenlileştirilmiş, daha küçük yapraklı ve ağaç başına daha fazla özellik değerlendiren beş yapı denendi. Dış fold başına yapı ve oy eşiği yalnız iç-CV tahminlerinden seçildi. Özellik sayısı bütün adaylarda 319 olarak kaldı.

| ID3 | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sabit baseline | 0.6883 | 0.9048 | 0.6378 | 0.6459 | 0.4260 | 0.6034 | 9.5 |
| Nested CV ile ayarlanmış | 0.6937 | 0.8857 | 0.6489 | 0.6485 | 0.4207 | 0.5845 | 11.4 |

**Karar:** Ayarlı ID3 accuracy ve recall'u çok az artırdı; fakat benign-ağırlıklı ana hedeflerde geriledi. FP/100 benign 9.5'ten 11.4'e yükseldi, projekte F1 0.6034'ten 0.5845'e ve MCC 0.4260'dan 0.4207'ye düştü. Bu nedenle **ayarlı ID3 reddedildi ve baseline ID3 korundu**.

İç-CV seçimi küçük yapraklı yapıyı 10, derin-düzenlileştirilmiş yapıyı 5, baseline'ı 4, daha fazla özellikli yapıyı 3 ve sığ yapıyı 3 fold'da seçti. Seçim kararlı tek bir ayarda birleşmedi.

Ham sonuçlar: `id3_319_tuned_nested_score.csv`, `id3_319_tuned_nested_folds.csv`, `id3_319_tuned_nested_predictions.csv` ve `id3_319_tuned_nested_inner_predictions.csv`.

## 6. Ayarlanmış modellerle eşit ve ağırlıklı soft voting

Baseline ID3, ayarlanmış CatBoost ve ayarlanmış AdaBoost aynı 319 özellik ve aynı dış fold'lar üzerinde birleştirildi. Sabit eşit/ağırlıklı üçlüler, iki-model birleşimleri ve iç CV'de dinamik seçilen ağırlıklar karşılaştırıldı. Her kombinasyonun karar eşiği yalnız ilgili dış eğitim fold'undaki iç-CV tahminlerinden belirlendi.

| Kombinasyon (ID3:CatBoost:AdaBoost) | Accuracy | Specificity | Recall | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|
| Önceki baseline ensemble (1:1:1) | 0.7477 | 0.9143 | 0.7089 | **0.4964** | **0.6633** | **8.6** |
| Ayarlı modeller, AdaBoost ağırlıklı (1:1:2) | 0.7315 | 0.9143 | 0.6889 | 0.4779 | 0.6507 | 8.6 |
| Ayarlı ID3 + CatBoost (1:1:0) | 0.7477 | 0.8952 | 0.7133 | 0.4862 | 0.6377 | 10.5 |
| Ayarlı modeller eşit (1:1:1) | 0.7387 | 0.8857 | 0.7044 | 0.4705 | 0.6191 | 11.4 |
| İç CV ile dinamik ağırlık | 0.7532 | 0.8571 | 0.7289 | 0.4724 | 0.5967 | 14.3 |

**Karar:** Ayarlı modeller arasında benign-ağırlıklı en iyi sabit birleşim `(1:1:2)` oldu. FP'yi 8.6'da tutmasına rağmen MCC ve projekte F1 mevcut baseline ensemble'dan düşük kaldı. Dinamik ağırlıklar da fold'lar arasında çok değişti ve FP'yi 14.3'e yükseltti. Bu nedenle **ayarlanmış ensemble reddedildi; önceki baseline ID3 + baseline CatBoost + baseline AdaBoost eşit soft voting nihai aday olarak korundu**.

Bu sonuç önemli bir noktayı gösterir: Bir temel modelin tek başına iyileşmesi, ensemble'ın mutlaka iyileşeceği anlamına gelmez. Ayar, modellerin hata çeşitliliğini azaltmış veya olasılık ölçeklerini eşit ortalama için daha az uyumlu hâle getirmiş olabilir.

Ham sonuçlar: `tuned_319_weighted_ensemble_scores.csv`, `tuned_319_weighted_ensemble_folds.csv` ve `tuned_319_weighted_ensemble_predictions.csv`.

## 7. İki aşamalı benign-odaklı hibrit model

Birinci aşamada modellerin yüksek güvenle benign bulduğu örnekleri doğrudan benign olarak ayıran bir güven kapısı kuruldu. Kapıdan geçmeyen kararsız örnekler ikinci aşamada baseline ID3 + CatBoost + AdaBoost eşit soft-voting ensemble ile değerlendirildi. Benign olasılık sınırı, gereken model uzlaşması (`2/3` veya `3/3`) ve ikinci aşama eşiği yalnız iç CV'de seçildi.

| Model | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline ensemble | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |
| İki aşamalı hibrit | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |

İç CV 25 dış fold'un 24'ünde üç modelin de olasılığının `≤0.10` olmasını isteyen çok katı `3/3` benign kapısını seçti. Ortalama benign kapısı 0.116, ikinci aşama eşiği 0.6748 oldu. Kapı dış tahminlerin yalnızca %0.18'ini doğrudan benign olarak ayırdı.

**Karar:** Hibrit yaklaşım performansı bozmadı fakat artırmadı; iç CV güvenli ek filtre bulamadığı için sistem pratikte baseline ensemble'a dönüştü. Ayrı bir hibrit model olarak ek karmaşıklık getirmesi gerekçesizdir. Nihai aday yine baseline üçlü ensemble'dır.

Ham sonuçlar: `hybrid_319_two_stage_score.csv`, `hybrid_319_two_stage_folds.csv`, `hybrid_319_two_stage_predictions.csv` ve `hybrid_319_two_stage_choices.csv`.

## 8. F1 odaklı lojistik stacking

ID3, CatBoost ve AdaBoost olasılıkları eşit ortalama yerine lojistik bir üst modele verildi. Üst modelin düzenlileştirmesi, sınıf ağırlığı ve karar eşiği meta-seviye iç CV ile seçildi; dış fold seçimden ayrı tutuldu. Optimizasyon şartnamedeki yaklaşık 100 benign + 20 patojenik dağılımındaki pozitif F1'i hedefledi, eşitlikte daha az FP seçildi.

| Model | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline eşit soft voting | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | **0.6633** | **8.6** |
| Lojistik stacking | 0.7586 | 0.8571 | 0.7356 | 0.7025 | 0.4791 | 0.6005 | 14.3 |

Stacking recall'u 0.7089'dan 0.7356'ya çıkardı; ancak FP/100 benign 8.6'dan 14.3'e yükseldi. Bu nedenle yarışmanın temel metriğine karşılık gelen projekte F1 0.6633'ten 0.6005'e, MCC ise 0.4964'ten 0.4791'e düştü. Üst modelin ortalama katsayıları ID3 için 2.396, CatBoost için 2.514 ve AdaBoost için yalnız 0.227 oldu.

**Karar:** Lojistik stacking reddedildi. Şartname dağılımında F1'i artırırken FP'yi düşük tutan en iyi doğrulanmış seçenek hâlâ baseline eşit soft-voting ensemble'dır.

Ham sonuçlar: `stacking_319_logistic_score.csv`, `stacking_319_logistic_folds.csv`, `stacking_319_logistic_predictions.csv` ve `stacking_319_logistic_choices.csv`.

## 9. Hard-negative learning

Her dış eğitim fold'unda baseline CatBoost iç-CV tahminleri kullanılarak benign olduğu hâlde patojenik tahmin edilen zor benign örnekler belirlendi. Bu örneklere `1.0, 1.5, 2.0, 3.0, 4.0` aday ağırlıkları verilerek CatBoost yeniden eğitildi. Zor örnek tespiti, ağırlık seçimi ve eşik seçimi dış test fold'una bakmadan gerçekleştirildi; 319 özellik değişmedi.

| CatBoost | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.7063 | 0.8381 | 0.6756 | 0.6539 | **0.4074** | **0.5437** | **16.2** |
| Hard-negative ağırlıklı | 0.7063 | 0.8190 | 0.6800 | 0.6516 | 0.3968 | 0.5262 | 18.1 |

Dış eğitim fold'u başına ortalama yalnız 1.92 zor benign saptandı. Ağırlık seçimleri kararlı değildi: `1.0` 5, `1.5` 7, `2.0` 3, `3.0` 4 ve `4.0` 6 fold'da seçildi.

**Karar:** Hard-negative öğrenme FP'yi düşürmedi; tersine 16.2'den 18.1'e çıkardı ve hem projekte F1'i hem MCC'yi düşürdü. Küçük benign örnek sayısı güvenilir bir zor-negatif profili öğrenmeye yetmedi. Deney CatBoost güvenlik kapısında reddedildi ve ensemble'a taşınmadı.

Ham sonuçlar: `catboost_319_hard_negative_score.csv`, `catboost_319_hard_negative_folds.csv`, `catboost_319_hard_negative_predictions.csv` ve `catboost_319_hard_negative_choices.csv`.

## 10. Seed/CV bagging

ID3 zaten 21 dengeli bootstrap ağacını oyladığı için önce CatBoost üzerinde beş-seed bagging denendi. Her iç ve dış fold'da beş bağımsız CatBoost aynı 319 özellikle eğitildi, olasılıkları ortalandı ve eşik yalnız iç-OOF ortalamalarından seçildi.

| CatBoost | MCC | Proj. F1 | FP / 100 benign | Recall |
|---|---:|---:|---:|---:|
| Tek seed baseline | 0.4074 | 0.5437 | 16.2 | 0.6756 |
| Beş-seed bagging | 0.4088 | 0.5503 | 15.2 | 0.6689 |

Bagging tek CatBoost'ta küçük bir benign-ağırlıklı iyileşme sağladığı için baseline ID3 ve AdaBoost ile ensemble güvenlik testine geçirildi.

| Ensemble | MCC | Proj. F1 | FP / 100 benign | Recall |
|---|---:|---:|---:|---:|
| Baseline üçlü | **0.4964** | **0.6633** | **8.6** | 0.7089 |
| ID3 + CatBoost Bag5 + AdaBoost | 0.4695 | 0.6104 | 12.4 | 0.7111 |

**Karar:** CatBoost bagging tek modelde küçük fayda sağlasa da ensemble içinde FP'yi 8.6'dan 12.4'e yükseltti; F1 ve MCC'yi düşürdü. Muhtemel neden, seed ortalamasının CatBoost varyansını azaltırken üç model arasındaki yararlı hata çeşitliliğini de azaltmasıdır. Bagged CatBoost nihai ensemble için reddedildi. Bu güvenlik kapısı başarısız olduğu için AdaBoost'a ek bagging hesaplaması taşınmadı.

Ham sonuçlar: `catboost_319_seed_bagging_score.csv`, `catboost_319_seed_bagging_folds.csv`, `catboost_319_seed_bagging_predictions.csv`, `catboost_319_seed_bagging_inner_predictions.csv`, `ensemble_319_catboost_bag5_score.csv`, `ensemble_319_catboost_bag5_folds.csv` ve `ensemble_319_catboost_bag5_predictions.csv`.

## 11. Takım arkadaşı sunumundan aktarılan regularized model yaklaşımı

Farklı veri setine ait skorlar taşınmadan yalnız yöntemler incelendi. Logistic Regression bizim 319 özellikte başarısız oldu: MCC 0.0478, projekte F1 0.2872 ve FP/100 benign 71.4. Düzenlileştirilmiş RandomForest tek başına MCC 0.4404, F1 0.5642 ve FP 16.2 verdi; ancak hata çeşitliliği ensemble içinde yararlı oldu.

| Aday | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP / 100 benign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eski guardrail: ID3 + CatBoost + AdaBoost | 0.7477 | 0.9143 | 0.7089 | 0.6992 | 0.4964 | 0.6633 | 8.6 |
| Yeni ana aday: ID3 + CatBoost + RandomForest | **0.7694** | **0.9143** | **0.7356** | **0.7190** | **0.5225** | **0.6798** | **8.6** |

Paired cluster bootstrap'ta MCC farkı +0.0266 (%95 GA -0.0231–0.0745), F1 farkı +0.0156 (-0.0723–0.0933) ve recall farkı +0.0271 (-0.0001–0.0556) oldu. Nokta tahminleri yeni adayı desteklese de MCC/F1 güven aralıkları sıfırı içerir.

**Güncel karar:** Aynı 319 özellikli `ID3 + CatBoost + RandomForest` eşit soft voting yeni ana adaydır. Önceki `ID3 + CatBoost + AdaBoost` eşit soft voting bağımsız test gelene kadar guardrail/yedek olarak korunur. Veri seti veya mevcut 319 özellikli dosya değiştirilmemiştir.

Ayrıntılı yöntem aktarımı: `TAKIM_ARKADASI_SUNUMUNDAN_YONTEM_AKTARIMI.md`.
