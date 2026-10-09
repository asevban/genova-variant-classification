# Deney 3 — Robust ve Maliyet Duyarlı RBF-SVM

## Amaç ve yöntem

319 özellikli veri üzerinde doğrusal olmayan bir RBF-SVM test edildi.

- Sayısal kolonlar: fold-içi medyan imputasyon + RobustScaler
- Kategorik kolonlar: fold-içi eksik kategori + one-hot encoding
- Adaylar: sınırlı `C`, `gamma` ve benign sınıf ağırlığı kombinasyonları
- Model/parametre ve karar eşiği yalnız 4-fold iç CV'de seçildi.
- Dış değerlendirme 5-fold × 5 tekrar nested CV ile yapıldı.
- SVM tek başına ve mevcut ensemble'a farklı ağırlıklarla eklenerek denendi.

25 dış fold'un 22'sinde `C=1, gamma=scale, class_weight=balanced`; üçünde `C=10` karşılığı seçildi.

## Sonuçlar

| Yapı | Macro-F1 | MCC | Proj. F1 | Recall | Specificity | FP/100 |
|---|---:|---:|---:|---:|---:|---:|
| **Mevcut ID3 + CatBoost + RF** | **0.7190** | **0.5225** | **0.6798** | **0.7356** | 0.9143 | 8.57 |
| RBF-SVM tek başına | 0.4888 | −0.0117 | 0.2673 | 0.7200 | 0.2667 | 73.33 |
| Ana ensemble + düşük %10 SVM | 0.6868 | 0.4767 | 0.6406 | 0.6956 | 0.9048 | 9.52 |
| Ana ensemble + yaklaşık %14 SVM | 0.6788 | 0.4665 | 0.6336 | 0.6844 | 0.9048 | 9.52 |
| Dört model eşit ağırlık | 0.6516 | 0.4371 | 0.6203 | 0.6422 | 0.9143 | 8.57 |
| ID3 + RF + SVM | 0.6196 | 0.4091 | 0.6127 | 0.5889 | **0.9333** | **6.67** |

ID3 + RF + SVM false-positive değerini düşürmüştür; ancak recall 0.5889'a ve projeksiyon F1 0.6127'ye gerilediği için kabul edilebilir değildir. SVM tek başına benign örneklerin çoğunu yanlış sınıflandırmış ve MCC sıfıra yakın/negatif çıkmıştır.

## Karar

RBF-SVM hem tek model olarak hem de mevcut ensemble'a ek bileşen olarak güvenlik kapısını geçememiştir. En iyi SVM destekli projeksiyon F1 sonucu 0.6406 ile baseline 0.6798'in altındadır ve FP de 8.57'den 9.52'ye yükselmiştir.

Bu nedenle **RBF-SVM reddedildi** ve ana model değiştirilmedi.

## Teknik not

Kullanılan scikit-learn 1.9 sürümü `SVC(probability=True)` için gelecekte kaldırılma uyarısı vermektedir. Bu bir çalışma hatası değildir; model başarı güvenlik kapısında zaten elendiği için daha pahalı ayrı kalibrasyon deneyi yapılmamıştır.

## Kanıt dosyaları

- `rbf_svm_319_ensemble_scores.csv`
- `rbf_svm_319_ensemble_folds.csv`
- `rbf_svm_319_ensemble_predictions.csv`
- `rbf_svm_319_nested_predictions.csv`
- `rbf_svm_319_nested_inner_predictions.csv`
- `rbf_svm_319_nested_choices.csv`

