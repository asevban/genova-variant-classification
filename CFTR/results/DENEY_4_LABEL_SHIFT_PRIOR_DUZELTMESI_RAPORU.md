# Deney 4 — Label-Shift / Sınıf Prior Düzeltmesi

## Amaç

Eğitim verisindeki patojenik oranı yaklaşık `%81.1`, yarışmada beklenen oran ise `20/120 = %16.7` olduğu için Bayes odds düzeltmesiyle olasılıklar hedef sınıf oranına taşındı.

Düzeltme iki şekilde uygulandı:

1. Üç modelin ortalama olasılığına tek seferde,
2. ID3, CatBoost ve Random Forest olasılıklarına ayrı ayrı, ardından ortalama alınarak.

Her ikisi hem sabit `0.50` eşiğiyle hem de yalnız iç CV'de seçilen eşikle değerlendirildi. Aynı 5-fold × 5 tekrar dış tahminleri kullanıldı.

## Sonuçlar

| Yöntem | Macro-F1 | MCC | Proj. F1 | Recall | Specificity | FP/100 |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline: ham olasılık + iç-CV eşik** | **0.7190** | **0.5225** | **0.6798** | 0.7356 | **0.9143** | **8.57** |
| Her bileşene prior düzeltmesi + iç-CV eşik | 0.7135 | 0.5080 | 0.6511 | 0.7356 | 0.8952 | 10.48 |
| Birleşik olasılığa prior düzeltmesi + iç-CV eşik | 0.7124 | 0.5031 | 0.6390 | **0.7378** | 0.8857 | 11.43 |
| Her bileşene prior düzeltmesi + sabit 0.50 | 0.1785 | 0.0584 | 0.0349 | 0.0178 | **1.0000** | **0.00** |
| Birleşik olasılığa prior düzeltmesi + sabit 0.50 | 0.1591 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | **0.00** |

## Yorum

Sabit 0.50 eşiği FP'yi sıfırlamış fakat neredeyse bütün patojenik örnekleri benign olarak sınıflandırmıştır. Bu nedenle pratik olarak kullanılamaz.

Prior düzeltmesinden sonra eşik yeniden iç CV'de seçildiğinde recall korunmuş fakat FP yükselmiş ve projeksiyon F1 düşmüştür. Bunun temel nedeni, mevcut baseline eşik seçiminin zaten 100 benign + 20 patojenik hedef dağılımını doğrudan optimize etmesidir. Monoton bir prior dönüşümü sıralamayı büyük ölçüde değiştirmez; yeniden eşik seçildiğinde ek fayda üretmemiştir.

## Karar

Bilinen hedef prior'ına yapılan doğrudan Bayes düzeltmesi **reddedildi**. Güncel yöntem olan ham ensemble olasılığı + fold-içi benign-ağırlıklı eşik daha başarılıdır.

BBSE gibi test verisinden etiketsiz prior tahmini, gerçek yarışma test özellikleri verilmeden değerlendirilemez. Test seti geldiğinde yalnız bir dağılım kayması denetimi olarak çalıştırılabilir; mevcut modelin yerine otomatik uygulanmamalıdır.

## Kanıt dosyaları

- `label_shift_prior_319_scores.csv`
- `label_shift_prior_319_folds.csv`
- `label_shift_prior_319_predictions.csv`

