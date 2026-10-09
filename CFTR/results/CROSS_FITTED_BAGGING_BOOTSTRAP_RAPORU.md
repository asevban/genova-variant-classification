# Cross-fitted Bagging — Variant_ID Kümeli Bootstrap

Her varyant tek bağımsız küme kabul edilmiştir. Benign ve patojenik sınıflar ayrı ayrı yeniden örneklenerek eğitim setindeki `21 benign + 90 patojenik` sayısı korunmuştur. Hesaplama 100.000 bootstrap tekrarıyla yapılmıştır.

|Karar yöntemi|Metrik|Nokta tahmini|%95 güven aralığı|
|---|---|---:|---:|
|probability_average|observed_f1|0.8553|0.7919 – 0.9102|
|probability_average|projected_f1|0.7580|0.5864 – 0.9024|
|probability_average|mcc|0.5717|0.4550 – 0.6923|
|probability_average|recall|0.7556|0.6667 – 0.8444|
|probability_average|specificity|0.9524|0.8571 – 1.0000|
|probability_average|fp_per_100_benign|4.7619|0.0000 – 14.2857|
|majority_vote|observed_f1|0.8553|0.7919 – 0.9091|
|majority_vote|projected_f1|0.7580|0.5864 – 0.9024|
|majority_vote|mcc|0.5717|0.4550 – 0.6831|
|majority_vote|recall|0.7556|0.6667 – 0.8444|
|majority_vote|specificity|0.9524|0.8571 – 1.0000|
|majority_vote|fp_per_100_benign|4.7619|0.0000 – 14.2857|
|conservative_4of5|observed_f1|0.8182|0.7448 – 0.8820|
|conservative_4of5|projected_f1|0.7224|0.5532 – 0.8679|
|conservative_4of5|mcc|0.5171|0.4042 – 0.6312|
|conservative_4of5|recall|0.7000|0.6000 – 0.7889|
|conservative_4of5|specificity|0.9524|0.8571 – 1.0000|
|conservative_4of5|fp_per_100_benign|4.7619|0.0000 – 14.2857|

## Yorum

Güven aralığı, aynı büyüklükte yeni varyant örneklemlerinde skorun ne kadar oynayabileceğini gösterir. Final test dağılımına uyarlanmış F1 için nokta tahmini tek başına yeterli değildir; özellikle alt sınır ve `FP/100 benign` üst sınırı birlikte değerlendirilmelidir. Bu analiz final test etiketlerini kullanmaz.
