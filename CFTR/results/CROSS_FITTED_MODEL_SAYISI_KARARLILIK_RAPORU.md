# Cross-fitted Bagging — 3, 4 ve 5 Model Kararlılığı

Tüm tekrar kombinasyonları değerlendirilmiştir: 3 model için 10, 4 model için 5, 5 model için 1 kombinasyon. Tek bir yüksek skorlu kombinasyon seçilmemiştir; amaç model sayısı arttıkça ortalama sonuç ve yayılımın nasıl değiştiğini ölçmektir.

|Model sayısı|Kombinasyon|Metrik|Ortalama|SS|Minimum|Maksimum|
|---:|---:|---|---:|---:|---:|---:|
|3|10|projected_f1|0.7037|0.0600|0.6056|0.7648|
|3|10|mcc|0.5456|0.0297|0.4882|0.5835|
|3|10|recall|0.7544|0.0122|0.7333|0.7667|
|3|10|specificity|0.9190|0.0392|0.8571|0.9524|
|3|10|fp_per_100_benign|8.0952|3.9203|4.7619|14.2857|
|4|5|projected_f1|0.7404|0.0362|0.6771|0.7648|
|4|5|mcc|0.5624|0.0194|0.5358|0.5835|
|4|5|recall|0.7533|0.0122|0.7333|0.7667|
|4|5|specificity|0.9429|0.0213|0.9048|0.9524|
|4|5|fp_per_100_benign|5.7143|2.1296|4.7619|9.5238|
|5|1|projected_f1|0.7580|0.0000|0.7580|0.7580|
|5|1|mcc|0.5717|0.0000|0.5717|0.5717|
|5|1|recall|0.7556|0.0000|0.7556|0.7556|
|5|1|specificity|0.9524|0.0000|0.9524|0.9524|
|5|1|fp_per_100_benign|4.7619|0.0000|4.7619|4.7619|

## Önceden belirlenmiş karar kuralı

En yüksek tek kombinasyon seçilmeyecektir. Daha fazla model, ortalama F1'i koruyor ve kombinasyonlar arası oynaklığı azaltıyorsa tercih edilir. Beş modelli sonuç F1=0.7580; üç modelli ortalama 0.7037 (SS=0.0600), dört modelli ortalama 0.7404 (SS=0.0362) bulunmuştur. Bu nedenle nihai adayda beş model korunur.
