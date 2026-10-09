# Nihai Model F1 Üstünlüğü — 100.000 Tekrarlı Paired Cluster Bootstrap

Her `Variant_ID` bir küme olarak örneklenmiş, aynı varyanta ait beş tekrar tahmini birlikte tutulmuştur. Benign ve patojenik varyantlar kendi sınıfları içinde yeniden örneklenerek sınıf oranı korunmuştur.

|Metrik|Ana model|Yedek model|Fark|%95 GA|Ana modelin üstünlük olasılığı|
|---|---:|---:|---:|---:|---:|
|observed_f1|0.8380|0.8201|0.0179|-0.0018 – 0.0376|%96.2|
|projected_f1|0.6798|0.6633|0.0165|-0.0770 – 0.0922|%66.0|
|mcc|0.5225|0.4964|0.0260|-0.0243 – 0.0730|%85.3|
|recall|0.7356|0.7089|0.0267|-0.0022 – 0.0533|%96.4|
|specificity|0.9143|0.9143|0.0000|-0.0571 – 0.0476|%44.6|
|fp_per_100_benign|8.5714|8.5714|0.0000|-4.7619 – 5.7143|%41.2|

Bir farkın %95 düzeyinde kesin üstünlük sayılması için güven aralığının tamamının sıfırın üzerinde olması gerekir.
