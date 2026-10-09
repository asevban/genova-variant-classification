# ID3 ve Üçlü Ensemble — Paired Bootstrap ve Uyuşmazlık

Her varyant için beş cross-fitted karar ortalanmış, iki model aynı varyant örnekleri üzerinde karşılaştırılmıştır. 100.000 tekrarlı bootstrap benign ve patojenik sınıflar içinde ayrı yürütülmüştür.

|Metrik|ID3|Üçlü|ID3−Üçlü|%95 GA|ID3 üstünlük olasılığı|
|---|---:|---:|---:|---:|---:|
|projected_f1|0.7838|0.7580|0.0258|-0.1179 – 0.1963|%57.9|
|mcc|0.5053|0.5717|-0.0664|-0.1574 – 0.0360|%9.0|
|recall|0.6444|0.7556|-0.1111|-0.1889 – -0.0444|%0.0|
|specificity|1.0000|0.9524|0.0476|0.0000 – 0.1429|%64.1|
|fp_per_100_benign|0.0000|4.7619|-4.7619|-14.2857 – 0.0000|%0.0|

## Varyant bazlı karar uyuşmazlığı

- İkisi de doğru: **77**
- İkisi de yanlış: **21**
- Yalnız ID3 doğru: **2**
- Yalnız üçlü doğru: **11**

ID3'ün F1 avantajı güven aralığı sıfırı içeriyorsa kesin üstünlük kabul edilmez. Recall ve MCC kaybı ile benign örnek sayısının azlığı ayrıca dikkate alınır.
