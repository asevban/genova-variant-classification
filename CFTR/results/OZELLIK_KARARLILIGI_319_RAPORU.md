# 319 Özellik Fold Kararlılığı

Information Gain her dış eğitim fold'unda ayrı hesaplandı (25 fold). Bu rapor kolon silmez; yalnız sıralama kararlılığını ölçer.

- En az 20/25 fold'da ilk 20: 14
- En az 20/25 fold'da ilk 50: 34
- Hiçbir fold'da ilk 50'ye girmeyen: 205

## En kararlı 20 özellik

|Özellik|Ort. IG|IG SS|İlk 20/25|İlk 50/25|Ort. sıra|
|---|---:|---:|---:|---:|---:|
|CAT_1|0.3178|0.0277|25|25|1.0|
|AA_2|0.2493|0.0314|25|25|2.3|
|AA_1|0.2202|0.0234|25|25|3.6|
|AL_27|0.1575|0.0323|25|25|8.4|
|AL_3|0.1641|0.0365|24|25|7.8|
|AL_33|0.1573|0.0293|24|25|8.6|
|AL_7|0.1712|0.0271|23|25|9.2|
|EK_9|0.1709|0.0285|23|25|9.4|
|AL_34|0.1468|0.0285|22|25|12.4|
|AL_28|0.1453|0.0366|22|25|13.0|
|AL_4|0.1339|0.0308|22|24|17.2|
|AL_327|0.1426|0.0246|21|25|13.6|
|AL_37|0.1415|0.0381|21|24|16.4|
|AL_38|0.1347|0.0315|20|25|17.6|
|AL_30|0.1395|0.0343|19|25|15.3|
|EK_7|0.1364|0.0242|17|25|17.3|
|AL_19|0.1241|0.0265|14|25|20.8|
|AL_32|0.1302|0.0344|13|24|20.4|
|AL_2|0.1232|0.0357|13|23|24.8|
|AL_35|0.1240|0.0289|11|25|22.8|

Düşük kararlılık tek başına silme gerekçesi değildir; özellik etkileşimleri ID3, CatBoost ve Random Forest içinde tek-değişkenli IG tarafından yakalanmayabilir.
