# Kalibrasyon ve SHAP Tanısal Raporu

Bu analiz mevcut modeli veya `0.7224` eşiğini değiştirmez.

## Kalibrasyon

|evaluation|n_predictions|brier_score|ece_10_equal_width|probability_mean|observed_positive_rate|
|---|---|---|---|---|---|
|pooled_5x5|555|0.111319|0.130922|0.722685|0.810811|
|cross_fitted_variant_average|111|0.109205|0.152635|0.722685|0.810811|

Brier ve ECE için daha düşük değer daha iyidir. Pooled 5×5 satırlar aynı varyantın beş tahminini içerir; cross-fitted varyant ortalaması her varyantı bir kez sayar ve ana tanısal gösterimdir. Bu sonuçlardan yeni eşik seçilmemiştir.

## CatBoost SHAP — ilk 20

|feature|mean_absolute_shap|family|
|---|---|---|
|EK_9|0.190287|EK|
|EK_7|0.124372|EK|
|AL_298|0.065931|AL|
|AL_327|0.044927|AL|
|AL_7|0.041099|AL|
|AL_296|0.039386|AL|
|AL_19|0.037669|AL|
|AL_289|0.032957|AL|
|AL_3|0.031342|AL|
|AL_34|0.029211|AL|
|AL_304|0.028935|AL|
|GRUP_AL1_AL211|0.026951|GRUP|
|EK_2|0.025572|EK|
|AL_4|0.024898|AL|
|AL_27|0.023171|AL|
|AL_30|0.022574|AL|
|AL_33|0.022197|AL|
|AL_2|0.021898|AL|
|AL_32|0.021364|AL|
|AL_148|0.020052|AL|

## Özellik ailesi toplamı

|family|mean_absolute_shap|
|---|---|
|AL|2.035822|
|EK|0.372766|
|GRUP|0.027406|
|CAT|0.009261|
|AA|0.000000|

SHAP yalnız dondurulmuş CatBoost bileşenini açıklar; ID3+CatBoost+RF ensemble'ının tamamının nedensel açıklaması değildir. Özellik silmek veya modeli yeniden seçmek için kullanılmamıştır.
