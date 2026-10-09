# Nested Ensemble Ağırlık Kararlılığı

Ağırlıklar dış test fold'una bakılmadan, her dış fold'un yalnız iç-CV tahminlerinden seçilmiştir. ID3, CatBoost ve Random Forest'ın her birine en az %10 ağırlık verilmiş; eşit ağırlığa yakınlık yalnız son bağlayıcı olarak kullanılmıştır. Böylece dış-fold skoruna bakarak tek bir sabit ağırlık seçilmemiştir.

|Değerlendirme|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|pooled_5x5|0.6136|0.4621|0.6956|0.8857|11.43|313|12|137|93|
|cross_fitted_bagged|0.6570|0.5020|0.7222|0.9048|9.52|65|2|25|19|

## En sık seçilen ağırlıklar

|ID3|CatBoost|Random Forest|Fold sayısı|
|---:|---:|---:|---:|
|0.6|0.1|0.3|4|
|0.1|0.1|0.8|3|
|0.8|0.1|0.1|3|
|0.3|0.1|0.6|2|
|0.1|0.8|0.1|2|
|0.1|0.2|0.7|2|
|0.4|0.3|0.3|2|
|0.2|0.2|0.6|1|
|0.1|0.3|0.6|1|
|0.2|0.1|0.7|1|

Referans eşit ağırlıklı beş-model cross-fitted sonuç: projeksiyon F1 `0.7580`, MCC `0.5717`, FP/100 benign `4.76`. Nested ağırlık seçimi bu referansı açık ve kararlı biçimde geçmiyorsa eşit ağırlık korunmalıdır.
