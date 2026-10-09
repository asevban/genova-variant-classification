# Cross-Fitted Bagging Deneyi

Her Variant_ID için beş tekrarda, o varyantı eğitimde görmeyen dış-fold modellerinin tahminleri birleştirildi. Böylece her varyant için tek cross-fitted karar üretildi.

|Yöntem|Accuracy|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|---:|---:|---:|---:|
|probability_average|0.7928|0.9524|0.7556|0.7451|0.5717|0.7580|4.76|
|majority_vote_3of5|0.7928|0.9524|0.7556|0.7451|0.5717|0.7580|4.76|
|conservative_vote_4of5|0.7477|0.9524|0.7000|0.7032|0.5171|0.7224|4.76|

Referans pooled 5×5 sonuç: projeksiyon F1 0.6798, MCC 0.5225, FP/100 8.57. Cross-fitted tek-karar sonuçları doğrudan aynı örnek sayısında değildir; yöntem seçimi için destekleyici sağlamlık kanıtıdır.
