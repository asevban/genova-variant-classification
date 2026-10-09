# Cross-fitted Bagging — Bir Modeli Eksiltme Testi

Her varyant için bulunan beş bağımsız cross-fitted olasılıktan sırasıyla biri çıkarılmış, kalan dört olasılık ve dört iç-CV eşiği ortalanmıştır. Amaç, başarıyı tek bir tekrar modelinin sürükleyip sürüklemediğini ölçmektir.

|Yapı|Model sayısı|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|all_5|5|0.7580|0.5717|0.7556|0.9524|4.76|68|1|22|20|
|drop_repeat_1|4|0.7580|0.5717|0.7556|0.9524|4.76|68|1|22|20|
|drop_repeat_2|4|0.7440|0.5490|0.7333|0.9524|4.76|66|1|24|20|
|drop_repeat_3|4|0.7580|0.5717|0.7556|0.9524|4.76|68|1|22|20|
|drop_repeat_4|4|0.6771|0.5358|0.7556|0.9048|9.52|68|2|22|19|
|drop_repeat_5|4|0.7648|0.5835|0.7667|0.9524|4.76|69|1|21|20|

Dört modelli yapıların projeksiyon F1 aralığı **0.6771–0.7648**, MCC aralığı **0.5358–0.5835**, FP/100 aralığı **4.76–9.52** olmuştur.
