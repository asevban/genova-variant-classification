# Nested Model Bileşeni Ablation Karşılaştırması

Her tekil, ikili ve üçlü yapı eşit ağırlıkla birleştirilmiş; eşik yalnız ilgili dış fold'un iç-CV tahminlerinden öğrenilmiştir. Modeller yeniden eğitilmemiştir.

|Yapı|Değerlendirme|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ID3|cross_fitted_bagged|0.7838|0.5053|0.6444|1.0000|0.00|58|0|32|21|
|ID3+CAT+RF|cross_fitted_bagged|0.7580|0.5717|0.7556|0.9524|4.76|68|1|22|20|
|RF|cross_fitted_bagged|0.6901|0.5600|0.7778|0.9048|9.52|70|2|20|19|
|ID3+RF|cross_fitted_bagged|0.6837|0.5477|0.7667|0.9048|9.52|69|2|21|19|
|ID3+CAT|cross_fitted_bagged|0.6638|0.5130|0.7333|0.9048|9.52|66|2|24|19|
|CAT+RF|cross_fitted_bagged|0.5928|0.4658|0.7222|0.8571|14.29|65|3|25|18|
|CAT|cross_fitted_bagged|0.5864|0.4550|0.7111|0.8571|14.29|64|3|26|18|
|ID3+CAT+RF|pooled_5x5|0.6798|0.5225|0.7356|0.9143|8.57|331|9|119|96|
|ID3+RF|pooled_5x5|0.6111|0.4841|0.7333|0.8667|13.33|330|14|120|91|
|ID3|pooled_5x5|0.6034|0.4260|0.6378|0.9048|9.52|287|10|163|95|
|CAT+RF|pooled_5x5|0.6010|0.4549|0.6956|0.8762|12.38|313|13|137|92|
|ID3+CAT|pooled_5x5|0.5994|0.4644|0.7133|0.8667|13.33|321|14|129|91|
|RF|pooled_5x5|0.5642|0.4404|0.7111|0.8381|16.19|320|17|130|88|
|CAT|pooled_5x5|0.5437|0.4074|0.6756|0.8381|16.19|304|17|146|88|

Karar yalnız en yüksek tek skora göre verilmez. Üçlü yapının ikililere göre F1/FP dengesi ve modeller arası tamamlayıcılığı birlikte değerlendirilir.
