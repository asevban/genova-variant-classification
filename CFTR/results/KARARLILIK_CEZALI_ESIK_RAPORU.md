# Kararlılık Cezalı Eşik Deneyi

|Yöntem|MCC|Proj. F1|Recall|Specificity|FP/100|Eşik ort±SS|
|---|---:|---:|---:|---:|---:|---:|
|lambda_025|0.5000|0.6201|0.7489|0.8667|13.33|0.708±0.044|
|lambda_050|0.4883|0.6391|0.7156|0.8952|10.48|0.716±0.046|
|lambda_100|0.4767|0.6405|0.6956|0.9048|9.52|0.722±0.050|

Referans: MCC 0.5225, projeksiyon F1 0.6798, FP/100 8.57. Eşik adayları yalnız iç OOF tahminlerindeki dört alt foldun ortalama F1 eksi λ×standart sapmasına göre seçildi.
