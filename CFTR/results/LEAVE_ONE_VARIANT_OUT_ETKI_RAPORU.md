# Leave-One-Variant-Out Etki Analizi

Bu analiz modeli yeniden seçmez; kaydedilmiş beş tekrarlı dış-fold tahminlerinden her Variant_ID kümesini sırayla çıkarıp toplu skorları yeniden hesaplar.

- Ana model referans projeksiyon F1: 0.6798
- Yedek model referans projeksiyon F1: 0.6633
- Ana-yedek farkı: 0.0165
- Tek varyant çıkarıldığında sıralamayı tersine çeviren varyant sayısı: 1
- En büyük mutlak ana-model F1 değişimi: VAR_001930, 0.0611

## En etkili 15 varyant

|Variant_ID|Etiket|Ana F1 değişimi|MCC değişimi|FP/100 değişimi|Ana-yedek F1 farkı|Sıralama değişti mi?|
|---|---:|---:|---:|---:|---:|---:|
|VAR_001930|0|0.0611|0.0199|-3.5714|-0.0018|Evet|
|VAR_001850|0|0.0256|0.0050|-1.5714|0.0331|Hayır|
|VAR_001977|0|0.0256|0.0050|-1.5714|0.0487|Hayır|
|VAR_001934|0|0.0091|-0.0025|-0.5714|0.0166|Hayır|
|VAR_001961|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001833|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001846|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001901|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001848|0|-0.0067|-0.0100|0.4286|0.0008|Hayır|
|VAR_001785|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001964|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001913|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001804|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001823|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|
|VAR_001871|0|-0.0067|-0.0100|0.4286|0.0164|Hayır|

Bir varyantın etkili çıkması onun silinmesi gerektiğini göstermez. Etiket veya veri hatası bağımsız olarak doğrulanmadıkça bütün varyantlar korunur.
