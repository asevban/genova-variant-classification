# Varyant Bazlı Hata ve Veri Kalitesi Denetimi

Her Variant_ID için ana modelin beş tekrardaki dış-fold tahminleri birleştirildi. Bu analiz etiket değiştirmez veya satır silmez; yalnız manuel inceleme adaylarını belirler.

## Özet

- CONSISTENT_FN: 12
- MAJORITY_ERROR: 11
- INTERMITTENT_ERROR: 24
- CONSISTENT_CORRECT: 64

- Beş tekrarda sürekli FP: 0
- Beş tekrarda sürekli FN: 12
- En az 3/5 tekrar yanlış: 23

## Sürekli yanlış sınıflandırılan varyantlar

|Variant_ID|Gerçek|Hata/5|Ort. skor|Skor SS|Eksik özellik|Bileşen anlaşmazlığı/5|
|---|---:|---:|---:|---:|---:|---:|
|VAR_001874|1|5|0.6179|0.0791|305|4|
|VAR_001958|1|5|0.6167|0.0605|118|3|
|VAR_001887|1|5|0.6149|0.0634|254|3|
|VAR_001857|1|5|0.6355|0.0262|41|2|
|VAR_001938|1|5|0.5874|0.0721|170|2|
|VAR_001858|1|5|0.6383|0.0529|0|2|
|VAR_001798|1|5|0.6257|0.0258|17|2|
|VAR_001982|1|5|0.5758|0.0398|179|1|
|VAR_001834|1|5|0.5174|0.0577|169|0|
|VAR_001976|1|5|0.4284|0.0175|1|0|
|VAR_001955|1|5|0.5895|0.0329|17|0|
|VAR_001820|1|5|0.4373|0.0440|0|0|

Bu kayıtlar otomatik olarak aykırı değer veya yanlış etiket kabul edilemez. Kaynak etiket, varyant gösterimi ve annotation değerleri bağımsız biyolojik kaynaktan doğrulanmadan veri değişikliği yapılmamalıdır.
