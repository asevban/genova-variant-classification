# 02 — Ön İşleme Kararları

| Karar | Uygulama | Gerekçe |
|---|---|---|
| Ham veri | Salt okunur kaynak olarak saklandı | İzlenebilirlik |
| Kimlik/hedef | Özellik CSV'lerinden ayrıldı | Doğrudan hedef ve kimlik sızıntısını önleme |
| `./.` | Anlamsal eksik kabul edildi | Kategorik sahte seviye oluşmasını önleme |
| Sayısal eksik | `NaN` bırakıldı | İmputasyonu modele ve training fold'una bırakma |
| Sabit sütun | V2+ için training fold'unda öğrenilir | Bilgi taşımayan boyutu azaltma |
| Birebir kopya | V2+ için training fold'unda öğrenilir | Tekrarlı ağırlığı azaltma |
| ≥%85 eksik | V3/A kollarında training fold'unda öğrenilir | Çok düşük kapsama stres testi |
| Outlier | Otomatik kırpılmadı | Biyolojik uç değerleri yanlış silmeme |
| SMOTE/ölçekleme/encoding | Statik CSV'ye uygulanmadı | CV sızıntısını ve model bağımlılığını önleme |

## Sızıntı sınırı

Her dış fold için preprocessing yalnızca outer-train üzerinde fit edilir. Hiperparametre, özellik seçimi, kalibrasyon ve threshold inner-CV içinde belirlenir. Outer-validation sadece tarafsız skor üretir. Final/test verisinde hiçbir eşik veya politika yeniden ayarlanmaz.

## Satır güvenliği

Hedef manifestindeki `duplicate_group_id`, aynı özellik profiline sahip satırları birlikte tutar. Split üretiminde stratifikasyon etiketi dengeler; grup kısıtı yinelenen satırların train ve validation'a dağılmasını engeller.
