# 05 — Takım Paylaşımı ve Uyarlanabilirlik

KANSER panelinde ham veri korunmuş, hedef/kimlik ayrılmış, V1–V3 ön işleme merdiveni ve A0–A4 kaynak-dayanıklılığı merdiveni kurulmuştur. Bütün adaylar ortak, grup-güvenli nested-CV bankasıyla sınanacak; dönüşümler yalnızca training fold'unda öğrenilecektir.

## Diğer panellere doğrudan aktarılabilecek ilkeler

- Kimlik ve hedefi model özelliklerinden ayırma.
- Yinelenen kayıtları aynı fold'da tutma.
- Bütün adaylarda aynı split ve seed'leri kullanma.
- Bir deneyde tek ana bileşeni değiştirme.
- Preprocessing, imputasyon, encoding ve seçimi fold içinde fit etme.
- F1 yanında MCC, sınıf bazlı skor ve kararlılığı raporlama.
- Threshold'u yalnızca OOF/inner-CV tahminleriyle seçip kilitleme.

## Panele göre yeniden belirlenmesi gerekenler

Eksiklik eşiği, nadir kategori kuralı, CV tekrar sayısı, grup tanımı, kalibrasyon yöntemi ve türetilmiş özellikler başka panele aynen kopyalanmaz. Yöntem, o panelin baseline'ına tek değişiklik olarak eklenir ve kendi split'lerinde yeniden sınanır.

Paylaşım kaydında satır/özellik/sınıf sayısı, eksiklik yapısı, kopya veya grup bilgisi, baseline, değiştirilen bileşen, split tasarımı, seed, OOF F1/MCC/sınıf skorları ve sızıntı kontrolleri bulunmalıdır.
