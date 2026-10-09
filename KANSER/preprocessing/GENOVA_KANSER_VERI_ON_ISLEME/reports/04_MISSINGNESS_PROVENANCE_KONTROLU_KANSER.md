# 04 — Missingness ve Kaynak Kestirmesi Kontrolü

Eksiklik oranı Label 1'de Label 0'a göre belirgin biçimde yüksektir. Bu ilişki faydalı sinyal olabilir; fakat biyolojiden bağımsız veri kaynağı, laboratuvar veya raporlama sürecini temsil etme riski vardır.

## Kontrollü deney

- A0: V3 referansı.
- A1: yalnızca `CAT_1/2` çıkarılır.
- A2: yalnızca 14 satır-temelli eksiklik özeti eklenir.
- A3: yalnızca 5 AA özelliği eklenir.
- A4: `CAT_1/2` hem ana tabandan hem missingness kaynaklarından çıkarılır; eksiklik + AA birlikte eklenir.

## Yorum kuralları

- A2 büyük artış, A1 büyük düşüş gösterirse performans kaynak/eksiklik izine bağımlı olabilir.
- A4, V4 Legacy'den zayıfsa Legacy tasarım `CAT_1/2` eksiklik izinden faydalanıyor olabilir.
- F1 artarken MCC, specificity veya benign F1 düşüyorsa politika dengeli kabul edilmez.
- Fold'lar veya tekrarlar arasında yüksek oynaklık varsa tek ortalama skor yeterli değildir.

Sonuç tablosunda her kol için OOF F1, MCC, sensitivity, specificity, benign F1, ortalama±SS, güven aralığı ve train–OOF farkı birlikte raporlanmalıdır. Kaynak grubu mevcutsa grup bazlı skor; yoksa missingness/CAT alt grup analizi ve adversarial validation kullanılmalıdır.
