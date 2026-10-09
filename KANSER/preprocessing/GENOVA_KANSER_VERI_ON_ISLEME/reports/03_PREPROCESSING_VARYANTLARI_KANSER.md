# 03 — Preprocessing Varyantları

V1–V3 ana ön işleme seçimi içindir. Her biri aynı sabit model ve split bankasında değerlendirilir. Tek değişken, ön işleme politikasıdır.

| Varyant | Değişiklik | Tam-veri referans boyutu |
|---|---|---:|
| V1 | Kategorik eksik standardizasyonu | 388×351 |
| V2 | V1 − sabitler − birebir kopyalar | 388×280 |
| V3 | V2 − eksikliği ≥%85 sütunlar | 388×262 |

A0, V3'tür. A1–A4 kontrollü dayanıklılık merdivenidir:

| Kol | V3'e göre tek ana soru | Boyut |
|---|---|---:|
| A1 | `CAT_1/2` çıkarılınca ne olur? | 388×260 |
| A2 | Yalnızca missingness özetleri eklenince ne olur? | 388×276 |
| A3 | Yalnızca AA türetimleri eklenince ne olur? | 388×267 |
| A4 | Kaynak-sütunları çıkarılmış birleşik tasarım dayanıklı mı? | 388×279 |

`V4_LEGACY`, eski birleşik dosyanın yeniden üretilebilir kopyasıdır. Bu sürümde `CAT_1/2` ana özelliklerden çıkarılsa da eksiklik özetlerine izleri girebilir. Bu nedenle kaynak-dayanıklı iddia için A4 kullanılır.

Seçim ölçütleri: pozitif-sınıf OOF F1, MCC, sensitivity, specificity, benign F1, train–OOF farkı, tekrarlar arası değişkenlik ve özellik sayısıdır. Benzer skor durumunda daha basit politika tercih edilir.
