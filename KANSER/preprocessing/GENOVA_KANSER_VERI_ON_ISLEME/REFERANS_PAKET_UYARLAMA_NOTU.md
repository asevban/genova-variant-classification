# PAH Referans Paketinden KANSER'e Uyarlama

İncelenen PAH ZIP'i ham veri, işlenmiş varyantlar, konfigürasyonlar, nested-CV bölünmeleri, fold-safe kod, testler, tablolar, raporlar, şekiller ve ön işleyici artefaktları içeriyordu. KANSER paketi aynı ana katmanları koruyacak şekilde yeniden kuruldu.

| PAH yaklaşımı | KANSER uyarlaması |
|---|---|
| Birden fazla preprocessing varyantı | V1–V3 ana seçim, A1–A4 kontrollü ablation |
| 5×10 dış CV + iç CV | 5×10 grup-güvenli dış CV + her dış fold için 4 katlı iç CV |
| Fold içinde öğrenilen dönüşümler | Sabit/kopya/yüksek-eksik kararlarını training fold'unda öğrenen `KanserPreprocessor` |
| Eksiklik tanıları | `NaN` ve `./.` birlikte anlamsal eksiklik; sınıf ve `CAT_1/2` provenance kontrolü |
| Test ve rapor katmanı | Şema, X/y ayrımı, boyut, sonsuz değer, fold hizası ve grup bütünlüğü testleri |

KANSER'e özgü en önemli ekleme, eksiklik örüntüsünün hedefle güçlü ilişkisi nedeniyle kaynak-kestirmesi testidir. Eski birleşik dosya `LEGACY` olarak korunmuş; düzeltilmiş A4 sürümünde `CAT_1/2` hem özelliklerden hem eksiklik hesaplarının kaynak listesinden çıkarılmıştır.

Referans ZIP'teki çalıştırma önbellekleri (`__pycache__`, `.pytest_cache`, `.pyc`) teslim paketine alınmamıştır. Bunlar bilimsel içeriğin parçası değildir ve farklı bilgisayarlarda yeniden oluşur.
