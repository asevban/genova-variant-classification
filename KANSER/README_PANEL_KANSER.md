# GENOVA KANSER Paneli - Final Teslim README

Bu klasör, GENOVA takımının KANSER paneli için hazırladığı bağımsız final teslim paketidir. Paket, final günü eğitim yapmadan yalnızca verilen KANSER test dosyası üzerinde tahmin üretmek üzere düzenlenmiştir.

## Panelin Kapsamı

KANSER paneli, KANSER paneline ait varyant kayıtları için benign/pathogenic sınıf tahmini üretir. Bu panel diğer panellerden bağımsızdır; KANSER verisi yalnızca KANSER final model hattına verilir.

Bu klasördeki yapı, jüri tarafından aşağıdaki amaçlarla incelenebilir:

- Final inference kodunun ve model artefaktlarının paket içinde bulunduğunu görmek.
- Test CSV dosyasının nereye konacağını ve çıktının nerede oluşacağını doğrulamak.
- Model, eşik ve ön işleme kararlarının final öncesinde dondurulduğunu kontrol etmek.
- Deney, sonuç ve jüri kanıt dosyalarının çalışma paketinden ayrıldığını görmek.

## Final Çalıştırma Özeti

Beklenen input dosyası:

```text
input/KANSER.csv
```

Çalıştırma komutu:

```powershell
.\RUN_PANEL.ps1
```

PowerShell çalışmazsa yedek komut:

```bat
RUN_PANEL.bat
```

Başarılı çalışmada beklenen çıktı:

```text
output/TEAM_885171_KANSER_FINAL.json
```

Başarılı ekranda `Validation: PASS` ve `GENOVA KANSER PANEL JSON READY` görülmelidir.

## Klasör Yapısı ve Anlamı

```text
KANSER
├── .venv
├── archive
├── config
├── experiments
├── final
├── input
├── jury
├── logs
├── output
├── preprocessing
├── results
├── src
├── test
├── README_PANEL.md
├── run_panel_final.py
├── RUN_PANEL.bat
└── RUN_PANEL.ps1
```

## Çalıştırma İçin Gerekli Ana Bileşenler

- `.venv`: KANSER panelinin çalışması için gerekli Python ortamıdır.
- `final`: Dondurulmuş final model artefaktları ve inference için gerekli dosyaları içerir.
- `config`: Takım bilgileri, panel registry, sınıf eşlemesi ve panel ayarlarını içerir.
- `src`: Input okuma, panel yönlendirme, JSON yazma ve validation kodlarını içerir.
- `preprocessing`: KANSER paneline ait ön işleme politika ve kanıt dosyalarını içerir.
- `run_panel_final.py`: Panel tahmin hattını çalıştıran ana Python dosyasıdır.
- `RUN_PANEL.ps1`: Windows PowerShell üzerinden paneli çalıştıran ana komuttur.
- `RUN_PANEL.bat`: PowerShell engeli durumunda yedek çalıştırma dosyasıdır.

## Kanıt ve İnceleme Klasörleri

- `results`: Model geliştirme, validasyon ve metrik kanıt dosyalarını içerir.
- `jury`: Jüri incelemesi için hazırlanmış model seçimi ve teknik açıklama dosyalarını içerir.
- `experiments`: Final modeline giden geçmiş deney ve araştırma kayıtlarını içerir.
- `archive`: Eski veya yardımcı dosyaları içerir; final inference hattının ana kaynağı değildir.
- `test`: Geliştirme ve kontrol amaçlı dosyalar içindir.

Bu klasörler, final tahmini üretirken model seçimi yapmak için kullanılmaz. Final tahmini `final`, `config`, `src` ve dondurulmuş `.venv` hattı üzerinden yapılır.

## Python Ortamı

Bu panel için beklenen Python sürümü:

```text
Python 3.11.3
```

Kontrol komutu:

```powershell
.\.venv\Scripts\python.exe --version
```

## Girdi ve Çıktı Şeması

Girdi dosyası panelin beklediği KANSER test CSV formatında olmalıdır. `Variant_ID` alanı final çıktı takibi için kritik kabul edilir.

Çıktı JSON dosyası panel tahminlerini, takım bilgisini ve validation uyumunu içerir. Bu JSON daha sonra ekip liderindeki `GENOVA_JSON_MERGER` aracıyla diğer panel JSON dosyalarıyla birleştirilebilir.

## Final Güvenlik Notları

- Final günü model yeniden eğitilmez.
- Model dosyaları, threshold ve config ayarları değiştirilmez.
- JSON çıktısı elle düzenlenmez.
- Input CSV dosyası elle satır silinerek veya değer uydurularak değiştirilmez.
- `Validation: PASS` görülmeden çıktı teslim edilmez.
- KANSER panelinde `Variant_ID` alanının korunması özellikle önemlidir.

## Sınırlılık

Bu paket bir klinik tanı sistemi değildir. KANSER paneli için yarışma kapsamında verilen test dosyası üzerinde dondurulmuş modelle tahmin üretir. Klinik kullanım için bağımsız dış validasyon ve uzman değerlendirmesi gerekir.

