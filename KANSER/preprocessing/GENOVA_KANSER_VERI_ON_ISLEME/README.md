# GENOVA KANSER Veri Ön İşleme Paketi

Bu paket, `YARISMA_TRAIN_KANSER.csv` verisini model karşılaştırmasına hazırlayan, yeniden üretilebilir ve fold-güvenli bir çalışma iskeletidir. Ham veri, sekiz özellik varyantı, hedef/kimlik manifesti, 5×10 dış ve 4 katlı iç CV bölünmeleri, tanısal tablolar, şekiller, kaynak kodu ve otomatik kontroller birlikte verilir.

## En kısa kullanım kararı

- Ana başlangıç adayı: `02_KANSER_COMPACT_NONREDUNDANT.csv`.
- Kontrol adayları: `01_KANSER_NATIVE_SAFE_FULL.csv` ve `03_KANSER_ROBUST_COVERAGE85.csv`.
- Kaynak/eksiklik dayanıklılığı: `A1`–`A4` ablation dosyaları.
- `04_..._LEGACY.csv`, önceki birleşik tasarımın izlenebilir kopyasıdır; final aday olarak doğrudan seçilmemelidir.

“En iyi” veri seti yalnızca CSV boyutuna bakılarak belirlenmez. Önce V1–V3 aynı model, aynı hiperparametreler, aynı split'ler ve aynı threshold yöntemiyle karşılaştırılır. Sonra seçilen ön işleme üzerinde modeller karşılaştırılır.

## Veri özeti

| Bilgi | Değer |
|---|---:|
| Satır | 388 |
| Ham sütun | 353 |
| Model özelliği | 351 |
| Label 0 / Label 1 | 120 / 268 |
| Anlamsal eksik hücre oranı | %57,56 |
| Sabit sütun | 69 |
| Birebir kopya olarak çıkarılan sütun | 61 |
| Eksikliği ≥%85 sütun | 19 |
| Yinelenen satır grubundaki satır | 3 |

Sayısal eksikler `NaN`, kategorik eksikler `__MISSING__` olarak korunur. Global imputasyon, ölçekleme, target encoding, SMOTE veya hedefe bakarak özellik seçimi yapılmaz.

## Dosyalar

| Kimlik | Özellik sayısı | Amaç |
|---|---:|---|
| V1 Native | 351 | Kayıpsız kontrol; yalnızca kategorik eksik gösterimi standarttır |
| V2 Compact | 280 | Sabit ve birebir kopya sütunları training fold'unda çıkarır |
| V3 Robust85 | 262 | V2'ye ek olarak eksikliği en az %85 olan sütunları çıkarır |
| V4 Legacy | 279 | Eski birleşik tasarım; karşılaştırılabilirlik için saklanır |
| A1 Source removed | 260 | V3'ten yalnızca `CAT_1` ve `CAT_2` çıkarılır |
| A2 Missingness only | 276 | V3'e yalnızca 14 eksiklik özeti eklenir |
| A3 AA only | 267 | V3'e yalnızca 5 aminoasit özelliği eklenir |
| A4 Source-resistant | 279 | A1 + kaynak sütunlarını dışlayan eksiklik özetleri + AA özellikleri |

Hedef ve kimlikler `data/processed/kanser/00_KANSER_TARGET_AND_METADATA.csv` içindedir. Özellik CSV'lerinde `Variant_ID` ve `Label` bulunmaz; satırlar `row_index` sırasıyla eşleşir.

## Yeniden üretme

Python 3.10+ ortamında:

```powershell
python -m pip install -r requirements.txt
python build_all.py
python tests/run_checks.py
```

İsteğe bağlı testler:

```powershell
python -m pytest -q
```

## Fold-güvenli kullanım

Paket içindeki CSV'ler inceleme ve deney iskeleti için tam eğitim verisinde üretilmiş referanslardır. Resmî CV skorunda sabit/kopya/yüksek-eksik sütun kararları her training fold'unda yeniden öğrenilmelidir:

```python
from genova.kanser.preprocessing import KanserPreprocessor

prep = KanserPreprocessor(policy="compact")
X_train = prep.fit_transform(raw_train)
X_valid = prep.transform(raw_valid)
```

İmputasyon, encoding, ölçekleme, özellik seçimi, yeniden örnekleme ve threshold belirleme de aynı şekilde yalnızca training/inner-CV içinde yapılır. Dış validation veya final/test etiketi karar değiştirmek için kullanılmaz.

## Klasörler

- `data/raw`: değiştirilmeyen kaynak CSV
- `data/processed/kanser`: özellik varyantları ve hedef manifesti
- `data/splits/kanser`: sabit, grup-güvenli nested-CV bankası
- `src/genova/kanser`: fit/transform ve üretim kodları
- `configs` ve `artifacts`: politika tanımları ile öğrenilmiş tam-veri referansları
- `reports`: EDA, kararlar, tablolar ve SVG şekiller
- `tests`: şema, sızıntı ve split kontrolleri

Bu çalışma yarışma/araştırma amaçlıdır; klinik tanı veya tedavi kararı üretmez.
