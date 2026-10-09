# Proje Dosya Yapısı

```text
GENOVA_KANSER_VERI_ON_ISLEME/
├── README.md
├── build_all.py
├── data/
│   ├── raw/YARISMA_TRAIN_KANSER.csv
│   ├── processed/kanser/        # hedef manifesti + V1–V4 + A1–A4
│   └── splits/kanser/           # 10 dış tekrar + 50 iç-CV dosyası
├── src/genova/kanser/           # şema, preprocessing, split ve tanı kodu
├── configs/kanser/              # her varyantın insan/makine okunur ayarı
├── artifacts/preprocessors/     # tam-veri referansında öğrenilen sütun politikaları
├── reports/
│   ├── tables/                  # denetim ve deney kayıt tabloları
│   └── figures/                 # SVG özet grafikler
└── tests/                       # otomatik bütünlük ve sızıntı kontrolleri
```

`artifacts/preprocessors` altındaki dosyalar inceleme referansıdır. Resmî CV'de bu listeler dışarıdan doğrudan uygulanmaz; aynı politika training fold'unda `KanserPreprocessor.fit()` ile yeniden öğrenilir.

`data/splits/kanser` değişmez deney bankasıdır. Aynı `duplicate_group_id` içindeki satırlar hiçbir validation sınırını geçmez. Böylece veri seti ve model farkları aynı örnekler üzerinde karşılaştırılır.
