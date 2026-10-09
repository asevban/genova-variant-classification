# KANSER Row-Level OOF

Bu dosya yalnızca KANSER paneline aittir ve GENOVA final klasör yapısı korunarak eklenmiştir.

`KANSER_ROW_LEVEL_OOF.csv`, 388 etiketli eğitim satırı için leakage-safe tekrarlı OOF ortalama skorunu ve kilitli deployment threshold sonrası tahmini içerir.

Sütunlar: `row_index`, `Variant_ID`, `Label`, `repeated_oof_mean_score`, `locked_oof_prediction`.

Etiket kodu: `0 = benign`, `1 = pathogenic`. Bu çıktı gerçek final test skoru değil, eğitim verisi üzerinde denetim ve hata analizi çıktısıdır.
