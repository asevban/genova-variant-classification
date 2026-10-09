# GENOVA JSON Merger Output

Birleştirilmiş JSON çıktısı varsayılan olarak burada oluşur:

`GENOVA_FINAL_SUBMISSION.json`

Varsayılan format `internal` formattır. Bu format izlenebilirlik için `panel`, `predicted_class` ve `predicted_prob` alanlarını korur.

Resmî kılavuz sadece `Variant_ID` ve `Label` isterse, merge scripti şu komutla minimal aday format üretebilir:

```powershell
python merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION_OFFICIAL.json --format official
```

Önemli: `official` formatı final günü açıklanan resmî şemayla mutlaka karşılaştırılmalıdır.
