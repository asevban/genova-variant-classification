# MASTER Hybrid Nested Governance Raporu

Bu rapor, Hybrid-C / Hybrid-E adaylarının aynı OOF üzerinde aşırı seçilme riskini kontrol etmek için yapılan nested governance doğrulamasını özetler.

## Protokol

- Preprocessing: `M3_missing_aware_compact`
- Outer seedler: `[42, 52, 62, 72, 82]`
- Outer fold sayısı: `25`
- Inner fold sayısı: `4`
- Her outer fold içinde ağırlık, kalibrasyon ve threshold yalnız inner OOF üzerinde seçildi.
- Outer holdout yalnız seçilmiş kararın final değerlendirmesinde kullanıldı.
- Final/test veri ve historical outer validation seçim için kullanılmadı.

## Kısa Karar

Nested doğrulamada en yüksek özet sıradaki aday: **Hybrid-E_bestMCC_nested**.

CATOPT-A referansına göre ortalama farklar:

| Metrik | Fark |
|---|---:|
| F1 | +0.0056 |
| MCC | +0.0058 |
| FP/3500 | -25.9 |
| AUROC | +0.0000 |
| Final weighted AUPRC | -0.0005 |

## Ortalama Outer Fold Sonuçları

| Strateji | Strict pass | F1 mean±std | MCC mean±std | FP/3500 mean±std | Sens mean±std | Spec mean±std | AUROC mean±std | AUPRC_final mean±std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid-E_bestMCC_nested | 0.48 | 0.5326±0.0353 | 0.4493±0.0424 | 413.1±84.0 | 0.6598±0.0337 | 0.8623±0.0280 | 0.8525±0.0177 | 0.5065±0.0392 |
| Hybrid-E_AUPRC_guard_nested | 0.40 | 0.5311±0.0349 | 0.4474±0.0420 | 415.0±86.1 | 0.6585±0.0344 | 0.8617±0.0287 | 0.8532±0.0172 | 0.5101±0.0353 |
| Hybrid-C_nested_selected | 0.36 | 0.5307±0.0337 | 0.4470±0.0404 | 416.0±82.8 | 0.6590±0.0356 | 0.8613±0.0276 | 0.8516±0.0176 | 0.5062±0.0391 |
| Hybrid-E_lowFP_nested | 0.36 | 0.5304±0.0307 | 0.4463±0.0373 | 397.7±70.1 | 0.6464±0.0389 | 0.8674±0.0234 | 0.8523±0.0179 | 0.5074±0.0386 |
| Hybrid-C_fixed_reported | 0.36 | 0.5262±0.0358 | 0.4427±0.0430 | 455.3±84.3 | 0.6791±0.0360 | 0.8482±0.0281 | 0.8531±0.0178 | 0.5074±0.0352 |
| Hybrid-E_fixed_AUPRC_reported | 0.28 | 0.5271±0.0369 | 0.4441±0.0445 | 447.6±99.4 | 0.6746±0.0441 | 0.8508±0.0331 | 0.8531±0.0180 | 0.5077±0.0352 |
| CATOPT-A_nested_reference | 0.28 | 0.5270±0.0373 | 0.4435±0.0452 | 439.0±98.7 | 0.6682±0.0434 | 0.8537±0.0329 | 0.8524±0.0180 | 0.5069±0.0351 |
| Hybrid-E_fixed_lowFP_reported | 0.24 | 0.5228±0.0362 | 0.4396±0.0431 | 468.7±103.6 | 0.6825±0.0504 | 0.8438±0.0345 | 0.8511±0.0179 | 0.4994±0.0363 |

## CATOPT-A'ya Göre Paired Farklar

Her strateji aynı outer seed/fold üzerinde CATOPT-A referansıyla karşılaştırılmıştır.

| Strateji | F1 farkı | MCC farkı | FP/3500 farkı | Final AUPRC farkı | F1 win | MCC win | FP <= ref | All3 win | Strict pass | Ref strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid-E_bestMCC_nested | +0.0056 | +0.0058 | -25.9 | -0.0005 | 0.60 | 0.60 | 0.72 | 0.56 | 0.48 | 0.28 |
| Hybrid-E_AUPRC_guard_nested | +0.0041 | +0.0039 | -24.0 | +0.0032 | 0.52 | 0.52 | 0.72 | 0.52 | 0.40 | 0.28 |
| Hybrid-C_nested_selected | +0.0037 | +0.0035 | -23.0 | -0.0007 | 0.60 | 0.60 | 0.76 | 0.52 | 0.36 | 0.28 |
| Hybrid-E_lowFP_nested | +0.0034 | +0.0028 | -41.2 | +0.0005 | 0.52 | 0.48 | 0.76 | 0.48 | 0.36 | 0.28 |

## Governance Yorumu

Bu doğrulama, önceki tek OOF üzerinde yapılan geniş weight-grid ve threshold seçiminden doğabilecek selection-overfit riskini ölçmek için tasarlanmıştır. Bir hibrit adayın resmi final pipeline yerine geçebilmesi için yalnız ortalama metriklerinin iyi olması yetmez; strict pass rate, metrik stabilitesi ve CATOPT-A'ya göre anlamlı/kararlı kazanım birlikte değerlendirilmelidir.

Nested kontrol sonrasında Hybrid-E ailesi CATOPT-A'ya göre daha iyi yönde sinyal vermeye devam etmiştir. En güçlü governance adayı **Hybrid-E_bestMCC_nested**; AUPRC korumayı daha öncelikli tutan alternatif ise **Hybrid-E_AUPRC_guard_nested** olarak kaydedilmelidir. Kazanımlar küçük ve fold varyansı yüksek olduğu için bu çıktı “kesin final kilidi” değil, “freeze sürecine geçebilecek final adayı” olarak yorumlanmalıdır.

## Çıktılar

- `results/modeling/nested_governance/nested_outer_fold_results.csv`
- `results/modeling/nested_governance/nested_governance_summary.csv`
- `results/modeling/nested_governance/nested_outer_predictions.csv`
- `results/modeling/nested_governance/nested_selection_details.csv`
- `results/modeling/nested_governance/hybrid_e_governance_freeze_manifest.json`
