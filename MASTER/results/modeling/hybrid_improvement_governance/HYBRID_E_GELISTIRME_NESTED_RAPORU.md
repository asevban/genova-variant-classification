# MASTER Hybrid-E Geliştirme Nested Governance Raporu

Bu rapor, kilitli Hybrid-E final adayını bozmadan denenebilecek geliştirme yollarını nested governance altında ölçer.

## Şartname / Governance Sınırı

- Kullanılan veri: yalnız sağlanan MASTER etiketli eğitim tablosu.
- Final/test veri kullanılmadı.
- Dış veri, genomic adres çözme/tersine arama veya yarışma dışı label lookup kullanılmadı.
- Preprocessing her inner/outer fold içinde yeniden fit edildi.
- Ağırlık, kalibrasyon, threshold ve fallback politikası yalnız inner OOF üzerinde seçildi.
- Outer holdout sadece seçilmiş kararın tek seferlik değerlendirmesinde kullanıldı.
- Tekrarlayan outer FP ID listesi model seçimi veya eğitim ağırlığı için kullanılmadı.

## Denenen Adımlar

1. FP tamponu: inner seçimde 400 yerine 375 ve 350 FP/3500 hedefleri denendi; sensitivity tabanı 0.62 olarak korundu.
2. Missingness guard: Q2/Q4 missingness binlerinde M2/M4 fallback skorları yalnız inner-CV seçilen bir skor dönüşümü olarak denendi.
3. Hard-benign güçlendirme: mevcut HBRF bileşeninin ağırlığı artırılmış aday vektörleri denendi; outer FP ID öğrenmesi yapılmadı.
4. AUPRC guard: mevcut AUPRC koruması ve FP375 tamponlu AUPRC challenger birlikte raporlandı.
5. TabPFN final modele dahil edilmedi.

## Kısa Karar

Özet sıralamada en yüksek aday: **Hybrid-E_bestMCC_nested**. Strict pass=0.48, F1=0.5326, MCC=0.4493, FP/3500=413.1.

## Final Uygulama Kararı

Mevcut `Hybrid-E_bestMCC_nested` freeze korunmalı. Denenen challenger'ların hiçbiri mevcut adayı aynı anda F1, MCC, strict pass ve FP kuyruğu açısından geçmedi.

- `Hybrid-E_lowFP_nested` ortalama FP'yi düşürdü; ancak F1, MCC ve strict pass geriledi.
- `Hybrid-E_AUPRC_guard_nested` final weighted AUPRC'yi artırdı; fakat F1/MCC tarafında mevcut BestMCC'nin altında kaldı.
- `Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested` AUPRC ve bazı missingness risklerini iyileştirme sinyali verdi; ama F1/MCC ve strict pass yeterli değil.
- FP350/FP375 buffer ve hard-benign-heavy challenger'lar kötü fold FP kuyruğunu güvenilir biçimde kesemedi.

## Ortalama Outer Fold Sonuçları

| strategy | outer_folds | strict_pass_rate | outer_f1_mean | outer_mcc_mean | outer_expected_fp_per_3500_mean | outer_expected_fp_per_3500_max | outer_sensitivity_mean | outer_specificity_mean | outer_final_weighted_auprc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 25.0000 | 0.4800 | 0.5326 | 0.4493 | 413.0971 | 600.0000 | 0.6598 | 0.8623 | 0.5064 |
| Hybrid-E_AUPRC_guard_nested | 25.0000 | 0.4000 | 0.5311 | 0.4474 | 415.0171 | 624.0000 | 0.6585 | 0.8617 | 0.5101 |
| Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested | 25.0000 | 0.4000 | 0.5267 | 0.4420 | 414.9714 | 595.2381 | 0.6519 | 0.8617 | 0.5092 |
| Hybrid-E_FP350_buffer_nested | 25.0000 | 0.3600 | 0.5310 | 0.4480 | 431.3219 | 672.0000 | 0.6711 | 0.8562 | 0.5060 |
| Hybrid-E_lowFP_nested | 25.0000 | 0.3600 | 0.5304 | 0.4463 | 397.7448 | 528.0000 | 0.6464 | 0.8674 | 0.5074 |
| Hybrid-E_hardBenign_FP375_nested | 25.0000 | 0.3600 | 0.5280 | 0.4443 | 433.2038 | 600.0000 | 0.6672 | 0.8556 | 0.5064 |
| Hybrid-E_AUPRC_FP375_nested | 25.0000 | 0.3600 | 0.5261 | 0.4411 | 418.8495 | 576.0000 | 0.6539 | 0.8604 | 0.5081 |
| Hybrid-E_FP375_buffer_nested | 25.0000 | 0.3200 | 0.5279 | 0.4435 | 418.8495 | 624.0000 | 0.6562 | 0.8604 | 0.5068 |

## Mevcut Hybrid-E BestMCC'ye Göre Fark

| strategy | delta_f1_mean | delta_mcc_mean | delta_fp_per_3500_mean | delta_final_auprc_mean | f1_win_rate | mcc_win_rate | fp_le_reference_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_FP350_buffer_nested | -0.0016 | -0.0012 | 18.2248 | -0.0005 | 0.2400 | 0.2400 | 0.6400 |
| Hybrid-E_AUPRC_guard_nested | -0.0015 | -0.0018 | 1.9200 | 0.0036 | 0.1200 | 0.1200 | 0.8800 |
| Hybrid-E_lowFP_nested | -0.0022 | -0.0029 | -15.3524 | 0.0010 | 0.3200 | 0.2800 | 0.8800 |
| Hybrid-E_hardBenign_FP375_nested | -0.0046 | -0.0050 | 20.1067 | -0.0000 | 0.3200 | 0.3200 | 0.5600 |
| Hybrid-E_FP375_buffer_nested | -0.0047 | -0.0058 | 5.7524 | 0.0004 | 0.2000 | 0.1600 | 0.7200 |
| CATOPT-A_nested_reference | -0.0056 | -0.0058 | 25.8819 | 0.0005 | 0.4000 | 0.4000 | 0.4400 |
| Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested | -0.0059 | -0.0072 | 1.8743 | 0.0028 | 0.2400 | 0.2000 | 0.6800 |
| Hybrid-E_AUPRC_FP375_nested | -0.0065 | -0.0081 | 5.7524 | 0.0017 | 0.1200 | 0.0800 | 0.6800 |

## CATOPT-A Referansına Göre Fark

| strategy | delta_f1_mean | delta_mcc_mean | delta_fp_per_3500_mean | delta_final_auprc_mean | f1_win_rate | mcc_win_rate | fp_le_reference_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 0.0056 | 0.0058 | -25.8819 | -0.0005 | 0.6000 | 0.6000 | 0.7200 |
| Hybrid-E_FP350_buffer_nested | 0.0040 | 0.0045 | -7.6571 | -0.0009 | 0.6000 | 0.6000 | 0.6400 |
| Hybrid-E_AUPRC_guard_nested | 0.0041 | 0.0039 | -23.9619 | 0.0032 | 0.5200 | 0.5200 | 0.7200 |
| Hybrid-E_lowFP_nested | 0.0034 | 0.0028 | -41.2343 | 0.0005 | 0.5200 | 0.4800 | 0.7600 |
| Hybrid-E_hardBenign_FP375_nested | 0.0010 | 0.0008 | -5.7752 | -0.0005 | 0.4400 | 0.4400 | 0.6400 |
| Hybrid-E_FP375_buffer_nested | 0.0009 | 0.0000 | -20.1295 | -0.0001 | 0.4800 | 0.4800 | 0.7200 |
| Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested | -0.0003 | -0.0015 | -24.0076 | 0.0023 | 0.4000 | 0.3600 | 0.7600 |
| Hybrid-E_AUPRC_FP375_nested | -0.0009 | -0.0023 | -20.1295 | 0.0012 | 0.4000 | 0.4000 | 0.8000 |

## En İyi Aday Missingness Dağılımı

| missingness_bin | mean_missing_rate | f1_final_prior | mcc_final_prior | specificity | sensitivity | fp_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Q1_low_missing | 0.0418 | 0.4274 | 0.3996 | 0.9725 | 0.3167 | 0.0275 |
| Q2 | 0.3727 | 0.4141 | 0.3098 | 0.7214 | 0.6975 | 0.2786 |
| Q3 | 0.8501 | 0.5902 | 0.5306 | 0.8402 | 0.8199 | 0.1598 |
| Q4_high_missing | 0.9735 | 0.3542 | 0.2243 | 0.6789 | 0.6297 | 0.3211 |

## Yorum

- Bir challenger mevcut Hybrid-E'yi hem F1/MCC hem FP kuyruğu bakımından geçmiyorsa final freeze bozulmamalı.
- FP tamponu ortalama FP'yi düşürüp F1/MCC'yi belirgin azaltıyorsa sadece risk raporunda tutulmalı.
- Q2/Q4 fallback, missingness shortcut riskini azaltırsa bile yalnız bu nested sonuçla freeze adayı olabilir.
- Hard-benign ağırlığı güçlenmiş adaylar outer FP ID'lerinden öğrenmediği için yöntemsel olarak savunulabilir; karar tamamen bu rapordaki nested metriklere bağlıdır.

## Çıktılar

- `results/modeling/hybrid_improvement_governance/improvement_outer_fold_results.csv`
- `results/modeling/hybrid_improvement_governance/improvement_outer_predictions.csv`
- `results/modeling/hybrid_improvement_governance/improvement_selection_details.csv`
- `results/modeling/hybrid_improvement_governance/improvement_summary.csv`
- `results/modeling/hybrid_improvement_governance/improvement_missingness_subgroup_summary.csv`
- `results/modeling/hybrid_improvement_governance/improvement_delta_vs_current.csv`
- `results/modeling/hybrid_improvement_governance/improvement_delta_vs_catopt.csv`
- `results/modeling/hybrid_improvement_governance/improvement_governance_metadata.json`

## Run Metadata

- Outer seedler: `[42, 52, 62, 72, 82]`
- Outer fold sayısı: `25`
- Inner fold sayısı: `4`
- Hybrid-E weight vector sayısı: `2987`
- Süre saniye: `2554.2`
