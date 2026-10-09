# MASTER Hybrid-E Strengthening Audit Raporu

Bu rapor, freeze edilen Hybrid-E adayını yeni bir model araması açmadan güçlendirmek için yapılan güvenlik ve stabilite kontrollerini özetler.

## Uygulanan Adımlar

| step | status | governance_note |
| --- | --- | --- |
| Frozen Hybrid-E artefact | applied | Frozen Hybrid-E procedure instantiated for final/test inference. |
| Inner-CV final selection | applied | Weights, calibration/prior correction and threshold selected only inside inner CV. |
| Calibration/Brier audit | applied | Existing nested predictions summarized; no new model search. |
| FP error analysis | applied | Repeated false positives identified from nested outer predictions. |
| Missingness subgroup stability | applied | Nested predictions checked across missingness quartiles. |
| CATOPT-A paired comparison | applied | Same outer seed/fold pairs compared against reference. |
| Multi-seed bagging | not_applied_to_final | Would be a new model variant; should be nested-validated before final use. |

## Calibration / Brier Özeti

| strategy | brier | final_prior_weighted_brier | log_loss | ece_10bin | mean_score |
| --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 0.2434 | 0.1750 | 0.7213 | 0.2397 | 0.4934 |
| Hybrid-E_lowFP_nested | 0.2153 | 0.1901 | 0.6485 | 0.1922 | 0.5408 |
| Hybrid-E_AUPRC_guard_nested | 0.2090 | 0.1992 | 0.6285 | 0.1773 | 0.5557 |
| CATOPT-A_nested_reference | 0.1321 | 0.2317 | 0.4268 | 0.0804 | 0.6682 |

## CATOPT-A Paired Fark Özeti

| strategy | delta_f1_mean | delta_mcc_mean | delta_fp_per_3500_mean | delta_final_auprc_mean | f1_win_rate | mcc_win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 0.0056 | 0.0058 | -25.8819 | -0.0005 | 0.6000 | 0.6000 |
| Hybrid-E_AUPRC_guard_nested | 0.0041 | 0.0039 | -23.9619 | 0.0032 | 0.5200 | 0.5200 |
| Hybrid-E_lowFP_nested | 0.0034 | 0.0028 | -41.2343 | 0.0005 | 0.5200 | 0.4800 |

## Fold Stabilitesi

| strategy | strict_pass_rate | f1_mean | f1_std | mcc_mean | mcc_std | fp_per_3500_mean | fp_per_3500_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 0.4800 | 0.5326 | 0.0353 | 0.4493 | 0.0424 | 413.0971 | 84.0115 |
| Hybrid-E_AUPRC_guard_nested | 0.4000 | 0.5311 | 0.0349 | 0.4474 | 0.0420 | 415.0171 | 86.1304 |
| Hybrid-E_lowFP_nested | 0.3600 | 0.5304 | 0.0307 | 0.4463 | 0.0373 | 397.7448 | 70.1193 |
| CATOPT-A_nested_reference | 0.2800 | 0.5270 | 0.0373 | 0.4435 | 0.0452 | 438.9790 | 98.7345 |

## Missingness Subgroup Özeti

| missingness_bin | mean_missing_rate | f1_final_prior | mcc_final_prior | specificity | sensitivity | fp_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Q1_low_missing | 0.0423 | 0.4289 | 0.4012 | 0.9726 | 0.3179 | 0.0274 |
| Q2 | 0.3761 | 0.4120 | 0.3076 | 0.7164 | 0.7011 | 0.2836 |
| Q3 | 0.8231 | 0.6770 | 0.6235 | 0.9078 | 0.7948 | 0.0922 |
| Q4_high_missing | 0.9702 | 0.3469 | 0.2198 | 0.6067 | 0.7049 | 0.3933 |

## FP Hata Analizi

Hybrid-E_bestMCC_nested için en az bir outer değerlendirmede FP olan benzersiz varyant sayısı: **106**.

En sık tekrar eden FP örnekleri ayrı CSV dosyasında saklandı.

## Çıktılar

- `results/modeling/final_hybrid_freeze/hybrid_e_strengthening_actions.csv`
- `results/modeling/final_hybrid_freeze/hybrid_e_calibration_brier_summary.csv`
- `results/modeling/final_hybrid_freeze/hybrid_e_paired_delta_vs_catopt.csv`
- `results/modeling/final_hybrid_freeze/hybrid_e_fold_stability_summary.csv`
- `results/modeling/final_hybrid_freeze/hybrid_e_missingness_subgroup_summary.csv`
- `results/modeling/final_hybrid_freeze/hybrid_e_fp_error_analysis.csv`
