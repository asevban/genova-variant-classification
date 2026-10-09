# TabPFN API Smoke Test and Hybrid-E Blend Screening

Bu rapor yalnız ara tarama çıktısıdır. TabPFN API ile üretilen seed=42, 5-fold OOF skorları kullanıldı; final/test etiketi kullanılmadı. Ağırlık ve threshold kararları burada outer OOF üzerinde seçildiği için bu bölüm resmi final governance kanıtı değildir.

## TabPFN Threshold Kontrolü

| case | available | threshold | final_f1 | final_mcc | final_specificity | final_sensitivity | expected_fp_per_3500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| threshold_0_50 | True | 0.5000 | 0.3863 | 0.3217 | 0.5032 | 0.9529 | 1490.4153 |
| best_mcc_overall | True | 0.8150 | 0.5280 | 0.4456 | 0.8450 | 0.6923 | 464.8562 |
| fp_le_400_best_mcc | True | 0.8440 | 0.5251 | 0.4393 | 0.8674 | 0.6393 | 397.7636 |
| strict_gate_best_mcc | False |  |  |  |  |  |  |

## Seed-42 Governance Referansı

| candidate | folds | f1 | mcc | specificity | sensitivity | expected_fp_per_3500 | final_weighted_auprc | strict_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_bestMCC_nested | 5 | 0.5241 | 0.4385 | 0.8562 | 0.6567 | 431.3905 | 0.5014 | 0.6000 |
| Hybrid-E_AUPRC_guard_nested | 5 | 0.5232 | 0.4378 | 0.8546 | 0.6591 | 436.1524 | 0.5098 | 0.4000 |
| Hybrid-E_lowFP_nested | 5 | 0.5243 | 0.4388 | 0.8642 | 0.6433 | 407.3524 | 0.5027 | 0.4000 |
| CATOPT-A_nested_reference | 5 | 0.5270 | 0.4427 | 0.8674 | 0.6370 | 397.7524 | 0.5083 | 0.2000 |

## Hybrid-E + TabPFN Küçük Ağırlık Taraması

Aşağıdaki tablo FP <= 400 ve sensitivity >= 0.62 koşulu altında en iyi post-hoc karışımları gösterir.

| base_strategy | tabpfn_weight | selection_status | threshold | final_f1 | final_mcc | final_specificity | final_sensitivity | expected_fp_per_3500 | final_weighted_auprc | delta_final_mcc | delta_expected_fp_per_3500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid-E_AUPRC_guard_nested | 0.3000 | fp_forced | 0.8360 | 0.5208 | 0.4337 | 0.8706 | 0.6254 | 388.1789 | 0.4883 | 0.0045 | -9.5847 |
| Hybrid-E_AUPRC_guard_nested | 0.2500 | fp_forced | 0.8350 | 0.5200 | 0.4327 | 0.8722 | 0.6207 | 383.3866 | 0.4887 | 0.0035 | -14.3770 |
| Hybrid-E_AUPRC_guard_nested | 0.2000 | fp_forced | 0.8310 | 0.5176 | 0.4299 | 0.8690 | 0.6236 | 392.9712 | 0.4888 | 0.0007 | -4.7923 |
| Hybrid-E_AUPRC_guard_nested | 0.0000 | fp_forced | 0.8240 | 0.5170 | 0.4291 | 0.8674 | 0.6259 | 397.7636 | 0.4874 | 0.0000 | 0.0000 |
| Hybrid-E_AUPRC_guard_nested | 0.0500 | fp_forced | 0.8290 | 0.5165 | 0.4285 | 0.8690 | 0.6219 | 392.9712 | 0.4882 | -0.0006 | -4.7923 |
| Hybrid-E_AUPRC_guard_nested | 0.1000 | fp_forced | 0.8280 | 0.5159 | 0.4278 | 0.8674 | 0.6242 | 397.7636 | 0.4886 | -0.0013 | 0.0000 |
| Hybrid-E_AUPRC_guard_nested | 0.0250 | fp_forced | 0.8270 | 0.5152 | 0.4269 | 0.8674 | 0.6230 | 397.7636 | 0.4879 | -0.0022 | 0.0000 |
| Hybrid-E_AUPRC_guard_nested | 0.0750 | fp_forced | 0.8290 | 0.5149 | 0.4265 | 0.8674 | 0.6225 | 397.7636 | 0.4885 | -0.0026 | 0.0000 |

## Strict Geçen Karışımlar

Strict koşulunu geçen Hybrid-E + TabPFN karışımı bulunmadı.

## Kısa Karar

TabPFN tek başına güçlü sensitivity/F1 sinyali verdi, ancak benign-heavy hedefte FP yükü yüksek kaldı. FP <= 400 zorlandığında TabPFN MCC/F1 düşüyor. Küçük ağırlıklı Hybrid-E karışımları bu seed-42 taramada sınırlı iyileşme gösterebilse de, resmi aday yapılmadan önce TabPFN skorunun inner seçimli nested governance akışına dahil edilmesi gerekir.
