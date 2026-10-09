# MASTER PAH-Style Governance Audit Report

Bu rapor PAH paketindeki kalite-guvence fikirlerini MASTER modelleme paketine uygular. Yeni bir final model egitmez; mevcut nested governance ve frozen artefact ciktisini denetler.

## Resmi Artefact Karari

- Primary official inference artefact: `results/modeling/final_hybrid_freeze/MASTER_Hybrid-E_bestMCC_nested_development_train_frozen.joblib`
- Prediction script: `scripts/master_predict_hybrid_e_frozen.py`
- Performance claim source: `results/modeling/nested_governance/HYBRID_NESTED_GOVERNANCE_REPORT.md`
- Accepted fallback: `results/modeling/final_hybrid_freeze/MASTER_Hybrid-C_nested_selected_all_labeled_frozen.joblib`

## Hybrid-E vs CATOPT-A Istatistikleri

| Test | Delta | Win rate | NB corrected SE | Normal approx p |
|---|---:|---:|---:|---:|
| F1 | +0.0056 | 0.60 | 0.0100 | 0.576 |
| MCC | +0.0058 | 0.60 | 0.0114 | 0.613 |
| FP/3500 | -25.9 | 0.72 | 43.5 | 0.552 |

Yorum: Hybrid-E ortalamada CATOPT-A'dan biraz iyi gorunuyor; fakat Nadeau-Bengio duzeltmesi kucuk deltalari temkinli yorumlamayi gerektirir.

## Fold Fragility

| Strategy | Strict pass | FP>400 | Sens<0.62 | F1<0.530 | Worst MCC | Max FP/3500 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid-E_bestMCC_nested | 12/25 | 12 | 3 | 11 | 0.3598 | 600.0 |
| CATOPT-A_nested_reference | 7/25 | 16 | 3 | 10 | 0.3488 | 619.0 |

## Missingness / Provenance

- Hybrid-E Q4 high-missing subgroup: F1=0.3462, MCC=0.2122, specificity=0.6789, FP/3500=963.2.
- Hybrid-E score Q4-vs-Q1 missingness separation AUC=0.7263.
- Missingness-signature-only label AUC=0.8035; bu deger provenance shortcut riskini canli tutar.

## Repeated FP / LOO Influence

- Hybrid-E repeated FP variant count: `106`
- Top LOO MCC improvement if removed: `+0.002097`
- LOO influence tablosu yeniden egitim yapmadan OOF tahminlerinden hesaplanir; row deletion karari yerine domain incelemesi icin oncelik listesi olarak kullanilmalidir.

## Output Files

- `results/modeling/final_hybrid_freeze/MASTER_ARTIFACT_REGISTRY.json`
- `results/modeling/pah_style_governance_audits/fold_fragility_summary.csv`
- `results/modeling/pah_style_governance_audits/hybrid_e_vs_catopt_nb_tests.csv`
- `results/modeling/pah_style_governance_audits/hybrid_e_vs_catopt_cluster_bootstrap_ci.csv`
- `results/modeling/pah_style_governance_audits/missingness_subgroup_stress.csv`
- `results/modeling/pah_style_governance_audits/missingness_adversarial_audit.csv`
- `results/modeling/pah_style_governance_audits/hybrid_e_repeated_false_positives.csv`
- `results/modeling/pah_style_governance_audits/hybrid_e_oof_loo_influence.csv`
