# MASTER MODEL_SELECTION

Final model: `final/models/MASTER_Hybrid-E_bestMCC_nested_development_train_frozen.joblib`

Secim gerekcesi: Resmi final model Hybrid-E bestMCC nested development_train frozen hattidir. Kaynak secim notlarinda threshold 0.337 ve M3 missing-aware compact ozellik seti korunmustur. Raporlanan gelistirme sonucu F1 0.5326 +/- 0.0353, specificity 0.8623 +/- 0.0280, MCC 0.4493 +/- 0.0424 ve AUROC 0.8525'tir.

Denenen aileler/hatlar: Hybrid-C/E, XGBoost, CatBoost, LightGBM, TabPFN ve MCC/FP odakli hedefli calismalar `experiments/` ve `results/` altina ayrildi.

Arsivlenen modeller: 4 non-primary/legacy serialized model `archive/` altindadir. Final inference bunlari kullanmaz.
