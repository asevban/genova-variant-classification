# CFTR MODEL_SELECTION

Final artefakt dizini: `final/models/artifacts`

Secim gerekcesi: Resmi final hat ID3 + CatBoost + RandomForest soft voting ensemble olarak korunmustur. Final sette 5 ID3 JSON, 5 CatBoost CBM ve 5 RandomForest joblib model artefakti vardir. Mevcut raporlarda ana esik 0.7224, nested-CV projected F1 0.6798, MCC 0.5225, specificity 0.9143, recall 0.7356 ve FP/100 benign 8.57 olarak yer alir.

Denenen aileler/hatlar: ID3, CatBoost, RandomForest, SVM, missingness, label shift, threshold stability, SHAP/calibration ve ensemble calismalari `experiments/` ve `results/` altina ayrildi.

Arsivlenen modeller: Final disi serialized model artefakti bu teslimde ayrica tespit edilmedigi icin sayi 0'dur.
