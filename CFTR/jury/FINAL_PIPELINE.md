# CFTR FINAL_PIPELINE

1. `test/` altinda tek CSV bulunur.
2. `run_all.py` `final/predict.py --mode main` cagrisi yapar.
3. Script `final/models/artifacts` altindaki 15 base model artefaktini ve `preprocessing.json` dosyasini yukler.
4. Dondurulmus preprocessing uygulanir.
5. ID3, CatBoost ve RandomForest repeat modelleri score uretir.
6. Soft voting ve sabit esik ile `Label` uretilir.
7. Cikti `output/CFTR_predictions.csv`, audit `output/CFTR_audit.csv`, drift raporu `output/CFTR_drift.json` olarak yazilir.
