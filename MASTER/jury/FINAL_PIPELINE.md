# MASTER FINAL_PIPELINE

1. `test/` altinda tek CSV bulunur.
2. `run_all.py` `final/predict.py` dosyasini cagirir.
3. `final/predict.py` `final/models/MASTER_Hybrid-E_bestMCC_nested_development_train_frozen.joblib` artefaktini yukler.
4. `master_preprocessing.py` ayni `final/` klasorunden kullanilir.
5. `predict.py` icindeki frozen artefakt skorlayici ile sabit preprocessing, feature alignment ve model tahmini uygulanir.
6. Cikti `output/MASTER_predictions.csv` dosyasina yazilir.
