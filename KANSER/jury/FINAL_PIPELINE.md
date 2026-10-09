# KANSER FINAL_PIPELINE

1. `test/` altinda tek CSV bulunur.
2. `run_all.py` `final/predict.py` dosyasini cagirir.
3. `final/predict.py` `final/models/KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib` artefaktini yukler.
4. Final preprocessing/model akisi joblib iceriginden uygulanir.
5. Sabit karar mantigi ile `prediction` ve `pathogenic_score` uretilir.
6. Cikti `output/KANSER_predictions.csv` dosyasina yazilir.
