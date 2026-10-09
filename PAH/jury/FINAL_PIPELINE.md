# PAH FINAL_PIPELINE

1. `test/` altinda tek CSV bulunur.
2. `run_all.py` panel icin `final/predict.py` cagirir.
3. `final/predict.py` `final/models/final_model_bundle_v2.pkl` dosyasini yukler.
4. Sema isim bazli dogrulanir.
5. Dondurulmus preprocessing ve 26 ozellikli pool uygulanir.
6. Model skoru, calibration ve prior correction sonrasi sabit esik ile etikete cevrilir.
7. Cikti `output/PAH_predictions.csv` dosyasina yazilir.
