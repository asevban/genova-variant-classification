# Yetkili Nihai Model Kimliği

Bu dosya, projedeki nihai model konusunda **tek yetkili kaynaktır**. Eski raporlardaki “aday”, “guardrail” veya AdaBoost ifadeleri tarihsel deney sonuçlarını anlatır.

## Ana yarışma modeli

- Kimlik: `CFTR_319_ID3_CATBOOST_RF_EQUAL_SOFT_VOTING_5SEED`
- Veri şeması: 319 özellik
- Bileşenler: ID3, CatBoost, Random Forest
- Birleştirme: her bileşende beş seed olasılığının ortalaması; ardından üç bileşenin eşit ortalaması
- Dondurulmuş eşik: `0.7224`
- Paket: `final/`
- Tahmin girişi: `final/predict.py`
- Çıktı doğrulaması: `final/verify_submission.py`

Model artefaktları seri hale getirilmiş ve paketlenmiştir: 5 ID3 JSON, 5 CatBoost CBM ve 5 Random Forest Joblib.

## Teknik yedek

- Kimlik: `CFTR_319_ID3_5SEED`
- Model: yalnız ID3, beş seed ortalaması
- Eşik: `0.780952380952381`
- Paket: ana teslimde ayrica paketlenmedi; yedek kimlik tarihsel kayit olarak korunur
- Kullanım: yalnız ana model teknik olarak çalışmazsa

`ID3 + CatBoost + AdaBoost` üretim modeli değildir; yalnız geçmiş karşılaştırma raporlarında bulunan tarihsel adaydır.

## Doğrulama durumu

- 111 eğitim örneği: 21 benign, 90 patojenik
- CV kanıtı: mevcut
- Eşik seçimi kanıtı: mevcut; her outer fold'un yalnız eğitim tarafındaki inner-CV ile
- Dondurulmuş üretim modeli: mevcut
- Etiketsiz final inference: smoke test ile doğrulandı
- Tekrarlı çalıştırma: aynı SHA256 tahmin çıktısı
- Ana nested-CV kanıtı: projekte F1 `0.6798`, MCC `0.5225`, FP/100 benign `8.57`

## Yarışma komutu

```powershell
powershell -ExecutionPolicy Bypass -File .\run_competition_inference.ps1 -TestCsv .\test.csv -OutputCsv .\tahmin.csv
```

Alternatif olarak panel kokundeki guncel `run_competition_inference.ps1` komutu kullanilabilir.

Test etiketi görüldükten sonra model, eşik, özellik listesi veya ana/yedek kimliği değiştirilmeyecektir.
