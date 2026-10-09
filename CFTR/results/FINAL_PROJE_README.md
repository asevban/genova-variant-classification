# CFTR Projesi — Nihai Çalışma Hattı

> **YETKİLİ NİHAİ KİMLİK:** Yarışmada kullanılacak tek ana model, `final_package/` içindeki 319 özellikli `ID3 + CatBoost + Random Forest` eşit soft-voting modelidir. Sabit eşik `0.7224`'tür. Tek teknik yedek, `final_package_backup/` içindeki yalnız-ID3 modelidir. Eski AdaBoost raporları tarihsel deney kanıtıdır ve üretimde kullanılmaz.

## Güncel veri seti

- Dosya: `YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv`
- Gözlem: 111
- Model özelliği: 319
- Kimlik: `Variant_ID`
- Hedef: `Label`
- `CAT_6` nihai veri hattında yoktur.
- `AL_6–AL_251` ve `AL_1–AL_211` çiftleri, yalnız eğitim fold'undan öğrenilen medyan/IQR parametreleriyle robust standartlaştırılıp iki birleşik kolona dönüştürülür.

Fold-içi yeniden üretim kaynağı `YARISMA_TRAIN_CFTR_REDUCED.csv`, sabit tam-veri çıktısı ise `YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv` dosyasıdır. Çapraz doğrulamada veri sızıntısını önlemek için model kodları birleşik kolon parametrelerini her eğitim fold'unda yeniden öğrenir.

## Nihai ana model

`ID3 + CatBoost + Random Forest — eşit ağırlıklı soft voting`

- Değerlendirme: 5-fold × 5 tekrar nested CV
- Eşik seçimi: Her dış fold'un 4-fold iç-CV tahminleri
- Hedef dağılım: 100 benign + 20 patojenik

| Metrik | Sonuç |
|---|---:|
| Accuracy | 0.7694 |
| Specificity | 0.9143 |
| Recall | 0.7356 |
| Macro-F1 | 0.7190 |
| MCC | 0.5225 |
| Projeksiyon F1 | 0.6798 |
| FP / 100 benign | 8.57 |

## Yedek model

`Yalnız ID3 — beş seed ortalaması, sabit eşik 0.780952`

Bu model yalnız ana pakette teknik çalışma hatası oluşursa kullanılır. `ID3 + CatBoost + AdaBoost` önceki bir guardrail adayıdır; güncel yedek veya ana model değildir.

## Temel kodlar

- `run_final_pipeline.ps1`: Araştırma sonuçlarını yeniden üreten değerlendirme hattıdır; yarışma tahmin komutu değildir.
- `run_competition_inference.ps1`: Dondurulmuş üretim artefaktlarıyla etiketsiz test dosyasından yarışma çıktısı üretir ve doğrular.
- `build_scenario1d_319_dataset.js`: 319 özellikli sabit veri çıktısını üretir.
- `id3_benign_weighted_322_vs_320.js`: ID3 nested-CV tahminlerini üretir; nihai senaryo anahtarı `id3_scenario1d_319`.
- `catboost_benign_weighted_finalists_nested_cv.py`: CatBoost tahminlerini üretir; nihai senaryo anahtarı `scenario1_robust_groups_no_cat6`.
- `evaluate_randomforest_319_nested.py`: Random Forest nested-CV tahminlerini üretir.
- `evaluate_319_randomforest_hybrids.py`: Nihai ana ve yedek ensemble karşılaştırmasını üretir.
- `sklearn_scenario1c_320_nested_cv.py 319`: Yalnız tarihsel AdaBoost karşılaştırmasını üretir; üretim tahmininde kullanılmaz.
- `bootstrap_final_f1_superiority.js`: Variant_ID kümeli 100.000 tekrarlı ana–yedek F1 üstünlük denetimini üretir.

## Nihai kanıtlar

Ana özet `results/GUNCEL_319_VERI_SETI_VE_SECILEN_ENSEMBLE.md`, geliştirme günlüğü `results/319_MODEL_GELISTIRME_ADIMLARI_RAPORU.md`, elenen deneyler ise `results/REDDEDILEN_YONTEMLER_OZETI.md` dosyasındadır.
F1 üstünlük güven aralığı `results/FINAL_F1_USTUNLUGU_BOOTSTRAP_RAPORU.md` dosyasındadır.
Yeni geliştirme serisinin özeti `results/YENI_MODEL_GELISTIRME_SERISI_SONUCLARI.md` dosyasındadır. Bu seride model ailesi değişmemiş; cross-fitted bagging güçlü final tahmin stratejisi adayı olmuştur.
