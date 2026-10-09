# Model Kartı — CFTR 319 Özellikli Ensemble

## Amaç

Klinik etkisi bilinmeyen CFTR varyantlarını, yarışma komitesinin sağladığı biyolojik ve hesaplamalı özelliklerden yararlanarak `benign (0)` veya `patojenik (1)` şeklinde tahmin etmek. Model klinik tanı koymaz; hesaplamalı karar desteği üretir.

## Nihai yapı

- ID3: 21 ağaç, maksimum derinlik 4, dengeli bootstrap.
- CatBoost: 40 iterasyon, derinlik 4, öğrenme oranı 0.05, dengeli sınıf ağırlığı.
- Random Forest: 150 ağaç, maksimum derinlik 5, minimum yaprak 3, `balanced_subsample`.
- Birleştirme: eşit ağırlıklı soft voting.
- Üretim: her bileşenin beş seed modeli ortalanır.
- Sabit karar eşiği: `0.7224`.
- Yedek: beş ID3 modelinin ortalaması, eşik `0.780952`.

## Doğrulama

- Düzen: dışta stratified 5-fold × 5 tekrar; eşik ve fold-içi dönüşümler yalnız dış eğitim tarafında 4-fold iç CV ile belirlenmiştir.
- Pooled nested-CV: projekte F1 `0.6798`, MCC `0.5225`, recall `0.7356`, specificity `0.9143`, FP/100 benign `8.57`.
- Cross-fitted varyant ortalaması: projekte F1 `0.7580`, MCC `0.5717`, recall `0.7556`, specificity `0.9524`, FP/100 benign `4.76`.
- Cross-fitted bootstrap %95 GA: F1 `0.5864–0.9024`, MCC `0.4550–0.6923`, recall `0.6667–0.8444`, FP/100 `0–14.29`.

`0.6798` bağımsız nested-CV ana sonucu; `0.7580` farklı bir tahmin toplama stratejisinin destekleyici sonucudur.

## Kalibrasyon

- Cross-fitted varyant ortalaması Brier Score: `0.1092`.
- 10 eşit genişlikli kutuda ECE: `0.1526`.
- Olasılıklar kusursuz kalibre değildir; yeni kalibrasyon seçimi yapılmamış ve eşik değiştirilmemiştir.

## Sağlamlık ve sızıntı denetimleri

- Tam veya %98 üzerinde benzer özellik satırı bulunmadı.
- 100.000 etiket permütasyonunda MCC ortalaması yaklaşık sıfıra düştü; `p=0.000010`.
- Yalnız eksiklik maskesi anlamlı sinyal taşıdı: Logistic Regression ROC-AUC `0.8592`. Bu durum final verisinde eksiklik driftinin özellikle izlenmesini gerektirir.
- Tekil/ikili/üçlü bileşen ablasyonu, fold/seed kırılganlığı ve eşik hassasiyeti raporlandı.

## Açıklanabilirlik

CatBoost SHAP analizinde en yüksek ortalama mutlak katkılar `EK_9`, `EK_7`, `AL_298`, `AL_327` ve `AL_7` özelliklerinde görüldü. Bu SHAP sonuçları yalnız CatBoost bileşenini açıklar ve özellik seçmek için kullanılmaz.

## Sınırlılıklar

- Eğitim verisi yalnız 111 varyant ve 21 benign örnek içerir; özellikle FP tahmini belirsizdir.
- İlk `351 → 322` özellik elemesi tam nested süreç içinde tekrarlanmadı.
- Eksiklik desenleri sınıfla ilişkili olduğundan farklı veri hazırlama hattında genelleme riski vardır.
- SHAP sonuçları nedensellik göstermez.
- Model bağımsız klinik karar veya hasta yönetimi için kullanılmamalıdır.

## Yarışma kullanım kuralı

Test etiketi görülmeden dondurulmuş ana model ve `0.7224` eşiği kullanılır. Yedek ID3 yalnız teknik hata halinde devreye alınır. Test verisi geldiğinde sütun, eksiklik, yeni kategori ve dağılım drift raporu kontrol edilir; test sonucuna göre model seçilmez.
