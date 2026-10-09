# Geçmiş Deneyler Tekrar Kontrolü

Bu kontrol, proje klasöründe kalan kodlar ve raporlar ile sohbet geçmişindeki kayıtlar birlikte değerlendirilerek hazırlanmıştır. Silinmiş ham sonuç dosyaları bulunan deneylerde rapor kanıtı esas alınmıştır.

| Yöntem | Durum | Kanıt / sonuç | Yeniden yapılacak mı? |
|---|---|---|---|
| Platt kalibrasyonu | Kesin denendi | Eski `ID3 + CatBoost + AdaBoost` ensemble: proj. F1 `0.6633 → 0.6162`, FP/100 `8.6 → 13.3` | Aynı eski yapı için hayır |
| İzotonik kalibrasyon | Kesin denendi | Eski ensemble: proj. F1 `0.5563`, MCC `0.4003`, FP/100 `13.3` | Hayır; küçük veri için ayrıca riskli |
| Güncel `ID3 + CatBoost + Random Forest` kalibrasyonu | Kanıt yok | Eski kalibrasyon AdaBoost'lu model üzerinde yapılmış | Teknik olarak yeni deney; düşük öncelik |
| ID3 hiperparametre ayarı | Kesin denendi | F1 `0.6034 → 0.5845`, FP `9.5 → 11.4`; reddedildi | Hayır |
| CatBoost hiperparametre ayarı | Kesin denendi | Tek modelde F1 `0.5437 → 0.5722`, FP `16.2 → 12.4`; fakat ensemble iyileşmedi | Hayır |
| AdaBoost hiperparametre ayarı | Kesin denendi | Tek model iyileşti; ayarlı ensemble ana modeli geçmedi | Hayır |
| Random Forest düzenlileştirme/parametre seçimi | Kesin denendi | Nested seçim dosyaları ve tahminleri mevcut; RF ensemble'a tamamlayıcı katkı verdi | Hayır |
| Sabit/dinamik voting ağırlıkları | Kesin denendi | Dinamik ağırlık FP'yi yaklaşık `14.3` yaptı; yakın tarihli güncel üçlü nested ağırlık deneyi de F1 `0.6570`, FP `9.52` verdi | Hayır |
| Hard ve soft voting kombinasyonları | Kesin denendi | Çok sayıda ikili/üçlü/dörtlü yapı karşılaştırıldı | Hayır |
| Bileşen ablation | Kesin denendi | Güncel üçlüde tekil, ikili ve üçlü yapılar yeniden karşılaştırıldı | Hayır |
| Lojistik stacking | Kesin denendi | F1 `0.6005`, MCC `0.4791`, FP `14.3`; reddedildi | Hayır |
| İki aşamalı benign hibrit | Kesin denendi | Ana modelle aynı sonucu verdi; ek karmaşıklık nedeniyle reddedildi | Hayır |
| Hard-negative CatBoost | Kesin denendi | FP `16.2 → 18.1`; reddedildi | Hayır |
| CatBoost seed bagging | Kesin denendi | Ensemble FP `8.6 → 12.4`; reddedildi | Hayır |
| Eşik stratejileri | Kesin denendi | Projeksiyon-F1, specificity kısıtlı, MCC, Macro-F1 ve sabit `0.50` karşılaştırıldı | Hayır |
| Eşik hassasiyeti | Kesin denendi | Güncel cross-fitted yapıda ±`0.02` analizi yapıldı | Hayır |
| 3/4/5 tekrar-model bagging | Kesin denendi | Beş model en kararlı sonuç verdi | Hayır |
| Eksiklik göstergeleri | Kesin denendi | F1 `0.6090`, FP `12.38`; reddedildi | Hayır |
| Özellik ailesi late fusion | Kesin denendi | En iyi F1 `0.6624`, FP `9.52`; reddedildi | Hayır |
| Robust RBF-SVM | Kesin denendi | MCC `-0.0117`, FP `73.33`; reddedildi | Hayır |
| Label-shift/prior düzeltmesi | Kesin denendi | En iyi F1 `0.6511`, FP `10.48`; reddedildi | Hayır |
| Conformal güven analizi | Başlatılmış fakat tamamlanmamış | Doğrulanmış sonuç yok | Modeli iyileştirmez; yalnız belirsizlik raporu için düşünülebilir |

## Karar

Önceden önerilen kalibrasyon ve hiperparametre çalışmalarının çoğu geçmişte zaten uygulanmıştır. Aynı deneyleri tekrarlamak, küçük veri üzerinde tekrar tekrar seçim yaparak validasyon overfitting riskini artırır. Bu nedenle bunlar yeniden çalıştırılmayacaktır.

Güncel Random Forest'lı ensemble üzerinde kalibrasyon teknik olarak denenmemiştir; ancak eski kalibrasyon sonuçları FP'yi belirgin artırdığı ve veri yalnız 21 benign örnek içerdiği için bu deney düşük öncelikli kabul edilmiştir. Bağımsız bir doğrulama seti olmadan yeni bir geniş ayar turuna girilmemesi daha güvenlidir.
