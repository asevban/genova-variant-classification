# Reddedilen Yöntemler — Nihai Özet

Referans ana model: 319 özellikli `ID3 + CatBoost + Random Forest`, Macro-F1 `0.7190`, MCC `0.5225`, projeksiyon F1 `0.6798`, FP/100 benign `8.57`.

| Deney | Temel sonuç | Reddedilme nedeni |
|---|---|---|
| `CAT_6` eklenerek 320 özellik | MCC 0.4756, proj. F1 0.6309, FP 10.48 | Aynı ana modelde bütün temel metrikler kötüleşti. |
| Ek Information Gain azaltması | Fold'lar 100–319 arasında kararsız seçim yaptı | Genellenebilir sabit kolon sayısı oluşmadı; CatBoost ve FP kötüleşti. |
| Platt/izotonik kalibrasyon | MCC/F1 artmadı | FP yükseldi veya genel başarı düştü. |
| MCC/Macro-F1 odaklı eşik | MCC 0.5144, FP 32.4 | Küçük MCC artışı çok büyük yanlış pozitif maliyeti oluşturdu. |
| Ayarlı ID3 | MCC 0.4207, proj. F1 0.5845, FP 11.4 | Baseline ID3'ten kötüydü. |
| Ayarlı CatBoost/AdaBoost ensemble | Ana modeli geçmedi | Tek model iyileşmeleri ensemble'a taşınmadı. |
| Dinamik/sabit farklı voting ağırlıkları | Dinamik FP yaklaşık 14.3 | Fold kararsızlığı ve düşük projeksiyon F1. |
| Lojistik stacking | MCC 0.4791, proj. F1 0.6005, FP 14.3 | Recall artışı FP ve F1 kaybını karşılamadı. |
| İki aşamalı hibrit | Ana modeli geçmedi | Küçük benign örneklemde ikinci aşama kararlı öğrenilemedi. |
| Hard-negative CatBoost | FP 16.2'den 18.1'e çıktı | Amaçlanan FP azalması oluşmadı. |
| CatBoost seed bagging | Ensemble FP 8.6'dan 12.4'e çıktı | Hata çeşitliliği azaldı. |
| Logistic Regression | MCC 0.0478, proj. F1 0.2872, FP 71.4 | Veri ilişkileri doğrusal model için uygun değildi. |
| Dört modelli ensemble | Üçlü ana modeli geçmedi | AdaBoost eklenmesi yararlı çeşitlilik sağlamadı. |
| Eksiklik göstergeleri | MCC 0.4674, proj. F1 0.6090, FP 12.38 | 16 ek gösterge ana ensemble'ı kötüleştirdi. |
| Özellik ailesi late fusion | En iyi benign hedef: proj. F1 0.6624, FP 9.52 | Bazı klasik metrikler artsa da benign-ağırlıklı hedef geriledi. |
| Robust RBF-SVM | Tek SVM MCC −0.0117, FP 73.33 | Tek başına ve ensemble içinde başarısız oldu. |
| Label-shift/prior düzeltmesi | En iyi düzeltilmiş proj. F1 0.6511, FP 10.48 | Mevcut fold-içi eşik hedef prior'ını zaten daha iyi yönetiyordu. |
| TabPFN | Bağımsız ön tarama | Nihai modele ait değildi; yalnız dış karşılaştırma deneyi olarak değerlendirildi. |
| Conformal güven analizi | Tamamlanmadı | Çalıştırılmış ve doğrulanmış sonuç üretmedi; nihai karara dahil edilmedi. |

Sonuç: Bu deneylerin hiçbiri referans modelin F1–FP dengesini güvenilir biçimde aşmadı. Nihai model değiştirilmedi.

