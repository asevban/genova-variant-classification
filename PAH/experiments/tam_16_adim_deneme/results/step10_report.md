# Adım 10 — Çok Güçlü Çiftleri Grafikle İnceleme

**Eşik notu:** Spearman `|r|≥0.90` eşiğini geçen çift **yok** (Adım 6 —
Faz 2'nin ön-bilgisiyle tutarlı, genişletilmiş kapsamda da doğrulandı).
Görev talimatına uygun şekilde eşik düşürüldü — **gözlenen en yüksek
Spearman değerinin (0.777) altına
inmeden**, en güçlü 4 çift seçildi.

## Çift Özeti

        pair  common_n  pearson_r  spearman_r en_iyi_form_R2  en_iyi_R2
AL_88-AL_121       123   0.962621    0.776947         linear   0.926640
   EK_7-EK_9       361   0.842766    0.774410         linear   0.710255
AL_23-AL_283        85   0.981941    0.766934         linear   0.964209
 AL_7-AL_103       169   0.990019    0.737979         linear   0.980137

## Eğri Uydurma Karşılaştırması (yalnızca gerçek ortak gözlemler, `v1`'in ham NaN'ı üzerinden)

        pair                      form                   params  r2_original_scale  loocv_rmse_original_scale  loocv_rmse_log_scale
AL_88_AL_121                    linear    a=1.689e-06, b=0.8639           0.926640                   0.000048                   NaN
AL_88_AL_121         log (y=a+b*ln(x))  a=0.0001574, b=7.82e-06           0.073159                   0.000161                   NaN
AL_88_AL_121 exponential (y=a*exp(bx))      a=6.075e-06, b=5934       -5154.107525                        NaN              2.168093
AL_88_AL_121           power (y=a*x^b)    a=9.595e-05, b=0.1848          -0.036339                        NaN              1.924192
   EK_7_EK_9                    linear         a=0.995, b=1.076           0.710255                   2.085856                   NaN
AL_23_AL_283                    linear    a=1.861e-06, b=0.9579           0.964209                   0.000035                   NaN
AL_23_AL_283         log (y=a+b*ln(x))  a=0.000227, b=1.312e-05           0.101130                   0.000174                   NaN
AL_23_AL_283 exponential (y=a*exp(bx))      a=8.851e-07, b=9489     -636114.260947                        NaN              8.050415
AL_23_AL_283           power (y=a*x^b)    a=0.0001868, b=0.3916          -0.122556                        NaN              7.628238
 AL_7_AL_103                    linear   a=-1.037e-06, b=0.9707           0.980137                   0.000017                   NaN
 AL_7_AL_103         log (y=a+b*ln(x)) a=0.0004835, b=3.814e-05           0.309963                   0.000102                   NaN
 AL_7_AL_103 exponential (y=a*exp(bx))      a=5.162e-06, b=8149       -6200.762616                        NaN              1.635108
 AL_7_AL_103           power (y=a*x^b)       a=0.07128, b=0.787           0.516119                        NaN              1.073418

## Sonuç — Dönüşümlü Model Öneriliyor mu?

Görev kuralı: dönüşümlü model yalnızca LOOCV hatasını belirgin azaltıyor,
orijinal ölçekte geçerli R² üretiyor, VE birkaç uç gözleme bağlı değilse
önerilir. Yukarıdaki tabloda hiçbir çift için log/üstel/kuvvet formu,
lineer forma göre **belirgin bir R² iyileşmesi göstermiyor** (fark <0.05
tüm çiftlerde) — **hiçbir dönüşüm bu turda önerilmiyor**, mevcut lineer
varsayım korunuyor. Grafikler: `figures/step10_*.png`.
