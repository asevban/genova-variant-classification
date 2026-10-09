# Adım 8 — Korelasyon + IG Birlikte Değerlendirme (Redundan Çift Adayları)

Spearman `|r|≥0.7` olan **14 çift**
bulundu (Adım 6). Her çiftin IG'si (Adım 2, betimsel) karşılaştırıldı;
**IG farkı >0.005 olan 9 kolon**, çiftinin daha
zayıf tarafı olarak **aday** işaretlendi (kesin eleme değil).

**Önemli:** İki sütun da otomatik çıkarılmadı — her çiftte yalnızca daha
düşük IG'li taraf aday, güçlü taraf tutulması öneriliyor. Bu adaylar Adım 13'te
nested karşılaştırmayla (tam 405 vs bu aday-set) test edilecek.

## En Belirgin IG Farkına Sahip 15 Çift

 col_i  col_j  spearman_r     IG_i     IG_j aday_cikarilacak_zayif_taraf
AL_117 AL_126    0.729516 0.004720 0.028230                       AL_117
AL_188 AL_260    0.721018 0.006692 0.022380                       AL_188
  AL_7 AL_103    0.737979 0.025938 0.011323                       AL_103
AL_224 AL_260    0.721657 0.008741 0.022380                       AL_224
  EK_7   EK_9    0.774410 0.062911 0.050860                         EK_9
 AL_23 AL_283    0.766934 0.014288 0.021120                        AL_23
 AL_88 AL_121    0.776947 0.018725 0.024710                        AL_88
AL_247 AL_283    0.718254 0.015849 0.021120                       AL_247
 AL_41  AL_47    0.702500 0.010087 0.005055                        AL_47
  EK_2   EK_3    0.713741 0.033020 0.028953                         EK_3
AL_121 AL_283    0.729135 0.024710 0.021120                       AL_283
 AL_88 AL_112    0.708194 0.018725 0.016151                       AL_112
 AL_88 AL_283    0.703569 0.018725 0.021120                        AL_88
AL_112 AL_145    0.701086 0.016151 0.015202                       AL_145

Tam liste: `results/step8_redundancy_candidates.csv`.
