# Adım 7 — Kategorik Özellikler İçin Uygun İlişki Ölçüleri

Kategoriler **rastgele tamsayı kodlanıp Pearson hesaplanmadı** — sahte
sıralama riski nedeniyle tüm ölçümler kategorik veri için uygun, sıralama-
bağımsız yöntemlerle yapıldı: Cramér's V (kategorik-kategorik / kategorik-
hedef), Eta/correlation ratio (kategorik-sayısal), Mutual Information
(kategorik-hedef).

## Kategorik-Kategorik: En Güçlü İlişkiler

col_i col_j  cramers_v       p_value   n
CAT_3 CAT_4   1.000000 1.939034e-307 372
CAT_4 CAT_5   1.000000 1.939034e-307 372
CAT_3 CAT_5   1.000000 1.939034e-307 372
CAT_5  AA_1   0.679171 2.985055e-111 372
CAT_4  AA_1   0.679171 2.985055e-111 372
CAT_3  AA_1   0.679171 2.985055e-111 372
CAT_5  AA_2   0.527202  4.595292e-60 372
CAT_4  AA_2   0.527202  4.595292e-60 372
CAT_3  AA_2   0.527202  4.595292e-60 372
CAT_1 CAT_2   0.446706  1.828464e-62 372

**Çapraz-doğrulama (önceki denetimin bağımsız tekrarı):** `CAT_3`↔`CAT_4`
Cramér's V = **1.0000**, `CAT_3`↔`CAT_5` = **1.0000** — bu üç
kolonun birebir aynı olduğu bulgusu (CLAUDE.md'ye eklenmişti) burada da
tam olarak V=1.0 ile doğrulandı.

## Kategorik-Hedef İlişkisi (Cramér's V + Mutual Information)

column  cramers_v_vs_Label  p_value   n  mutual_info_vs_Label
  AA_1            0.180454 0.041882 372              0.042405
 CAT_3            0.168515 0.005741 372              0.017577
 CAT_4            0.168515 0.005741 372              0.017577
 CAT_5            0.168515 0.005741 372              0.017577
 CAT_1            0.138742 0.147928 372              0.040114
  AA_2            0.103299 0.241838 372              0.034664
 CAT_2            0.050562 0.335485 372              0.012714

## Kategorik-Sayısal: En Güçlü 15 Eta İlişkisi (n≥30, güvenilir)

**Metodolojik not (Adım 6'daki aynı tuzağın burada da yakalanması):** Filtre
uygulanmadan önce en yüksek eta değerleri (~0.99) hep **n=14** gibi çok
düşük ortak-gözlem sayılarında çıktı — Adım 6'daki Pearson sahte-yükseklik
sorunuyla birebir aynı desen. **60 kombinasyon** n<30
iken eta≥0.7 gösteriyordu, bunlar güvenilmez kabul edilip filtrelendi.
Aşağıdaki tablo yalnızca n≥30 olan, güvenilir sonuçları gösteriyor:

categorical numeric      eta   n
      CAT_2   AL_84 0.882949 144
      CAT_1  AL_271 0.864681 129
      CAT_1  AL_235 0.853059 129
      CAT_1  AL_199 0.848342 129
      CAT_1  AL_270 0.831438 129
      CAT_1   AL_10 0.827379 220
      CAT_1  AL_311 0.825149 227
      CAT_1   AL_20 0.824370 122
      CAT_1  AL_151 0.824231 175
      CAT_1  AL_198 0.819178 129
      CAT_2  AL_202 0.818083  95
      CAT_1  AL_202 0.810941 129
      CAT_2   AL_96 0.795447 144
      CAT_1  AL_109 0.778420 175
      CAT_1  AL_207 0.774932 129

Tam sonuçlar (filtresiz, `n` kolonuyla birlikte — yorumlarken mutlaka
kontrol edin): `results/step7_categorical_categorical_cramers_v.csv` (kategorik-kategorik,
tüm çiftler), `step7_categorical_vs_target.csv` (hedef ilişkisi), `step7_categorical_numeric_eta.csv`
(kategorik-sayısal, 2401 kombinasyon).
