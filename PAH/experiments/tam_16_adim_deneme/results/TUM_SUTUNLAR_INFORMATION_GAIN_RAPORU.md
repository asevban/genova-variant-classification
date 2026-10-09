# Adım 2 — Tüm 353 Sütun İçin Information Gain (Yalnızca Keşif)

> **Düzeltme 1 uygulandı:** Bu tablo ham veri (372 satırın tamamı) üzerinde
> hesaplandı — dış-fold'ların da göründüğü bir istatistik, bu yüzden
> **hiçbir eleme kararına doğrudan girdi olarak kullanılmıyor**. Adım 3'ün
> önerdiği eleme adayları birer **hipotez**; gerçek doğrulama yalnızca
> Adım 13'ün nested prosedüründe yapılıyor.

Formül: `IG(Label, X) = H(Label) - H(Label|X)` (log2, bit cinsinden).
Sayısal kolonlarda tüm ara-noktalar eşik adayı olarak denendi (ID3-tarzı);
eksiklik/doluluk ayrımının ayrıca bilgilendirici olup olmadığı da
kaydedildi. Gain Ratio, dal-büyüklüğü entropisine bölünerek hesaplandı.

## En Yüksek IG'li 20 Sütun

column        type       IG  gain_ratio  missing_rate
  EK_7     numeric 0.062911    0.054321      0.029570
AL_323     numeric 0.061567    0.040015      0.338710
  AA_1 categorical 0.061177    0.014719      0.024194
 CAT_1 categorical 0.057872    0.017596      0.362903
AL_306     numeric 0.057470    0.037609      0.338710
AL_334     numeric 0.055832    0.050857      0.338710
AL_331     numeric 0.054224    0.041271      0.338710
AL_296     numeric 0.052575    0.041936      0.338710
AL_302     numeric 0.052416    0.033768      0.338710
AL_300     numeric 0.051594    0.034236      0.338710
  EK_9     numeric 0.050860    0.052930      0.029570
  AA_2 categorical 0.050010    0.011947      0.024194
AL_319     numeric 0.049932    0.051549      0.338710
AL_329     numeric 0.048804    0.041951      0.338710
AL_327     numeric 0.048095    0.047958      0.338710
AL_326     numeric 0.048018    0.030856      0.338710
AL_318     numeric 0.047847    0.040121      0.338710
AL_298     numeric 0.047044    0.033477      0.338710
AL_333     numeric 0.046380    0.030245      0.338710
AL_313     numeric 0.046202    0.043571      0.338710

## Genel İstatistikler

- Toplam taranan kolon: 351
- IG ≈ 0 (≤1e-6) olan kolon sayısı: 1
- Eksiklik/doluluk ayrımı bilgilendirici olan kolon sayısı: 48

Tam tablo: `results/id3_feature_importance.csv`.
