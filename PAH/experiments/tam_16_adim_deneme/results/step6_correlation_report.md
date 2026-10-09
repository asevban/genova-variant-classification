# Adım 6 — Tüm Sayısal Sütun Çiftleri İçin İkili Korelasyon

`AL_`+`EK_` = 343 sayısal kolon, **C(343,2) = 58,653 çift**
tarandı (yalnızca `AL_`-`AL_` değil, `AL_`-`EK_` ve `EK_`-`EK_` de dahil —
GENOVA'nın resmi pipeline'ında bu kapsamlı tarama yoktu).

**Metodolojik not (ilk denemede yakalanan bir hata):** GENOVA'nın kendi
`AL_`-`AL_` taraması `min_periods=30` kullanıyor — düşük eşiklerin (ör. 10)
az sayıda ortak-dolu satırla sahte-yüksek korelasyon ürettiği zaten
doğrulanmıştı. Bu script'in **ilk çalıştırmasında `min_periods` unutuldu**
ve en güçlü çiftlerin ortak-n'i yalnızca **14** çıktı (aynı sahte-yüksek-
korelasyon tuzağı) — bu ampirik olarak fark edilip **`min_periods=30`
eklenerek düzeltildi**, aşağıdaki sayılar düzeltilmiş hâle aittir.

## Genel Özet

- Hesaplanabilen çift: 27,429 (%46.8)
- **Hesaplanamayan çift: 31,224** (%53.2) — nedeni:
hesaplanamama_nedeni
sabit değer (std=0)                                   26775
yetersiz ortak gözlem (n=13<30, min_periods eşiği)     1926
yetersiz ortak gözlem (n=14<30, min_periods eşiği)     1431
yetersiz ortak gözlem (n=12<30, min_periods eşiği)     1008
hesaplanamadı (bilinmeyen neden)                         66
yetersiz ortak gözlem (n=9<30, min_periods eşiği)        18
- **Çok güçlü (|r|≥0.90) çift sayısı: 222**
- Çok zayıf (|r|<0.05) çift sayısı: 8,300

### Pearson |r| Dağılımı (yalnızca hesaplanabilen çiftler)

strength_pearson
çok zayıf    22251
zayıf         3375
orta          1219
güçlü          362
çok güçlü      222

## En Güçlü 10 Çift

 col_i  col_j  common_n  pearson_r  spearman_r strength_pearson
 AL_13 AL_112       169   0.991139    0.641704        çok güçlü
  AL_7 AL_103       169   0.990019    0.737979        çok güçlü
AL_112 AL_133       202   0.987358    0.666492        çok güçlü
AL_103 AL_296       185   0.986632    0.647900        çok güçlü
AL_103 AL_121       202   0.986630    0.677011        çok güçlü
AL_112 AL_323       185   0.986377    0.611110        çok güçlü
 AL_13 AL_133       169   0.985847    0.676125        çok güçlü
  AL_7 AL_296       211   0.984129    0.663221        çok güçlü
 AL_13 AL_121       169   0.984095    0.627850        çok güçlü
  AL_7 AL_121       169   0.983318    0.650525        çok güçlü

## Sonuç — `|r|≥0.90` Eşiği: Pearson vs Spearman AYRIMI KRİTİK

**Pearson'da 222 çift `|r|≥0.90` eşiğini geçiyor — ama Spearman'da
bu sayı: 0.**

Bu, Faz 2'nin PRECHECK'te doğruladığı "AL_-AL_'de `|r|≥0.90` çift yok" bulgusuyla
**çelişmiyor** — o bulgu `AL_correlation_summary.json`'da özellikle **Spearman**
ile hesaplanmıştı (GENOVA'nın `01_eda_pah.py`'si `df[AL_COLS].corr(method="spearman",
min_periods=30)` kullanıyor). Spearman'da genişletilmiş (AL_-EK_/EK_-EK_ dahil)
taramada da **0 çift** bulunuyor —
Phase 2'nin bulgusu genişletilmiş kapsamda da **doğrulandı**.

**Pearson'daki 222 "çok güçlü" çift neredeyse tamamen aykırı-değer kaynaklı
görünüyor** — `AL_` kolonlarının ağır sağa-çarpık/sıfır-şişkin yapısı
(CLAUDE.md) nedeniyle birkaç uç değer Pearson'ı domine edebiliyor. En çarpıcı
örnek: `AL_235`↔`AL_271` — **Pearson r=0.978, Spearman r=0.003** (rank ilişkisi
pratikte SIFIR). Bu, GENOVA'nın "AL_ için Spearman kullan, Pearson değil"
kararının (`01_eda_pah.py` yorum satırı) genişletilmiş kapsamda da **bağımsız
olarak yeniden doğrulandığı** anlamına geliyor — Adım 10'un grafik incelemesi
bu yüzden Spearman'a göre en güçlü çiftlere odaklanacak.

Tam sonuçlar: `results/step6_all_numeric_pairs.csv` (58,653 satır),
en güçlü/en zayıf 500'lük listeler ayrı dosyalarda (Pearson'a göre sıralı —
yorumlarken Spearman kolonunu da mutlaka kontrol edin).
