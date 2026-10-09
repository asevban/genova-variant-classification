# PAH Paneli — Aşama E Öncesi Literatür ve Yöntem Taraması

> Kaynak: Claude Research (kapsamlı web taraması), TEKNOFEST 2026 PAH panelinin
> tam bağlamı verilerek yürütüldü. Bu rapor bir karar belgesi değil — E0/E1
> başlamadan önce mevcut planın literatürle ne kadar uyumlu olduğunu ve
> hangi küçük eklemelerin değerli olabileceğini gösteren bir referans.

## TL;DR

- En yüksek getirili üç hamle: (1) basit imputasyon + eksiklik maskesi
  (zaten yapıyoruz — doğrulandı), (2) prior-shift'i kalibrasyon+SLD+eşik
  zinciriyle ele almak (zaten planlı — doğrulandı), (3) TabPFN v2/v2.5'i
  CatBoost'la birlikte çekirdek model olarak denemek (zaten E2'de var,
  öncelik teyit edildi).
- Prior-shift (eğitim %83,3 → final ~%28,6) projenin en kritik problemi;
  SLD'nin **yalnızca kalibre olasılıklarla güvenilir olduğu** (Alexandari,
  Kundaje, Shrikumar, ICML 2020) literatürle doğrulandı — E3→E4 sırası
  (önce kalibrasyon, sonra SLD) zaten doğru sırada.
- Model ailesi: gradient boosting (CatBoost/LightGBM/XGBoost) + TabPFN v2
  ikilisi + bunların stacking/blend'i (ICR Kaggle yarışmasının kazanan
  çözümleri XGBoost+TabPFN) küçük-N'de en sağlam yaklaşım — mevcut E2
  portföyüyle uyumlu.

## Ana Bulgular

### 1. Eksik veri — mevcut strateji doğrulandı

Särkkä ve ark. (2025, *NAR Genomics and Bioinformatics*, "Comparison of
missing data handling methods for variant pathogenicity predictors") —
doğrudan aynı problem (ClinVar/gnomAD varyant patojenite, yüksek eksiklik)
üzerinde 14 yöntem test etmiş. Sonuç: **basit imputasyon (mean/0),
MissForest/MICE gibi karmaşık yöntemlerden anlamlı şekilde iyi değil**,
bazen daha kötü. Missingness indicator lojistik regresyonda en iyi
sonuçlardan biri. gnomAD allel frekansı eksikliği, **daha yüksek
patojenite olasılığıyla pozitif korelasyonlu** bulunmuş — GENOVA'nın
"tüm AL_ eksik → potansiyel patojenite sinyali" hipotezini bağımsız
doğruluyor.

**Sonuç:** Mevcut ön işleme stratejisi (basit doldurma + `al_all_missing`
gibi eksiklik göstergeleri) literatürle uyumlu, değişiklik gerekmiyor.

### 2. Küçük-N tablo verisi model karşılaştırması

TabPFN v2 (Hollmann ve ark., *Nature* 2025) küçük-orta ölçekli (≤10.000
satır) tablo verisinde CatBoost'u normalize ROC AUC'ta 0,187 puan geçiyor.
TabPFN v2.5 (Prior Labs, Kasım 2025), ≤10.000 satır/500 özellik veri
setlerinde XGBoost'a karşı **%100 kazanma oranı** bildiriyor. Ancak
bağımsız bir çalışma (Ye ve ark., arXiv:2502.17361), **yüksek-boyutlu**
(features/samples oranı yüksek) veri setlerinde RealMLP ve CatBoost'un
TabPFN v2'yi geçebildiğini gösteriyor — PAH'ın 25 özellik/369 satır oranı
bu sınıra yakın, dikkatli olunmalı.

**Sonuç:** TabPFN v2/v2.5'i E2'de öncelikli dene, ama CatBoost/LightGBM'i
"yedek" değil eşit aday olarak tut — yüksek-boyut nüansı nedeniyle hangisi
kazanır önceden belli değil, nested CV karar versin.

### 3. Prior-shift düzeltmesi — sıra kritik

Alexandari, Kundaje, Shrikumar (ICML 2020): SLD'nin (maximum likelihood
prior düzeltmesi) güvenilir olması için **olasılıkların önce kalibre
edilmiş olması şart**. Kalibrasyonsuz SLD, BBSE/RLLS gibi alternatiflerden
bile kötü performans gösterebiliyor.

**Sonuç:** E3 (kalibrasyon) → E4 (SLD) sırası zaten doğru, değişmiyor.
Hedef prevalans (~%28,6) şartnameden **bilindiği** için SLD'de bunu tahmin
etmeye çalışmak yerine sabit değer olarak kullanmak (mevcut plan) daha
güvenli — literatürdeki SLD kırılganlığı çoğunlukla prior-tahmini
aşamasından geliyor.

### 4. Küçük-N kalibrasyon

Isotonic regression küçük kalibrasyon setlerinde (n<500) aşırı-uyum
riski taşıyor; Platt/Beta calibration daha güvenilir. Bu, E3'ün zaten
kurduğu önceliklendirmeyle (Platt öncelikli, Beta ikinci, Isotonic
yalnızca karşılaştırma amaçlı) birebir örtüşüyor.

### 5. ICR Kaggle yarışması (en yakın benzer örnek)

Anonim özellikler, n≈617, dengesiz sınıf, prior-shift içeren tıbbi tanı
yarışması. Kazanan/üst çözümler neredeyse evrensel olarak **XGBoost +
TabPFN ensemble** kullandı. Önemli ders: public leaderboard'a değil
**nested/cross-validation skoruna güvenin** — eşikleme yapan bazı üst
public-skor çözümleri özel test setinde çöktü.

### 6. Adversarial validation / provenance

Standart yöntem doğrulandı: train/test (veya kaynak A/B) ayırt eden bir
sınıflandırıcı, AUC≈0,5 hedefi, yüksek-AUC veren kolonları çıkar/GroupKFold
uygula. Genomik-spesifik risk: "data circularity" (VariPred 2024) — aynı
genin tüm varyantlarının aynı etiketi taşıması, modelin biyolojik sinyal
yerine gen/kaynak kimliğini öğrenmesine yol açabilir. `CAT_` kolonları
(özellikle kalite bayrağı, arkaik genom genotipi) en olası sızıntı
taşıyıcıları.

### 7. Küçük-N ensemble/stacking

Meta-öğrenici basit (lojistik regresyon) olmalı, taban modellerin OOF
tahminleriyle eğitilmeli (aynı veriyle hem taban hem meta eğitmek sızıntı
yaratır). Heterojen taban modeller (boosting+doğrusal+foundation model)
çeşitlilik sağladığında stacking tek-modelden iyi; çeşitlilik yoksa
marjinal. Mevcut E6 planıyla (gerçekten farklı modeller şartı) uyumlu.

### 8. Entropy/ID3 ensemble vs boosting

Genel benchmarklar boosting'in klasik ağaç yöntemlerini geçtiğini
gösteriyor — GENOVA'nın kendi deneylerinin (Faz 1/3) bulduğu sonuçla
tutarlı. ID3 ensemble'ın tek başına baş model değil, stack'e çeşitlilik
üyesi olarak değerlendirilmesi önerisi literatürle uyumlu.

## Yeni, Değerlendirilmeye Değer Bir Fikir

**Satır-bazlı AL_ özet istatistikleri:** Her varyant için, kendi AL_
kolonları arasında (334 kolon) max/min/ortalama allel frekansı ve gözlenen
popülasyon sayısı gibi özet özellikler türetmek — ACMG BA1/BS1 ("çok sık
görülen = benign") ve PM2 ("hiç görülmemiş = patojenik olabilir")
kriterlerinin doğrudan sayısal karşılığı. Bu, mevcut `v3`'ün korelasyon-
kümesi grup-özetlerinden (belirli kümeler için min/max/median/n_positive)
**farklı** — o kümeye özel, bu tüm AL_ evrenine satır-bazlı bir özet.
Şu an pipeline'da yok. Düşük efor, biyolojik olarak gerekçeli, dış-skor
içermiyor — E2'de bir aday özellik seti olarak denenmeye değer (resmi
Aşama D pipeline'ına dokunmadan, `ZORT_*` adaylarıyla aynı şekilde E2'nin
kendi nested karşılaştırmasında test edilebilir).

**Sonuç:** E2'de test edildi (`ZORT_*` ile birlikte), `catboost/v1`'e
karşı paired karşılaştırmada kaybetti (kazanma oranı %30, ort. ΔF1
−0,0026), resmi havuza eklenmedi (bkz. `06_MODEL_SECIM_RAPORU_PAH.md`,
"`ZORT_*` ve `AL_`-özet aday özellik setleri" bölümü).

## Caveatlar

- Särkkä ve ark.'ın (AMISS) yazarları ticari bir genetik şirketinden;
  downstream sınıflandırıcıları RF/LR'ydi, boosting değil — kendi
  verinizde ablation ile doğrulayın.
- TabPFN v2.5'in "%100 kazanma" iddiası geliştiricisinin kendi raporundan;
  yüksek-boyut nüansı nedeniyle PAH'ta peşinen üstünlük varsaymayın.
- n=369 kaçınılmaz yüksek varyans demek — model kararlarını tek fold/tek
  tekrara değil, 10-tekrarlı dış CV'nin std-hatalarına dayandırın.
- "Eksik = nadir = patojenik" olasılıksal bir sinyal, kesin kural değil
  (gnomAD v4 bazı patojenik varyantların popülasyonda görülebildiğini
  gösteriyor).

## Sonuç — Aşama E Planına Etkisi

Taramanın **büyük kısmı mevcut E-F planını doğruluyor**, radikal bir
değişiklik gerektirmiyor. Üç küçük, somut ekleme değerli:
1. TabPFN v2.5 spesifik olarak E2'ye not düşülmeli (v2'nin yanına).
2. Yüksek-boyut nüansı (features/samples oranı) E2'ye karar-kuralı olarak eklenmeli.
3. Satır-bazlı AL_ özet istatistikleri, E2'de denenecek yeni bir aday özellik seti olarak eklenmeli.

Bunların dışında hiçbir kesin kural/sıra değişmiyor.
