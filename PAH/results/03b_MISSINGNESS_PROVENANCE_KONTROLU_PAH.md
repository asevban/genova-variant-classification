# PAH Paneli — `al_all_missing` Göstergesinin Kaynak-Provenance Confound Kontrolü

**Ek kontrol tarihi:** Aşama D Düzeltme Turu (Görev 1). Aşama D'nin o zamanki final 27-özellikli havuzunda yer alan `al_all_missing` göstergesinin `CAT_1` (gnomAD alt-popülasyonu) ve `CAT_2` (AllofUs) ile ilişkisini test eder. **Yalnızca betimsel/tanısal analiz** — hiçbir sınıflandırıcı eğitilmedi, karar Aşama D'nin final havuzunu değiştirmiyor. *(Güncelleme notu: sonraki bir turda `AL_296_missing` göstergesi kaldırıldı ve havuz 24 özelliğe indi — bkz. `reports/03_OZELLIK_SECIMI_PAH.md`; `al_all_missing` havuzda kalmaya devam ediyor, buradaki bulgular geçerliliğini koruyor.)*

Veri: `data/processed/pah/v2.parquet` (369 satır, dedup sonrası, `al_all_missing` zaten hesaplı).

---

## Bulgu — Beklenenden Daha Güçlü Bir İlişki

Önceden beklenen soru "hangi `CAT_1`/`CAT_2` alt-kategorisi `al_all_missing=1` satırları arasında aşırı temsil ediliyor?" idi. İlk crosstab denemesinde (yalnızca gözlenen kategori değerleriyle, `pandas.crosstab` varsayılan `dropna=True` ile) `al_all_missing=1` tarafı **tamamen boş** çıktı — bu, sorunun kendisinin yanlış çerçevelendiğini ortaya çıkardı: `al_all_missing=1` olan satırların **hiçbirinde** gözlenen bir `CAT_1`/`CAT_2` değeri yok; hepsi `CAT_1` VE `CAT_2` açısından da eksik.

### Çapraz Tablo 1 — `al_all_missing` × `CAT_1` (eksiklik ayrı kategori olarak dahil)

| CAT_1 | al_all_missing=0 | al_all_missing=1 | toplam | al_all_missing=1 içindeki payı | veri setindeki payı | aşırı-temsil oranı |
|---|---|---|---|---|---|---|
| **(missing)** | 43 | **89** | 132 | **%100.0** | %35.77 | **2.80×** |
| gnomADe_NFE | 66 | 0 | 66 | %0.0 | %17.89 | 0.00× |
| gnomADg_NFE | 32 | 0 | 32 | %0.0 | %8.67 | 0.00× |
| *(diğer 22 kategori)* | — | 0 | — | %0.0 | — | 0.00× |

`al_all_missing=1` olan 89 satırın **%100'ü** `CAT_1` açısından eksik; bu, veri setinin genelinde `CAT_1` eksiklik oranının (%35.77) **2.80 katı**. Diğer 24 gözlenen `CAT_1` kategorisinden hiçbiri `al_all_missing=1` grubunda tek bir satırda bile görünmüyor.

### Çapraz Tablo 2 — `al_all_missing` × `CAT_2`

| CAT_2 | al_all_missing=0 | al_all_missing=1 | toplam | al_all_missing=1 içindeki payı | veri setindeki payı | aşırı-temsil oranı |
|---|---|---|---|---|---|---|
| **(missing)** | 136 | **89** | 225 | **%100.0** | %60.98 | **1.64×** |
| AllofUs_EUR | 62 | 0 | 62 | %0.0 | %16.80 | 0.00× |
| *(diğer 5 kategori)* | — | 0 | — | %0.0 | — | 0.00× |

Aynı desen: `al_all_missing=1` satırlarının **%100'ü** `CAT_2` açısından da eksik (veri setinin geneli %60.98).

### Bağımsızlık Testi (Ki-kare / Cramér's V)

| Kolon | χ² | serbestlik derecesi | p-değeri | Cramér's V |
|---|---|---|---|---|
| `CAT_1` | 210.59 | 24 | 9.2×10⁻³² | **0.755** |
| `CAT_2` | 75.07 | 7 | 1.4×10⁻¹³ | **0.451** |

Cramér's V ölçeğinde (0=ilişki yok, 1=tam belirlenim) **0.755 çok büyük bir etki büyüklüğü** — pratikte `al_all_missing=1` olmak, `CAT_1` eksikliğinin neredeyse kesin bir alt-kümesi. `CAT_2` için 0.451 de orta-büyük bir etki.

### Ek gözlem: `Label` ile ilişki

`al_all_missing=1` (n=89): patojenik oranı **%97.75** (87/89). `al_all_missing=0` (n=280): patojenik oranı %78.93. Genel ortalama %83.47. Tek başına AUC = 0.625.

---

## Karar

**Yoğunlaşma/ilişki AÇIKÇA VAR** (Cramér's V=0.755 CAT_1 için "çok büyük" eşiğin bile üzerinde; ilişki yönü tam kapsayıcı: `al_all_missing=1 ⟹ CAT_1 eksik VE CAT_2 eksik`, istisnasız 89/89). Görev talimatındaki karar mantığına göre bu **"provenance-confound riski VAR"** kategorisine giriyor.

**Yorum (iki rakip hipotez, bu kontrolle ayırt edilemiyor):**
1. **Biyolojik tutarlılık hipotezi:** Bir varyant hiçbir popülasyon veritabanında (ne gnomAD ne AllofUs) hiç gözlenmemişse, doğal olarak ne `AL_` frekans değeri ne de `CAT_1`/`CAT_2` kaynak etiketi üretilebilir — üçü de aynı temel olayın (popülasyonda hiç görülmeme) farklı yansımaları olur. Bu senaryoda `al_all_missing` gerçek bir nadirlik/patojenite sinyali taşır (ACMG PM2 kriteriyle uyumlu: popülasyon veritabanlarında yokluk, patojenite kanıtı).
2. **Kaynak/pipeline-artefaktı hipotezi:** Eğitim setindeki varyantlar farklı küratasyon partilerinden/kaynaklardan derlenmiş olabilir; bir parti (örn. yalnızca ClinVar'dan gnomAD'a çapraz-referans yapılmadan alınan patojenik varyantlar) sistematik olarak hem `AL_` hem `CAT_1`/`CAT_2` alanlarını boş bırakmış olabilir — bu durumda ilişki biyolojiden değil, veri toplama sürecinden kaynaklanır ve final test setinde **aynı oranda tekrarlanacağının garantisi yoktur**.

Bu iki hipotez arasında bu betimsel kontrolle **ayrım yapılamıyor** — ikisi de veriyle tutarlı.

**Aksiyon:** `al_all_missing`, Aşama D'nin 27-özellikli final havuzundan **çıkarılmadı** (görev talimatına uygun olarak bu turda özellik havuzu değiştirilmiyor). Bunun yerine:
- Bu bulgu **Aşama F'nin adversarial validation adımına önceliklendirilmiş girdi olarak bağlanıyor**: F1'de zaten `CAT_1`/`CAT_2` dağılım kayması test edilecek; `al_all_missing` (ve dolaylı olarak `CAT_1`/`CAT_2` eksiklik göstergeleri) bu testte **öncelikli/işaretli özellik** olarak ele alınmalı — eğer adversarial model `al_all_missing`'i yüksek önemle kullanıyorsa, bu tek başına hipotez 2'yi (pipeline-artefaktı) güçlendirir.
- İleride (Aşama E/F) model, `al_all_missing`'in katkısını **`CAT_1`/`CAT_2` eksiklik göstergeleriyle birlikte** değerlendirmeli; üçü arasındaki neredeyse tam çakışma nedeniyle, üçünü birden kullanmak ek bilgi katmıyor olabilir (multicollinearity), bu da Aşama D.1'in ayrıca not ettiği "elastic-net gürültüsü" bulgusuyla da örtüşüyor.

**Sonuç:** Confound riski **VAR**, aksiyon **Aşama F'ye bağlandı**, final özellik havuzu bu turda değiştirilmedi.

---

## Ek Kontrol — `ek_cat_block_missing` ve `AL_296_missing` Göstergelerinin Fisher's Exact Test ile Doğrulanması

**Ek kontrol tarihi:** Bağımsız bir EDA notebook'unun (`01_eda_pah_ipynb.ipynb`) çapraz karşılaştırmasında, `dataset_versions.py`'nin v2'ye eklediği 4 eksiklik göstergesinden (`al_all_missing`, `ek_cat_block_missing`, `AL_296_missing`, `EK_3_missing`) ikisinin gerekçesi sorgulandı: `ek_cat_block_missing`'in dayandığı 11-satırlık desen küçük olduğu için ki-kare testinin geçerlilik şartını (beklenen hücre≥5) sağlamayabilir; `AL_296_missing` ise `al_all_missing` ile yüksek korele (r≈0.80) olduğu bilindiğinden gereksiz tekrar olabilir. Bu iki iddia, veri: `data/processed/pah/v2.parquet` (369 satır) üzerinde **Fisher's exact test** (`scipy.stats.fisher_exact`) ile doğrulandı — Fisher testi, ki-kare'nin aksine küçük beklenen hücre sayılarında da geçerlidir, bu yüzden `ek_cat_block_missing` için doğru araç budur. Ham sonuçlar: `reports/tables/missingness_indicator_fisher_test.json`.

### `ek_cat_block_missing` × `Label`

| | Label=0 | Label=1 |
|---|---|---|
| gösterge=0 | 57 | 304 |
| gösterge=1 | 4 | 4 |

- Ki-kare geçerlilik şartı **sağlanmıyor** (beklenen minimum hücre = 1.322 < 5) — doğrulandı, şüphe haklı çıktı.
- **Fisher's exact test: p=0.0283, odds ratio=0.19.** Ki-kare'nin geçersizliğine rağmen, geçerli alternatif test yine de p<0.05 ile anlamlı çıkıyor.
- `al_all_missing=1` alt-kümesinde (n=89) `ek_cat_block_missing` neredeyse hiç varyans taşımıyor (yalnızca 4/89 satırda 1) — bu iki gösterge büyük ölçüde **farklı** satırları işaretliyor (düşük örtüşme), yani `al_all_missing`'in doğrudan bir tekrarı değil.
- **Dikkat:** Anlamlılık yalnızca **8 satırlık** (4 benign + 4 patojenik) çok küçük bir hücreye dayanıyor — tek bir satırın etiketi değişse sonuç kolayca değişebilir. İstatistiksel olarak **kırılgan** bir sinyal.

### `AL_296_missing` × `Label`

| | Label=0 | Label=1 |
|---|---|---|
| gösterge=0 | 55 | 191 |
| gösterge=1 | 6 | 117 |

- Koşulsuz: **Fisher p=6.08×10⁻⁶, odds ratio=5.62** — çok güçlü görünüyor, ki-kare de geçerli (beklenen min=20.3≥5).
- **Ama koşullu analiz farklı bir tablo çiziyor.** `al_all_missing=1` alt-kümesinde (n=89), `AL_296_missing` **sabit** (89/89 satırda =1) — bu alt-grup içinde sıfır varyans, dolayısıyla sıfır ek ayırt edicilik. `al_all_missing`, `AL_296_missing`'in işaretlediği satırların büyük çoğunluğunu (123 satırdan 89'unu) zaten kapsıyor.
- Geriye kalan **34 satır** (`AL_296_missing=1` ama `al_all_missing=0`) gerçek bir ek kapsama alanı oluşturuyor. Ama bu 34 satırın **`al_all_missing=0` popülasyonu (n=280) içindeki** Label ilişkisi test edildiğinde: **Fisher p=0.184 — anlamlı değil** (ki-kare de geçerli, beklenen min=7.16).
- **Sonuç: `AL_296_missing`'in koşulsuz anlamlılığı (p=6×10⁻⁶) tamamen `al_all_missing` ile örtüşmesinden geliyor.** Hiçbir koşulda (`al_all_missing=1` içinde de, `al_all_missing=0` içinde de) bağımsız, doğrulanabilir bir ek sinyal taşımıyor.

### Karar Önerisi (kod değiştirilmedi — yalnızca öneri)

| Gösterge | Öneri | Gerekçe |
|---|---|---|
| **`AL_296_missing`** | **Düşürülmesi önerilir** | Koşulsuz anlamlılığı tamamen `al_all_missing` ile çakışmadan kaynaklanıyor; iki alt-popülasyonun hiçbirinde (koşullu) bağımsız ayırt edicilik yok. |
| **`ek_cat_block_missing`** | **Şimdilik kalsın, ama kırılgan olarak işaretlendi** | `al_all_missing` ile düşük örtüşme var (doğrudan redundant değil), Fisher testi p<0.05 veriyor — ama yalnızca 8 satırlık aşırı küçük bir hücreye dayanıyor, istatistiksel olarak narin. |

**Bu turda hiçbir kod değişikliği yapılmadı** (`dataset_versions.py`, `fold_features.py` dokunulmadı). Eğer `AL_296_missing`'in düşürülmesi onaylanırsa, bu değişiklik v2/v3/v4'ün yeniden üretilmesini **ve** Aşama D.1'in nested özellik seçiminin (`feature_selection.py`, ~17 dakika) yeniden çalıştırılmasını gerektirir — bu, ayrı bir onaylı görev olarak ele alınmalı.

---

## Ek Kontrol — Kısmi Korelasyon / Koşullu Information Gain (Faz 2, bağımsız deney)

**Kontrol tarihi:** 2026-08-16. Kaynak: `experiments/outlier_partial_corr_analizi/`
(bağımsız araştırma deneyi, resmi pipeline'ın parçası değil), kullanıcı
tarafından bağımsız olarak doğrulandı.

`al_all_missing=1` alt-grubunda (n=89) `CAT_1` mekanik olarak sabit (tüm
satırlarda 0,0 — çünkü bu alt-grubun tamamında `CAT_1` zaten eksik, bu
bölümün başındaki Cramér's V=0,755 bulgusunun doğrudan sonucu). Bu nedenle
asıl soru `al_all_missing=0` alt-grubunda (n=280) `CAT_1`'in `Label` ile
bağımsız bir ilişkisi olup olmadığı.

Sonuç: koşulsuz ilişki zayıf (`r=0,055, p=0,29`), ama `al_all_missing=0`
alt-grubuna daraltılınca ilişki güçleniyor (`r≈0,25-0,26, p<0,0001`).
Dörtte-birlik çapraz tabloda patojenik oranı `%61,6 → %79,4 → %89,0 → %86,4`
— genel olarak yükselen ama tam monoton olmayan bir desen (son dilimde
hafif düşüş).

**Yorum (temkinli):** Bu, `CAT_1`'in yalnızca `al_all_missing`'in bir
yansıması olmadığını, `al_all_missing`'den bağımsız kendi bilgisini
taşıdığını gösteriyor — saf pipeline-artefaktı hipotezine karşı bir kanıt
ağırlığı. Ama bu **kesin değil**: doz-cevap deseni tam monoton değil, ve
bu tek başına `CAT_1`'in kaynağının (biyolojik mi, farklı bir küratasyon
partisi mi) ne olduğunu belirlemiyor. Yukarıdaki "Karar" bölümünde
belirtilen iki rakip hipotez arasındaki nihai ayrım hâlâ **Aşama F'nin
adversarial validation adımına** bağlı — bu bulgu yalnızca ek bir veri
noktası, nihai karar değil.

## Ek — Aşama E2-EK Tier 3: Provenance Alt-Grup Performans Kırılımı

Bu bir ablasyon değil — final model (CatBoost/v4_from_v2, Strateji B)
**yeniden eğitilmedi/aranmadı**; `e2_model_comparison.csv`'nin zaten
seçtiği `best_params` aynen yeniden kullanılarak 50 dış fold'un tüm
dış-test tahminleri (3690 satır = 10 tekrar × 369) toplandı ve
`al_all_missing=0` / `al_all_missing=1` alt-gruplarına bölünüp **ayrı
ayrı** skorlandı (`src/genova/pah/e2ek_provenance.py`, çıktı: `reports/
tables/e2ek_provenance_subgroup_predictions.csv`).

| Alt-grup | n | n_benign | n_patojenik | F1 | MCC | Specificity | FP | FN |
|---|---|---|---|---|---|---|---|---|
| `al_all_missing=0` | 2800 | 590 | 2210 | 0,9053 | 0,4525 | 0,3847 | 363 | 82 |
| `al_all_missing=1` | 890 | 20 | 870 | **0,9852** | **−0,0125** | **0,0000** | 20 | 6 |

**Bulgu — F1 tek başına yanıltıcı, MCC/specificity gerçek durumu
gösteriyor:** `al_all_missing=1` alt-grubunda F1 çok yüksek görünüyor
(0,9852) ama bu tamamen alt-grubun aşırı dengesiz olmasından kaynaklanıyor
(890 satırın yalnızca 20'si benign, ~%2,2) — model bu alt-grupta
**benign'i hiçbir zaman doğru tahmin etmiyor** (specificity=0,0000, 20
benign satırın 20'si de FP) ve MCC neredeyse sıfır/hafif negatif
(−0,0125) — yani gerçek ayırt edicilik **yok**, model bu alt-grupta fiilen
"her zaman patojenik de" stratejisine eşdeğer davranıyor.

**Önemli ölçek uyarısı:** Bu 20 benign satır, pooled 10-tekrarlı dış-CV'nin
**yalnızca 2 benzersiz varyantına** karşılık geliyor (`VAR_002693` ve
`VAR_003234`, her biri 10 tekrarda birer kez test'e düşüyor). Yani bu
specificity=0,0 bulgusu **n=2 bağımsız gözleme** dayanıyor — çarpıcı ve
tutarlı (10 tekrarın **hepsinde** aynı sonuç) ama istatistiksel olarak
genellenebilir bir kanıt değil, yalnızca güçlü bir işaret. Ayrıca
`VAR_003234`, bu raporun ve `CLAUDE.md`'nin başka bir yerinde zaten
belgelenen **`conflict_group_1`** çelişkili-profil grubunun üyesi (bkz.
split bankası tasarımı) — yani bu iki "her zaman yanlış tahmin edilen"
benign varyanttan biri, zaten bağımsız olarak bilinen bir etiket-belirsizliği
taşıyor; onun yanlış tahmini kısmen `al_all_missing`'den değil, bu önceden
bilinen etiket belirsizliğinden kaynaklanıyor olabilir.

**Sonuca etkisi:** Bu bulgu, raporun başındaki iki rakip hipotezi (biyolojik
sinyal vs. pipeline-artefaktı) tek başına çözmüyor, ama Aşama F'nin
adversarial validation adımına somut, önceliklendirilmiş bir hedef
sağlıyor: final modelin `al_all_missing=1` alt-grubundaki benign tespiti
neredeyse tamamen kör — bu alt-grup F1 dominant metrikte gizlenen bir
zayıflık, F1'in tek başına yanıltıcı olabileceğinin bir başka somut örneği
(bkz. `04_BASELINE_YENIDEN_OLCUM_PAH.md`'deki benzer eşik-artefaktı
tartışması).

---

## NİHAİ DEĞERLENDİRME (Aşama F1, adversarial validation sonrası)

> Bu bölüm, raporun en başından beri açık bırakılan "biyolojik mi,
> pipeline-artefaktı mı?" sorusuna, artık elimizdeki **tüm** kanıtla
> nihai bir yargıyla cevap veriyor. Yukarıdaki içerik silinmedi — bu,
> ona eklenen kapanış.

### Toplanan kanıtın tam dökümü

| Kanıt | Yön | Güç |
|---|---|---|
| Cramér's V=0,755 (`al_all_missing`×`CAT_1`, tam kapsayıcı: 89/89) | Belirsiz (ikisi de tutarlı) | Çok güçlü ilişki, ama nedensellik yönü ayırt edilemiyor |
| `AL_296_missing`'in Fisher testiyle bağımsız sinyal taşımadığının doğrulanması | Hafif pipeline-lehine | `al_all_missing` ile tam örtüşme — "birden fazla kolonun aynı artefaktı tekrarladığı" deseni |
| Kısmi korelasyon (Faz 2): `al_all_missing=0` alt-grubunda `CAT_1`↔Label ilişkisi (r≈0,25-0,26, p<0,0001) | Hafif biyoloji-lehine | `CAT_1`'in `al_all_missing`'den bağımsız bir miktar kendi sinyali var — saf artefakt değil |
| E2-EK Tier 3: `al_all_missing=1` alt-grubunda specificity=0,000, MCC≈0 | Pipeline-lehine (dolaylı) | Zayıf — yalnızca 2 benzersiz varyanta dayanıyor |
| **Madde 11: `al_all_missing`/`CAT_1` final modelden çıkarmanın maliyeti YOK (p=0,83)** | **Güçlü pipeline-lehine** | Gerçek/vazgeçilmez biyolojik sinyal olsaydı çıkarmanın ölçülebilir bir bedeli olması beklenirdi |
| **F1 (bu bölüm): `AL_26`/`AL_12`/`AL_7`/`AL_49`'un eksiklik deseni, benign alt-kümede `CAT_1`-varlığını AUC 0,77-1,00 ile neredeyse mükemmel tahmin ediyor** | **Çok güçlü pipeline-lehine** | Bu dört kolonun ham (doldurmadan önceki) NaN deseni, `CAT_1` dolu olan **hiçbir** benign satırda görülmüyor — allel-frekansı biyolojisinin, idari bir kaynak-etiketiyle bu kadar keskin/istisnasız örtüşmesi biyolojik olarak beklenmez; klasik bir "aynı veri toplama partisi, hem etiketi hem ölçümü birlikte boş bırakmış" imzası |

### Nihai yargı

**Kanıtın toplam ağırlığı, pipeline/kaynak-provenance artefaktı hipotezini
açıkça baskın hipotez olarak destekliyor — saf biyolojik hipotez büyük
ölçüde çürütülmüş sayılır, ama %100 sıfırlanmış değil.**

Gerekçe: (1) madde 11'in "çıkarmanın bedeli yok" bulgusu ve (2) bu turun
"belirli AL_ kolonlarının eksiklik deseni, kaynak-etiketini neredeyse
mükemmel tahmin ediyor" bulgusu, ikisi birlikte, sinyalin büyük kısmının
**gerçek allel-frekansı biyolojisinden değil, hangi veritabanının
sorguya cevap verdiğinden** kaynaklandığını gösteriyor. Karşı-kanıt
(kısmi korelasyon bulgusu) tamamen göz ardı edilmiyor — `CAT_1`'in
`al_all_missing`'den bağımsız, küçük bir kendi sinyali olabileceğini
gösteriyor — ama bu, "baskın mekanizma pipeline-artefaktıdır" yargısını
değiştirmeyecek kadar zayıf bir karşı-ağırlık.

**Pratik sonuç:** `al_all_missing`/`CAT_1` zaten madde 11'de (maliyetsiz
olduğu için) çıkarıldı. Bu turda bulunan `AL_26`/`AL_12`/`AL_7`/`AL_49`
için bir ablasyon adayı üretildi ve madde 10'un araçlarıyla ölçüldü —
ama bu sefer maliyet madde 11'deki kadar temiz/sıfır değil (nominal
düşüş var, F1/MCC/specificity/Monte-Carlo'nun **hepsinde aynı yönde**,
istatistiksel olarak k=5'te anlamlılığa ulaşmasa da) — bu yüzden
otomatik olarak uygulanmadı, kullanıcı onayı bekleniyor (bkz.
`06_MODEL_SECIM_RAPORU_PAH.md` Aşama F1 bölümü).

**`03b` bu nihai değerlendirmeyle KAPANIYOR.**

---

## Nihai Karar (F1 sonrası)

`al_all_missing`/`CAT_1` provenance riski nedeniyle çıkarıldı (madde 11,
maliyetsiz). Ayrıca F1'in adversarial validation'ı 4 yeni riskli özellik
buldu (`AL_26, AL_12, AL_7, AL_49` — adversarial AUC 0,77-1,00, eksiklik
deseni `CAT_1` doluluğuna neredeyse birebir bağlı). Bunların çıkarılması
**büyük ve istatistiksel olarak sağlam bir performans maliyeti** taşıyor
(10/10 tekrarlı-bölme testinde tutarlı, ortalama −0,0877±0,0272
weighted-F1). Bu, madde 11'in "bedava sigorta" mantığından farklı —
burada gerçek bir takas var. **Karar: model korundu, risk bilinçli
kabul edildi.**

Gerekçe: (1) bu özelliklerin (özellikle `AL_49`'un, Faz 1/3'te bağımsız
doğrulanmış ikincil sinyali) muhtemelen gerçek biyolojik bilgi de
taşıdığı; (2) saf şortkat mi gerçek sinyal mi olduğu mevcut araçlarla
ayırt edilemiyor; (3) kanıtlanmamış bir şüphe üzerine büyük bir
performans fedakarlığı yapmanın gerekçesiz olduğu değerlendirildi. Bu
risk, F3'ün sağlamlık stres testlerinde (kaynak-dağılımı kayması
senaryoları) ayrıca izlenmeli.

---

## Güncelleme (F3 sonrası)

F3'ün sağlamlık testi, `CAT_1` dağılımı kayarsa modelin ciddi şekilde
bozulabileceğini gösterdi (kritik yönde AUC 0,831→0,573). Ancak takip
testi (karar matrisi: 26 vs 23 vs 22 özellik), bu kırılganlığın 4 riskli
özellikten (`AL_26/12/7/49`) kaynaklanmadığını kanıtladı — bunları
çıkarmak (kısmen ya da tamamen) çöküşü düzeltmiyor (B: 0,5727, A ile
pratikte aynı; C: 0,4937, A'dan bile kötü), yalnızca ek maliyet
getiriyor.

**Kök neden karakterizasyonu (yalnızca betimsel EDA, model yok):**
`CAT_1`-boş alt-küme (n=132) ile `CAT_1`-dolu alt-küme (n=237)
karşılaştırıldığında:

- **Label dağılımı neredeyse özdeş** — `CAT_1`-boş %83,3 patojenik,
  `CAT_1`-dolu %83,5 patojenik (genel ortalama %83,5). Bu, alt-kümenin
  "aşırı tek-etiketli" olmadığını, sorunun etiket dengesizliğinden
  gelmediğini gösteriyor.
- **`AL_` eksiklik oranında çarpıcı bir fark var:** `CAT_1`-boş
  alt-kümede kolon-bazlı ortalama `AL_` eksikliği **%91,4**, `CAT_1`-dolu
  alt-kümede yalnızca **%36,6**. (`EK_`'te küçük bir fark var — %8,8 vs
  %6,3 — `AA_`'da fark yok — %1,5 vs %1,7.)
- `al_all_missing=1` (89 satır) `CAT_1`-boş alt-kümenin **tam bir
  alt-kümesi** (89/89 = %100 örtüşme o yönde), ama `CAT_1`-boş
  satırların yalnızca %67,4'ünü kapsıyor — geri kalan 43 satır tüm
  `AL_` bloğu eksik olmasa da yine de çok yüksek `AL_` eksikliği
  taşıyor.
- `CAT_1`-boş satırların **%81'i** (107/132) aynı zamanda `CAT_2`
  (AllofUs) açısından da boş — yani `CAT_1`-boş olmak, genel olarak
  "hiçbir popülasyon veritabanında gözlenmeme" durumuyla güçlü bir
  şekilde örtüşüyor.

**Yorum:** Bu alt-küme "farklı bir etiket dağılımı" taşıyan bir grup
değil — **yapısal olarak çok daha az `AL_` bilgisi içeren, küçük
(132 satırlık) bir alt-popülasyon.** `CAT_1`-boş satırlarda eğitildiğinde
model, `AL_` özelliklerinin neredeyse tamamının sıfıra doldurulduğu bir
veriyle karşılaşıyor — gerçek `AL_` sinyalini öğrenecek neredeyse hiç
örneği yok. `CAT_1`-dolu satırlarda test edildiğinde ise model, hiç
öğrenmediği (gerçek, sıfır-olmayan) `AL_` değerleriyle karşılaşıp
başarısız oluyor. Bu, **belirli bir kolonun** değil, **tüm `AL_`
bloğunun eksiklik-oranı uyumsuzluğunun** sonucu — bu yüzden 3-4 kolonu
çıkarmak sorunu çözmüyor, çünkü sorun kolonlarda değil, alt-kümenin
kendisinde.

**Ek nüans (03b'nin orijinal iki rakip hipotezine dönerek):** Bu yeni
bulgu (neredeyse özdeş etiket oranı + `CAT_1`-boşluğunun `CAT_2`-boşluğu
ile de güçlü örtüşmesi) aslında raporun başındaki **Hipotez 1'i**
(biyolojik tutarlılık — hiçbir popülasyon veritabanında hiç gözlenmemiş
bir varyantın doğal olarak hem `AL_` hem `CAT_1`/`CAT_2` alanlarının boş
kalması) bir miktar güçlendiriyor; saf "farklı küratasyon partisi"
(Hipotez 2) hikâyesinden biraz uzaklaştırıyor. Bu, önceki "pipeline-
artefaktı baskın" yargısını **geçersiz kılmıyor** (adversarial AUC
bulgusu hâlâ geçerli, `AL_26/12/7/49` hâlâ `CAT_1`'i neredeyse mükemmel
tahmin ediyor) — yalnızca kök nedenin muhtemelen "farklı bir veri
toplama partisi" değil, "gerçekten az gözlenmiş/nadir varyantların
doğal, tutarlı bir yansıması" olduğunu düşündürüyor. Pratik sonuç
değişmiyor: her iki durumda da final test setinde kaynak-kompozisyonu
farklıysa risk gerçek.

**Karar: model `A` (26 özellik, `final_model_bundle_v2.pkl`) korunuyor.**
Hiçbir alternatif (B/C) çöküşü düzeltmiyor, bedelsiz bir çözüm yok, bedel
ödeyip fayda almamanın anlamı yok. Bu kırılganlık düzeltilebilir bir hata
değil — veri setinin doğal bir sınırlaması olarak **açıkça belgeleniyor**.
Jüri sunumunda bu risk gizlenmeyecek: model, kaynak-kompozisyonu
eğitimden ciddi farklı olan (özellikle `AL_` eksiklik oranı çok yüksek
olan) varyantlarda güvenilirliğini kaybedebilir.

**`03b` bu güncellemeyle TAMAMEN KAPANDI.**
