# Korelasyonlu Kolonları Matematiksel Olarak Birleştirme – Kanıt Raporu

## 1. Mentorun istediği çalışma

Mentorun önerisi şu şekilde uygulanmıştır:

> Benzer özellikleri yalnızca silmek yerine aynı matematiksel ölçeğe getir, tek bir birleşik kolon gibi kullandır ve bu veriyle modeli yeniden eğit. Birleştirilecek özellikleri aralarındaki korelasyonlara göre belirle.

Bu işlemin amacı:

1. Aynı veya benzer bilgiyi taşıyan kolonların tekrarını azaltmak,
2. Birbiriyle farklı ölçeklerde bulunan değerleri karşılaştırılabilir hâle getirmek,
3. Bir özellik grubunu tek bir birleşik özellik olarak modele vermek,
4. Kolon sayısını düşürürken tahmin bilgisini mümkün olduğunca korumak,
5. Birleştirmenin gerçekten faydalı olup olmadığını çapraz doğrulamayla kanıtlamaktır.

## 2. Başlangıç veri seti

İlk Information Gain ve gereksizlik elemesinden sonra oluşan veri seti kullanılmıştır:

- Gözlem sayısı: `111`
- Model özelliği: `322`
- Benign (`Label=0`): `21`
- Patojenik (`Label=1`): `90`
- Kimlik sütunu `Variant_ID` modele verilmemiştir.

Başlangıç dosyası: `YARISMA_TRAIN_CFTR_REDUCED.csv`

## 3. Kolonlar neye göre eşleştirildi?

322 özellik içindeki bütün sayısal özellik çiftleri Pearson ve Spearman korelasyonlarıyla karşılaştırılmıştır.

- Pearson: doğrusal ilişkiyi ölçer.
- Spearman: sıralamaya dayalı monoton ilişkiyi ölçer.
- Çok güçlü Pearson sınırı: `|r| ≥ 0.90`

Bulunan yedi çok güçlü çift:

|Çift|Pearson|Spearman|
|---|---:|---:|
|`AL_5 – AL_16`|0.93474|0.29286|
|`AL_5 – AL_112`|0.93471|0.60714|
|`AL_5 – AL_40`|0.93254|0.47500|
|`AL_6 – AL_251`|0.92097|0.73626|
|`AL_4 – AL_38`|0.92008|0.12500|
|`AL_5 – AL_304`|0.90728|0.00357|
|`AL_1 – AL_211`|0.90545|0.63956|

Bu çiftlerden aşağıdaki dört korelasyon grubu oluşturulmuştur:

```text
GRUP_AL5       = AL_5, AL_16, AL_112, AL_40, AL_304
GRUP_AL6_AL251 = AL_6, AL_251
GRUP_AL4_AL38  = AL_4, AL_38
GRUP_AL1_AL211 = AL_1, AL_211
```

## 4. Neden ham ortalama alınmadı?

Özelliklerin sayısal büyüklükleri ve dağılımları birbirinden farklıdır. Ham değerleri doğrudan toplamak veya ortalamak, büyük ölçekli kolonun birleşik özelliği kontrol etmesine neden olur.

Bu nedenle her özellik önce robust biçimde aynı matematiksel çerçeveye alınmıştır:

\[
z_{robust} = \frac{x-medyan}{IQR}
\]

Burada:

- `medyan`: ilgili özelliğin eğitim verisindeki medyanı,
- `IQR`: üçüncü çeyrek ile birinci çeyrek arasındaki farktır.

Bir satırdaki birleşik kolon daha sonra şu şekilde hesaplanmıştır:

\[
Grup(X_1,\ldots,X_k)=ortalama(z_{robust,1},\ldots,z_{robust,k})
\]

Bir grup üyesi eksikse yalnız o satırda mevcut olan standartlaştırılmış değerlerin ortalaması alınmıştır. Bütün grup üyeleri eksikse birleşik değer de eksik bırakılmıştır.

## 5. Veri sızıntısı nasıl önlendi?

Medyan ve IQR değerleri bütün veri setinden bir kez hesaplanmamıştır. Her çapraz doğrulama katında:

1. Medyan ve IQR yalnız dış eğitim bölümünden öğrenilmiştir.
2. Aynı parametreler iç doğrulama ve dış test bölümüne uygulanmıştır.
3. Dış test satırları preprocessing parametrelerini etkilememiştir.
4. Birleşim sonrası ilgili ham kolonlar model girdisinden çıkarılmıştır.

Bu yöntem, test bilgisinin eğitim/preprocessing aşamasına sızmasını engellemiştir.

Uygulamanın kod kanıtı:

- `composite_feature_nested_cv.js`: bütün grup seçeneklerinin ID3 deneyi
- `catboost_scenario1_composite_nested_cv.py`: 322 özellikli senaryoda CatBoost grup deneyi
- `catboost_benign_weighted_finalists_nested_cv.py`: finalistlerin benign ağırlıklı nested CV deneyi
- `id3_benign_weighted_322_vs_320.js`: 322 ve birleşik 320 özellikli ID3 karşılaştırması

## 6. Denenen birleştirme senaryoları

|Senaryo|İşlem|Son özellik sayısı|
|---|---|---:|
|1A – Referans|322 özellik değiştirilmeden kullanıldı|322|
|Yalnız `GRUP_AL5`|Beş AL_5 ilişkili kolon tek kolonda birleştirildi|314 özellikli karşılaştırma yapısı|
|Yalnız `GRUP_AL6_AL251`|İki kolon tek kolonda birleştirildi|314 özellikli karşılaştırma yapısı|
|Yalnız `GRUP_AL4_AL38`|İki kolon tek kolonda birleştirildi|314 özellikli karşılaştırma yapısı|
|Yalnız `GRUP_AL1_AL211`|İki kolon tek kolonda birleştirildi|314 özellikli karşılaştırma yapısı|
|Bütün gruplar|Dört korelasyon grubu birlikte birleştirildi|314 özellikli yapı|
|Sağlam gruplar|Yalnız AL_6–AL_251 ve AL_1–AL_211 birleştirildi|320 özellikli Senaryo 1C|

Grafik incelemesinde `AL_6–AL_251` ve `AL_1–AL_211` çiftlerinin Spearman değerleri diğer birçok çifte göre daha yüksek ve ilişkileri daha tutarlı olduğu için “sağlam gruplar” olarak korunmuştur.

Senaryo 1C dönüşümü:

```text
AL_6 + AL_251  → GRUP_AL6_AL251
AL_1 + AL_211  → GRUP_AL1_AL211

4 ham kolon çıkarıldı
2 birleşik kolon eklendi
322 özellik → 320 özellik
```

## 7. İlk CatBoost karşılaştırması

İlk CatBoost deneyi, daha dengeli Macro-F1 ve MCC seçimine göre yapılmıştır. Dış değerlendirme 5-fold stratified CV × 5 tekrar, iç eşik seçimi 4-fold CV'dir.

|Model|Özellik|Accuracy|Specificity|Recall|Macro-F1|MCC|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|
|Senaryo 1A – ham azaltılmış|322|0.8396|0.5810|0.9000|0.7396|0.4792|41.9|
|**Senaryo 1C – sağlam gruplar birleşik**|**320**|**0.8396**|**0.6381**|0.8867|**0.7503**|**0.5023**|**36.2**|
|Senaryo 2A – sekiz temsilci çıkarılmış|314|0.8306|0.5905|0.8867|0.7317|0.4641|41.0|
|Senaryo 2B – bileşik 314|314|0.8396|0.6190|0.8911|0.7469|0.4946|38.1|

Senaryo 1C'nin ham 322 özellikli CatBoost'a etkisi:

- Özellik sayısı: `322 → 320`
- Specificity: `0.5810 → 0.6381` (`+0.0571`)
- Macro-F1: `0.7396 → 0.7503` (`+0.0107`)
- MCC: `0.4792 → 0.5023` (`+0.0231`)
- FP/100 benign: `41.9 → 36.2` (`−5.7`)
- Recall: `0.9000 → 0.8867` (`−0.0133`)

**İlk sonuç:** Korelasyonlu çiftleri robust biçimde birleştirme, CatBoost'ta iki kolon azaltırken MCC ve Macro-F1'ı artırmış, FP'yi azaltmıştır. Bu nedenle Senaryo 1C finalist yapılmıştır.

## 8. Benign ağırlıklı nihai doğrulama

Yarışma testinin yaklaşık 100 benign + 20 patojenik olacağı bilgisi nedeniyle modeller aynı benign ağırlıklı nested CV düzeninde yeniden karşılaştırılmıştır.

- Dış değerlendirme: 5-fold stratified CV × 5 tekrar
- İç değerlendirme: 4-fold stratified CV
- Eşik seçim hedefi: projeksiyon F1
- Eşitlikte: daha az FP, sonra daha yüksek recall
- Ön işleme ve eşik seçimi yalnız eğitim fold'larında yapılmıştır.

|Model|Özellik|Eşik|Accuracy|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|**ID3 – ham azaltılmış**|**322**|15.48/21 oy|**0.7369**|**0.8857**|**0.7022**|**0.6863**|**0.4684**|**0.6177**|**11.4**|
|ID3 – Senaryo 1C|320|15.84/21 oy|0.7279|0.8476|0.7000|0.6738|0.4372|0.5687|15.2|
|CatBoost – Senaryo 1C|320|0.686 olasılık|0.7099|0.8286|0.6822|0.6558|0.4061|0.5373|17.1|

Bu skorların doğrudan CSV kanıtı:

```text
results/id3_catboost_benign_agirlikli_nihai_skorlar.csv
```

### ID3'te birleştirmenin etkisi

322 özellikli ID3 ile 320 özellikli birleşik ID3 karşılaştırıldığında:

- Özellik sayısı: `322 → 320`
- MCC: `0.4684 → 0.4372` (`−0.0312`)
- Macro-F1: `0.6863 → 0.6738` (`−0.0125`)
- Projeksiyon F1: `0.6177 → 0.5687` (`−0.0490`)
- FP/100 benign: `11.4 → 15.2` (`+3.8`)
- Recall: `0.7022 → 0.7000` (`−0.0022`)

**Nihai ID3 sonucu:** Birleştirme ID3'e katkı sağlamamıştır. ID3, korelasyonlu kolonları ayrı ayrı kullanırken daha iyi karar sınırları oluşturmuştur.

## 9. Sonuç neden modele göre değişti?

CatBoost ve ID3 aynı özellik mühendisliğine aynı tepkiyi vermek zorunda değildir:

- CatBoost birleşik ve sürekli bir özet özellikten faydalanabilir.
- ID3 her ham kolonda ayrı eşik belirleyerek ilişkilerin farklı bölgelerini kullanabilir.
- Ortalama almak, iki kolondaki bazı ayırt edici uç veya eksik değer desenlerini yumuşatabilir.
- Korelasyon yalnız iki özelliğin birlikte değişimini gösterir; Label için aynı karar bilgisini taşıdıklarını garanti etmez.

Bu nedenle korelasyonla birleştirme matematiksel olarak mantıklı bir aday üretmiş, fakat son karar mutlaka model bazında cross-validation skoruyla verilmiştir.

## 10. Kanıt dosyaları

|Kanıt|Dosya|
|---|---|
|Nihai skor tablosu|`results/id3_catboost_benign_agirlikli_nihai_skorlar.csv`|
|Nihai karşılaştırma açıklaması|`results/ID3_CATBOOST_BENIGN_AGIRLIKLI_NIHAI_KARSILASTIRMA.md`|
|Korelasyon çiftleri ve IG gerekçeleri|`results/KORELASYON_ASAMASI_VE_ADAY_OZELLIKLER.md`|
|Tam proje özeti|`results/PROJE_BASLANGICTAN_ENSEMBLE_ASAMASINA_TAM_RAPOR.md`|
|Birleştirme uygulama kodu|`composite_feature_nested_cv.js`|
|ID3 322–320 karşılaştırma kodu|`id3_benign_weighted_322_vs_320.js`|
|CatBoost senaryo kodu|`catboost_scenario1_composite_nested_cv.py`|
|Benign finalist kodu|`catboost_benign_weighted_finalists_nested_cv.py`|
|Üretilmiş robust veri|`YARISMA_TRAIN_CFTR_COMPOSITE_ROBUST.csv`|
|Dönüşüm parametreleri ve kolon sırası|`YARISMA_TRAIN_CFTR_COMPOSITE_ROBUST.preprocessing.json`|

## 11. Nihai karar

Mentorun istediği korelasyona göre eşleştirme, aynı matematiksel ölçeğe taşıma, tek kolona birleştirme ve yeniden model eğitme işlemleri uygulanmıştır.

1. Korelasyon grupları oluşturulmuştur.
2. Özellikler eğitim fold'unda öğrenilen medyan ve IQR ile robust standardize edilmiştir.
3. Standartlaştırılmış değerler ortalanarak birleşik kolonlar üretilmiştir.
4. Ham grup üyeleri çıkarılarak kolon sayısı azaltılmıştır.
5. CatBoost ve ID3 modelleri nested CV ile yeniden eğitilmiştir.
6. CatBoost'un ilk dengeli değerlendirmesinde 322'den 320 özelliğe inmek MCC'yi `0.4792 → 0.5023` artırmıştır.
7. Benign ağırlıklı nihai karşılaştırmada ise en iyi tek model 322 özellikli ID3 olmuştur.

**Son yorum:** Mentorun önerdiği yöntem denenmiş ve CatBoost için faydalı bir alternatif üretmiştir; ancak bütün modeller için otomatik olarak daha iyi değildir. Yarışmanın benign ağırlıklı hedefinde ana tek model olarak 322 özellikli ID3 korunmuş, 320 özellikli birleşik CatBoost ise farklı hata örüntüsü sağladığı için ensemble bileşeni olarak değerlendirilmiştir.

## 12. Mentöre verilecek kısa açıklama

> Information Gain elemesinden sonra kalan 322 özellik içinde Pearson ve Spearman korelasyonları kullanılarak benzer kolonlar eşleştirildi. Ham değerlerin ölçekleri farklı olduğu için doğrudan ortalama alınmadı; her CV fold'unda yalnız eğitim verisinden öğrenilen medyan ve IQR ile robust standardizasyon yapıldı. Aynı gruptaki standartlaştırılmış değerlerin ortalaması alınarak tek bir birleşik kolon üretildi ve ham grup üyeleri çıkarıldı. En güvenilir AL_6–AL_251 ve AL_1–AL_211 çiftleriyle özellik sayısı 322'den 320'ye düşürüldü. İlk CatBoost nested CV karşılaştırmasında MCC 0.4792'den 0.5023'e yükseldi ve 100 benign içindeki FP 41.9'dan 36.2'ye düştü. Ancak benign ağırlıklı nihai değerlendirmede 322 özellikli ID3, MCC 0.4684 ve FP 11.4 ile 320 özellikli birleşik ID3'ten daha iyi sonuç verdi. Bu nedenle birleştirme CatBoost için yararlı bir alternatif, fakat ID3 için uygun olmayan bir dönüşüm olarak değerlendirildi.
