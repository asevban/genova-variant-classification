# CFTR Projesi – Slayt Gösterisi Metni

## Slayt 1 — CFTR varyant sınıflandırmasında özellik azaltma ve model seçimi

**Alt başlık:** Ham veriden benign odaklı ensemble modeline

**Konuşmacı notu:** Bu çalışmada amaç yalnızca yüksek accuracy elde etmek değil; benign ağırlıklı yarışma verisinde false positive sayısını azaltırken patojenik varyantları yakalama gücünü mümkün olduğunca korumaktır.

---

## Slayt 2 — Çalışmanın temel problemi sınıf dağılımı değişimidir

- Eğitim verisi: 111 varyant
- 21 benign, 90 patojenik
- Ham dosya: 353 sütun
- Gerçek model özelliği: 351
- Beklenen test: yaklaşık 100 benign + 20 patojenik

**Ana mesaj:** Model eğitimde patojenik çoğunluk görürken testte benign çoğunlukla karşılaşacaktır.

**Konuşmacı notu:** `Variant_ID` ve `Label` model özelliği değildir. Test dağılımının eğitimden ters olması, varsayılan 0.50 karar eşiğini ve yalnız accuracy ile seçim yapmayı riskli hâle getirir.

---

## Slayt 3 — Başarıyı tek bir skorla değil, hata dengesiyle ölçtük

- FP/100 benign: Kaç benign örnek yanlışlıkla patojenik deniyor?
- Specificity: Benign örnekleri doğru tanıma oranı
- Recall: Patojenik örnekleri yakalama oranı
- MCC: Dengesiz sınıflarda dengeli genel başarı
- Macro-F1: İki sınıfa eşit önem
- Projeksiyon F1: 100 benign + 20 patojenik varsayımındaki tahmini F1

**Ana mesaj:** FP azalırken recall ve MCC'nin ne kadar değiştiği birlikte izlenmelidir.

---

## Slayt 4 — İlk ID3 modeli benignleri yeterince koruyamadı

|Ölçüt|İlk ID3 sonucu|
|---|---:|
|Accuracy|0.7775|
|Specificity|0.3810|
|Recall|0.8700|
|Macro-F1|0.6284|
|MCC|0.2572|
|FP/100 benign|61.9|

**Ana mesaj:** Yüksek recall, benign sınıftaki çok yüksek yanlış pozitif oranını gizliyordu.

**Konuşmacı notu:** Eğitimde patojenik sınıfın çoğunluk olması nedeniyle model patojenik tahmine eğilimliydi. Bu nedenle özellik analizi ve karar eşiği düzenlemesi gerekliydi.

---

## Slayt 5 — Information Gain ile her sütunun hedef bilgisi ölçüldü

|Özellik|Information Gain|
|---|---:|
|CAT_1|0.29575|
|AA_2|0.22024|
|AA_1|0.19078|
|AL_3|0.16187|
|AL_7|0.16044|
|AL_6|0.15521|
|EK_9|0.15513|

**Ana mesaj:** IG, bir özelliğin Label belirsizliğini tek başına ne kadar azalttığını gösterdi.

**Konuşmacı notu:** Sütun bazlı IG ile her özellik bağımsız incelendi. ID3 ağacındaki bölünmelerle de veri bütünü içinde koşullu katkıları gözlendi. Düşük IG tek başına otomatik silme kararı olarak kullanılmadı.

---

## Slayt 6 — İlk elemede 29 gereksiz özellik çıkarıldı

**Özellik sayısı:** `351 → 322`

Çıkarılanlar:

`CAT_4`, `CAT_5`, `AL_191`, `AL_195`, `AL_197`, `AL_200`, `AL_204`, `AL_208`, `AL_212`, `AL_220`, `AL_227`, `AL_231`, `AL_233`, `AL_236`, `AL_240`, `AL_244`, `AL_248`, `AL_250`, `AL_252`, `AL_253`, `AL_256`, `AL_263`, `AL_267`, `AL_269`, `AL_272`, `AL_276`, `AL_280`, `AL_284`, `AL_292`

**Konuşmacı notu:** Kararda Information Gain, eksiklik, sabitlik/tekrarlılık ve alan anlamı birlikte kullanıldı.

---

## Slayt 7 — İlk özellik elemesi FP’yi düşürdü, fakat bir ödünleşim oluşturdu

|Yapı|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|---:|---:|---:|
|Ham 351 özellik|0.8762|0.6778|0.6661|0.4386|0.5902|12.4|
|322 özellik|0.9524|0.5867|0.6229|0.4224|0.6430|4.8|

**Ana mesaj:** FP düştü; ancak recall ve dengeli skorlar da geriledi.

**Konuşmacı notu:** `FP=4.8` ön sonuçtur. Karar eşiği ile değerlendirme aynı CV tahminlerinde yapıldığı için iyimser olabilir. Daha sonra nested CV ile `11.4` olarak daha gerçekçi biçimde ölçüldü.

---

## Slayt 8 — Hard Voting tek ağacın kararsızlığını azalttı

- 21 ID3 ağacı
- Dengeli bootstrap örnekleri
- Her ağaçta rastgele özellik alt kümesi
- Nihai karar: ağaç oylarının toplamı
- Oy eşiği yükselirse FP azalır; recall düşebilir

**Ana mesaj:** Eşik, yarışma dağılımına göre modelin hata dengesini belirleyen ana araçtır.

---

## Slayt 9 — İkili korelasyon tekrarlı bilgiyi aradı

- 316 sayısal özellik
- 49.770 benzersiz çift
- 36.596 hesaplanabilen çift
- 13.174 yetersiz ortak gözlem veya sabit değer nedeniyle hesaplanamayan çift
- 7 çok güçlü Pearson çifti
- 14.480 çok zayıf çift

**Yöntemler:** Pearson, Spearman, Cramér's V ve Eta

**Konuşmacı notu:** Pearson doğrusal, Spearman monoton ilişkiyi ölçtü. Kategorik sütunlara keyfî sayı verilip Pearson uygulanmadı.

---

## Slayt 10 — Yüksek korelasyon tek başına kolon silmek için yeterli değildi

|Çift|Pearson|Spearman|İlk aday|
|---|---:|---:|---|
|AL_5–AL_16|0.935|0.293|AL_16|
|AL_5–AL_112|0.935|0.607|AL_112|
|AL_5–AL_40|0.933|0.475|AL_40|
|AL_6–AL_251|0.921|0.736|AL_251|
|AL_4–AL_38|0.920|0.125|AL_38|
|AL_5–AL_304|0.907|0.004|AL_304|
|AL_1–AL_211|0.905|0.640|AL_211|

**Ana mesaj:** Tutulacak temsilci; IG, Label ilişkisi, eksiklik ve CV ile seçildi.

---

## Slayt 11 — Korelasyon hattı 314 özellikli alternatif oluşturdu

İlk çıkarma adayları:

`AL_16`, `AL_112`, `AL_40`, `AL_251`, `AL_38`, `AL_304`, `AL_211`, `CAT_6`

**Özellik sayısı:** `322 → 314`

`CAT_6` kanıtları:

- IG: `0.00551`
- Label ile Cramér's V: `0.06543`
- Kontrollerin `%93.15`inde düşük koşullu IG

**Ana mesaj:** 314 özellikli yapı ana veri setinin yerine otomatik geçmedi; alternatif senaryo olarak korundu.

---

## Slayt 12 — Üçlü analiz yeni sayısal kolon silme kararı üretmedi

- 5.209.260 benzersiz üçlü
- 15.627.780 parsiyel ilişki
- 99.540 Label merkezli kontrol
- 49 zayıf sayısal aday
- 103.362 koşullu Information Gain hesabı
- İki doğrulama yönteminin kesişiminde yeni sayısal aday: `0`

**Ana mesaj:** Doğrusal olarak zayıf görünen özellikler doğrusal olmayan hedef bilgisi taşıyabildiği için korundu.

---

## Slayt 13 — Grafikler yüksek Pearson’ın birkaç noktaya bağlı olduğunu gösterdi

- Her güçlü çiftte yalnız 14–15 ortak gerçek gözlem
- Noktaların çoğu sıfıra yakın kümeleniyor
- Birkaç büyük değer Pearson’ı yükseltiyor
- Spearman birçok çiftte belirgin biçimde düşük
- Normalizasyon ilişkiyi güvenilir hâle getirmiyor

**Önerilen görsel:** `YEDI_COK_GUCLU_CIFT_MATPLOTLIB.png`

**Ana mesaj:** Bütün yüksek korelasyonlu çiftlerin doğrudan ortalamasını almak güvenli değildi.

---

## Slayt 14 — AL_112 log dönüşümü FP’yi azalttı ama genel dengeyi bozdu

|Yapı|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|---:|---:|---:|
|Temel 314|0.8286|0.6956|0.6652|0.4184|0.5450|17.1|
|AL_112_LOG|0.8952|0.6022|0.6189|0.3897|0.5665|10.5|

**Ana mesaj:** Daha az FP tek başına yeterli değildi; recall, Macro-F1 ve MCC düştüğü için log özellik reddedildi.

---

## Slayt 15 — Aykırı gözlemler veri hatası kanıtı olmadığı için silinmedi

|Durum|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|
|Temel|0.4184|0.5450|17.1|
|VAR_001964 çıkarılmış|0.3794|0.5043|21.0|
|VAR_001873 çıkarılmış|0.4323|0.5712|14.3|
|VAR_001806 çıkarılmış|0.4167|0.5610|15.0|
|VAR_001885 çıkarılmış|0.3940|0.5284|18.0|
|Dördü birlikte çıkarılmış|0.3400|0.4902|20.0|

**Ana mesaj:** Tek gözlemdeki küçük CV kazancı biyolojik bir kaydı silmek için yeterli kanıt değildir.

---

## Slayt 16 — Korelasyonlu çiftler tek kolonda birleştirilerek de denendi

- Her fold'da medyan ve IQR yalnız eğitim verisinden öğrenildi
- Özellikler robust standardize edildi
- Gruptaki mevcut standartlaştırılmış değerlerin ortalaması alındı
- Ham grup üyeleri çıkarıldı

Güvenilir çiftler:

- `AL_6 + AL_251`
- `AL_1 + AL_211`

**Ana mesaj:** Veri sızıntısı olmadan, benzer özellikleri tek temsilci yerine birleşik bilgi olarak kullanmayı denedik.

---

## Slayt 17 — İki özellik senaryosu birlikte korundu

|Senaryo|Tanım|Özellik|
|---|---|---:|
|1A|29 elemeden sonraki ham azaltılmış yapı|322|
|1C|İki güvenilir çift birleştirildi|320|
|2A|Sekiz korelasyon adayı çıkarıldı|314|
|2B|Güvenilir çiftler birleşik, diğer adaylar çıkarılmış|314|

**İlk CatBoost sonucu:** Senaryo 1C, MCC `0.5023` ile dengeli ön karşılaştırmanın kazananıydı.

**Konuşmacı notu:** Bu ön karşılaştırmada FP hâlâ yüksekti. Yarışma hedefi için eşikler daha sonra benign ağırlıklı nested CV içinde yeniden seçildi.

---

## Slayt 18 — Benign ağırlıklı nested CV’de en iyi tek model 322 özellikli ID3 oldu

|Model|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|
|ID3 – 322|0.4684|0.6177|11.4|
|ID3 – 320|0.4372|0.5687|15.2|
|CatBoost – 320|0.4061|0.5373|17.1|
|CatBoost – 314|0.3621|0.4830|22.9|

**Ana mesaj:** Birleştirme CatBoost’a sınırlı katkı sağlasa da ID3 özellikleri ayrı kullanırken daha başarılı oldu.

---

## Slayt 19 — PyCaret farklı algoritma ailelerini taramak için kullanıldı

**Preprocessing:**

- Sayısal eksikler: medyan
- Kategorik eksikler: mod
- Robust normalizasyon
- One-hot encoding
- PCA, otomatik outlier silme ve otomatik feature selection kapalı
- 5-fold stratified CV
- Sıralama ölçütü: MCC

**İlk adaylar:** AdaBoost `0.4862`, Extra Trees `0.4086`, Decision Tree `0.3391`

**Ana mesaj:** İlk PyCaret skorları nihai model seçimi değil, aday taramasıydı.

---

## Slayt 20 — PyCaret finalistleri nested CV’de ID3’ü geçemedi

|Model|Eşik|Recall|MCC|Proj. F1|FP/100|
|---|---:|---:|---:|---:|---:|
|ID3|15.48/21|0.7022|0.4684|0.6177|11.4|
|Extra Trees|0.792|0.7111|0.3820|0.4902|23.8|
|AdaBoost|0.694|0.5178|0.2423|0.4037|21.0|

**Ana mesaj:** Extra Trees yalnız küçük bir recall artışı sağladı; bunun karşılığında yaklaşık 12.4 ek FP üretti.

---

## Slayt 21 — Ensemble taramasında 33 kombinasyon değerlendirildi

- Modeller: ID3, CatBoost, Extra Trees, AdaBoost
- İkili, üçlü ve dörtlü kombinasyonlar
- Hard voting
- Eşit ağırlıklı soft voting
- İç CV'de ağırlık seçilen soft voting
- Ağırlık ve eşik seçimi yalnız iç OOF tahminlerinde

**Ana mesaj:** Model çeşitliliğinin benign hatalarını azaltıp azaltmadığını veri sızıntısı olmadan sınadık.

---

## Slayt 22 — En iyi benign odaklı sonuç ID3 + CatBoost soft voting oldu

|Ölçüt|Tek ID3|ID3 + CatBoost|Değişim|
|---|---:|---:|---:|
|Specificity|0.8857|0.9238|+0.0381|
|Recall|0.7022|0.6422|−0.0600|
|Macro-F1|0.6863|0.6541|−0.0322|
|MCC|0.4684|0.4445|−0.0239|
|Proj. F1|0.6177|0.6349|+0.0172|
|FP/100 benign|11.4|7.6|−3.8|

**Ana mesaj:** Ensemble FP’yi azalttı; bunun bedeli patojenik recall ve genel dengede düşüş oldu.

---

## Slayt 23 — F1 neden beklenenden düşük görünüyor?

**ID3, 100 benign + 20 patojenik için yaklaşık:**

- TP: `14.0`
- FP: `11.4`
- Precision: `0.552`
- Recall: `0.702`
- F1: `0.618`

**Ana mesaj:** Testte benign sayısı fazla olduğu için orta düzey bir FP oranı bile precision'ı ve dolayısıyla F1'ı güçlü biçimde düşürüyor.

**Konuşmacı notu:** Düşük F1 yalnız modelin kötü olduğu anlamına gelmez. Eğitim-test prevalans değişimi, küçük örneklem, yüksek özellik sayısı ve eksiklikler problemi zorlaştırmaktadır.

---

## Slayt 24 — Kolon azaltma sürecinin özeti

`351 ham özellik`

`−29 IG/gereksizlik adayı`

`322 ana özellik hattı`

`−8 korelasyon adayı`

`314 alternatif hat`

**Kalıcı kararlar:**

- Ana tek-model hattı: 322 özellik
- CatBoost alternatif hattı: 320 özellikli Senaryo 1C
- Kalıcı outlier silme: yok
- AL_112_LOG: kullanılmıyor

---

## Slayt 25 — Nihai seçim hedefe göre yapılmalıdır

### Benign odaklı yarışma seçimi

**ID3 + CatBoost eşit soft voting**

- FP/100: `7.6`
- Projeksiyon F1: `0.6349`
- MCC: `0.4445`

### Dengeli tek-model seçimi

**322 özellikli ID3 Hard Voting**

- FP/100: `11.4`
- Recall: `0.7022`
- Macro-F1: `0.6863`
- MCC: `0.4684`

**Son karar:** Bağımsız testte iki aday birlikte karşılaştırılmalıdır.

---

## Slayt 26 — Sonraki adım bağımsız test ve üretim tutarlılığıdır

1. Eğitim verisiyle nihai eşik ve ensemble ağırlıklarını yalnız CV içinde seç
2. Her modelin kolon listesini ve kolon sırasını kaydet
3. Eksik değer ve ölçekleme parametrelerini yalnız train'den öğren
4. Aynı preprocessing'i test verisine değişmeden uygula
5. Tek ID3 ve ID3+CatBoost tahminlerini birlikte üret
6. Nihai seçimi bağımsız test skoru ve FP–recall maliyetine göre ver

**Kapanış mesajı:** Kolon azaltmanın amacı yalnız daha az sütun değil; daha güvenilir, açıklanabilir ve test dağılımına uygun bir karar sistemi oluşturmaktır.
