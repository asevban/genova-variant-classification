# CFTR Model Geliştirme Süreci: Amaç, Etki ve Kararlar

## 1. Projenin hedefi ve değerlendirme düzeni

Amaç, CFTR varyantlarını `Patojenik (1)` ve `Benign (0)` olarak sınıflandırırken yarışmanın benign ağırlıklı final dağılımında pozitif sınıf F1 skorunu yükseltmek ve yanlış pozitifleri düşük tutmaktır.

- Eğitim verisi: **111 varyant** — 90 patojenik, 21 benign
- Beklenen final test dağılımı: yaklaşık **20 patojenik, 100 benign**
- Başlangıç: **351 model özelliği**
- Güncel veri seti: **319 model özelliği**
- Kimlik kolonu `Variant_ID` modele özellik olarak verilmemiştir.
- Temel değerlendirme: dışta `5-fold × 5 tekrar`, içte `4-fold` nested çapraz doğrulama
- Ana metrik: final dağılımına uyarlanmış pozitif sınıf F1
- Destekleyici metrikler: MCC, recall, specificity ve FP/100 benign

Ön işleme, model/parametre seçimi ve eşik öğrenimi yalnız eğitim fold'unda yapılmıştır. Dış test fold'u seçimlerde kullanılmamıştır.

## 2. Veri ve özellik geliştirme çalışmaları

### 2.1 Information Gain ile ilk özellik elemesi — Etkili, kabul edildi

**Neden yapıldı?** Hedef sınıf hakkında çok az bilgi taşıyan, sabit veya büyük ölçüde gereksiz kolonları azaltmak; küçük örneklemde model karmaşıklığını düşürmek.

**Ne yapıldı?** Her sütunun sınıf etiketine ilişkin Information Gain değeri hesaplandı. Bilgi kazancı, eksiklik oranı, sabitlik ve model doğrulama sonuçları birlikte değerlendirildi.

**Sonuç:** Model özellik sayısı `351 → 322` oldu; **29 özellik çıkarıldı**. Önceki deneylerde FP/100 benign değerinin yaklaşık `4.8` seviyesine kadar düştüğü görülse de bu değer daha sonraki nested-CV düzeniyle birebir karşılaştırılabilir değildir.

**Karar:** İlk azaltma kalıcı olarak kabul edildi.

### 2.2 İkili ve üçlü korelasyon analizi — Tanısal olarak etkili

**Neden yapıldı?** Aynı veya çok benzer bilgiyi taşıyan kolonları belirlemek ve kolon sayısını matematiksel olarak azaltma olasılığını incelemek.

**Ne yapıldı?** Sayısal kolonlarda Pearson ve Spearman; kategorik kolonlarda uygun sayısal kodlama/ilişki ölçümleri kullanıldı. Güçlü ve zayıf çiftler ile üçlü gruplar çıkarıldı. Korelasyon adayları Information Gain ile ayrıca kontrol edildi.

**Sonuç:** Korelasyon tek başına kolon silme gerekçesi yapılmadı. Bazı yüksek korelasyonların yalnız birkaç ortak gözlemden veya aykırı değerden kaynaklandığı görüldü.

**Karar:** Analiz aday üretmekte yararlı oldu; doğrudan toplu kolon silme yapılmadı.

### 2.3 Korelasyonlu kolonları robust birleştirme — Modele bağlı sonuç

**Neden yapıldı?** Mentorun “benzer kolonları aynı matematiksel çerçeveye alıp tek kolon gibi kullandırma” önerisini uygulamak.

**Ne yapıldı?** `AL_6–AL_251` ve `AL_1–AL_211` çiftleri, her fold'da yalnız eğitim bölümünden öğrenilen medyan ve IQR ile robust standartlaştırıldı; standartlaştırılmış değerlerin ortalaması birleşik kolon olarak kullanıldı.

**Sonuç:** Özellik sayısı `322 → 320` oldu. İlk CatBoost karşılaştırmasında MCC `0.4792 → 0.5023`, FP `41.9 → 36.2` ile iyileşme görüldü. Fakat benign ağırlıklı nihai değerlendirmede 322 özellikli ID3, 320 özellikli birleşik ID3'ten daha iyi kaldı.

**Karar:** Birleştirme CatBoost için yararlı bir alternatifti; bütün modeller için otomatik üstünlük sağlamadı.

### 2.4 Korelasyonla sekiz kolon çıkarılan 314 özellikli senaryo — Etkisiz

**Neden yapıldı?** Çok güçlü ilişkilerde tekrarlı bilgiyi daha agresif azaltmak.

**Sonuç:** 314 özellikli temsilci/bileşik senaryolar 320 ve 322 özellikli alternatifleri güvenilir biçimde geçmedi. Düşük örnek sayılı korelasyonların kolon silme kararını kırılganlaştırdığı görüldü.

**Karar:** Sekiz kolonluk agresif eleme nihai veri setine alınmadı.

### 2.5 Aykırı değer çıkarma — Etkisiz ve riskli

**Neden yapıldı?** Bazı yüksek Pearson korelasyonlarının tek bir uç gözlem tarafından oluşturulup oluşturulmadığını görmek.

**Ne yapıldı?** Yedi güçlü çiftte aykırı değer varken/yokken grafikler, korelasyonlar, Information Gain ve model skorları karşılaştırıldı.

**Sonuç:** Aykırı gözlem çıkarıldığında bazı korelasyonlar keskin biçimde kayboldu; ancak gözlemin hatalı olduğuna ilişkin bağımsız kanıt bulunmadı. Bazı çıkarma denemeleri model skorunu değiştirse de sonuçlar kararlı değildi.

**Karar:** Hiçbir varyant kalıcı olarak silinmedi.

### 2.6 `CAT_6` çıkarılması — Etkili, kabul edildi

**Neden yapıldı?** `CAT_6` düşük ve kararsız bilgi taşıyor, mevcut ensemble'ın benign ayrımını olumsuz etkiliyor olabilir düşüncesi.

**Sonuç:** Aynı ana modelde 320 özellikli sürüm MCC `0.4756`, projeksiyon F1 `0.6309`, FP `10.48` verdi. `CAT_6` çıkarılmış 319 özellikli sürüm MCC `0.5225`, F1 `0.6798`, FP `8.57` verdi.

**Karar:** `CAT_6` çıkarıldı ve **319 özellikli veri seti** kabul edildi.

### 2.7 Ek fold-içi Information Gain azaltması — Etkisiz

**Neden yapıldı?** 319 özelliği 100, 150, 200 veya 250 özelliğe indirerek daha sade model elde etmek.

**Sonuç:** Fold'lar farklı özellik sayıları seçti; sabit ve genellenebilir bir kolon kümesi oluşmadı. CatBoost F1'i geriledi, FP yükseldi.

**Karar:** Ek eleme reddedildi; bütün ana modellerde aynı 319 özellik korundu.

### 2.8 Yeni özellik ve alternatif temsil deneyleri — Etkisiz

Eksiklik göstergeleri, özellik ailesi late fusion ve `AL_112_LOG` gibi dönüşümler denendi. Eksiklik göstergeleri F1 `0.6090`, MCC `0.4674`, FP `12.38`; late fusion en iyi benign hedefte F1 `0.6624`, FP `9.52` verdi.

**Karar:** Yeni özellikler ve alternatif temsiller nihai 319 özellikli veri setine alınmadı.

## 3. Tek model taraması ve geliştirme

### 3.1 ID3 — Yararlı temel model

**Neden kullanıldı?** Information Gain ile doğal uyumu, küçük veri setinde yorumlanabilir olması ve benign sınıfta düşük FP üretmesi.

319 özellikli sabit ID3 sonucu:

- MCC: `0.4260`
- Projeksiyon F1: `0.6034`
- FP/100 benign: `9.52`
- Recall: `0.6378`

Ayarlı ID3, F1 `0.5845` ve FP `11.4` verdi.

**Karar:** Hiperparametre ayarı reddedildi; baseline ID3 korundu.

### 3.2 CatBoost — Yararlı fakat tek başına yetersiz

**Neden kullanıldı?** Sayısal ve kategorik özellikleri birlikte işleyebilmesi, eksik değerlerle çalışabilmesi ve doğrusal olmayan ilişkileri yakalayabilmesi.

- Baseline: MCC `0.4074`, F1 `0.5437`, FP `16.2`
- Nested ayarlı: MCC `0.4133`, F1 `0.5722`, FP `12.4`

**Karar:** Tek modelde küçük iyileşme vardı; ancak ayarlanmış sürüm ensemble'ı iyileştirmediği için nihai üçlüde baseline/kararlı yapı kullanıldı.

### 3.3 AdaBoost ve Extra Trees — Ana model için yetersiz

PyCaret aday taraması sonrası aynı nested-CV düzeninde kontrol edildiler. AdaBoost bazı ensemble'larda tamamlayıcı oldu; Extra Trees'in küçük recall farkı yüksek FP maliyetine değmedi.

**Karar:** AdaBoost eski yedek ensemble'da korundu; Extra Trees nihai modele alınmadı.

### 3.4 Random Forest — Ensemble içinde etkili, kabul edildi

**Neden kullanıldı?** Ağaç çeşitliliği ile ID3 ve CatBoost'un hatalarını tamamlamak.

- Tek Random Forest: MCC `0.4404`, F1 `0.5642`, FP `16.2`
- Eski `ID3 + CatBoost + AdaBoost`: MCC `0.4964`, F1 `0.6633`, FP `8.57`
- Yeni `ID3 + CatBoost + Random Forest`: MCC `0.5225`, F1 `0.6798`, FP `8.57`

**Karar:** Random Forest, AdaBoost'un yerine ana ensemble'a alındı. FP aynı kalırken recall, F1 ve MCC yükseldi.

### 3.5 Logistic Regression ve Robust RBF-SVM — Etkisiz

- Logistic Regression: MCC `0.0478`, F1 `0.2872`, FP `71.4`
- Robust RBF-SVM: MCC `-0.0117`, FP `73.33`

**Karar:** Veri ilişkileri bu modeller için uygun bulunmadı; ikisi de reddedildi.

### 3.6 TabPFN — Bağımsız deney, nihai modele alınmadı

TabPFN yalnız karşılaştırma amacıyla 319, 320 ve 322 özellikli veriler üzerinde denendi. Kullanıcı kararı ve yarışma çalışma koşulları nedeniyle nihai modele veya ensemble'a eklenmedi.

## 4. Ensemble geliştirme deneyleri

### 4.1 Hard voting ve soft voting — Soft voting kabul edildi

**Neden yapıldı?** Hard voting yalnız sınıf kararlarını, soft voting ise model güven olasılıklarını birleştirir. Modellerin farklı güven düzeylerinden yararlanmak hedeflendi.

Çok sayıda ikili, üçlü ve dörtlü kombinasyon denendi. Güncel ana yapı **ID3 + CatBoost + Random Forest eşit soft voting** oldu.

### 4.2 Sabit ve dinamik ağırlıklar — Etkisiz

Geçmişte çeşitli ağırlıklar denenmiş; dinamik ağırlık FP'yi yaklaşık `14.3` seviyesine yükseltmişti. Güncel üçlüde yapılan nested ağırlık deneyi de:

- F1: `0.6570`
- MCC: `0.5020`
- FP/100: `9.52`

verdi. Eşit ağırlıklı güncel sonuç `0.7580 / 0.5717 / 4.76` idi.

**Karar:** Ağırlık optimizasyonu reddedildi; eşit ağırlık korundu.

### 4.3 Lojistik stacking — Etkisiz

Recall yükselse de F1 `0.6005`, MCC `0.4791`, FP `14.3` oldu.

**Karar:** FP artışı nedeniyle reddedildi.

### 4.4 İki aşamalı benign güven kapısı — Etkisiz

İç CV çok katı bir benign kapısı seçti ve yapı pratikte baseline ensemble ile aynı sonucu verdi.

**Karar:** Ek karmaşıklık performans katkısı sağlamadığı için reddedildi.

### 4.5 Hard-negative öğrenme — Etkisiz

Zor benign örnekler CatBoost eğitiminde daha yüksek ağırlıklandırıldı. FP `16.2 → 18.1`, F1 `0.5437 → 0.5262` oldu.

**Karar:** Amaçlanan FP azalması oluşmadığı için reddedildi.

### 4.6 CatBoost seed bagging — Tek modelde sınırlı, ensemble'da etkisiz

Tek CatBoost'ta FP `16.2 → 15.2` ile küçük iyileşme sağladı; fakat ensemble FP'si `8.6 → 12.4`, F1'i `0.6633 → 0.6104` oldu.

**Karar:** Ensemble'a alınmadı.

### 4.7 Olasılık kalibrasyonu — Etkisiz

Eski `ID3 + CatBoost + AdaBoost` ensemble üzerinde fold-içi Platt ve izotonik kalibrasyon uygulandı:

| Yöntem | Proj. F1 | MCC | FP/100 |
|---|---:|---:|---:|
| Kalibrasyonsuz | 0.6633 | 0.4964 | 8.6 |
| Platt | 0.6162 | 0.4931 | 13.3 |
| İzotonik | 0.5563 | 0.4003 | 13.3 |

**Karar:** Kalibrasyon FP'yi artırdığı için reddedildi. Güncel Random Forest'lı üçlüde tekrar denenmedi; küçük veri üzerinde tekrar tekrar ayar seçme riski nedeniyle düşük öncelikli bırakıldı.

### 4.8 Label-shift/prior düzeltmesi — Etkisiz

Beklenen final sınıf oranına olasılık düzeltmesi uygulandı. En iyi F1 `0.6511`, FP `10.48` oldu.

**Karar:** Fold-içi benign-ağırlıklı eşik mevcut prior farkını daha iyi yönettiği için reddedildi.

## 5. Eşik geliştirme çalışmaları

### 5.1 Nested benign-ağırlıklı eşik — Etkili, kabul edildi

**Neden yapıldı?** Eğitim verisi patojenik, final test seti benign ağırlıklı olduğu için sabit `0.50` eşiği uygun değildi.

Her dış fold için eşik yalnız iç CV'de, yaklaşık `100 benign + 20 patojenik` projeksiyonunda F1'i yükseltecek şekilde seçildi.

**Karar:** Güncel eşik yöntemi olarak kabul edildi.

### 5.2 Alternatif eşik hedefleri — Etkisiz

- En yüksek MCC: MCC `0.5144`, fakat FP `32.4`
- En yüksek Macro-F1: FP `32.4`
- Sabit `0.50`: FP `53.3`
- Specificity ≥ `0.95`: FP `2.9`, fakat recall `0.4511`

**Karar:** Yarışmanın F1–FP dengesi için projeksiyon-F1 eşiği korundu.

### 5.3 Kararlılık cezalı eşik — Etkisiz

F1'in fold'lar arasındaki standart sapmasına ceza uygulandı. Denemeler F1 `0.6201–0.6405`, MCC `0.4767–0.5000`, FP `9.52–13.33` verdi.

**Karar:** Baseline eşiği geçemediği için reddedildi.

## 6. Overfitting ve kararlılık denetimleri

### 6.1 Variant_ID kümeli bootstrap — Gerekli ve yararlı denetim

Ana model ile yedek model 100.000 tekrarlı paired bootstrap ile karşılaştırıldı. Ana modelin nokta tahmini daha iyi olsa da F1 ve MCC farklarının güven aralıkları sıfırı içerdi.

**Katkısı:** Ana modelin kesin üstün olduğunun iddia edilmesini önledi; örneklem belirsizliğini ölçtü.

### 6.2 Varyant ve fold kırılganlığı — Yararlı denetim

- 64 varyant tutarlı doğru
- 24 varyant aralıklı hata
- 11 varyant çoğunlukla hata
- 12 varyant tutarlı FN
- Tutarlı FP: 0

`VAR_001930` çıkarıldığında ana/yedek sıralamasının değişebildiği görüldü. Hata veya yanlış etiket kanıtı olmadığı için çıkarılmadı.

### 6.3 Cross-fitted beş-model bagging — Etkili aday

Her varyant için, o varyantı eğitimde görmeyen beş dış-fold modelinin olasılıkları ortalandı:

- Projeksiyon F1: `0.7580`
- MCC: `0.5717`
- Recall: `0.7556`
- Specificity: `0.9524`
- FP/100 benign: `4.76`

Bootstrap güven aralıkları:

- F1: `0.5864–0.9024`
- MCC: `0.4550–0.6923`
- FP/100: `0.00–14.29`

**Karar:** Güçlü final eğitim stratejisi adayıdır; ancak yalnız 21 benign bulunduğu için FP güven aralığı geniştir. Standart pooled nested-CV ile aynı değerlendirme formatı olmadığından `0.6798 → 0.7580` kesin performans artışı olarak sunulmamalıdır.

### 6.4 Model sayısı kararlılığı — Beş modeli destekledi

| Cross-fitted model sayısı | Ortalama proj. F1 | F1 SS | Ortalama FP/100 |
|---:|---:|---:|---:|
| 3 | 0.7037 | 0.0600 | 8.10 |
| 4 | 0.7404 | 0.0362 | 5.71 |
| 5 | 0.7580 | — | 4.76 |

**Karar:** Tek bir iyi alt kombinasyon seçilmedi; beş model daha yüksek ve daha kararlı sonuç verdiği için korundu.

### 6.5 Eşik hassasiyeti — Orta düzey kırılganlık bulundu

Varsayılan eşikte F1 `0.7580`, FP `4.76` oldu. Eşik ±`0.02` değiştiğinde F1 `0.6966–0.7580`, FP `4.76–9.52` aralığına geldi.

**Karar:** En iyi görünen yeni eşik seçilmedi. İç CV'de öğrenilen varsayılan eşik korundu.

### 6.6 Bileşen ablation ve ID3 yedek adayı

Cross-fitted tek-karar değerlendirmesinde:

| Model | Proj. F1 | MCC | Recall | FP/100 |
|---|---:|---:|---:|---:|
| ID3 | 0.7838 | 0.5053 | 0.6444 | 0.00 |
| ID3 + CatBoost + Random Forest | 0.7580 | 0.5717 | 0.7556 | 4.76 |

ID3'ün F1 farkı `+0.0258` görünse de %95 güven aralığı `-0.1179–0.1963`, üstünlük olasılığı `%57.9` oldu. Yalnız ID3'ün doğru bildiği 2, yalnız üçlü ensemble'ın doğru bildiği 11 varyant bulundu. Üçlünün recall üstünlüğü güven aralığıyla doğrulandı.

**Karar:** ID3'ün yalnız 21 benign örnekteki `0 FP` sonucu kesin üstünlük sayılmadı. Üçlü ensemble ana model, ID3 düşük-FP yedek aday olarak korundu.

## 7. Güncel nihai karar

### Veri seti

`YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv`

- 111 varyant
- 319 model özelliği
- `Variant_ID` yalnız kimlik
- `Label` hedef
- Bütün ensemble bileşenleri aynı 319 özelliği kullanır.

### Ana model

**ID3 + CatBoost + Random Forest — eşit ağırlıklı soft voting**

Standart `5×5` nested-CV sonucu:

- Accuracy: `0.7694`
- Specificity: `0.9143`
- Recall: `0.7356`
- Macro-F1: `0.7190`
- MCC: `0.5225`
- Projeksiyon F1: `0.6798`
- FP/100 benign: `8.57`

Cross-fitted beş-model olasılık ortalaması adayı:

- Projeksiyon F1: `0.7580`
- MCC: `0.5717`
- FP/100 benign: `4.76`

### Yedek modeller

1. **Düşük-FP yedek:** cross-fitted ID3
2. **Eski guardrail:** ID3 + CatBoost + AdaBoost eşit soft voting — F1 `0.6633`, MCC `0.4964`, FP `8.57`

## 8. Genel sonuç

Model geliştirme çalışmalarında gerçek ve kalıcı katkı sağlayan adımlar şunlardır:

1. İlk Information Gain/gereksizlik elemesi
2. Güvenilir korelasyonlu çiftlerin fold-içi robust birleştirilmesi
3. `CAT_6` çıkarılarak 319 ortak özelliğe geçilmesi
4. Random Forest'ın AdaBoost yerine ensemble'a eklenmesi
5. Benign ağırlıklı nested eşik seçimi
6. Beş cross-fitted modelin olasılık ortalaması

Kalibrasyon, geniş hiperparametre ayarları, stacking, dinamik ağırlıklar, agresif kolon azaltma, hard-negative öğrenme, seed bagging ve alternatif sınıflandırıcılar güvenilir iyileşme sağlamadığı için reddedilmiştir.

Bu aşamada aynı eğitim verisi üzerinde daha fazla yöntem taramak validasyon overfitting riskini artırabilir. En güvenli yaklaşım, ana ve yedek modeli dondurmak; final test geldiğinde etiketlere erişmeden aynı kolon sırası, ön işleme ve önceden belirlenmiş eşikle tahmin üretmektir.
