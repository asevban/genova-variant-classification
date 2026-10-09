# CFTR Projesi – Ham Veriden Ensemble Aşamasına Tam Süreç Raporu

## 1. Projenin amacı

Amaç, CFTR varyantlarını sınıflandırırken gereksiz ve tekrarlı özellikleri azaltmak, benign ağırlıklı yarışma testinde yanlış pozitifleri düşürmek ve modelin sınıflar arasındaki dengeli başarısını korumaktır.

- `Label=0`: benign
- `Label=1`: patojenik
- Eğitim verisi: 111 gözlem; 21 benign, 90 patojenik
- Ham dosya: 353 sütun = `Variant_ID` + `Label` + 351 gerçek model özelliği
- Yarışma için beklenen yaklaşık dağılım: 100 benign + 20 patojenik

`Variant_ID` yalnız gözlem kimliğidir ve modele özellik olarak verilmemiştir.

## 2. Ölçütler neden kullanıldı?

- **FP (False Positive):** Gerçekte benign olan bir varyantın patojenik tahmin edilmesidir. `FP/100 benign = 100 × (1-specificity)` olarak raporlandı. Benign ağırlıklı testte özellikle önemlidir.
- **Specificity:** Benign örnekleri doğru tanıma oranıdır. Yükseldikçe FP azalır.
- **Recall:** Patojenik örnekleri yakalama oranıdır. FP'yi azaltırken recall'ın aşırı düşmemesi gerekir.
- **MCC:** Dört karışıklık matrisi hücresini de kullanan dengeli bir ölçüttür. Sınıf dengesizliğinde accuracy'den daha güvenilirdir; `1` mükemmel, `0` rastgele düzeye yakın sonuçtur.
- **Macro-F1:** İki sınıfın F1 değerlerine eşit önem verir.
- **Projeksiyon F1:** CV'deki specificity ve recall'ın yaklaşık 100 benign + 20 patojenik yarışma dağılımına taşınmasıyla hesaplanan tahmini F1'dir; gerçek bağımsız test sonucu değildir.

## 3. Ham veri ve ilk Information Gain / ID3 analizi

Her sütunun hedef hakkında ne kadar bilgi taşıdığı Information Gain (IG) ile ölçüldü. Hedef entropisi `0.699772 bit` bulundu. Hem sütunların tek başına IG değerleri hem de ID3 ağacının veri bütünü içinde kullandığı koşullu bölünmeler incelendi.

En yüksek IG örnekleri:

|Özellik|IG|
|---|---:|
|CAT_1|0.29575|
|AA_2|0.22024|
|AA_1|0.19078|
|AL_3|0.16187|
|AL_7|0.16044|
|AL_6|0.15521|
|EK_9|0.15513|

İlk tek-ID3 kontrolünde 10 tekrar × 5-fold stratified CV kullanıldı. Sonuçlar: accuracy `0.7775`, specificity `0.3810`, recall `0.8700`, Macro-F1 `0.6284`, MCC `0.2572`. Yaklaşık 100 benign için FP `61.9` idi. Bu model patojenik çoğunluk sınıfını iyi yakalıyor, fakat benignleri yeterince koruyamıyordu.

## 4. İlk gereksizlik elemesi: 351 özellikten 322 özelliğe

IG, eksiklik, sabitlik/tekrarlılık ve alan anlamı birlikte değerlendirilerek 29 özellik çıkarıldı:

`CAT_4`, `CAT_5`, `AL_191`, `AL_195`, `AL_197`, `AL_200`, `AL_204`, `AL_208`, `AL_212`, `AL_220`, `AL_227`, `AL_231`, `AL_233`, `AL_236`, `AL_240`, `AL_244`, `AL_248`, `AL_250`, `AL_252`, `AL_253`, `AL_256`, `AL_263`, `AL_267`, `AL_269`, `AL_272`, `AL_276`, `AL_280`, `AL_284`, `AL_292`.

Sonuç: **351 gerçek model özelliği → 322 özellik**. Eski raporlardaki `352 → 322` ifadesi `Variant_ID`yi özellik havuzu sayımına dahil eden adlandırmadır; model girdisi açısından doğru sayı `351 → 322`dir.

21 dengeli-bootstrap ID3 ağacından oluşan Hard Voting ön karşılaştırması:

|Yapı|Özellik|Eşik|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Ham özellikler|351|16/21|0.8762|0.6778|0.6661|0.4386|0.5902|12.4|
|29 aday çıkarılmış|322|17/21|0.9524|0.5867|0.6229|0.4224|0.6430|4.8|

Bu ön sonuçta özellik eleme FP'yi ciddi biçimde düşürdü; ancak recall, Macro-F1 ve MCC de düştü. Ayrıca `4.8` değeri, oy eşiğinin değerlendirmede kullanılan aynı CV tahminlerinden seçilmesi nedeniyle iyimser olabilirdi. Bu nedenle daha sonra nested CV ile yeniden ölçüldü.

## 5. Hard Voting seçimi

Tek bir ID3 ağacının küçük veride kararsız olmasını azaltmak için 21 ID3 ağacı dengeli bootstrap örnekleri ve rastgele özellik alt kümeleriyle eğitildi. Hard Voting'de her ağaç sınıf oyu verir; patojenik karar için gereken oy sayısı eşik olarak seçilir. Eşik yükseldikçe genellikle FP azalır, fakat patojenik recall da düşebilir.

## 6. İkili korelasyon analizi

322 özellik içinde 316 sayısal özellik için bütün benzersiz çiftler karşılaştırıldı:

- Toplam sayısal çift: 49.770
- Hesaplanabilen: 36.596
- Yetersiz ortak gözlem veya ortak satırlarda sabit değer nedeniyle hesaplanamayan: 13.174
- Pearson `|r|≥0.90` olan çok güçlü çift: 7
- Her iki yöntemde `|r|<0.10` olan çok zayıf çift: 14.480

Pearson doğrusal, Spearman sıralamaya dayalı monoton ilişkiyi ölçtü. Kategorik değişkenlere keyfî sayı kodu verip Pearson/Spearman uygulamak yerine kategorik-kategorik için Cramér's V, kategorik-sayısal için Eta kullanıldı.

Yedi çok güçlü çift ve ilk temsilci-eleme adayları:

|Çift|Pearson|Spearman|Çıkarma adayı ve IG|Tutulan eş ve IG|
|---|---:|---:|---|---|
|AL_5–AL_16|0.93474|0.29286|AL_16: 0.03039|AL_5: 0.11417|
|AL_5–AL_112|0.93471|0.60714|AL_112: 0.01867|AL_5: 0.11417|
|AL_5–AL_40|0.93254|0.47500|AL_40: 0.03375|AL_5: 0.11417|
|AL_6–AL_251|0.92097|0.73626|AL_251: 0.03523|AL_6: 0.15521|
|AL_4–AL_38|0.92008|0.12500|AL_38: 0.13115|AL_4: 0.13115|
|AL_5–AL_304|0.90728|0.00357|AL_304: 0.09363|AL_5: 0.11417|
|AL_1–AL_211|0.90545|0.63956|AL_211: 0.07923|AL_1: 0.11806|

`AL_38` düşük IG'li değildir; eşit IG durumunda `AL_4`ün Label ile doğrudan ilişkisi daha güçlü olduğu için temsilci olarak `AL_4` tutuldu. Kategorik analizde `CAT_6` için IG `0.00550697` ve Label ile Cramér's V `0.06543` bulundu. Böylece geçici çıkarma listesi yedi sayısal özellik + `CAT_6` olmak üzere sekiz sütuna ulaştı ve alternatif veri hattı **322 → 314 özellik** oldu.

## 7. Üçlü/koşullu analiz ve IG doğrulaması

Üçlü analizde `r(X,Y|Z)` parsiyel korelasyonu ile üçüncü özellik kontrol edildiğinde iki özellik arasındaki ilişkinin sürüp sürmediği incelendi:

- 5.209.260 benzersiz üçlü
- Üç kontrol yönüyle 15.627.780 parsiyel ilişki
- Label merkezli 99.540 kontrol ilişkisi
- Parsiyel korelasyonda 49 zayıf sayısal aday

Ancak korelasyon hedef bilgisini tek başına ölçmez. Bu nedenle `I(X;Label|Z)` koşullu Information Gain ile 103.362 kontrol değeri hesaplandı. 49 sayısal adayın hiçbiri iki yöntemin şartlarını birlikte sağlamadı. Yalnız `CAT_6`, kontrollerin `%93.15`inde düşük koşullu IG gösterdi. Sonuç olarak üçlü analiz yeni sayısal kolon silme kararı üretmedi; 49 aday korundu.

## 8. Çok güçlü çiftlerin grafikleri, normalizasyon ve log dönüşümü

Yedi çift için saçılım grafikleri çizildi; lineer, logaritmik, üstel ve kuvvet ilişkileri karşılaştırıldı. Ortak gerçek gözlem sayısı yalnız 14–15 idi. Pearson'ın yüksek, Spearman'ın çoğu çiftte düşük olması birkaç etkili gözlemin Pearson'ı şişirdiğini gösterdi.

Eksikler medyanla tamamlanarak z-skor ve Min–Max ölçekleme incelendi. Normalizasyon yalnız ölçekleri ortaklaştırır; ilişkinin güvenilirliğini artırmaz. Bu nedenle bütün yüksek korelasyonlu çiftlerin doğrudan ortalamasını almak uygun bulunmadı.

`AL_5–AL_112` için üstel model aday oldu:

|Model|R²|LOOCV RMSE|
|---|---:|---:|
|Lineer|0.87368|0.00909|
|Üstel|0.97698|0.00566|

İlişkiyi doğrusallaştırmak için `AL_112_LOG=ln(AL_112)` denenmiştir. Nested CV sonucu:

|Yapı|Özellik|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|
|Temel korelasyon modeli|314|0.8286|0.6956|0.6652|0.4184|0.5450|17.1|
|AL_112_LOG ekli|315|0.8952|0.6022|0.6189|0.3897|0.5665|10.5|

Log dönüşümü FP'yi azalttı; fakat recall, Macro-F1 ve MCC'yi düşürdü. Dengeli genel başarı açısından doğrulanmadığı için ana modele alınmadı.

## 9. Aykırı gözlem incelemesi

Korelasyonu en fazla değiştiren gözlemler tek tek çıkarılarak aynı nested CV düzeninde test edildi. Bir gözlemin yalnız korelasyonu düşürmesi silme gerekçesi kabul edilmedi; doğrulanmış veri hatası veya tutarlı model kazancı arandı.

|Durum|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|
|Gözlemler korunmuş|0.8286|0.6956|0.6652|0.4184|0.5450|17.1|
|VAR_001964 çıkarılmış|0.7900|0.6911|0.6461|0.3794|0.5043|21.0|
|VAR_001873 çıkarılmış|0.8571|0.6854|0.6673|0.4323|0.5712|14.3|
|VAR_001806 çıkarılmış|0.8500|0.6822|0.6568|0.4167|0.5610|15.0|
|VAR_001885 çıkarılmış|0.8200|0.6822|0.6484|0.3940|0.5284|18.0|
|Dördü birlikte çıkarılmış|0.8000|0.6494|0.6107|0.3400|0.4902|20.0|

`VAR_001873` çıkarıldığında sayısal olarak küçük bir artış görüldü; fakat tek gözlemlik, küçük veri setine duyarlı bu kazanç veri/biyolojik hata kanıtı değildir. `VAR_001964` ve diğer çıkarmalar genel olarak performansı kötüleştirdi. Bu nedenle hiçbir satır kalıcı olarak silinmedi.

## 10. Korelasyonlu özellikleri tek kolonda birleştirme

Mentorun önerisi doğrultusunda benzer sütunlar yalnız silinmek yerine matematiksel olarak aynı çerçeveye alındı. Her CV fold'unda medyan ve IQR yalnız eğitim kısmından öğrenildi, değerler robust standardize edildi ve mevcut standartlaştırılmış grup değerlerinin ortalaması alındı. Böylece veri sızıntısı önlendi.

Gruplar:

- `GRUP_AL5`: AL_5, AL_16, AL_112, AL_40, AL_304
- `GRUP_AL6_AL251`: AL_6, AL_251
- `GRUP_AL4_AL38`: AL_4, AL_38
- `GRUP_AL1_AL211`: AL_1, AL_211

İlk ID3 ön deneyinde en iyi tek grup `AL_1–AL_211` oldu: Macro-F1 `0.6763`, MCC `0.4332`, FP `17.1`; 314 özellikli temsilci-seçim referansı Macro-F1 `0.6392`, MCC `0.3967`, FP `14.3` idi. Bütün grupları birleştirmek fayda sağlamadı.

## 11. İki veri senaryosu

- **Senaryo 1:** İlk 29 elemeden sonraki 322 özellikli veri.
- **Senaryo 2:** Buna ek olarak sekiz korelasyon adayının temsilci seçimiyle çıkarıldığı 314 özellikli veri.
- **Senaryo 1C:** 322'den başlayıp yalnız daha güvenilir `AL_6–AL_251` ve `AL_1–AL_211` çiftlerini birleştiren 320 özellikli yapı.
- **Senaryo 2B:** Güvenilir çiftleri birleştirip diğer korelasyon adaylarını çıkaran 314 özellikli yapı.

İlk CatBoost nested karşılaştırması farklı, daha dengeli seçim hedefiyle şu sonucu verdi:

|Senaryo|Özellik|Specificity|Recall|Macro-F1|MCC|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|
|1A – ham azaltılmış|322|0.5810|0.9000|0.7396|0.4792|41.9|
|1C – güvenilir gruplar|320|0.6381|0.8867|0.7503|0.5023|36.2|
|2A – temsilci seçim|314|0.5905|0.8867|0.7317|0.4641|41.0|
|2B – bileşik|314|0.6190|0.8911|0.7469|0.4946|38.1|

Bu aşamada 1C en iyi dengeli CatBoost adayıydı; fakat FP seviyeleri yarışmanın benign ağırlıklı hedefi için yüksekti. Bu yüzden eşik amacı daha sonra doğrudan 100 benign + 20 patojenik projeksiyonuna göre değiştirildi.

## 12. Benign ağırlıklı nested CV ile adil yeniden karşılaştırma

Dış döngü 5-fold stratified CV × 5 tekrar, iç döngü 4-fold CV olarak kullanıldı. Ön işleme ve eşik yalnız dış eğitim bölümünde öğrenildi; dış test fold'u seçim görmedi.

|Model|Özellik|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|
|ID3 Hard Voting – 322|322|0.8857|0.7022|0.6863|0.4684|0.6177|11.4|
|ID3 Hard Voting – Senaryo 1C|320|0.8476|0.7000|0.6738|0.4372|0.5687|15.2|
|CatBoost – Senaryo 1C|320|0.8286|0.6822|0.6558|0.4061|0.5373|17.1|
|CatBoost – Senaryo 2B|314|0.7714|0.6822|0.6393|0.3621|0.4830|22.9|

Bu daha güvenilir değerlendirmede en iyi tek model **322 özellikli ID3 Hard Voting** oldu. Korelasyonlu çiftleri birleştirmek CatBoost'a küçük yapısal yarar sağlayabilse de ID3'te performansı düşürdü. Özellik mühendisliğinin etkisinin modele bağlı olduğu görüldü.

## 13. PyCaret model taraması

322 özellikli veri üzerinde PyCaret ile farklı algoritmalar tarandı. İlk taramada AdaBoost MCC `0.4862`, Extra Trees `0.4086` ile aday oldu; ancak bu değerler sabit karar kuralına dayalı aday tarama skorlarıydı. Aynı benign ağırlıklı nested CV ile yeniden değerlendirme:

|Model|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|
|ID3 – 322|0.8857|0.7022|0.6863|0.4684|0.6177|11.4|
|Extra Trees – 322|0.7619|0.7111|0.6565|0.3820|0.4902|23.8|
|AdaBoost – 322|0.7905|0.5178|0.5354|0.2423|0.4037|21.0|

Extra Trees recall'da ID3'ten yalnız `0.0089` daha yüksekken 100 benign başına yaklaşık `12.4` ek FP üretti. Tek model olarak ID3 üstün kaldı.

## 14. Ensemble deneyi

ID3, CatBoost, Extra Trees ve AdaBoost'un bütün ikili, üçlü ve dörtlü kombinasyonları hard voting, eşit soft voting ve iç CV'de ağırlık seçilen soft voting ile denendi. Toplam 33 yapı değerlendirildi. Ağırlıklar ve karar eşikleri yalnız iç OOF tahminlerinden seçildi.

En iyi sonuçlar:

|Kombinasyon|Yöntem|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---|---:|---:|---:|---:|---:|---:|
|ID3 + CatBoost|Eşit soft voting|0.9238|0.6422|0.6541|0.4445|0.6349|7.6|
|Dört model|Hard voting|0.9143|0.6556|0.6609|0.4484|0.6291|8.6|
|ID3 + CatBoost|Hard voting|0.9238|0.6089|0.6309|0.4173|0.6120|7.6|
|Tek ID3 referansı|—|0.8857|0.7022|0.6863|0.4684|0.6177|11.4|

Benign ağırlıklı projeksiyonda **ID3 + CatBoost eşit soft voting** en yüksek F1'i (`0.6349`) ve en düşük FP'yi (`7.6`) verdi. Tek ID3'e göre FP `11.4 → 7.6`, specificity `0.8857 → 0.9238` oldu. Bedeli recall'ın `0.7022 → 0.6422`, MCC'nin `0.4684 → 0.4445` ve Macro-F1'ın `0.6863 → 0.6541` düşmesidir.

## 15. Kronolojik kolon özeti

|Aşama|İşlem|Model özelliği|Kalıcı durum|
|---|---|---:|---|
|Ham veri|351 predictor|351|Başlangıç|
|IG/gereksizlik elemesi|29 sütun çıkarıldı|322|Ana veri hattı|
|Korelasyon temsilci seçimi|8 sütun daha çıkarıldı|314|Alternatif senaryo|
|Üçlü analiz|Yeni sayısal sütun silinmedi|314|49 aday korundu|
|AL_112_LOG|Bir özellik eklendi|315|Nested CV sonrası reddedildi|
|Senaryo 1C|İki güvenilir çift birleştirildi|320|CatBoost alternatifi|
|Outlier analizi|Satır silme denenip geri çevrildi|Değişmedi|Hiçbir satır kalıcı silinmedi|
|Nihai tek model hattı|322 özellikli ID3|322|En dengeli tek model|
|Nihai benign odaklı ensemble|ID3 + CatBoost soft voting|Model başına kendi hattı|En düşük FP / en yüksek proj. F1|

## 16. Son karar

İki hedefe göre iki güçlü aday vardır:

1. **Benign ağırlıklı yarışma F1'i ve düşük FP öncelikliyse:** ID3 + CatBoost eşit soft voting; projeksiyon F1 `0.6349`, FP/100 benign `7.6`, MCC `0.4445`.
2. **Patojenik recall, Macro-F1 ve genel denge öncelikliyse:** 322 özellikli tek ID3 Hard Voting; recall `0.7022`, Macro-F1 `0.6863`, MCC `0.4684`, FP/100 benign `11.4`.

111 satırlık küçük veri nedeniyle farklar kesin üstünlük kanıtı değildir. Nihai seçim bağımsız testte yapılmalıdır. Test verisine eğitimde öğrenilen kolon listesi, kolon sırası, eksik değer işlemleri, robust ölçekleme/grup dönüşümleri, model ağırlıkları ve karar eşiği aynen uygulanmalıdır.

## 17. Mentöre verilebilecek kısa özet

> Ham veride 351 model özelliği vardı. Information Gain ve gereksizlik kontrolleriyle 29 özellik çıkarılarak 322 özelliğe inildi. İkili Pearson/Spearman ve kategorik ilişki analizleri sekiz ek korelasyon adayı üretti; ancak üçlü parsiyel korelasyon ve koşullu IG, CAT_6 dışında yeni bir silme kararını desteklemedi. Çok güçlü görünen yedi çiftin yalnız 14–15 ortak gözleme sahip olduğu ve korelasyonlarının etkili noktalara duyarlı olduğu grafiklerle gösterildi. Log dönüşümü, normalizasyon ve aykırı gözlem silme deneyleri nested CV ile sınandı; genel MCC/Macro-F1 kazancı doğrulanmadığı için kalıcı uygulanmadı. Korelasyonlu çiftleri robust ölçekleyip tek kolonda birleştirme CatBoost'ta sınırlı fayda sağladı, fakat ID3'te 322 ham azaltılmış yapı daha iyi kaldı. PyCaret taramasında Extra Trees ve AdaBoost aday oldu fakat benign ağırlıklı nested doğrulamada ID3'ü geçemedi. Son ensemble taramasında ID3+CatBoost eşit soft voting FP'yi 100 benign başına 11.4'ten 7.6'ya düşürüp projeksiyon F1'i 0.6177'den 0.6349'a çıkardı; buna karşılık recall ve MCC düştü. Bu nedenle benign odaklı ana aday ensemble, dengeli yedek aday ise 322 özellikli tek ID3'tür.

## 18. PyCaret parametreleri ve seçim gerekçeleri

PyCaret çalışması iki aşamadan oluşmuştur. İlk aşama çok sayıda algoritma arasından aday bulmak için yapılan **model taraması**, ikinci aşama ise taramada öne çıkan modellerin benign ağırlıklı **nested cross-validation** ile doğrulanmasıdır. İlk taramada kapsamlı hiperparametre optimizasyonu yapılmamış; modeller büyük ölçüde varsayılan başlangıç ayarlarıyla karşılaştırılmıştır.

### 18.1. PyCaret `setup` parametreleri

```python
exp.setup(
    data=data,
    target="Label",
    index=False,
    ignore_features=["Variant_ID"],
    categorical_features=categorical,
    train_size=0.90,
    preprocess=True,
    numeric_imputation="median",
    categorical_imputation="mode",
    max_encoding_ohe=10,
    normalize=True,
    normalize_method="robust",
    remove_multicollinearity=False,
    remove_outliers=False,
    fix_imbalance=False,
    pca=False,
    feature_selection=False,
    fold_strategy=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=73000
    ),
    session_id=2026,
    n_jobs=-1,
    html=False,
    verbose=False
)
```

|Parametre|Verilen değer|Neden kullanıldı?|
|---|---|---|
|Veri seti|322 özellikli azaltılmış veri|İlk 29 özellik elemesinden sonraki ana hattı taramak için|
|`target`|`Label`|Tahmin edilmek istenen sınıf|
|`ignore_features`|`Variant_ID`|Kimlik sütununun biyolojik tahmin özelliği olmaması|
|`categorical_features`|`CAT_*` ve `AA_*`|Bu sütunların sayısal değil kategorik olarak işlenmesi|
|`train_size`|`0.90`|Küçük veri setinde tarama eğitimi için daha fazla satır bırakmak|
|`preprocess`|`True`|Eksik değer, kodlama ve ölçekleme işlemlerini pipeline içinde yapmak|
|Sayısal eksik tamamlama|Medyan|Ortalamaya göre aykırı değerlerden daha az etkilenmesi|
|Kategorik eksik tamamlama|Mod / en sık değer|Eksik kategorileri eğitim verisindeki en yaygın değerle tamamlamak|
|`max_encoding_ohe`|`10`|Düşük kardinaliteli kategorileri sıralama varsayımı oluşturmadan one-hot kodlamak|
|Normalizasyon|Robust|Medyan ve IQR kullandığı için aykırı değerlere daha dayanıklı olmak|
|Multicollinearity removal|Kapalı|Korelasyon elemesini ayrı ve kontrollü deneylerle yapmak|
|Outlier removal|Kapalı|Doğrulanmış hata olmadan biyolojik gözlemleri otomatik silmemek|
|`fix_imbalance`|Kapalı|111 satırlık küçük veride sentetik/otomatik dengelemenin ek yanlılık oluşturmaması|
|PCA|Kapalı|Özellik isimlerini ve yorumlanabilirliği korumak; PCA'yı bu aşamanın dışında tutmak|
|Feature selection|Kapalı|IG tabanlı özellik seçimi daha önce ayrı olarak yapıldığı için|
|CV|5-fold stratified|Her fold'da benign/patojenik oranını mümkün olduğunca korumak|
|`shuffle`|`True`|Satır sırasından kaynaklanan sistematik etkiyi azaltmak|
|`random_state`|`73000`|Aynı fold'ları tekrar üretebilmek|
|`session_id`|`2026`|PyCaret işlemlerini tekrarlanabilir yapmak|
|`n_jobs`|`-1`|Mevcut işlemci çekirdeklerini paralel kullanmak|

### 18.2. Taranan modeller ve sıralama kuralı

```python
requested = [
    "lr", "knn", "nb", "dt", "rf", "et",
    "ada", "gbc", "lda", "qda", "lightgbm", "rbfsvm"
]

exp.compare_models(
    include=included,
    fold=folds,
    sort="MCC",
    n_select=6,
    turbo=False,
    errors="ignore"
)
```

Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, LDA, QDA, LightGBM ve RBF-SVM tarandı. Modeller accuracy yerine **MCC** ile sıralandı; çünkü eğitim sınıfları dengesizdir ve MCC her iki sınıftaki doğru/yanlış kararları birlikte değerlendirir. `turbo=False`, PyCaret'ın daha yavaş modelleri de taramaya dahil edebilmesi için kullanıldı. İlk altı model sonraki değerlendirme için raporlandı.

İlk taramadaki en yüksek MCC değerleri:

|Model|PyCaret tarama MCC|
|---|---:|
|AdaBoost|0.4862|
|Extra Trees|0.4086|
|Decision Tree|0.3391|
|LDA|0.3079|
|Random Forest|0.2924|
|Naive Bayes|0.2615|

Bu değerler nihai sonuç kabul edilmedi. PyCaret taramasında standart sınıf kararları kullanıldığı ve model seçimi ile değerlendirme tam olarak nested biçimde ayrılmadığı için yalnızca **aday belirleme sonucu** olarak yorumlandı.

### 18.3. Finalistlerin manuel nested CV parametreleri

Taramada öne çıkan ve sınıf olasılığı üretebilen AdaBoost ile Extra Trees, ID3 ve CatBoost ile aynı dış test satırlarında yeniden değerlendirildi.

AdaBoost:

```python
AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=seed
)
```

- `n_estimators=50`: Varsayılan, kontrollü başlangıç karmaşıklığı.
- `learning_rate=1.0`: Her zayıf öğrenicinin standart katkısı.
- `random_state`: Sonuçların tekrar üretilebilmesi.

Extra Trees:

```python
ExtraTreesClassifier(
    n_estimators=100,
    random_state=seed,
    n_jobs=-1
)
```

- `n_estimators=100`: Tek bir ağacın oynaklığını azaltacak bir ağaç topluluğu oluşturmak.
- `random_state`: Aynı rastgele yapıyı tekrar üretebilmek.
- `n_jobs=-1`: Ağaçları paralel eğitmek.

Bu aşamada model hiperparametreleri ayrıca optimize edilmemiştir. Amaç, PyCaret'ın bulduğu algoritma türlerinin standart ve karşılaştırılabilir ayarlarla mevcut ID3'ü geçip geçmediğini sınamaktır. Aynı anda hem algoritmayı hem çok sayıda hiperparametreyi seçmek, yalnız 111 satır bulunan veri setinde seçim yanlılığı riskini artırabilirdi.

### 18.4. Finalist nested CV ve eşik seçimi

- Dış değerlendirme: 5-fold stratified CV × 5 tekrar
- İç eşik seçimi: 4-fold stratified CV
- Olasılık eşikleri: `0.05–0.95`, adım `0.01`
- Birinci seçim ölçütü: 100 benign + 20 patojenik dağılımda projeksiyon F1
- Eşitlikte ikinci ölçüt: daha az FP
- Sonraki eşitlik ölçütü: daha yüksek recall
- Sayısal preprocessing: medyan tamamlama + RobustScaler
- Kategorik preprocessing: en sık değerle tamamlama + bilinmeyen kategoriyi kabul eden one-hot encoding

Ön işleme parametreleri ve karar eşiği yalnız dış eğitim fold'unda öğrenildi. Dış test fold'u model veya eşik seçimine katılmadı.

|Model|Ortalama eşik|Specificity|Recall|Macro-F1|MCC|Proj. F1|FP/100 benign|
|---|---:|---:|---:|---:|---:|---:|---:|
|ID3 Hard Voting|15.48/21 oy|0.8857|0.7022|0.6863|0.4684|0.6177|11.4|
|Extra Trees|0.792|0.7619|0.7111|0.6565|0.3820|0.4902|23.8|
|AdaBoost|0.694|0.7905|0.5178|0.5354|0.2423|0.4037|21.0|

### 18.5. PyCaret aşamasının sonucu

PyCaret, doğrudan nihai modeli seçmek için değil, farklı algoritma aileleri arasından aday bulmak için kullanıldı. Extra Trees ve AdaBoost ilk taramada güçlü görünse de benign ağırlıklı nested CV'de 322 özellikli ID3'ü geçemedi. Extra Trees recall'da ID3'ten yalnız `0.0089` daha yüksekken 100 benign başına yaklaşık `12.4` daha fazla FP oluşturdu. AdaBoost hem MCC hem recall hem de projeksiyon F1 bakımından geride kaldı.

Bu nedenle PyCaret sonucunda tek model kararı değiştirilmedi: **322 özellikli özel ID3 Hard Voting ana tek-model adayı olarak korundu**. Extra Trees ve AdaBoost yalnız farklı hata örüntüleri sağlayabilecek ensemble adayları olarak saklandı.

## 19. Güncel nihai karar – 319 özellikli yeniden model seçimi

Son aşamada bütün modellerin aynı kolon sayısını kullanması şartıyla 320 özellikli yapı değerlendirilmiş, ardından `%98.2` eksik ve tek benzersiz dolu değere sahip `CAT_6` çıkarılarak 319 özellikli Senaryo 1D oluşturulmuştur.

319 özellik üzerinde ID3, CatBoost, Extra Trees ve AdaBoost'un 33 ensemble kombinasyonu yeniden taranmıştır. Güncel kazanan:

```text
ID3 + CatBoost + AdaBoost
Eşit ağırlıklı Soft Voting
319 ortak özellik
```

|Ölçüt|Skor|
|---|---:|
|Accuracy|0.7477|
|Specificity|0.9143|
|Recall|0.7089|
|Macro-F1|0.6992|
|MCC|0.4964|
|Projeksiyon F1|0.6633|
|FP/100 benign|8.6|

Bu karar önceki 320 özellikli `ID3 + CatBoost + Extra Trees ağırlıklı soft voting` kararının yerini almıştır. Ancak veri yalnız 111 satır olduğundan bağımsız test doğrulaması hâlâ gereklidir.
