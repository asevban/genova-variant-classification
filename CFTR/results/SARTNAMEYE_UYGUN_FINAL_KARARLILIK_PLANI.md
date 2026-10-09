# Şartnameye Uygun Final Kararlılık Planı

## Şartnameden alınan bağlayıcı çerçeve

- Üniversite ve üzeri seviyesinde temel sıralama metriği, pozitif sınıf için `TP`, `FP` ve `FN` üzerinden hesaplanan **F1 skorudur**.
- **MCC**, asimetrik sınıf dağılımındaki genel başarıyı değerlendiren destekleyici metriktir; ana seçim metriği değildir.
- CFTR final test setinin yaklaşık dağılımı **20 patojenik + 100 benign** olacaktır.
- Final test seti **etiketsiz** verilecektir. Model ve eşik seçimi yalnız eğitim verisindeki iç validasyon sonuçlarıyla dondurulmalıdır.
- Varyantların gerçek etiketini harici kaynaklardan bulmaya çalışmak yasaktır. Yalnız yarışma komitesinin verdiği varyant profilleri kullanılmalıdır.
- Kod, ön işleme, model eğitimi ve değerlendirme süreci çalışabilir, yeniden üretilebilir ve açıkça dokümante edilmiş olmalıdır.
- Jüri kodu tekrar çalıştırıp beyan edilen sonuçları yeniden üretmeyi isteyebilir.

## Projedeki ölçüm önceliği

Model seçimi aşağıdaki sırayla yapılacaktır:

1. CFTR final dağılımına göre tahmin edilen pozitif sınıf F1
2. F1 güven aralığı ve tekrarlar arasındaki kararlılık
3. FP/100 benign ve FN dengesi
4. MCC
5. Model sadeliği ve yeniden üretilebilirlik

`Macro-F1` raporlanabilir; ancak şartnamedeki üniversite finalinin temel seçim metriği olarak kullanılmayacaktır.

## Uygun kararlılık kontrolleri

### 1. Variant_ID kümeli bootstrap güven aralığı — Uygun ve uygulandı

Aynı varyanta ait tahminlerin bağımsız örnekler gibi sayılmasını önlemek için yeniden örnekleme `Variant_ID` düzeyinde yapılmıştır. Ana modelin yedek modele göre gözlenen F1 üstünlüğü vardır; fakat yüzde 95 güven aralığı sıfırı içerdiğinden üstünlük kesinleşmemiştir. Bu sonuç, final modelini yalnız tek ortalama skora göre seçmememiz gerektiğini göstermektedir.

### 2. Fold ve tekrar kırılganlığı — Uygun ve uygulandı

Nested çapraz doğrulama sonuçları fold ve tekrar düzeyinde incelenmiştir. Tek bir fold veya varyantın sonucu sürükleyip sürüklemediği ayrıca leave-one-variant-out analiziyle kontrol edilmiştir. `VAR_001930`, ana ve yedek model sıralamasını değiştirebilen etkili örnek olarak bulunmuştur; etiketi veya veri kalitesi hakkında bağımsız kanıt olmadan çıkarılmayacaktır.

### 3. Eşik hassasiyeti — Uygun ve uygulandı

Eşik yalnız iç eğitim fold'larında, yaklaşık `100 benign + 20 patojenik` final dağılımına göre seçilmiştir. Kararlılık cezalı eşik denemeleri temel yöntemi geçemediği için reddedilmiştir. Final test etiketine bakarak eşik değiştirilmeyecektir.

### 4. Ensemble ablation — Uygun ve uygulandı

Üçlü ensemble, daha sade ikili ve alternatif bileşimlerle karşılaştırılmıştır. Ana aday `ID3 + CatBoost + Random Forest` eşit soft voting modelidir. Yedek aday `ID3 + CatBoost + AdaBoost` modelidir. Bileşenlerin katkısı ayrı ayrı karşılaştırılmış, yalnız daha karmaşık olduğu için model kabul edilmemiştir.

### 5. Cross-fitted olasılık ortalaması — Uygun, güçlü aday ve uygulandı

Her varyant için, o varyantı eğitimde görmeyen beş dış-fold modelinin olasılıkları ortalanmıştır. Elde edilen sonuçlar:

- Accuracy: `0.7928`
- Specificity: `0.9524`
- Recall: `0.7556`
- Macro-F1: `0.7451`
- MCC: `0.5717`
- Final dağılımına göre tahmini pozitif F1: `0.7580`
- 100 benign başına FP: `4.76`

Bu sonuç güçlüdür; fakat standart pooled nested-CV sonucu ile birebir aynı değerlendirme düzeni değildir. Final modele alınmadan önce aynı tahmin tablosu üzerinde kümeli bootstrap, model sayısı azaltma ve eşik çevresi kararlılığı ile doğrulanacaktır.

### 6. Özellik kararlılığı — Uygun ve uygulandı

Özelliklerin fold'lar boyunca önem sıralamasında ne kadar kararlı olduğu ölçülmüştür. Bu analiz yalnız denetim amacı taşımaktadır. Sırf bir özellik seyrek seçildi diye yeni bir kolon elemesi yapılmayacaktır; çünkü kullanıcı yeni özellik eklenmesini veya veri setinin yeniden değiştirilmesini istememiştir ve nihai veri seti 319 özellikte dondurulmuştur.

## Uygulanmayacak işlemler

- Final test etiketleriyle eşik, kolon veya model seçmek
- Harici veri tabanlarından varyantın gerçek sınıfını bulmak veya etiketi düzeltmek
- TabPFN'yi nihai ensemble'a eklemek
- Yeni özellik üretmek
- Yalnız tek CV skorunu yükselttiği için varyant silmek
- Ana değerlendirmeyi accuracy veya Macro-F1'e göre yapmak

## Sıradaki güvenli doğrulama sırası

1. Cross-fitted bagging sonucuna `Variant_ID` kümeli bootstrap güven aralığı uygulamak.
2. Beş fold modelinden biri çıkarıldığında skorların ne kadar değiştiğini ölçmek.
3. Olasılık ortalamasında 3, 4 ve 5 fold-model kullanımını karşılaştırmak.
4. Seçilen eşiğin yakın çevresinde F1, FP ve FN değişimini ölçmek.
5. Ana ensemble bileşen ağırlıklarını küçük bir aralıkta değiştirerek sonucun kırılganlığını kontrol etmek.
6. Son yapılandırmayı, kolon listesini, rastgele tohumları ve tahmin formatını dondurmak.

Bu işlemlerin tamamı yalnız eğitim verisi ve out-of-fold tahminlerle yapılmalıdır. Final test seti geldiğinde yalnız dondurulmuş ön işleme ve model tahmin adımları çalıştırılmalıdır.

## Yeni doğrulama sonuçları

### Cross-fitted bagging bootstrap

100.000 tekrarlı, sınıf-korumalı `Variant_ID` bootstrap sonucunda olasılık ortalaması için:

- Tahmini final F1: `0.7580` — %95 GA: `0.5864–0.9024`
- MCC: `0.5717` — %95 GA: `0.4550–0.6923`
- Recall: `0.7556` — %95 GA: `0.6667–0.8444`
- FP/100 benign: `4.76` — %95 GA: `0.00–14.29`

Nokta tahmini güçlüdür; ancak yalnız 21 benign eğitim örneği bulunduğu için FP güven aralığı geniştir.

### Bir cross-fitted modeli eksiltme

Beş tekrar tahmininden biri sırayla çıkarıldığında dört modelli yapıların tahmini final F1 değeri `0.6771–0.7648`, MCC değeri `0.5358–0.5835` ve FP/100 benign değeri `4.76–9.52` arasında değişmiştir. Dördüncü tekrar çıkarıldığında ikinci bir benign örnek FP'ye dönmüş ve F1 `0.6771` olmuştur. Dolayısıyla başarı tamamen tek modele bağlı değildir; fakat karar sınırına yakın en az bir benign varyant nedeniyle orta düzeyde kırılganlık vardır. Beş modelin tamamını kullanmak dört modelli alt yapılardan daha güvenli kabul edilmiştir.

### Model sayısı kararlılığı

Tek bir iyi alt küme seçilmeden bütün kombinasyonlar karşılaştırılmıştır. Üç modelli yapıların ortalama tahmini final F1'i `0.7037` (SS `0.0600`), dört modelli yapıların `0.7404` (SS `0.0362`), beş modelli yapının `0.7580` olmuştur. FP/100 benign ortalaması sırasıyla `8.10`, `5.71` ve `4.76` bulunmuştur. Bu nedenle beş model korunmuştur.

### Eşik hassasiyeti

Varsayılan eşik farkı `0.00` için tahmini final F1 `0.7580`, MCC `0.5717`, FP/100 benign `4.76` bulunmuştur. Eşiğin ±`0.02` yakın çevresinde F1 `0.6966–0.7580`, FP/100 benign `4.76–9.52` aralığında değişmiştir. Özellikle eşiği düşürmek ikinci bir benign örneği FP'ye çevirebildiğinden model eşik açısından tamamen düz bir plato göstermemektedir. Tarama yalnız hassasiyet denetimidir; en iyi görünen ofset seçilmemiş ve iç CV ile öğrenilen varsayılan eşik korunmuştur.

### Ensemble ağırlık kararlılığı

Her dış fold için ağırlıklar yalnız iç-CV tahminlerinden seçilmiştir. Nested ağırlık seçimi cross-fitted değerlendirmede F1 `0.6570`, MCC `0.5020`, FP/100 benign `9.52` üretmiş; eşit ağırlıklı üçlü yapının `0.7580`, `0.5717`, `4.76` sonuçlarını geçememiştir. Fold'larda seçilen ağırlıkların belirgin biçimde değişmesi de sabit ve güvenilir bir ağırlık deseni bulunmadığını göstermiştir. Ağırlık optimizasyonu reddedilmiş, eşit ağırlık korunmuştur.

### Model bileşeni ablation ve ID3 karşılaştırması

Tekil, ikili ve üçlü yapıların eşikleri yalnız iç-CV tahminlerinden öğrenilmiştir. Cross-fitted tek-karar değerlendirmesinde ID3, `0 FP` nedeniyle tahmini F1 `0.7838` üretmiş; ancak recall `0.6444` ve MCC `0.5053` olmuştur. Üçlü ensemble tahmini F1 `0.7580`, recall `0.7556`, MCC `0.5717` ve FP/100 benign `4.76` üretmiştir.

Paired bootstrap'ta ID3'ün F1 farkı `+0.0258` olsa da %95 güven aralığı `-0.1179–0.1963` ve üstünlük olasılığı yalnız `%57.9` bulunmuştur. Üçlü modelin recall üstünlüğü ise güven aralığıyla doğrulanmıştır: ID3−üçlü recall farkı `-0.1111`, %95 GA `-0.1889–-0.0444`. Varyant düzeyinde yalnız ID3'ün doğru bildiği 2, yalnız üçlü modelin doğru bildiği 11 örnek vardır. Bu nedenle ID3'ün 21 benign örnekte gözlenen sıfır FP sonucu kesin üstünlük sayılmamış; daha dengeli ve tamamlayıcı üçlü ensemble ana aday olarak korunmuştur. ID3 düşük-FP yedek adayıdır.
