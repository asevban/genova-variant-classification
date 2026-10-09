# KANSER Paneli — Benign Ağırlıklı Final Modeli

**Rapor tarihi:** 24 Ağustos 2026  
**Final için doğrulanan sınıf dağılımı:** 3.000 benign / 500 patojenik  
**Pozitif sınıf:** `1 = patojenik`  
**Negatif sınıf:** `0 = benign`

## 1. Kısa sonuç

Final için seçilen model tek bir algoritma değil, iki farklı modelin **soft-voting** birleşimidir:

- **%75 CatBoost:** Information Gain / mutual information ile seçilmiş 20 özellik;
- **%25 Balanced Random Forest:** mRMR ile seçilmiş 40 özellik;
- **kilitli karar eşiği:** `0,486465580331`;
- **kullanılan eğitim satırı:** 388;
- **silinen eğitim satırı:** 0.

Her iki bileşen de `source_free_robust85` veri ön işleme politikasını kullanır. Bu politika veri kaynağını ezberleme riski taşıyan `CAT_1` ve `CAT_2` sütunlarını çıkarır; tamamen boş, sabit ve birebir tekrarlı sütunları eğitim katında öğrenerek temizler; semantik eksikliği en az %85 olan ham sütunları çıkarır; fakat bilgi taşıyan farklı eksiklik örüntülerini tekilleştirilmiş eksiklik göstergeleri olarak korur.

Seçilen modelin **10 tekrar × 5 dış katlı nested-CV** sonucunun, 3.000/500 final dağılımına yansıtılmış ortalaması şöyledir:

| Ölçüt | Ortalama | Tekrarlar arası std. |
|---|---:|---:|
| Hedef dağılım F1 | **0,6220** | 0,0284 |
| Hedef dağılım MCC | **0,5563** | 0,0345 |
| Precision | **0,5838** | 0,0514 |
| Patojenik duyarlılığı | **%66,87** | %2,12 |
| Benign özgüllüğü | **%91,92** | %1,71 |
| Hedef dağılım doğruluk | **%88,34** | %1,38 |
| ROC-AUC | **0,8974** | 0,0070 |
| 3.000/500'e uyarlanmış AUPRC | **0,5637** | 0,0310 |

Bu değerler **gerçek final test skoru değildir**. Etiketli 388 eğitim satırındaki sınıf-içi hata oranlarının finalde de benzer kalacağı varsayımıyla yapılan nested-CV projeksiyonudur. Gerçek 3.500 satırlık final dosyasının özellik dağılımı değişirse sonuç da değişebilir.

## 2. Neden eski model doğrudan kullanılmadı?

Eğitim verisinde 120 benign ve 268 patojenik örnek vardır; patojenik oranı yaklaşık %69,07'dir. Finalde ise patojenik oranı yalnızca %14,29 olacaktır. Modelin duyarlılık ve özgüllüğü aynı kalsa bile bu değişim precision ve F1'i ciddi biçimde değiştirir. Bu nedenle eğitim dağılımındaki F1'i en yüksek model, final dağılımında en yüksek F1'i veren model olmak zorunda değildir.

F1 için optimum karar eşiğinin taban orana ve skorların dağılımına bağlı olması literatürde gösterilmiştir. Bu yüzden `0,50` varsayılan eşiğini kullanmak yerine eşik sadece iç çapraz doğrulama tahminlerinde arandı ve final test etiketleri görülmeden kilitlendi. Ayrıca dengesiz veri değerlendirmesinde ROC tek başına yeterli olmadığı için hedef prevalansa uyarlanmış precision, F1 ve PR-AUC birlikte izlendi.

## 3. Veri ve sızıntı kontrolleri

### 3.1 Kullanılan veri

- Ham CSV: `YARISMA_TRAIN_KANSER.csv`
- Satır: 388
- Toplam sütun: 353
- Model adayı ham özellik: 351
- Benign: 120
- Patojenik: 268
- Ham dosya SHA-256: `bc695838a7e6b77413bfa7c0725a3907b15edc6a2f0cb0da3ae9bdc8f5a905cb`

### 3.2 Aynı satırların katlar arasında sızması engellendi

Veride 386 benzersiz satır grubu vardır. Bir grupta birbirinin aynısı olan 3 satır bulundu. Bu satırlar rastgele farklı katlara ayrılsaydı model validasyon örneğini eğitimde görmüş olabilirdi. Bu nedenle `StratifiedGroupKFold` mantığı kullanıldı ve aynı satır grubundaki örneklerin tamamı hep aynı tarafta tutuldu.

### 3.3 Ön işleme ve özellik seçimi kat içinde öğrenildi

Aşağıdakilerin hiçbiri CV başlamadan bütün veride öğrenilip dış katlara taşınmadı:

- hangi sütunun tamamen boş, sabit veya tekrar olduğu;
- hangi sütunun eksiklik oranının %85'i geçtiği;
- medyan doldurma değerleri;
- kategorik kodlama sözlüğü;
- Information Gain, ANOVA F-score ve mRMR sıralamaları;
- sınıf ağırlıkları;
- karar eşiği.

Her dış katta bu kararlar sadece dış eğitim bölümünde, eşik ise onun içindeki 4 katlı OOF tahminlerinde öğrenildi.

## 4. Taranan yöntemler

“Bütün modeller” ifadesi matematiksel olarak sonsuz bir uzaydır. Bu çalışmada küçük, geniş ve eksik değerli tabular veri için makul olan **12 model ailesi** ve bunların kontrollü varyantları tarandı:

1. Logistic Regression
2. Linear SVM
3. RBF SVM
4. Random Forest
5. Extra Trees
6. HistGradientBoosting
7. LightGBM
8. XGBoost
9. CatBoost
10. Linear Discriminant Analysis
11. Gaussian Naive Bayes
12. Balanced Random Forest

İlk geniş taramada **408 model konfigürasyonu**, ikinci politika ve örnekleme aktarımında **162 konfigürasyon**, ensemble taramasında **195 birleşim** değerlendirildi. En iyi ve birbirinden farklı 12 finalist tekrarlı nested-CV'ye alındı.

### 4.1 Özellik seçme yöntemleri

- Tüm özellikler
- ANOVA F-score
- Mutual information / Information Gain
- mRMR: hedefle ilişkili fakat birbirinin tekrarı olmayan özellikleri öne çıkarır
- 20, 40, 80 ve daha geniş özellik kümeleri

### 4.2 Sınıf dengesizliği yöntemleri

- Ağırlıksız eğitim
- Dengeli sınıf ağırlığı
- Benign için 4 kat ve 8 kat maliyet
- Final 3.000/500 dağılımına göre hesaplanan hedef-prior ağırlığı
- Random oversampling: eşit ve hedef oranlı
- SMOTE
- Balanced Random Forest'ın sınıf dengeli örneklemesi

SMOTE ve rastgele aşırı örnekleme yalnızca eğitim katında uygulandı. Dış validasyon satırları hiçbir zaman çoğaltılmadı veya sentetik veri üretiminde kullanılmadı.

### 4.3 Birleştirme ve kalibrasyon yöntemleri

- Soft voting
- Hard voting: AND ve OR kararları
- Olasılık çarpımı
- Platt-kalibrasyonlu soft voting
- Lojistik stacking
- Bilinen final öncülüne Bayes / label-shift olasılık düzeltmesi

Kalibrasyon ve label-shift düzeltmesi teorik olarak anlamlı olmasına rağmen bu küçük veri üzerinde sabit `0,50` eşikli Platt + prior düzeltmesi F1'i `0,4335`'e düşürdü. En iyi hard-voting sonucu `0,6181` oldu. Bu yüzden ikisi de final modele alınmadı.

## 5. Ön işleme politikalarının karşılaştırılması

Aşağıdaki tablo ilk eleme taramasındaki en iyi model sonucunu gösterir. Bunlar finalist nested-CV sonucu değildir ve aday seçimi amacıyla kullanılmıştır.

| Ön işleme | En iyi tarama F1 | Kısa yorum |
|---|---:|---|
| `source_free_robust85` | **0,6712** | Kaynak sütunları yok, yüksek eksik ham sütunlar temiz, eksiklik örüntüsü korunuyor |
| `source_aware_info` | 0,6680 | `CAT_1/2` korunuyor; kaynak ezberleme riski nedeniyle tercih edilmedi |
| `a5_info_preserving` | 0,6645 | Yüksek eksik ham özellikleri daha fazla koruyor |
| `source_free_compact` | 0,6594 | Kaynak sütunları yok, daha az agresif temizlik |
| `a4_corrected` | 0,6470 | Önceki A4 mantığının düzeltilmiş karşılaştırıcısı |

`source_free_robust85`, hem güçlü tarama sonucu verdiği hem de kaynağa bağlı kestirme yol riskini azalttığı için nihai iki bileşende de kullanıldı.

## 6. Nihai modelin yapısı

### 6.1 CatBoost bileşeni — %75

- Ön işleme: `source_free_robust85`
- Özellik seçimi: Information Gain / mutual information
- Özellik sayısı: 20
- Sınıf ağırlığı: 3.000/500 hedef oranına göre benign hatasını daha pahalı yapan ağırlık
- Ağaç derinliği: 6
- Tur: 220
- Öğrenme oranı: 0,04
- `l2_leaf_reg`: 20
- `random_strength`: 4

### 6.2 Balanced Random Forest bileşeni — %25

- Ön işleme: `source_free_robust85`
- Özellik seçimi: mRMR
- Özellik sayısı: 40
- Ağaç sayısı: 300
- Maksimum derinlik: 8
- Minimum yaprak örneği: 2
- `max_features`: `sqrt`
- Sınıf dengeli bootstrap örneklemesi

CatBoost'un 20 özelliğinin tamamı Random Forest'ın 40 özellikli kümesi içinde yer alır. İkinci model ek 20 özellik ve farklı ağaç kurma mantığıyla aynı örneklerde farklı hatalar yapabildiği için birleşim tek CatBoost'tan daha iyi sonuç verdi.

### 6.3 Soft voting ve eşik

İki modelin patojenik skorları aşağıdaki gibi birleştirilir:

`birleşik_skor = 0,75 × CatBoost_skoru + 0,25 × BalancedRF_skoru`

Birleşik skor `0,486465580331` veya üzerindeyse sonuç patojenik (`1`), altındaysa benign (`0`) olarak üretilir. Bu eşik, tekrarlı OOF skorlarıyla öğrenildi ve model dosyasının içine kaydedildi. Final test sonucu görüldükten sonra değiştirilmemelidir.

## 7. Finalistlerin tekrarlı nested-CV karşılaştırması

| Sıra | Model | F1 ort. ± std. | MCC ort. | Özgüllük | Duyarlılık |
|---:|---|---:|---:|---:|---:|
| 1 | **%75 CatBoost + %25 Balanced RF** | **0,6220 ± 0,0284** | **0,5563** | **%91,92** | **%66,87** |
| 2 | %50 CatBoost + %50 LightGBM | 0,6199 ± 0,0198 | 0,5541 | %92,25 | %65,78 |
| 3 | %50 CatBoost + %50 XGBoost | 0,6171 ± 0,0246 | 0,5504 | %91,58 | %66,98 |
| 4 | En iyi Platt soft birleşimi | 0,6194 ± 0,0351 | 0,5533 | %91,75 | %66,83 |
| 5 | En iyi hard-voting / AND | 0,6181 ± 0,0197 | 0,5513 | %91,67 | %67,01 |

Birinci ve ikinci model arasındaki F1 farkı yalnızca `0,0021`'dir ve veri küçüktür. Bu nedenle “birinci model istatistiksel olarak kesin üstündür” denemez. Önceden tanımlanan seçim kuralı — önce hedef F1, sonra MCC ve hedef AUPRC — birinci modeli seçti. Daha düşük tekrar oynaklığı istenirse CatBoost + LightGBM birleşimi mantıklı yedektir; yarışma için kilitlenen ana seçim yine birinci modeldir.

## 8. Önceki benign-ağırlıklı modelle adil karşılaştırma

Önceki LightGBM sonucu daha önce tek bir nested-CV düzeninde `0,6231` olarak raporlanmıştı. Fakat yeni çalışmayla doğrudan karşılaştırılabilmesi için o model de **aynı 10 tekrar × 5 grup-güvenli dış kat** üzerinde yeniden ölçüldü.

| Ölçüt | Önceki LightGBM | Yeni ensemble | Değişim |
|---|---:|---:|---:|
| Hedef F1 | 0,6023 | **0,6220** | **+0,0197** |
| Hedef MCC | 0,5342 | **0,5563** | **+0,0221** |
| Precision | 0,5264 | **0,5838** | **+0,0574** |
| Patojenik duyarlılığı | **%70,93** | %66,87 | -%4,07 puan |
| Benign özgüllüğü | %89,25 | **%91,92** | **+%2,67 puan** |

Yeni model benign-ağırlıklı finalde daha az yanlış alarm üretmek için bir miktar patojenik duyarlılığından vazgeçmektedir. Bu, sınıf dağılımına ve yarışmanın F1 ölçütüne göre bilinçli bir değiş tokuştur; klinik kullanım kararı olarak yorumlanmamalıdır.

## 9. 3.000 benign / 500 patojenik için beklenen hata matrisi

Nested-CV'deki ortalama duyarlılık ve özgüllük final dağılımında aynı kalırsa yaklaşık sonuç:

| Gerçek / Tahmin | Benign | Patojenik |
|---|---:|---:|
| Gerçek benign | **2.758 doğru** | **243 yanlış pozitif** |
| Gerçek patojenik | **166 yanlış negatif** | **334 doğru** |

Ondalıklı ham projeksiyon `TN=2757,5`, `FP=242,5`, `FN=165,67`, `TP=334,33` değeridir. Tablodaki tam sayılar yalnızca okunabilirlik için yuvarlanmıştır. Gerçek final hata matrisi değildir.

Önceki LightGBM'e göre yaklaşık 80 daha az benign örneğin yanlışlıkla patojenik işaretlenmesi; buna karşılık yaklaşık 20 daha fazla patojenik örneğin kaçırılması beklenir.

## 10. Literatür taramasının çalışmaya etkisi

1. **F1 ve eşik:** Lipton, Elkan ve Naryanaswamy, F1-optimal kararın taban orana ve skor dağılımına bağlı olduğunu gösterir. Bu nedenle eşik iç OOF tahminlerinde, doğrudan 3.000/500 hedef dağılım F1'i için seçildi.  
   Kaynak: [Optimal Thresholding of Classifiers to Maximize F1 Measure](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/)

2. **PR-AUC kullanımı:** Saito ve Rehmsmeier, dengesiz veri değerlendirmesinde Precision–Recall eğrisinin ROC'a göre daha bilgilendirici olabildiğini gösterir. Bu yüzden ROC-AUC yanında hedef prevalansa uyarlanmış AUPRC raporlandı.  
   Kaynak: [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

3. **Label shift / prior değişimi:** Saerens ve arkadaşlarının öncül olasılık düzeltmesi ile Lipton ve arkadaşlarının label-shift çalışması incelendi. Olasılık düzeltmesi deneysel olarak sınandı fakat bu veri üzerinde seçilen eşik yöntemini geçemedi. Projeksiyonun geçerli olması için temel varsayım, sınıf oranı değişirken sınıf içi özellik dağılımının yaklaşık sabit kalmasıdır.  
   Kaynaklar: [Adjusting the Outputs of a Classifier to New a Priori Probabilities](https://pubmed.ncbi.nlm.nih.gov/11747533/), [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html), [Maximum Likelihood with Bias-Corrected Calibration Is Hard-To-Beat at Label Shift Adaptation](https://proceedings.mlr.press/v119/alexandari20a.html)

4. **Model seçimi yanlılığı:** Cawley ve Talbot, sonlu veri üzerinde hiperparametre ve model seçiminin CV sonucuna aşırı uyum sağlayabileceğini vurgular. Bu nedenle finalistler 10 tekrarlı dış değerlendirme ve yalnız eğitim kısmında 4 katlı iç eşik seçimiyle ölçüldü.  
   Kaynak: [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html)

5. **SMOTE:** Chawla ve arkadaşlarının sentetik azınlık örneği üretme yöntemi sadece eğitim katında uygulandı. Bu veri setinde seçilen ağırlıklandırma + Balanced Random Forest yaklaşımını geçmediği için final modele alınmadı.  
   Kaynak: [SMOTE: Synthetic Minority Over-sampling Technique](https://www.jair.org/index.php/jair/article/view/10302)

6. **CatBoost ve diğer boosting aileleri:** CatBoost'un ordered boosting yaklaşımı ile LightGBM ve XGBoost'un ağaç tabanlı boosting yöntemleri aday havuzuna alındı. Son karar makalelerdeki genel üstünlük iddiasına göre değil, aynı fold'larda ölçülen kanser verisi sonucuna göre verildi.  
   Kaynaklar: [CatBoost: Unbiased Boosting with Categorical Features](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

## 11. Sonucun nasıl yorumlanması gerekir?

- `0,6220`, gerçek final F1'i değil, en dürüst eldeki hedef-prior tahminidir.
- Tek bir şanslı bölmenin sonucu değildir; 10 farklı 5-kat bölmesinin ortalamasıdır.
- Buna rağmen 388 satır küçüktür ve aday kısa listesi bu aynı eğitim verisindeki ön taramadan çıkmıştır. Dolayısıyla sonuçta hâlâ model-seçimi iyimserliği olabilir.
- Birinci ve ikinci finalist arasındaki fark küçük olduğu için kesin algoritmik üstünlük iddia edilmemelidir.
- Final verisinde yalnız sınıf oranı değil, veri kaynağı veya ölçüm süreci de değişirse label-shift varsayımı bozulur.
- Nihai ve bağımsız ölçüm, organizatörün hiç görülmemiş 3.500 satırlık final testidir.

## 12. Teslim edilen dosyalar

- `KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib`: ön işleme, iki bileşen ve kilitli eşiği içeren model
- `predict_final_model.py`: ham final CSV'den skor ve sınıf tahmini üretir
- `RUNTIME_VERSIONS.txt`: doğrulanan paket sürümleri
- `RESULTS.json`: seçilen modelin makine-okunur özeti ve SHA-256 değerleri
- `04_repeated_nested_summary.csv`: finalistlerin tekrarlı nested-CV özeti
- `05_repeated_nested_by_repeat.csv`: tekrar bazlı sonuçlar
- `06_nested_fold_and_selection_log.csv`: dış ve iç kat/eşik kayıtları
- `07_prevalence_stress_test.csv`: farklı final prevalans senaryoları
- `09_final_selected_features.csv`: iki bileşenin seçtiği özellikler
- `12_targeted_ensemble_summary.csv`: hard voting, kalibrasyon ve stacking karşılaştırmaları
- `15_previous_models_fair_summary.csv`: önceki LightGBM'in aynı protokoldeki adil karşılaştırması

Model dosyası SHA-256:

`5f5aa34252080b8905bf0116a72d316266798ea9d8a114e7c37211be1a18bd33`

## 13. Tahmin üretme

Final test CSV'si ham eğitim CSV'siyle aynı özellik adlarına sahip olmalıdır. `Label` sütunu zorunlu değildir ve final testinde bulunmaması beklenir.

```powershell
python predict_final_model.py --input final_test.csv --output final_predictions.csv
```

Çıktıdaki temel alanlar:

- `pathogenic_score`: birleşik model skoru;
- `prediction`: `0 = benign`, `1 = patojenik`;
- `prediction_name`: okunabilir sınıf adı.

Model taze bir süreçte yeniden yüklenerek 388 satır üzerinde test edildi; bütün skorlar sonlu bulundu ve tahmin betiği başarıyla çalıştı.

## 14. 31 Ağustos ikinci geçiş: amino-asit zenginleştirmesi

Benzer varyant-patojenisite çalışmalarında REVEL gibi yöntemler farklı fonksiyonel ve evrimsel kanıtları ensemble içinde birleştirir; CADD ve RENOVO da çoklu anotasyonları makine öğrenmesiyle bir araya getirir. Bizim CSV'de bu dış anotasyonların ham değerleri bulunmadığı için dış veri ekleyip yapay skor üretmedim. Bunun yerine yalnızca mevcut `AA_1` ve `AA_2` sembollerinden şu hedef-bağımsız değişkenleri türeten yeni bir politika sınadım:

- hidrofobiklik farkı ve mutlak farkı;
- amino-asit hacmi farkı;
- polarite farkı;
- yük farkı ve yük değişimi;
- stop-gain / stop-loss ve aynı amino-asit göstergeleri;
- amino-asit çiftinin kategorik temsili.

Yeni politika 9 bileşen ve 11 finalist ile yine 10 tekrar × 5 dış katlı, 4 katlı iç OOF eşikli nested-CV'de test edildi. En iyi yeni aday, **%75 AA-zengin CatBoost + %25 önceki Balanced Random Forest** oldu:

| Ölçüt | Mevcut kilitli model | AA-zengin challenger |
|---|---:|---:|
| Hedef F1 | **0,621995** | 0,621760 |
| Hedef MCC | 0,556257 | **0,557469** |
| F1 std. | 0,028428 | **0,019802** |
| Precision | 0,583790 | **0,596491** |
| Özgüllük | 0,919167 | **0,925000** |
| Duyarlılık | **0,668657** | 0,654104 |

F1 yarışmanın birincil seçme ölçütü olduğu için ana model değiştirilmedi. Fark F1 açısından yalnızca `0,000235` ve pratik olarak belirsizlik bandının içindedir; challenger ise MCC, precision, benign özgüllüğü ve tekrarlar arası oynaklıkta daha iyi görünmektedir. Bu nedenle AA-zengin model pakette **yedek/araştırma adayı** olarak tutuldu, ana final tahmin dosyasının yerine geçirilmedi. Challenger çıktıları `outputs/GENOVA_KANSER_FINAL_BENIGN_SHIFT_V3` klasöründedir.

Bu ikinci taramanın ana sonucu şudur: mevcut veride yeni biyolojik özellikler eklemek tek başına genelleme F1'ini artırmadı. Gerçekten daha büyük bir sıçrama için finalden önce dış ve güvenilir anotasyonlar (ör. gen/protein konumu, popülasyon frekansı, CADD/REVEL/SpliceAI veya benzeri) veri sözleşmesine eklenmeli ve bunlar yeni bir dış doğrulama kohortunda sınanmalıdır. Dış anotasyonlar olmadan bu skorları tahmin etmek veri sızıntısı ve sahte performans riski oluşturur.

İncelenen benzer çalışmalar: [REVEL ensemble](https://pmc.ncbi.nlm.nih.gov/articles/PMC5065685/), [CADD](https://pmc.ncbi.nlm.nih.gov/articles/PMC3992975/) ve [RENOVO](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059374/). Bu çalışmaların ortak ve bizim panele uyarlanabilir tarafları; bağımsız kanıtları birleştirmek, varyant düzeyinde sızıntıyı önlemek ve sürekli skorları sabit bir eşikle birlikte raporlamaktır.

## 15. Son eşik ve özgüllük denetimi

Ana modelin tekrarlı OOF ortalama skorları üzerinde tüm benzersiz eşikler tarandı. Özgüllük en az %90, %92, %93 ve %94 olacak şekilde ayrı ayrı en yüksek hedef-prior F1 arandı. Dört kısıtın tamamında aynı eşik seçildi:

- eşik: `0,486465580331`;
- OOF-fit tanısı: F1 `0,6610`, MCC `0,6069`, özgüllük `%95,00`;
- bu tanı aynı OOF skorlarında eşik seçildiği için bağımsız performans değildir;
- dış nested-CV'nin daha güvenilir tahmini F1 `0,6220`, MCC `0,5563`, özgüllük `%91,92` olarak kalır.

Sonuç olarak özgüllük kısıtı koymak için eşiği tekrar değiştirmeye gerek görülmedi. Eşik denetiminin tüm sonuçları `16_threshold_specificity_audit.csv` ve `16_threshold_specificity_audit.json` dosyalarındadır.
