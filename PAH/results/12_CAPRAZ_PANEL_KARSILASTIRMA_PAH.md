# PAH Paneli — Çapraz-Panel Karşılaştırmalı Değerlendirme (CFTR / MASTER)

> **Kapsam ve kaynak notu:** Bu rapor, PAH'ın ön işleme sonrası tüm
> çalışmasını (E0-E6 + EK turları + P0-P2 + F1-F6) (1) danışman Osman
> Hoca'nın çapraz-panel rehberlik notlarına ve (2) CFTR/MASTER
> panellerinin doğrulanmış yaklaşım özetlerine göre denetler. **Kardeş
> panellerin kod/veri deposuna erişim yoktur** — CFTR/MASTER hakkındaki
> her ifade, bu göreve girdi olarak verilen doğrulanmış özetlere
> dayanır; bu özetlerin ötesindeki hiçbir şey varsayılmamıştır ve
> bilinmeyen noktalar açıkça "bilinmiyor" diye işaretlenmiştir.
>
> Bu tur **saf inceleme**dir — hiçbir kod, model veya mevcut rapor
> içeriği değiştirilmemiştir.

---

## 1. Yönetici Özeti

**Osman Hoca'nın PAH'a özel olumlu notu — "PAH panelindeki nested CV
yapısını koruyun, mevcut group-aware ve nested değerlendirme yaklaşımı
güçlü kurulmuş" — bugün hâlâ geçerlidir, ve o not verildiğinden bu yana
yapı zayıflamak yerine güçlenmiştir.** Somut kanıt:

- Sabit split bankası (`data/splits/pah/`, 10 tekrar × 5 dış × 4 iç
  fold) bir kez üretildi ve **hiçbir deneyde yeniden üretilmedi** — F1,
  F3, F4, F5, F6 turlarının tamamı boyunca dosya mtime'ı değişmedi.
  Bu, Osman Hoca'nın "aynı panelde tüm adayları aynı splitlerle
  karşılaştırın" kuralının en katı uygulaması.
- Group-aware yapı korunuyor: çelişkili-profil grubu
  (`conflict_group_1`) `StratifiedGroupKFold` ile hiçbir fold sınırını
  geçmiyor, bu bir testle (`test_split_bank_pah.py`) sürekli
  doğrulanıyor.
- Nested disiplin, o notun verildiği tarihten sonra **daha da
  sıkılaştı**: P0 turunda `permutation_selected` içinde gerçek bir
  dış-test sızıntısı bulundu ve düzeltildi; özellik havuzu global
  değil **fold-lokal** hale getirildi ve bu bytecode düzeyinde bir
  regresyon testiyle (`test_p0_fold_local_feature_selection_pah.py`)
  kilitlendi.

**Genel değerlendirme:** PAH, Osman Hoca'nın 13 maddelik listesinin
büyük çoğunluğunu karşılıyor; sızıntı disiplini, nested değerlendirme,
tekrarlı-CV, preprocessing-vs-model etkisinin ayrıştırılması ve
mean±std raporlaması eksiksiz. **En derin/en özgün katkısı, provenance
(kaynak-kısayolu) riskini üç ayrı turda (F1 → F3 → F3 karar matrisi →
kök-neden EDA) kovalayıp kök nedene inmesidir** — bu, verilen özetlere
göre kardeş panellerin hiçbirinin bu derinlikte yapmadığı bir iştir ve
diğer panellere doğrudan örnek gösterilebilir.

**Bulunan boşluklar dört başlıkta toplanıyor ve hiçbiri metodolojik bir
hata değil — hepsi "raporlama tamlığı/sunum" düzeyinde:**

| Boşluk | Büyüklük | Öncelik |
|---|---|---|
| **AUROC hiçbir yerde model performans metriği olarak raporlanmıyor** | Gerçek, tek net eksik metrik | **Yüksek** (Osman Hoca'nın açık listesinde var) |
| Sensitivity yalnızca 2 yerde var (F2 CI tablosu, F3 shift testleri), E2-E6 çekirdeğinde yok | Orta | **Yüksek** (aynı gerekçe, ama kısmen kapalı) |
| Tek, birleşik "nihai karar tablosu" yok — bilgi 4 ayrı tabloya dağılmış | Orta, saf sunum | **Yüksek** (ucuz, jüri-görünür) |
| Mutlak-eşik fold kırılganlık sayımı (CFTR tarzı) yalnızca 1 örnekte var | Küçük-orta | Orta |

Kritik nokta: **bu boşlukların hiçbiri yeni model eğitimi gerektirmiyor.**
Gerekli tüm ham veri (OOF olasılıkları, per-fold tahminler) zaten
`reports/tables/` altında kayıtlı; eksik metrikler bu mevcut dosyalardan
**yeniden eğitim yapmadan** hesaplanabilir.

---

## 2. Dokuz Eksende Değerlendirme

### Eksen 1 — Çoklu-metrik raporlama tamlığı

**Osman Hoca'nın beklentisi:** F1, Sensitivity, Specificity, MCC,
Balanced Accuracy, AUPRC, AUROC birlikte raporlanmalı.

**PAH'ın mevcut durumu (dosya-dosya doğrulandı):**

| Kaynak tablo | F1 | MCC | Spec | Sens | AUPRC | AUROC | Bal.Acc |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `e2_model_comparison.csv` (E2 çekirdek, 800 satır) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `e2_model_comparison_v4fixed.csv` (P0) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `e3_calibration_metrics.csv` (E3) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `e5_threshold_selection.csv` (E5) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `e6_ensemble_metrics.csv` (E6) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `f0_uncertainty_analysis.csv` (F2, CI'li) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `f3_source_shift_tests.csv` (F3) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ (`raw_auc`) | ❌ |

**Bulgular:**

1. **AUROC gerçek bir eksik.** Depo genelinde `roc_auc_score` yalnızca
   dört dosyada kullanılıyor: `feature_selection.py` (Aşama D'nin
   *yardımcı* grup-ablation modelleri), `f1_adversarial_validation.py`
   (kaynak-ayrımı teşhisi), `f3_robustness_stress_tests.py` ve
   `f3_decision_matrix.py` (shift testlerinde `raw_auc`). **Hiçbiri
   final modelin performans raporu değil** — hepsi teşhis amaçlı.
   Yani E2-E6'nın 16 model×versiyon kombinasyonunun hiçbiri için AUROC
   raporlanmadı. Osman Hoca'nın listesinde açıkça yer aldığı için bu
   savunulabilir bir eksik değil, kapatılmalı.

2. **Sensitivity kısmen kapalı, ama en kritik yerde eksik.** F2 turunda
   `src/genova/metrics.py::sensitivity` eklendi ve final modelin CI
   tablosuna girdi; F3'ün shift testleri de sensitivity raporluyor. Ama
   **E2'nin model seçim kararının verildiği tablo sensitivity
   içermiyor** — yani "hangi model kazandı" kararı sensitivity'ye
   bakılmadan verildi. Pratikte bu kararı değiştirmezdi (F1 ve
   specificity birlikte raporlanıyordu, sensitivity bu ikisiyle güçlü
   ilişkili), ama Osman Hoca'nın "sadece F1'e bakmayın" uyarısının tam
   hedefi bu tür bir boşluk.

3. **Balanced Accuracy hiçbir yerde yok.** Ancak bu, sensitivity ve
   specificity'nin aritmetik ortalaması olduğu için, ikisi
   raporlandığında **türetilebilir** — bağımsız bir ölçüm turu
   gerektirmez. Düşük öncelikli.

4. **AUPRC yalnızca E2'de var**, kalibrasyon/eşik/ensemble
   aşamalarında düşmüş. E3-E6'nın eşik-bağımsız bir ayrım ölçüsü
   taşımaması, "kalibrasyon ayrımı bozdu mu" sorusunun doğrudan
   cevaplanamaması demek (dolaylı olarak Brier/logloss üzerinden
   izleniyor).

**Kardeş panellerle karşılaştırma:** MASTER'ın karar kriteri
("F1 korunuyor VE FP≤400/3500 VE Sensitivity≥0,62 VE AUROC/AUPRC
düşmüyor") **AUROC ve Sensitivity'yi bir kapı koşulu olarak** içeriyor
— yani MASTER bu iki metriği yalnızca raporlamıyor, karar kuralına
gömüyor. CFTR'nin Stage7 fragility tablosu da MCC/Sensitivity/
Specificity üçlüsünü birlikte kullanıyor. **Bu eksende PAH iki kardeş
panelin de gerisinde.**

**Boşluk: GERÇEK, öncelik YÜKSEK.** (Kapatma maliyeti düşük — bkz.
Öneri #1.)

---

### Eksen 2 — Çift-threshold raporlama

**Osman Hoca'nın beklentisi:** hem sabit 0,50 hem CV-optimize eşik
sonucu **ayrı ayrı** verilmeli.

**PAH'ın mevcut durumu:** `e5_threshold_selection.csv` bu şartı
**tam olarak ve sistematik biçimde** karşılıyor. Kolonlar:

```
chosen_threshold, f1_chosen, mcc_chosen, specificity_chosen, weighted_f1_chosen,
                  f1_fixed05, mcc_fixed05, specificity_fixed05, weighted_f1_fixed05
```

Yani üç aday × 50 fold'un tamamı için **hem seçilmiş eşik hem sabit
0,5** sonuçları yan yana kayıtlı, ve rapor bunu açık bir karar
tablosuna dönüştürmüş (`06`, "Karar kuralı (madde 6)" bölümü):
adaptif eşik üç adayda da sabit 0,5'i net biçimde geçiyor (+0,126 ile
+0,287 arası, %95 GA'ları sıfırdan uzak).

**Nüans — "sistematik olarak her adımda mı?" sorusunun dürüst cevabı:
hayır, ama bunun sağlam bir gerekçesi var.** E2 (model karşılaştırma)
**yalnızca** sabit 0,5 kullandı, adaptif eşik yok; E3 (kalibrasyon)
eşik-bağımsız metriklere (Brier/logloss) odaklandı. Bu bilinçli bir
aşama ayrımı: E2 bir *model/versiyon* karşılaştırmasıdır, eşik seçimi
E5'in işidir ve raporda bu açıkça yazılmıştır ("E2 bir model/versiyon
karşılaştırma aşaması, eşik seçimi E5'in işi; F1'in yanında
eşik-bağımsız AUPRC de raporlanıyor"). Bu, Osman Hoca'nın **başka bir**
maddesiyle ("preprocessing etkisi ile model etkisi birbirine
karıştırılmamalı") tam uyumlu bir tasarım tercihi — her aşamada tek bir
değişkeni oynatmak.

Ayrıca E5, sabit 0,5'in **neden** SLD-sonrası ölçekte anlamsızlaştığını
da açıklıyor (SLD düzeltmesi olasılıkları ~0,82'den ~0,37-0,40'a
çekiyor, bu ölçekte 0,5 aşırı tutucu kalıyor) — yani "sabit 0,5" sayısı
körü körüne değil, yorumlanarak raporlanmış.

**Kardeş panellerle karşılaştırma:** CFTR'nin Stage5'te ön-seçilen
eşikleri Stage7'de "yalnızca destekleyici kanıt" seviyesine düşürmesi,
PAH'ın E5'te eşiği baştan **nested** seçmesiyle karşılaştırıldığında
PAH lehine bir fark — PAH'ın eşiği hiçbir zaman ön-seçim riski
taşımadı, çünkü seçim en baştan iç örnekleme kapatıldı ve bu bir
sızıntı testiyle (`test_process_fold_threshold_choice_is_unaffected_by_
dis_test_contents`) kanıtlandı.

**Boşluk: YOK.** Bu eksen PAH'ın güçlü olduğu bir alan.

---

### Eksen 3 — Fold-düzeyi kırılganlık raporlama tarzı

**CFTR'nin yaptığı:** her finalist için "kaç fold'da MCC<0,40, kaç
fold'da Sensitivity<0,85, kaç fold'da Specificity<0,70" — **mutlak bir
kalite eşiğinin altına düşen fold sayısı.**

**PAH'ın mevcut durumu — üç farklı araç var, ama biri eksik:**

1. **mean±std:** her tabloda, eksiksiz. ✅
2. **Kazanma oranı (win rate):** çok yaygın ve sistematik — E1'e karşı
   paired kazanma oranı (%58-82), Platt/Beta fold sayımı (40/50, 49/50,
   42/50), ensemble-vs-solo (%30-38), F1 Ek'in 10/10'u, F3 karar
   matrisinin 1/10 ve 0/10'u. Bu **göreli** (A vs B) bir fold sayımıdır.
3. **Örnek-düzeyi bootstrap CI + NB testi:** F2/madde 10'da, doğru
   metodolojiyle. ✅
4. **Mutlak-eşik fold sayımı:** **neredeyse yok.** Depoda bulunan tek
   örnek E1-EK'in ağırlıklandırma tablosundaki "Specificity=0 olan fold
   sayısı" sütunu (A: 2/50, B: 1/50, C: 0/50) — ve bu, tek bir metriğin
   tek bir (dejenere) eşiği için, tek bir tabloda.

**Kritik soru: PAH'ın std/CI/kazanma-oranı üçlüsü CFTR'nin yaklaşımının
yerini tutuyor mu? — Hayır, tam olarak tutmuyor, çünkü farklı bir soruya
cevap veriyorlar:**

- Kazanma oranı **göreli** bir soruya cevap verir: "A, B'den kaç
  fold'da iyi?" Bu, iki aday **ikisi de kötüyse** bunu göstermez.
- std, dağılımın **genişliğini** verir ama **kuyruğun nerede**
  olduğunu vermez. Ortalama 0,62 ± 0,09'luk bir dağılımda kaç fold'un
  0,45'in altına düştüğü std'den doğrudan okunamaz (dağılım simetrik
  varsayılmadıkça).
- CFTR'nin sayımı **mutlak** bir soruya cevap verir: "bu model kaç
  fold'da kabul edilemez bölgeye düşüyor?"

Bu, Osman Hoca'nın **"ana hedef en yüksek tek skor değil, kararlı ve
güvenilir sistem — fold/seed arası çok dalgalanan bir model,
ortalaması yüksek olsa bile tercih edilmemeli"** maddesinin doğrudan
ölçüm aracıdır. PAH'ın E5 tablosunda final adayın ağırlıklı-F1'i
**0,6187 ± 0,0855** — std, ortalamanın ~%14'ü, yani hatırı sayılır bir
dalgalanma. Kaç fold'un örneğin 0,50'nin altında olduğu şu an
raporlanmıyor, ama **`e5_threshold_selection.csv`'de zaten satır satır
kayıtlı** — yalnızca sayılmamış.

**Boşluk: GERÇEK ama KÜÇÜK-ORTA, öncelik ORTA.** Yeni hesaplama değil,
mevcut CSV'lerden saf bir sayım. Tamamlayıcı bir görselleştirme/tablo
olarak eklenmeli — mevcut araçların yerine değil, yanına.

---

### Eksen 4 — Önsel-kayması / benign-heavy stres testi genişliği

**Üç panelin karşılaştırması:**

| Panel | Yöntem | Kapsanan aralık (patojenik önsel) | Nokta sayısı |
|---|---|---|---|
| **CFTR** | Önsel-kayması eğrisi | **0,10 – 0,81** | 6 |
| **MASTER** | Tek senaryo, çok-koşullu FP-kapılı test | Tek nokta (final kompozisyon) | 1 (ama çok-koşullu) |
| **PAH — E5** | Değerlendirme yeniden-ağırlıklandırma | 0,20 – 0,38 | 5 |
| **PAH — F3** | Monte Carlo yeniden-örnekleme | 0,20 – 0,50 | 5 |
| **PAH — F2** | Monte Carlo (şartname kompozisyonu) | 0,286 (100/250) | 1 |

**PAH'ın durumu iyi, ama iki farklı mekanizma karıştırılmamalı.** PAH
aslında **iki bağımsız** önsel-stres testi yapıyor ve bu bir güç:

- **E5'in duyarlılık eğrisi** (%20/25/28,6/33/38): eşik ve SLD sabit
  tutulup yalnızca *değerlendirme ağırlıklandırması* değiştiriliyor —
  "gerçek önsel tahminimizden farklı çıkarsa skorumuz ne olur?"
  Sonuç: düz, monotonik plato; ±10 puanlık sapmada ~0,13-0,17 değişim,
  çöküş yok.
- **F3'ün benign-ağırlıklı taraması** (benign %50/60/70/71,4/80,
  yani patojenik önsel 0,50→0,20): OOF'tan gerçekten *yeniden
  örnekleme* — "test setinin kompozisyonu gerçekten değişirse ne olur?"
  Sonuç: yine düzgün monotonik (F1 0,745→0,559), ani çöküş yok.

İkisi birlikte patojenik önsel **0,20-0,50** bandını iki farklı
mekanizmayla kapsıyor. Osman Hoca'nın "benign-heavy senaryo mutlaka
test edilmeli" şartı **karşılanmış** durumda (şartname noktası %28,6
dahil, hatta %20'ye kadar daha agresif benign-ağırlıklı senaryolar da).

**Eksik olan tek aralık: 0,20'nin altı.** CFTR 0,10'a kadar inmiş.
PAH'ın en uç benign-ağırlıklı noktası 0,20 (benign %80). 0,10 (benign
%90) noktası test edilmemiş. Bunun **pratik önemi sınırlı** — şartname
%28,6 diyor ve %20 zaten bunun belirgin altında bir güvenlik marjı —
ama CFTR'nin eğrisi kadar geniş bir "en kötü durum" resmi vermiyor.

Diğer yönde (0,50 üstü, CFTR'nin 0,81'i) PAH'ın *eğitim* dağılımı
zaten %83,3 patojeniktir ve E1/E2'nin tüm ham F1 sayıları fiilen bu
noktada ölçülmüştür — yani o uç **dolaylı olarak** kapsanmıştır, ayrıca
taranmasına gerek yok.

**MASTER ile farkı — burada PAH'ın eksiği yok, farklı bir tasarım var:**
MASTER tek noktada ama **çok-koşullu bir kapı** (FP sayısı + sensitivity
+ AUROC/AUPRC eşzamanlı) kullanıyor. PAH çok noktada ama **tek metrikli**
(F1, sonra ağırlıklı-F1). MASTER'ın FP-kapısı n=2931'de anlamlı
(FP≤400/3500 gibi mutlak bir sayı ancak büyük örneklemde stabildir);
PAH'ın n=369'unda ve yalnızca 61 benzersiz benign örneğinde böyle bir
mutlak FP eşiği **çok gürültülü olurdu** — bu yüzden PAH'ın çok-noktalı
eğri yaklaşımı bu örneklemde daha bilgilendirici.

**Boşluk: ÇOK KÜÇÜK (yalnızca 0,10 noktası), öncelik DÜŞÜK.**

---

### Eksen 5 — Ensemble arama disiplini ve ölçek-farkındalığı

**Ölçek karşılaştırması:**

| Panel | n | Ensemble arama uzayı |
|---|---|---|
| MASTER | ~2931 | ~197.000 (Hybrid-C) + ~48.000 (Hybrid-E) ağırlık kombinasyonu |
| PAH | **369** | α ∈ [0;1], adım 0,1 → **11 aday** × 3 blend yöntemi, her biri nested |

**Değerlendirme: PAH'ın küçük arama uzayı bu örneklemde DOĞRU bir
karardır, taklit edilmesi gereken bir eksiklik değildir.** Gerekçe:

PAH'ın nested yapısında ensemble ağırlığı her dış fold'un **iç**
örnekleminde seçiliyor. O iç örneklem ~295 satır, içinde yalnızca ~49
benign. 197.000 aday ağırlık kombinasyonunu bu kadar küçük ve azınlık
sınıfı bu kadar seyrek bir örneklemde taramak, **seçim-aşırı-uyumunun
ders kitabı örneği** olurdu: aday sayısı arttıkça, iç örneklemin
gürültüsüne en iyi uyan ağırlığın seçilme olasılığı artar ve iç-skor
optimistik biçimde şişer. MASTER'ın n=2931'i (ve ~500 patojenik/~3000
benign kompozisyonu) bu riski taşımaz; PAH'ın n=369'u taşır.

Bunun ampirik doğrulaması PAH'ın kendi verisinde zaten var: E6'da
kısıtlı ağırlıklı blend'in iç araması **50 fold'un 18'inde (%36)
α=1,0 seçti** — yani LightGBM'e sıfır ağırlık verdi, "ensemble
yapmamayı" seçti. 11 adaylık bir uzayda bile iç arama üyelerden birini
sık sık tamamen atıyorsa, uzayı 4 kat büyütmek sinyal değil gürültü
eklerdi.

**Kritik soru: bu gerekçe PAH'ın kendi raporlarında AÇIKÇA yazılı mı,
yoksa zımni mi kalmış?**

**Cevap: kısmen zımni.** `06`'nın E6 bölümü **kararın kendisini** çok
iyi gerekçelendiriyor — OOF korelasyonunun ılımlılığı (0,764), α=1,0
seçim sıklığı (%36), solo modele karşı negatif fark (−0,0014, kazanma
oranı %30), ve "karmaşıklık kendiliğinden ödül değil" ilkesi hepsi
yazılı. Ama **"arama uzayını neden 11 adayla sınırladık"** sorusunun
cevabı — yani örneklem büyüklüğüne bağlı seçim-aşırı-uyumu gerekçesi —
E6 bölümünde açıkça yazılmamış. Hiperparametre ızgaraları için benzer
bir gerekçe tablosu var ("Neden dar" sütunu, ama oradaki gerekçe
**hesaplama bütçesi**, istatistiksel aşırı-uyum değil).

Jüri "MASTER 197 bin kombinasyon taramış, PAH neden 11?" diye sorarsa,
cevap şu an raporlardan **doğrudan okunamıyor** — çıkarılabiliyor ama
yazılı değil. Bu, savunulabilir bir kararın belgelenmemiş olması
durumu.

**Boşluk: KARAR DOĞRU, GEREKÇE EKSİK BELGELENMİŞ. Öncelik ORTA**
(tek paragraflık bir ekleme).

---

### Eksen 6 — Seçim-aşırı-uyumu (selection overfitting) karşı disiplin

**İki panelin yaklaşımı:**

| | PAH | MASTER |
|---|---|---|
| Mekanizma | Tekrarlı-bölme (10 farklı `StratifiedKFold`, seed 0-9) + Nadeau-Bengio düzeltmeli paired t-test | 25 tekrar (5 seed × 5 fold), iç OOF'ta seçim + dış holdout'ta tek skorlama |
| Özet metrik | **Kazanma sayısı** (örn. 10/10) + **yön-işaretli marj** (±std) + **NB p-değeri** | **"Strict koşulu geçen fold oranı"** (örn. 0,48) |
| Doğası | İstatistiksel test tabanlı, **göreli** (A vs B) | Kapı-koşulu tabanlı, **mutlak** |

**PAH'ın uygulaması gerçekten yaygın ve tutarlı** — bu disiplin tek bir
yerde değil, en az beş ayrı kararda uygulanmış:

- **Madde 7** (`depth=3` vs `depth=5`): tekrarlı-bölme teşhisi + NB
  kapanış testi → mevcut model korundu.
- **Madde 9** (A/B/C ağırlıklandırma): NB testi → eleme kararı sağlam.
- **Madde 16** (genişletilmiş CatBoost ızgarası): yeni aday kazanmadı,
  `depth=5` korundu.
- **F1 Ek** (22 vs 26 özellik): 10 tekrar → 26 özellik 10/10 kazandı,
  marj −0,0877 ± 0,0272.
- **F3 karar matrisi** (B/C vs A): 10 tekrar + NB, ikisi birden.

Ve bu disiplinin **değeri somut olarak kanıtlandı**: F3 karar
matrisinde B adayı için iki yöntem **yön konusunda bile anlaşmadı**
(10-tekrar marjı −0,0259, NB'nin nominal farkı +0,0218, p=0,83) — yani
tek bir karşılaştırmaya güvenilseydi, hangi karşılaştırma seçildiğine
göre zıt sonuçlara varılabilirdi. Bu, Osman Hoca'nın "sadece en iyi
koşuyu raporlamayın" uyarısının neden haklı olduğunun canlı kanıtı.

**Hangisi daha bilgilendirici? — İkisi farklı şeyler ölçüyor, ve
PAH'ınki bir noktada eksik kalıyor:**

- **PAH'ın NB p-değeri**, "bu fark gerçek mi yoksa fold gürültüsü mü?"
  sorusuna cevap verir. Bağımlı fold'lar için varyans düzeltmesi
  yaptığı için naif t-testten **daha muhafazakâr** — bu doğru ve
  savunulabilir bir tercih (küçük örneklemde yanlış-pozitif keşfi
  önler).
- **PAH'ın kazanma sayısı** (10/10) bunu tamamlar ve etkinin
  **tutarlılığını** gösterir.
- **MASTER'ın "strict geçme oranı"** ise bambaşka bir soruya cevap
  verir: "bu model kaç fold'da **mutlak kalite kapısını** geçiyor?" —
  hiçbir rakip aday olmadan, kendi başına.

**PAH'ın eksik olduğu nokta tam olarak burası ve Eksen 3 ile aynı
kökten geliyor:** PAH'ın tüm fold-sayımları **göreli**. Final modelin
"kaç fold'da kendi başına kabul edilebilir performans gösterdiği"
hiçbir yerde raporlanmıyor. Final aday ağırlıklı-F1'i 0,6187 ± 0,0855
— jüri "peki bu modelin en kötü fold'u ne kadar kötü, ve kaç fold kötü
bölgede?" diye sorarsa cevap mevcut raporlarda yok (ham veri var,
sayım yok).

**Değerli mi? — Evet, ama PAH'ın NB disiplininin YERİNE değil,
YANINA.** MASTER'ın oranı istatistiksel anlamlılık vermez (NB verir);
PAH'ın NB'si mutlak kalite resmi vermez (oran verir). İkisi tamamlayıcı.

**Boşluk: GERÇEK, öncelik ORTA.** (Eksen 3'ün önerisiyle **aynı**
uygulamada birleşiyor — tek bir iş.)

---

### Eksen 7 — Provenance / shortcut-risk araştırma derinliği ⭐

**Bu eksen PAH'ın en güçlü yanıdır ve diğer panellere doğrudan örnek
gösterilmelidir.**

**CFTR'nin karşılık gelen bulgusu (verilen özete göre):** TEAMX-05'te
`AL_all_missing` göstergesinin yalnızca 7 satırda aktif olduğu, 7'sinin
de patojenik olduğu tespit edilmiş, "shortcut riski taşıyor" diye
**not edilmiş ve havuza sokulmamış.** Yani: risk **fark edildi ve
kaçınıldı** — ama mekanizması araştırılmadı, sonuçları ölçülmedi.
*(CFTR'nin bunun ötesinde bir analiz yapıp yapmadığı verilen özette yer
almıyor — bilinmiyor.)*

**MASTER'ın karşılık gelen pratiği:** KANSER'den devralınan disiplinde
"eksiklik bilgisini ayrı ablation olarak ölçme" var — yani ölçüm var,
ama provenance/kaynak-kayması özelinde bir stres testi verilen özette
yok. *(Var olup olmadığı bilinmiyor.)*

**PAH'ın yaptığı — dört aşamalı, giderek derinleşen bir kovalamaca:**

1. **`03b` (Aşama D turu):** `al_all_missing`'in `CAT_1`/`CAT_2` ile
   confound'u — çapraz tablolar, Cramér's V (0,755), Fisher testleri,
   kısmi korelasyon. *Risk tespit edildi.*
2. **Madde 11 (P1):** `al_all_missing`/`CAT_1` havuzdan çıkarıldı;
   maliyet ölçüldü ve **maliyetsiz** çıktı (p=0,83). *Ucuz olan
   temizlik yapıldı.*
3. **F1 (adversarial validation):** Asıl soru soruldu — "temizlikten
   sonra havuzda hâlâ provenance sinyali var mı?" Yardımcı bir
   sınıflandırıcı benign sınıf içinde kaynak-1'i **AUC=1,0000 ±
   0,0000** ile ayırdı. Özellik bazlı tarama 4 suçlu buldu (`AL_26`
   AUC=1,000; `AL_12`/`AL_7` 0,949; `AL_49` 0,769) ve **mekanizmayı
   doğruladı**: bu kolonların ham NaN deseni `CAT_1` boşken sistematik
   olarak boş, `v2`'nin sıfır-doldurması bu boşluğu `0,0`'a çevirip
   neredeyse kusursuz bir kaynak göstergesi yaratıyor. Ablasyon
   denendi; 10 tekrarlı bölmede **10/10** gerçek maliyet çıktı
   (−0,0877 ± 0,0272) — yani madde 11'in aksine "bedava" değil.
4. **F3 (sağlamlık stres testleri) — asıl kanıt:** Risk teorik
   kalmadı, **gerçek bir çöküşe dönüştürülüp ölçüldü.** `CAT_1`
   doluluk-geçişi testinde, `CAT_1`-boş alt-kümede eğitilip
   `CAT_1`-dolu alt-kümede test edilen model **F1=0,818 → 0,059**
   (mutlak düşüş 0,759, karar eşiğinin ~5 katı) ve — kritik olarak —
   **eşikten ve kalibrasyondan tamamen bağımsız raw AUC de 0,831 →
   0,573'e** (rastgele tahmine yakın) düştü. Yani bu bir eşik
   artefaktı değil, gerçek bir genelleme çöküşü.
5. **F3 karar matrisi — hipotezin çürütülmesi:** "Bu 4 özelliği
   çıkarırsak çözülür" varsayımı doğrudan sınandı ve **çürütüldü**:
   B (23 özellik, AUC=0,5727) A ile pratikte **aynı**, C (22 özellik,
   AUC=0,4937) A'dan **daha kötü**. Yani suçlu bu özellikler değildi.
6. **Kök-neden karakterizasyonu (saf EDA, model yok):** `CAT_1`-boş
   alt-küme (n=132) etiket dengesinde farklı **değil** (%83,3 vs
   %83,5 patojenik — neredeyse birebir aynı), ama `AL_` bloğunun
   eksiklik oranında **çarpıcı** biçimde farklı: **%91,4 vs %36,6**.
   Yani o alt-kümede eğitilen bir model `AL_` sinyalini hiç öğrenemiyor
   — çöküşün nedeni birkaç kolon değil, alt-kümenin yapısal
   temsilsizliği.
7. **F4 (SHAP) ile mekanizmanın görsel doğrulanması:** Riskli 4
   özelliğin `CAT_1`-boş satırlardaki ortalama |SHAP| katkısı,
   `CAT_1`-dolu satırlara göre 1,4×-3,0× **daha düşük** — modelin o
   alt-kümede bu özellikleri kullanamadığının bağımsız kanıtı.

**Değerlendirme:** PAH bu konuda **bir riski fark etmekle
yetinmedi** — riski ölçtü, çözüm önerdi, çözümü test etti, çözümün
işe yaramadığını dürüstçe raporladı, kök nedeni buldu, ve bulguyu
üçüncü bir yöntemle (SHAP) doğruladı. Özellikle değerli olan iki nokta:

- **Negatif sonucun dürüstçe raporlanması.** "4 özelliği çıkarmak
  çözer" hipotezi PAH'ın kendi hipoteziydi ve yanlış çıktı; rapor bunu
  gizlemek yerine "⚠️ BEKLENMEDİK BULGU" başlığıyla öne çıkardı.
- **Modelin gereksiz yere bozulmaması.** Çözüm işe yaramadığı için
  model değiştirilmedi — "bir şey yapmış olmak için" maliyet
  ödenmedi, ve bu karar gerekçesiyle `03b`'ye belgelendi.

Bu, verilen özetlere göre kardeş panellerin hiçbirinin bu derinlikte
yapmadığı bir iştir. **Boşluk: YOK — bu bir üstünlüktür.** (Bkz.
Bölüm 4.)

---

### Eksen 8 — Özellik-seçimi sızıntı kontrolü

**PAH'ın durumu:** P0 turunda `feature_selection.py::permutation_
selected` fonksiyonunun **dış-test verisine eriştiği** tespit edildi —
yani permutation importance dış-test üzerinde hesaplanıyor ve seçilen
özellik havuzu dolaylı olarak dış-testten bilgi alıyordu. Bu **gerçek
bir sızıntıydı** ve düzeltildi:

- Havuz **global**'den **fold-lokal**'e çevrildi (her dış fold kendi
  havuzunu üretir).
- Eski (leaky) havuz silinmedi, `_ARCHIVED_leaky.json` olarak
  arşivlendi — izlenebilirlik korundu.
- Düzeltme **bytecode düzeyinde bir regresyon testiyle** kilitlendi
  (`test_p0_fold_local_feature_selection_pah.py`): fonksiyonun imzası
  ve derlenmiş kodu dış-teste erişemeyeceğini doğruluyor, eski
  5-argümanlı çağrı `TypeError` veriyor.
- Etkisi ölçüldü: `e2_model_comparison_v4fixed.csv` ile eski/yeni
  paired karşılaştırma yapıldı, performans farkı bulunmadı — yani
  sızıntı sonuçları şişirmemişti, ama **yine de düzeltildi.**

Bu, Osman Hoca'nın iki maddesinin kesişimi: "feature selection mutlaka
fold içinde yapılmalı" **ve** "data leakage kesinlikle olmamalı".

**Kardeş paneller hakkında — açık soru:** CFTR ve MASTER'ın
feature-selection adımlarında permutation-importance tabanlı bir seçim
olup olmadığı, varsa bunun dış-test'e erişip erişmediği **verilen
özetlerde yer almıyor — bilinmiyor.** MASTER'ın "KANSER'den devralınan
disiplin"i arasında "fold-safe öğrenme" sayılıyor, bu olumlu bir işaret
ama permutation importance özelinde bir garanti değil.

**Öneri (PAH'ın doğrudan kontrol edemeyeceği, takım-içi bir aksiyon):**

> Diğer panellerin (CFTR, MASTER, KANSER) feature-selection kodunda
> permutation importance veya benzeri bir "model çıktısına bakarak
> özellik seçen" adım varsa, bu adımın **dış-test/holdout verisine
> erişip erişmediği** kontrol edilmeli. PAH'ta bu tam olarak böyle bir
> sızıntıydı ve gözle fark edilmesi zordu (fonksiyon imzasına bakmak
> gerekti). PAH'ın `test_p0_fold_local_feature_selection_pah.py`
> testi bu kontrol için doğrudan bir şablon olarak paylaşılabilir.

**Boşluk: PAH'ta YOK (bulundu ve düzeltildi). Kardeş panellerde
BİLİNMİYOR — takım-içi kontrol önerisi olarak kayda geçirildi.**

---

### Eksen 9 — Nihai karar tablosu formatı

**Osman Hoca'nın istediği format:**
`Preprocessing | Model | F1 | MCC | Sensitivity | Specificity | AUPRC | AUROC | Threshold | Std`

**PAH'ın mevcut durumu: böyle tek, birleşik bir tablo YOK.** Bilgi en az
dört ayrı tabloya dağılmış ve **her biri farklı bir eşik/ölçek
rejiminde** ölçülmüş (bu, tabloları basitçe yan yana koymayı da
engelliyor):

| Tablo | Nerede | İçerdiği | Eşik rejimi |
|---|---|---|---|
| E2 karşılaştırma | `06`, satır 106-123 | F1, MCC, AUPRC, Spec | Sabit 0,5, kalibrasyonsuz |
| E5 nihai karşılaştırma | `06`, satır 561-566 | F1, MCC, Spec, Ağırlıklı-F1 | Adaptif, SLD-sonrası |
| Aşama E nihai özet | `06`, satır 5-13 | Yalnızca ağırlıklı-F1 | Adaptif |
| F2 CI tablosu | `10`, satır 26-31 | F1, MCC, Spec, **Sens** + %95 CI | Final bundle, eşik=0,35 |

Yani jürinin "final model ne kadar iyi?" sorusuna tek bir yerden cevap
alması mümkün değil — dört tabloyu okuyup, hangisinin hangi eşik
ölçeğinde olduğunu anlaması gerekiyor. Üstelik **hiçbirinde
Sensitivity + AUPRC + AUROC birlikte yok** (Eksen 1'in bulgusu burada
somutlaşıyor).

**Kardeş panellerle karşılaştırma:** CFTR'nin Stage7 tablosu üç
finalisti tek bir yerde, aynı metrik setiyle karşılaştırıyor ve buna
fold-fragility + eşik-duyarlılık penceresi ekliyor. MASTER'ın strict
kapısı da tek bir satırda çok-metrikli bir karar veriyor. **Bu eksende
PAH her iki kardeş panelin de gerisinde** — metodoloji değil, sunum
olarak.

**Ek bir zorluk (dürüstlük notu):** PAH'ın tabloları farklı eşik
rejimlerinde olduğu için birleşik tablo yaparken **hangi rejimde
raporlandığı açıkça belirtilmeli**, yoksa E2'nin 0,9266'sı ile E5'in
0,7988'i yan yana konduğunda "model kötüleşmiş" gibi yanlış bir izlenim
doğar (raporda bu tuzak zaten uzun uzun açıklanmış: E5'in düşük ham
F1'i **beklenen ve istenen** bir sonuç). Birleşik tablo bu nüansı
korumalı.

**Boşluk: GERÇEK, öncelik YÜKSEK** (ucuz + jüri-görünürlüğü en yüksek
madde).

---

## 3. Önceliklendirilmiş Öneri Listesi

> Efor tahminleri, **yeni model eğitimi gerektirmeyen** işler için
> verilmiştir. Dördünün hiçbiri split bankasına, `final_model_bundle_
> v2.pkl`'e veya `predict.py`'ye dokunmaz.

### 🔴 Öneri 1 — Eksik metrikleri mevcut OOF'lardan geriye dönük hesapla
**(Eksen 1 + 9 — en yüksek değer/maliyet oranı)**

**Ne:** AUROC ve Sensitivity'yi (ve türev olarak Balanced Accuracy'yi)
final aday için ve mümkünse E2'nin ilk 3-5 kombinasyonu için hesapla.

**Neden yapılabilir:** Gerekli ham veri **zaten diskte**:
`e3_calibrated_oof_predictions.csv` (44.280 satır, tüm dış-test
tahminleri) ve `e4_prior_corrected_probabilities.csv` (11.070 satır).
AUROC eşik-bağımsız olduğu için doğrudan bu olasılıklardan çıkar;
sensitivity için `src/genova/metrics.py::sensitivity` **zaten mevcut**
(F2'de eklendi, 4 testi var).

**Efor:** ~1-1,5 saat (tek bir yeni script + rapor bölümü). **Model
eğitimi yok, yeniden fit yok.**

**n=369 kısıtı açısından:** Sorun yok — bu bir *raporlama* eklemesi,
yeni bir seçim kararı değil. Hiçbir aşırı-uyum riski yaratmaz, çünkü
hiçbir karar bu metriklere göre yeniden verilmeyecek (mevcut kararlar
donmuş durumda).

**Uyarı:** Bu metrikler **geriye dönük raporlama** olarak eklenmeli,
"yeni kanıt bulundu, model değişmeli" kapısı olarak değil. Eğer AUROC
beklenmedik bir şey gösterirse, bu ayrı bir tartışma konusudur —
otomatik bir model değişikliği tetiklememelidir.

---

### 🔴 Öneri 2 — Tek birleşik "Nihai Karar Tablosu" ekle
**(Eksen 9)**

**Ne:** `06_MODEL_SECIM_RAPORU_PAH.md`'nin **başına** (veya
`PAH_SUBMISSION_INTERFACE.md`'nin model kartının yanına) Osman Hoca'nın
formatında tek bir tablo:

```
Preprocessing | Model | F1 | MCC | Sens | Spec | AUPRC | AUROC | Threshold | Std
```

**Kritik tasarım şartı:** Tabloya **"eşik rejimi" sütunu veya dipnotu**
eklenmeli — E2 (sabit 0,5, kalibrasyonsuz) ile E5/final (adaptif,
SLD-sonrası) satırlarının farklı ölçeklerde olduğu görünür olmalı.
Aksi halde tablo, raporun 500 satır boyunca açıkladığı nüansı
(düşük ham F1'in beklenen olduğu) tek hamlede yok eder.

**Efor:** ~30-45 dakika (Öneri 1 tamamlandıktan sonra; öncesinde
AUROC/Sens sütunları boş kalır).

**Bağımlılık:** Öneri 1'den sonra yapılmalı.

---

### 🟡 Öneri 3 — Mutlak-eşik fold kırılganlık tablosu (CFTR tarzı)
**(Eksen 3 + 6 — tek işte birleşiyor)**

**Ne:** Final aday için (ve istenirse E5'in üç adayı için) mevcut
`e5_threshold_selection.csv`'nin 50 satırından saf bir sayım:

```
Kaç fold'da ağırlıklı-F1 < 0,50?
Kaç fold'da MCC < 0,25?
Kaç fold'da Specificity < 0,60?
Kaç fold'da Sensitivity < 0,60?   (Öneri 1'den sonra)
En kötü fold değeri / en iyi fold değeri
```

Bu aynı zamanda MASTER'ın "strict geçme oranı"nın PAH karşılığını
üretir: **eşzamanlı** bir kapı tanımlayıp (örn. wF1≥0,55 VE Spec≥0,60)
50 fold'un kaçının geçtiğini raporlamak.

**Neden değerli:** PAH'ın tüm mevcut fold-sayımları **göreli** (A vs B).
Final modelin **kendi başına** kaç fold'da kabul edilebilir olduğu şu
an hiçbir yerde yazmıyor. Final aday 0,6187 ± 0,0855 — std ortalamanın
~%14'ü, yani bu soru meşru.

**Efor:** ~45 dakika (saf `pandas` sayımı, mevcut CSV'den; yeniden
hesaplama yok).

**n=369 kısıtı açısından:** Eşikler **keyfi seçilmemeli** ve
"kaybeden adayı elemek" için kullanılmamalı — 50 fold zaten aynı 369
satırın tekrarlı bölünmesi olduğu için bu sayımlar bağımsız gözlem
değildir. Yalnızca **tanımlayıcı** (descriptive) bir kırılganlık resmi
olarak sunulmalı, karar kuralı olarak değil. Bu sınır tabloya not
düşülmeli.

---

### 🟡 Öneri 4 — Ensemble arama ölçeği kararının gerekçesini yaz
**(Eksen 5)**

**Ne:** `06`'nın E6 bölümüne tek paragraf: "Ensemble ağırlığı araması
neden 11 adayla (α adım 0,1) sınırlandı?" — gerekçe: iç örneklemin ~295
satır / ~49 benign olması, aday sayısı arttıkça iç-skor
optimizasyonunun gürültüye uyma riski, ve bunun ampirik işareti
(%36 fold'da α=1,0 seçilmesi). MASTER'ın ~197.000'lik aramasının
n=2931'de neden güvenli, n=369'da neden güvensiz olduğu bir cümleyle
belirtilmeli.

**Efor:** ~15 dakika.

**Not:** Bu bir **savunma** belgesi — jüri karşılaştırma yaparsa PAH'ın
küçük aramasının bilinçli bir istatistiksel karar olduğu, kaynak
kısıtı veya ihmal olmadığı görünür olur.

---

### 🟢 Öneri 5 — Önsel taramasına 0,10 noktası ekle
**(Eksen 4 — düşük öncelik)**

**Ne:** F3'ün benign-ağırlıklı taramasına benign %90 (patojenik önsel
0,10) noktasını ekleyerek CFTR'nin aralığının alt ucunu yakala.

**Efor:** ~20 dakika (mevcut `monte_carlo_final_f1_simulation`
çağrısına bir oran daha eklemek).

**Dürüst değerlendirme:** Pratik değeri sınırlı — şartname %28,6 diyor,
PAH zaten %20'ye kadar test etmiş durumda ve eğri düz/monotonik
çıkıyor, %10'da da sürpriz beklenmiyor. **Yalnızca çapraz-panel
karşılaştırılabilirliği** için değerli. Zaman kısıtı varsa atlanabilir.

---

### ⛔ Uygulanması ÖNERİLMEYEN (n=369 kısıtı nedeniyle)

| Kardeş panel pratiği | Neden PAH'a uygun değil |
|---|---|
| **MASTER'ın ~197.000'lik ensemble ağırlık taraması** | İç örneklem ~295 satır / ~49 benign. Bu ölçekte arama, seçim-aşırı-uyumunun ders kitabı örneği olur. PAH'ın 11-adaylık nested araması bu örneklem için doğru ölçek. (Bkz. Eksen 5.) |
| **MASTER'ın mutlak FP-sayısı kapısı (FP≤400/3500)** | Mutlak FP eşikleri ancak büyük örneklemde stabildir. PAH'ın 61 benzersiz benign örneğinde FP sayısı fold'lar arası çok gürültülü olur; PAH'ın oransal/ağırlıklı-F1 yaklaşımı daha uygun. |
| **MASTER'ın Isotonic kalibrasyonu** | PAH'ta E3'te doğrudan test edildi ve **açık aşırı-uyum** gösterdi (çok az benzersiz kalibre değer). Beta seçildi. Bu, "örneklem büyüklüğüne göre kalibratör seçimi" ilkesinin PAH tarafında da bağımsız olarak doğrulandığı bir nokta — MASTER'ın n=2931'i Isotonic'i kaldırabiliyor, PAH'ın n=369'u kaldıramıyor. **İki panelin farklı seçim yapması bir tutarsızlık değil, doğru bir uyarlamadır.** |
| **CFTR'nin varyant-kümesi bazlı bootstrap'ı** | PAH'ın `v1`'inde `Variant_ID` zaten benzersiz (369 satır = 369 varyant, 3 tam-yinelenen satır dedüplike edilmiş). Aynı varyantın tekrarlı OOF tahmini yok, dolayısıyla PAH'ın örnek-düzeyi bootstrap'ı **zaten** varyant düzeyindedir — CFTR'nin ek kümeleme adımı PAH'ta gereksiz. *(Bu, PAH'ın CFTR'nin gerisinde olduğu bir nokta değil, veri yapısının farklı olmasıdır.)* |

---

## 4. PAH'ın Diğer Panellere Örnek Olabilecek Güçlü Yanları

> Bu bölüm, çapraz-panel fikir alışverişinde **PAH'ın verebileceği**
> pratikleri listeler (CFTR'nin "TEAMX" turunda başka panellerden fikir
> alması gibi).

### ⭐ 1. Provenance riskini kök nedene kadar kovalamak (Eksen 7 — en güçlü katkı)

PAH, bir kısayol riskini "fark edip kaçınmakla" yetinmedi; **riski
ölçülebilir bir çöküşe dönüştürdü** (`CAT_1` doluluk-geçişi testi:
F1 0,818→0,059, raw AUC 0,831→0,573), **önerdiği çözümü test etti**,
çözümün **işe yaramadığını dürüstçe raporladı** (B: 0,5727 ≈ A; C:
0,4937 < A), ve **kök nedeni buldu** (`CAT_1`-boş alt-kümede `AL_`
eksiklik oranı %91,4 vs %36,6 — sorun özellikler değil, alt-kümenin
yapısal temsilsizliği).

**Diğer panellere doğrudan aktarılabilir şablon:**

> Bir özellik "kaynak göstergesi" gibi görünüyorsa, onu havuzdan
> çıkarmadan önce **kaynak-geçişi testi** yap: veriyi kaynak
> değişkenine göre ikiye böl, birinde eğit diğerinde test et, **her iki
> yönde de**. Sonra kalibrasyondan bağımsız olduğunu kanıtlamak için
> **raw AUC'yi de** ölç. Yön asimetrisi (PAH'ta bir yönde F1=0,930,
> diğer yönde 0,059) tek başına çok bilgilendiricidir.

CFTR'nin `AL_all_missing` bulgusu (7 satır, hepsi patojenik) tam olarak
bu testin uygulanabileceği bir durum — ve **7 satırın hepsinin aynı
etikette olması**, PAH'ın `CAT_1`-boş alt-kümesindekinden bile daha
keskin bir sinyal. *(CFTR'nin bu testi yapıp yapmadığı bilinmiyor;
yapılmadıysa öneriliyor.)*

### ⭐ 2. Negatif ve beklenmedik sonuçları öne çıkarmak

PAH'ın raporlarında en az dört yerde, **kendi hipotezinin çürütüldüğü**
başlık altında raporlanmış:

- "⚠️ BEKLENMEDİK BULGU" (F3 karar matrisi: 4 özelliği çıkarmak
  çözmüyor)
- F4/SHAP: "riskli 4 özelliğin katkısı küçük olmalı" beklentisi
  **doğrulanmadı** (`AL_7` global sıralamada #1 çıktı) — rapor bunu
  gizlemek yerine yazdı ve ne anlama geldiğini yeniden yorumladı.
- Madde 7: NB testinin "çelişkili sonuç" verdiği açıkça yazıldı, karar
  ertelendi.
- F1: ablasyonun "ne temiz maliyetsiz ne net maliyetli" olduğu, "net
  bir etiket yapıştırmak yanıltıcı olur" diye raporlandı.

Bu, Osman Hoca'nın **"sadece en iyi koşuyu raporlamayın"** maddesinin
en katı yorumudur.

### ⭐ 3. Tek karşılaştırmaya asla güvenmemek (tekrarlı-bölme + NB ikilisi)

PAH'ın standart hâline gelmiş deseni — **10 farklı rastgele bölme ile
kazanma sayısı + yön-işaretli marj + NB düzeltmeli p-değeri** — en az
beş kararda uygulandı (madde 7, 9, 16, F1 Ek, F3 karar matrisi).

**Değerinin somut kanıtı:** F3 karar matrisinde iki yöntem B adayı için
**yön konusunda bile anlaşmadı** (10-tekrar marjı −0,0259 vs NB'nin
nominal +0,0218, p=0,83). Tek bir karşılaştırmaya güvenilseydi, hangi
karşılaştırmanın seçildiğine göre zıt kararlar verilebilirdi.
**Küçük örneklemli panellerin (özellikle CFTR, n≈111) doğrudan
benimseyebileceği bir disiplin.**

### ⭐ 4. Sabit split bankası + bytecode düzeyinde sızıntı testi

Split bankası bir kez üretilip donduruldu ve **hiçbir deneyde yeniden
üretilmedi** — F1'den F6'ya kadar tüm turlar boyunca dosya mtime'ı
değişmedi. Bu, Osman Hoca'nın "aynı splitlerle karşılaştırın"
maddesinin en katı uygulaması ve panel-arası olarak da benimsenebilir.

Buna ek olarak PAH, bulduğu gerçek sızıntıyı (P0,
`permutation_selected`) yalnızca düzeltmekle kalmadı, **fonksiyon
imzası ve derlenmiş bytecode düzeyinde bir regresyon testiyle**
kilitledi — yani sızıntının geri gelmesi teknik olarak imkânsız hale
getirildi. Bu test (`test_p0_fold_local_feature_selection_pah.py`)
diğer panellere doğrudan bir şablon olarak paylaşılabilir (bkz.
Eksen 8'in açık sorusu).

### ⭐ 5. Nested eşik seçiminin sızıntı testiyle kanıtlanması

PAH'ın E5'i, eşiğin dış-teste bakılmadan seçildiğini **iddia etmekle
kalmıyor, test ediyor**: aynı sabit iç örneklem için dış-test içeriği
tamamen değiştirildiğinde `chosen_threshold` **birebir aynı** kalıyor,
ama dış-teste bağlı metrikler değişiyor. Aynı desen E6'da ensemble
ağırlığı (`α`) için de tekrarlanmış.

CFTR'nin Stage5'te ön-seçilen eşikleri Stage7'de "yalnızca destekleyici
kanıt" seviyesine düşürmek zorunda kalması, bu tür bir testin baştan
kurulmasının değerini gösteriyor — **PAH'ın eşiği hiçbir zaman bu
sorunu yaşamadı.**

### ⭐ 6. "Bir şey yapmış olmak için" maliyet ödememek

F3'ün bulgusundan sonra üç seçenek (26/23/22 özellik) ölçüldü, hiçbiri
kırılganlığı düzeltmedi, ve **model değiştirilmedi** — çünkü fayda
sağlamayan bir değişiklik için performans maliyeti ödemek anlamsızdı.
Karar ve gerekçesi `03b`'ye ve `09`'a belgelendi, sınırlama
`PAH_SUBMISSION_INTERFACE.md`'nin "Bilinen Kısıtlar" bölümüne taşındı.

Aynı disiplin madde 7 (`depth=3` reddedildi), madde 16 (genişletilmiş
ızgara `depth=5`'i değiştirmedi) ve E6'da (ensemble reddedildi) de
görülüyor. Osman Hoca'nın **"ana hedef en yüksek tek skor değil,
kararlı ve güvenilir sistem"** maddesinin pratikteki karşılığı budur.

---

## 5. Sonuç

**Osman Hoca'nın PAH'a özel olumlu notu geçerliliğini koruyor** —
nested + group-aware yapı bozulmadı, aksine P0'ın sızıntı düzeltmesi ve
F1-F6'nın stres testleriyle güçlendi.

Bulunan dört boşluğun **hiçbiri metodolojik hata değil**; hepsi
raporlama tamlığı/sunum düzeyinde ve **hiçbiri yeni model eğitimi
gerektirmiyor** — gerekli ham veri (44.280 satırlık kalibre OOF
tahminleri dahil) zaten `reports/tables/` altında kayıtlı.

En yüksek öncelikli iki iş (Öneri 1 ve 2: eksik metriklerin geriye
dönük hesaplanması + tek birleşik karar tablosu) toplam ~2 saatlik bir
efor ve jüri görünürlüğü en yüksek boşluğu kapatıyor.

Buna karşılık PAH'ın provenance araştırması (Eksen 7), tekrarlı-bölme
disiplini (Eksen 6) ve sızıntı-testi kültürü (Eksen 2, 8), verilen
özetlere göre kardeş panellerin gerisinde değil **önündedir** ve
çapraz-panel fikir alışverişinde PAH'ın vereceği asıl katkıdır.
