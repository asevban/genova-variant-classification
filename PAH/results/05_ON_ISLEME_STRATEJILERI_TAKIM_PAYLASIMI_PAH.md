# PAH Paneli — Ön İşleme Stratejileri (Takım Paylaşım Özeti)

> **Amaç:** Bu dosya, PAH panelinde şu ana kadar (Aşama A-D + Tamamlayıcı Deney Turu) **hangi stratejinin neden seçildiğini, neyin reddedildiğini ve hangi kanıta dayandığını** strateji-bazında özetler — panel/görev kronolojisine göre değil, karar kategorisine göre. Diğer panelleri (MASTER, KANSER, CFTR) çalışan takım arkadaşlarının kendi yaklaşımlarıyla satır satır karşılaştırması için hazırlandı.
>
> **Kapsam dışı:** Aşama E (gerçek model eğitimi, hiperparametre arama, eşik/kalibrasyon) henüz başlamadı. Buradaki tüm sayılar ya betimsel (EDA) ya da sabit/ayarlanmamış diagnostic modellerle (`LogisticRegression`, `ExtraTreesClassifier`, `DummyClassifier`) yapılan ön işleme karşılaştırmalarıdır — **final model performansı değildir.**
>
> **Detaylı kaynaklar:** `reports/01_EDA_RAPORU_PAH.md`, `reports/02_ON_ISLEME_KARARLARI_PAH.md`, `reports/03_OZELLIK_SECIMI_PAH.md`, `reports/03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`, `reports/04_PREPROCESSING_MERDIVENI_PAH.md`. Kod: `src/genova/pah/`.

---

## Hızlı Karşılaştırma Tablosu

| Strateji alanı | Ne kullandık | Neyi reddettik |
|---|---|---|
| CV tasarımı | Dış 5-fold × 10 tekrar + iç 4-fold, `StratifiedGroupKFold`, sabit split bankası (dosyaya yazılı, bir daha üretilmez) | Tek bölme, k-fold (grupsuz), her deneyde yeniden rastgele bölme |
| Eksik veri göstergesi | Yalnızca istatistiksel olarak gerekçeli az sayıda blok/temsilci gösterge (4 kolon) | 334 kolona körü körüne bire bir eksiklik göstergesi |
| Sayısal doldurma | Kolon tipine göre ayrı strateji: sıfır (frekans-tipi), medyan (skor-tipi, fold-içi) | Tüm kolonlara tek/global doldurma stratejisi |
| Kategorik kodlama | One-hot (düşük kardinalite) + frekans kodlama (yüksek kardinalite, sızıntısız) | Ham target/mean encoding |
| Sayısal dönüşüm | Kolon-tipine özel: log1p / logit / rank-quantile, yapısal (etiketten bağımsız) sınıflandırmayla seçilmiş | Tüm sürekli kolonlara tek dönüşüm |
| Ölçekleme | Model-ailesine göre ayrı, Aşama E'nin nested iç döngüsünde seçilen bir hiperparametre (v3'e artık gömülü değil); ölçekleyici seçimi ampirik test edildi | "Ölçekleme her zaman güvenlidir" varsayımı — LogReg'de RobustScaler ampirik olarak reddedildi; bulgu üzerine `v3`'ten koşulsuz `RobustScaler` adımı kaldırıldı |
| Outlier | Ölçüldü (IQR+MAD), **hiçbir satır silinmedi** | Otomatik/eşik-bazlı outlier temizliği |
| Yüksek-eksiklik filtreleme | Denendi (4 eşik × 2 model, 50 fold), **ampirik olarak reddedildi** | Varsayıma dayalı ön-filtreleme |
| Özellik seçimi | 4 bağımsız yöntemin ≥2'sinde stabil olan konsensüs havuzu, tamamen nested (fold-içi) | Tek yöntem (özellikle sadece elastic-net veya sadece GBDT) |
| Sızıntı disiplini | fit/transform sınıfları + iki ayrı sağlamlaştırılmış test kategorisi (birim test + duman testi) | Notebook'ta tek seferlik global fit |
| Dış prediktör kullanımı | Yok — proje kuralı gereği asla eklenmedi | AlphaMissense/REVEL/EVE gibi skorlar |

---

## A. Cross-Validation / Split Stratejisi

**Ne kullandık:** Dış **5-fold × 10 tekrar (Repeated Stratified)**, iç **4-fold**, `sklearn.model_selection.StratifiedGroupKFold`. Split indeksleri `data/splits/pah/` altında **bir kez** üretilip dosyaya yazıldı (61 dosya: 10 dış-tekrar + 50 iç-fold + manifest); tüm sonraki deneyler (özellik seçimi, ölçekleme karşılaştırması, preprocessing merdiveni) bu dosyaları okuyor, kendi bölmesini üretmiyor.

**Neden:**
- Yalnızca **62 benign örnek** var (372 satırın %16.7'si) → tek bölme yerine tekrarlı stratifikasyon, küçük-örneklem varyansını azaltıyor.
- 5 satırlık bir grup **öznitelik profili birebir aynı ama etiketi çelişkili** (3 patojenik/2 benign, `group_id="conflict_group_1"`, EDA'da tespit edildi) — bu grup fold sınırını geçerse aynı satırların hem train hem test'te (farklı etiketle) görünme riski var. `StratifiedGroupKFold`, `group_id` kolonuyla bu grubu her zaman aynı tarafta tutuyor.
- Sabit split bankası, farklı deneyler arasındaki performans farkının **bölme şansından değil yöntem farkından** geldiğini garanti ediyor (nested doğrulama zorunluluğu, bkz. CLAUDE.md kural #7).

**Doğrulama (kod ile, sıfır ihlal):** 369 satırın her repeat'te tam/çakışmasız kapsanması, train/test kesişimi olmaması, çelişkili-grubun 50 dış + 250 iç fold kombinasyonunun tamamında aynı tarafta kalması, iç fold'ların kendi dış fold'unun eğitim alt-kümesi olması.

**Karşılaştırma sorusu takıma:** Diğer panellerde benign/patojenik oranı ne kadar dengesiz? Çelişkili-etiket grubu (aynı profil, farklı etiket) var mı — varsa `GroupKFold` kullanıyor musunuz?

---

## B. Eksik Veri (Missingness) Stratejisi

**Ne kullandık — üç ayrı bileşen:**

| Bileşen | Ne yapar | Nerede kullanıldı |
|---|---|---|
| `BlockMissingIndicator` | Bir kolon **grubunun tamamının** eksik olduğu satırlar için tek 0/1 gösterge | `al_all_missing` (334 `AL_` kolonunun tamamı eksik, 92/369 satır), `ek_cat_block_missing` |
| `SelectedMissingIndicators` | Config'den verilen **açık liste** için gösterge (istatistiksel olarak p<0.05 çıkan alt-küme) | 1 temsilci kolon (`AL_296_missing` — sonradan bağımsızlık testiyle elendi, aşağıya bkz.) |
| `ConstantFillImputer` / `MedianImputerWithIndicator` | Kolon tipine göre sıfır ya da eğitim-fold medyanı | `AL_` → sıfır (frekans-tipi, "hiç gözlenmeme" gerçek bir değer), `EK_3` (%47.85 eksik, skor-tipi) → medyan |

**Neden bu üçlü ayrım:**
- **Kör/global işlem yasak** (CLAUDE.md kural #4): 334 `AL_` kolonuna otomatik bire-bir eksiklik göstergesi eklemek yerine, yalnızca EDA'da istatistiksel olarak anlamlı çıkan az sayıda gösterge (toplam 4 yeni kolon) kullanıldı.
- Doldurma stratejisi kolonun **anlamına** göre seçildi: `AL_` frekans/oran-tipi kolonlarda "popülasyon veritabanında hiç gözlenmeme" gerçek bir sinyal olabileceği için sıfıra yakın doldurma savunulabilir; ama bu varsayım kesin değil, bu yüzden **hem sıfır-dolu hem ham-NaN versiyonu aynı kod yoluyla üretilebiliyor** (ağaç modelleri NaN'ı kendi işleyebilir).
- `EK_3` bir korunmuşluk skoru — sıfır doldurma yanlış bir sinyal verir, medyan daha nötr. Medyan **yalnızca eğitim fold'undan** hesaplanıyor (sızıntı testiyle doğrulandı).

**Ampirik olarak reddedilen alternatif — yüksek-eksiklik kolon filtreleme:** CFTR panelinden gelen yöntemle, %70/80/90/95 eksiklik eşiklerinde `AL_` kolon filtrelemesi 4 eşik × 2 model × 50 dış fold ile test edildi. Sonuç: etki gürültü seviyesinde ve model-ailesine göre yön değiştiriyor (ExtraTrees +0.13..+0.21pp, LogReg −0.18..−0.64pp) — **net kazanç yok, final pipeline'a alınmadı.** Bu artık varsayım değil, ölçülmüş bir ret.

**Bilinen açık risk — kaynak-provenance confound:** `al_all_missing=1` olan 89 satırın **%100'ü** hem `CAT_1` hem `CAT_2` (kaynak/popülasyon etiketleri) açısından da eksik (Cramér's V=0.755 / 0.451, p<10⁻¹³). Bu, göstergenin gerçek bir biyolojik sinyal mi (popülasyonda hiç gözlenmeme → patojenite, ACMG PM2) yoksa bir veri-toplama/kaynak artefaktı mı olduğunu ayırt edemiyor. **Özellik çıkarılmadı**, Aşama F adversarial validation'a öncelikli test noktası olarak bağlandı.

**Karşılaştırma sorusu takıma:** Diğer panellerde blok-seviye eksiklik var mı (bir grup kolonun tamamının aynı satırlarda eksik olması)? Varsa aynı confound riskini (kaynak/provenance ile örtüşme) kontrol ettiniz mi?

---

## C. Kategorik Değişken Kodlama Stratejisi

**Ne kullandık:**

| Sınıf | Kolonlar | Mantık |
|---|---|---|
| `NominalOneHotEncoder` | `CAT_3/4/5` (4 seviye, homozigot genotip), `AA_1/2` (20/21 seviye) | Düşük kardinalite, heterozigot form yok → additive/ordinal kodlama biyolojik olarak anlamsız |
| `FrequencyEncoder` | `CAT_1` (24 seviye, gnomAD alt-popülasyonu), `CAT_2` (7 seviye, AllofUs) | Yüksek kardinalite; `fit`'te `y` (Label) **hiç kullanılmıyor** — yalnızca eğitim-fold'daki göreli frekans öğreniliyor |

**Neden:** Naif target/mean encoding, fold içindeki etiket bilgisini doğrudan kodlamaya sızdırır — klasik bir leakage kaynağı. Frekans kodlaması etiketten tamamen bağımsız olduğu için bu riski yapısal olarak ortadan kaldırıyor; `strategy="native"` seçeneğiyle CatBoost gibi native-categorical destekleyen modeller için `category` dtype'a da çevrilebiliyor (model seçimine bırakıldı).

**Bilinen açık nokta — CAT_1 multi-hot ihtiyacı:** 5 satırda (%1.35) `CAT_1` gerçekten çok-değerli (`"gnomADe_AFR&gnomADe_AMR&...&gnomADe_SAS"`, 9 alt-popülasyonun birleşimi). `FrequencyEncoder` bunu tek atomik "sözde kategori" gibi ele alıyor — semantik bilgi kaybı var ama düşük satır sayısı nedeniyle bu turda düzeltme **kod değişikliği olarak uygulanmadı, yalnızca belgelendi** (gerekli `MultiHotEncoder` tasarımı rapor edildi, onay bekliyor).

**Karşılaştırma sorusu takıma:** Panelinizde `&`/çoklu-değer içeren kategorik hücreler var mı (ör. birleşik popülasyon etiketleri)? Frequency/one-hot kodlayıcınız bunu sessizce yanlış mı işliyor?

---

## D. Sayısal Dönüşüm & Ölçekleme Stratejisi

**Ne kullandık — kolon tipi, veriden yapısal olarak (etikete bakmadan) sınıflandırılarak seçildi (`classify_al_columns`):**

| Kolon tipi | Sayı | Dönüşüm | Neden |
|---|---|---|---|
| Sabit-değerli `AL_` | 90 | Yok | Varyans yok, dönüşüm anlamsız |
| Frekans-tipi `AL_` | 162 | `log1p` | Ağır sağa çarpık, sıfır-şişkin |
| Oran-tipi [0,1] `AL_` | 82 | `LogitTransformer` (eps=1e-6) | [0,1]→sınırsız ölçek; epsilon-offset tam 0/1'de ±inf patlamasını önlüyor |
| Sınırsız-ölçekli `EK_` (1,2,7,8,9) | 5 | `RankQuantileHarmonizer` | Farklı native ölçekler (ör. `EK_2`: [-10.8,6.17]) → eğitim-fold quantile'larına göre ortak [0,1]'e getiriliyor, NaN korunuyor |
| Zaten [0,1] `EK_` (4,5,6) | 3 | Yok | Ek harmonizasyon gerekmiyor |

**Ölçekleme — model-ailesine göre ayrı, Aşama E'ye ertelenmiş bir karar:** Ağaç modelleri (ileride) ham/NaN-toleranslı veriyi kullanacak; ölçek-duyarlı modeller (LogReg, SVM) için `ColumnwiseScaler` (RobustScaler/QuantileTransformer sarmalayıcı) hazır ama artık `v3`'e (veya `pah_pipeline_transformed.joblib`'e) gömülü değil — aşağıdaki ampirik bulgu üzerine çıkarıldı, model-ailesine göre Aşama E'nin nested iç döngüsünde ayrı bir hiperparametre olarak seçilecek. `fit` yalnızca eğitim fold'unda çağrılabiliyor.

**Ampirik doğrulama (Tamamlayıcı Deney Turu) — beklenmedik bulgu:** "Ağaç modelleri ölçekten bağımsız, RobustScaler doğrusal modeller için güvenli" varsayımı 50 dış fold üzerinde 3 sabit model ile test edildi:

| Varyant | ExtraTrees F1 | LogReg F1 |
|---|---|---|
| Ölçeksiz | 0.9115 | 0.8421 |
| StandardScaler | 0.9111 | **0.8635** |
| RobustScaler | 0.9133 | **0.6898** |

- ExtraTrees'in ölçeğe kayıtsız olduğu doğrulandı (varsayım teyit).
- LogReg'de `RobustScaler`, ölçeksiz duruma göre bile **belirgin şekilde kötü** (F1 −15 puan) — `StandardScaler` ise beklendiği gibi iyileştiriyor. **Bu, "RobustScaler her zaman güvenli" varsayımını çürüttü.**
- Ek olarak, ölçekleyici seçim script'inin otomatik mantığının (model ayrımı yapmadan max-F1'e göre seçim) bu LogReg çöküşünü maskeleyen bir metodolojik zafiyet taşıdığı da tespit edildi — kendi hatamız olarak açıkça raporlandı.

**Sonuç/karar:** Ölçekleyici seçimi kalıcı olarak değiştirilmedi (bu tur "modelleme yapma" kuralına tabiydi) ama **Aşama E'nin nested iç döngüsünde model-ailesine göre ayrı bir hiperparametre olarak ele alınması** notu kesinleşti — sabit/tek ölçekleyici varsayılmayacak.

**Karşılaştırma sorusu takıma:** Ölçekleyici seçiminizi (Standard/Robust/Quantile) model ailesine göre ayrı mı test ettiniz, yoksa tek bir "genel kabul görmüş" seçimi mi tüm modellere uyguladınız? Bizim veride bu varsayım yanlış çıktı.

---

## E. Aykırı Değer (Outlier) Stratejisi

**Ne kullandık:** 253 sayısal kolonda formal IQR + MAD/modified-z taraması, satır-bazlı outlier yükü hesaplandı, benign(n=62) vs patojenik(n=310) karşılaştırması (Mann-Whitney U) yapıldı.

**Bulgular:** IQR ortalama oran %8.9/satır (medyan %5.3), MAD ortalama %17.4 (medyan %12.7). Benign/patojenik karşılaştırması **tutarsız**: IQR açısından anlamlı fark var (p=0.0086), MAD açısından yok (p=0.1456) — bu tutarsızlık gizlenmeden dürüstçe raporlandı.

**Karar:** **Hiçbir satır silinmedi.** Outlier'lar yalnızca ölçüldü ve modelleme aşamasına (Aşama E, sağlam/robust model seçimi ya da winsorization gibi kararlar için) bir risk notu olarak taşındı — bu turun kapsamı "ölç ve belgele"ydi, "temizle" değil.

**Neden satır silinmedi:** n=369 gibi küçük bir örneklemde, özellikle 62 benign örneğin bir kısmı gerçek biyolojik uç-değerler olabilir (nadir patojenik varyantların aykırı skorlar alması beklenir) — istatistiksel "aykırı" ile "hatalı veri" aynı şey değil; körü körüne silme gerçek sinyali kaybetme riski taşır.

**Karşılaştırma sorusu takıma:** Panelinizde outlier temizliği (satır silme/winsorization) uyguladınız mı? Uyguladıysanız, benign/patojenik gruplar arasında outlier oranı farklı mı (bizde tutarsız/zayıf bir fark var) — temizlik bu farkı yapay olarak silikleştirebilir mi diye kontrol ettiniz mi?

---

## F. Özellik Seçimi Stratejisi

**Ne kullandık — 4 bağımsız yöntem, her biri 50 dış fold'un yalnızca eğitim kısmında (tamamen nested):**

| Yöntem | Araç | Fold-içi "seçildi" kuralı |
|---|---|---|
| Mutual information | `mutual_info_classif` | O fold'da en yüksek skorlu ilk 30 |
| Elastic-net stability selection | `LogisticRegression(elasticnet)`, 10 bootstrap | Bootstrap'ların ≥%50'sinde katsayı≠0 |
| Model-tabanlı (GBDT) | LightGBM (yardımcı/diagnostic, 100 ağaç) | En yüksek `feature_importance` ilk 30 |
| Permutation importance | Aynı fold'un LightGBM'i, dış test fold'unda ölçüldü | Ortalama önem > 0 |

**Stabilite kuralı:** 50 dış tekrarın **≥%60'ında** seçilen özellik, o yöntem için "stabil" sayılıyor. **Final havuz kuralı:** bir özellik **en az 2 bağımsız yöntemde** stabil ise havuza giriyor (24 özellik, 405 aday içinden).

**Neden tek yöntem yeterli değildi:** Elastic-net'in kendi eşiği tek başına çok gevşek (405 özelliğin %41'i geçiyor) — n≈p (369 satır / ~405 özellik) az-belirlenmiş rejiminde L1/L2 cezası yeterince seyreltici olamıyor, gürültü gerçek sinyal gibi görünebiliyor. Bu yüzden **konsensüs (≥2 yöntem)** şartı getirildi.

**Ekstra titizlik — bağımsızlık kontrolü:** GBDT importance ve permutation importance aynı fold-içi LightGBM'den türediği için **istatistiksel olarak bağımsız değil**. Bu, "≥2 yöntemde stabil" kuralının bazı özellikler için aslında "1.5 bağımsız yöntem" anlamına gelebileceği fark edildi ve ayrıca raporlandı: final 24 özelliğin 21'i (%87.5) bu bağımlı ikilinin ötesinde bağımsız bir üçüncü yöntemle (MI veya elastic-net) de teyitli; 3'ü (`AL_318`, `AL_12`, `AL_22`) yalnızca bağımlı ikiliyle stabil — bunlar daha temkinli değerlendirilmeli notuyla işaretlendi.

**Doğrulama — döngüsellik/meta-prediktör kontrolü:** Grup ablation (`EK_` çıkarılınca AUC −0.040, en büyük kayıp) ve izole-grup modelleri (hiçbir grup tek başına 0.673 AUC'yi geçmiyor) iki bağımsız yöntemle çapraz kontrol edildi — veri setinde gizli, döngüsel bir meta-prediktör (örn. sızmış bir dış skor) olmadığı teyit edildi.

**Karşılaştırma sorusu takıma:** Özellik seçiminiz kaç bağımsız yöntemin konsensüsüne dayanıyor? Kullandığınız yöntemler (ör. GBDT importance + SHAP) gerçekten bağımsız mı, yoksa aynı modelin iki farklı çıktısı mı (bizde olduğu gibi kısmen bağımlı çıkabiliyor)?

---

## G. Sızıntı Önleme (Leakage) — Genel İlkeler

Tüm ön işleme sınıfları sklearn `fit`/`transform` arayüzünde yazıldı; `fit` **yalnızca kendisine verilen (eğitim) veriden** öğreniyor:

- Medyan doldurma → yalnızca eğitim fold'unun medyanı (test: `test_median_imputer_uses_training_fold_median_not_validation_data`)
- Frekans kodlama → yalnızca eğitim fold'unun kategori frekansları (test: sızıntı testi mevcut)
- Rank/quantile harmonizasyonu → yalnızca eğitim fold'unun quantile'ları
- Ölçekleyici → yalnızca eğitim fold'unda fit

**Bilinen/belgelenen kısmi sızıntı riskleri (özellikle işaretlendi, gizlenmedi):**
1. **Korelasyon-kümesi grup-özet özellikleri (v3, 48 kolon):** Satır-bazlı hesaplama sızıntısız, ama küme **üyeliği** (hangi `AL_` kolonlarının birlikte özetleneceği) tüm veri setinde (Label kullanılmadan ama tüm satırlarla) hesaplandı — düşük şiddetli, yapısal bir sızıntı. Aşama E'de tam titizlik istenirse fold-içi yeniden hesaplanmalı.
2. **One-hot kategori kümesi:** Hangi kategori değerlerinin göründüğü eğitim-fold'undan öğreniliyor (`fit`), ama mevcut `v3.parquet` demo/referans amaçlı tüm veri setinde fit edildi — sınıfın kendisi fold-içi çağrılabilir durumda.

Bu ayrım bilinçli: **kodun kendisi** (sınıflar) sızıntısız tasarlandı ve test edildi; **bazı statik referans dosyaları** (v3.parquet gibi) demo/keşif amaçlı tüm veri üzerinde üretildi ve bu açıkça etiketlendi — Aşama E'de gerçek performans ölçümü bu dosyaları değil, fold-içi yeniden-fit edilmiş hallerini kullanacak.

**Karşılaştırma sorusu takıma:** Statik/demo veri dosyalarınız (tüm veride fit edilmiş) ile fold-içi nested versiyonlarınız arasında net bir ayrım var mı? Hangisinin "gerçek performans ölçümü" için kullanılacağı kod/dokümantasyonda açık mı?

---

## H. Kod Mimarisi — fit/transform + Pipeline Serileştirme

**Ne kullandık:** Tüm ön işleme mantığı `src/genova/pah/` altında test edilebilir, `BaseEstimator, TransformerMixin` tabanlı sınıflar olarak yazıldı (notebook değil). Aşama B'nin sınıfları iki `sklearn.pipeline.Pipeline`'da birleştirilip `joblib` ile serileştirildi:
- `pah_pipeline_tree.joblib` — ağaç modelleri kolu (native categorical, ölçeksiz)
- `pah_pipeline_transformed.joblib` — dönüştürülmüş kol (dönüşüm+harmonizasyon+kodlama, ölçeksiz — bkz. bölüm D'nin güncellenmiş bulgusu; eski adı `pah_pipeline_scaled.joblib`'ti, gömülü `RobustScaler` kaldırılınca yeniden adlandırıldı)

Her ikisinde de **sınıflandırıcı yok** (`any(hasattr(step, "predict") ...)` → `False`, doğrulandı) — yalnızca `fit`/`transform`.

**Neden `Pipeline` (`ColumnTransformer` değil):** Özel transformer'lar tüm-dataframe-girdi/tüm-dataframe-çıktı şeklinde tasarlandı (kolon alt-kümesi değil, satır-bazlı/grup-bazlı mantık içeriyorlar) — `ColumnTransformer`'ın kolon-bölme modeli buraya uymuyor.

**Karşılaştırma sorusu takıma:** Ön işleme kodunuz sklearn `Pipeline`/`ColumnTransformer` uyumlu mu, yoksa serbest fonksiyonlar mı? Serileştirilmiş (joblib/pickle) bir ön işleme artefaktınız var mı, yoksa her deneyde yeniden mi çalıştırıyorsunuz?

---

## I. Deneysel Doğrulama Metodolojisi (CFTR'den Öğrenilen Yaklaşım)

Bu, PAH'a özgü değil ama diğer panellere taşınabilir bir **metodoloji** stratejisi:

1. **Sabit/ayarlanmamış diagnostic modeller** (`LogisticRegression(class_weight="balanced")`, `ExtraTreesClassifier(n_estimators=200, random_state=42)`, `DummyClassifier`) — hiçbir hiperparametre taraması yapılmadan, yalnızca ön işleme kararlarını A/B test etmek için kullanıldı. Amaç "en iyi model" değil, "bu ön işleme adımı gerçekten katkı sağlıyor mu?" sorusuna nicel cevap.
2. **Tek-değişkenli merdiven (ladder) yaklaşımı:** v1→v2→v3→v4 zinciri 6 basamağa (M-0..M-5) bölündü, her basamak **tek bir bileşen** ekliyor, hepsi aynı split bankasında (50 dış fold) ölçüldü. Bu, "toplam pipeline iyi çalışıyor" ifadesini "şu spesifik adım şu kadar katkı sağlıyor, şu adım katkı sağlamıyor" seviyesine indirgedi.
   - En büyük katkı: `AL_` sıfır-doldurma + `EK_3` medyan-doldurma (LogReg F1 +5.71 puan) — ham NaN'ı kullanılabilir hale getirmek, dönüşüm/ölçeklemeden çok daha belirleyici çıktı.
   - Ölçülebilir katkı sağlamayan adım: yalnızca gösterge eklemek (doldurma olmadan) — gösterge kendi başına değil, doldurmayla **birlikte** değer katıyor.
3. **Varsayımları ampirik test etmek, literatüre güvenmemek:** Ölçekleme (bölüm D) ve yüksek-eksiklik filtreleme (bölüm B) kararları "genel kabul" yerine bu projenin kendi verisinde, split bankasıyla test edildi — ikisinde de literatür-kaynaklı ilk varsayım kısmen/tamamen revize edildi.
4. **"Neden alternatifi seçmedim" dokümantasyonu:** Her büyük kararın yanına, reddedilen alternatiflerin gerekçesi de yazıldı (bkz. `00_DUZELTME_OZETI_PAH.md` ve `04_PREPROCESSING_MERDIVENI_PAH.md` sonları) — yalnızca "ne yaptık" değil "ne yapmadık ve neden" kayıt altında.

**Karşılaştırma sorusu takıma:** Panelinizde ön işleme kararlarını (doldurma, kodlama, ölçekleme) sabit/basit modellerle ayrı ayrı A/B test ettiniz mi, yoksa doğrudan final modelin performansına mı baktınız (bu durumda hangi adımın ne kattığını ayırt etmek zor olabilir)?

---

## J. Henüz Yapılmayanlar / Açık Noktalar (Aşama E-F'ye Bırakılan)

Şeffaflık için: bu strateji dokümanı yalnızca ön işlemeyi kapsıyor, aşağıdakiler **henüz yapılmadı**:

- **Sınıf dengesizliği/prior düzeltmesi:** Eğitimde %83.3 patojenik, final test beklentisi ≈%28.6 — bu farkı ele alacak kalibrasyon/eşik stratejisi Aşama E/F'de belirlenecek, şu ana kadar hiçbir eşik/kalibrasyon kararı verilmedi.
- **Adversarial validation:** `al_all_missing`/`CAT_1`/`CAT_2` confound riskini (bölüm B) train/test dağılım farkı açısından test edecek adım, Aşama F'ye planlandı.
- **`CAT_1` multi-hot düzeltmesi:** Belgelendi, kod değişikliği olarak uygulanmadı (bölüm C).
- **Korelasyon-kümesi üyeliğinin fold-içi yeniden hesaplanması:** Şu an tüm veri setinde sabit (bölüm G, risk #1) — Aşama E'de tam titizlik istenirse ele alınmalı.
- **Model seçimi ve hiperparametre optimizasyonu:** Hiç başlamadı.

**Karşılaştırma sorusu takıma:** Panelinizde final test setindeki sınıf dağılımı da eğitimden farklı mı? Prior-shift/kalibrasyon stratejinizi ne zaman (hangi aşamada) belirlemeyi planlıyorsunuz?
