# PAH Paneli — Aşama B: Ön İşleme Politikası ve Gerekçeleri

Kod: `src/genova/pah/{schema,missingness,encoding,transforms}.py`. Testler: `tests/test_preprocessing_pah.py` (21 test, hepsi geçiyor) + her modülde doctest (tümü geçiyor). Tüm sınıflar sklearn `fit`/`transform` arayüzünde; `fit` yalnızca kendisine verilen veriden (eğitim fold'u) öğrenir — nested CV içinde fold-güvenli çağrılabilir.

---

## 1. Şema/sentinel doğrulayıcı (`schema.py`)

- `validate_schema`: zorunlu kolonlar, `Variant_ID` benzersizliği, `Label∈{0,1}`, grup kolon sayıları (`AL_`=334, `CAT_`=6, `EK_`=9, `AA_`=2) kontrol eder, hata fırlatmaz — bir rapor dict'i döner ki çağıran taraf loglayabilsin/karar verebilsin.
- `scan_sentinels`: -999/-9999/9999/999/-1 için körü körüne "yok" varsaymak yerine gerçekten tarar, hangi kolonda kaç kez geçtiğini raporlar. **Gerekçe:** CLAUDE.md "Bu panelde literal sentinel değer... yok" diyor ama Aşama A.1 talimatı "kendi taramanla yeniden doğrula, körü körüne varsayma" diyor — kod, ham veri üzerinde çalıştırıldığında gerçekten 0 sentinel buluyor (doğrulandı), ama fonksiyon her yeni veri versiyonunda yeniden çalıştırılabilir kalıyor.

## 2. Eksiklik işleyici (`missingness.py`)

| Sınıf | Ne yapar | Neden |
|---|---|---|
| `BlockMissingIndicator` | Bir kolon grubunun **tamamının** eksik olduğu satırlar için tek 0/1 gösterge | EDA A.3.2: 92 satırda (`AL_` tümü) ve 11 satırda (`EK_` hariç EK_3 + `CAT_3/4/5` tümü) blok-seviye eksiklik var — bunu tek satır/tek gösterge olarak modele vermek, 334 ayrı sütun eklemekten hem daha az aşırı-uyum riskli hem de daha yorumlanabilir. |
| `SelectedMissingIndicators` | Config'den verilen **açık kolon listesi** için gösterge üretir | Kesin kural #3 ("kör/global işlem yok") gereği: 334 `AL_` kolonuna otomatik gösterge eklenmez. Kolon listesi, `reports/tables/missingness_label_association.csv`'deki p<0.05 sonuçlarından, blok-tekrarlarını eleyerek elle seçilir (bkz. Aşama C v2 config'i). |
| `ConstantFillImputer` | `strategy="zero"` ile sıfır doldurma, `strategy="none"` ile ham NaN korunur | `AL_` frekans/oran-tipi kolonlar için "popülasyon veritabanında hiç gözlenmeme = aşırı nadirlik" gerekçesiyle sıfıra yakın doldurma savunulabilir (CLAUDE.md), ama bu varsayım kesin değil — bu yüzden **hem sıfır-dolu hem ham-NaN versiyonu aynı kod yoluyla üretilebiliyor**, ağaç tabanlı modeller (ileride) NaN'ı kendi başına yorumlayabildiği için ham hali de saklanıyor. |
| `MedianImputerWithIndicator` | Gösterge + eğitim-fold medyanıyla doldurma | `EK_3` için: %47.85 eksiklik oranı çok yüksek, sıfır doldurma anlamsız (EK_3 bir korunmuşluk skoru, 0 gerçek bir değer aralığı içinde yanlış bir sinyal verir) — medyan daha güvenli nötr bir doldurma. Medyan **yalnızca `fit()`'e verilen (eğitim) veriden** hesaplanıyor, testte doğrulandı (`test_median_imputer_uses_training_fold_median_not_validation_data`). |

**Seçilen ek eksiklik göstergeleri (v2 için):** EDA A.3.2/A.3.3 bulgusuna göre, `AL_` grubunda 14 benzersiz eksiklik deseni var; en anlamlısı (p=0.000033, n=126 satır, "AL_ tümü eksik" 92-satırlık bloktan daha geniş bir alt-küme) tek bir temsilci kolonun (`AL_296`) göstergesiyle yakalanıyor — bu deseni paylaşan tüm kolonlar birebir aynı 0/1 vektörünü ürettiği için tekrar eklemek bilgi katmıyor. Toplamda v2'ye eklenen gösterge sayısı: `al_all_missing` (blok) + `ek_cat_block_missing` (blok) + `EK_3_missing` (medyan-imputer içinde otomatik) + 1 temsilci `AL_296_missing` = **4 yeni sütun**, 334 değil.

## 3. Kategorik kodlayıcı (`encoding.py`)

| Sınıf | Uygulandığı kolonlar | Gerekçe |
|---|---|---|
| `NominalOneHotEncoder` | `CAT_3/4/5`, `AA_1/2` | Düşük kardinalite (4 ve 20 seviye), heterozigot form yok (EDA A.6) → additive/ordinal kodlama biyolojik olarak anlamsız, one-hot doğru seçim. `handle_unknown="ignore"` ile görülmeyen kategori sıfır-vektöre düşer, hata fırlatmaz. |
| `FrequencyEncoder` | `CAT_1`, `CAT_2` | 24 ve 7 seviye, `Label`'a bakmadan yalnızca eğitim-fold'daki göreli frekansı öğrenir (`fit`'te `y` parametresi hiç kullanılmıyor) — naif target encoding sızıntısından kaçınmak için bilinçli tercih. `strategy="native"` alternatifi CatBoost gibi native-categorical destekleyen modeller için kolonu `category` dtype'a çevirip bırakıyor; hangi stratejinin kullanılacağı ileride model seçimine göre config'den belirlenecek. |

**CAT_2 özel notu:** EDA A.5, `CAT_2` (AllofUs) alt-gruplarında patojenik oranında büyük hücrelerde bile ~9 puanlık sapmalar buldu — bu sinyal gerçek olabilir de olmayabilir de (kaynak-provenance riski). Frekans kodlaması bu belirsizliği çözmüyor ama en azından **sızıntısız** taşıyor; ham target-encoding kesinlikle kullanılmadı.

## 4. Dönüşüm/ölçekleme (`transforms.py`) — yalnızca ölçek-duyarlı model kolu için

| Sınıf | Uygulandığı kolonlar | Gerekçe |
|---|---|---|
| `Log1pTransformer` | 162 frekans-tipi `AL_` kolonu | Ağır sağa çarpık, sıfır-şişkin dağılım (EDA A.6) → log1p sıfırı koruyarak çarpıklığı azaltır. |
| `LogitTransformer` (eps=1e-6) | 82 oran-tipi [0,1] `AL_` kolonu | [0,1] sınırlı oranlar için logit, sınırsız ölçeğe taşır; epsilon-offset tam 0/1 değerlerinde ±inf patlamasını önler (test: `test_logit_transformer_does_not_blow_up_at_exact_0_or_1`). |
| — (dönüşümsüz) | 90 sabit-değerli `AL_` kolon | EDA A.6: bu kolonlar gözlendiğinde tek bir değer alıyor (89'u `1.0`, `AL_185` `264690.0`) — varyans yok, dönüşüm anlamsız. Bilgi içerikleri zaten eksiklik göstergelerinde. |
| `RankQuantileHarmonizer` | `EK_1,2,7,8,9` (sınırsız native ölçek) | EDA A.6: bu kolonlar farklı, sınırsız ölçeklerde (`EK_2`: [-10.8,6.17], `EK_9`: [-7.84,11.93]) — kör ortalama/karşılaştırma anlamsız. Eğitim-fold quantile'larına göre ortak [0,1] ölçeğine getiriliyor, NaN korunuyor (test: `test_rank_quantile_harmonizer_preserves_nan`). |
| — (dönüşümsüz) | `EK_4/5/6` | Zaten native [0,1] aralığında, ek harmonizasyon gerektirmiyor. |
| `ColumnwiseScaler` (`RobustScaler`/`QuantileTransformer` sarmalayıcı) | Log1p/logit/harmonize edilmiş tüm sürekli kolonlar | Yalnızca ölçek-duyarlı modeller (örn. lojistik regresyon, SVM) için ayrı bir kol; ağaç tabanlı modeller (LightGBM/XGBoost/CatBoost) ham (log/logit dönüşümsüz de olabilir) NaN-toleranslı veriyi kullanacak — CLAUDE.md'nin "ağaç modelleri hattına karıştırılmayacak" kuralına uyulmuştur. `fit` yalnızca eğitim fold'unda çağrılır; arayüz ileride fold-içi kullanılabilir. |

`classify_al_columns` fonksiyonu bu üç alt-grubu (`constant`/`ratio_type`/`frequency_type`) veriden **yapısal olarak** (Label'a bakmadan) çıkarır — bu nedenle tam veri setinde veya herhangi bir fold'da çağrılması sızıntı oluşturmaz (etiket-bağımsız bir sınıflandırma).

## 5. Test kapsamı

`tests/test_preprocessing_pah.py` — 21 test:
- Şema: geçerli/geçersiz şema tespiti, sentinel tarama (var/yok).
- Eksiklik: blok gösterge doğruluğu, bilinmeyen kolon hatası, sıfır/ham-NaN stratejileri, **medyan doldurmanın yalnızca eğitim fold'undan öğrenildiği** (sızıntı testi — validation'daki uç değer medyanı etkilemiyor).
- Kodlama: one-hot boyutu, görülmeyen kategori sıfır-vektörü, frekans kodlamasının eğitim-fold frekanslarını kullandığı (sızıntı testi), native-categorical dtype dönüşümü.
- Dönüşüm: log1p/logit doğruluğu, logit'in 0/1 sınırında patlamaması, rank/quantile'ın [0,1] aralığına düştüğü ve NaN'ı koruduğu, ölçekleyicinin yalnızca eğitim fold'undan fit edildiği (sızıntı testi), `classify_al_columns`'ın üç tipi doğru ayırdığı.

Ayrıca ham veri (`data/raw/YARISMA_TRAIN_PAH.csv`) üzerinde uçtan uca bir duman testi (smoke test) çalıştırıldı: şema geçerli, 0 sentinel, 90/82/162 sabit/oran/frekans ayrımı, 92 satırlık `AL_` bloğu doğru tespit edildi, tüm dönüşümler sonrası `inf`/kalan `NaN` yok — Aşama C'deki versiyon üretim script'i bu bileşenleri doğrudan kullanacak.

---

## Ek — Aşama D Düzeltme Turu Notları

### 6. v3 Ek Kolonlar — Tam Liste ve Gerekçe (Görev 2)

`set(v3.columns) - set(v2.columns)` ile hesaplanan tam fark: **105 yeni kolon**. Ayrıca v2'de var olup v3'te bulunmayan 5 kolon var: `AA_1, AA_2, CAT_3, CAT_4, CAT_5` (bunlar one-hot açılımlarıyla yer değiştirdi, düşmedi). 352 kolon **isim olarak aynı** kalıyor ama bir kısmının **değeri** dönüştürüldü (aşağıda ayrıca not edildi).

| Grup | Kolon sayısı | Kaynak | Fold-güvenlik durumu |
|---|---|---|---|
| **One-hot açılımlar** (`CAT_3_*`, `CAT_4_*`, `CAT_5_*`, `AA_1_*`, `AA_2_*`) | 57 (5+5+5+21+21) | `NominalOneHotEncoder` | Satır-bazlı (her satırın kendi kategori değerinden üretilir); kategori kümesinin **hangi değerleri içerdiği** eğitim-fold'undan öğrenilir (`fit`), sızıntı riski yalnızca kategori kümesi seçiminde — sınıf zaten fold-içi çağrılabilir (bkz. Bölüm 3), v3.parquet dosyası tam veri setinde fit edildi (mevcut sızıntı notuyla tutarlı). |
| **Korelasyon-kümesi grup-özet özellikleri** (`AL_cluster{id}_min/max/median/n_positive`) | 48 (12 küme × 4 istatistik) | `build_v3` içindeki grup-özet bloğu | **Satır-bazlı** (`block.min(axis=1)` vb. — yalnızca o satırın kendi `AL_` değerlerinden hesaplanır, eğitim-fold istatistiği FIT EDİLMİYOR); bu haliyle kendi başına sızıntı riski taşımaz. **Ancak** hangi `AL_` kolonlarının hangi kümeye ait olduğu (`reports/tables/AL_correlation_clusters.csv`, |r|>0.5 eşiği), Aşama A.7'de **tüm 372/369 satır** üzerinde hesaplanan global Spearman korelasyonundan geliyor — Label'dan bağımsız (etiket kullanılmadı) ama test-fold satırlarının değerlerini de gördü. Bu, düşük şiddetli ama gerçek bir "yapısal" sızıntı: küme **üyeliği** (hangi `AL_` kolonlarının birlikte özetleneceği) train/test ayrımı gözetmeden seçildi. Aşama E'de tam titizlik isteniyorsa küme yapısı da fold-içi yeniden hesaplanmalı; şimdilik `reports/03_OZELLIK_SECIMI_PAH.md`'deki genel sızıntı notuna ek olarak burada da işaretleniyor. |

**352 ortak-isimli kolon içindeki değer dönüşümleri (yeni kolon değil, ama gözden kaçabilir):**
- 162 frekans-tipi `AL_` kolonu: `log1p` uygulandı (v2'de ham/sıfır-dolu değer, v3'te log1p'li değer — isim aynı).
- 82 oran-tipi `AL_` kolonu: epsilon-ofsetli `logit` uygulandı.
- `EK_1,2,7,8,9`: rank/quantile harmonizasyonu ile [0,1]'e taşındı.
- **`CAT_1`, `CAT_2`: v2'de `category` dtype (native, kodlanmamış); v3'te `FrequencyEncoder` ile sayısal frekans değerine dönüştürüldü.** İsim aynı kaldığı için `set()` farkı bunu YAKALAMAZ — bu notla açıkça belirtiliyor.
- ~~Tüm sürekli kolonlar en son `RobustScaler` ile ölçeklendi.~~ **Güncelleme:** Bu adım `v3`'ten kaldırıldı (bkz. §11'in güncellenmiş sonucu ve §12) — v3 artık ölçeksiz. Bu satır, o dönemde v3'ün gerçek içeriğini yansıtan tarihsel bir kayıt olarak korunuyor.

### 7. CAT_1 Çok-Değerlilik — Encoding Düzeltme İhtiyacı (Görev 3)

`reports/01_EDA_RAPORU_PAH.md`'deki düzeltme notunda tespit edildiği gibi, `CAT_1`'de 5 satır (`VAR_002655, VAR_002560, VAR_003161, VAR_002835, VAR_002589`) gerçekten çok-değerli: `"gnomADe_AFR&gnomADe_AMR&...&gnomADe_SAS"` (9 alt-popülasyonun TÜMÜNÜN birleşimi).

**Önceki davranış (düzeltildi):** `CAT_1`, `CAT_3/4/5`/`AA_1/2`'nin aksine `NominalOneHotEncoder`'a değil, `FrequencyEncoder`'a veriliyordu. `FrequencyEncoder.fit`, bu `&`-birleşik string'i `value_counts()` ile **tek, atomik bir kategori** olarak sayıyordu (369 satırda 5/369 ≈ %1.35 frekans) — kod hata vermiyordu, geçerli bir sayısal değer üretiyordu, ama semantik olarak yanlıştı: bu 5 satırın aslında `gnomADe_AFR`, `gnomADe_AMR`, ..., `gnomADe_SAS` etiketlerinin **her birine birden** ait olduğu bilgisi kayboluyordu; bunun yerine diğer 24 kategoriyle ilgisiz, kendi başına 25. bir "sözde kategori" gibi ele alınıyordu.

**UYGULANDI (frekans-ortalaması yöntemiyle):** `src/genova/pah/encoding.py`'ye `MultiValueFrequencyEncoder(FrequencyEncoder)` eklendi. Tasarım, ilk düşünülen **multi-hot** (her `gnomADe_*` etiketi için ayrı 0/1 sütun) yaklaşımından **farklı**, bilinçli bir tercih: `CAT_1`'i tek bir sürekli kolon olarak koruyup, çok-değerli bir satır için bileşenlerinin **eğitim-fold'undaki tekil (single-valued satırlardan hesaplanan) frekanslarının ortalamasını** kodlanmış değer olarak veriyor — tekil-değerli satırlar `FrequencyEncoder` ile birebir aynı davranıyor (testle doğrulandı). Bu tercih, `CAT_1`'in geri kalan pipeline'da (frekans kodlanmış tek bir sayısal kolon) taşıdığı şekli koruyor — multi-hot yaklaşımı 9 yeni sütun ekleyip yalnızca 5/369 satır (%1.35) için anlamlı olacaktı, bu da boyut artışına değmeyen bir maliyet olurdu. Görülmeyen bir bileşen 0.0 katkı sağlıyor (`FrequencyEncoder`'ın unseen-kategori davranışıyla tutarlı). Kullanıldığı yerler: `dataset_versions.py::build_v3`, `fold_features.py::build_fold_features`, `serialize_pipelines.py::build_transformed_pipeline`, `preprocessing_ladder.py::build_step` (`CAT_2` hâlâ düz `FrequencyEncoder` kullanıyor, çok-değerli satır deseni yalnızca `CAT_1`'de var). `tests/test_preprocessing_pah.py`'ye 4 yeni test eklendi (tekil-değer tutarlılığı, çok-değerli ortalama, görülmeyen bileşen, sızıntı).

### 8. v1 Kolon Aritmetiği Teyidi (Görev 5)

Kod ile doğrulandı: `assert 'CAT_6' not in v1.columns` ✅, `assert 'group_id' in v1.columns` ✅. Ham veri 353 kolon (`Variant_ID`+`Label`+334 `AL_`+6 `CAT_`+9 `EK_`+2 `AA_`) → `CAT_6` (1 kolon) düşürüldü → 352 → `group_id` (1 kolon) eklendi → **353**. `v1.parquet` da 353 kolon — "353=353" örtüşmesi **tesadüf değil, tam olarak bu iki işlemin (1 düşür + 1 ekle) birbirini götürmesinden** kaynaklanıyor; doğrulandı.

### 9. Test Sayısı Tutarlılığı (Görev 6)

**Güncelleme (Aşama E Öncesi Denetim, `reports/06_ASAMA_E_ONCESI_DENETIM_PAH.md`, Orta-1):**
Bu bölümün önceki hali "toplam 27 test — `test_preprocessing_pah.py`: 24, `test_split_bank_pah.py`: 3" diyordu. `pytest --collect-only -q` ile yeniden doğrulandı: **`test_preprocessing_pah.py` aslında 24 değil 25 test içeriyor** (önceki "21→22→24" kümülatif-ekleme anlatısının son basamağı bir eksik sayılmış — tam olarak hangi ek testin bu anlatıya dahil edilmediği geriye dönük olarak kesin belirlenemedi, bu yüzden sahte bir kesinlikle yeniden kurulmuyor; burada yalnızca güncel, doğrulanmış sayı esas alınıyor). Ayrıca Tamamlayıcı Deney Turu'nda (bkz. `03c_ID3_YONTEM_EKLEME_PAH.md`) `tests/test_id3_feature_selection_pah.py` (5 test) eklendi.

**Güncel, doğrulanmış durum (`pytest --collect-only -q tests/`):**

| Dosya | Test sayısı |
|---|---|
| `test_preprocessing_pah.py` | 25 |
| `test_split_bank_pah.py` | 3 |
| `test_id3_feature_selection_pah.py` | 5 |
| **Toplam** | **33** |

Aşağıdaki tarihsel anlatı (Aşama B'den bu yana kümülatif ekleme süreci) doğru yönde ama son
rakamı artık güncel değil; bir sayım hatası değil, zamanla kümülatif ekleme sürecinden
kaynaklanıyor — orijinal metin tarihsel kayıt olarak korunuyor: Aşama B raporundaki "21 test"
rakamı yalnızca `test_preprocessing_pah.py`'nin **ilk yazıldığı haliydi**. Sonradan ekler
oldu: (a) `encoding.py`'deki category-dtype hatasının düzeltilmesiyle bir regresyon testi
(`test_frequency_encoder_handles_category_dtype_input_column`) eklendi, (b) Aşama D'nin
`fold_features.py` fold-güvenlik testleri (`test_build_fold_features_on_real_v1_data_produces_dense_aligned_matrices`,
`test_build_fold_features_median_impute_uses_train_fold_only`) aynı dosyaya eklendi. Ayrıca
Aşama D Düzeltme Turu'nda `tests/test_split_bank_pah.py` (3 test) yeni dosya olarak eklendi.

### 10. Sonsuz (inf) Değer Taraması (Tamamlayıcı Deney Turu, Görev 2)

Tüm 6 veri seti versiyonunun (`v1`, `v2`, `v2_raw_nan`, `v3`, `v4_from_v2`, `v4_from_v3`) sayısal kolonları `np.isinf()` ile tarandı — **0 `inf`/`-inf` değeri bulundu**, hiçbir versiyonda. Özellikle `v3` (426 sayısal kolon, `log1p`/`logit` dönüşümlü `AL_` kolonlarını içeriyor) sıfır çıktı; bu, `LogitTransformer`'ın epsilon-ofsetinin (`eps=1e-6`, bkz. §4) tam 0/1 değerlerinde `log(x/(1-x))`'in ±∞'a kaçmasını gerçek veride de başarıyla önlediğinin doğrudan kanıtı — daha önce yalnızca birim testle (`test_logit_transformer_does_not_blow_up_at_exact_0_or_1`) sentetik veride doğrulanmıştı, şimdi gerçek 369 satırlık veri üzerinde de teyitli. Düzeltme gerekmedi.

### 11. Ölçekleme Kararının Ampirik Doğrulaması (Tamamlayıcı Deney Turu, Görev 4)

Şu ana kadar "ağaç modelleri ölçekten bağımsız, RobustScaler/QuantileTransformer ölçek-duyarlı modeller için" kararı literatüre dayanıyordu, PAH verisinde hiç ölçülmemişti. Split bankasının 10 tekrarı (50 dış fold), 3 sabit diagnostic model (`LogReg`, `ExtraTrees`, `Dummy`) ve basamak 3 (log1p/logit/harmonizasyon uygulanmış, ölçeklemesiz) üzerine kurulu 3 varyant (ölçeksiz / `StandardScaler` / `RobustScaler`) ile ölçüldü (`src/genova/pah/preprocessing_experiments.py::run_scaling_comparison`, çıktı `reports/tables/scaling_comparison.csv`).

| Varyant | Model | F1 (patojenik) | MCC | Specificity | AUPRC |
|---|---|---|---|---|---|
| Ölçeksiz | ExtraTrees | 0.9115 | 0.2539 | 0.1711 | 0.9482 |
| StandardScaler | ExtraTrees | 0.9111 | 0.2462 | 0.1589 | 0.9477 |
| RobustScaler | ExtraTrees | 0.9133 | 0.2716 | 0.1775 | 0.9492 |
| Ölçeksiz | LogReg | 0.8421 | 0.1576 | 0.3493 | 0.8884 |
| StandardScaler | LogReg | **0.8635** | 0.2409 | 0.4054 | 0.9130 |
| RobustScaler | LogReg | **0.6898** | 0.0163 | 0.4340 | 0.8406 |
| — | Dummy | 0.9093 | 0.0 | 0.0 | 0.8348 |

**Bulgu 1 (beklendiği gibi, doğrulandı):** `ExtraTrees` üç varyant arasında pratik olarak kayıtsız (F1 0.9111–0.9133 arası, <0.3 puanlık fark, fold-üstü std ~0.03 içinde) — "ağaç modelleri ölçekten bağımsız" varsayımı bu veride teyit edildi.

**Bulgu 2 (beklenmedik, varsayımı KISMEN REVİZE ediyor):** `LogReg` ölçeklemeye hiç kayıtsız değil, ama yön beklenenin tersi: `RobustScaler`, `StandardScaler`'a göre değil, **ölçeklemesiz duruma göre bile belirgin şekilde daha kötü** (F1 0.6898 vs 0.8421 ölçeksiz — 15 puanlık düşüş, MCC neredeyse sıfıra iniyor). `StandardScaler` ise beklendiği gibi ölçeksiz duruma göre iyileşme sağlıyor (F1 +2.1 puan). Bu veri setinde `AL_`'nin log1p/logit sonrası dağılımı ve `EK_`'nin zaten [0,1]'e harmonize edilmiş yapısı, medyan/IQR bazlı `RobustScaler`'ın ölçeklendirmesini `LogReg`'in `class_weight="balanced"` optimizasyonu için sistematik olarak kötüleştiriyor olabilir (kesin nedensellik bu turun kapsamı dışında, yalnızca ölçüldü ve raporlanıyor).

**Metodolojik not — otomatik "kazanan" seçimi yanıltıcı çıktı:** `preprocessing_experiments.py::main()`, kazanan ölçekleyiciyi model ayrımı yapmadan (Dummy hariç tüm satırların) maksimum F1'ine göre seçiyor; bu ölçüt `RobustScaler`'ı seçti (`ExtraTrees`'in marjinal 0.0018 puanlık üstünlüğü sayesinde), ama bu seçim `LogReg`'deki 15 puanlık çöküşü tamamen maskeliyor. Basamak 4/5 (Görev 5 merdiveni) bu yüzden `RobustScaler` ile hesaplandı — sonuçlar aşağıda (§ `04_PREPROCESSING_MERDIVENI_PAH.md`) bu çöküşü doğrudan gösteriyor.

**Sonuç ve Aşama E için çıkarım:** Mevcut v3/pipeline tasarımı (`RobustScaler`, aykırı-değer dirençliliği gerekçesiyle seçilmişti — bkz. A.9 outlier bulguları) `ExtraTrees`/ağaç ailesi için sorunsuz kalıyor, ama **doğrusal modeller için körü körüne varsayılmamalı**: ölçekleyici seçimi, Aşama E'nin nested iç döngüsünde model-ailesine göre ayrı bir hiperparametre olarak ele alınmalı (örn. doğrusal aday modeller için `StandardScaler` iç döngüde denenmeli, `RobustScaler` sabit varsayılmamalı). Bu, mevcut varsayımı **doğrulamıyor, revize ediyor** — CLAUDE.md'nin genel kuralı yanlış değildi ama PAH'ın bu spesifik veri yapısında `RobustScaler`'ın doğrusal modeller için "güvenli" olduğu varsayımı bu ölçümle çürütüldü.

**Uygulama güncellemesi:** Bu bulgu üzerine `dataset_versions.py::build_v3`'ten (ve `serialize_pipelines.py`'nin ürettiği pipeline'dan) koşulsuz `RobustScaler` adımı **kaldırıldı** — `v3.parquet` artık dönüştürülmüş + grup-özellikli ama **ölçeksiz**. Ölçekleme, Aşama E'nin nested iç döngüsünde model-ailesine göre ayrı bir hiperparametre olarak seçilecek. Bu, yukarıdaki bulgunun doğrudan, gecikmesiz uygulanmasıdır — "revize ediyor" tespiti artık koda yansıtılmış durumda.

### 12. Ön İşleme Pipeline'ları — Serialize Edilmiş Kullanım (Tamamlayıcı Deney Turu, Görev 7)

Aşama B'nin sınıfları (şema doğrulama → blok eksiklik göstergeleri → doldurma → [yalnızca dönüştürülmüş kolda: dönüşüm + harmonizasyon] → kodlama) iki `sklearn.pipeline.Pipeline` nesnesinde birleştirildi ve `joblib` ile kaydedildi:

- `artifacts/preprocessors/pah_pipeline_tree.joblib` — v2 hattına karşılık gelir (ağaç modelleri için; kategorikler native `category` dtype, ek kodlama/ölçekleme yok).
- `artifacts/preprocessors/pah_pipeline_transformed.joblib` — v3'ün çekirdek dönüşümlerine karşılık gelir (log1p/logit/harmonizasyon + tam kategorik kodlama — `CAT_1` için `MultiValueFrequencyEncoder`, `CAT_2` için düz `FrequencyEncoder`; v3'ün korelasyon-kümesi grup-özet özellikleri — `build_v3`'ün inline mantığı, bir Aşama B sınıfı değil — kapsam dışı bırakıldı). **Güncelleme:** Bu pipeline önceden `pah_pipeline_scaled.joblib` adıyla ve gömülü bir `RobustScaler` adımıyla üretiliyordu; ölçeklemenin model-ailesine göre değiştiği bulgusu (yukarıdaki §11) üzerine ölçekleme adımı kaldırıldı ve dosya, artık yanıltıcı olmayan bir isimle (`pah_pipeline_transformed.joblib`) yeniden üretildi.

**`SchemaGate` şema notu:** Bu iki pipeline `v1.parquet`'i (ham CSV değil) girdi alıyor; `v1`'de `CAT_6` zaten düşürülmüş olduğundan (5 `CAT_` kolonu, ham CSV'nin 6'sı değil), `SchemaGate`'e `V1_EXPECTED_GROUP_COUNTS = {"AL": 334, "CAT": 5, "EK": 9, "AA": 2}` override'ı verildi (`schema.py::validate_schema`'ya eklenen opsiyonel `expected_group_counts` parametresi ile). Ham CSV → v1 geçişinin şema doğrulaması zaten `build_v1` içinde bir kez ham şemaya (`CAT: 6`) göre yapılıyor — bu iki katman birbirini tekrar etmiyor, farklı noktaları koruyor.

Her iki pipeline da yalnızca `fit`/`transform` adımları içerir, doğrulandı: `any(hasattr(step, "predict") for _, step in pipe.steps)` her ikisi için de `False` — hiçbir sınıflandırıcı yok.

**Kullanım örneği (transform, predict DEĞİL):**

```python
import joblib
import pandas as pd

v1 = pd.read_parquet("data/processed/pah/v1.parquet")
tree_pipe = joblib.load("artifacts/preprocessors/pah_pipeline_tree.joblib")

# DİKKAT: bu pipeline zaten tüm v1 üzerinde fit edildi (kaydedilmiş haliyle) --
# yeni/gerçek bir train/test ayrımında kullanmak için fold-güvenliği korumak
# adına yeniden `fit(train_df)` çağrılmalı, kaydedilmiş `.joblib` doğrudan
# `transform` için değil, yalnızca hızlı keşif/demo amaçlı kullanılmalı.
X_tree = tree_pipe.transform(v1)
print(X_tree.shape)  # (369, 356)
```
