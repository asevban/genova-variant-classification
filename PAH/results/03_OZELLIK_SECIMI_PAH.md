# PAH Paneli — Aşama D: Split Bankası + Nested Özellik Seçimi

**Not:** Bu aşamada model eğitimi yapılmadı, F1/eşik hesaplanmadı. Bu raporda geçen tüm AUC değerleri yalnızca özellik seçimi/ayırt edicilik için kullanılan hafif yardımcı sınıflandırıcılara (LightGBM, elastic-net lojistik regresyon) ait diagnostik ölçümlerdir — final model adayı değildir.

---

## D.0 — Split Bankası

Kod: `src/genova/pah/split_bank.py`. Girdi: `data/processed/pah/v1.parquet` (369 satır, dedup sonrası). Kütüphane: `sklearn.model_selection.StratifiedGroupKFold`. Seed: `BASE_SEED=42`, her tekrar için `42+repeat_idx`; iç fold seed'i `outer_seed*1000+outer_fold_idx`.

**Tasarım:** Dış **5-fold × 10 tekrar** (Repeated Stratified), iç **4-fold**. Grup-farkındalık `v1.parquet`'teki `group_id` kolonuyla sağlanıyor — yalnızca 1 çelişkili-profil çifti (`VAR_003238`/`VAR_003234`, `group_id="conflict_group_1"`) var, geri kalan 367 satırın her biri kendi tekil grubu. `StratifiedGroupKFold` bu iki satırı hem dış hem iç bölmelerde her zaman aynı tarafta tutuyor.

**Çıktı dosyaları:** `data/splits/pah/outer_fold_repeat{00..09}.json` (10 dosya, her biri 5 dış fold'un train/test `Variant_ID` listesini içerir) + `inner_fold_repeat{RR}_outer{F}.json` (50 dosya, her dış fold'un kendi eğitim kısmına uygulanan 4 iç fold'u içerir) + `manifest.json` (tasarım özeti). Toplam 61 dosya.

**Doğrulama (kod ile, `reports/tables/` altına değil doğrudan çalıştırılarak kontrol edildi):**
- Her repeat'te 5 test fold'u 369 satırı **tam ve çakışmasız** kapsıyor (0 eksik, 0 fazla).
- Hiçbir fold'da train/test kesişimi yok.
- Çelişkili-profil çifti (`conflict_group_1`) her dış VE her iç bölmede **her zaman aynı tarafta** — 50 dış fold + 250 iç fold kombinasyonunun tamamında sıfır ihlal.
- İç fold'lar her zaman kendi dış fold'unun eğitim alt-kümesinin bir alt-bölmesi (subset kontrolü geçti).
- Test fold'larındaki patojenik oranı 0.70–0.95 aralığında (genel ortalama 0.834'e yakın) — makul stratifikasyon kalitesi, küçük n=369'da beklenen varyasyon dahilinde.

**Toplam tespit edilen sorun: 0.** Split bankası bu haliyle sabitlendi, bir daha üretilmeyecek.

---

## D.1 — Özellik Seçimi Yöntem Karşılaştırması

Kod: `src/genova/pah/feature_selection.py` + `src/genova/pah/fold_features.py`. Her biri 50 dış fold'un (5×10) **yalnızca o fold'un eğitim kısmında** çalıştırıldı; hiçbir imputasyon/kodlama/seçim adımı test fold'unun satırlarını görmeden fit edilmedi (`fold_features.build_fold_features`, Aşama B sınıflarının fold-içi yeniden-fit edilmiş hâli — bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md`'deki sızıntı notu).

### Yöntemler ve seçim kuralı (her dış fold içinde)

| Yöntem | Uygulama | Fold-içi "seçildi" kuralı |
|---|---|---|
| Mutual information | `sklearn.feature_selection.mutual_info_classif` | O fold'da en yüksek skorlu ilk 30 özellik |
| Elastic-net stability selection | `LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga")`, 10 bootstrap alt-örnekleme (%80), standardize edilmiş özellikler | Bootstrap'ların ≥%50'sinde katsayı sıfır değil |
| Model-tabanlı (GBDT) | `LightGBM` (100 ağaç, max_depth=4, hafif/seçim-amaçlı) | O fold'da en yüksek `feature_importance` ilk 30 özellik |
| Permutation importance | Fold'un eğitiminde fit edilen LightGBM, **dış test fold'unda** ölçülen AUC üzerinden permütasyon (5 tekrar) | Ortalama önem > 0 |

**Stabilite kuralı (görev tanımına göre):** Bir özellik, 50 dış tekrarın **≥%60'ında** ("seçildi" olarak işaretlendiği fold sayısı / 50) o yöntem için stabil sayılır. Bu eşik her 4 yöntem için **ayrı ayrı** hesaplandı: `reports/tables/feature_selection_stability.csv`.

> **Düzeltme notu (Aşama D Düzeltme Turu — `AL_296_missing` kaldırıldı, `min_periods` düzeltildi):** Bu bölümdeki tüm sayılar, `AL_296_missing` göstergesinin `src/genova/pah/dataset_versions.py`/`fold_features.py`'den kaldırılması (Fisher's exact test ile gerekçelendirildi, bkz. `reports/03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`) ve `AL_` korelasyonunun `min_periods=30` ile düzeltilmesi (bkz. `reports/01_EDA_RAPORU_PAH.md` A.7) sonrasında **50 dış fold'un tamamı yeniden çalıştırılarak** güncellendi. Aday özellik sayısı 406'dan **405**'e düştü (yalnızca `AL_296_missing` eksildi). Aşağıdaki tüm tablolar güncel sonuçları yansıtıyor.

### Nedensel Ayrıştırma — Havuz Değişiminin Ne Kadarı Hangi Düzeltmeden Geldi?

Bu turda iki bağımsız düzeltme art arda yapıldı (`AL_296_missing` kaldırma + `min_periods=10→30`), ilk bakışta etkileri karışmış görünüyor. **Ayrıştırma sonucu: değişimin %100'ü `AL_296_missing` kaldırılmasından geldi, `min_periods` düzeltmesinin bu spesifik pipeline'a (Aşama D.1 nested özellik seçimi) hiçbir etkisi olmadı.** Gerekçe iki kanıtla doğrulandı, tam bir 2×2 yeniden-çalıştırma matrisine (her biri ~17 dk) gerek kalmadan:

1. **Yapısal (kod) kanıt:** `src/genova/pah/feature_selection.py` ve `src/genova/pah/fold_features.py` içinde `AL_correlation_clusters.csv`, `AL_spearman_corr.csv` veya herhangi bir korelasyon-türevi dosyaya/kümeleme sonucuna referans **yok** (`grep` ile doğrulandı, sıfır eşleşme). `fold_features.build_fold_features` yalnızca blok eksiklik göstergeleri + `EK_3`/`EK_` blok medyan-doldurma + `AL_` sıfır-doldurma + `CAT_1/2` frekans + `CAT_3/4/5`,`AA_1/2` one-hot kodluyor — korelasyon kümelerine dayanan grup-özet özellikleri (min/maks/medyan/pozitif-sayım) yalnızca `dataset_versions.py::build_v3`'te üretiliyor, **Aşama D.1'in hiçbir yerinde kullanılmıyor**.
2. **Ampirik kanıt:** `reports/tables/AL_correlation_clusters.csv` dosyası geçici olarak diskten kaldırılıp `build_fold_features` tek bir fold üzerinde tekrar çalıştırıldı — dosya **hiç yokken bile** hatasız, birebir aynı şekilde (405 kolonlu) çalıştı. Bu, dosyanın pipeline'a hiçbir girdi sağlamadığını doğrudan kanıtlıyor.

Dolayısıyla iki-yönlü ayrıştırma tablosu basitleşiyor (yapay bir 2×2 matrisi gerektirmez):

| Müdahale | Aşama D.1 havuzuna etkisi |
|---|---|
| `AL_296_missing` kaldırma | **Tüm gözlenen değişimin kaynağı** (406→405 aday özellik, 27→24 final havuz) |
| `min_periods=10→30` düzeltmesi | **Sıfır etki** (Aşama D.1 bu dosyayı hiç okumuyor; yalnızca `v3.parquet`'in grup-özet özelliklerini — ayrı bir kod yolu — etkiledi) |

Bu net bir sonuç; iki müdahale arasında etkileşim (interaction) yok çünkü ikinci müdahalenin bu pipeline'a giden hiçbir yolu yok.

> **Ek düzeltme notu (Soru 3 — `all_features` birleşim hatası):** `feature_selection.py::main()`'de stabilite tablosunun kolon evreni (`all_features`) önceden yalnızca **son işlenen fold'un** (repeat 9, dış fold 4) kolon kümesinden alınıyordu — bir fold'da seçilen ama son fold'un kolon kümesinde bulunmayan bir özellik varsa, sessizce stabilite tablosundan düşerdi. Bu, **tüm 50 fold'un kolon birleşimini** alacak şekilde düzeltildi (`all_features_seen` kümesi döngü içinde biriktiriliyor). Düzeltme sonrası 50 fold'un tamamı yeniden çalıştırıldı — **sonuç birebir aynı çıktı** (405 aday özellik, 24-özellikli final havuz, ablation/izole-model AUC'leri 6 ondalık basamağa kadar özdeş). Bu, düzeltmeden önceki çalıştırmada son fold'un kolon kümesinin zaten tam birleşime eşit olduğu (yani hatanın bu özel veri seti/split bankası kombinasyonunda pratikte zararsız kaldığı) anlamına geliyor — ama düzeltme, gelecekteki farklı veri/split kombinasyonlarında sessiz veri kaybını önlemek için kalıcı olarak koda işlendi. **Bu raporun tüm sayıları artık bu düzeltmeyi içeren, resmi nihai çalıştırmaya aittir.**

| Yöntem | Kendi ≥%60 eşiğini geçen özellik sayısı (405 içinden) |
|---|---|
| Mutual information | 4 |
| GBDT importance | 20 |
| Permutation importance | 22 |
| **Elastic-net stability** | **165** |

Elastic-net'in kendi eşiği diğer üç yöntemden **çok daha gevşek** — 405 özelliğin %41'i geçiyor. Bu, n=369/p≈405 rejiminde (özellik sayısı örneklem büyüklüğüne yakın) L1/L2 karışık cezanın yeterince seyreltici olmadığını, birçok ilişkisiz özelliğin de bootstrap'ların yarısında sıfırdan farklı katsayı alabildiğini gösteriyor — **gürültü, gerçek sinyal değil**.

**Nüans (Aşama D Düzeltme Turu, bağımsız EDA notebook'u ile çapraz kontrol):** Bu gevşekliğin ilk yorumu "çoklu-doğrusallık nedeniyle" idi, ama bu tam doğru değil/eksik bir gerekçe. `AL_` grubu içi Spearman korelasyonu **min_periods=30** ile yeniden hesaplandı (bkz. `reports/01_EDA_RAPORU_PAH.md` A.7 — projenin önceki `min_periods=10` sonucu, yalnızca 13 ortak-dolu satıra dayanan sahte-yüksek bir korelasyon (0.92) içerdiği için düzeltildi) ve `AL_` grubunda aslında **ciddi bir çoklu-doğrusallık olmadığını** doğruladı: en yüksek ikili korelasyon yalnızca 0.777 (`AL_121`-`AL_88`), medyan 0.094, ortalama 0.124 — hiçbir çift "neredeyse özdeş" (|r|>0.9) seviyesine ulaşmıyor. Bu sonuç, bağımsız EDA notebook'unun (`01_eda_pah_ipynb.ipynb`) aynı eşikle bulduğu sayılarla birebir örtüşüyor.

Dolayısıyla elastic-net'in gevşekliğinin **asıl birincil nedeni**, güçlü çoklu-doğrusallıktan çok, **n≈p az-belirlenmiş (underdetermined) rejimi** olmalı: 369 satır ile 406 özellik neredeyse aynı büyüklükte olduğunda, L1/L2 karışık ceza güçlü ilişkili özellik çiftleri olmadan da yeterince seyreltici olamaz — bootstrap alt-örneklemelerinin her birinde farklı, kısmen rastgele özellik alt-kümeleri sıfırdan-farklı katsayı alabilir. Çoklu-doğrusallık bu etkiyi güçlendirebilecek bir faktördür ama burada gözlenen gevşekliğin *gerekli* nedeni değildir. Bu nedenle final aday havuzu tek bir yönteme (özellikle elastic-net'e) değil, **yöntemler arası uzlaşıya** dayandırıldı.

### Final aday havuzu

**Kural:** Bir özellik **en az 2 yöntemde** ≥%60 stabil ise final aday havuzuna girer (`reports/tables/v4_final_feature_pool.json`). Gerekçe: yukarıdaki tabloda görüldüğü gibi tek-yöntem eşiği (özellikle elastic-net) tek başına güvenilir değil; ≥2 bağımsız yöntemin uzlaşması gürültüyü büyük ölçüde eler.

| Kaç yöntemde stabil (0-4) | Özellik sayısı |
|---|---|
| 0 | 230 |
| 1 | 151 |
| 2 | 14 |
| 3 | 8 |
| 4 | 2 |

**Final havuz büyüklüğü: 24 özellik** (405 içinden, ≥2 yöntem uzlaşan; önceki turda 27 idi — `AL_296_missing`'in kaldırılması dış fold'lardaki yardımcı model/seçici davranışını hafifçe değiştirdi: 4 özellik eşiğin altına düştü (`AL_306`, `AL_328`, `AL_331`, `AL_7`), 1 yeni özellik eşiği geçti (`AL_84`), 23 özellik ortak kaldı). Tüm 4 yöntemde stabil olan yalnızca 2 özellik var, **değişmedi**: **`EK_7`** ve **`AL_300`** — `EK_7`'nin bu kadar güçlü ve tutarlı çıkması, Aşama A.3.5'teki tek-özellik AUC taramasında da en yüksek skoru (0.715) alan kolon olmasıyla birebir örtüşüyor; döngüsellik şüphesi yaratacak kadar yüksek değil (bkz. D.1 izole-grup sonuçları altında), makul bir gerçek sinyal olarak yorumlanıyor.

24 özellikli havuzun grup dağılımı: **16 `AL_`, 6 `EK_`, 1 `CAT_` (`CAT_1`), 1 blok göstergesi (`al_all_missing`), 0 `AA_`.** Bu dağılım, aşağıdaki grup ablation/izole-model sonuçlarıyla tam tutarlı.

---

## D.2 — Grup Ablation (bir grup çıkarılınca AUC değişimi)

50 dış fold ortalaması, tam yardımcı LightGBM modeli baz alınarak:

| Model | Ortalama AUC | Std |
|---|---|---|
| **Tam model** | **0.775** | 0.072 |
| AL_ çıkarılmış | 0.760 (Δ=-0.015) | 0.079 |
| **EK_ çıkarılmış** | **0.734 (Δ=-0.040)** | 0.087 |
| AA_ çıkarılmış | 0.782 (Δ≈+0.007) | 0.076 |
| CAT_ çıkarılmış | 0.781 (Δ≈+0.006) | 0.066 |

Kaynak: `reports/tables/group_ablation_summary.csv` (özet), `group_ablation_raw.csv` (50 fold'un ham değerleri). *(`AL_296_missing` kaldırıldıktan ve `min_periods` düzeltildikten sonra yeniden çalıştırıldı; sayılar önceki turdan (0.777/0.760/0.734/0.777/0.776) yalnızca gürültü seviyesinde farklı, yorum değişmedi.)*

**Yorum:** `EK_` grubunun çıkarılması en büyük performans kaybına yol açıyor (-0.040 AUC) — 9 kolonluk küçük bir grup olmasına rağmen orantısız katkı sağlıyor. `AL_` grubu da anlamlı katkı sağlıyor (-0.015) ama daha ölçülü — 334 kolonluk devasa boyutuna rağmen katkısının mütevazı olması, EDA A.6'daki "90 kolon sabit-değerli, bilgi taşımıyor" bulgusuyla tutarlı. `AA_` ve `CAT_` gruplarının çıkarılması **pratik olarak sıfır etkili** (gürültü seviyesinde, hatta hafif pozitif) — bu iki grup tam modelde neredeyse hiçbir ek katkı sağlamıyor.

---

## D.3 — İzole-Grup Modelleri (bir grup tek başına ne kadar yetiyor)

| Grup | Ortalama AUC (tek başına) | Std |
|---|---|---|
| `AL_` | 0.672 | 0.083 |
| `EK_` | 0.673 | 0.067 |
| `AA_` | 0.538 | 0.080 |
| `CAT_` | 0.528 | 0.075 |

Kaynak: `reports/tables/isolated_group_models_summary.csv` (özet), `isolated_group_models_raw.csv` (ham). *(Yeniden çalıştırma sonrası — bu tablo, `AL_296_missing`'in kaldırılmasından pratik olarak etkilenmedi: değerler önceki turla (0.671/0.673/0.538/0.528) gürültü seviyesinde aynı.)*

**Yorum — Ablation ile izole-model sonuçları birlikte okunmalı:**
- `AL_` ve `EK_` tek başlarına neredeyse eşit güçte (~0.67 AUC) — ikisi de gerçek, bağımsız sinyal taşıyor. `EK_` yalnızca 9 kolonla bu performansa ulaşırken `AL_` bunu 334 kolonla ancak eşliyor — kolon-başına bilgi yoğunluğu `EK_`'de çok daha yüksek, ablation sonucuyla (`EK_` çıkarılınca daha büyük kayıp) tutarlı.
- `AA_` (0.538) ve `CAT_` (0.528) neredeyse rastgele tahmin (0.50) seviyesinde — tek başlarına neredeyse hiç ayırt edicilik yok. Ablation'da da bu gruplar çıkarılınca performans değişmiyor. **İki bulgu birbirini doğruluyor: `AA_`/`CAT_` bu panelde zayıf sinyal taşıyor.**
- **Döngüsellik/meta-prediktör kontrolü:** Hiçbir grup, tek başına bile, Aşama A.3.5'teki 0.90 şüpheli-eşiğine yaklaşmıyor (en yüksek izole-grup AUC'si 0.673). Bu, hem tekil-kolon taramasının hem de grup-seviyesindeki bu ek kontrolün **aynı sonuca vardığını** — veri setinde gizlenmiş, döngüsel bir meta-prediktör olmadığını — iki bağımsız yöntemle teyit ediyor.

---

## Ek Not (Aşama D Düzeltme Turu, Görev 4) — GBDT ve Permutation Importance Bağımsız Değil

D.1'deki 4 yöntemden ikisi — **GBDT importance** ve **permutation importance** — aynı fold-içi eğitilmiş LightGBM modelinden türüyor: GBDT importance modelin kendi `feature_importances_` çıktısı, permutation importance ise **aynı modelin** dış test fold'unda özellik karıştırılarak ölçülen performans kaybı. Bu ikisi, modelin öğrendiği aynı ağaç yapısına bağımlı olduğu için **istatistiksel olarak bağımsız değil** — ikisinin aynı fikirde olması, tamamen farklı iki modelleme yaklaşımının (örn. GBDT + doğrusal model) uzlaşmasından daha zayıf bir kanıt sayılmalı. Bu nedenle "≥2 yöntemde stabil" kuralı, bir özelliğin **yalnızca** GBDT+permutation ikilisiyle stabil çıktığı durumlarda, gerçekte "1.5 bağımsız yöntem" gibi okunmalı; asıl güçlü kanıt **mutual information** veya **elastic-net** gibi yapısal olarak farklı bir üçüncü yöntemin de aynı özelliği doğrulamasıdır.

**Final 24-özellikli havuzun konsensüs gücü dökümü** (`reports/tables/feature_selection_stability.csv` üzerinden hesaplandı; `AL_296_missing` kaldırıldıktan ve `min_periods` düzeltildikten sonra yeniden hesaplandı):

| Konsensüs türü | Özellik sayısı |
|---|---|
| **Yalnızca GBDT+permutation ile stabil** (bağımsız olmayan ikili, daha zayıf kanıt) | **3** (`AL_318`, `AL_12`, `AL_22`) |
| **En az bir bağımsız üçüncü yöntemle de teyitli** (MI veya elastic-net + en az 1 diğer) | **21** |

Havuzun **%87.5'i (21/24)** GBDT+permutation ikilisinin ötesinde bağımsız bir yöntemle de doğrulanıyor (önceki turda %81, 22/27) — bu, havuzun büyük çoğunluğunun gerçekten güçlü, yöntem-çeşitliliğine dayanan bir konsensüse oturduğunu gösteriyor. Geriye kalan 3 özellik (`AL_318`, `AL_12`, `AL_22`) yalnızca bağımlı ikiliyle stabil — bunlar Aşama E'de daha temkinli değerlendirilmeli (örn. ablation'da tek başlarına çıkarılıp etkisi ölçülebilir), ama bu turda havuzdan çıkarılmadı (görev kapsamı: yeni özellik seçimi/model eğitimi yok, yalnızca dokümantasyon).

---

## v4 — Özellik-Seçilmiş Veri Setleri

`src/genova/pah/dataset_versions.py::build_v4` — final 24-özellik havuzu (`reports/tables/v4_final_feature_pool.json`) v2 ve v3'e uygulanarak **iki** dosya üretildi (görev tablosundaki "v2 ve v3'ün ... indirgenmiş hâlleri" ifadesiyle uyumlu, ikisi de ayrı ayrı korunuyor):

- `data/processed/pah/v4_from_v2.parquet` (369×27: `Variant_ID`+`Label`+`group_id` + 24 özellik, v2'nin AL_ sıfır-doldurma/blok-gösterge kodlamasıyla)
- `data/processed/pah/v4_from_v3.parquet` (369×27: aynı 24 özellik, v3'ün log1p/logit/harmonize dönüşümleriyle — **güncelleme:** `RobustScaler` artık v3'e gömülü değil, bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md` §11)

Config'ler: `configs/pah/v4_from_v2.yaml`, `configs/pah/v4_from_v3.yaml`.

**Sızıntı notu (v2/v3 ile aynı):** Bu iki dosya da referans/deney iskeleti amaçlıdır. Havuzun kendisi (hangi 24 özelliğin seçildiği) zaten 50 dış fold'un yalnızca eğitim kısımlarından nested biçimde türetildi — bu kısım sızıntısız. Ama v4 dosyalarının içindeki sayısal değerler (medyan doldurma, frekans kodlama, ölçekleme) yine tam veri setinde fit edildi; Aşama E'de gerçek performans ölçümü fold-içi yeniden fit gerektirecek.

---

## Ek Not — `v3` Ölçekleyici Kaldırma + `CAT_1` Çok-Değerli Frekans Düzeltmesi Sonrası Havuz Değişimi

Bu turda iki bağımsız düzeltme uygulandı: (A) `v3`'ten koşulsuz gömülü `RobustScaler` kaldırıldı
(bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md` §11 güncellemesi), (B) `CAT_1`'in çok-değerli
(`&`-birleşik) satırları artık `MultiValueFrequencyEncoder` ile (bileşen popülasyonların
tekil-frekans ortalaması) kodlanıyor, eskiden olduğu gibi tek atomik bir kategori olarak değil
(bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md` §7). Düzeltme B, aday özellik matrisinin `CAT_1`
kolonunu değiştirdiği için Aşama D.1'in tüm 50-fold nested özellik seçimi (`feature_selection.py`)
yeniden çalıştırıldı (Düzeltme A ölçeklemeyi kaldırdığı için `feature_selection.py`'nin kendi
aday matrisini —  `fold_features.py::build_fold_features`, hiç ölçekleme içermiyordu zaten —
etkilemedi, yalnızca Düzeltme B'nin etkisi ölçüldü).

**Final havuz değişti: 24 → 25 özellik.**

| Değişim | Özellik(ler) |
|---|---|
| Havuza **giren** | `AL_306`, `EK_9` |
| Havuzdan **çıkan** | `AL_84` |
| Değişmeyen (ortak) | 23 özellik |

**Mekanizma — üçü de `permutation_importance`'ın %60 eşiğini geçmesi/altına düşmesiyle açıklanıyor:**

| Özellik | `mutual_information` | `gbdt_importance` | `elasticnet_stability` | `permutation_importance` | Sonuç |
|---|---|---|---|---|---|
| `AL_306` | 0.70→0.70 (aynı) | 0.30→0.22 | 0.12→0.12 (aynı) | **0.58→0.62** (eşiği geçti) | 1→2 yöntem: **girdi** |
| `EK_9` | 0.28→0.28 (aynı) | 0.78→0.78 (aynı) | 0.48→0.48 (aynı) | **0.56→0.60** (eşiği geçti) | 1→2 yöntem: **girdi** |
| `AL_84` | 0.02→0.02 (aynı) | 0.02→0.02 (aynı) | 0.74→0.74 (aynı) | **0.60→0.54** (eşiğin altına düştü) | 2→1 yöntem: **çıktı** |

Üç değişimin de yalnızca `permutation_importance`'tan kaynaklanması mantıklı: bu yöntem, her fold'da
fit edilen yardımcı LightGBM modelinin **dış test fold'undaki** performans katkısını ölçüyor —
`CAT_1`'in düzeltilmiş kodlaması modelin genel yapısını (ve dolayısıyla hangi özelliğin ne kadar
"tutulduğunu") hafifçe değiştirdi. `mutual_information` (özellik-bazlı, modelden bağımsız) ve
`elasticnet_stability` (kendi bootstrap'ları) bu üç özellik için **hiç değişmedi** — beklenen,
çünkü `CAT_1`'in kendi değeri değişse de bu iki yöntem `AL_306`/`EK_9`/`AL_84`'ün DEĞERLERİNE değil
kendi ayrı istatistiklerine bakıyor. Üçü de önceden zaten **eşiğin çok yakınında** (0.56-0.60
bandında) — yani bu, büyük bir sinyal değişikliği değil, zaten marjinal olan özelliklerin ince bir
kayması.

**`CAT_1`'in kendi stabilite oranı da arttı** (havuzda zaten kalıyordu, ama konsensüs gücü güçlendi):

| Yöntem | Önce | Sonra |
|---|---|---|
| `mutual_information` | 0.0 | 0.02 |
| `gbdt_importance` | 0.8 | 0.9 |
| `elasticnet_stability` | 1.0 | 1.0 |
| `permutation_importance` | 0.58 | **0.62** (eşiği geçti) |
| **Stabil olduğu yöntem sayısı** | **2/4** | **3/4** |

`CAT_1`, düzeltmeden önce yalnızca 2/4 yöntemde stabildi (havuza zar zor giriyordu); düzeltmeden
sonra 3/4 yöntemde stabil — daha semantik olarak doğru bir kodlamanın, aynı zamanda daha tutarlı/
güçlü bir istatistiksel sinyal ürettiğine işaret ediyor (5/369 satırlık küçük bir düzeltmenin
ölçülebilir ama küçük bir etkisi var, beklenen büyüklükte).

**v4 dosyaları yeniden üretildi:** `v4_from_v2.parquet`/`v4_from_v3.parquet` artık 369×28 (25
özellik + `Variant_ID`/`Label`/`group_id`), önceki 369×27'den güncellendi. Bu raporun yukarıdaki
"D.1" bölümündeki "24 özellik" ve ilgili sayılar artık bu güncel 25-özellik sonucuna aittir; tam
detaylı tablo (`feature_selection_stability.csv`) da bu turda yeniden üretildi.

---

## Ek Not (Aşama E Öncesi Denetim, Orta-5) — Meinshausen-Bühlmann PFER Üst Sınırı

`reports/06_ASAMA_E_ONCESI_DENETIM_PAH.md` (Orta-5), elastic-net stabilite seçiminin "165/405
özellik kendi %60 eşiğini geçiyor" bulgusunun (yukarıda "gürültü, gerçek sinyal değil" diye
yorumlandı) literatürdeki resmi bir hata-kontrolü ölçütüyle nicelleştirilmesini istedi.
Meinshausen & Bühlmann (2010, *"Stability Selection"*, JRSS-B) Theorem 1, `π>1/2` iken
yanlış-seçilen özellik sayısının beklenen üst sınırı (PFER) için şu formülü veriyor:

**E(V) ≤ (1/(2π−1)) · (q²/p)**

— burada `q`: tek bir uygulamada (burada: tek bir dış fold'da) ortalama seçilen özellik
sayısı, `p`: aday özellik evreni, `π`: stabilite eşiği.

**Bu projenin sayılarıyla (mevcut `feature_selection_stability.csv`'den, yeni deney
gerekmeden hesaplandı):**

- `p = 405` (aday özellik evreni)
- `π = 0.60` (projenin stabilite eşiği)
- `q = 188.92` — `elasticnet_stability` kolonundaki 405 oranın toplamı; bu toplam,
  doğrusallık (linearity of expectation) gereği **fold başına ortalama kaç özelliğin
  elastic-net tarafından seçildiğinin** doğrudan bir tahminidir (yeni bir fold-bazlı
  hesaplama gerekmedi, mevcut tablodan türetildi).
- **PFER üst sınırı = (1/(2×0.60−1)) × (188.92²/405) ≈ 440.6**

**Yorum — sınır anlamsız (vacuous):** 440.6, aday özellik evreninin kendisinden (405)
**büyük** — yani "en fazla ~441 yanlış-pozitif özellik olabilir" ifadesi, zaten yalnızca 405
aday olduğu için hiçbir bilgi taşımıyor. M&B'nin bağı yalnızca `q≪p` (gerçekten seyrek/sparse)
rejiminde anlamlı; burada `q≈189`, `p=405`'in neredeyse yarısı — **hiç seyrek değil**. Bu,
raporun daha önce ampirik olarak vardığı sonucu ("n≈p az-belirlenmiş rejim, çoklu-doğrusallık
değil") **formel bir istatistiksel argümanla da doğruluyor**: elastic-net stabilite
seçiminin **tek başına**, bu veri ölçeğinde hiçbir sertifikalı hata kontrolü sağlamadığı
matematiksel olarak gösterilmiş oluyor — ≥2-yöntem konsensüs kuralının (tek yönteme
güvenmemek) neden gerekli olduğunu güçlendiren ek bir kanıt.

**Yaklaşıklık uyarısı:** M&B'nin `1/(2π−1)` sabiti, orijinal makalede tek-aşamalı,
tamamlayıcı-çift (complementary-pairs) %50 alt-örnekleme varsayımı altında türetildi.
Projenin `elasticnet_selected()` fonksiyonu ise **iki aşamalı** bir prosedür kullanıyor: önce
fold içinde %80 alt-örneklemeyle 10 bootstrap (bir özellik bu bootstrap'ların ≥%50'sinde
sıfırdan-farklı katsayı alırsa "fold-seçilmiş" sayılıyor), sonra bu fold-seçim sonucu 50 dış
fold üzerinden %60 eşiğiyle stabilize ediliyor. Bu, M&B'nin tek-aşamalı türetiminden farklı;
yukarıdaki sayı bu yüzden **sertifikalı bir garanti değil, yaklaşık bir tanısal/diyagnostik
gösterge** olarak sunuluyor — yine de sonucun yönü (bağın anlamsız/vacuous çıkması) o kadar
büyük bir marjla (441 > 405) gerçekleşiyor ki, %50→%80 alt-örnekleme farkının bu niteliksel
sonucu değiştirmesi olası değil.

---

## Özet Tablo

| Bileşen | Sonuç |
|---|---|
| Split bankası | Dış 5×10 tekrar + iç 4-fold, `StratifiedGroupKFold`, 0 ihlal |
| En güçlü grup (ablation) | `EK_` (-0.040 AUC kaybı) |
| En zayıf gruplar | `AA_`, `CAT_` (izole AUC ~0.53, ablation etkisi ~0) |
| Şüpheli döngüsellik | Yok (grup ve tekil-kolon seviyesinde iki bağımsız kontrolle doğrulandı) |
| Final stabil özellik havuzu | **24 özellik** (16 AL_, 6 EK_, 1 CAT_, 1 blok göstergesi) — Aşama D Düzeltme Turu'nda `AL_296_missing` kaldırılıp `min_periods` düzeltildikten sonra 27'den güncellendi |
| Tüm 4 yöntemde stabil | `EK_7`, `AL_300` (değişmedi) |

---

## P0 Düzeltme Turu — `permutation_selected` Sızıntısı ve Fold-Lokal Havuz Mimarisi

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, madde 1-4. Yukarıdaki
> tüm bölümler (Özet Tablo dahil) **korunmuştur, silinmemiştir** — bu bölüm
> yalnızca eklenmiştir. Yukarıdaki içerik artık **tarihsel** kabul edilmeli;
> resmî final havuz artık tek bir sabit dosya değil, fold-başına üretilen bir
> havuz kümesidir (aşağıda açıklanmıştır).

### Bulunan sızıntı

Eski `permutation_selected(X_train, y_train, X_test, y_test, seed)`, `sklearn.
inspection.permutation_importance`'ı doğrudan **dış-test** (`X_test, y_test`)
üzerinde çalıştırıyordu — dış-test etiketleri, o dış-fold'un özellik-seçim
kararına doğrudan giriyordu. Bu skor daha sonra 50 dış-fold'un hit-oranlarıyla
**global** olarak birleştirilip (`build_v4_final_pool`) TEK sabit bir
25-özellik havuzu (`v4_final_feature_pool.json`) üretiyordu — bu havuz sonra
TÜM 50 dış-fold'un `v4_from_v2`'sinde (dolayısıyla E2-E6'nın tamamında)
kullanılıyordu. Nested doğrulamanın "dış-test hiçbir seçim kararına giremez"
ilkesini ihlal eden kritik bir sızıntıydı (CLAUDE.md Kesin Kural #1-2).

### Düzeltme

**1) `permutation_selected` (P0-1):** artık yalnızca `(X_train, y_train, seed,
val_frac=0.2)` alıyor — dış-test'e hiçbir şekilde erişimi yok. Kendi iç
stratified `train_test_split`'ini (`val_frac=0.2`) ayırıyor, yardımcı GBDT'yi
iç-fit kısmında eğitip permutation-importance'ı iç-val kısmında ölçüyor.

**2) Fold-lokal havuz mimarisi (P0-2):** `fold_versions.py::build_v4_from_v2/
v3` hiç değiştirilmedi — zaten bir `pool` parametresi alıyorlardı, mimari
olarak fold-lokal'a hazırdılar. Eksik olan, bu `pool`'un HER dış-fold için
AYRI türetilmesiydi. Yeni `compute_fold_local_pool(X_train, y_train, seed)`
fonksiyonu, `elasticnet_selected`'in zaten sahip olduğu felsefeyi (kendi
train_ids'i içinde bootstrap resampling ile stabilite oranı) MI/GBDT/
permutation'a da genelleştiren `_resampled_hit_rate` yardımcısını kullanıyor
— dört yöntemin dördü de artık **yalnızca o fold'un kendi train_ids'i
içindeki** 10 alt-örneklemden (%80, yerine koymadan) bir stabilite oranı
üretiyor, ardından aynı "≥2/4 yöntem ≥%60" kuralı **o fold'a özel** uygulanıp
o fold'un kendi havuzu döndürülüyor. Eski global-birleşim (`build_v4_final_
pool`) fonksiyonu SİLİNMEDİ, yalnızca artık çağrılmıyor — docstring'i
tarihsel izlenebilirlik için "ESKİ/ARŞİV" olarak güncellendi.

Sonuç: **50 dış-fold artık 50 farklı özellik havuzuna sahip olabilir** (ve
fiilen sahip) — bu, tasarım gereği beklenen ve doğru bir sonuçtur, bir
regresyon değildir (bkz. aşağıdaki testler).

### Yeni testler (`tests/test_p0_fold_local_feature_selection_pah.py`, 6 test)

- İmza + bytecode kontrolü: `permutation_selected` ve `compute_fold_local_pool`'un
  hiçbir parametresi/değişkeni `X_test`/`y_test` adını taşımıyor.
- Eski (sızıntılı) 5-argümanlı çağrı şekli artık `TypeError` fırlatıyor.
- İki farklı dış-fold'un (sentetik, biri `AL_1` diğeri `AL_2` bilgilendirici
  olacak şekilde kurulmuş, >30 sütunlu — `TOP_K_MI/GBDT=30`'un gerçekten
  filtre olarak çalışması için) havuzlarının **aynı olmak zorunda olmadığı**
  doğrulandı — eski global-havuz regresyonunu yakalayacak asıl test.
- `id3_feature_selection` testleriyle aynı desende bir fold-güvenlik testi:
  aynı `train_ids`, farklı (bozulmuş) `test_ids` → `X_train`/havuz değişmiyor.

Tam paket: **117/117 test geçti** (111 eski + 6 yeni), regresyon yok.

### Adım 3 — Eski/Yeni Havuz Karşılaştırması

Düzeltilmiş prosedür 50 dış-fold'un tamamında çalıştırıldı (`python -m
genova.pah.feature_selection`). Eski sabit havuz `v4_final_feature_pool.json`
**silinmedi**, `v4_final_feature_pool_ARCHIVED_leaky.json` adıyla arşivlendi
(orijinal dosya da E3-E6'nın bu turda dokunulmayan bağımlılığı bozulmasın diye
yerinde bırakıldı). Yeni çıktı: `v4_fold_local_pools.json` (50 fold × kendi
havuzu) + `feature_selection_stability_fold_local.csv` (uzun format, fold
başına 4-yöntem oranları).

| Ölçüt | Eski (global, sabit) | Yeni (fold-lokal, 50 fold) |
|---|---|---|
| Havuz sayısı | 1 | 50 |
| Havuz büyüklüğü | 25 (sabit) | ort=27.32, std=4.54, min=18, max=37 |
| Fold-lokal havuzların birleşimi (≥1 fold'da seçilen) | — | 100 farklı özellik |

**Örtüşme (her fold-lokal havuz vs. eski 25-özellik havuz):**

- Eski havuzun ortalama ne kadarı her fold-lokal havuzda da var: **%69,9** (std %7,4)
- Fold-lokal havuzun ortalama ne kadarı eski havuzda da var: **%65,0** (std %8,1)
- Ortalama Jaccard benzerliği: **0,505** (std 0,060)
- Eski 25 özelliğin **hiçbiri** yeni prosedürde 50 fold'un tamamında sıfır kez
  seçilmedi (`eskide olup YENI hicbir foldda secilmeyenler: []`) — yani eski
  havuzdaki hiçbir özellik "tamamen yanlış" çıkmadı, yalnızca stabilite
  oranları fold'a göre değişiyor.
- En sık seçilen (≥%90/50 fold): `EK_7` (%100), `AL_300` (%98), `CAT_1`
  (%98), `EK_5` (%92), `EK_8` (%90), `AL_49` (%90) — eski havuzun en güçlü
  üyeleriyle örtüşüyor.
- Eski havuzun en zayıf iki üyesi (`AL_306` %26, `AL_277` %12/50 fold) yeni
  prosedürde de en düşük stabiliteye sahip — eski sızıntılı yöntemin bu iki
  özelliği havuza sokmasının sınırda/marjinal olduğunu doğruluyor.

**Yorum:** Havuzların tam olarak örtüşmemesi (~%65-70 örtüşme, Jaccard≈0,50)
**beklenen bir sonuçtur, hata değildir** — her fold artık yalnızca kendi 295
satırlık train'inden (n≈295, p=405) bir stabilite tahmini üretiyor; bu ölçekte
(Aşama D'nin PFER analizinin de gösterdiği gibi, bkz. yukarıki bölüm) tekil-
fold stabilite tahminleri doğası gereği gürültülü. Havuzun **çekirdeği**
(en sık seçilen ~10-15 özellik) fold'lar arasında tutarlı; kenar özellikler
(marjinal stabilite) fold'a göre değişiyor.

### Adım 4 — `v4_from_v2` Yeniden Ölçümü (CatBoost/RF/LightGBM/XGBoost)

Hiçbir hiperparametre yeniden aranmadı — `e2_model_comparison.csv`'nin ZATEN
kaydettiği `best_params` aynen yeniden kullanıldı; tek değişen değişken
özellik havuzunun kaynağı (global sabit → fold-lokal). `e2_model_comparison.
csv`'nin v4_from_v2 için FİİLEN sahip olduğu (model, weighting_variant)
kombinasyonlarının hepsi (catboost/RF → A/B/C, lightgbm/xgboost → yalnızca B)
50 dış-fold'da yeniden ölçüldü. Sonuç **yeni** bir dosyaya yazıldı:
`reports/tables/e2_model_comparison_v4fixed.csv` (`e2_model_comparison.csv`
üzerine yazılmadı).

**Paired karşılaştırma (yeni − eski, aynı model/weighting/repeat/outer_fold,
n=50 çift her satırda):**

| Model | Ağırlıklandırma | ΔF1 (ort±std) | ΔMCC (ort±std) | Δspecificity (ort±std) | paired-t p |
|---|---|---|---|---|---|
| CatBoost | A_no_weight | −0,0049 ± 0,0110 | −0,0517 ± 0,1111 | −0,0433 ± 0,0955 | 0,003 |
| **CatBoost** | **B_fixed_minority_2x (final aday)** | **−0,0039 ± 0,0198** | **−0,0248 ± 0,1458** | **+0,0005 ± 0,1082** | **0,174** |
| CatBoost | C_data_driven_spw | −0,0081 ± 0,0223 | −0,0426 ± 0,1147 | −0,0136 ± 0,1283 | 0,013 |
| LightGBM | B_fixed_minority_2x | −0,0054 ± 0,0135 | −0,0549 ± 0,1169 | −0,0497 ± 0,1201 | 0,007 |
| Random Forest | A_no_weight | −0,0011 ± 0,0120 | −0,0040 ± 0,1688 | −0,0006 ± 0,1158 | 0,530 |
| Random Forest | B_fixed_minority_2x | −0,0002 ± 0,0119 | +0,0033 ± 0,1348 | +0,0128 ± 0,0946 | 0,887 |
| Random Forest | C_data_driven_spw | −0,0002 ± 0,0143 | −0,0234 ± 0,1112 | −0,0193 ± 0,0842 | 0,922 |
| XGBoost | B_fixed_minority_2x | −0,0040 ± 0,0182 | −0,0388 ± 0,1367 | −0,0298 ± 0,1276 | 0,127 |

**Final aday için (CatBoost/v4_from_v2/B_fixed_minority_2x) bootstrap %95 GA**
(10.000 paired-bootstrap tekrarı, F1 farkı üzerinde): **[−0,0093, +0,0014]**
— sıfırı kapsıyor, yani fark istatistiksel olarak sıfırdan ayırt edilemiyor.
Fold-lokal havuzun ortalama büyüklüğü bu 50 fold'da 27,32 (eski sabit: 25).

**Sürpriz bir sapma var mı? Hayır.** Sekiz (model, ağırlıklandırma)
kombinasyonunun **hepsi** aynı yönde (yeni ≤ eski, F1 farkı hep negatif veya
~0) ve büyüklük olarak küçük (|ΔF1|≤0,008). Bu, sızıntının **teorik olarak
beklenen yönüyle** tam örtüşüyor: dış-test etiketleriyle seçilen bir havuz, o
AYNI dış-test'te değerlendirildiğinde hafifçe şişirilmiş görünür (test'e
"göre" seçilmiş olduğu için); sızıntıyı kapatmak bu yapay şişirmeyi kaldırır.
Naif paired t-testin bazı kombinasyonlarda (ör. CatBoost/A, CatBoost/C,
LightGBM/B) p<0,05 çıkması, **büyük bir etki bulunduğu anlamına gelmiyor** —
projenin kendi P1-10 bulgusunun (fold-düzeyi CI, 10 tekrarın aynı 369 satırı
paylaştığı için bağımsızlık varsayımını ihlal ettiğinden gerçek belirsizliği
olduğundan dar gösteriyor) burada da geçerli olduğu unutulmamalı; asıl karar-
belirleyici sayı, final aday için sıfırı kapsayan bootstrap GA'dır. Bağımsız
"sızıntısız 15-özellik" denemesinin öngördüğü "yaklaşık maliyetsiz" beklentisi
(o zamanki tahmin +0,0091, hafif iyi yönde) ile bu ölçümün yönü (hafif negatif,
istatistiksel olarak sıfırdan ayırt edilemez) **aynı niteliksel sonuca**
varıyor: **düzeltmenin ölçülebilir bir performans maliyeti yok.**

### Sonuç

Muhtemel sonuç doğrulandı: **"aynı model, artık savunulabilir bir gerekçeyle."**
Final aday (CatBoost/v4_from_v2, B ağırlıklandırma) için performans, düzeltme
öncesi/sonrası istatistiksel olarak ayırt edilemez; havuzun kimliği fold'a
göre makul ölçüde değişiyor ama çekirdek özellikler (EK_7, AL_300, CAT_1 vb.)
tutarlı kalıyor. E3-E6'nın (kalibrasyon/SLD/eşik/ensemble) bu düzeltilmiş
`v4_from_v2`'ye göre yeniden koşulması **ayrı bir turda** yapılacak.
