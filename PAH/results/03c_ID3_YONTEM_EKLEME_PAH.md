# PAH Paneli — Aşama D.1'e 5. Yöntem: Entropy-Criterion Decision Tree Stabilite Seçimi

> **Ekleme türü: tamamlayıcı.** Bu rapor Aşama D.1'in mevcut 4 özellik seçimi yöntemine
> (mutual information, elastic-net stability selection, GBDT importance, permutation
> importance) dokunmadan, aynı split bankası (`data/splits/pah/`, dış 5×10 = 50 fold), aynı
> aday özellik evreni (`fold_features.build_fold_features`, 405 kolon) ve aynı stabilite
> formatını (%60 eşiği) kullanan bir 5. yöntem ekler. `src/genova/pah/feature_selection.py`,
> `fold_features.py`, `split_bank.py`, `dataset_versions.py`, `missingness.py`, `encoding.py`,
> `transforms.py`, `schema.py` — hiçbiri değiştirilmedi, yalnızca import edilip okundu.
> `data/splits/pah/`, `data/processed/pah/*.parquet`, `configs/pah/*.yaml`,
> `reports/tables/feature_selection_stability.csv`, `v4_final_feature_pool.json` **değişmedi**.
> Kod: `src/genova/pah/id3_feature_selection.py` (yeni). Testler:
> `tests/test_id3_feature_selection_pah.py` (5 yeni test, hepsi geçiyor).
>
> **Bu rapor bir karar belgesi değildir, yalnızca bulgu raporudur.** Havuza dahil etme önerisi
> aşağıda var, ama uygulanmadı — onay Teknofest PDR'ye bırakılıyor.

---

## Terminoloji Notu

`sklearn.tree.DecisionTreeClassifier(criterion="entropy")` kullanıldı. Bu, **tarihsel ID3
algoritmasının kendisi değildir** — ID3 yalnızca kategorik özelliklerle ve çok-yollu
(multi-way) bölünmeyle çalışır, budama yapmaz. Burada kullanılan, Information Gain
kriterinin modern **CART** (ikili bölünme, sürekli değişken desteği, isteğe bağlı derinlik
sınırı) çatısı altındaki karşılığıdır. Doğru ifade: **"ID3'ün information-gain mantığını CART
çatısında uyguladık"** — "saf ID3 kullandık" değil. Jüriye sunumda bu ayrım korunmalı.

---

## Adım 0 — Mevcut Metodoloji Teyidi (salt okunur)

`feature_selection.py` ve `fold_features.py` okunarak doğrulandı:

- **Aday özellik evreni:** `fold_features.build_fold_features`'ın döndürdüğü sayısal matris —
  blok eksiklik göstergeleri (`al_all_missing`, `ek_cat_block_missing`) + `EK_3`/`EK_` blok
  medyan-doldurma + `AL_` sıfır-doldurma + `CAT_1/2` frekans kodlama + `CAT_3/4/5`,`AA_1/2`
  one-hot kodlama. `feature_selection.py::main()`'de bu matrisin 50 fold'un **birleşimi**
  (`all_features_seen`) üzerinden **405 kolon** olduğu doğrulandı (bkz. `reports/03_OZELLIK_SECIMI_PAH.md`
  D.1 bölümü, "Aday özellik sayısı 406'dan 405'e düştü" notuyla tutarlı).
- **Stabilite kuralı:** Bir özellik, bir yöntemde, 50 dış fold'un (5 dış-fold × 10 tekrar)
  **≥%60'ında** ("hits/50 ≥ 0.60") o yöntem tarafından seçilmiş sayılıyorsa o yöntem için
  "stabil". Final havuz kuralı: bir özellik **en az 2 yöntemde** stabilse final havuza girer.
  `feature_selection.py`'deki `STABILITY_THRESHOLD = 0.60` sabiti ve `n_methods_stable_at_60pct`
  hesaplaması bu yeni modülde **birebir aynı** (`STABILITY_THRESHOLD`, aynı `SEED`/fold seed
  formülü: `SEED + repeat_idx*100 + fold_idx`) kopyalanarak doğrudan karşılaştırılabilirlik
  sağlandı.

---

## Adım 1 — `max_depth` Duyarlılık Analizi

369 satırlık (≈300 satırlık eğitim fold'u) küçük bir veri setinde derin bir ağaç kolayca
ezberler (aşırı öğrenir); bu da stabilite ölçümünü anlamsızlaştırır (her fold'da farklı,
gürültüye dayalı bir bölünme kümesi seçilir). Bu riski nicelleştirmek için 5 aday derinlik
(`3, 4, 5, 6, 8`), aynı 50 dış fold üzerinde, fold özellik matrisleri yalnızca bir kez inşa
edilip 5 derinlik arasında paylaşılarak (gereksiz yeniden hesaplama önlenerek) test edildi:

| `max_depth` | Fold başına ort. seçilen özellik sayısı | Std | Stabil havuz büyüklüğü (≥%60) |
|---|---|---|---|
| 3 | 6.00 | 0.69 | 2 |
| 4 | 9.44 | 1.31 | 2 |
| 5 | **13.20** | 2.04 | **3** |
| 6 | 16.86 | 2.48 | 3 |
| 8 | 22.46 | 3.25 | 4 |

**Gözlem:** Fold başına seçilen özellik sayısı derinlikle yaklaşık doğrusal artıyor (6.00'dan
22.46'ya), ama **stabil havuz büyüklüğü** çok daha yavaş büyüyor ve `max_depth=5` ile `6`
arasında bir platoya oturuyor (3=3). `max_depth=8`'de tekrar artışa geçmesi (4), derinlik
arttıkça fold'lar arası tutarlılığın değil, fold-özel gürültünün (369 satırlık veri setinde
her fold'un kendi rastgele bölünme kombinasyonu) baskın hale geldiğinin işareti.

**Seçilen derinlik: `max_depth=5`.** Gerekçe: (a) `max_depth≤4` neredeyse yalnızca 1-2
bölünmeye izin veriyor — bu, ağacın yalnızca en güçlü 1-2 sinyali yakalayabildiği, daha zengin
bir stabilite tablosu üretemeyen bir rejim; (b) `max_depth=5` ve `6` aynı stabil-havuz
büyüklüğünü (3) veriyor — bu bir platoyu işaret ediyor, ve platonun **daha sığ** (daha az
aşırı-öğrenmeye yatkın) ucu tercih edildi (Occam's razor / parsimony — n≈369 gibi küçük bir
örneklemde entropy-tabanlı ağaçların bilinen aşırı-öğrenme riski göz önünde bulundurularak);
(c) `max_depth=8`'in stabil havuzu büyütmeye devam etmesi (4), bu derinlikte artık gerçek
tutarlı sinyalden çok fold-özel gürültünün eklendiğine işaret ediyor — sığ tarafta durmak daha
temkinli.

---

## Adım 2 — ID3'ün Kendi Stabil Havuzu (`max_depth=5`, 50 dış fold)

`reports/tables/id3_feature_selection_stability.csv` (tam tablo, 405 satır) üretildi.
**ID3 (entropy, `max_depth=5`) ile stabil olan (≥%60) yalnızca 3 özellik var:**

| Özellik | ID3 stabilite oranı | Diğer 4 yöntemde stabil mi (ayrı ayrı) | Diğer 4 yöntemden kaç tanesinde stabil |
|---|---|---|---|
| **`EK_7`** | 0.88 | MI ✅ (0.94) · GBDT ✅ (1.0) · Elastic-net ✅ (1.0) · Permutation ✅ (0.94) | 4/4 |
| **`AL_49`** | 0.78 | MI ❌ (0.24) · GBDT ✅ (0.88) · Elastic-net ✅ (1.0) · Permutation ✅ (0.86) | 3/4 |
| **`CAT_1`** | 0.68 | MI ❌ (0.0) · GBDT ✅ (0.8) · Elastic-net ✅ (1.0) · Permutation ❌ (0.58) | 2/4 |

Kaynak: `reports/tables/five_method_stability_comparison.csv` (405 satır, 5 yöntemin tamamı
yan yana + havuz simülasyon bayrakları).

**Gözlem — ID3 gürültüye karşı son derece muhafazakâr:** 405 aday özellik içinde yalnızca 3'ü
(%0.7) ID3'ün kendi %60 eşiğini geçiyor — bu, diğer 4 yöntemin hiçbirinin eşiğinden (MI: 4,
GBDT: 20, Permutation: 22, Elastic-net: 165 — bkz. `03_OZELLIK_SECIMI_PAH.md` D.1) daha düşük,
elastic-net'in gevşekliğinin tam tersi bir uç. Sığ, tek-ağaçlı ve budama içermeyen bir modelin
küçük örneklemde bu kadar seçici davranması beklenen bir sonuç: her fold'da yalnızca en güçlü
birkaç bölünme mümkün (derinlik 5 ile en fazla 2⁵−1=31 iç düğüm), bu da zayıf/orta sinyalli
özelliklerin fold'lar arası tutarlı şekilde seçilme şansını düşürüyor.

---

## Adım 3 — Havuz Simülasyonu (bilgi amaçlı, uygulanmadı)

`reports/tables/five_method_pool_simulation.json`:

```json
{
  "current_pool_size_4_methods": 24,
  "simulated_pool_size_5_methods": 24,
  "entered_pool_with_id3": [],
  "left_pool_with_id3": []
}
```

**ID3 eklenince final havuz (≥2/5 kuralı) hiç değişmiyor: hâlâ 24 özellik, ne giren ne çıkan
var.** Neden: "≥2 yöntemde stabil" kuralı monoton — bir özelliğin zaten ≥2/4 yöntemde stabil
olması, 5. bir yöntem eklendiğinde onu havuzdan çıkaramaz (aynı 2 yöntem hâlâ sayılıyor); yeni
bir özelliğin havuza girmesi için ID3'ün, **halihazırda tam olarak 1/4 yöntemde stabil olan**
bir özelliği desteklemesi gerekirdi. Yukarıdaki tabloda görüldüğü gibi ID3'ün 3 stabil
özelliğinin **üçü de zaten ≥2/4 diğer yöntemde stabildi** (`EK_7`: 4/4, `AL_49`: 3/4, `CAT_1`:
2/4) — dolayısıyla ID3'ün oyu, havuzu **genişletmek** yerine **zaten en güçlü konsensüse sahip
3 özelliği bir 5. bağımsız yöntemle de teyit etmiş oldu.**

---

## Adım 3b — Provenance-Riskli Özellikler: `al_all_missing` ve `CAT_1`

Görev talimatına göre özellikle işaretlenmesi istenen iki özellik:

| Özellik | ID3 stabilite oranı | ID3'te stabil mi (%60 eşiği) | Yorum |
|---|---|---|---|
| **`CAT_1`** | 0.68 | ✅ **Evet** — ID3'ün yalnızca 3 stabil özelliğinden biri | ID3, yapısal olarak diğer 4 yöntemden tamamen farklı (tek, sığ, budamasız ağaç) — buna rağmen `CAT_1`'i bağımsız olarak güçlü bir sinyal olarak işaretliyor. Bu, `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'deki confound bulgusuyla (`al_all_missing` ↔ `CAT_1`/`CAT_2` eksikliği arasında Cramér's V=0.755) **aynı yönde** ek bir kanıt: `CAT_1`'in ayırt ediciliği gerçek ve yöntemden bağımsız olarak tekrar üretilebiliyor — ama bu ayırt ediciliğin biyolojik mi yoksa kaynak-provenance artefaktı mı olduğu konusunda **hiçbir yeni bilgi vermiyor** (ID3 nedensellik ayırt edemez, yalnızca ayırt edicilik ölçer). |
| **`al_all_missing`** | 0.30 | ❌ Hayır — eşiğin belirgin şekilde altında | Beklenmedik bir asimetri: `al_all_missing`, D.1'in mevcut 4 yönteminden 2'sinde (elastic-net, permutation) zaten stabil ve final 24-havuzunda yer alıyor, `Label` ile tek başına AUC=0.625 taşıyor (bkz. `03b`) — ama ID3'ün sığ ağacında fold'ların yalnızca %30'unda bir bölünmede kullanılıyor. Olası açıklama: `EK_7`/`AL_49`/`CAT_1` gibi sürekli/çok-seviyeli özellikler, ikili (0/1) bir gösterge kolonundan daha fazla farklı eşik noktası sunduğu için entropy-tabanlı bir ağaçta genellikle daha yüksek bilgi kazancı sağlıyor — bu, `al_all_missing`'in sinyalinin zayıf olduğu anlamına gelmiyor, yalnızca sığ bir ağacın kök/üst düzey bölünmelerinde ikili göstergelere göre sürekli özellikleri sistematik olarak tercih edebileceği anlamına geliyor. |

**Sonuç bu iki özellik için:** ID3, `CAT_1`'in provenance-confound riskini **güçlendiren** bağımsız bir kanıt sağlıyor (ID3'ün 405 özellik içinden seçtiği yalnızca 3 özellikten biri olması, bu bulgunun rastgele değil dikkat çekici bir sinyal gücüne işaret ediyor); `al_all_missing` için ise ID3 ne bulguyu ne güçlendiren ne de çürüten nötr bir sonuç veriyor (yöntem-bağımlı bir seçicilik farkı, sinyal kaybı değil). **Aşama F'nin adversarial validation adımına önerilen ek not:** `CAT_1`, yalnızca mevcut confound bulgusu nedeniyle değil, artık bağımsız bir 5. yöntemin de onu en-güçlü-3-sinyalden biri olarak işaretlemesi nedeniyle **öncelikli test edilmeli**.

---

## Karar Değil, Bulgu

Bu turda:
- Mevcut 4 yönteme, mevcut 24-özellikli final havuza, split bankasına veya herhangi bir
  v1-v4/config dosyasına **hiçbir değişiklik yapılmadı.**
- ID3 yöntemi kendi başına **çok muhafazakâr** (405'te yalnızca 3 stabil özellik) ve mevcut
  havuzu **hiç değiştirmiyor** (0 giren, 0 çıkan) — bu haliyle havuza eklenmesi pratik bir fark
  yaratmıyor.
- **Öneri (uygulanmadı, onaya bağlı):** ID3'ün 3 stabil özelliği zaten mevcut havuzun **en
  güçlü konsensüslü** üyeleri (`EK_7` 4/4, `AL_49` 3/4, `CAT_1` 2/4) — bu, ID3'ün gelecekte
  "beşinci bağımsız doğrulayıcı" olarak resmi konsensüs kuralına kalıcı şekilde eklenmesinin
  düşük risk/düşük ek fayda taşıdığını düşündürüyor (havuzu büyütmüyor, yalnızca en güçlü
  üyeleri teyit ediyor). Asıl değerli çıktısı özellik seçiminden çok, `CAT_1`'in provenance-risk
  değerlendirmesine sağladığı bağımsız kanıt — bu nedenle ID3'ü **Aşama F'nin adversarial
  validation kontrolüne bir girdi olarak** taşımak, resmi 5. seçim yöntemi yapmaktan daha
  yüksek getirili görünüyor. Nihai karar Teknofest PDR'ye bırakılıyor.

---

## Kısa Özet

- **Tutarlılık:** ID3, mevcut 4 yöntemle **çelişmiyor** — 3 stabil özelliğinin üçü de zaten
  mevcut havuzda ve en yüksek konsensüse sahip üyeler arasında.
- **Yeni sinyal:** Havuz büyüklüğü/içeriği açısından **yok** (0 giren/çıkan özellik). Ama
  `CAT_1`'in provenance-confound riski için **bağımsız, güçlendirici bir kanıt** üretti.
- **Sonraki adım önerisi:** ID3'ü resmi 5. seçim yöntemi olarak koda kalıcı şekilde eklemek
  yerine, bulgusunu (`CAT_1` işaretlemesi) Aşama F adversarial validation planına not olarak
  taşımak — ama bu bir öneri, uygulama onay bekliyor.

## Teslim Edilen Dosyalar

- `src/genova/pah/id3_feature_selection.py` (yeni modül)
- `tests/test_id3_feature_selection_pah.py` (5 yeni test, hepsi geçiyor; tam suite 33/33 yeşil, regresyon yok)
- `reports/tables/id3_max_depth_sensitivity.csv`
- `reports/tables/id3_feature_selection_stability.csv`, `id3_feature_selection_stable_pool.json`
- `reports/tables/five_method_stability_comparison.csv`, `five_method_pool_simulation.json`
- `reports/03c_ID3_YONTEM_EKLEME_PAH.md` (bu dosya)

Mevcut hiçbir dosya (`feature_selection.py`, `fold_features.py`, `split_bank.py`,
`dataset_versions.py`, `missingness.py`, `encoding.py`, `transforms.py`, `schema.py`,
`data/splits/pah/*`, `data/processed/pah/*.parquet`, `configs/pah/*.yaml`,
`reports/tables/feature_selection_stability.csv`, `v4_final_feature_pool.json`) değiştirilmedi.
