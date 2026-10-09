# Entropy-Tree Ensemble Deneyi — Faz 1: Model Karşılaştırması

> ⚠️ **Bu rapor bağımsız, araştırma amaçlı bir deneyin sonucudur — resmi GENOVA
> PAH pipeline'ının hiçbir parçasını değiştirmez.** `data/`, `configs/`,
> `reports/tables/`, `src/genova/pah/*.py` — hiçbirine yazılmadı (bkz. rapor
> sonundaki izolasyon doğrulaması). Sonuç önerisi bir E2 aday önerisidir,
> uygulama değildir — ayrı bir onay gerekir.

---

## Yönetici Özeti

**E2'ye resmi aday olarak eklenmeye değer mi? BELİRSİZ/DÜŞÜK ÖNCELİK** —
gerekçe aşağıda. Ensemble ailesi (Model B) tek-ağaç referansı ve tam-özellik-
uzayı varyantını (Model C) açıkça geçiyor, ve **prevalans-ayarlı metrikte**
(hedef %28.6 patojenik) basit "her zaman patojenik" tuzağından kaçınarak
gerçek bir ayırt edicilik gösteriyor. Ama mevcut Aşama D.1 havuzunun zaten
sağladığı sinyalin ötesine geçen, yeni bir bilgi kaynağı **değil** — aynı
25-özellik havuzunu kullanıyor, farklı bir model ailesiyle. `CAT_1`
ablasyonu bu ensemble ailesinde **istatistiksel olarak anlamsız** bir fark
gösterdi (p=0.10, N=50 eşli fold) — ne "CAT_1 kritik" ne "CAT_1 gereksiz"
sonucuna varmaya yeten güçte bir kanıt.

---

## 1. Metodoloji Doğrulaması

### PRECHECK (modellemeden önce, tüm maddeler PASS)

`results/precheck_log.txt`'ten özet — **10/10 madde PASS**:
- Split bank: 50 dış fold (10 tekrar × 5), her birinin 4 iç fold'u — doğrulandı.
- `build_fold_features` **import edildi, yeniden yazılmadı** (kaynak dosya
  yolu doğrudan doğrulandı: `src/genova/pah/fold_features.py`).
- `v4_final_feature_pool.json`'dan taze okunan liste: **25 özellik**, görev
  metnindeki referans listeyle **birebir eşleşti**.
- `build_fold_features()` çıktısı: **405 kolon**, `Label`/`Variant_ID` yok,
  25-özellik havuzunun tamamı bu 405 kolonun içinde.
- Kaynak-kod taraması: `data/`, `configs/`, `reports/tables/`,
  `src/genova/pah/*.py`'a hiçbir yazma girişimi **bulunamadı** (hem ilk
  script yazımından önce hem tüm script'ler yazıldıktan sonra iki kez
  çalıştırıldı, ikisinde de temiz).

### Kural 1 — Split bankası olduğu gibi kullanıldı

`run_experiment.py`, `data/splits/pah/outer_fold_repeat{RR}.json` ve
`inner_fold_repeat{RR}_outer{F}.json` dosyalarını doğrudan okuyor
(`json.loads(...read_text())`), yeni bir `StratifiedGroupKFold` çağrısı
**hiçbir yerde yok**. Split bankası dosyalarının mtime'ı deney öncesi/sonrası
karşılaştırıldı (bkz. §5) — değişmedi.

### Kural 2 — Nested eşik seçimi

`run_experiment.py::run_ensemble_model()` → `_inner_threshold_search()`:
her dış fold için 4 iç fold'da **ayrı** ensemble'lar eğitilip (yalnızca
iç-eğitim satırlarında), iç-val'de 11-19/21 oy-eşiği aranıyor, 4 iç fold'un
F1'i eşik-başına ortalanıp en iyi eşik seçiliyor. Ardından **tamamen yeni**
bir ensemble (`final_ens`), dış-eğitimin **tamamında** fit ediliyor, dış-test
yalnızca bu sabit eşikle **tek seferde** skorlanıyor (`ensemble.py::predict`,
`threshold=best_t` — dış-test satırları hiçbir eşik aramasına girmiyor,
kod satır satır izlenebilir: `run_ensemble_model()` içinde `best_t`
hesaplandıktan SONRA `final_ens.fit(X_train_sel, ...)` ve
`final_ens.predict(X_test_sel, threshold=best_t)` çağrılıyor — sıra kod
tarafından zorunlu kılınıyor).

### Kural 3 — Fold-güvenli preprocessing

Her `build_fold_features(v1, al_columns, train_ids, test_ids)` çağrısı
doğrudan `genova.pah.fold_features`'tan import edildi (bkz. PRECHECK'in
kaynak-yolu doğrulaması). Bu, resmi pipeline'ın tüm düzeltmelerini (blok
eksiklik göstergeleri, `MultiValueFrequencyEncoder` ile `CAT_1` kodlaması,
`AL_` sıfır-doldurma, `EK_3` medyan-doldurma) otomatik miras alıyor —
deneyde hiçbir paralel/yeniden-icat edilmiş imputasyon/encoding kodu yok.

---

## 2. Model A/B/C/D Karşılaştırma Tablosu

50 dış fold ortalama ± std:

| Model | Özellik | Yapı | F1 (ham) | MCC | Recall | Specificity | **F1 (prevalans-ayarlı, %28.6)** | **FP/100 benign** |
|---|---|---|---|---|---|---|---|---|
| A | 25 | Tek ağaç | 0.792 ± 0.064 | 0.275 ± 0.109 | 0.709 ± 0.098 | 0.636 ± 0.116 | 0.545 ± 0.065 | 36.4 ± 11.6 |
| **B** | 25 | 21-ağaç ensemble | **0.854 ± 0.035** | **0.374 ± 0.124** | **0.799 ± 0.062** | 0.652 ± 0.145 | **0.608 ± 0.075** | 34.8 ± 14.5 |
| C | 405 (rastgele 70/ağaç) | 21-ağaç ensemble | 0.840 ± 0.039 | 0.328 ± 0.123 | 0.783 ± 0.066 | 0.612 ± 0.120 | 0.574 ± 0.066 | 38.8 ± 12.0 |
| D | 24 (B − `CAT_1`) | 21-ağaç ensemble | 0.848 ± 0.041 | 0.373 ± 0.134 | 0.788 ± 0.069 | **0.663 ± 0.146** | 0.610 ± 0.089 | **33.7 ± 14.6** |

**Gözlemler:**
- **Model B (ana hipotez), dört model içinde en iyi/en istikrarlı sonucu
  veriyor**: en yüksek ham F1 (0.854, en düşük std=0.035 — en tutarlı),
  en yüksek MCC ve recall.
- **Model C (405 özellik, ağaç başına rastgele 70), B'yi hiçbir metrikte
  geçemiyor** — hem ham hem prevalans-ayarlı F1'de B'den düşük, FP/100
  benign'de en kötü model. Bu, Aşama D.1'in nested özellik seçiminin
  (405→25 indirgeme) rastgele bir alt-kümeden **gerçekten daha iyi**
  olduğunu bağımsız bir model ailesiyle de doğruluyor — mevcut özellik
  seçimi sürecine ek bir güven kanıtı.
- **Model A (tek ağaç), beklenildiği gibi en zayıf/en değişken model**
  (en yüksek std'lerden biri, en düşük tüm metrikler).

### Kritik bağlam — "ham F1" tek başına yanıltıcı

Aynı split bankasının aynı fold'larında, **hiçbir şey öğrenmeyen, her zaman
"patojenik" diyen** teorik bir taban çizgisi hesaplandı (eğitim fold'unun
kendi ~%83 patojenik oranıyla): **ham F1 ≈ 0.909** — Model B'nin 0.854'ünden
**daha yüksek**. Bu, resmi projenin split bankası + eğitim dağılımıyla
ölçülen ham F1'in, dengesiz bir sınıf dağılımında yanıltıcı olabileceğinin
somut bir hatırlatıcısı (CLAUDE.md'nin zaten vurguladığı prior-shift
endişesiyle birebir örtüşüyor). **Prevalans-ayarlı metrikte tablo tersine
dönüyor:**

| | Ham F1 | Prevalans-ayarlı F1 (%28.6) | FP / 100 benign |
|---|---|---|---|
| "Her zaman patojenik" (taban) | **0.909** | 0.445 | **100.0** (her benign hasta yanlış işaretlenir) |
| Model B (bu deney) | 0.854 | **0.608** | **34.8** |

Yani Model B, ham F1'de trivial taban çizgisinden düşük görünse de, **gerçek
yarışma dağılımını simüle eden metrikte onu büyük farkla geçiyor** — 100
yanlış-pozitif/100 benign yerine ~35. Bu, `class_weight="balanced"` +
dengeli-bootstrap tasarımının **tam olarak amaçlandığı gibi çalıştığının**
kanıtı: eğitim dağılımına aşırı uyum sağlamak yerine, dengeli bir
ayırt-edicilik öğreniyor.

---

## 3. Eşik Seçimi Bulgusu

`results/threshold_results.csv`'den: **50 dış fold'un 49'unda, üç ensemble
model için de (B/C/D) seçilen eşik 11/21 (basit çoğunluk)** — yalnızca 1
fold'da 12/21 seçildi. Ortalama 11.02 ± 0.14.

**Yorum:** `class_weight="balanced"` (her ağaçta) + dengeli bootstrap
(ensemble seviyesinde) zaten iki ayrı dengesizlik-düzeltme katmanı
sağlıyor — bu iki katman birlikte, oy-eşiğinin ince ayarına neredeyse hiç
ihtiyaç bırakmıyor gibi görünüyor (basit çoğunluk zaten yakın-optimal).
Bu, nested eşik aramasının **gereksiz olduğu** anlamına gelmiyor (arama
hâlâ doğru/gerekli bir adım, yalnızca bu spesifik tasarımda sonucun çoğunluk
oyuna yakınsadığını gösteriyor) — ama ileride bu ensemble tasarımı
geliştirilirse, eşik aramasının hesaplama maliyetini basit çoğunlukla
karşılaştırmalı bir duyarlılık kontrolü değerli olabilir.

---

## 4. `CAT_1` Ablasyonu (Model B vs Model D)

**Yalnızca Model B üzerinde** yapıldı — Model B, dış-fold sonuçlarına
bakılarak "en iyisi seçildiği" için değil, **görev tanımı gereği önceden,
tasarım olarak** ana hipotez seçildiği için (bkz. görev metni) — bu,
ablasyonun gizli bir model-seçim adımından sızıntı almamasını sağlıyor.

| | Model B (25, `CAT_1` dahil) | Model D (24, `CAT_1` çıkarılmış) | Fark (B−D) |
|---|---|---|---|
| F1 (ham) | 0.854 ± 0.035 | 0.848 ± 0.041 | +0.0053 ± 0.0220 |
| MCC | 0.374 ± 0.124 | 0.373 ± 0.134 | +0.0011 ± 0.0679 |
| Specificity | 0.652 | **0.663** | −0.011 |
| FP/100 benign | 34.8 | **33.7** | +1.1 (D biraz daha iyi) |

**Eşli istatistiksel testler (N=50 fold, aynı split'ler üzerinde):**
- F1: paired t-test p=**0.096**, Wilcoxon p=**0.321** — **anlamlı değil**.
- MCC: paired t-test p=**0.912** — **anlamlı değil** (neredeyse tam sıfır fark).
- Yön tutarsız: 24/50 fold'da B>D, 20/50'de B<D, 6/50'de eşit — sistematik
  bir yön yok, gürültü seviyesinde.

**Yorum:** Bu entropy-tree-ensemble ailesinde, `CAT_1`'in çıkarılması
**ölçülebilir, istatistiksel olarak anlamlı bir performans kaybına yol
açmıyor**. Bu, `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'nin açık
bıraktığı "biyolojik mi, provenance mi" sorusuna doğrudan cevap
**vermiyor** (görev metninin de belirttiği gibi bu ayrı bir soru, Aşama F
adversarial validation'ın işi) — ama şunu ekliyor: `CAT_1`'in **tahmin
gücü**, en azından bu model ailesinde, diğer 24 özellik tarafından büyük
ölçüde **yedekleniyor/örtüşüyor** görünüyor (specificity'nin `CAT_1`
çıkarılınca hafifçe İYİLEŞMESİ bile ilginç — çok küçük, gürültü seviyesinde
bir etki, ama en azından "CAT_1 olmadan performans çöküyor" senaryosunu
desteklemiyor). Aşama D.1'in kendi verisiyle de tutarlı: `CAT_1` zaten
havuzda en marjinal üyelerden biri (yalnızca 3/4 yöntemde stabil, `EK_7`/
`AL_49` gibi 4/4'lük üyelerden farklı olarak — bkz.
`reports/03_OZELLIK_SECIMI_PAH.md`).

---

## 5. İzolasyon Doğrulaması

`data/`, `configs/`, `reports/tables/`, `src/genova/pah/*.py` altındaki
tüm dosyaların mtime'ı deney öncesi (155 dosya) ve sonrası karşılaştırıldı
— **sıfır fark**. Deneyin ürettiği tüm dosyalar yalnızca
`experiments/entropy_tree_ensemble_deneme/` altında:
`scripts/{precheck,entropy_tree,ensemble,threshold_search,run_experiment}.py`,
`config.yaml`, `README.md`,
`results/{precheck_log.txt,model_comparison.csv,model_comparison_summary.csv,threshold_results.csv,cat1_ablation.csv,run_experiment.log,final_report.md}`.
`figures/` klasörü bu turda boş bırakıldı (bu rapor tablo-tabanlı, ek görsel
üretilmedi — istenirse ayrı bir istekle eklenebilir).

---

## 6. Sonuç: E2 Aday Önerisi

**BELİRSİZ/DÜŞÜK ÖNCELİK — resmi aday olarak önerilmiyor, ama tamamen
reddedilmiyor.**

**Lehte:**
- Model B, tek-ağaç ve rastgele-özellik varyantlarını her metrikte geçiyor
  — ensemble + nested eşik tasarımı **kendi içinde** tutarlı ve mantıklı
  çalışıyor.
- Prevalans-ayarlı metrikte trivial taban çizgisini büyük farkla geçiyor —
  dengesizlik-farkındalığı gerçek.
- Düşük std (0.035 F1) — 50 fold boyunca istikrarlı.

**Aleyhte:**
- Bu ensemble, Aşama D.1'in zaten seçtiği **aynı 25 özelliği** kullanıyor —
  yeni bir bilgi kaynağı değil, yalnızca farklı bir model ailesi. Mevcut/
  gelecek Aşama E adaylarının (LightGBM/XGBoost/CatBoost gibi, zaten
  entropy/information-gain mantığını GBDT çatısında kullanan, çok daha
  olgun ve düzenlileştirilmiş kütüphaneler) bu basit ensemble'ı geçmemesi
  için hiçbir neden yok — doğrudan bir karşılaştırma bu turda yapılmadı
  (kapsam dışı, yalnızca A/B/C/D kendi arasında karşılaştırıldı).
  **Bu, en büyük boşluk: bu ensemble'ın Aşama E'nin diğer aday ailelerine
  karşı nasıl durduğu hâlâ bilinmiyor.**
- `CAT_1` ablasyonunun anlamsız çıkması iki yönlü yorumlanabilir: (a) model
  ailesi `CAT_1`'in sinyalini yeterince kullanamıyor (ensemble'ın bir
  zayıflığı), ya da (b) `CAT_1` gerçekten redundant (proje seviyesinde bir
  bulgu) — bu deney ikisini ayırt edemiyor.
- Yalnızca 4 sabit parametre kombinasyonu test edildi (`max_depth=4` vb.
  hiç taranmadı) — bu, "en iyi entropy-tree ensemble" değil, "bir entropy-
  tree ensemble" sonucu.

**Öneri:** Bu turda E2'ye resmi aday olarak **eklenmesin**; ama sonuçlar
(özellikle prevalans-ayarlı metrikteki tutarlılık) Aşama E'nin model
portföyü tasarlanırken bir "hızlı/basit taban çizgisi" olarak referans
alınabilir. Eğer E2 aşamasında zaten LightGBM/XGBoost/CatBoost gibi daha
güçlü aday aileler test ediliyorsa, bu ensemble'ın ek bir katkı sağlaması
düşük ihtimal — ama doğrudan karşılaştırma yapılmadan bu kesin değil.
