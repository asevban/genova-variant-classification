# PAH Paneli — Aşama E (Modelleme) Öncesi Kapsamlı Denetim

> **Denetim türü:** Karar-destek denetimi. Hiçbir kod/veri/config değiştirilmedi; tüm bulgular
> dosya okuma, `grep`, salt-okunur Python/pytest çalıştırmaları ve web araması ile doğrulandı.
> Kapsam: `src/genova/pah/`, `tests/`, `reports/`, `configs/pah/`, `data/` (provenance),
> `notebooks/`, `CLAUDE.md`, `PROJE_DOSYA_YAPISI_PAH.md`. Aşama E henüz başlamadı; bu denetim
> yalnızca Aşama A-D'yi kapsıyor.
>
> **Kapsam notu:** Görev tanımı `notebooks/02_pipeline_model_ready_PAH.ipynb` adlı bir dosyanın
> incelenmesini istiyordu — bu dosya **repoda mevcut değil** (`notebooks/` yalnızca
> `01_eda_pah.py` ve `id3_followup_analysis.py` içeriyor), ve `model_ready_dataset` adında bir
> dosyaya da hiçbir yerde referans yok. Bu proje eksikliği değil, görev talimatının projenin
> güncel durumuyla uyuşmayan bir varsayımı — düzeltilmeden, olduğu gibi raporlanıyor.

---

## 1. Yönetici Özeti

**Bulgu sayısı: 2 Kritik, 8 Orta, 6 Düşük/Bilgi.**

**Genel değerlendirme: ŞARTLI EVET.** Aşama D.1'in özellik seçimi metodolojisi, split bankası
bütünlüğü, sızıntı disiplini ve resmi metrik (F1) tutarlılığı bu denetimde **yeniden test
edildi ve doğrulandı** — hiçbirinde sayısal bir hata bulunamadı. Ancak **2 Kritik bulgu**
doğrudan CLAUDE.md'nin kendi kırmızı çizgilerini (kural #3 "kalıcı mantık notebook'ta
yaşamaz", kural #5 "her karar izlenebilir") ihlal ediyor ve jüri önünde kolayca yakalanabilir
niteliktedir:

1. Projenin en kritik tek dosyası — final 24-özellik havuzunu tanımlayan
   `v4_final_feature_pool.json` — hiçbir committed script tarafından üretilmiyor; tek-seferlik,
   kayıt dışı bir adımın ürünü.
2. Proje haritası (`PROJE_DOSYA_YAPISI_PAH.md`) en az 6 yerde doğrulanabilir şekilde güncel
   değil (eski özellik sayıları, eski dosya boyutları, kaldırılmış bir gösterge, yanlış bir
   script-çıktı iddiası).

Bu iki bulgu **Aşama E başlamadan önce düzeltilmeli** — düzeltmeleri küçük efor gerektiriyor
(bir script fonksiyonu + belge güncellemesi) ama düzeltilmeden bırakılırsa risk büyük (kod
çalıştırıldığında rapor edilen sonuçlar yeniden üretilemez görünebilir). Bunların dışında
kalan Orta/Düşük bulgular (prior-shift hazırlığı, kaynak-kısayolu taramasının kapsamı,
kalibrasyon/eşik protokolünün somutlaşmamış olması) Aşama E'nin **planına** dahil edilmesi
gereken, ama Aşama E'nin **başlamasını engellemeyen** noktalar.

| Önem | Sayı |
|---|---|
| Kritik | 2 |
| Orta | 8 |
| Düşük / Bilgi (çoğu "sorun bulunamadı" negatif sonucu) | 6 |

---

## 2. Bölüm 1 — İç Tutarlılık Denetimi

### K1 [KRİTİK] — `v4_final_feature_pool.json`'ın üretici script'i yok

**Bulgu:** `dataset_versions.py::build_v4` (ve dolayısıyla `v4_from_v2.parquet`/
`v4_from_v3.parquet`) `reports/tables/v4_final_feature_pool.json` dosyasını **tüketiyor**
(satır 216-218: `if pool_path.exists(): pool = json.loads(...)`), ama bu dosyayı **hiçbir
committed script yazmıyor**. `feature_selection.py::main()` (Aşama D.1'in tek özellik-seçimi
script'i) yalnızca iki dosya yazıyor: `feature_selection_stability.csv` ve
`feature_selection_stable_pool.json` — ikincisi **"≥1 yöntemde stabil"** kuralını kullanıyor
(satır 200: `stability["stable_in_any_method"] = stability["n_methods_stable_at_60pct"] > 0`),
`v4_final_feature_pool.json`'ın kullandığı **"≥2 yöntemde stabil"** kuralını DEĞİL.

**Kanıt:**
- Repo genelinde `grep "open(.*v4_final|to_json.*v4_final|json.dump.*v4_final"` → **sıfır
  eşleşme**. Dosyaya yazan hiçbir kod yok.
- mtime kanıtı: `feature_selection_stability.csv`/`feature_selection_stable_pool.json` →
  2026-08-07 14:26:01; `v4_final_feature_pool.json` → 2026-08-07 **16:25:29** (~2 saat sonra,
  ayrı bir işlem).
- `v4_final_feature_pool.json`'ın `"rationale"` alanı **İngilizce** yazılmış — projenin geri
  kalanı (kod yorumları, rapor, config'ler) tamamen Türkçe. Bu, dosyanın script yerine
  tek-seferlik bir REPL/komut-satırı adımıyla üretildiğinin güçlü bir işareti.
- `PROJE_DOSYA_YAPISI_PAH.md` satır 131 açıkça (ve **yanlış**) şunu iddia ediyor: *"`python -m
  genova.pah.feature_selection` → split bankasını okuyup `reports/tables/feature_selection_*`
  ve `v4_final_feature_pool.json` dosyalarını üretir (~17 dk)."* — bu iddia doğrudan koddan
  çürütüldü.

**İçerik doğru mu?** Evet — `v4_final_feature_pool.json`'daki 24 özellik, `≥2/4` kuralı
`feature_selection_stability.csv`'ye elle uygulandığında çıkan sonuçla **birebir eşleşiyor**
(bu denetimde çapraz doğrulandı: `five_method_stability_comparison.csv`'nin
`in_current_pool_ge2_of_4` bayrağıyla aynı 24 isim). **Sayılar yanlış değil — yalnızca
üretim yolu kayıt dışı.**

**Neden önemli:** CLAUDE.md kural #3 ("kalıcı mantık notebook'ta/script-dışı yaşamaz") ve
kural #5 ("hangi kod bu sonucu üretti sorusu her zaman cevaplanabilmeli") doğrudan ihlal
ediliyor. Jüri `feature_selection.py`'yi yeniden çalıştırıp ardından `dataset_versions.py`'yi
çalıştırırsa: (a) `feature_selection.py` `v4_final_feature_pool.json`'ı **hiç yazmaz**, (b)
`dataset_versions.py` mevcut (eski) dosyayı sessizce kullanmaya devam eder ya da dosya
silinmişse `"v4 atlandi"` mesajıyla adımı atlar. Sonuç: "nasıl üretildi?" sorusuna kod
üzerinden cevap verilemez — tam olarak CLAUDE.md'nin uyardığı senaryo.

**Önerilen aksiyon:** `feature_selection.py::main()`'e (ya da ayrı küçük bir
`select_final_pool.py` script'ine), `stability["n_methods_stable_at_60pct"] >= 2` filtresini
uygulayıp `v4_final_feature_pool.json`'ı yazan ~15 satırlık bir adım ekle; `rationale` metnini
Türkçeye çevir; `PROJE_DOSYA_YAPISI_PAH.md`'yi düzelt.

**Tahmini efor:** Küçük (bir fonksiyon + 2 rapor cümlesi; split bankasına veya mevcut
sonuçlara dokunmuyor).

---

### K2 [KRİTİK] — `PROJE_DOSYA_YAPISI_PAH.md` çok yönlü olarak güncel değil

**Bulgu:** Proje haritası olarak tanımlanan bu dosya, Aşama D Düzeltme Turu'ndaki
düzeltmelerin (27→24 özellik, `AL_296_missing` kaldırma) **hiçbirini** yansıtmıyor. Altı ayrı
doğrulanabilir hata:

| İddia (`PROJE_DOSYA_YAPISI_PAH.md`) | Gerçek (bu denetimde doğrulandı) |
|---|---|
| `v2.parquet`: "369×357", "+seçili gösterge (`AL_296_missing`)" | (369, **356**); `AL_296_missing` kodda ve tüm parquet dosyalarında **yok** (kaldırıldı) |
| `v3.parquet`: "369×457" | (369, **428**) |
| `v4_from_v2/v3.parquet`: "369×30", "27 kararlı özellik" | (369, **27**); **24** özellik |
| `v4_final_feature_pool.json`: "27 özellik" | **24** özellik |
| "`feature_selection.py` ... `v4_final_feature_pool.json` dosyalarını üretir" | **Yanlış** — bkz. K1 |
| "`tests/` — pytest testleri (**27** test, hepsi geçiyor)" | `pytest --collect-only` → **25** (`test_preprocessing_pah.py`) **+ 3** (`test_split_bank_pah.py`) **= 28** (id3 testleri hariç tarihsel taban bile 27 değil, 28) |

**Kanıt:** Her satır doğrudan `pd.read_parquet(...).shape`, `python -m pytest --collect-only
-q`, ve dosya okumasıyla bu denetimde bağımsız olarak yeniden üretildi (yukarıdaki bölümlerde
ayrıntılı).

**Neden önemli:** Bu dosya açıkça *"Yeni bir oturumda veya başka bir kişi tarafından
okunduğunda projenin haritası olarak kullanılabilir"* diye tanımlanmış. Jüri veya yeni bir
takım üyesi bu dosyayı okuyup gerçek dosyalarla karşılaştırırsa (ki bu tam olarak bu denetimin
yaptığı şey), en az 6 yerde tutarsızlık bulur — güven kaybı riski yüksek, özellikle "27 test"
iddiasının kendisi ironik biçimde `02_ON_ISLEME_KARARLARI_PAH.md` §9'un "test sayısı
tutarlılığı" başlığı altında **bir kez zaten "çözüldü" diye kapatılmış bir konu** olması
nedeniyle.

**Önerilen aksiyon:** Dosyadaki tüm sayısal iddiaları yukarıdaki tabloyla güncelle. K1
düzeltildikten sonra script-akış açıklamasını da güncelle.

**Tahmini efor:** Küçük (metin düzenleme, ~10 satır).

---

### Orta-1 — `02_ON_ISLEME_KARARLARI_PAH.md` §9'daki test sayısı da hâlâ yanlış (25≠24)

**Bulgu:** §9, `test_preprocessing_pah.py`'nin **24** test içerdiğini iddia ediyor.
`pytest --collect-only -q tests/test_preprocessing_pah.py` → **25 test**. Bu, id3 testlerinin
eklenmesinden bağımsız, önceden var olan bir sayım hatası (bu rapor bölümü zaten iki kez bir
"tarihsel test sayısı" anlatısı kurmuş — 21→22→24 — ve son basamak yine de yanlış çıkmış).

**Kanıt:** `python -m pytest tests/test_preprocessing_pah.py --collect-only -q` çıktısı →
"25 tests collected".

**Önem: Orta.** Sonucu/kararı etkilemiyor (kozmetik), ama K2 ile aynı kategoriden bir "rapor
kendi iddiasını doğrulamamış" örneği — küçük ama toplu olarak "sayılara güvenilir mi"
izlenimini zedeliyor.

**Aksiyon:** §9'u "25 test" olarak düzelt, `PROJE_DOSYA_YAPISI_PAH.md` ile birlikte.

**Efor:** Küçük.

---

### Orta-2 — `CAT_3`, `CAT_4`, `CAT_5` birebir aynı (duplike) kolonlar — hiçbir rapor/CLAUDE.md'de fark edilmemiş

**Bulgu:** Ham veri üzerinde doğrudan hesaplandı: **`CAT_3 == CAT_4 == CAT_5`, 372 satırın
tamamında (NaN dahil), istisnasız.** `value_counts` üçü için birebir aynı: `G/G=113, C/C=100,
A/A=77, T/T=71, NaN=11`. Bu üç kolon aslında **tek bir sinyalin üç kopyası**, üç bağımsız
kolon değil.

**Kanıt:** `(df["CAT_3"].fillna("_NA_") == df["CAT_4"].fillna("_NA_")).mean() == 1.0` ve aynı
şekilde `CAT_3`↔`CAT_5`, `CAT_4`↔`CAT_5` — üçü de `1.0`. Bu denetimde ham CSV üzerinde
doğrudan doğrulandı (daha önce hiçbir raporda/CLAUDE.md'de bu duplikasyon belgelenmemiş;
CLAUDE.md yalnızca "her biri 4 seviye... nominal kategorik" diyor, kopya olduklarını
belirtmiyor).

**Neden önemli:** `NominalOneHotEncoder(columns=["CAT_3","CAT_4","CAT_5","AA_1","AA_2"])` bu
tek sinyali **3 kez** one-hot'luyor (15 kolon üretiyor, gerçek bilgi 5 kolonluk). v3'ün 57
one-hot kolonundan 10'u saf duplikasyon. **Sonucu değiştirmedi** — final 24-özellik havuzuna
hiçbir `CAT_3/4/5` one-hot kolonu girmedi (yalnızca `CAT_1` var), grup ablation/izole-model
sonuçları (`CAT_` grubunun zaten zayıf sinyalli, ~0.53 AUC olduğu) bu duplikasyondan bağımsız
olarak da geçerli. Ama Aşama D.1'in ara-adım istatistiklerini (stabilite oranları, elastic-net
bootstrap davranışı) hafifçe distort ediyor olabilir — zayıf bir sinyal 3 kopyaya yayılınca
her kopyanın bootstrap alt-örneklemelerinde ayrı ayrı "hayatta kalma" şansı artabilir
(elastic-net'in zaten belgelenmiş 165/405 gevşekliğine küçük bir katkı olası, ama izole test
edilmedi — spekülasyon olarak işaretleniyor).

**Önem: Orta.** Final kararı değiştirmiyor, ama (a) CLAUDE.md'nin "doğrulanmış veri
gerçekleri" bölümüne hiç girmemiş temel bir veri gerçeği eksik, (b) Aşama E'de biri bu üç
kolonu "3 bağımsız genotip" sanıp örn. ayrı ayrı ağırlıklandırma/etkileşim özelliği üretirse
yanlış varsayımla ilerleyebilir.

**Önerilen aksiyon:** CLAUDE.md'nin "PAH paneli — doğrulanmış veri gerçekleri" bölümüne ekle:
*"`CAT_3`, `CAT_4`, `CAT_5` birebir aynı kolon (372/372 satırda identik) — 3 ayrı bilgi değil,
1 sinyalin 3 kopyası."* Aşama E'de `NOMINAL_COLS`'tan `CAT_4`/`CAT_5`'i çıkarıp yalnızca
`CAT_3`'ü tutmak değerlendirilebilir (bu turda **uygulanmadı**, yalnızca öneri).

**Tahmini efor:** Küçük (dokümantasyon); kod değişikliği yapılırsa Orta (v2/v3 yeniden üretimi
+ ilgili testler).

---

### Düşük/Bilgi-1 — `CAT_1`'in çok-değerli anomalisi: "3 bilgi tipi karışımı" hipotezi test edildi ve **reddedildi**

**Bulgu:** Görev, `CAT_1`'in 5 satırlık `&`-birleşik anomalisinin ("gnomADe_AFR&gnomADe_AMR&
...&gnomADe_SAS") üç farklı bilgi tipinin (popülasyon+kalite+arkaik) tek kolonda birleşmesiyle
açıklanıp açıklanamayacağını sordu. **Kanıt karşı yönde:** birleşik string'in **9 token'ının
tamamı** aynı `gnomADe_` önekini taşıyor — homojen tip (popülasyon). Üç farklı bilgi tipi
karışmış olsaydı token'lar farklı formatlarda olurdu (örn. bir kalite-bayrağı token'ı "PASS"
gibi, bir genotip token'ı "G/G" gibi görünürdü) — bu **gözlenmiyor**.

**Sonuç:** Mevcut yorum (`00_DUZELTME_OZETI_PAH.md`: "çoklu popülasyonda gözlenme") **veriyle
doğrulandı**, değişiklik gerekmiyor.

**Önem: Düşük/Bilgi** (negatif sonuç — mevcut belgeleme zaten doğruydu, sadece teyit edildi).

---

### Düşük/Orta-2 — `CAT_` kolonlarının şartname 3-kategori taksonomisiyle eşleştirilmesi (spekülatif, veri-temelli)

Kolon isimleri anonim olduğu için **kesin eşleme mümkün değil**; aşağıdaki değerlendirme
gözlenen değer örneklerine dayanıyor, açıkça spekülatif işaretleniyor:

| Kolon | Gözlenen değerler | En olası kategori | Kanıt gücü |
|---|---|---|---|
| `CAT_1` (24 seviye) | `gnomADe_NFE`, `gnomADg_AFR`, ... | (a) popülasyon etiketi | **Güçlü** — string'ler doğrudan gnomAD alt-popülasyon isimleri |
| `CAT_2` (7 seviye) | `AllofUs_EUR`, `AllofUs_AFR`, ... | (a) popülasyon etiketi | **Güçlü** — All of Us kaynak/popülasyon etiketleri |
| `CAT_3`=`CAT_4`=`CAT_5` (4 seviye, birebir aynı) | `G/G`, `T/T`, `A/A`, `C/C` | (c) arkaik genotip **veya** başka bir tekil-SNP genotip alanı | **Orta** — genotip formatı (b) kalite-bayrağı olasılığını dışlıyor (kalite bayrakları tipik olarak PASS/FAIL/skor formatında olur), ama (a) popülasyon ile (c) arkaik-genotip arasında kesin ayrım yapılamıyor |
| `CAT_6` (%100 eksik) | Hiç gözlenmedi | Belirsiz — olası (b) kalite bayrağı adayı | **Yok** (veri bu dilimde hiç dolu değil, doğrulanamaz) |

**Provenance-confound yorumuna etkisi:** `CAT_1`/`CAT_2`'nin popülasyon-etiketi olduğu güçlü
kanıtla doğrulandığından, `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'nin **mevcut
"kaynak-provenance" çerçevesi geçerliliğini koruyor** — "kalite bayrağı" alternatif yorumu
`CAT_1`/`CAT_2` için **desteklenmiyor**. Ek bulgu: bu denetimde ölçülen `CAT_3`(=4=5)
eksikliğinin `al_all_missing=1` alt-kümesindeki oranı yalnızca **%7.6** (7/92) — genel orana
(%3.0) göre ~2.6× fazla ama `CAT_1`/`CAT_2`'nin **neredeyse tam örtüşmesine** (%100, Cramér's
V=0.755) kıyasla **çok daha zayıf**. Bu, `CAT_3/4/5`'in `CAT_1`/`CAT_2`'den **farklı bir
kaynak/mekanizma** taşıdığı hipotezini destekliyor (belki gerçekten arkaik-genotip gibi ayrı
bir bilgi ekseni) — ama kesin değil.

**Önem: Düşük-Orta** (spekülatif, kod değişikliği gerektirmiyor; `03b`'ye bir dipnot olarak
eklenebilir).

**Aksiyon:** `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'ye bu tabloyu ve yorumu bir ek not
olarak ekle (rapor bu denetimde değiştirilmedi, öneri olarak bırakılıyor).

---

### Düşük/Bilgi-3 — MCC/F1 metrik tutarlılığı: **sorun bulunamadı**

**Bulgu:** `preprocessing_experiments.py::main()` satır 103'te ölçekleyici seçimi doğrudan
`.idxmax()` ile **`f1_pathogenic_mean`** üzerinden yapılıyor (MCC değil). `feature_selection.py`'nin
4 yöntemi de MCC kullanmıyor: mutual information (bilgi kazancı skoru), GBDT importance
(`feature_importances_`), elastic-net (bootstrap'larda sıfırdan-farklı katsayı sayısı),
permutation importance (`scoring="roc_auc"`). **Hiçbir yerde MCC'ye göre bir seçim/ayarlama
yapılmıyor** — MCC yalnızca tamamlayıcı/raporlama istatistiği olarak kullanılıyor, tam
şartnamenin (Bölüm 7.3) istediği gibi.

**Önem: Düşük/Bilgi** (negatif sonuç, doğrulanmış — ama görev istediği için raporlanıyor).

---

## 3. Bölüm 2 — Metodolojik Sağlamlık Denetimi

### Düşük/Bilgi-4 — Sızıntı taraması: `fold_features.py`, `feature_selection.py`, `split_bank.py` — **sorun bulunamadı**

- `fold_features.build_fold_features`: her adım (`BlockMissingIndicator`,
  `MedianImputerWithIndicator`, `ConstantFillImputer`, `FrequencyEncoder`,
  `NominalOneHotEncoder`) `.fit(train_df)` ile eğitim satırlarında fit ediliyor, `test_df`'e
  yalnızca `.transform()` uygulanıyor — kod satır satır okunarak doğrulandı, sızıntı yok.
- `feature_selection.py`: `Label` yalnızca stratifikasyon hedefi ve model `y` girdisi olarak
  kullanılıyor; hiçbir groupby/agregasyon istatistiğine sızmıyor.
- `split_bank.py`: `StratifiedGroupKFold(...).split(df, df["Label"], df["group_id"])` — `Label`
  yalnızca stratifikasyon için (beklenen, standart kullanım), başka hiçbir yerde işlenmiyor.

**Önem: Düşük/Bilgi** (negatif sonuç, doğrulanmış).

---

### Orta-3 — Prior-shift hazırlığı: hiç test edilmemiş (ama açıkça ertelenmiş bir konu)

**Bulgu:** Repo genelinde `saerens|prior_shift|calibrat|adversarial` taraması yalnızca `.md`
dosyalarında (planlama metni) eşleşiyor — **hiçbir `.py` dosyasında** (`src/`, `tests/`,
`notebooks/`) bu kavramların bir uygulaması yok. Bugüne kadarki tüm CV/deney sonuçları
standart stratified split kullanıyor (train dağılımını, %83 patojenik, yansıtan test
fold'larıyla ölçülüyor) — final beklenen dağılımı (~%28.6 patojenik) simüle eden **hiçbir
deney yapılmamış**.

**Not — görev talimatının bir varsayımı düzeltiliyor:** Görev metni "projenin zaten
kullandığı yaklaşımlarla (Saerens-Latinne-Decaestecker, adversarial validation, Clopper-Pearson
CI)" diyor — bu **yanlış bir öncül**. `Saerens` ve `Clopper` terimleri repoda **hiçbir yerde**
geçmiyor (doğrulandı, sıfır eşleşme). Adversarial validation yalnızca `03b`'de **planlanmış**
(Aşama F'ye bağlanmış), henüz **uygulanmamış**. Bu üçü "zaten kullanılıyor" değil, "henüz
kullanılmamış, kısmen planlanmış" durumda — rapor bu gerçeği olduğu gibi yansıtıyor,
uydurma/varsayılan bir kullanım iddia etmiyor.

**Önem: Orta.** CLAUDE.md zaten bunu "ileride kalibrasyon/eşik aşamasında kritik olacak" diye
not düşmüş (yani "hiç düşünülmemiş" değil), ama Aşama E'nin CV protokolüne **somut bir adım
olarak** henüz girmemiş.

**Önerilen aksiyon:** Aşama E'nin nested CV tasarımına, iç döngüde (ya da ayrı bir
duyarlılık-analizi adımında) sentetik/simüle-edilmiş benign-ağırlıklı bir test alt-kümesi
üzerinde mevcut yöntemlerin (özellik havuzu, model) nasıl davrandığını ölçen bir kontrol ekle.
Posterior-ayarlama için **Saerens, Latinne & Decaestecker (2002)** yöntemi ("Adjusting the
outputs of a classifier to new a priori probabilities: a simple procedure", *Neural
Computation*) somut, doğrulanmış bir aday — bkz. Bölüm 4.

**Tahmini efor:** Orta (Aşama E'nin protokol tasarımına yeni bir adım; kod değişikliği bu
denetimin kapsamı dışı, yalnızca planlama önerisi).

---

### Orta-4 — Kaynak-kısayolu (source-shortcut) taraması yalnızca `al_all_missing` için yapılmış, havuzdaki diğer 23 özellik için değil

**Bulgu:** `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md` yalnızca `al_all_missing` ×
`CAT_1`/`CAT_2` ilişkisini test ediyor. Final 24-özellik havuzundaki diğer özellikler
(`EK_7`, `AL_49`, `AL_300`, `AL_26`, vb.) için `CAT_1`/`CAT_2` (kaynak/popülasyon etiketi)
ile bağımsız bir ilişki testi **yapılmamış**. `03d_ID3_TAKIP_ANALIZI_PAH.md` bunu kısmen
genişletti (`CAT_1`'in ID3 ağaçlarındaki yapısal konumu) ama sistematik bir "24 özelliğin
tamamı × `CAT_1`/`CAT_2`" tarama tablosu hâlâ yok.

**Önem: Orta.** Aşama F'nin adversarial validation adımı muhtemelen bunu doğal olarak
kapsayacak, ama plana **açıkça yazılmalı**: "yalnızca `al_all_missing` değil, final havuzun
tamamı test edilecek."

**Önerilen aksiyon:** Aşama F planına bu genişletmeyi ekle (bu denetimde uygulanmadı, yalnızca
plan notu).

**Tahmini efor:** Küçük (plan notu); Aşama F'de gerçek analiz Orta efor.

---

### Düşük/Bilgi-5 — Split bankası/`GroupKFold` bütünlüğü: spot-check ile **yeniden doğrulandı, sorun yok**

**Bulgu (bu denetimde bağımsız ölçüldü, önceki rapora güvenilmeden):**
- 3 tekrar (repeat 0, 4, 9) × 5 dış fold = 15 fold spot-check: `conflict_group_1`
  (`VAR_003238`/`VAR_003234`) her fold'da **0 ihlal** ile aynı tarafta.
- Tüm 50 dış fold'da `n_train + n_test = 369` (tam, çakışmasız) — doğrulandı.
- Spot-check'teki test fold'larında patojenik oranı **0.743–0.904** aralığında (önceki
  raporun iddia ettiği 0.70–0.95 aralığıyla tutarlı).

**Önem: Düşük/Bilgi** (negatif sonuç, bağımsız olarak yeniden doğrulandı).

---

### Orta-5 — Küçük-n güvenilirliği: n≈p rejimi, literatürle karşılaştırma

**Bulgu:** Aşama D.1'de dış eğitim fold'u ~295-296 satır, aday özellik evreni 405 (final 24) —
klasik bir n≈p/n<p (az-belirlenmiş) rejim. Meinshausen & Bühlmann'ın (2010) orijinal
"Stability Selection" makalesi (*JRSS-B*) %50 alt-örnekleme kullanıyor; proje %80 kullanıyor
(`ELASTICNET_SUBSAMPLE_FRAC=0.8`) — farklı ama makul bir seçim. Makalenin teorik hata-kontrolü
(PFER — yanlış-seçilen özellik sayısının beklenen üst sınırı), seçilen eşiğe (`π`) ve aday
özellik sayısına (`p`) duyarlı; projenin kendi elastic-net bulgusu (165/405'in kendi %60
eşiğini geçmesi — çok gevşek) bu teorik hassasiyetin **pratikte gözlemlendiğinin somut
kanıtı**. **≥2-yöntem konsensüs kuralı**, tek-yöntem stability selection'ın bilinen zayıflığına
karşı literatürle uyumlu, makul bir düzeltme (birden fazla bağımsız yöntemin kesişimi, PFER
kontrolünü informal ama pratik şekilde güçlendiriyor).

**Önem: Orta/Bilgi** — mevcut yaklaşım literatürle **uyumlu**, ama resmi bir sayısal
hata-sınırı (PFER) hiç hesaplanıp raporlanmamış. Bunu eklemek düşük efor, yüksek jüri-güven
katkısı sağlar.

**Önerilen aksiyon:** Meinshausen-Bühlmann formülüyle (`q_Λ`, `π` ve `p`'den) final havuz için
yaklaşık bir PFER üst sınırı hesaplayıp `03_OZELLIK_SECIMI_PAH.md`'ye ek not olarak ekle.

**Tahmini efor:** Küçük (mevcut sayılardan türetilebilecek bir hesap, yeni deney gerekmiyor).

---

## 4. Bölüm 3 — Literatür ve Dış Kaynak Karşılaştırması

Her kaynak doğrulanabilir şekilde isimlendirildi (arama motoruyla teyit edildi); doğrulanamayan
hiçbir kaynak "gerçekmiş gibi" sunulmadı.

### 4.1 Eksik veri işleme — doğrudan destekleyici kanıt bulundu

**Kaynak:** Särkkä, Myöhänen, Marinov, Saarinen, Lahti, Fortino & Paananen (2025), *"Comparison
of missing data handling methods for variant pathogenicity predictors"*, **NAR Genomics and
Bioinformatics** (ilk olarak bioRxiv 2022.06.17.496578). AMISS çatısı altında 14 eksik-veri
yöntemi (6 basit imputasyon, 4 MICE varyantı, 3 diğer, + eksiklik-göstergesi genişletmesi)
karşılaştırılmış.

**Bulgu:** *"Sofistike eksik veri yöntemleri kullanmak gerekli değil — basit
koşulsuz imputasyon yöntemleri, hatta sıfır-imputasyonu bile, daha yüksek performans veriyor
ve önemli ölçüde hesaplama zamanı tasarrufu sağlıyor."*

**Projeye uygulanabilirlik:** **Doğrudan destekleyici.** Projenin `AL_` için sıfır-doldurma +
`EK_3` için medyan-doldurma + blok eksiklik göstergesi yaklaşımı, bu makalenin bulgularıyla
**tam örtüşüyor** — literatür, projenin zaten yaptığı basit-imputasyon tercihini
doğruluyor. Ek olarak makale, eksiklik-göstergesi genişletmesini de ayrı bir yöntem olarak test
etmiş (proje de bunu blok-seviyesinde kullanıyor) — iki yaklaşımın birleşimi (basit imputasyon
+ seçici gösterge) literatürün doğrudan önerdiği kombinasyona yakın.

### 4.2 VEP geliştirme/yayınlama en iyi pratikleri

**Kaynak:** Livesey, Badonyi, Dias, Frazer, Kumar, Lindorff-Larsen, McCandlish, Orenbuch,
Shearer, Muffley, Foreman, Glazer, Lehner, Marks, Roth, Rubin, Starita & Marsh (2024/2025),
*"Guidelines for releasing a variant effect predictor"*, arXiv:2404.10807 (sonrasında Genome
Biology'de yayınlandı). Büyük, tanınmış bir yazar konsorsiyumu (Debora Marks, Frederick Roth,
Joseph Marsh gibi alanın önde gelen isimleri dahil).

**Bulgu:** Açık-kaynak erişilebilirlik, şeffaf metodoloji, standart ölçekli/yorumlanabilir
skor sunumu, ve **eğitim verisinin titiz açıklanması** vurgulanıyor.

**Projeye uygulanabilirlik:** **Kısmen uygulanabilir.** Proje zaten config-driven versiyon
üretimi ve `leakage_caveat` alanlarıyla "eğitim verisinin titiz açıklanması" ilkesini
uyguluyor. Ama K1/K2 bulguları (izlenebilirlik kopukluğu) tam olarak bu rehberin vurguladığı
"şeffaf metodoloji" ilkesine aykırı — bu literatür referansı, K1/K2'nin düzeltilmesi için ek
bir dışsal gerekçe sağlıyor.

### 4.3 Veri sirkülerliği (data circularity) — projenin kendi endişesine akademik bir çerçeve

**Bulgu (birden fazla kaynaktan, genel/yerleşik bir kavram olarak doğrulandı — VEP
benchmarking literatüründe standart terminoloji):** "Tip 1 sirkülerlik" (değerlendirmede
kullanılan varyantların eğitimde de kullanılmış olması) ve VEP-destekli ClinVar
etiketlemesinin kendisinin yeni bir yanlılık kaynağı olabileceği (etiketleme insan-kaynaklı
olmayabilir) literatürde iyi belgelenmiş bir sorun.

**Projeye uygulanabilirlik:** **Doğrudan uygulanabilir, terminoloji katkısı.** Projenin
`al_all_missing`/`CAT_1`/`CAT_2` "kaynak-provenance riski" olarak adlandırdığı endişe, VEP
literatüründeki "veri sirkülerliği" kavramının bir örneği. Aşama F'nin adversarial validation
planına bu terminolojiyi (Tip 1/Tip 2 sirkülerlik ayrımı) eklemek, jüri karşısında "biz bunu
biliyoruz ve adı var" güvenini artırır — proje zaten doğru şeyi sezgisel olarak yapıyor, isim
vermek ek bir jüri-iletişim faydası sağlar.

### 4.4 Küçük-n dengesiz tablo yarışmaları — ICR (Kaggle)

**Kaynak:** "ICR — Identifying Age-Related Conditions" Kaggle yarışması (küçük-n, dengesiz
sınıf, tıbbi tablo verisi — PAH paneline yapısal olarak en yakın bulunan genel yarışma).

**Bulgu:** TabPFN (küçük tablo problemleri için tasarlanmış transformer), hız ve
kendi-normalize-etme özellikleri nedeniyle yarışmacılar arasında yaygın kullanılmış; sınıf
dengesizliği için hibrit örnekleme (azınlık sınıfını yukarı, çoğunluk sınıfını aşağı örnekleme)
kullanılmış.

**Projeye uygulanabilirlik:** **Kısmen zaten planlanmış.** `PROJE_DOSYA_YAPISI_PAH.md`
(v3.parquet açıklaması) zaten *"ölçek-duyarlı modeller (lojistik regresyon, **TabPFN**,
RealMLP) için"* diyerek TabPFN'i öngörmüş — bu literatür bulgusu bu seçimi **doğrudan
destekliyor**. Hibrit up/down-sampling ise projenin şu anki `class_weight="balanced"`
yaklaşımına bir **alternatif/ek** olarak Aşama E'nin model-seçim aşamasında değerlendirilebilir
— düşük efor, potansiyel orta getiri.

### 4.5 Küçük-örneklem oran tahmini için istatistiksel pratik

**Kaynak:** Clopper & Pearson (1934) — temel/klasik "exact" binom güven aralığı yöntemi (görev
metninde adı geçti, ama repoda hiç kullanılmıyor — bkz. Bölüm 2, "Saerens/Clopper" taraması).

**Projeye uygulanabilirlik:** **Doğrudan uygulanabilir, düşük efor.** Yalnızca 62 benign
örnek olduğu için, Aşama E/F'de raporlanacak herhangi bir oran-metriği (specificity, MCC'nin
bileşenleri) normal-yaklaşıklı (Wald) güven aralığı yerine Clopper-Pearson "exact" aralığıyla
raporlanmalı — küçük n'de Wald aralıkları güvenilmez şekilde dar çıkabilir. Bu, mevcut hiçbir
kod veya raporda **uygulanmıyor** (görev metninin "zaten kullanılıyor" varsayımı yanlıştı,
bkz. Orta-3).

---

## 5. Bölüm 4 — Hiç Düşünülmemiş Olabilecek Konular

### Orta-6 — Kalibrasyon: kavram var, somut yöntem yok

CLAUDE.md'de "kalibrasyon" kelimesi **3 kez geçiyor** (kural #1, #2, ve genel giriş) — yani
konsept tamamen "hiç düşünülmemiş" değil. Ama **hiçbir somut yöntem adı** (Platt scaling,
isotonic regression, temperature scaling vb.) hiçbir dosyada geçmiyor. **Önem: Orta.**
**Aksiyon:** Aşama E planına, hangi kalibrasyon yönteminin nested iç döngüde deneneceğine dair
somut bir madde eklenmeli.

### Orta-7 — Eşik ayarlama (threshold tuning): protokol taslağı yok

Benzer şekilde, CLAUDE.md kural #1/#2 eşik seçimini "yalnızca eğitim fold'unda fit edilir"
diye genel kuralla kapsıyor, PDR'deki eski eşik (0,359) yalnızca tarihsel referans olarak
anılıyor (kural #6). Ama **hangi metrik üzerinden** (F1 doğrudan mı, yoksa
precision-recall eğrisi üzerinden mi), **hangi CV katmanında** (iç mi dış mı) eşiğin
aranacağına dair somut bir taslak henüz hiçbir dosyada yok. **Önem: Orta.**

### Düşük/Bilgi-6 — Panel-özgü vs panel-ortak mimari: **zaten karar verilmiş**, ek aksiyon gerekmiyor

CLAUDE.md açıkça ve kesin bir dille şunu söylüyor: *"Dört bağımsız panel var: MASTER, KANSER,
PAH, CFTR — her biri ayrı modellenir, birleştirilmez."* Bu soru **zaten cevaplanmış** bir karar
— "hiç düşünülmemiş konu" olarak sunmak yanlış olur. Panel-ortak bir mimari (örn. multi-task
learning) literatürde var olan bir teknik olsa da, projenin kendi kesin kuralı gereği **kapsam
dışı**. Bu maddede yeni bir aksiyon **önerilmiyor**.

### Orta-8 — Açıklanabilirlik (SHAP vb.) ve ön işlemenin tersine-çevrilebilirliği: gerçek bir boşluk olabilir

Repoda `SHAP`, `açıklanabilirlik`, `interpretability` kelimelerine (veya TEKNOFEST jüri
sunumunun bu konuda ne istediğine dair) hiçbir açık referans bulunamadı. Bu, ya (a) şartname
bunu gerektirmiyor ya da (b) gerçekten hiç düşünülmemiş bir gereksinim. Mevcut özel
transformer sınıflarının (`Log1pTransformer`, `LogitTransformer`, `FrequencyEncoder`) hiçbiri
`inverse_transform` metodu sunmuyor — yalnızca `ColumnwiseScaler`'ın sardığı sklearn native
`RobustScaler`/`StandardScaler` bu metodu miras yoluyla taşıyor. Eğer final teslim bir SHAP
analizi veya "özellik X şu kadar katkı sağladı, orijinal ölçekte bu şu anlama geliyor" tarzı bir
yorumlanabilirlik çıktısı gerektiriyorsa, dönüşümlerin tersine çevrilmesi gerekecek ve şu anki
kod tabanı buna hazır değil.

**Önem: Orta** (gerçekliği TEKNOFEST şartnamesine bağlı, doğrulanamadı — takım kaptanının
şartnameyi kontrol etmesi gerekiyor).

**Aksiyon:** Şartnamede açıklanabilirlik/SHAP gereksinimi olup olmadığını netleştir; varsa,
`transforms.py`/`encoding.py` sınıflarına `inverse_transform` eklenmesi Aşama E öncesi planlanmalı.

---

## 6. Öncelik Sıralı Aksiyon Listesi

| # | Bulgu | Önem | Efor | Aksiyon |
|---|---|---|---|---|
| 1 | K1 — `v4_final_feature_pool.json` üretici script yok | **Kritik** | Küçük | `feature_selection.py`'ye (veya ayrı script'e) ≥2-yöntem filtresi + dosya yazımı ekle |
| 2 | K2 — `PROJE_DOSYA_YAPISI_PAH.md` çok yönlü stale | **Kritik** | Küçük | Tabloda verilen 6 düzeltmeyi uygula |
| 3 | Orta-1 — `02_ON_ISLEME_KARARLARI_PAH.md` §9 test sayısı (25≠24) | Orta | Küçük | Sayıyı düzelt |
| 4 | Orta-2 — `CAT_3`/`CAT_4`/`CAT_5` duplikasyonu belgesiz | Orta | Küçük (dok.) / Orta (kod) | CLAUDE.md'ye veri gerçeği ekle; Aşama E'de kolon indirgeme değerlendir |
| 5 | Orta-3 — Prior-shift hiç test edilmemiş | Orta | Orta | Aşama E CV protokolüne SLD-tabanlı duyarlılık kontrolü ekle |
| 6 | Orta-4 — Kaynak-kısayolu taraması yalnızca `al_all_missing`'de | Orta | Küçük (plan) | Aşama F planına "havuzun tamamı" genişletmesini yaz |
| 7 | Orta-5 — PFER üst sınırı hiç hesaplanmamış | Orta | Küçük | Mevcut sayılardan PFER hesapla, `03_OZELLIK_SECIMI_PAH.md`'ye ekle |
| 8 | Orta-6/7 — Kalibrasyon/eşik protokolü somutlaşmamış | Orta | Orta (planlama) | Aşama E planına somut yöntem adları ekle |
| 9 | Orta-8 — Açıklanabilirlik gereksinimi belirsiz | Orta | Küçük (araştırma) | Şartnameyi kontrol et, gerekirse `inverse_transform` planla |
| 10 | Düşük/Orta — `CAT_` taksonomi eşleştirmesi | Düşük-Orta | Küçük | `03b`'ye dipnot ekle (opsiyonel) |
| — | Diğer tüm Düşük/Bilgi maddeleri (sızıntı, split-bank, MCC/F1, CAT_1 anomali, panel-mimari) | Düşük/Bilgi | — | Aksiyon gerekmiyor — negatif sonuçlar zaten doğrulandı |

**Genel tavsiye:** Madde 1 ve 2 (Kritik) Aşama E başlamadan önce düzeltilmeli — ikisi de küçük
efor. Madde 3-9 (Orta) Aşama E'nin **planına** yazılabilir, Aşama E'nin **başlangıcını**
engellemez; bunlardan 5, 6, 8 doğrudan Aşama E'nin CV/kalibrasyon tasarımının bir parçası
olacağı için, planlama en geç Aşama E'nin ilk taslağıyla birlikte yapılmalı.
