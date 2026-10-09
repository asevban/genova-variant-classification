# PAH Paneli — Aşama A: Kapsamlı EDA Raporu

**Ham veri:** `data/raw/YARISMA_TRAIN_PAH.csv` — 372 satır, 353 kolon.
**Üretim script'i:** `notebooks/01_eda_pah.py` (yalnızca keşif/görselleştirme; kalıcı mantık yok). Tüm figürler `reports/figures/`, tüm ham çıktı tabloları `reports/tables/` altındadır; bu rapor onlara atıfla yazılmıştır. Seed=42.

---

## A.1 — Şema Doğrulama

| Kontrol | Sonuç |
|---|---|
| Satır × Kolon | 372 × 353 |
| `Variant_ID` benzersiz mi | Evet, 372/372, null yok |
| `Label` değer kümesi | {0, 1}, null yok |
| `AL_` kolon sayısı | 334 |
| `CAT_` kolon sayısı | 6 |
| `EK_` kolon sayısı | 9 |
| `AA_` kolon sayısı | 2 |
| dtype dağılımı | 344 × float64, 8 × object (Variant_ID + CAT_1..CAT_6 + AA_1/AA_2 karışık), 1 × int64 (Label) |

Kaynak: `reports/tables/schema_validation.json`. Beklenen 353 kolon (`Variant_ID` + `Label` + 334+6+9+2 = 351) ile satır satır uyuşuyor: 2+334+6+9+2 = 353. ✅ CLAUDE.md'deki önceden doğrulanmış gerçeklerle birebir örtüşüyor.

**Sentinel tarama:** -999, -9999, 9999, 999, -1 literal değerleri sayısal kolonlarda taranmıştır, hiçbiri bulunmamıştır — eksiklik doğrudan NaN olarak kodlanmış. (Bu, `preprocessing.py` içindeki `scan_sentinels` fonksiyonuyla kod olarak da doğrulanabilir hale getirilmiştir, bkz. Aşama B.)

---

## A.2 — Hedef Dengesizliği

Eğitim setinde **310 patojenik (%83.3) / 62 benign (%16.7)**. Final yarışma test setinde beklenen patojenik oranı **≈%28.6** — yön tersine dönüyor (eğitimde patojenik çoğunluk, finalde beklenen azınlık). Bu fark bu görevde ele alınmayacak (kalibrasyon/eşik ayrı görev), ama:

- Nested CV tasarımında **stratifikasyon eğitim dağılımına göre** yapılacak (Aşama D.0), performans metriği (F1, ileride) eğitim dağılımında ölçülecek — final dağılımdaki performans farklı olabilir, bu normal ve beklenen.
- Özellik seçimi ve model karmaşıklığı kararları bu kayma göz önünde bulundurularak **muhafazakar** tutulmalı (aşırı uyum, azınlık sınıfı olan benign'e özel önem verilmeli — zaten 62 örnek düşük hacim nedeniyle risk taşıyor).

Görsel: `reports/figures/target_imbalance.png`

---

## A.3 — Eksiklik Analizi (Çok Katmanlı)

### A.3.1 Kolon ve satır bazlı eksiklik

- `AL_` grubu kolon bazlı eksiklik: min ≈%27 — maks ≈%96 (`reports/tables/missing_rate_by_column.csv`, `reports/figures/missing_rate_by_column.png`).
- Satır bazlı eksiklik oranı dağılımı: `reports/figures/missing_rate_by_row.png` — belirgin çok-modlu dağılım (bir grup satır ~%0 eksik, bir grup satır ~%95+ eksik → AL_ tamamen eksik bloğu).

### A.3.2 Eksiklik-deseni kümeleme

`AL_`+`EK_` ikili eksiklik matrisi üzerinde Jaccard mesafesiyle hiyerarşik kümeleme (6 küme), sonuç `reports/tables/missing_pattern_clusters.csv`:

| Küme | n | Patojenik oranı | AL_ tümü-eksik oranı |
|---|---|---|---|
| 1 | 8 | 0.625 | 0.00 |
| 2 | 4 | **1.000** | 0.00 |
| 3 | 21 | 0.714 | 0.00 |
| 4 | 242 | 0.822 | **0.380** (=92/242) |
| 5 | 75 | 0.907 | 0.00 |
| 6 | 22 | 0.864 | 0.00 |

Küme 4 içindeki 92 satır tam olarak bilinen "tüm `AL_` eksik" bloğu ile örtüşüyor (372 satırın 92'si, %24.7) — **doğrulandı**. Küme 2 (n=4, %100 patojenik) dikkat çekici ama örneklem çok küçük (n=4), genelleştirilebilir bir sinyal olarak güvenilmemeli; sadece not düşülüyor.

**Gizli ikinci bir blok bulundu:** `AL_` kolonlarının ikili eksiklik "parmak izi" (hangi 372 satırda eksik) incelendiğinde, 334 kolon yalnızca **14 benzersiz eksiklik desenine** indirgeniyor (`AL_` kolonları tek tek değil, gruplar halinde birlikte eksik/gözlenmiş). Bu, `AL_` grubunun kaynak veritabanı düzeyinde (muhtemelen gnomAD alt-tablosu bazında) blok halinde eksik olduğunu gösteriyor — kör kolon-bazlı imputasyon yerine blok yapısını koruyan bir yaklaşım (blok göstergesi + sıfır doldurma) bu nedenle doğru tercih.

Ayrıca `EK_1,2,4,5,6,7,8,9` (EK_3 hariç) ve `CAT_3,4,5` **aynı 11 satırda birlikte eksik** (missing_rate=%2.96, tam örtüşen desen) — bu, `AL_` bloğundan bağımsız, ayrı bir "kaynak satırında hiç genotip/korunmuşluk verisi yok" bloğu. Bu 11 satırın 7'si aynı zamanda `AL_` tümü-eksik 92 satırlık bloğun içinde, 4'ü değil — yani iki blok kısmen ama tam olarak örtüşmüyor, birbirinden bağımsız iki eksiklik mekanizması var.

### A.3.3 Bilgilendirici eksiklik testi

Her `AL_`/`EK_`/`CAT_` kolonunun eksiklik göstergesi (0/1) ile `Label` arasında point-biserial korelasyon (`reports/tables/missingness_label_association.csv`):

- **p<0.05 olan kolon sayısı: 53/348** (41 `AL_`, 9 `EK_` — tüm EK_ grubu, 3 `CAT_` — CAT_3/4/5).
- `EK_1,2,4-9` ve `CAT_3,4,5` için ortak p=0.0092, r=-0.135 (yukarıdaki 11-satırlık ortak blok nedeniyle — bağımsız 12 test değil, tek bir ortak sinyal).
- `AL_` grubunda p<0.05 olan 41 kolon da benzer şekilde birkaç ortak blok deseninden geliyor (bkz. A.3.2 — 14 benzersiz desen), bağımsız 41 sinyal değil.

**Önemli metodolojik not:** Bu p-değerleri çoklu-karşılaştırma düzeltmesi (Bonferroni/FDR) uygulanmadan ham haliyle raporlanmıştır ve birçoğu aynı alttaki birkaç bloktan (14 AL_ deseni + 1 EK/CAT bloğu) türediği için **istatistiksel olarak bağımsız değildir**. Aşama B'de eksiklik göstergesi seçimi, kolon sayısını değil **benzersiz blok sayısını** temel alacak (kör 41+9+3 gösterge eklenmeyecek).

### A.3.4 Yalnızca-eksiklik ayırt edicilik testi

- Tek kolon bazında maksimum eksiklik-göstergesi AUC'si **0.635** (birkaç AL_ kolonu, ortak blok nedeniyle aynı değeri paylaşıyor) — tek başına orta-düşük.
- `AL_` tümü-eksik blok göstergesi tek başına AUC = **0.619**.
- **Tam eksiklik matrisi (334+9+5 = 348 kolonun yalnızca 0/1 eksiklik göstergesi, öznitelik değerleri hariç) ile 5-fold CV lojistik regresyon AUC = 0.752 (±0.066).** Bu, eksiklik deseninin TEK BAŞINA etiketi orta-güçlü düzeyde tahmin edebildiğini gösteriyor — dürüstçe belirtilmesi gereken bir bulgu. Bu, "popülasyon veritabanında hiç gözlenmeme = nadirlik/patojenite sinyali" hipotezini destekliyor (CLAUDE.md'deki öneriyle tutarlı) ama aynı zamanda modelin eksiklik desenine aşırı güvenmemesi için Aşama D'de grup ablation/izole model testleriyle çapraz doğrulanmalı.

Kaynak: `reports/tables/missingness_indicator_auc.csv`, `reports/tables/missingness_only_cv_auc.json`, `reports/tables/block_missing_auc.json`.

### A.3.5 Tek-özellik AUC taraması (döngüsellik/meta-prediktör tespiti)

`AL_`+`EK_` kolonlarının her biri, mevcut olduğu satırlarda `Label` ile tek değişkenli AUC'ye göre sıralandı (`reports/tables/univariate_auc_scan.csv`, ilk 30 kolon `reports/figures/univariate_auc_top30.png`).

**En yüksek AUC: `EK_7` = 0.715.** İkinci sırada birkaç `AL_` kolonu 0.64–0.69 aralığında.

**Şüpheli kolon (AUC>0.90) BULUNMADI** (`reports/tables/univariate_auc_suspicious.csv` boş). Bu, veri setinde gizlenmiş bir meta-prediktör (REVEL/CADD/PolyPhen benzeri, ClinVar ile döngüsel ilişkili bir skor) **olmadığına dair kanıt** — hiçbir tek kolon etiketi tek başına neredeyse mükemmel tahmin etmiyor. `EK_7`'nin göreli yüksekliği (0.715) makul bir evrimsel korunmuşluk sinyali olarak yorumlanabilir, döngüsellik şüphesi için yetersiz. Bu bulgu Aşama D'deki izole-grup model sonuçlarıyla çapraz doğrulanacak.

---

## A.4 — Yinelenen / Çelişkili Satır Analizi

Profil karşılaştırması `Variant_ID` hariç tüm 351 öznitelik kolonu üzerinden **tam string eşleşmesi** ile yapıldı (352 satır × 352 satır floating-point tolerans belirsizliği olmadan, tam ondalık string temsili karşılaştırıldı).

**Bulgu — CLAUDE.md'deki iki ayrı görünen gerçek aslında tek bir iç içe geçmiş yapı:**

1 adet **5 satırlık "çelişkili profil üst-kümesi"** bulundu — bu 5 satırın `Variant_ID`/`Label` hariç tüm öznitelik profili birebir aynı:

| Variant_ID | Label |
|---|---|
| VAR_003238 | 1 |
| VAR_003234 | 0 |
| VAR_003237 | 1 |
| VAR_003225 | 0 |
| VAR_003241 | 1 |

Bu üst-küme, profil+etiket birlikte karşılaştırıldığında **2 alt-gruba ayrışıyor**:
- **3 satırlık tam-yinelenen alt-grup (Label=1):** VAR_003238, VAR_003237, VAR_003241 — profil ve etiket birebir aynı.
- **2 satırlık tam-yinelenen alt-grup (Label=0):** VAR_003234, VAR_003225 — profil ve etiket birebir aynı.

Yani CLAUDE.md'deki "3 tam yinelenen satır" gerçeği, bu 5 satırlık üst-kümenin Label=1 olan alt-grubudur; "5 satırlık çelişkili grup" ise bu iki alt-grubun (3+2) birleşimidir. **Bu ikisi bağımsız fenomen değil, iç içedir.**

`pandas.duplicated(subset=öznitelikler+Label, keep="first")` standart dedup işlemi uygulandığında **tam olarak 3 satır** düşer (3-grubundan 2 fazlalık + 2-grubundan 1 fazlalık = 3), geriye **2 temsilci satır** (VAR_003238 [Label=1], VAR_003234 [Label=0]) kalır — bunlar hâlâ aynı profile sahip ama farklı etiketli, yani **giderilemez etiket belirsizliği** dedup sonrası da devam ediyor. Bu 2 satır Aşama C'de ortak bir `group_id` alacak, Aşama D.0'daki split bankasında GroupKFold ile aynı fold'da tutulacak.

Başka tam yinelenen/çelişkili grup bulunamadı (372 satırın 367'si benzersiz profil).

Kaynak: `reports/tables/conflicting_profile_supergroups.csv`, `reports/tables/full_duplicate_subgroups.csv`, `reports/tables/duplicate_conflict_summary.json`.

### Yakın-kopya taraması (caveat ile)

`AL_`+`EK_` sayısal kolonları medyan-doldurma + standardizasyon sonrası en-yakın-komşu (Euclidean) taraması yapıldı (`reports/tables/near_duplicate_candidates.csv`). En yakın 7 satır (mesafe ≈1e-7, pratikte sıfır) tam olarak yukarıdaki 5-satırlık çelişkili grup + 2 fazladan satır (VAR_002796, VAR_002977) olarak çıktı.

**Önemli caveat:** VAR_002796 ve VAR_002977'nin `AA_1`/`AA_2` (amino asit) değerleri sırasıyla (P,C) ve (E,L) — 5-satırlık gruptaki tüm satırlar için bu alanlar NaN. Yani bu iki satır **gerçekte farklı varyantlar**; yakın-kopya taraması yalnızca `AL_`+`EK_` sayısal kolonlarını kullandığı ve bu satırların `AL_` bloğu tamamen eksik olup aynı medyana doldurulduğu için sahte biçimde "yakın" görünüyorlar. Bu tarama yöntemi **kategorik/AA_ kolonlarını içermiyor**, dolayısıyla tam-eşleşme analizindeki (A.4 üstü) sonuç otoriter kabul edilmeli; yakın-kopya taraması yalnızca ek bir sinyal, kesin kanıt değil. İkinci en yakın çift (VAR_002598/VAR_002839, mesafe≈0.20) gerçek bir yakın-kopya adayı olabilir ama eşiği net biçimde aşmıyor, ayrı bir işlem gerektirmiyor.

---

## A.5 — Kategori–Etiket İlişkisi

### CAT_1 (gnomAD alt-popülasyonu, 24 seviye)

En büyük sapmalar (`reports/tables/CAT_1_label_crosstab.csv`, `reports/figures/cat1_cat2_label_rates.png`): `gnomADg_MID` (n=1, oran=0.0, sapma=-0.83), `AFR` (n=3, oran=0.33, sapma=-0.50), `gnomADg_FIN`/`gnomADg_ASJ` (n=2 her biri, oran=0.5) — **tüm belirgin sapmalar n≤10 gibi çok küçük hücrelerde**, gürültü olma ihtimali yüksek. En kalabalık seviyeler (`gnomADe_NFE` n=66, `gnomADg_NFE` n=32) genel ortalamaya (0.833) yakın kalıyor. **Kaynak-provenance riski CAT_1 için düşük** — büyük hücrelerde belirgin sapma yok.

`CAT_1` %36.3 eksik (135/372 satır) — CLAUDE.md'de belirtilmemiş yeni bir bulgu, Aşama B'de not edilmeli.

> **Düzeltme notu (Aşama D Düzeltme Turu, Görev 3):** `reports/figures/cat1_cat2_label_rates.png` grafiğinde görünen `"gnomADe_AFR&gnomADe_AMR&gnomADe_ASJ&gnomADe_EAS&gnomADe_FIN&gnomADe_MID&gnomADe_NFE&gnomADe_REMAINING&gnomADe_SAS"` etiketli çubuğun bir görselleştirme/gruplama hatası olup olmadığı kontrol edildi. **Sonuç: ham veride gerçekten var, EDA kodunda hata yok.** `df['CAT_1'].astype(str).str.contains('&').sum()` → **5 satır** (`CAT_2`'de 0). Bu 5 satırın (`VAR_002655, VAR_002560, VAR_003161, VAR_002835, VAR_002589`, hepsi `Label=1`) `CAT_1` değeri, tam olarak 9 tekil `gnomADe_*` seviyesinin ("AFR","AMR","ASJ","EAS","FIN","MID","NFE","REMAINING","SAS") `&` ile birleşimi — yani bu satırlar **gerçekten çok-değerli (multi-label)**: varyant, gnomADe'nin TÜM alt-popülasyonlarında gözlenmiş (kozmopolit/yaygın varyant), tek bir alt-popülasyona atfedilemiyor. Grafik ve tablo bu değeri doğru şekilde ham veriden yansıtıyor; düzeltme gerekmedi, figür yeniden üretilmedi. **Ancak** bu, `CAT_1`'in şu anki tekil-kategori kodlamasının (bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md`'deki "CAT_1 çok-değerlilik" notu) semantik olarak eksik kaldığı anlamına geliyor — düzeltme önerisi orada belgelendi, bu turda kod değiştirilmedi.

### CAT_2 (AllofUs, 7 seviye) — özel inceleme

| CAT_2 | n | Patojenik oranı | Sapma |
|---|---|---|---|
| AllofUs_MID | 3 | 0.667 | -0.167 |
| AllofUs_AFR | 23 | 0.739 | -0.094 |
| AllofUs_AMR | 13 | 0.769 | -0.064 |
| AllofUs_OTH | 19 | 0.842 | +0.009 |
| AllofUs_EAS | 17 | 0.882 | +0.049 |
| AllofUs_EUR | 62 | 0.919 | +0.086 |
| AllofUs_SAS | 7 | 1.000 | +0.167 |

`CAT_2` %61.3 eksik (228/372) — CLAUDE.md'de belirtilmemiş, oldukça yüksek bir eksiklik oranı. Doldurulan satırlarda (144/372) `AllofUs_AFR`/`AllofUs_AMR` belirgin şekilde ortalamanın altında, `AllofUs_EUR`/`AllofUs_SAS` üstünde patojenik oranına sahip. En büyük hücre bile (`AllofUs_EUR`, n=62) 8.6 puanlık bir sapma gösteriyor — bu, gnomAD alt-popülasyon etiketinden (CAT_1) daha belirgin bir kaynak-provenance sinyali. **Bu ilişki gerçek bir popülasyon-genetiği etkisi de olabilir (bazı patojenik varyantlar belirli popülasyonlarda gerçekten daha sık gözlenir) ya da kohort-kompozisyon artefaktı da olabilir** — ayırt edilemiyor, bu nedenle CAT_2 encoding kararı (Aşama B) fold-içi frekans kodlaması ile sızıntısız tutulacak, ham target-encoding kullanılmayacak.

---

## A.6 — Grup Bazlı Univaryant Dağılımlar

### AL_ grubu

334 `AL_` kolonu üç alt-tipe ayrıldı (`reports/tables/AL_type_classification.json`, `reports/tables/AL_column_stats.csv`):

| Tip | Kolon sayısı | Tanım |
|---|---|---|
| Frekans-tipi | 162 | Sağa çarpık, çoğunlukla küçük pozitif değerler |
| Oran-tipi ([0,1], çok sayıda tam 0/1) | 82 | [0,1] aralığında, >%5 tam 0 veya 1 değer |
| **Sabit-değerli (tek gözlenen değer)** | **90** | Gözlendiğinde her zaman aynı tek değeri alıyor |

**Yeni ve önemli bulgu — 90 `AL_` kolonu sabit değerli:** Bu kolonların 89'u yalnızca **1.0** değerini alıyor (gözlendiğinde), geri kalan `AL_185` ise yalnızca **264690.0** değerini alıyor (muhtemelen bir popülasyon "allele number"/kohort büyüklüğü denklemi — varyanttan bağımsız sabit bir payda). Bu 90 kolonun **taşıdığı tek bilgi, değeri değil eksiklik/varlık durumudur** — sürekli bir sayısal sinyal olarak ele alınmaları (log1p, ölçekleme vb.) anlamsız ve gürültü ekler. Aşama B'de bu kolonlar için: (a) değer olarak sabit tutulacak (hiçbir dönüşüm uygulanmayacak, zaten bilgi taşımıyor), (b) varlık/yokluk bilgisi zaten blok-seviye ve kolon-seviye eksiklik göstergeleriyle yakalanıyor, (c) Aşama D özellik seçiminde bu kolonların ham hali muhtemelen elenecek (sıfır varyans → model için faydasız), yalnızca göstergeleri tutulacak.

Kalan 244 kolon (162 frekans-tipi + 82 oran-tipi) gerçek sürekli sinyal taşıyor. Frekans-tipi kolonlar log1p ile, oran-tipi kolonlar epsilon-ofsetli logit ile ele alınacak (Aşama B.4). Örnek dağılımlar: `reports/figures/AL_distributions.png`.

### EK_ grubu

| Kolon | n (dolu) | Ortalama | Min | Maks | [0,1] mi | Eksiklik |
|---|---|---|---|---|---|---|
| EK_1 | 361 | 5.66 | 2.91 | 6.17 | Hayır | %2.96 |
| EK_2 | 361 | 4.67 | -10.80 | 6.17 | Hayır | %2.96 |
| EK_3 | 194 | 2.54 | -7.54 | 5.81 | Hayır | **%47.85** |
| EK_4 | 361 | 0.95 | 0.00 | 1.00 | **Evet** | %2.96 |
| EK_5 | 361 | 0.84 | 0.00 | 1.00 | **Evet** | %2.96 |
| EK_6 | 361 | 0.91 | 0.00 | 1.00 | **Evet** | %2.96 |
| EK_7 | 361 | 6.14 | -1.67 | 10.00 | Hayır | %2.96 |
| EK_8 | 361 | 0.56 | -1.51 | 0.76 | Hayır | %2.96 |
| EK_9 | 361 | 7.60 | -7.84 | 11.93 | Hayır | %2.96 |

Kaynak: `reports/tables/EK_column_stats.csv`, `reports/figures/EK_boxplots.png`. `EK_4/5/6` gerçekten [0,1] normalize; `EK_1,2,7,8,9` sınırsız native ölçekte (bazıları negatif) — **kör ortalama/toplama kesinlikle uygulanamaz**, Aşama B'de kolon-bazlı rank/quantile harmonizasyonu gerekli. `EK_3` diğerlerinden çok daha fazla eksik (%47.85 vs %2.96) — kendi ayrı eksiklik göstergesi + medyan doldurma alacak (CLAUDE.md'deki plana uygun). `EK_1,2,4-9` (EK_3 hariç) tam olarak aynı 11 satırda birlikte eksik (A.3.2'deki blok ile aynı) — bu blok için ayrı bir "EK_blok_eksik" göstergesi düşünülebilir (Aşama B kararı).

### AA_1 / AA_2 (amino asit harfi)

20 standart amino asit harfinin tümü her iki kolonda da gözleniyor, dağılım dengesiz ama patolojik bir anomali yok (`reports/tables/AA_1_letter_freq.csv`, `AA_2_letter_freq.csv`). En sık AA_1: P(35), L(34), R(33); en sık AA_2: V(40), S(36), L(32). Nominal kategorik olarak one-hot ile kodlanacak (20 seviye, düşük kardinalite).

### CAT_3 / CAT_4 / CAT_5

Her üçü de yalnızca 4 seviye alıyor: `G/G`, `T/T`, `A/A`, `C/C` — **hiçbir heterozigot form (`G/T`, `A/C` vb.) gözlenmedi**, kod ile doğrulandı. Bu nedenle bu kolonlar dozaj/additive (0/1/2) kodlama yerine **nominal kategorik (one-hot, 4 seviye)** olarak ele alınacak; additive kodlama biyolojik olarak anlamsız olurdu çünkü heterozigot referans noktası yok.

---

## A.7 — Redundans / Korelasyon Kümeleme (AL_ grubu)

334 `AL_` kolonu arasında Spearman korelasyonu hesaplandı (eksik-değer-toleranslı, **min_periods=30**), hiyerarşik kümeleme ile çoklu eşiklerde bağımsız küme sayısı:

| |r| eşiği | Bağımsız küme sayısı | Çok-kolonlu küme sayısı | En büyük küme |
|---|---|---|---|
| >0.99 | 334 | 0 | 1 |
| >0.95 | 334 | 0 | 1 |
| >0.90 | 334 | 0 | 1 |
| >0.80 | 334 | 0 | 1 |
| >0.70 | 327 | 7 | 2 |
| >0.50 | 295 | 13 | **13** |

Kaynak: `reports/tables/AL_correlation_cluster_counts.csv`, `reports/tables/AL_correlation_clusters.csv` (her kolon için farklı eşiklerdeki küme etiketi).

> **Düzeltme notu (Aşama D Düzeltme Turu — `min_periods` tutarsızlığı, bağımsız EDA notebook'u ile çapraz kontrol):** Bu bölüm önceden `min_periods=10` ile hesaplanmıştı ve en yüksek ikili korelasyonu 0.92 (`AL_184`-`AL_30`) olarak raporlamıştı. Doğrulandı: bu 0.92 değeri yalnızca **13 ortak-dolu satıra** dayanıyordu — küçük-örneklem gürültüsü, gerçek bir ilişki değil (bu panelde `AL_` eksikliği %27-96 arasında olduğundan, `min_periods=10` gibi düşük bir eşik iki kolonun yalnızca birkaç satırda örtüştüğü durumlarda bile korelasyon hesaplamasına izin veriyor, ve az sayıda gözlemle Spearman korelasyonu kolayca ±1'e yakın uç değerlere sıçrayabiliyor). **`min_periods=30`'a geçildi** (istatistiksel olarak daha güvenilir alt sınır) ve tüm korelasyon-türevi dosyalar (`AL_spearman_corr.csv`, `AL_correlation_clusters.csv`, `AL_correlation_cluster_counts.csv`, `AL_correlation_summary.json`) yeniden üretildi. Yeni sonuç, bağımsız EDA notebook'unun (`01_eda_pah_ipynb.ipynb`, aynı eşikle) bulduğu sayılarla **birebir örtüşüyor**: en yüksek ikili korelasyon 0.777 (`AL_121`-`AL_88`), medyan 0.094, ortalama 0.124 — hiçbir çift 0.90'a yaklaşmıyor. Hesaplanabilir hücre oranı (244 değişken `AL_` kolonu üzerinden, 90 sabit-değerli kolon hariç): %85.6.

**Yorum (güncellendi):** `min_periods=30` ile `AL_` grubunda ciddi bir çoklu-doğrusallık **yok** — |r|>0.7 eşiğinde bile yalnızca 7 çok-kolonlu küme var (önceki, güvenilmez min_periods=10 sonucunun iddia ettiği 21 kolonluk kümeler artık desteklenmiyor). Gevşek eşikte (|r|>0.5) 295 bağımsız küme oluşuyor, en büyük küme 13 kolon içeriyor. Aşama C'deki grup-özet özellik üretimi (min/maks/medyan/pozitif-sayısı) için **|r|>0.5 kümeleme temel alınacak** (daha anlamlı, aşırı parçalı olmayan gruplama) — bu kümeleme değiştiği için `v3`/`v4` veri setleri de yeniden üretildi (bkz. `reports/02_ON_ISLEME_KARARLARI_PAH.md`).

Ayrıca bu düzeltme, `reports/03_OZELLIK_SECIMI_PAH.md`'deki elastic-net stability selection'ın gevşek eşiğinin (%41 özellik geçiyor) **çoklu-doğrusallıktan değil, n≈p (369 satır/406 özellik) az-belirlenmiş rejiminden** kaynaklandığı yorumunu da doğruluyor (o rapor ayrıca güncellendi).

---

## A.8 — Near-Constant Feature Taraması (Tamamlayıcı Deney Turu)

> **Ek kontrol tarihi:** Tamamlayıcı Deney Turu, Görev 1. Bir başka panelin (CFTR) ön işleme sürecinden esinlenerek eklendi. Aşama A.6 yalnızca **tam sabit** (`nunique<=1`) `AL_` kolonlarını (90 tanesi) tespit etmişti; burada **near-constant** (dolu değerlerin belirli bir oranı aynı) taraması `AL_`'nin 244 "değişken" kolonuna + `EK_` (9) + `AA_1/2` ve `CAT_3/4/5`'in one-hot açılımına (57 kolon, `v3.parquet`'ten) + `CAT_1/2`'nin frekans-kodlanmış hâline (2 kolon) uygulandı — toplam **312 kolon**.

| Eşik | Near-constant kolon sayısı | Grup dağılımı | 24-özellik havuzuyla örtüşme |
|---|---|---|---|
| ≥%99 | 2 | 1 `AL_` (`AL_325`), 1 one-hot (`AA_1_W`) | **Yok** |
| ≥%95 | 34 | 29 one-hot (`AA_`/`CAT_3-5`), 5 `AL_` | **Yok** |

Kaynak: `reports/tables/near_constant_scan.csv`.

**Bulgu:** Near-constant kolonların ezici çoğunluğu (%95 eşiğinde 34'ün 29'u) **one-hot açılımından geliyor** — beklenen bir yapısal sonuç: nadir bir kategorinin (örn. `AA_1_W`, yalnızca 3/372 satırda görülen Triptofan) one-hot açılımı otomatik olarak "%99 sıfır" bir kolon üretir. Bu, yeni bir veri kalitesi sorunu değil, one-hot kodlamanın doğal bir yan etkisi.

**En önemli sonuç:** Her iki eşikte de near-constant çıkan **hiçbir kolon Aşama D.1'in final 24-özellik havuzunda yer almıyor** — nested özellik seçimi süreci (mutual information + elastic-net + GBDT + permutation importance, 50 dış fold) bu düşük-varyanslı kolonları zaten kendiliğinden eleye­rek doğru kararı veriyor. Bu, **ayrı bir near-constant ön-filtreleme adımına ihtiyaç olmadığının** ampirik kanıtı (bkz. Görev 9 kutusu, `reports/04_PREPROCESSING_MERDIVENI_PAH.md`).

## A.9 — Outlier Analizi (IQR + MAD) (Tamamlayıcı Deney Turu)

> **Ek kontrol tarihi:** Tamamlayıcı Deney Turu, Görev 3. Şimdiye kadar yalnızca çarpıklık (log1p/logit kararı için) incelenmişti; burada resmî bir aykırı-değer taraması yapıldı. Veri: `v1.parquet` (369 satır, sabit-olmayan 253 sayısal kolon: 244 `AL_` + 9 `EK_`).

**Yöntem:** Her kolon için (a) **IQR kuralı** (`[Q1-1.5×IQR, Q3+1.5×IQR]` dışı) ve (b) **modified z-score** (`0.6745×(x-medyan)/MAD`, `|z|>3.5`) ile aykırı-değer adayları işaretlendi. Her satır için "kaç kolonda aykırı değer taşıyor" oranı (**outlier yükü**) hesaplandı.

| Ölçüm | IQR | MAD (modified z-score) |
|---|---|---|
| Kolon başına ortalama aykırı-satır oranı | %8,9 | %17,4 |
| Satır-bazlı outlier yükü (medyan) | %5,3 | %12,7 |

MAD yöntemi, bu panelin ağır çarpık dağılımlarına daha duyarlı olduğu için (beklenen), IQR'den belirgin biçimde daha fazla aday işaretliyor.

**Sınıflar arası karşılaştırma** (Mann-Whitney U testi, `reports/tables/outlier_class_comparison.json`):

| Yöntem | Benign medyan (n=60) | Patojenik medyan (n=305) | p-değeri |
|---|---|---|---|
| IQR | **0,0602** | 0,0486 | **0,0086** (anlamlı) |
| MAD | 0,1377 | 0,1250 | 0,1456 (anlamlı değil) |

Görsel: `reports/figures/outlier_load_by_class.png`.

**Bulgu ve yorum:** IQR yöntemine göre **benign satırlar patojenik satırlardan istatistiksel olarak anlamlı derecede daha yüksek outlier yükü taşıyor** (MAD yönteminde aynı yön ama anlamlı değil — yöntem-bağımlı bir bulgu, ihtiyatla okunmalı). Bunun iki olası açıklaması var: (1) gerçek bir biyolojik/istatistiksel örüntü (benign varyantlar bu öznitelik uzayında daha "sıra dışı" konumlanıyor olabilir), (2) küçük örneklem (n=62 benign) artefaktı — az sayıda örnekte tek tük uç değer, oranı kolayca şişirebilir. **Bu iki hipotez arasında bu analizle ayrım yapılamıyor.**

**Kesin karar: hiçbir satır silinmedi.** Gerekçe: (a) küçük örneklemde (özellikle 62 benign) satır silmek orantısız bilgi kaybı riski taşır, (b) genomik popülasyon frekansı verisinde "uç değer" çoğu zaman gerçek ve bilgilendirici bir aşırı-nadirlik sinyalidir, hata değil — tıpkı `al_all_missing`'in nadirlik sinyali olarak yorumlanması gibi (bkz. Aşama A.3.4). **Bu bulgu modelleme aşamasına (Aşama E) açık bir risk notu olarak taşınıyor:** eğer bir model benign sınıfını sistematik olarak yanlış sınıflandırıyorsa, bu outlier-yükü farkının bir nedeni olup olmadığı ayrıca incelenmeli.

---

## Aşama B için Çıkarımlar (Özet)

1. **Eksiklik göstergeleri — kör değil, seçici:** 334 `AL_` kolonuna tek tek gösterge eklenmeyecek. Bunun yerine: (a) blok-seviye "AL_ tümü eksik" göstergesi (92 satır, tek gösterge), (b) blok-seviye "EK_(EK_3 hariç)+CAT_3/4/5 tümü eksik" göstergesi (11 satır, tek gösterge), (c) `EK_3`'ün kendi göstergesi (%47.85 eksik, ayrı mekanizma). A.3.2'deki 14 benzersiz `AL_` alt-deseninden en bilgilendirici olanlar (p<0.05 VE tekil/temsilci desen) ayrıca değerlendirilecek, ama kör 41-kolon eklenmeyecek.
2. **90 sabit-değerli `AL_` kolonu** ham değer olarak dönüştürülmeyecek (log1p/logit anlamsız) — bilgi içerikleri zaten eksiklik göstergelerinde. Aşama D özellik seçimi bu kolonları muhtemelen eleyecek.
3. **AL_ dönüşümü:** 162 frekans-tipi kolon → log1p; 82 oran-tipi [0,1] kolon → epsilon-ofsetli logit; 90 sabit kolon → dönüşümsüz bırak.
4. **EK_ harmonizasyonu:** `EK_4/5/6` zaten [0,1], dönüşümsüz bırakılabilir; `EK_1,2,7,8,9` (sınırsız) kolon-bazlı rank/quantile transform ile ortak ölçeğe getirilecek; `EK_3` ayrı gösterge+medyan doldurma.
5. **Kategorik kodlama:** `CAT_3/4/5` ve `AA_1/2` → one-hot (nominal, düşük kardinalite, heterozigot yok). `CAT_1` (24 seviye, düşük provenance riski) ve `CAT_2` (7 seviye, belirgin ama açıklaması net olmayan provenance sinyali) → fold-içi frekans kodlaması, asla ham target encoding değil. `CAT_6` tamamen boş → düşür.
6. **Dedup:** `pandas.duplicated(subset=öznitelikler+Label, keep="first")` ile 3 satır düşürülecek (369 satır kalacak); geriye kalan 2 satırlık (VAR_003238, VAR_003234) giderilemez çelişki, ortak `group_id` alacak.
7. **Şüpheli döngüsellik kolonu yok** (tüm tek-özellik AUC'ler ≤0.715) — Aşama D'de yine de izole-grup modelleriyle çapraz doğrulanacak, özellikle `EK_7`'nin göreli yüksek AUC'si izlenecek.
8. **Grup-özet özellikler (Aşama C v3):** |r|>0.5 korelasyon kümelerine göre (270 bağımsız küme, en büyük 21 kolon) `AL_` grubu içi min/maks/medyan/pozitif-sayısı özetleri üretilecek.
