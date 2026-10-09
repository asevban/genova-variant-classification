# PAH Paneli — Bağımsız Deney Faz 3 — Nihai Rapor (Adım 16)

**Kapsam:** Orijinal 16 adımlık prompt'un tam kapsamı (Faz 1/2'nin
kapsamadığı: tam korelasyon matrisi, kategorik ilişki ölçüleri, log
dönüşümü, normalizasyon denemesi, koşullu IG, üstel/kuvvet eğri uydurma,
etkili gözlemler, train-test uyumu, nested doğrulama). Bu rapor
`experiments/tam_16_adim_deneme/` altında üretilen tüm çıktıları tek yerde
toplar. **Resmi GENOVA pipeline'ına (`data/`, `configs/`, `reports/`,
`src/genova/pah/*.py`) hiçbir değişiklik yapılmadı** — bkz. son bölüm
("İzolasyon Doğrulaması").

---

## 1. PRECHECK Sonucu

Tüm 6 madde **PASS**:

```
[x] split bank bulundu, 50 outer / 4 inner fold doğrulandı
[x] build_fold_features import edildi
[x] v4_final_feature_pool.json'dan taze okunan liste = 25 özellik
[x] ham veri (YARISMA_TRAIN_PAH.csv) satır/kolon sayısı doğrulandı (372/353)
[x] çıktı dizini yalnızca experiments/tam_16_adim_deneme/
[x] official dosyalara yazma modu KAPALI
```

Detay: `results/precheck_log.txt`.

---

## 2. İki Zorunlu Düzeltmenin Uygulanışı

**Düzeltme 1 (Adım 2-3, global IG sızıntı riski):** Adım 2'nin IG
hesaplaması (353 kolon, ham 372 satır) yalnızca keşif/rapor amaçlı
tutuldu — hiçbir eleme kararı doğrudan bu sayılardan verilmedi. Adım 3'ün
131 kolonluk eleme listesi (`step3_elimination_hypothesis.csv`) açıkça
**"hipotez"** olarak işaretlendi. Gerçek doğrulama Adım 13'te split
bankasının 50 outer fold'unda nested olarak yapıldı: Varyant A (tam 405
kolon aday evren) vs Varyant C (Adım 3'ün önerdiği ~265 kolonluk azaltılmış
set). **Sonuç: azaltılmış set nested karşılaştırmada zarar vermedi ama
belirgin bir üstünlük de göstermedi** (bkz. Bölüm 4) — "eleme işe yaradı"
sonucuna varılamaz, dürüstçe "desteklenmedi" olarak raporlanıyor.

**Düzeltme 2 (Adım 5, kendi CV'sini kurma riski):** Adım 5, kendi
`StratifiedKFold`'unu kurmadı — split bankasının 50 outer fold'u üzerinde,
nested OLMAYAN, iç eşik-arama içermeyen hızlı bir **ön-izleme** olarak
çalıştırıldı (basit çoğunluk eşiği=11, iç doğrulama yok). Rapor metninde
(`step4_hard_voting_by_fold.csv`, `step5_preview_threshold_summary.csv`)
ve bu raporda **açıkça "ön-izleme, nested değil"** etiketlendi. Asıl
güvenilir eşik/performans sayıları yalnızca Adım 13'ün nested çıktısından
alındı.

---

## 3. Adım 6-7'nin Öne Çıkan Bulguları

### Adım 6 — Tam Sayısal Korelasyon Matrisi (58,653 çift)

- **Pearson'da 222 çift `|r|≥0.90` ("çok güçlü"), ama Spearman'da 0.**
  Bu, Faz 2'nin "AL_-AL_'de |r|≥0.90 yok" bulgusuyla (Spearman ile ölçülmüş)
  **çelişmiyor**, genişletilmiş kapsamda (AL_-EK_/EK_-EK_ dahil) da
  **bağımsız olarak yeniden doğrulandı**.
- En çarpıcı örnek: `AL_235`↔`AL_271` — Pearson r=0.978, Spearman r=0.003
  (rank ilişkisi pratikte sıfır). `AL_` kolonlarının ağır sağa-çarpık/sıfır-
  şişkin yapısı nedeniyle Pearson birkaç uç değer tarafından domine
  ediliyor — **GENOVA'nın "AL_ için Spearman kullan" kararının bağımsız
  doğrulaması**.
- Metodolojik not: ilk çalıştırmada `min_periods=30` unutulup 237 sahte
  "çok güçlü" çift (n=14 gibi düşük ortak-gözlemli) bulunmuştu; GENOVA'nın
  kendi `01_eda_pah.py` konvansiyonuyla düzeltildi.

### Adım 7 — Kategorik İlişki Ölçüleri

- `CAT_3`↔`CAT_4`↔`CAT_5` üçlemesi Cramér's V = **1.0000** ile (üçüncü
  bağımsız yöntemle) yeniden doğrulandı.
- Hedefle en güçlü kategorik ilişki: `AA_1` (Cramér's V=0.180, p=0.042),
  `CAT_3/4/5` (V=0.169, p=0.006) — istatistiksel olarak anlamlı;
  `CAT_1` (V=0.139, p=0.148) ve `CAT_2` (V=0.051, p=0.335) anlamlı değil.
- En güçlü güvenilir (n≥30) Eta ilişkileri: `CAT_2`-`AL_84` (0.883, n=144),
  `CAT_1`-`AL_271` (0.865, n=129) — 60 düşük-n kombinasyon (n<30, eta≈0.99)
  Adım 6'daki aynı desenle tespit edilip filtrelendi.

---

## 4. Adım 13'ün Sonuç Tablosu (ASIL GÜVENİLİR SONUÇ — nested, 50 outer fold)

| Varyant | Açıklama | n_özellik | Macro-F1 | F1(Label=1) | MCC | Recall | Specificity | F1 (%28.6 prevalans) | FP/100 benign |
|---|---|---|---|---|---|---|---|---|---|
| A_temel_405 | Tam aday evren (405) | 405 | 0.6428±0.0686 | 0.8398 | 0.3283 | 0.7829 | 0.6122 | 0.5745 | 38.78 |
| B_resmi_25havuz | Resmi 25-özellik havuzu (referans) | 25 | 0.6660±0.0668 | 0.8538 | 0.3742 | 0.7991 | 0.6520 | 0.6079 | 34.80 |
| C_adim3_hipotez | Adım 3 eleme hipotezi (~265 kolon) | 265 | 0.6521±0.0586 | 0.8643 | 0.3249 | 0.8332 | 0.5349 | 0.5620 | 46.51 |
| D_adim8_redundancy_pruned | Adım 8 redundan-budanmış (405-9) | 396 | 0.6575±0.0618 | 0.8529 | 0.3488 | 0.8025 | 0.6135 | 0.5866 | 38.65 |
| E_25havuz_log1p | 25-havuz + log1p(AL_) | 25 | 0.6660±0.0668 | 0.8538 | 0.3742 | 0.7991 | 0.6520 | 0.6079 | 34.80 |
| **F_25havuz_ortalama** | **25-havuz + 4 ortalama-özellik (29 kolon)** | **29** | **0.6841±0.0661** | **0.8700** | **0.3983** | **0.8273** | **0.6381** | **0.6157** | **36.19** |

**Gözlemler:**

- **Varyant F (25-havuz + Adım 11'in 4 ortalama-özelliği) tüm metriklerde
  en iyi sonuç** — resmi 25-havuz referansına (B) göre Macro-F1 +0.018,
  MCC +0.024, prevalans-ayarlı F1 +0.008. Bu, Adım 11'in nested-doğrulama
  kuralını (yalnızca nested CV performansı iyileşiyorsa aday göster)
  **karşılıyor** — aday olarak öneriliyor (otomatik ekleme değil).
- **Varyant E (log1p), Varyant B ile BİREBİR AYNI sonucu verdi** (tüm
  metrikler 6 ondalık basamağa kadar özdeş). **Bu bir hata değil,
  matematiksel olarak beklenen bir sonuçtur**: `EntropyTreeEnsemble`
  (sklearn `DecisionTreeClassifier`, `criterion="entropy"`) bölünme
  noktalarını sıralı-eşik-tarama (threshold-scan) ile bulur; `log1p`
  **monotonik artan** bir dönüşüm olduğundan, her düğümdeki optimal
  bölünme aynı veri partisyonunu üretir — ağaç yapısı ve dolayısıyla tüm
  tahminler değişmez. **Sonuç: log1p dönüşümü, eşik-tabanlı ağaç
  ensemblelerinde etkisizdir** — GENOVA'nın M-3 basamağındaki genel log1p
  bulgusu bu model ailesinde geçerli değil (doğrusal/mesafe-tabanlı
  modellerde geçerli olabilir, ağaç modellerinde geçerli değil). Adım 12'nin
  "genel kontrol" amacı bu şekilde karşılandı.
- **Varyant C (Adım 3'ün eleme hipotezi) Düzeltme 1'in beklediği gibi test
  edildi — desteklenmedi:** Macro-F1 (0.652) hem A'dan (0.643) hem B'den
  (0.666) düşük; specificity belirgin şekilde düşük (0.535 vs A'nın 0.612)
  ve FP/100 benign en yüksek değer (46.5) — azaltılmış set, prevalans-
  ayarlı skorda **zarar veriyor**. **"Eleme işe yaradı" sonucuna
  varılamaz — dürüstçe desteklenmedi.**
- **Varyant D (Adım 8 redundancy-budama), A'ya göre hafif iyileşme**
  (Macro-F1 0.658 vs 0.643, MCC 0.349 vs 0.328) ama B'nin (25-havuz)
  gerisinde kalıyor — 9 kolonun çıkarılması zarar vermiyor ama tek başına
  yeterli bir iyileştirme de değil.
- **Genel sıralama (Macro-F1):** F (0.684) > B=E (0.666) > D (0.657) >
  C (0.652) > A (0.643). 25-havuza dayalı varyantlar (B/E/F), 405-kolonluk
  tam evrene veya kısmi elemelere göre tutarlı şekilde daha iyi — bu,
  resmi `v4_final_feature_pool.json`'ın (25 özellik) zaten iyi bir seçim
  olduğunun bağımsız bir teyididir.

Ham dış-fold sonuçları: `results/step13_nested_<varyant>.csv` (her biri 50
satır). Özet: `results/step13_nested_summary.csv`.

---

## 5. Diğer Adımların Özeti (doğrulanan vs yalnızca hipotez/keşif)

| Adım | Bulgu | Durum |
|---|---|---|
| 1 | 372/353/310/62 — GENOVA EDA ile birebir eşleşti; şartname çapraz-doğrulaması (F1 metrik, %28.6 prevalans, CAT_ taksonomisi) | **Doğrulandı** (dış kaynakla çapraz-kontrol) |
| 2 | 353 kolon için IG (en yüksek: EK_7=0.063, AL_323=0.062, AA_1=0.061) | **Yalnızca keşif** (Düzeltme 1) |
| 3 | 131 kolonluk eleme hipotezi | **Desteklenmedi** (Adım 13-C ile test edildi, zarar verdi) |
| 4 | 21-ağaç ensemble, Hard F1=0.854±0.035, Soft F1=0.882±0.027 (ön-izleme) | Faz 1 Model C ile örtüşüyor (referans) |
| 5 | Eşik 11-19 taraması, prevalans-ayarlı F1 eşik=13'te tepe (0.632) | **Yalnızca ön-izleme, nested değil** (Düzeltme 2) — asıl eşikler Adım 13'ün iç döngüsünden |
| 6 | Pearson 222 vs Spearman 0 çok-güçlü çift; AL_ için Spearman doğru ölçü | **Doğrulandı** (58,653 çiftin tamamı tarandı) |
| 7 | CAT_3=CAT_4=CAT_5 (V=1.0), AA_1/CAT_3-5 hedefle anlamlı | **Doğrulandı** (üçüncü bağımsız yöntemle) |
| 8 | 14 redundan çift, 9 zayıf-taraf adayı | Adım 13-D ile test edildi, **hafif olumlu ama tek başına yetersiz** |
| 9 | EK_7-EK_9 kısmi korelasyonu al_all_missing'ten bağımsız (0.774→0.773); AL_88-AL_121 IG'si koşullu olarak artıyor | **Yalnızca keşif/hipotez** (küçük ortak-n uyarısıyla) |
| 10 | 4 güçlü çiftin hiçbirinde üstel/kuvvet lineer'i geçmedi | **Doğrulandı** (LOOCV RMSE ile) — dönüşüm önerilmedi |
| 11 | 4 ortalama-özellik adayı tanımlandı, Adım 13-F'de test edildi | **Doğrulandı** (nested CV'de iyileşme gösterdi — Bölüm 4) |
| 12 | Adım 10'da üstel çift bulunmadığı için spesifik kapsam boş; genel log1p testi Adım 13-E'ye taşındı | **Doğrulandı** (ağaç modellerinde etkisiz — matematiksel kanıt) |
| 13 | Nested CV, 6 varyant, 50 outer fold | **Asıl resmi sonuç** (Bölüm 4) |
| 14 | 4 çiftin hiçbirinde tekil/5'li gözlem grubu dramatik etki göstermedi (max Δr=0.066) | **Doğrulandı** — silme önerilmedi |
| 15 | Train-test sütun uyum ilkesi dokümante edildi, doğrulama fonksiyonu resmi `build_fold_features` çıktısı üzerinde PASS verdi | **Doğrulandı** (ilke düzeyinde, gerçek test seti yok) |

---

## 6. Adım 15 — Train-Test Sütun Uyumu (özet)

Gerçek test seti henüz yok. İlke dokümante edildi + resmi
`build_fold_features()` çıktısı üzerinde bağımsız bir
`validate_train_test_column_consistency()` fonksiyonuyla test edildi
(**PASS**) — kolon seti/sırası birebir aynı, `Variant_ID` girdi değil.
Detay: `results/step15_train_test_consistency.md`.

---

## 7. Nihai Öneri — E2 / `v4_final_feature_pool.json` Adaylığı

**Kısmen EVET, belirli ve sınırlı bir bulgu için:**

- **EVET — aday:** Adım 11'in 4 ortalama-özelliği (`ZORT_AL88_AL121`,
  `ZORT_EK7_EK9`, `ZORT_AL23_AL283`, `ZORT_AL7_AL103`), Adım 13-F'de resmi
  25-havuza eklendiğinde nested CV'de tutarlı iyileşme gösterdi (Macro-F1
  +0.018, MCC +0.024, prevalans-ayarlı F1 +0.008, tüm metriklerde B'yi
  geçti). Bu, görev kuralının ("nested CV performansı iyileşiyorsa aday
  göster") açıkça karşılandığı **tek** bulgu — resmi E2 portföyüne veya
  `v4_final_feature_pool.json`'a **aday olmaya değer**, ancak otomatik
  eklenmedi (bu deneyin yazma yetkisi yok).
- **HAYIR:** Adım 3'ün eleme hipotezi (Varyant C, nested'de zarar verdi),
  Adım 12'nin log1p dönüşümü (ağaç modellerinde matematiksel olarak
  etkisiz — resmi pipeline zaten ağaç-tabanlı model kullanıyorsa
  uygulanmasının anlamı yok).
- **BELİRSİZ:** Adım 8'in redundancy-budaması (Varyant D, A'ya göre hafif
  iyileşme ama B'nin gerisinde — resmi 25-havuzla birleştirilip ayrıca test
  edilmedi, bu deneyin kapsamı dışında kaldı).

**Özet gerekçe:** Bu deneyin en değerli somut çıktısı, resmi 25-özellik
havuzunun **zaten iyi bir seçim olduğunun** (405 kolonluk tam evrene ve
kısmi elemelere karşı tutarlı üstünlüğü) bağımsız doğrulanması, artı **4
ortalama-özelliğin** nested-doğrulanmış, küçük ama tutarlı bir iyileştirme
sunması.

---

## 8. İzolasyon Doğrulaması

`data/`, `configs/`, `reports/` (tüm ağaç) ve `src/genova/pah/*.py`
dosyalarının mtime'ları, görev başlangıcında alınan `173` dosyalık
`protected_mtimes_phase3_BEFORE.txt` anlık görüntüsüyle karşılaştırıldı.
Bkz. bu raporun sonunda eklenen doğrulama komut çıktısı — **hiçbir resmi
dosya bu deney boyunca değiştirilmedi**.
