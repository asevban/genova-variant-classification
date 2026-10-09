# PAH Paneli — Aşama E (E0-E6) Sonrası Kapsamlı Denetim

> ⚠️ **GÜNCEL DEĞİL (P2 madde 22 notu):** Bu denetim, E*-EK/P0/P1/P2
> geliştirmelerinden önce üretildi — güncel değil, tarihsel kayıt olarak
> korunuyor. Burada "bulgu" olarak işaretlenen birçok madde (özellikle
> özellik-seçimi sızıntısı, objective mismatch, kalibrasyon yöntem-seçimi,
> ağırlıklandırma adaleti) sonraki P0-P2 turlarında ayrıca ele alındı ve
> çözüldü/belgelendi — güncel durum için `06_MODEL_SECIM_RAPORU_PAH.md`
> ve `09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`'ye bakın.

> **Denetim disiplini:** `06_ASAMA_E_ONCESI_DENETIM_PAH.md` ile aynı ilke —
> bu doküman kod değiştirmez, yalnızca okur/doğrular/raporlar. Her bulgu
> Kritik/Orta/Düşük etiketli, kanıtlı (dosya+satır/sayı). Sorun
> bulunamayan alt-maddeler de açıkça "kontrol edildi, sorun yok" diye
> işaretlendi — hiçbiri sessizce atlanmadı.

## 1. Yönetici Özeti

**Toplam bulgu: 0 Kritik, 8 Orta, 4 Düşük.**

**Aşama F'ye geçmeye hazır mı: EVET, şartlı.** E0-E6'nın kendisi —
metodoloji, sızıntı disiplini, nested prosedürler, testler — sağlam;
hiçbir Kritik bulgu yok, hiçbir hesaplama/sonuç hatası bulunamadı. Bulunan
sorunların tamamı **dokümantasyon güncelliği** (rapor başlığı/özeti E6'ya
kadar güncellenmemiş, proje haritası E0-E6'nın 14 yeni dosyasını hiç
görmüyor) ve **tekrar-üretilebilirlik cilası** (versiyon pinleme yok, tek
komutluk bir "baştan sona çalıştır" listesi yok) kategorisinde — hiçbiri
E'nin bilimsel sonuçlarını geçersiz kılmıyor, hiçbiri Aşama F'nin
başlamasını teknik olarak engellemiyor. Önerilen: Aşama F'ye paralel
olarak (ya da hemen öncesinde, ayrı ve kısa bir görev olarak) bu 8 Orta
bulgunun düzeltilmesi.

**En kritik 3 bulgu (Orta, ama en görünür/en yanıltıcı olanlar):**
1. **`06_MODEL_SECIM_RAPORU_PAH.md`'nin başlığı ve "Kapsam" özeti hâlâ
   yalnızca E2'yi anlatıyor** — dosya artık E2-E6'yı kapsıyor ama başlık
   ve en üstteki özet bunu yansıtmıyor (Bölüm 2).
2. **Aynı dosyada, E2'nin "Öneri" cümlesi ("CatBoost/v1'i birincil aday
   olarak taşı") E6'nın nihai kararıyla (CatBoost/v4_from_v2, solo)
   doğrudan çelişiyor** ve hiçbir yerde geriye dönük düzeltilmemiş
   (Bölüm 2, madde 1).
3. **`artifacts/preprocessors/pah_pipeline_tree.joblib`, sklearn'ün
   `check_is_fitted()` kontrolünden geçemiyor** (kök neden bulundu ve
   doğrulandı: pipeline'ın son adımı — `ConstantFillImputer` —
   sklearn'ün "fit edilmiş" kuralına uyan `_`-sonekli bir öznitelik hiç
   set etmiyor). `.transform()` fiilen çalışıyor (test edildi), yani
   fonksiyonel bir kırılma değil, ama E-F dokümanının bu dosya için
   verdiği "sorunsuz" notuyla çelişen, gelecekte bir savunma-amaçlı
   `check_is_fitted()` çağrısını yanlışlıkla başarısız kılacak bir
   kusur (Bölüm 4, madde 6).

---

## 2. Bölüm 1 — Zincir Bütünlüğü (PRECHECK)

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| `data/splits/pah/` tüm dosyalar E0-öncesi | ✅ Geçti | 61 dosya, en yeni mtime `manifest.json` = 1785932835 (2026-08-06 civarı); en erken E-aşaması dosyası (`src/genova/pah/e1_baseline.py`) mtime = 1786826265 (2026-08-17 14:57) — split bankası ~9,5 gün önce donmuş, E0-E6 boyunca dokunulmamış. |
| `v4_final_feature_pool.json` E2'de onaylanan 25 özellikle birebir aynı | ✅ Geçti | `n_features=25`; mtime 2026-08-15 19:43:41 — E0-E6 boyunca hiç değişmedi (aşağıdaki mtime kontrolüyle de doğrulandı). Liste (E2'nin fold_versions.py'de kullandığı havuzla birebir aynı): EK_7, AL_300, AL_171, AL_301, CAT_1, AL_329, AL_49, EK_6, AL_26, EK_2, EK_1, AL_12, AL_298, AL_306, AL_8, AL_317, AL_318, AL_330, AL_277, AL_334, AL_22, al_all_missing, EK_9, EK_8, EK_5. |
| `data/processed/pah/*.parquet` E0'dan beri değişmemiş | ✅ Geçti | 6 dosyanın hepsi mtime 2026-08-15 19:52:55 — hepsi aynı saniyede yazılmış (tek seferlik `dataset_versions.py` koşumu), E-aşamasından ~46 saat önce. |
| `artifacts/preprocessors/*.joblib` E0-E6 arasında dokunulmamış | ✅ Geçti (mtime) / ⚠️ ama bkz. Bölüm 4 madde 6 | mtime 2026-08-15 19:13:35, iki dosya da — dokunulmadığı doğrulandı. Ama **içeriklerinde önceden var olan** (E0-E6'nın sebep olmadığı) bir fit-durumu kusuru bulundu, ayrıntı Bölüm 4/m.6. |
| `experiments/` E0-E6 arasında dokunulmamış | ✅ Geçti | En yeni dosya `experiments/tam_16_adim_deneme/scripts/step13_nested_cv.py`, mtime 2026-08-15 ~23:37 — E-aşamasından önce. |
| `pytest tests/ -q` güncel toplam | ✅ **81 test, 81 geçti** | Dosya dosya: `test_calibration_pah.py`=6, `test_e2_candidate_features_pah.py`=5, `test_e3_calibration_run_pah.py`=1, `test_e4_prior_correction_pah.py`=4, `test_e5_threshold_selection_pah.py`=4, `test_e6_ensemble_pah.py`=5, `test_fold_versions_pah.py`=8, `test_id3_feature_selection_pah.py`=5, `test_metrics.py`=11, `test_preprocessing_pah.py`=29, `test_split_bank_pah.py`=3. Toplam=81. **Not:** `e1_baseline.py`, `e1_threshold_check.py`, `e2_oof_correlation.py`, `e2b_run.py`, `e2b_raw_nan_run.py`, `models.py`, `e3_calibration_run.py`'nin orkestrasyon kısmı için ayrı test dosyası yok — bu, her aşamanın raporunda **açıkça belgelenmiş** bir tasarım kararı (yalnızca zaten test edilmiş yapı taşlarını yeniden kullanan "lehçe" kodu, ayrıca test edilmiyor), sessiz bir boşluk değil. |
| `models/pah/baseline_frozen.pkl` E1 sayılarıyla tutarlı | ✅ Geçti | `summary_v1['f1']` = mean 0.91986/std 0.024186 → 0,9199±0,0242; `summary_v1['specificity']` = mean 0.26083/std 0.10575 → 0,2608±0,1057 — raporla birebir eşleşiyor. |
| `reports/tables/e*.csv` satır sayıları | ✅ Hepsi beklenen | `e2_model_comparison.csv`=(800,13), `e2_oof_predictions_repeat0.csv`=(369,4), `e3_calibration_metrics.csv`=(600,11), `e3_calibrated_oof_predictions.csv`=(44280,8), `e4_prior_corrected_probabilities.csv`=(11070,8) [3 aday×10 tekrar×369=11070, beklenenle uyumlu], `e5_threshold_selection.csv`=(150,21), `e6_ensemble_metrics.csv`=(150,12), `e6_oof_correlation_matrix.csv`=(3,4, index dahil → 3×3 matris). |

**Bölüm 1 sonucu: FAIL yok. Zincir bütünlüğü tam.**

---

## 3. Bölüm 2 — Rapor İçi Tutarlılık (`06_MODEL_SECIM_RAPORU_PAH.md`, 696 satır, tam okuma yapıldı)

### Madde 1 — E2'nin "Öneri" bölümü artık geçersiz bir sıralama taşıyor (**Orta**)

**Tam alıntı (satır 270-273):**
> "**Öneri:** E3'e (kalibrasyon) **CatBoost/v1**'i birincil aday olarak
> taşı. `v4_from_v2` (25-özellik, CatBoost) ikinci aday olarak — F1'de
> marjinal geride ama yorumlanabilirlik/paketleme (Aşama D'nin özellik
> seçimi hattıyla tutarlılık, F4/F6'da avantaj) için değerli bir takas."

**Çelişki (satır 666-672, E6 nihai sonucu):**
> "### Nihai öneri: **Solo CatBoost/v4_from_v2, ensemble değil**
> ... Aşama F'ye ... **CatBoost/v4_from_v2** ... tek final aday olarak
> taşınması öneriliyor."

Satır 270'teki "CatBoost/v1 birincil" ifadesi, E3-E6'nın ürettiği kanıtla
(E5: CatBoost/v4_from_v2 ağırlıklı-F1'de en iyisi; E6: solo model
ensemble'ı yeniyor) **hiçbir yerde geriye dönük düzeltilmedi** — dosyayı
yalnızca 253-278. satırlar arası (E2'nin özeti) okuyan biri yanlış nihai
kararla ayrılır. **Önerilen düzeltme:** satır 270-278'e "(NOT: bu öneri
E3-E6'nın nested kanıtıyla güncellendi, bkz. satır 666 — nihai aday
CatBoost/v4_from_v2, solo)" gibi bir geriye-atıf notu eklenmeli, ya da
bu bölüm E6 sonucuna işaret eden bir üst-not alarak "tarihsel/aşama-içi
karar" olarak çerçevelenmeli.

### Madde 2 — Kalibratör tutarlılığı: sorun bulunamadı

E3'ün önerdiği kalibratör (Beta, satır 349-355) E4 (satır 414-415: "E3'ün
Beta-kalibre dış-test olasılıklarına"), E5 (satır 444-445: "Beta
kalibratörüyle") ve E6'da (satır 612: "Beta ile kalibre edildi") **tutarlı
biçimde** kullanılmış. **Kontrol edildi, sorun bulunamadı.**

### Madde 3 — SLD sabitleri: sorun bulunamadı

`w1 = 0,286/0,833 = 0,343337`, `w0 = 0,714/0,167 = 4,275449` yalnızca bir
kez (satır 419-420) yazılmış; dosyanın başka hiçbir yerinde farklı
yuvarlanmış bir versiyonu yok (`grep` ile doğrulandı — `0,343`/`4,275`
yalnızca bu iki satırda geçiyor). **Kontrol edildi, sorun bulunamadı.**

### Madde 4 — E5'in eşikleri E6'da tutarlı kullanılıyor mu: sorun bulunamadı

E5 tablosu (satır 462-466): CatBoost/v1=0,2847; CatBoost/v4_from_v2=0,3315;
LightGBM/v1=0,2737. E6, bu **solo** eşikleri doğrudan tekrar yazmıyor
(doğru — E6'nın kendi blend-özel eşikleri var, satır 623-625: örn.
ağırlıklı blend için 0,3242, farklı ve doğru şekilde farklı bir kavram).
E6'nın gönderme yaptığı tek sayı, CatBoost/v4_from_v2'nin ağırlıklı-F1'i
(0,6187) — bu da satır 487/524'teki E5 değeriyle birebir eşleşiyor (satır
602, 640). **Kontrol edildi, sorun bulunamadı.**

### Madde 5 — Nihai öneri dosyanın başındaki "Kapsam"a yansımamış (**Orta**)

Dosyanın **başlığı** (satır 1): `# PAH Paneli — Aşama E2: Model Portföyü
ve Nested Karşılaştırma` — hâlâ yalnızca E2'yi adlandırıyor. "Kapsam"
bölümü (satır 3-22) yalnızca 2a/2b/2c alt-aşamalarını anlatıyor, E3/E4/
E5/E6'dan **hiç bahsetmiyor** — oysa dosyanın geri kalan ~490 satırı
(satır 280-696) tam olarak bu dört aşamayı belgeliyor. Bir okuyucu yalnızca
başlığa/kapsam özetine bakarsa dosyanın E2'de bittiğini sanır.
**Önerilen düzeltme:** başlık `# PAH Paneli — Aşama E2-E6: Model
Portföyü, Kalibrasyon, Eşik Seçimi ve Ensemble Değerlendirmesi` gibi
güncellenmeli; "Kapsam" bölümüne E3-E6'nın birer satırlık özeti eklenmeli
(nihai karar dahil: CatBoost/v4_from_v2, solo, Beta+SLD+nested eşik).

### Madde 6 — Dosya genelinde sayı tutarlılığı: sorun bulunamadı (bir potansiyel karışıklık notu hariç)

`0,9199` üç farklı ama gerçek bağlamda kullanılıyor: (a) E1 baseline'ın
F1'i (±0,0242 std ile, satır 188/192/522), (b) E2'de LightGBM/v1'in
KENDİ F1'i (±0,0257 std ile, satır 80/267/315) — ikisi **farklı
ölçümler** olup yalnızca ilk 4 ondalıkları rastlantısal örtüşüyor; std
değerleri (0,0242 vs 0,0257) her yerde doğru ayırt edici, bu yüzden gerçek
bir hata değil, ama hızlı okuyan biri için potansiyel bir karışıklık
kaynağı — **Düşük önemde bir okunabilirlik notu** olarak kaydedildi,
düzeltme gerektirmiyor. `0,359` ve `0,2608` her geçtiği yerde tutarlı.
**Kontrol edildi, gerçek bir tutarsızlık bulunamadı.**

---

## 4. Bölüm 3 — Proje Haritası ve Hafıza Güncelliği

### Madde 1 — E0-E6'nın 14 yeni `.py` dosyası `PROJE_DOSYA_YAPISI_PAH.md`'de yok (**Orta**)

`find src -name "*.py" -newermt "2026-08-17 14:00:00"` çıktısı (14
dosya) ile `PROJE_DOSYA_YAPISI_PAH.md` içinde `grep` yapıldı — **hiçbiri
(0/14) bulunamadı** (`grep` exit code 1). Eksik dosyalar ve işlevleri:

| Dosya | İşlevi | Haritada olmaması neden sorun |
|---|---|---|
| `src/genova/metrics.py` | Resmi F1/MCC/specificity/ağırlıklı-F1 modülü | Projenin **en kritik** modülü (tüm sıralama kararı buna dayanıyor), haritada hiç yok |
| `src/genova/pah/e1_baseline.py` | E1 baseline yeniden-ölçüm | — |
| `src/genova/pah/fold_versions.py` | v1-v4 fold-güvenli inşacılar (E2'nin temeli) | Yeni sızıntı-güvenlik katmanı, dokümante değil |
| `src/genova/pah/models.py` | Nested-CV çatısı + model sarmalayıcıları | — |
| `src/genova/pah/e2_candidate_features.py` | ZORT_*/AL_-özet | — |
| `src/genova/pah/e2b_run.py`, `e2b_raw_nan_run.py` | 2b çalıştırıcıları | — |
| `src/genova/pah/e1_threshold_check.py`, `e2_oof_correlation.py` | E2-sonrası doğrulamalar | — |
| `src/genova/pah/calibration.py`, `e3_calibration_run.py` | E3 kalibrasyon | — |
| `src/genova/pah/e4_prior_correction.py` | E4 SLD | — |
| `src/genova/pah/e5_threshold_selection.py` | E5 eşik seçimi | — |
| `src/genova/pah/e6_ensemble.py` | E6 ensemble | — |

**Önerilen düzeltme:** `PROJE_DOSYA_YAPISI_PAH.md`'ye bu 14 dosya için
(mevcut `src/genova/pah/` tablosuna ek satırlar olarak) kısa açıklama
satırları eklenmesi.

### Madde 2 — E0-E6'nın ürettiği tablolar da haritada yok (**Orta**)

Aynı `grep` testi `reports/tables/e2_model_comparison.csv` vb. için
tekrarlandı — **0 eşleşme**. `e2_model_comparison.csv`, `e2_oof_
predictions_repeat0.csv`, `e3_calibration_metrics.csv`, `e3_calibrated_
oof_predictions.csv`, `e4_prior_corrected_probabilities.csv`, `e5_
threshold_selection.csv`, `e6_ensemble_metrics.csv`, `e6_oof_correlation_
matrix.csv` — 8 tablo, hiçbiri `PROJE_DOSYA_YAPISI_PAH.md`'de yok. Diğer
`.md` raporlarında da (yalnızca `06_MODEL_SECIM_RAPORU_PAH.md` hariç)
referans edilmiyor. **Önerilen düzeltme:** aynı harita dosyasına
`reports/tables/` bölümüne bu 8 satır eklenmeli.

### Madde 3 — `CLAUDE.md`'nin "Şu anki durum" bölümü stale (**Orta**)

**Tam alıntı (satır 78-80):**
> "## Şu anki durum
>
> Aşama A-D (EDA → ön işleme → özellik seçimi, 25-özellik final havuz)
> tamamlandı ve doğrulandı; split bankası donduruldu. **Aşama E
> (modelleme) başlamak üzere.** Aktif olarak hangi alt-aşamada olduğumuz
> ayrıca görev mesajında belirtilir (bu dosyada güncel tutulmaz)."

"Aşama E başlamak üzere" artık yanlış — Aşama E'nin tamamı (E0-E6)
bitti. Dosyanın kendisi ince taneli alt-aşama takibini yapmayacağını
zaten belirtiyor (bu kısım sorun değil), ama kaba taneli "hangi Aşama"
bilgisi bile güncel değil. **Önerilen güncel metin:**
> "Aşama A-E (EDA → ön işleme → özellik seçimi → modelleme/kalibrasyon/
> önsel-düzeltme/eşik/ensemble) tamamlandı ve doğrulandı; final aday
> CatBoost/v4_from_v2 (Beta kalibrasyon + SLD önsel-düzeltmesi + nested
> eşik≈0,33), solo (ensemble E6'da reddedildi). Aşama F (adversarial
> validation, sağlamlık, açıklanabilirlik, paketleme) başlamak üzere."

### Madde 4 — Planlanan vs gerçek dosya adları (**Düşük**)

| E-F dokümanının planı | Gerçekte üretilen | Değerlendirme |
|---|---|---|
| `models.py` | `models.py` (+ `fold_versions.py`) | Birebir + ek destek modülü |
| `calibration.py` | `calibration.py` (+ `e3_calibration_run.py`) | Birebir + orkestrasyon ayrımı |
| `threshold.py` | `e5_threshold_selection.py` | İsim evrimi |
| `ensemble.py` | `e6_ensemble.py` | İsim evrimi |
| *(plansız)* | `e1_baseline.py`, `e1_threshold_check.py`, `e2_oof_correlation.py`, `e2b_run.py`, `e2b_raw_nan_run.py`, `e2_candidate_features.py`, `e4_prior_correction.py`, `metrics.py` | Görev-akışı sırasında ortaya çıkan meşru genişlemeler |

**Değerlendirme:** Kasıtlı bir isimlendirme evrimi — `eN_*.py` deseni
`e1_baseline.py`'de başladı (E1'in kendi görevinde, plan dışı bir isim
zaten gerekiyordu) ve tutarlılık için sonraki tüm E-alt-aşama dosyalarına
uygulandı; her dosya hangi E-alt-aşamasına ait olduğunu adından
doğrudan okutuyor (planın soyut `threshold.py`/`ensemble.py`'sinden daha
izlenebilir). Dokümantasyon eksikliği değil, bilinçli/tutarlı bir
kural. **Düşük önem, aksiyon önerilmiyor** (yalnızca Madde 1'in harita
güncellemesi bunu yansıtsın yeterli).

---

## 5. Bölüm 4 — Bilinen Açık Uçlar Kayıt Defteri (14 madde)

| # | Madde | Durum | Kanıt | Aksiyon? |
|---|---|---|---|---|
| 1 | `CAT_3=CAT_4=CAT_5` aynı kolon | ✅ Kayıtlı, hâlâ doğru | `CLAUDE.md:39` — "Aşama E'de değerlendirilebilir (UYGULANMADI)" notu hâlâ geçerli; `dataset_versions.py` E0-E6'da hiç değişmedi (mtime 2026-08-15) | Yok |
| 2 | `ek_cat_block_missing` Fisher/"kırılgan ama kalsın" | ✅ Kayıtlı, hâlâ doğru | `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md:67-100`, karar tablosu satır 100: "Şimdilik kalsın, ama kırılgan olarak işaretlendi" | Yok |
| 3 | v3'ün küme-özet özellikleri tüm-veride hesaplanıyor, ama 25-havuzu etkilemiyor | ✅ Teyit edildi, hâlâ doğru | `dataset_versions.py:114-141` değişmedi; `v4_final_feature_pool.json` (25 özellik, hiçbiri `AL_clusterN_*` değil) E0-E6'da değişmedi (mtime sabit) — E2'nin `fold_versions.py::build_v4_from_v3`'ü bu doğrulamayı kod düzeyinde de teyit ediyor (kümeleri hiç kullanmıyor) | Yok |
| 4 | Demo notebook güncel pipeline'la senkron değil, F6'ya ertelendi | ✅ Kayıtlı, hâlâ doğru | `PROJE_DOSYA_YAPISI_PAH.md:67`, ayrıntılı not (27-özellik/CAT_1 eski-FrequencyEncoder sorunları) | F6'da ele alınacak (zaten planlı) |
| 5 | `model_ready_dataset.parquet/json` üretilmemiş, kimse yanlışlıkla notebook'u çalıştırmadı | ✅ Doğrulandı | `find . -iname "model_ready_dataset*"` → 0 sonuç | Yok |
| 6 | sklearn versiyon pinleme yok + `pah_pipeline_tree.joblib` fit-durumu sorunu | ⚠️ **Orta** — kısmen doğrulandı | **Versiyon pinleme:** `pyproject.toml` içeriği yalnızca `[tool.pytest.ini_options]` — hiçbir bağımlılık (sklearn/xgboost/catboost/lightgbm/pandas...) hiçbir yerde pinlenmemiş, CLAUDE.md'nin "tekrar üretilebilir olmalı" kuralına (BAĞLAM) aykırı bir risk, bağımsız olarak doğrulandı. **joblib fit-durumu:** `pah_pipeline_tree.joblib` yüklenip `check_is_fitted()` çağrıldığında `NotFittedError` fırlatıyor (bağımsız olarak yeniden üretildi). **Kök neden bulundu:** pipeline'ın son adımı `ConstantFillImputer`, `fit()`'i hiçbir `_`-sonekli öznitelik set etmiyor (kasıtlı — durumsuz bir dönüştürücü); sklearn'ün `check_is_fitted()`'ı son adımın özniteliklerine bakıyor, bu yüzden yanlış-negatif veriyor. `.transform()` fiilen doğru çalışıyor (test edildi, 5×356 çıktı üretti) — **fonksiyonel bir kırılma değil**, ama E-F dokümanının bu dosya için "sorunsuz" notuyla çelişiyor. Konuşma geçmişinde iddia edilen spesifik "E1'de fark edilen XGBoost uyarısı" olayını bu oturumun bağlamında ya da repo'da doğrulayamadım (bkz. madde 7) — o kısmı onaylamıyorum, yalnızca reddetmiyorum da. | **Evet** — iki ayrı düzeltme: (a) `pyproject.toml`'a bağımlılık pinleri eklenmeli, (b) `ConstantFillImputer`/`BlockMissingIndicator`/`SchemaGate`'in `fit()`'ine sembolik bir `self.fitted_ = True` özniteliği eklenmesi (davranışı değiştirmez, yalnızca `check_is_fitted()` uyumluluğu sağlar) — ayrı bir görev olarak önerilir |
| 7 | XGBoost pickle versiyon-uyarısı (E1'de fark edilen) | ❓ **Doğrulanamadı** — Düşük | `models/pah/baseline_frozen.pkl` bu oturumda yeniden yüklendi, **hiçbir uyarı üretmedi** (aynı xgboost 2.1.4 ortamı). Repo'da (`reports/`, `CLAUDE.md`) bu konuda hiçbir kayıt yok. Bu iddiayı ne doğrulayabiliyorum ne çürütebiliyorum — mevcut bağlamımda bu olayın gerçekleştiğine dair kanıt bulamadım. Spekülasyonla karıştırmamak için "doğrulanamadı" olarak işaretliyorum, "gerçekleşmedi" demiyorum. | Madde 6'nın versiyon-pinleme düzeltmesi bu riski zaten genel olarak azaltır, ek aksiyon önerilmiyor |
| 8 | TabPFN v2/v2.5 engeli | ✅ Kayıtlı, hâlâ doğru | `06_MODEL_SECIM_RAPORU_PAH.md:24-45` — iki bloker (Python 3.8 ortamı + Prior Labs hesap/token) ayrıntılı belgeli, bu denetimde tekrar teyit edildi | Yok |
| 9 | ZORT_*/AL_-özet reddi literatür raporuna geri bağlanmamış | ⚠️ Düşük | `07_LITERATUR_TARAMASI_ASAMA_E_ONCESI_PAH.md:112-143` bu adayları E2'de denenecek diye tanımlıyor ama E2'nin sonucuna (ret) geri dönmüyor — ret yalnızca `06_MODEL_SECIM_RAPORU_PAH.md:149-175`'te var. İkisi de "kabul edilebilir" (görevin kendi ifadesiyle) ama tutarlılık için `07` raporuna tek satırlık bir "Sonuç: E2'de test edildi, reddedildi (bkz. 06 raporu)" eklenmesi önerilir | Düşük öncelik, isteğe bağlı |
| 10 | PFER notu `03_OZELLIK_SECIMI_PAH.md`'de duruyor mu, dosyaya dokunulmadı mı | ✅ Doğrulandı | `03_OZELLIK_SECIMI_PAH.md:216-238` (Meinshausen-Bühlmann bölümü) mevcut; dosya mtime 2026-08-15 19:55 — E0-E6 boyunca değişmedi | Yok |
| 11 | Faz 3 `n_features_final` (25 vs 29) düzeltmesi kalıcı mı | ✅ Doğrulandı | `experiments/tam_16_adim_deneme/results/step13_nested_summary.csv` satır F: `n_features_final=29` (25-havuz+4 ortalama-özellik), `final_report.md:94` ile birebir tutarlı | Yok |
| 12 | `VERI_ON_ISLEME_OZETI_PAH.md`'nin 27→24→25 anlatısı hâlâ eksik mi | ⚠️ Düşük, doğrulandı (hâlâ eksik) | Dosya içinde satır 95 "24 özellik" diyor (o zamanki D-denetimi anlık görüntüsü), satır 159 "25-özellik havuzuyla" diyor — ikisi arasındaki (CAT_1 çok-değerli düzeltmesinin getirdiği 24→25 geçişi) hiçbir yerde açık anlatılmıyor. Dosya mtime 2026-08-15 (E0-E6 kapsamı dışı, dokunulmadı) — düşük öncelikli olarak kalması **uygun**, jüri hazırlığına ertelenmiş durumu hâlâ geçerli | Jüri hazırlığında ele alınacak (zaten planlı) |
| 13 | `reports/05_ACIK_SORULAR.md` henüz açılmamış olmalı | ✅ Doğrulandı | Dosya mevcut değil (`ls` → No such file) — F5'te açılacak, sorun değil | Yok (F5'i bekliyor) |
| 14 | `conflict_group_1` (VAR_003238/VAR_003234) hiçbir E0-E6 fold'unda bölünmedi | ✅ Somut kanıtla doğrulandı | `outer_fold_repeat00.json`'ın 5 fold'unun her birinde iki Variant_ID **her zaman aynı tarafta** (fold 0,1,3,4: ikisi de train; fold 2: ikisi de test) — spot-check + yapısal argüman: split bankası E0-E6 boyunca hiç değişmediği ve tüm `fold_versions.py`/`e3-e6` kodu train/test id listelerini doğrudan bu JSON dosyalarından okuduğu için garanti tüm 500 (50 dış×10) fold için de geçerli | Yok |

---

## 6. Bölüm 5 — Metodolojik Bütünlük (uçtan-uca spot-check)

### Madde 1 — E3 çıktısının E4/E5/E6 boyunca sessizce yeniden üretilmediği doğrulandı

mtime zinciri: `e3_calibrated_oof_predictions.csv`/`e3_calibration_
metrics.csv` = 2026-08-18 10:36:15 (ikisi de aynı saniyede, E3'ün tek
koşumu) → `e4_prior_corrected_probabilities.csv` = 15:23:20 → `e5_
threshold_selection.csv` = 15:45:45 → `e6_ensemble_metrics.csv` =
16:42:53. Monoton artan, doğru nedensellik sırasında (E3→E4→E5→E6), ve
E3'ün dosyaları E4/E5/E6'nın hiçbirinde daha sonraki bir mtime'a
sıçramamış — **hiçbiri E3'ün çıktısını sessizce yeniden üretmemiş,
yalnızca okumuş.** ✅ Doğrulandı.

### Madde 2 — SLD formülünün elle doğrulanması

Rastgele seçilen satır: `lightgbm/v1, repeat=0, outer_fold=4,
VAR_003018`, Beta-kalibre ham olasılık (E3) = `0,6361374302738354`.

```
w1 = 0,286/0,833 = 0,343337...
w0 = 0,714/0,167 = 4,275449...
p_yeni = w1·p / (w1·p + w0·(1-p))
       = 0,343337×0,636137 / (0,343337×0,636137 + 4,275449×(1-0,636137))
       = 0,12311115464403147
```

`e4_prior_corrected_probabilities.csv`'deki kayıtlı değer: **aynı satır,
aynı model/versiyon/repeat/outer_fold/variant_id için `0,12311115464403147`**
— tam kayan-nokta hassasiyetine kadar birebir eşleşiyor. ✅ Doğrulandı.

### Madde 3 — Tek satırlık "Aşama E — Nihai Özet" var mı

**Yok.** Final adayın (CatBoost/v4_from_v2, Beta kalibrasyon, SLD, eşik≈
0,3315) E1 baseline'ına karşı özeti şu an **üç ayrı yerde parça parça**
duruyor: E5'in "Üç adayın nihai karşılaştırması" tablosu (satır 518-525,
ama üç adayı da gösteriyor, tek final adayı değil), E6'nın "Nihai öneri"
paragrafı (satır 666-672, ama sayısal özet tablosu yok, yalnızca düz
metin), ve E2'nin (artık kısmen geçersiz, bkz. Bölüm 2/m.1) "Genel
değerlendirme" tablosu. **Öneri (uygulanmadı, yalnızca önerildi):**
`06_MODEL_SECIM_RAPORU_PAH.md`'nin en başına (Kapsam'dan hemen sonra)
ya da en sonuna, tek bir "Aşama E — Nihai Özet" tablosu eklensin:

| Alan | Değer |
|---|---|
| Final aday | CatBoost, `v4_from_v2` (25 özellik) |
| Kalibratör | Beta |
| Önsel düzeltmesi | SLD (w1=0,343337, w0=4,275449) |
| Eşik | ~0,3315 (nested, dış-fold bazlı, ort±std) |
| Ağırlıklı-F1 (final önsel) | 0,6187±0,0855 |
| E1 baseline'ına göre | F1 ekseninde gerçek kazanan (paired kazanma oranı %62); specificity/ağırlıklı-F1 karşılaştırması eşik-ölçeği farkı nedeniyle doğrudan kıyaslanamaz (bkz. E5) |
| Ensemble | Denendi (E6), solo modeli geçemedi — reddedildi |

---

## 7. Bölüm 6 — Uçtan-Uca Yeniden Üretilebilirlik (**Orta**)

**Madde 1:** Hiçbir dosyada (README yok, `CLAUDE.md`'de yok, `06`
raporunda yok) E0'dan E6'ya **sıralı bir komut listesi** yazılı değil.
Her `.py` dosyasının kendi docstring'inde ayrı ayrı "Çalıştırma: python -m
genova.pah.X" notu var, ama bunları hangi sırada (ve hangi ara
çıktıların hangi sonrakine girdi olduğunu) birleştiren tek bir kaynak
yok. Jüri (ya da başka bir mühendis), sırayı yalnızca dosya adlarından
ve mtime'lardan çıkarsamak zorunda kalır.

**Madde 2 (öneri, uygulanmadı):** `reports/06_MODEL_SECIM_RAPORU_PAH.md`'ye
ya da ayrı bir `REPRODUCE.md`'ye şu sıralı liste eklenmeli (E-F
dokümanının "jüri kodu yeniden çalıştırabilir... tek komutla tekrar
üretilebilir olmalı" kuralına doğrudan bağlı):

```
1. python -m genova.metrics                    # (test edilir, çalıştırılmaz — modül)
2. python -m genova.pah.e1_baseline             # E1: models/pah/baseline_frozen.pkl
3. python -m genova.pah.models                  # E2a: reports/tables/e2_model_comparison.csv
4. python -m genova.pah.e2b_run                 # E2b (ZORT_*/AL_özet)
5. python -m genova.pah.e2b_raw_nan_run         # E2b (v2_raw_nan)
6. python -m genova.pah.e1_threshold_check      # E2-sonrası Doğrulama 1
7. python -m genova.pah.e2_oof_correlation      # E2-sonrası Doğrulama 2
8. python -m genova.pah.e3_calibration_run      # E3: e3_calibration_metrics.csv + e3_calibrated_oof_predictions.csv
9. python -m genova.pah.e4_prior_correction     # E4: e4_prior_corrected_probabilities.csv (girdi: adım 8 çıktısı)
10. python -m genova.pah.e5_threshold_selection # E5: e5_threshold_selection.csv (girdi: adım 3 + 9)
11. python -m genova.pah.e6_ensemble            # E6: e6_ensemble_metrics.csv (girdi: adım 3 + 8 + 9)
```

---

## 8. Öncelik Sıralı Aksiyon Listesi

Hiçbiri Aşama F'yi engellemiyor; hepsi ayrı, kısa bir "E-sonrası
temizlik" görevi olarak ele alınabilir:

1. **(Orta)** `06_MODEL_SECIM_RAPORU_PAH.md`: başlık + Kapsam bölümünü
   E2-E6'yı yansıtacak şekilde güncelle; satır 270'teki stale "CatBoost/
   v1 birincil" önerisine E6'ya atıf notu ekle; öneri edilen "Aşama E —
   Nihai Özet" tablosunu ekle (Bölüm 2/m.1,5; Bölüm 5/m.3).
2. **(Orta)** `PROJE_DOSYA_YAPISI_PAH.md`'ye 14 yeni `.py` dosyası + 8
   yeni tablo için satırlar ekle (Bölüm 3/m.1,2).
3. **(Orta)** `CLAUDE.md`'nin "Şu anki durum" satırını güncelle
   (Bölüm 3/m.3).
4. **(Orta)** `pyproject.toml`'a bağımlılık versiyon pinleri ekle;
   `ConstantFillImputer`/`BlockMissingIndicator`/`SchemaGate`'e
   `check_is_fitted()` uyumluluğu için sembolik `fitted_` özniteliği
   ekle (Bölüm 5, 14-madde tablosu #6).
5. **(Orta)** Sıralı bir `REPRODUCE.md` (ya da `06` raporuna ek bölüm)
   ekle (Bölüm 6).
6. **(Düşük, isteğe bağlı)** `07_LITERATUR_TARAMASI_ASAMA_E_ONCESI_PAH.md`'ye
   ZORT_*/AL_-özet reddinin tek satırlık geri-atfı (14-madde tablosu #9).

Hiçbir Kritik bulgu yok; Düşük bulguların kalanı (isimlendirme evrimi,
VERI_ON_ISLEME_OZETI'nin eksik anlatısı, XGBoost-uyarısı iddiasının
doğrulanamaması) mevcut haliyle kalabilir.
