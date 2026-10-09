# PAH Paneli — Aşama F Öncesi Yapılacaklar

> Kaynak: bağımsız dış denetim (2 tur) + kullanıcının kendi bağımsız
> doğrulaması (kod okuma + sayıları yeniden hesaplama). Aşağıdaki her
> madde, en az bir tarafça koddan/ham veriden doğrulanmıştır — hiçbiri
> yalnızca iddia değildir.

## Durum notu (P2 madde 22 — güncel, P0-P2 kapanışı)

**P0 (1-6): TAMAMLANDI.** Kritik özellik-seçimi sızıntısı düzeltildi,
fold-lokal havuz mimarisi kuruldu, final model hattı (`f0_final_model.py`
→ `predict.py`) baştan üretildi.

**P1 (7-11, 13-14): TAMAMLANDI** (**12 hariç** — F1'e/adversarial
validation'a devredildi, F ile paralel/sonra ele alınacak). Objective
mismatch araştırıldı (madde 7 — sonuç: mevcut model korundu, NB testi
yeni adayı doğrulamadı); kalibrasyon yöntem-seçimi fold-içine taşındı
(madde 8); ağırlıklandırma XGBoost/LightGBM/Elastic-Net'e genişletildi
(madde 9 — sonuç: eleme kararı sağlam); örnek-düzeyi CI/NB/Monte Carlo
resmi istatistik yöntemi yapıldı (madde 10); provenance-riskli özellikler
çıkarıldı, `final_model_bundle_v2.pkl` yeni resmi final oldu (madde 11);
terminoloji ve `classify_al_columns` sıra hatası düzeltildi (madde 13-14).

**P2 (16-19): TAMAMLANDI.** CatBoost ızgarası genişletildi (madde 16 —
sonuç: model korundu); RF'nin eksik-değer adaleti netleştirildi (madde 17
— zayıflık büyük ölçüde doldurma-artefaktıydı); 6 metodoloji testi
denetlendi + eksik olan tamamlandı (madde 18); EK/madde script'leri
idempotent yapıldı (madde 19).

**P3 (20-24): bu turda tamamlandı** (`REPRODUCE.md`, `CLAUDE.md`, bu
dosya, `PROJE_DOSYA_YAPISI_PAH.md`, ZIP hijyeni).

**Sonraki adım: Aşama F (adversarial validation, madde 12 dahil,
sağlamlık stres testleri, SHAP açıklanabilirlik, final paketleme)** —
ayrı bir onayla başlayacak.

## Karar

**Aşama F'ye şu an geçilmemeli.** Final 25-özellik havuzunun (`v4_final_
feature_pool.json`) 10 özelliği, dış-test etiketleriyle hesaplanan
permutation importance sayesinde havuza girmiş — bu, Aşama D.1'in "nested"
olduğu iddiasını geçersiz kılıyor ve ona dayalı tüm E2-E6 sonuçlarını
(final aday dahil) etkiliyor.

**Ama panik gerekmiyor:** Düzeltmenin ölçülen performans maliyeti sıfıra
yakın (bağımsız bir "sızıntısız 15-özellik" denemesi, mevcut sonuçla
istatistiksel olarak eşdeğer çıktı — hatta hafif iyi). Yani muhtemel
sonuç "aynı model, artık savunulabilir bir gerekçeyle" olacak — sıfırdan
farklı bir proje değil, mevcut çalışmanın **temizlenmesi.**

---

## P0 — Zorunlu, F'den önce

| # | Görev | Neden | Etkilenen dosyalar |
|---|---|---|---|
| 1 | `permutation_selected`'ı düzelt — dış-test (`X_test, y_test`) yerine, dış-eğitim fold'unun kendi iç bölmesini (inner-val) kullansın | **Kritik sızıntı** — dış-test etiketi doğrudan özellik-seçim kararına giriyor | `feature_selection.py` |
| 2 | `build_v4_final_pool`'u global/tek-seferlik üretimden çıkar, fold-içi çağrılabilir hale getir; `fold_versions.py`'nin `build_v4_from_v2/v3`'ü artık dış-fold'a özel havuz alsın | Aynı kök sorunun mimari yansıması | `feature_selection.py`, `fold_versions.py` |
| 3 | Düzeltilmiş prosedürle Aşama D.1'i yeniden çalıştır; eski/yeni havuzu karşılaştır, farkı raporla | Doğrulama | `feature_selection_stability.csv`, `v4_final_feature_pool.json` (eskisini arşivle, silme) |
| 4 | CatBoost/RF/LightGBM/XGBoost'u düzeltilmiş `v4` ile 50 dış fold'da yeniden ölç | E2 tablosunun geçerliliği | `e2_model_comparison.csv` (yeni dosyaya yaz, üzerine yazma) |
| 5 | Final eğitim tarifi + serileştirme + `predict.py` — 369'un tamamında tek bir cross-fit prosedürüyle: özellik seti + hiperparametre + kalibratör + tek eşik üretilip `joblib` olarak paketlensin | Şu an deploy edilebilir hiçbir şey yok (yalnızca `baseline_frozen.pkl` var) | Yeni modül: `f0_final_model.py`, `models/pah/`, `predict.py` |
| 6 | `CLAUDE.md`'nin "Şu anki durum" paragrafını düzelt; Kesin Kural #2'ye *"özellik seçimi dahil, dış-test hiçbir seçim kararına giremez"* ibaresi ekle | Bu hatanın bir daha olmaması için | `CLAUDE.md` |

## P1 — Final modelden önce zorunlu (F ile paralel/sonra yapılabilir, F'yi bloklamıyor)

| # | Görev | Neden |
|---|---|---|
| 7 | ✅ **UYGULANDI (mevcut model korundu, gerekçeli — bkz. 06 raporu P1 madde 7 kapanış bölümü)** — `nested_cv_evaluate`'in iç seçim ölçütünü (şu an eşik=0,5 + eğitim-önseli F1) tam boru hattına (kalibrasyon+SLD¹+uyarlanabilir eşik sonrası önsel-ağırlıklı F1) çevir | Objective mismatch — iç seçim ile final hedef aynı değil |
| 8 | ✅ **UYGULANDI (bkz. 06 raporu P1 madde 8 bölümü)** — Kalibrasyon yöntemi (Platt/Beta/Isotonic) seçimini fold-içine taşı; ECE/reliability/slope-intercept ekle | Yöntem seçimi şu an dış-test Brier'ine bakılarak global yapılıyor; "Beta üçünde de kazandı" iddiası CatBoost/v1'de yanlış çıktı (Platt daha iyi) |
| 9 | ✅ **UYGULANDI** — A/B/C ağırlıklandırma karşılaştırmasını tüm final adaylara (özellikle XGBoost) uygula. Sonuç: 4 kombinasyonun (XGBoost/v1, LightGBM/v1, Elastic-Net/v3, Elastic-Net/v4_from_v3) 4'ünde de A (ağırlıksız) en iyi çıktı, ama hiçbiri NB testinde final adaya (CatBoost/v4_from_v2) karşı anlamlı üstünlük göstermedi (p≥0,068 hepsinde) — mevcut eleme kararı sağlam, final model değişmedi. Bkz. `06_MODEL_SECIM_RAPORU_PAH.md` P1 madde 9 bölümü | Şu an yalnızca RF+CatBoost'ta yapıldı, XGBoost'un elenmesi bu yüzden adil değil |
| 10 | ✅ **UYGULANDI** — Örnek-düzeyi bootstrap + Nadeau-Bengio düzeltmesi + 100/250 (şartname) Monte Carlo final-F1 simülasyonunu resmi CI yöntemi yap | Mevcut fold-düzeyi CI (genişlik≈0,047, kendi elimizle doğrulandı) gerçek belirsizliği olduğundan dar gösteriyor |
| 11 | ✅ **UYGULANDI** — Ablation: `al_all_missing` hariç, `CAT_1` hariç — zaten "bedava" çıktığı ölçüldü (p=0,83, maliyetsiz), yine de resmi havuzdan çıkar | Provenance riski düşük öncelikli ama düzeltmesi ücretsiz |
| 12 | Shift stres testi: `CAT_2` leave-one-category-out, missingness-pattern holdout, benign-ağırlıklı validasyon | F1'in (adversarial validation) kapsamına doğal olarak giriyor, ayrıca planlanmalı |
| F1 | ✅ **TAMAMLANDI (model korundu, provenance riski bilinçli kabul edildi ve belgelendi — bkz. 03b nihai bölüm)** — Aşama F1 (adversarial validation): Kaynak-1/2 testleri + özellik-bazlı tarama, `AL_26/AL_12/AL_7/AL_49` provenance-riskli bulundu (adversarial AUC 0,77-1,00); ablasyon denendi, 10 tekrarlı-bölme testinde büyük/tutarlı bir performans maliyeti (10/10, −0,0877±0,0272) doğrulandı — madde 11'in aksine "bedava" değil; final model (`final_model_bundle_v2.pkl`, 26 özellik) korundu, risk F3'e devredildi. Not: madde 12 (`CAT_2` leave-one-category-out vb. spesifik shift-stres testleri) ayrı, hâlâ açık | — |
| F2 | ✅ **TAMAMLANDI** — Bootstrap güven aralıkları resmi kapanışı: `sensitivity` metriği eklendi (`src/genova/metrics.py`, 4 yeni test), `f0_uncertainty_analysis.csv` yeniden üretildi (F1/MCC/specificity/sensitivity, `n_boot=2000`), resmi rapor `reports/10_BOOTSTRAP_GUVEN_ARALIKLARI_PAH.md`'ye yazıldı (dosya numarası notu: `08` çakıştığı için `10` kullanıldı). E-F dokümanının "final test setinde ~13-20 benign" YANLIŞ ifadesi düzeltildi (gerçek neden: final set 250 benign içeriyor, geniş CI bizim 61-benign geliştirme setimizden kaynaklanıyor) | — |
| F3 | ✅ **TAMAMLANDI (kritik kırılganlık bulundu, kök nedeni özellik değil alt-küme yapısı, model korundu, sınırlama belgelendi)** — `CAT_1` doluluk-geçişi testi ciddi bir çöküş buldu (kritik yönde AUC 0,831→0,573); takip (26 vs 23 vs 22 özellik karar matrisi) bunun `AL_26/12/7/49`'dan kaynaklanmadığını kanıtladı (özellik çıkarma çöküşü düzeltmiyor); kök neden karakterizasyonu `CAT_1`-boş alt-kümenin (n=132) etiket dengesinde değil `AL_` eksiklik oranında (%91,4 vs %36,6) yapısal olarak farklı olduğunu gösterdi. Model `A` (`final_model_bundle_v2.pkl`, 26 özellik) korundu, sınırlama `03b`'ye belgelendi. *(Not: bu satır, aynı maddenin ilk turundaki daha erken/"DURDURULDU" notunu — o zamanki ara bulgu — günceller; ara not karışıklığı önlemek için kaldırıldı.)* | — |
| F4 | ✅ **TAMAMLANDI** — SHAP açıklanabilirlik (final model değiştirilmedi, saf açıklama turu): global sıralamada `AL_7` #1/26 — F1/F3'ün riskli 4 özelliğinin (`AL_26/12/7/49`) SHAP katkısının "küçük" olacağı beklentisi doğrulanmadı (tersine önemliler), ama `CAT_1`-doluluğuna göre kırılımda hepsi `CAT_1`-boş alt-kümede belirgin düşük katkı gösterdi (1,4×-3,0×) — F3'ün "model bu özellikleri o alt-kümede öğrenemiyor" mekanizmasını görsel/sayısal olarak doğruladı. Aşama D stabilite sıralamasıyla Spearman=0,40 (orta, beklenen bir sapma — SHAP model-spesifik, Aşama D yöntem-ortalaması). Bkz. `06_MODEL_SECIM_RAPORU_PAH.md` Aşama F4 bölümü | — |
| F5 | ✅ **TAMAMLANDI (PAH'ın kendi tarafı)** — Panel birleştirme hazırlığı: `predict.py` çıktısına `predicted_probability` (SLD-düzeltilmiş, ham) ve `panel="PAH"` kolonları eklendi (`predicted_label` eski `Label`'ın yeniden adlandırılmış hâli), `PAH_SUBMISSION_INTERFACE.md` sözleşmesi yazıldı, panel-arası kararlar (tek/çok dosya, `Variant_ID` çakışma riski, ortak eşik, rapor formatı) `reports/F5_ACIK_SORULAR_PANEL_BIRLESTIRME.md`'de takım koordinasyonu için açık soru olarak bırakıldı (cevaplanmadı, yalnızca listelendi). Gerçek panel-arası birleştirme diğer panellerin hazır olmasını bekliyor | — |
| F6 | ✅ **TAMAMLANDI** — Final paketleme denetimi (hafif, hedefli): F0-P2'nin çoğu zaten yapılmıştı; 4 kozmetik/keşfedilebilirlik boşluğu bulunup kapatıldı — (1) `RankQuantileHarmonizer`'ın v3-hattına özgü, final modeli etkilemeyen kusuru `reports/11_BILINEN_KUSURLAR_PAH.md`'ye belgelendi, (2) kök `README.md` eklendi, (3) `PROJE_DOSYA_YAPISI_PAH.md`'nin rapor tablosuna eksik olan 04-11 + `F5_ACIK_SORULAR_...` satırları eklendi, (4) `PAH_SUBMISSION_INTERFACE.md`'ye kısa model kartı eklendi. Sürüm pinleri, `check_is_fitted`, `predict.py` uç-durumları yeniden doğrulandı — hepsi sorunsuz. Kod/model/test mantığı değişmedi (yalnızca bir test yorumunun dosya referansı güncellendi), `pytest tests/ -q` 163/163. **Aşama F (F1-F6) tamamen kapandı** | — |
| 13 | ✅ **UYGULANDI** — `e4_prior_correction.py`'yi yeniden adlandır ("SLD" değil, "kapalı-form Bayes önsel düzeltmesi" — SLD, önsel bilinmediğinde EM ile tahmin eder, burada önsel biliniyor); docstring'deki `w1=TRAIN/FINAL` yazım hatasını düzelt (kod zaten doğru, `FINAL/TRAIN`) | Terminoloji/dokümantasyon doğruluğu |

¹ **Terminoloji notu:** bu belgede "SLD" ifadesi tarihsel tutarlılık için
olduğu gibi bırakıldı; gerçek yöntem SLD (Saerens-Latinne-Decaestecker)
değil, hedef önsel BİLİNDİĞİ için kullanılan basitleştirilmiş **kapalı-form
Bayes (Elkan-Noto tarzı) önsel düzeltmesi**dir (bkz. madde 13, ve
`e4_prior_correction.py`'nin güncellenmiş docstring'i).

## P1'e yeni eklenen (2. denetim turunda bulundu)

| # | Görev | Neden |
|---|---|---|
| 14 | ✅ **UYGULANDI** — `classify_al_columns`'ın sıfır-doldurmadan önce/sonra çağrılma sırasına göre tamamen farklı sonuç verdiğini düzelt (90/82/162 vs 24/309/1 — kendi elimle doğrulandı) — `serialize_pipelines.py`'nin ürettiği `pah_pipeline_transformed.joblib`, nested CV'de fiilen kullanılan `v3` ile **aynı boru hattı değil** | Ciddi, önceden fark edilmemiş bir tutarsızlık — F6 paketlemesine girerse yanıltıcı olur |
| 15 | `pyproject.toml`'daki sürüm pinlerinin gerçekten uygulandığından emin ol; çapraz-ortam testinde fold-başına fark 0,023'e kadar çıkabiliyor | Jüri farklı ortamda farklı sayı görebilir |

## P2 — Faydalı (F'yi bloklamıyor, kalite artışı)

| # | Görev |
|---|---|
| 16 | ✅ **UYGULANDI** — CatBoost ızgarasını genişlet — `learning_rate`, early stopping ekle (2 adaydan 6 adaya). Sonuç: mevcut model (`depth=5, lr=0,05`) 6 adayın en iyisi olarak kazandı (doğru ölçütle, tek 4-fold bölünmede) — model korundu, tekrarlı-bölme/NB testi gerekmedi. Bkz. `06_MODEL_SECIM_RAPORU_PAH.md` P2 madde 16 bölümü |
| 17 | ✅ **UYGULANDI** — RF/KNN'in eksik-değer işlemesini (medyan-doldurma) diğer ağaç modelleriyle (native NaN) eşitle ya da farkı raporda açıkça belgele. Sonuç: `AL_` için sıfır-doldurma (v2'nin stratejisi), medyan-doldurmaya göre RF'nin MCC'sini ~5-9× artırıyor (0,04-0,06 → 0,31-0,33) — zayıflık büyük ölçüde doldurma-artefaktıydı, ama tam değil (hâlâ CatBoost'un MCC'sinin altında). Final model kararı değişmedi (RF zaten final değil). Bkz. `06_MODEL_SECIM_RAPORU_PAH.md` P2 madde 17 bölümü |
| 18 | ✅ **UYGULANDI (denetlendi + eksik tamamlandı)** — Aşağıdaki 6 metodoloji testini yaz. Denetim sonucu: 5/6 zaten P0/P1'in yan ürünü olarak yazılmıştı (isim farklı); yalnızca #4 (iç/dış ölçüt tutarlılığı) gerçekten eksikti, `tests/test_methodology_regression_pah.py`'ye 2 yeni test eklendi (ic-CV bölme paylaşımı + `final_model_bundle_v2.pkl`'in kod ile hâlâ tutarlı/güncel olduğu — ikisi de geçti, sürüklenme yok). Bkz. o dosyanın modül docstring'indeki tam denetim tablosu |

```
test_feature_selection_never_sees_outer_test
test_feature_pool_is_fold_local
test_calibration_method_choice_is_fold_local
test_inner_objective_matches_outer_metric
test_final_pipeline_roundtrip
test_ci_uses_sample_level_resampling
```

| 19 | ✅ **UYGULANDI** — `e2ek_run.py` gibi EK scriptlerini idempotent yap. `idempotent_io.py::upsert_csv` (yeni) 6 çağrı noktasında (5 dosya: `models.py`, `e2ek_run.py`, `e3ek_rf_calibration.py`, `e4ek_rf_prior_correction.py`, `e5ek_rf_threshold_selection.py`) `mode="a"`'nın yerini aldı; 5 yeni testle (`test_idempotent_io_pah.py`) doğrulandı |

## P3 — Dokümantasyon (düşük öncelik ama ucuz)

| # | Görev |
|---|---|
| 20 | ✅ **UYGULANDI** — `REPRODUCE.md`'yi kapsamlı güncelle — E1-EK...E6-EK + P0-P2'nin tamamı (32 adım, 6 bölüm) eklendi, "87 test" → **158** |
| 21 | ✅ **UYGULANDI** — `CLAUDE.md`'deki "5 satırlık çelişki grubu" ifadesine dedup-sonrası gerçeği (2 satır, `conflict_group_1`) netleştiren parantez eklendi (2 yerde) |
| 22 | ✅ **UYGULANDI** — `08_ASAMA_E_SONRASI_DENETIM_PAH.md`'nin başına güncel-olmadığı notu eklendi; bu dosyanın (09) başına da P0-P2'nin tamamının durumunu özetleyen not eklendi |
| 23 | ✅ **UYGULANDI** — `PROJE_DOSYA_YAPISI_PAH.md`'ye E*-EK + P0-P2'nin ürettiği ~50 yeni dosya/artefakt referansı eklendi (src/, tests/, reports/tables/, models/pah/, kök dizin) |
| 24 | ✅ **UYGULANDI** — ZIP hijyeni: `.gitignore` oluşturuldu (yoktu) + mevcut `catboost_info`/`__pycache__`/`.pytest_cache` temizlendi; notebook'un kaynak kodu zaten göreli yol kullanıyordu (yalnızca eski bir çıktı hücresinde mutlak yol vardı, kozmetik); `configs/pah/v4_from_v2.yaml` ve `v4_from_v3.yaml`'da gerçek mojibake bulundu ve düzeltildi |

---

## Kullanıcının (Teknofest Pdr) eklediği nüanslar — denetimin kendi metnine göre yumuşatılmış/netleştirilmiş noktalar

- **Madde 11 (`al_all_missing`/`CAT_1` ablasyonu):** Denetimin ilk turda "Kritik shortcut riski" dediği bu özellik, ikinci turdaki kendi ablasyon deneyinde **maliyetsiz** çıktı (p=0,83) — yani öncelik P1'den P2'ye yakın bir yere çekilebilir, ama "bedava" olduğu için yine de yapılmalı.
- **Madde 10 (taze split bankası, repeat 10-19):** Denetimin önerdiği bu çözüm gerçek bir bağımsız holdout **değil** — aynı 369 satırın farklı bir bölünmesi. Faydalı bir ek kontrol ama "artık genelleme tahmini tarafsız" demek için yeterli değil; bu sınırı jüri sunumunda da açıkça belirtmek gerekir.
- **`AL_correlation_clusters.csv`'nin tüm-veride hesaplanması:** İkinci turda ölçüldü, final adayı (`v4_from_v2`) **hiç etkilemiyor** (final havuzda hiçbir küme-özet kolonu yok) — bu madde P1'den P3'e indirildi, yalnızca dokümantasyon notu yeterli.

## Önerilen uygulama sırası

1. **P0 (1-6)** — tek, kontrollü bir görev turu: düzeltme → yeniden koşum → eski/yeni havuz karşılaştırması → DURDUR, onay bekle.
2. **P0-5 (final model hattı)** — ayrı bir tur, P0'ın geri kalanı onaylandıktan sonra.
3. **P1 (7-15)** — F1 (adversarial validation) ile doğal olarak örtüşenler (12) F'ye devredilebilir; geri kalanı (7-9, 13-15) F'den önce ya da paralel, küçük ayrı turlar halinde.
4. **P2-P3** — zaman kalırsa, F6'ya kadar herhangi bir noktada.
