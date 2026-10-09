# PAH Paneli — Aşama E2-E6: Model Portföyü, Kalibrasyon, Eşik Seçimi ve Ensemble Değerlendirmesi

## Aşama E — Nihai Özet

| Alan | Değer |
|---|---|
| Final aday | CatBoost, `v4_from_v2` (25 özellik) |
| Kalibratör | Beta |
| Önsel düzeltmesi | SLD¹ (w1=0,343337, w0=4,275449) |
| Eşik | ~0,3315 (nested, dış-fold bazlı, ort±std) |
| Ağırlıklı-F1 (final önsel) | 0,6187±0,0855 |
| E1 baseline'ına göre | F1 ekseninde gerçek kazanan (paired kazanma oranı %62); specificity/ağırlıklı-F1 doğrudan kıyaslanamaz (eşik-ölçeği farkı, bkz. E5) |
| Ensemble | Denendi (E6), solo modeli geçemedi — reddedildi |

¹ **Terminoloji notu (P1 madde 13):** bu belgede ve kodda "SLD" olarak anılan
düzeltme, gerçek SLD (Saerens-Latinne-Decaestecker) değildir — SLD, hedef
önsel **bilinmediğinde** EM ile iteratif tahmin eder; burada hedef önsel
(%28,6, şartnameden) **biliniyor**, dolayısıyla kullanılan yöntem SLD'nin
basitleştirilmiş **kapalı-form Bayes (Elkan-Noto tarzı) önsel düzeltmesi**dir.
Formül/sabitler/sonuçlar etkilenmedi, yalnızca isimlendirme düzeltiliyor;
metnin geri kalanında "SLD" ifadesi tarihsel tutarlılık için olduğu gibi
bırakıldı.

## Kapsam

`GENOVA_PAH_ClaudeCode_Prompt_AsamaE-F_GUNCEL.md`'nin E2-E6 önceliklendirmesine göre:

- **Aşama 2a (çekirdek matris): TAMAMLANDI.** CatBoost/XGBoost/LightGBM ×
  {v1, v2, v4_from_v2}, Elastic-Net × {v3, v4_from_v3}. TabPFN v2 hariç
  (aşağıya bakın — engellendi, atlandı).
- **Aşama 2b (genişletme): TAMAMLANDI.** Üç ağaç modeli × `v2_raw_nan`;
  `ZORT_*` ve `AL_`-özet aday özellik setleri, en iyi çıkan model ailesiyle
  (CatBoost) `v1` üzerine eklenerek denendi.
- **Aşama 2c (RealMLP/TabM, TabPFN×v3): BAŞLANMADI.** Görev metninde
  "zaman kalırsa" olarak işaretlenmişti; 2a+2b'nin toplam hesaplama
  maliyeti (~3 saat, aşağıya bakın) ve TabPFN'in tamamen engellenmiş olması
  nedeniyle bu turda kapsam dışı bırakıldı. Onay olursa ayrı bir tur olarak
  yapılabilir.
- **Aşama E3 (nested kalibrasyon): TAMAMLANDI.** Üç aday (CatBoost/v1,
  CatBoost/v4_from_v2, LightGBM/v1) Platt/Beta/Isotonic ile karşılaştırıldı;
  Beta üçünde de kazandı, Isotonic açık aşırı-uyum gösterdi.
- **Aşama E4 (final önsel düzeltmesi, SLD): TAMAMLANDI.** Beta-kalibre
  olasılıklar, eğitim önselinden (%83,3) final önsele (%28,6) kapalı-form
  Bayes düzeltmesiyle taşındı.
- **Aşama E5 (nested eşik seçimi): TAMAMLANDI.** Üç aday için de ayrı
  ayrı, dış-teste hiç bakmadan, önsel-ağırlıklı F1'i maksimize eden eşik
  seçildi; adaptif eşik, sabit 0,5'i üçünde de net biçimde geçti.
- **Aşama E6 (ensemble): TAMAMLANDI.** CatBoost/v4_from_v2 + LightGBM/v1
  ikilisi (OOF korelasyon kanıtına göre seçildi) üç blend yöntemiyle
  denendi; en iyi blend bile solo CatBoost/v4_from_v2'yi geçemedi —
  **nihai öneri solo CatBoost/v4_from_v2** (yukarıdaki özet tabloya bakın).

Tüm sonuçlar `data/splits/pah/`'ın **değiştirilmeden** okunan 50 dış
fold'unda (10 tekrar × 5 fold); hiperparametre araması yalnızca her dış
fold'un kendi iç 4-fold'unda, dar/sabit ızgaralarla (aşağıya bakın).
Split bankasına veya `data/processed/pah/`'a hiçbir yazma yapılmadı.

## TabPFN v2 — Sonradan Denendi, KAPATILDI (model değişmedi)

İki bağımsız bloker tespit edildi, ikisi de kullanıcıyla görüşülüp bu tur
için **atlanmasına** karar verildi:

1. **Ortam blokeri:** Bu oturumun ana Python'u 3.8; PyPI'de TabPFN v2/v2.5
   yalnızca Python ≥3.9 gerektiriyor, `pip install tabpfn` burada yalnızca
   eski TabPFN v1'i (0.1.11, 2022, ≤100 özellik/≤1000 satır sınırı)
   buluyor. Ayrı bir Python 3.11 sanal ortamı kurularak gerçek TabPFN v2
   (8.3.0, Prior Labs) başarıyla kuruldu.
2. **Hesap/lisans blokeri:** TabPFN v2, yerel çıkarım için önceden eğitilmiş
   model ağırlıklarını indirmeden önce bir Prior Labs (ux.priorlabs.ai)
   hesabı + lisans onayı + API token istiyor — bu benim adıma
   yapılamayacak bir üçüncü-taraf kayıt işlemi.

Sonuç (o turda): TabPFN v2/v2.5, ne ana ortamda ne de ayrı kurulan 3.11
ortamında çalıştırılamadı. `07_LITERATUR_TARAMASI_ASAMA_E_ONCESI_PAH.md`'nin
"CatBoost/RealMLP TabPFN'i bu rejimde geçebiliyor, üstünlüğü varsayma"
uyarısı bu yüzden ampirik olarak sınanamadı — E2 sonuçları TabPFN'i
içermiyor. İleride bir Prior Labs API token sağlanırsa, kurulu 3.11
ortamı (`C:\tmp\tabpfn_env`) ve mevcut `fold_versions.py`/`metrics.py`
altyapısı doğrudan kullanılabilir.

### Güncelleme — token sağlandı, gerçekten çalıştırıldı, sonra KAPATILDI

Kullanıcı bir Prior Labs API token'ı oluşturdu (hiçbir dosyaya/rapora/koda
yazılmadı, yalnızca tek seferlik ortam değişkeni olarak kullanıldı).
İzole `C:\tmp\tabpfn_env` (Python 3.11.3, `tabpfn==8.3.0`) hâlâ
duruyordu — token ile lisans/kimlik doğrulama başarıyla tamamlandı,
TabPFN v2 gerçekten çalıştırıldı.

**Sentetik veri sağlamlık kontrolü:** `make_classification` (n=200)
üzerinde AUROC=0,9226, tek fit ~25s — TabPFN gerçekten çalışıyor
doğrulandı.

**PAH verisiyle, split bankasının 50 dış fold'unda, sabit eşik=0,5
(kalibrasyonsuz — E2'nin rejimiyle aynı), hiçbir hiperparametre araması
yapılmadan** (TabPFN in-context learning, arama gerektirmiyor):

| Versiyon | F1 | MCC | AUPRC | Specificity | Sensitivity | AUROC | Ort. süre/fold |
|---|---|---|---|---|---|---|---|
| `v2` | 0,9321±0,0230 | 0,4772±0,1347 | 0,9517±0,0217 | 0,3057±0,1129 | 0,9938±0,0132 | 0,8075±0,0523 | 24,1s |
| `v4_from_v2` | 0,9330±0,0212 | 0,4812±0,1397 | 0,9529±0,0225 | 0,3169±0,1193 | 0,9925±0,0113 | 0,8221±0,0503 | 6,7s |

CatBoost/v4_from_v2'nin **aynı sabit-eşik/kalibrasyonsuz rejimdeki**
E2 sayılarıyla (F1=0,9266±0,0230, AUROC=0,8364, bkz. Revize turunun
"Tam karşılaştırma tablosu") karşılaştırıldığında: TabPFN'in ham F1'i
marjinal olarak yüksek (+0,006), AUROC'u marjinal olarak düşük
(−0,014) — bu düzeyde **açık bir üstünlük ya da açık bir kötülük
göstermiyor**, iki model pratikte yakın.

**Kapatma kararı:** Bu turda başlatılan daha derin karşılaştırma
(TabPFN'in kendi nested kalibrasyon+SLD+eşik-seçim boru hattından
geçirilip final adaya karşı Nadeau-Bengio testiyle weighted-F1
üzerinden karşılaştırılması — Adım 4) **tamamlanmadan kullanıcı
tarafından durduruldu**: takımın diğer üyeleri TabPFN'i (muhtemelen
diğer panellerde/bağlamlarda) zaten denemiş ve kötü sonuç almışlar.
Bu dış geri bildirim, PAH'ın kendi ölçtüğü "belirgin üstünlük yok"
bulgusuyla tutarlı — TabPFN'in ölçülebilir bir kazanç sağlamadığı iki
bağımsız kaynaktan (PAH'ın kendi raw dış-CV'si + takımın diğer
denemeleri) doğrulanmış oldu. **Final model (`CatBoost/v4_from_v2`,
`final_model_bundle_v2.pkl`) değişmedi, split bankasına/`predict.py`'ye
dokunulmadı.**

**Not — ölçülmeyen kısım açıkça işaretleniyor:** Nested kalibrasyon+SLD+
weighted-F1 karşılaştırması (Adım 4-5) çalıştırılmadan durduruldu; bu
yüzden final adaya karşı istatistiksel (NB p-değeri) bir karşılaştırma
**yok** — yalnızca yukarıdaki ham (sabit eşik=0,5) sayılar mevcut. Bu
kasıtlı bir eksiklik, unutma değil.

**Toplam hesaplama süresi:** ~53 dakika (50-fold ham değerlendirme
[v2+v4_from_v2, 100 fit] ~26dk + kalibrasyon boru hattı verisi
[v4_from_v2, 50 outer+200 inner fit, Adım 4'ün yarım kalan kısmı] ~28dk,
paralel çalıştırıldığı için gerçek duvar-saati ~28dk).

### Kod / dosyalar

- `src/genova/pah/g3_export_folds_for_tabpfn.py`, `g3_export_calibration_
  pipeline.py`, `g3_tabpfn_calibrate_and_compare.py` (yeni, ana ortamda
  — split bankasını okuyup TabPFN ortamına aktarma + sonradan
  kalibrasyon/karşılaştırma).
- İzole ortamda (`C:\tmp\tabpfn_env`, repo dışı) iki gecici script
  (`eval_folds.py`, `predict_folds.py`) — repo'ya dahil değil.
- **Bulunup düzeltilen bir hata:** ilk ihracat turunda dış fold seed'i
  yanlışlıkla tüm tekrarlar için sabit `seed=42` verilmişti
  (`split_bank.py::main()`'in gerçek kuralı `seed=BASE_SEED+repeat_idx`)
  — bu, 10 "tekrarın" aslında aynı tek 5-fold'un kopyası olmasına yol
  açıyordu. Fark edilip düzeltildi, düzeltilmiş split'in split bankasının
  kendi diskteki dosyalarıyla birebir eşleştiği doğrulandı, tüm
  hesaplamalar SIFIRDAN yeniden çalıştırıldı — yukarıdaki sayılar
  düzeltilmiş, doğru split'e ait.

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı. API token hiçbir dosyaya/kalıcı yere yazılmadı.

## Sabit hiperparametre ızgaraları (dar, bütçe kontrollü)

| Model | Izgara | Sabit | Neden dar |
|---|---|---|---|
| CatBoost | depth ∈ {3,5} (lr=0,05 sabit) | iterations=100 | ~2s/fit — 50 fold × 5 fit × iterasyon başına maliyet çok yüksek |
| XGBoost | max_depth ∈ {3,5} × lr ∈ {0,03; 0,1} | n_estimators=200, min_child_weight=5 | E1 baseline'ıyla aynı mimari aile |
| LightGBM | num_leaves ∈ {7,15} × lr ∈ {0,03; 0,1} | n_estimators=200 | — |
| Elastic-Net | C ∈ {0,1; 1,0} × l1_ratio ∈ {0,3; 0,7} × scaler ∈ {robust, standard} | solver=saga, max_iter=1000 | ölçekleyici araması dahil, yine de 8 aday |

Azınlık sınıf (Label=0) örneklerine ×2,0 `sample_weight` — E1'in politikasıyla
tutarlı, tüm E2 model ailelerine aynen uygulandı (ayrı bir hiperparametre
olarak aranmadı). Karar eşiği tüm E2 ölçümlerinde **sabit 0,5** — E2 bir
model/versiyon karşılaştırma aşaması, eşik seçimi E5'in işi; F1'in yanında
eşik-bağımsız AUPRC de raporlanıyor.

Toplam hesaplama süresi: 2a ~2 saat (CatBoost ~60dk, XGBoost ~26dk,
LightGBM ~8dk, Elastic-Net ~27dk), 2b ~50dk (ZORT/AL-özet ~17dk,
`v2_raw_nan` ~32dk) — hepsi arka planda, tam 50-fold sadakatiyle koşuldu
(bkz. `CLAUDE.md` kural 7: split bankası küçültülmeden kullanıldı).

## Tam karşılaştırma tablosu

`reports/tables/e2_model_comparison.csv` (800 satır = 16 kombinasyon × 50
dış fold). Özet (dış-CV ortalama±std, F1'e göre sıralı):

| Model | Versiyon | F1 | MCC | AUPRC | Specificity |
|---|---|---|---|---|---|
| CatBoost | v1 | **0,9313±0,0214** | 0,4897±0,1163 | 0,9417±0,0281 | 0,3740±0,0971 |
| CatBoost | v1+ZORT_* | 0,9303±0,0231 | 0,4881±0,1155 | 0,9450±0,0282 | 0,3857±0,1074 |
| CatBoost | v1+AL_özet | 0,9287±0,0218 | 0,4743±0,1165 | 0,9422±0,0285 | 0,3636±0,0978 |
| CatBoost | v2_raw_nan | 0,9272±0,0219 | 0,4582±0,1135 | 0,9405±0,0271 | 0,3561±0,0974 |
| CatBoost | v4_from_v2 | 0,9266±0,0230 | 0,4637±0,1287 | 0,9568±0,0253 | 0,3767±0,1104 |
| CatBoost | v2 | 0,9265±0,0201 | 0,4383±0,1444 | 0,9428±0,0285 | 0,3360±0,1163 |
| LightGBM | v1 | 0,9199±0,0257 | 0,3926±0,1375 | 0,9373±0,0343 | 0,3153±0,1073 |
| XGBoost | v1 | 0,9186±0,0265 | 0,4043±0,1390 | 0,9344±0,0368 | 0,3542±0,1186 |
| LightGBM | v2_raw_nan | 0,9181±0,0246 | 0,3764±0,1431 | 0,9384±0,0321 | 0,3131±0,1142 |
| LightGBM | v4_from_v2 | 0,9178±0,0234 | 0,4184±0,1231 | 0,9556±0,0254 | 0,3926±0,1020 |
| XGBoost | v2_raw_nan | 0,9176±0,0267 | 0,3970±0,1334 | 0,9328±0,0381 | 0,3495±0,1170 |
| LightGBM | v2 | 0,9168±0,0232 | 0,3724±0,1279 | 0,9359±0,0372 | 0,3171±0,1004 |
| XGBoost | v4_from_v2 | 0,9150±0,0273 | 0,4060±0,1194 | 0,9511±0,0315 | 0,3925±0,1204 |
| XGBoost | v2 | 0,9114±0,0265 | 0,3504±0,1328 | 0,9299±0,0423 | 0,3161±0,1083 |
| Elastic-Net | v4_from_v3 | 0,9001±0,0279 | 0,3275±0,1280 | 0,9437±0,0301 | 0,3763±0,1601 |
| Elastic-Net | v3 | 0,8989±0,0281 | 0,3556±0,1138 | 0,9399±0,0311 | 0,4191±0,1206 |

## E1 baseline'ıyla eşleştirilmiş (paired, aynı 50 fold) karşılaştırma

Ham ortalama farkları yanıltıcı olabilir (farklı fold'larda ölçülmüş
sanılabilir) — bu yüzden her kombinasyon, E1'in **aynı** 50 dış fold'undaki
(`models/pah/baseline_frozen.pkl::outer_cv_results_v1`) sonucuyla fold-fold
eşleştirilip fark alındı:

| Model | Versiyon | ΔF1 (ort.) | Kazanma oranı | ΔSpecificity (ort.) |
|---|---|---|---|---|
| CatBoost | v1+ZORT_* | **+0,0104** | **%82** | +0,1249 |
| CatBoost | v1 | +0,0114 | %78 | +0,1131 |
| CatBoost | v1+AL_özet | +0,0088 | %66 | +0,1028 |
| CatBoost | v2_raw_nan | +0,0073 | %66 | +0,0952 |
| CatBoost | v4_from_v2 | +0,0067 | %62 | +0,1158 |
| CatBoost | v2 | +0,0066 | %58 | +0,0751 |
| LightGBM | v1 | +0,0000 | %38 | +0,0545 |
| XGBoost | v1 | −0,0013 | %32 | +0,0933 |
| LightGBM | v2_raw_nan | −0,0018 | %40 | +0,0523 |
| LightGBM | v4_from_v2 | −0,0021 | %34 | +0,1318 |
| XGBoost | v2_raw_nan | −0,0023 | %36 | +0,0887 |
| LightGBM | v2 | −0,0030 | %30 | +0,0562 |
| XGBoost | v4_from_v2 | −0,0048 | %32 | +0,1317 |
| XGBoost | v2 | −0,0084 | %26 | +0,0553 |
| Elastic-Net | v4_from_v3 | −0,0198 | %12 | +0,1155 |
| Elastic-Net | v3 | −0,0210 | %16 | +0,1583 |

**Yorum — asıl bulgu:** Yalnızca **CatBoost'un altı varyantı** E1
baseline'ını F1'de tutarlı biçimde geçiyor (kazanma oranı %58-82).
XGBoost/LightGBM ise F1'de pratikte **düz** (kazanma oranı ~%30-40, ortalama
fark ±0,003 civarı, gürültü seviyesinde — 50 fold'un F1 std'si ~0,02-0,03).
Buna karşın **specificity hemen hemen her kombinasyonda** (Elastic-Net dahil)
baseline'dan belirgin şekilde daha iyi (+0,05 ile +0,16 arası). Bunun tek
nedeni "daha iyi model" değil — kısmen bir **eşik artefaktı**: E1 PDR'nin
sabit eşiğini (0,359) kullanıyordu, E2 ise tüm kombinasyonlarda sabit 0,5
kullanıyor; daha yüksek eşik otomatik olarak daha fazla benign tahmini
üretir. Bu yüzden specificity farkını "model E2'de gerçekten daha iyi
ayırt ediyor" diye okumak yerine, **F1 farkının** (eşikten nispeten daha az
etkilenen) asıl sinyal olduğu kabul edildi — ve F1'de gerçek/tutarlı
kazanan yalnızca CatBoost ailesi.

## `v2` vs `v2_raw_nan` (sıfır-dolgu vs ham-NaN)

| Model | Ort. Δ(F1_raw − F1_v2) | raw_nan kazanma oranı |
|---|---|---|
| CatBoost | +0,0007 | %50 |
| XGBoost | +0,0061 | %60 |
| LightGBM | +0,0013 | %46 |

**Kazanan: pratik bir fark yok.** CatBoost ve LightGBM için tamamen kararsız
(~%50 kazanma oranı), XGBoost'ta hafif bir eğilim ham-NaN lehine (+0,006,
%60) ama fark std'nin (~0,025-0,027) çok altında — istatistiksel olarak
anlamlı değil. `v1`/`v2` (sıfır-dolgu) ile karşılaştırıldığında da CatBoost
zaten en iyi F1'i sıfır-dolgu **olmayan**, ham `v1`'de veriyor. Sonuç:
AL_ sıfır-doldurma politikası (v2'nin varsayılanı) bu ağaç modeli
üçlüsünde performans kaybına yol açmıyor, ama getirisi de yok — basitlik
lehine, ekstra karmaşıklık (iki ayrı AL_ doldurma varyantı taşımak)
gerekçelendirilmiyor.

## `ZORT_*` ve `AL_`-özet aday özellik setleri (yalnızca CatBoost, en iyi aile)

İki aday da CatBoost+v1'in **üzerine** eklenerek, aynı 50 fold'da,
`catboost/v1`'e karşı paired karşılaştırıldı (ham tablodaki E1-baseline
karşılaştırması değil, doğrudan `catboost/v1`'e karşı):

| Aday | Ort. ΔF1 (vs catboost/v1) | Kazanma oranı |
|---|---|---|
| `v1` + `ZORT_*` (4 özellik) | −0,0010 | %32 |
| `v1` + `AL_`-özet (4 özellik) | −0,0026 | %30 |

**Karar: ikisi de resmi havuza girmiyor.** Her ikisi de `catboost/v1`'e
karşı paired karşılaştırmada kazanma oranı %30-32 (yani fold'ların
çoğunda **kaybediyor**) — Faz 3 deneyinin farklı bir model/versiyon
kombinasyonunda (entropy-tree ensemble, 25-özellik havuzu) bulduğu +0,018
Macro-F1 iyileşmesi burada (CatBoost + ham `v1`, resmi pozitif-sınıf F1)
**doğrulanmadı**. Bu, E-F dokümanının kendi karar kuralına birebir uyuyor:
"yalnızca nested CV'de performans iyileşmesi gösterirse ana modele aday
gösterilir" — göstermedi, dolayısıyla önerilmiyor.

`AL_`-özet özellikleri için önemli bir düzeltme not edilmeli: ilk
uygulamada 334 AL_ kolonunun tamamı (90 "constant" kolon dahil — bazıları
~264690 gibi frekans-ölçeğinden tamamen kopuk sabit değerler taşıyor)
satır-bazlı max/min/ortalamaya dahil edilmişti, bu da özeti anlamsızlaştırıyordu.
Düzeltilmiş sürüm yalnızca `classify_al_columns`'ın (fold-içi, train-only
fit edilen) `ratio_type + frequency_type` alt-kümesini (334'ün ~244'ü)
kullanıyor — sonuçlar bu düzeltilmiş sürümle üretildi.

## E2-Sonrası Doğrulama 1: Eşik-Artefaktı Hipotezi Ölçüldü

Yukarıdaki "genel bir artefakt" iddiası, E1'in **aynı** sabit-hiperparametreli
XGBoost/v1 modeli, aynı 50 dış fold'da, **yeniden eğitilmeden**, yalnızca
olasılık çıktıları bu kez saklanarak (`e1_baseline.py`'den `build_v1_features`/
`FIXED_HYPERPARAMS` aynen içe aktarılıp) iki farklı eşikle (0,359 ve 0,5)
skorlanarak doğrudan ölçüldü (`src/genova/pah/e1_threshold_check.py`,
çıktı: `models/pah/e1_threshold_check_proba.pkl`).

| Eşik | F1 | Specificity |
|---|---|---|
| 0,359 (PDR/E1) | 0,9199±0,0242 | 0,2608±0,1057 |
| 0,5 (E2) | 0,9180±0,0254 | 0,3626±0,1230 |
| **Fark (yalnızca eşik, aynı model)** | **−0,0019** | **+0,1018** |

F1 satırı, E1 raporundaki orijinal sayıyla (0,9199±0,0242) birebir eşleşerek
determinizmi de doğruladı.

**Sonuç: hipotez doğrulandı, hatta ilk tahminden daha güçlü.** Yalnızca
eşik değişikliği (aynı model, aynı fold'lar), E2'nin CatBoost/v1 ile E1
arasında gözlenen paired specificity farkının (+0,1131) **%90'ını**
(0,1018 / 0,1131) tek başına açıklıyor. Yani E2'nin raporladığı geniş
specificity iyileşmesinin ezici çoğunluğu **model değişikliğinden değil,
eşik değişikliğinden** kaynaklanıyor — "kısmen artefakt" ifadesi aslında
bir yetersiz-tahmindi.

Ama önemli bir ayrım var: **F1 üzerinde eşik değişikliğinin yönü ters** —
0,5'e geçmek F1'i hafifçe **düşürüyor** (−0,0019), CatBoost/v1'in E1'e
karşı gözlenen F1 kazancını (+0,0114) değil. Yani F1 ekseninde ölçülen
"CatBoost gerçekten daha iyi" sonucu bu eşik-artefaktından **etkilenmiyor**
— eşik etkisi F1'i CatBoost lehine değil aleyhine çalışıyor, dolayısıyla
model-seçim kararının (CatBoost/v1'in F1'de gerçek bir kazanan olduğu)
dayanağı sağlam kalıyor. Yalnızca specificity'ye dayalı yorumlar (örn.
"model X specificity'de Y kadar daha iyi") bundan böyre eşik farkı
kontrol edilmeden okunmamalı.

## E2-Sonrası Doğrulama 2: OOF Tahmin Korelasyonu (üçüncü model kararı)

CatBoost/v1, LightGBM/v1 ve Elastic-Net/v3'ün out-of-fold (OOF) olasılık
tahminleri, split bankasının `repeat=0` tekrarının 5 dış fold'u (369
satırı sızıntısız ve tam olarak birer kez kapsayan bir partisyon)
üzerinden üretildi. Hiçbir yeni hiperparametre araması yapılmadı — her
fold için E2'nin `nested_cv_evaluate`'inin **o fold'da zaten seçtiği**
`best_params` (`e2_model_comparison.csv`) aynen yeniden kullanıldı, yalnızca
E2 sırasında saklanmayan olasılık çıktısını yakalamak için yeniden fit
edildi (`src/genova/pah/e2_oof_correlation.py`, çıktı:
`reports/tables/e2_oof_predictions_repeat0.csv`).

| Çift | Pearson r | p |
|---|---|---|
| CatBoost/v1 ↔ LightGBM/v1 | 0,7609 | 6,1×10⁻⁷¹ |
| CatBoost/v1 ↔ Elastic-Net/v3 | **0,7525** | 1,5×10⁻⁶⁸ |

**Ölçüme göre:** Elastic-Net/v3, CatBoost/v1 ile **daha düşük** korele
(0,7525 < 0,7609) — talimatın literal kararı kuralına göre "gerçek
çeşitlilik" bu. Ama fark küçük (0,0084) ve her ikisi de CatBoost'la zaten
yüksek korele (>0,75) — yani bu, dramatik bir çeşitlilik farkı değil,
marjinal bir eğilim.

Bu ölçüm tek başına yeterli değil: E2'nin ana karşılaştırmasında Elastic-Net
(hem v3 hem v4_from_v3), E1 baseline'ına karşı paired kazanma oranı yalnızca
%12-16 ile **tutarlı ve büyük farkla geride** kalıyordu (bkz. yukarı),
LightGBM ise baseline'la pratik olarak **eşit** (%38 kazanma oranı, F1
ortalaması E1'inkiyle aynı). E6'nın kendi karar kuralı da bunu destekliyor:
zayıf bir ikinci modelin ensemble'a eklenmesi, yalnızca marjinal bir
korelasyon düşüşü sağlıyorsa kendiliğinden avantaj değildir.

**Net öneri (ölçüme dayalı, varsayıma değil): LightGBM/v1.** Diversite
ölçümü Elastic-Net'i marjinal olarak öne çıkarsa da (0,0084 farkla),
LightGBM/v1'in standalone performansı E1 baseline'ıyla pratik olarak eşit
iken Elastic-Net %12-16 kazanma oranıyla anlamlı ölçüde geride — E3'e
zayıf ama "farklı" bir model taşımanın E6'da somut bir getirisi bu
ölçümle kanıtlanmadı. Elastic-Net, eğer E6'da mimari çeşitlilik özellikle
öncelikliyse ikinci bir alternatif olarak saklı tutulabilir, ama birincil
üçüncü-aday önerisi LightGBM/v1'dir.

## Genel değerlendirme ve E3 önerisi (güncellendi)

**En iyi 3-5 kombinasyon (F1'e göre, E1'e karşı paired kazanma oranıyla
birlikte):**

1. **CatBoost / v1** — F1=0,9313±0,0214, kazanma oranı %78, en basit
   (ham veri, ek mühendislik yok). F1 kazancı eşik-artefaktından
   etkilenmiyor (yukarıdaki Doğrulama 1'e bakın).
2. **CatBoost / v1+ZORT_\*** — F1=0,9303±0,0231, kazanma oranı %82 (en
   yüksek kazanma oranı ama `catboost/v1`'e karşı kendisi kaybediyor —
   bkz. yukarı, resmi havuza önerilmiyor).
3. **CatBoost / v2_raw_nan** — F1=0,9272±0,0219, kazanma oranı %66.
4. **CatBoost / v4_from_v2** — F1=0,9266±0,0230, kazanma oranı %62 (25
   özellik, en yorumlanabilir/paketlenebilir aday).
5. **LightGBM / v1** — F1=0,9199±0,0257, üçüncü E3 adayı (bkz. Doğrulama 2,
   ölçüme dayalı net öneri).

**Öneri:** E3'e (kalibrasyon) **CatBoost/v1**'i birincil aday olarak
taşı. `v4_from_v2` (25-özellik, CatBoost) ikinci aday olarak — F1'de
marjinal geride ama yorumlanabilirlik/paketleme (Aşama D'nin özellik
seçimi hattıyla tutarlılık, F4/F6'da avantaj) için değerli bir takas.

*(NOT: Bu öneri E2 zamanındaki ara-durumu yansıtıyor. E3-E6'nın nested
kanıtı bu kararı güncelledi — nihai final aday CatBoost/v4_from_v2,
solo, aşağıya bkz. — bu bölümün geri kalanı yalnızca tarihsel/E2-anındaki
gerekçe olarak korunuyor, dosyanın başındaki "Aşama E — Nihai Özet"
tablosu ve E6 bölümündeki "Nihai öneri" güncel/bağlayıcı olandır.)*

Üçüncü aday: **LightGBM/v1** (ölçülmüş OOF korelasyon + standalone
performans dengesine göre, bkz. Doğrulama 2). Elastic-Net (hem v3 hem
v4_from_v3) F1'de tutarlı ve büyük farkla (kazanma oranı %12-16) geride
kaldığı ve OOF-diversite avantajı yalnızca marjinal olduğu için E3'e
taşınması önerilmiyor.

## Aşama E3: Nested Kalibrasyon

Üç aday (CatBoost/v1, CatBoost/v4_from_v2, LightGBM/v1), her biri
`data/splits/pah/`'ın aynı 50 dış fold'unda, E2'nin o fold için **zaten
seçtiği** `best_params` ile (yeniden arama yok) kalibre edildi.

**Sızıntı-kritik prosedür (uygulandığı gibi):** her dış fold için,
dış-train kendi 4 iç fold'unda **çapraz-fit** edildi (taban model her iç
fold'da iç-train'de fit edilip iç-val'de olasılık üretti; 4 iç-val kümesi
dış-train'in **tamamını** sızıntısız kaplıyor — `test_e3_calibration_run_
pah.py` bunu gerçek split bankası üzerinde doğruluyor: çapraz-fit çıktı
kümesi tam olarak dış-train `Variant_ID` kümesine eşit, dış-test kümesiyle
kesişimi boş). Kalibratör bu çapraz-fit olasılıklarla fit edildi; taban
model ayrıca **tüm** dış-train'de yeniden fit edilip dış-test'te ham
olasılık üretti, kalibratör bu ham olasılığa yalnızca **transform**
uygulandı — hiçbir zaman dış-test'in kendi verisiyle fit edilmedi.

`betacal` paketi bu oturumda kuruldu (1.1.0), üç yöntem de (Platt/Beta/
Isotonic) çalıştırıldı.

### Sonuçlar (50 dış fold, ortalama±std)

| Model | Yöntem | Brier | Log-loss | F1 | Ort. benzersiz olasılık sayısı |
|---|---|---|---|---|---|
| CatBoost/v1 | Platt | 0,1022±0,0218 | 0,3497±0,0591 | 0,9306±0,0233 | 72,3 |
| CatBoost/v1 | **Beta** | **0,1023±0,0226** | 0,3518±0,0644 | **0,9322±0,0234** | 72,3 |
| CatBoost/v1 | Isotonic | 0,1035±0,0232 | 0,3968±0,1323 | 0,9283±0,0220 | **10,9** |
| CatBoost/v1 | Ham (kalibrasyonsuz) | 0,1054±0,0195 | 0,3574±0,0547 | 0,9313±0,0214 | 72,3 |
| CatBoost/v4_from_v2 | **Beta** | **0,0987±0,0245** | **0,3330±0,0693** | **0,9271±0,0232** | 73,4 |
| CatBoost/v4_from_v2 | Platt | 0,0988±0,0242 | 0,3342±0,0665 | 0,9267±0,0229 | 73,4 |
| CatBoost/v4_from_v2 | Ham | 0,0995±0,0223 | 0,3345±0,0649 | 0,9266±0,0230 | 73,4 |
| CatBoost/v4_from_v2 | Isotonic | 0,1021±0,0247 | 0,3907±0,1364 | 0,9250±0,0228 | **11,3** |
| LightGBM/v1 | **Beta** | **0,1090±0,0282** | **0,3677±0,0795** | **0,9213±0,0280** | 73,5 |
| LightGBM/v1 | Isotonic | 0,1110±0,0293 | 0,4014±0,1237 | 0,9204±0,0271 | **10,4** |
| LightGBM/v1 | Platt | 0,1118±0,0292 | 0,3779±0,0771 | 0,9215±0,0260 | 65,5 |
| LightGBM/v1 | Ham | 0,1184±0,0366 | 0,6418±0,3140 | 0,9199±0,0257 | 68,7 |

**Ham olasılıkların log-loss'u her üç modelde de belirgin kötü**
(0,357-0,642) — ağaç modellerinin (özellikle LightGBM'in, 0,642) ham
olasılıkları aşırı-güvenli/kötü kalibre; kalibrasyon bunu tutarlı biçimde
düzeltiyor (log-loss'ta %10-45 iyileşme). Brier'de iyileşme daha mütevazı
ama yine de her üç modelde de kalibre yöntemler (Isotonic hariç) ham'ı
geçiyor.

### Isotonic açıkça aşırı-uyum gösteriyor — beklenen bulgu doğrulandı

Üç modelin üçünde de Isotonic:
- **Ortalama benzersiz kalibre olasılık sayısı yalnızca ~10-11**
  (Platt/Beta'nın ~65-73'üne karşı) — n≈295'lik dış-train'de PAV
  algoritmasının ürettiği basamak fonksiyonu aşırı kaba, çoğu farklı ham
  skor aynı birkaç düzeye sıkıştırılıyor.
- **En kötü log-loss** (0,40 civarı, Platt/Beta'nın ~0,33-0,38'ine karşı)
  ve std'si de en yüksek (0,12-0,13 vs ~0,06-0,08) — bazı fold'larda
  ciddi kötüleşme.
- Brier ve F1'de de üç modelin üçünde de **en düşük/en kötü** sırada.

Beta-vs-Isotonic paired Brier karşılaştırması (aynı fold'lar): Isotonic
kazanma oranı yalnızca %30-56 (tutarlı bir üstünlük yok, CatBoost/v4_
from_v2'de belirgin kaybediyor). **Isotonic hiçbir model için
önerilmiyor.**

### Platt vs Beta — ikisi de sağlam, Beta hafif öne çıkıyor

Paired Brier karşılaştırması (aynı 50 fold): CatBoost/v1'de Beta kazanma
oranı %52 (pratikte berabere), CatBoost/v4_from_v2'de %56 (hafif Beta
lehine), LightGBM/v1'de %72 (**Beta belirgin daha iyi**, ort. fark
−0,0028). Beta hiçbir modelde Platt'tan anlamlı ölçüde kötü değil, en
azından bir modelde (LightGBM) net üstün.

**Model başına önerilen kalibratör:**

| Model | Önerilen | Gerekçe |
|---|---|---|
| CatBoost/v1 | **Beta** (Platt istatistiksel olarak eşdeğer) | En iyi F1 (0,9322), Brier/logloss Platt'a çok yakın |
| CatBoost/v4_from_v2 | **Beta** | En iyi Brier+logloss+F1, üçünde de birinci |
| LightGBM/v1 | **Beta** | En iyi Brier+logloss+F1, Isotonic'e karşı %72 kazanma oranıyla en net üstünlük |

F1@0,5 kabaca kontrolü: kalibrasyon hiçbir modelde F1'i **bozmuyor**
(CatBoost/v1'de Beta ham'a göre +0,0009, CatBoost/v4_from_v2'de +0,0005,
LightGBM/v1'de +0,0014 — hepsi std'nin çok altında, yani "ne bozuyor ne
belirgin iyileştiriyor", tam olarak E-F dokümanının öngördüğü sonuç).
Kesin eşik/F1 kararı E5'in işi; E3'ün burada ürettiği kalibre OOF
olasılıklar (`e3_calibrated_oof_predictions.csv`) E4'ün önsel-düzeltmesi
ve E5'in eşik taramasına doğrudan girdi olacak.

## Testler

`pytest tests/ -q` → **65/65 geçti** (58 önceki + 6 `test_calibration_
pah.py` + 1 `test_e3_calibration_run_pah.py` — ikincisi çapraz-fit'in
gerçek split bankasında dış-train'i tam kapladığını ve dış-test'e hiç
dokunmadığını doğrulayan sızıntı testi). E2-sonrası doğrulama scriptleri
(`e1_threshold_check.py`, `e2_oof_correlation.py`, `e3_calibration_run.py`
orkestrasyon kısmı) yalnızca zaten test edilmiş yapı taşlarını
(fold_versions, models, e1_baseline, calibration) yeniden kullanan
orkestrasyon kodu olduğu için ayrıca test edilmedi (aynı proje
konvansiyonu: `e1_baseline.py`/`e2b_run.py` de test edilmiyor).

## Kod / dosyalar

- `src/genova/pah/fold_versions.py` — v1/v2/v2_raw_nan/v4_from_v2/v3/
  v4_from_v3 için fold-güvenli inşacılar (yeni).
- `src/genova/pah/models.py` — nested-CV çatısı, model sarmalayıcıları,
  hiperparametre ızgaraları (yeni).
- `src/genova/pah/e2_candidate_features.py` — `ZORT_*`/`AL_`-özet
  fold-güvenli ek özellikler (yeni).
- `src/genova/pah/e2b_run.py`, `e2b_raw_nan_run.py` — 2b çalıştırıcıları
  (yeni).
- `src/genova/pah/e1_threshold_check.py` — E2-sonrası Doğrulama 1 (yeni).
- `src/genova/pah/e2_oof_correlation.py` — E2-sonrası Doğrulama 2 (yeni).
- `src/genova/pah/calibration.py` — Platt/Beta/Isotonic kalibratör
  sarmalayıcıları (yeni).
- `src/genova/pah/e3_calibration_run.py` — E3 çapraz-fit kalibrasyon
  çatısı (yeni).
- `reports/tables/e2_model_comparison.csv` — tam 800 satırlık ham sonuç.
- `reports/tables/e2_oof_predictions_repeat0.csv` — 369×3 OOF olasılık
  matrisi (Doğrulama 2).
- `reports/tables/e3_calibration_metrics.csv` — 3 model × 4 yöntem × 50
  fold = 600 satırlık kalibrasyon metrikleri.
- `reports/tables/e3_calibrated_oof_predictions.csv` — 3 model × 4 yöntem
  × 50 fold'un tüm dış-test tahminleri (E4/E5 girdisi).
- `models/pah/e1_threshold_check_proba.pkl` — E1'in 50 fold'luk ham
  olasılık çıktıları (Doğrulama 1).
- `tests/test_fold_versions_pah.py`, `tests/test_e2_candidate_features_
  pah.py`, `tests/test_calibration_pah.py`, `tests/test_e3_calibration_
  run_pah.py` — yeni testler.

`data/splits/pah/`, `data/processed/pah/`, `experiments/`,
`artifacts/preprocessors/` içine hiçbir yazma yapılmadı;
`v4_final_feature_pool.json` değiştirilmedi. E3'te hiçbir hiperparametre
yeniden aranmadı (E2'nin `best_params`'ı aynen kullanıldı). Hiçbir model
dondurulmadı (E2/E3 yalnızca karşılaştırma).

## Aşama E4: Final Önsel Düzeltmesi (SLD¹ — bkz. yukarıdaki terminoloji notu)

Kapalı-form düzeltme, E3'ün Beta-kalibre dış-test olasılıklarına
(`e3_calibrated_oof_predictions.csv`, `method="beta"`) doğrudan uygulandı
— **model yeniden eğitilmedi**, yalnızca olasılık ölçeği dönüştürüldü:

```
w1 = 0,286 / 0,833 = 0,343337
w0 = 0,714 / 0,167 = 4,275449
p_yeni = w1·p / (w1·p + w0·(1−p))
```

| Aday | Ort. proba (Beta, SLD-öncesi) | Ort. proba (SLD-sonrası) |
|---|---|---|
| CatBoost/v1 | 0,8244 | **0,3794** |
| CatBoost/v4_from_v2 | 0,8240 | **0,4028** |
| LightGBM/v1 | 0,8314 | **0,3708** |

Beklenen büyük kayma doğrulandı: eğitim önselinin (%83,3) final önseline
(%28,6) göre çok daha yüksek olması, ortalama olasılığı ~0,82-0,83'ten
~0,37-0,40'a çekiyor — model sıralaması (AUC) değişmedi, yalnızca olasılık
ölçeği ve dolayısıyla anlamlı eşiğin konumu değişti. Çıktı:
`reports/tables/e4_prior_corrected_probabilities.csv` (`beta_sld` metod
etiketiyle, 11.070 satır = 3 aday × 10 tekrar × 369 satır).

### Prior düzeltmesinin gerçek katkısı (sınır analizi — dış denetim bulgusu)

**Kapalı-form Bayes önsel düzeltmesi strictly monotonic bir dönüşümdür;
eşik zaten ampirik olarak optimize edildiği için hard-label F1 açısından
yeni bir karar sınırı ailesi yaratmaz.** SLD uzayındaki final eşiği
(`0,35`), Beta uzayındaki **≈0,8702** eşiğine birebir denktir:

```
sld(p) = w1·p / (w1·p + w0·(1−p)) = 0,35
  →  p* = 0,35·w0 / (0,65·w1 + 0,35·w0) = 0,87021847
```

Elle doğrulandı: `sld_correct(0,87021847) = 0,35`, ve 200.000 rastgele
olasılık noktasında `sld(p) ≥ 0,35` ile `p ≥ 0,87021847` kuralları
**sıfır** karar uyuşmazlığı üretiyor (bkz.
`tests/test_e4_prior_correction_pah.py::test_sld_threshold_equivalence`).

**Adımın korunma gerekçesi performans değil**, üç somut sebep:

1. **Eşik arama ızgarasını anlamlı bölgeye yoğunlaştırır.** E5'in araması
   0,05-0,95 aralığında 0,01 adımlarla yapılıyor; SLD sonrası optimum
   ~0,27-0,35 civarına düşüyor, yani ızgaranın orta-yoğun bölgesine.
   Düzeltme olmasaydı optimum ~0,87 civarında, ızgaranın seyrek/uç
   bölgesinde aranacaktı.
2. **`predicted_probability` çıktısını hedef popülasyonda yorumlanabilir
   bir posterior yapar.** `predict.py`'nin ürettiği olasılık, final test
   dağılımı (%28,6 patojenik) altındaki gerçek posterior olarak
   okunabilir — çok-panel birleştirmede (bkz. `F5_ACIK_SORULAR_PANEL_
   BIRLESTIRME.md` madde 3, ortak eşik/kalibrasyon sorusu) bu doğrudan
   kullanılabilir bir özelliktir.
3. **Prevalence shift'in açıkça modellendiğini gösterir** — şartnamenin
   eğitim/test dağılım farkının farkında olunduğunun ve hesaba
   katıldığının belgesi.

> ⚠️ **Jüri sunumu için sınır ifadesi:** Bu adım hakkında **"prior
> düzeltmesi F1'i artırdı" iddiası YAPILMAMALIDIR.** Doğru ifade:
> "prevalence shift'i açıkça modelleyerek eşiği yorumlanabilir ve
> transfer edilebilir bir ölçeğe taşıdık; hard-label performansı
> eşdeğer bir eşikle de elde edilebilirdi."

## Aşama E5: Nested Eşik Seçimi

### Prosedür (uygulandığı gibi)

Her dış fold için: (1) E3'ün `cross_fit_probabilities`'i aynı `best_
params` ile yeniden çalıştırılıp dış-train'in tamamı için sızıntısız ham
iç-çapraz-fit olasılıkları elde edildi; (2) bu olasılıklar o fold için
yeniden fit edilen (E3'ünkiyle matematiksel olarak birebir aynı,
deterministik) Beta kalibratörüyle dönüştürülüp E4'ün SLD formülüyle
düzeltildi; (3) önsel-ağırlıklı (`w1`/`w0`) F1'i (`f1_binary_positive_
weighted`, bu görev için `genova/metrics.py`'ye eklendi) 0,05-0,95
aralığında 0,01 adımlarla maksimize eden eşik seçildi (plato varsa
ortası); (4) bu eşik yalnızca **bir kez**, o fold'un E4-çıktısı (dış-
test'in SLD-düzeltilmiş olasılıkları) üzerinde uygulandı.

**Sızıntı testi** (`test_e5_threshold_selection_pah.py::test_process_
fold_threshold_choice_is_unaffected_by_dis_test_contents`): aynı sabit
iç-örneklem için dış-test içeriğini tamamen değiştirdiğimizde
`chosen_threshold`'un **birebir aynı** kaldığını, ama dış-test'e bağlı
metriklerin (`f1_chosen`) farklı çıktığını doğrudan doğruluyor —
fonksiyonun dış-test'i yalnızca eşik SEÇİLDİKTEN SONRA okuduğunu
kanıtlıyor.

### Seçilen eşikler (E1'in 0,359'una göre)

| Aday | Seçilen eşik (ort±std) | Aralık |
|---|---|---|
| CatBoost/v1 | **0,2847±0,0765** | 0,13-0,55 |
| CatBoost/v4_from_v2 | **0,3315±0,0731** | 0,13-0,50 |
| LightGBM/v1 | **0,2737±0,0525** | 0,14-0,37 |

**Önemli karşılaştırılabilirlik notu:** bu sayılar E1'in 0,359'uyla
doğrudan sayısal olarak kıyaslanamaz — E1'in eşiği ham (kalibrasyonsuz,
SLD-düzeltmesiz) XGBoost/v1 olasılıkları üzerinde tanımlıydı, buradaki
eşikler ise Beta-kalibre + SLD-düzeltilmiş bir olasılık ölçeğinde. Üçü de
E1'in 0,359'undan **düşük** çıkıyor, ama bunun nedeni "daha agresif bir
model" değil — SLD düzeltmesi olasılıkları ortalama ~0,82'den ~0,37-0,40'a
çektiği için, sabit 0,5 gibi bir eşik SLD-sonrası ölçekte aşırı
tutucu/az-duyarlı kalıyor (nerdeyse her şeyi benign tahmin ediyor);
optimum eşik bu düşük ortalamayı telafi etmek için de düşük çıkıyor.

### Karar kuralı (madde 6): üçünde de **adaptif eşik** kazanıyor

Ağırlıklı-F1'de (dış-test, hedef-önsel ağırlıklarıyla), adaptif eşik ile
sabit SLD-sonrası 0,5 eşiği arasındaki fark, fold-bazlı bootstrap (B=5000,
%95 GA) ile test edildi:

| Aday | Ağırlıklı-F1 (adaptif) | Ağırlıklı-F1 (sabit 0,5) | Ort. fark | %95 GA | Karar |
|---|---|---|---|---|---|
| CatBoost/v1 | 0,5802±0,0870 | 0,3903±0,1540 | +0,1898 | [0,1383; 0,2435] | **Adaptif** |
| CatBoost/v4_from_v2 | 0,6187±0,0855 | 0,4931±0,0805 | +0,1256 | [0,0981; 0,1532] | **Adaptif** |
| LightGBM/v1 | 0,5818±0,1057 | 0,2948±0,1767 | +0,2870 | [0,2262; 0,3511] | **Adaptif** |

Üçünde de ortalama iyileşme, GA genişliğinin yarısının **kat kat üzerinde**
(en düşük oran bile CatBoost/v4_from_v2'de ~4,5×) — bu, madde 6'nın
"basit eşiği tercih et" eşiğinin tam tersi, **çok net ve sağlam bir
sonuç**: sabit 0,5, SLD-sonrası olasılık ölçeğinde aşırı tutucu davranıp
neredeyse her şeyi benign tahmin ediyor (specificity ~0,93 ama F1 çöküyor
— bkz. aşağıdaki tablo), adaptif eşik bunu düzeltiyor. **Üç aday için de
adaptif (dış-fold-bazlı seçilen) eşik öneriliyor, sabit bir eşik değil.**

### Duyarlılık eğrisi (final önsel %20-%38 arası)

Sabit (28,6%'da seçilmiş) eşik + SLD-düzeltmesi korunarak, yalnızca
DEĞERLENDİRME ağırlıklandırması (gerçek önsel farklı çıkarsa) değiştirildi:

| Aday | %20 | %25 | %28,6 | %33 | %38 | Aralık (maks−min) |
|---|---|---|---|---|---|---|
| CatBoost/v1 | 0,4914 | 0,5466 | 0,5802 | 0,6158 | 0,6505 | 0,1591 |
| CatBoost/v4_from_v2 | 0,5463 | 0,5919 | 0,6187 | 0,6466 | 0,6731 | 0,1268 |
| LightGBM/v1 | 0,4901 | 0,5470 | 0,5818 | 0,6189 | 0,6552 | 0,1652 |

**Sonuç: geniş, düz bir plato — dar/kırılgan bir optimum değil.** Üçünde
de eğri **düzgün ve monotonik** (ani bir uçurum/kırılma yok); ±10 puanlık
sapmada ağırlıklı-F1 yalnızca ~0,13-0,17 puan (ölçeğin ~%25'i) değişiyor,
performans **çökmüyor**. Final önselinin tam olarak %28,6 çıkmaması
(örneğin %20 ya da %38 çıkması) sistemi felç etmiyor, kademeli/öngörülebilir
bir performans değişikliğine yol açıyor — bu, sabit bir eşiğin ±5-10
puanlık bir prevalans belirsizliğine karşı **sağlam** olduğu anlamına
geliyor.

### Üç adayın nihai karşılaştırması (E1'e göre)

| Aday | F1 (adaptif eşik, ham) | MCC | Specificity | Ağırlıklı-F1 (asıl karşılaştırma) |
|---|---|---|---|---|
| E1 (referans) | 0,9199±0,0242 | — | 0,2608±0,1057 | (hesaplanmadı, SLD/E5 öncesi) |
| CatBoost/v1 | 0,8263±0,0751 | 0,3330±0,1197 | 0,6248±0,1705 | **0,5802±0,0870** |
| CatBoost/v4_from_v2 | 0,7988±0,0739 | 0,3659±0,0849 | 0,7533±0,1535 | **0,6187±0,0855** |
| LightGBM/v1 | 0,8426±0,0789 | 0,3531±0,1389 | 0,6027±0,1580 | **0,5818±0,1057** |

**Yorum-kritik nüans:** "F1 (adaptif eşik, ham)" sütunu, E1/E2/E3'le aynı
tanımı (dış-test'in kendi — hâlâ eğitim-dağılımlı, %83 patojenik —
satırları üzerinde, kalibrasyonsuz) kullanıyor gibi görünse de, artık
**farklı bir amaç için ayarlanmış bir eşikle** ölçülüyor: eşik final önsele
(%28,6) göre optimize edildiği için, hâlâ eski (yüksek patojenik oranlı)
kompozisyonlu dış-test fold'larında bilerek daha fazla "benign" tahmin
üretiyor — bu yüzden E1'in 0,9199'una kıyasla düşük çıkması **beklenen ve
istenen** bir sonuç, "model kötüleşti" anlamına gelmiyor. **Asıl anlamlı
karşılaştırma Ağırlıklı-F1 sütunu** — bu, dış-test satırlarını final
önselini yansıtacak şekilde yeniden tartarak, sistemin **gerçek dağıtım
koşullarında** nasıl performans göstereceğinin en iyi tahminidir. Bu
ölçekte üç aday da CatBoost/v4_from_v2 > LightGBM/v1 ≈ CatBoost/v1 sırasını
gösteriyor — ama farklar (0,58-0,62) std'lerin (~0,09-0,11) içinde,
kesin bir sıralama iddiası için tek başına yeterli değil (E6'nın ensemble
karşılaştırmasında yeniden değerlendirilebilir).

Specificity'nin adaptif eşikte E1'e göre çok daha yüksek çıkması
(0,60-0,75 vs 0,26) da aynı SLD-mantığının beklenen bir sonucu: final
popülasyonun ağırlıklı benign çoğunluğu göz önüne alındığında, model artık
benign'i daha iyi ayırt etmesi gerektiğini "biliyor."

## Testler (E4+E5)

`pytest tests/ -q` → **76/76 geçti**: E3 sonrası eklenenler — 4 yeni
(`test_e4_prior_correction_pah.py`, SLD formülünün cebirsel özellikleri:
p=0,5'te w1/(w1+w0)'a eşitlik, tekdüzelik, sınır değerleri) + 3 yeni
(`genova/metrics.py`'ye eklenen `f1_binary_positive_weighted` için) + 4
yeni (`test_e5_threshold_selection_pah.py`, eşik seçiminin dış-test'e
bakmadan yapıldığını doğrulayan sızıntı testi dahil).

## Kod / dosyalar (E4+E5 eklentisi)

- `src/genova/metrics.py` — `f1_binary_positive_weighted` eklendi (E5'in
  önsel-ağırlıklı eşik araması için).
- `src/genova/pah/e4_prior_correction.py` — SLD düzeltmesi (yeni).
- `src/genova/pah/e5_threshold_selection.py` — nested eşik seçimi çatısı
  (yeni).
- `reports/tables/e4_prior_corrected_probabilities.csv` — 3 aday × 10
  tekrar × 369 satır SLD-düzeltilmiş olasılıklar.
- `reports/tables/e5_threshold_selection.csv` — 150 satır (3 aday × 50
  fold): seçilen eşik, F1/MCC/specificity (adaptif+sabit), duyarlılık
  eğrisi sütunları.
- `tests/test_e4_prior_correction_pah.py`, `tests/test_e5_threshold_
  selection_pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`'a, `experiments/`'e,
`v4_final_feature_pool.json`'a hiçbir yazma yapılmadı. E4/E5'te hiçbir
hiperparametre/kalibratör yeniden aranmadı (E2/E3'ün seçimleri aynen
kullanıldı, yalnızca ucuz/deterministik olarak yeniden hesaplandı).

## Aşama E6: Ensemble

### Adım 1 — OOF korelasyon matrisi (ensemble'dan önce ölçüldü)

Üç adayın kalibre+SLD-düzeltilmiş dış-test olasılıkları, `repeat=0`'ın 5
dış fold'unda (369 satırlık sızıntısız tam partisyon,
`e4_prior_corrected_probabilities.csv`'den doğrudan) ikili Pearson
korelasyonuna tabi tutuldu:

| | CatBoost/v1 | CatBoost/v4_from_v2 | LightGBM/v1 |
|---|---|---|---|
| **CatBoost/v1** | 1,000 | 0,820 | 0,760 |
| **CatBoost/v4_from_v2** | 0,820 | 1,000 | 0,764 |
| **LightGBM/v1** | 0,760 | 0,764 | 1,000 |

**Karar noktası — beklenenden ılımlı ama yönü doğru:** `CatBoost/v1`↔
`CatBoost/v4_from_v2` korelasyonu (0,820), E-F dokümanının örnek eşiği
olan >0,90'ı **geçmiyor**, ama yine de üç çiftin en yükseği ve diğer
ikisinden (0,760, 0,764) belirgin farklı. Daha çarpıcı olan: LightGBM,
**her iki CatBoost varyantına da neredeyse eşit** korele (0,760 vs
0,764, fark ihmal edilebilir) — yani "hangi ikili en düşük korele"
sorusunun cevabı pratikte kayıtsız, gerçek ayrım CatBoost/v1 ile
CatBoost/v4_from_v2'nin birbirine en yakın çift olması. Üçünü birden
zorlamak yerine, E-F dokümanının önerdiği ikili seçildi: **CatBoost/
v4_from_v2 + LightGBM/v1** — hem düşük-korele çiftlerden biri hem de
daha güçlü solo performansı (E5'te ağırlıklı-F1=0,6187, üç adayın en
iyisi) taşıyan üyeyi içeriyor. `CatBoost/v1` üçüncü üye olarak
denenmedi: solo performansı zaten en düşüktü (0,5802) ve `CatBoost/
v4_from_v2`'yle en yüksek korelasyona sahip, yani marjinal çeşitlilik
katkısı sınırlı olurdu.

### Adım 2 — Ensemble inşası: 3 blend yöntemi karşılaştırması

Her dış fold için, iki üyenin (`CatBoost/v4_from_v2`, `LightGBM/v1`) iç
çapraz-fit örneklemi ayrı ayrı yeniden hesaplandı (model yeniden
aranmadı, E2'nin `best_params`'ı kullanıldı), Beta ile kalibre edildi,
ve üç blend tarifi **her biri kendi nested E4(SLD)+E5(eşik) prosedüründen
geçirilerek** karşılaştırıldı — basit ortalama ve rank-average'ın serbest
parametresi yok, kısıtlı ağırlıklı blend'in ağırlığı (`α ∈ [0,1]`, adım
0,1) yalnızca iç örneklemde ağırlıklı-F1'i maksimize edecek şekilde
arandı. Dış-test skorlaması, E3'ün zaten kaydettiği Beta-kalibre dış-test
olasılıkları (`e3_calibrated_oof_predictions.csv`) okunarak yapıldı —
model dış-test için yeniden fit edilmedi.

| Blend yöntemi | Ağırlıklı-F1 (ort±std) | F1 (ham) | Seçilen eşik (ort±std) |
|---|---|---|---|
| **Ağırlıklı (kısıtlı)** | **0,6173±0,0945** | 0,8026±0,0710 | 0,3242±0,0605 |
| Rank-average | 0,6106±0,1042 | 0,7711±0,0427 | 0,0560±0,0101 |
| Basit ortalama | 0,6092±0,0983 | 0,8307±0,0788 | 0,2838±0,0751 |

**Kazanan: Ağırlıklı (kısıtlı) blend** — ama farkı diğer ikisinden çok
küçük (0,006-0,008), üçü de std'nin içinde pratik olarak eşdeğer. Daha
çarpıcı bir bulgu: ağırlıklı blend'in kendi α seçiminde, **50 fold'un
18'inde (%36) α=1,0 seçildi** (yani LightGBM'e sıfır ağırlık verildi, iç
örneklem LightGBM'i eklemenin performansı **düşürdüğünü** gösterdi) ve
**hiçbir fold'da α=0,0 seçilmedi** (LightGBM tek başına hiçbir zaman
tercih edilmedi). Ortalama α=0,772 — iç arama, ensemble'ı zaten büyük
ölçüde CatBoost/v4_from_v2'ye çeken bir ağırlık dağılımına yakınsıyor.
Bu, Adım 1'in "iki üye o kadar da farklı davranmıyor" bulgusuyla tutarlı
bir ön sinyal.

### Adım 3 — Karar: Ensemble solo modele karşı paired karşılaştırma

En iyi solo model (`CatBoost/v4_from_v2`, E5'ten ağırlıklı-F1=0,6187),
üç blend yöntemine karşı **aynı 50 dış fold'da paired** karşılaştırıldı,
fold-bazlı bootstrap (B=5000, %95 GA) ile:

| Blend yöntemi | Ort. fark (ensemble−solo) | Kazanma oranı | %95 GA | Karar |
|---|---|---|---|---|
| Basit ortalama | −0,0096 | %38 | [−0,0287; 0,0098] | **Solo model** |
| Rank-average | −0,0081 | %36 | [−0,0304; 0,0154] | **Solo model** |
| **Ağırlıklı (kazanan)** | **−0,0014** | **%30** | **[−0,0145; 0,0118]** | **Solo model** |

**Sonuç: üçünde de solo model tercih ediliyor — net ve tek yönlü.**
En iyi blend yöntemi (ağırlıklı) bile solo modele karşı **negatif**
ortalama fark veriyor (kazanma oranı yalnızca %30, yani fold'ların
çoğunda ensemble solo modelden **kötü**); %95 güven aralığı sıfırı
kapsıyor, yani istatistiksel olarak anlamlı bir fark yok. Karar kuralı
("ensemble, solo modelden bootstrap GA'nın yarısından daha az bir
iyileşme sağlıyorsa daha basit olanı tercih et") burada trivial olarak
tetikleniyor — iyileşme negatif olduğu için yarı-GA eşiğinin çok altında.

**Neden beklenen bir sonuç:** Adım 1'in OOF korelasyonu (0,764) ve E2'nin
kendi bulgusu (LightGBM baseline'a karşı pratikte düz, F1'de gerçek bir
kazanan değil) bir araya geldiğinde, LightGBM'i CatBoost/v4_from_v2'ye
eklemek gerçek bir çeşitlilik kazancından çok gürültü ekliyor — kısıtlı
ağırlıklı blend'in kendi iç aramasının %36 oranında LightGBM'i tamamen
devre dışı bırakması (α=1,0) bunu zaten kendi içinde doğruluyor.

### Nihai öneri: **Solo CatBoost/v4_from_v2, ensemble değil**

Karmaşıklık kendiliğinden ödül değil — E-F dokümanının kendi ilkesi bu
sonuçla doğrulandı. Aşama F'ye (adversarial validation, sağlamlık,
açıklanabilirlik, paketleme) **CatBoost/v4_from_v2** (E3'ün Beta
kalibratörü + E4'ün SLD düzeltmesi + E5'in nested-seçilen eşiğiyle,
solo) tek final aday olarak taşınması öneriliyor.

## Testler (E6)

`pytest tests/ -q` → **81/81 geçti**: E5 sonrası eklenenler — 5 yeni
(`test_e6_ensemble_pah.py`), bunlardan biri ensemble ağırlığının
(`alpha`) ve eşiğinin dış-teste bakmadan seçildiğini doğrulayan sızıntı
testi (aynı sabit iç örneklem için dış-testi tamamen değiştirip
`chosen_threshold`/`alpha`'nın birebir aynı kaldığını, yalnızca dış-teste
bağlı metriklerin değiştiğini doğruluyor — E5'in testiyle aynı desen).

## Kod / dosyalar (E6 eklentisi)

- `src/genova/pah/e6_ensemble.py` — OOF korelasyon matrisi + 3 blend
  yöntemi + nested E4/E5 entegrasyonu (yeni).
- `reports/tables/e6_oof_correlation_matrix.csv` — Adım 1'in korelasyon
  matrisi.
- `reports/tables/e6_ensemble_metrics.csv` — 150 satır (3 blend yöntemi
  × 50 fold).
- `tests/test_e6_ensemble_pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`'a, `experiments/`'e,
`v4_final_feature_pool.json`'a hiçbir yazma yapılmadı. E6'da hiçbir
model hiperparametresi/kalibratörü yeniden aranmadı; yalnızca ensemble
ağırlığı (`α`) ve eşik, E5'le aynı nested disiplinle arandı.

---

## Aşama E2-EK: Model Portföyü ve Ağırlıklandırma Genişletmesi

E1-EK'in `weighting.py` (A/B/C) altyapısı hazır olduğundan, üç literatür
kaynağının önerdiği **Random Forest** (çekirdek aday) ve **KNN**
(`v4_from_v2` üzerinde ikincil baseline) E2'nin karşılaştırma tablosuna
eklendi; ayrıca A/B/C'nin RF ve CatBoost/v4_from_v2 üzerindeki etkisi
ölçüldü. `reports/tables/e2_model_comparison.csv`'ye **450 yeni satır**
eklendi (1250 = 800 önceki + 450 yeni); mevcut satırlar silinmedi,
yalnızca geriye-dönük uyumluluk için `weighting_variant` kolonu eklendi
(hepsi `B_fixed_minority_2x` olarak etiketlendi — zaten o stratejiyle
üretilmişlerdi). **Bu görev hiçbir final model kararı vermiyor** — E2'nin
mevcut CatBoost/v4_from_v2 (E3-E6 boyunca taşınan aday) hâlâ değişmedi,
burada yalnızca portföy genişletiliyor.

### Tier 1 — Random Forest ve KNN

RF, XGBoost/CatBoost/LightGBM'in aksine ham NaN/native kategorik kabul
etmiyor — `src/genova/pah/dense_numeric_features.py` (yeni, fold-güvenli:
kategorik kodlama + kalan NaN'ı yalnızca train-fold medyanıyla doldurma)
her versiyonu RF/KNN'e uygun tam-sayısal bir matrise çeviriyor.

| Model | Versiyon | F1 | MCC | Specificity |
|---|---|---|---|---|
| RF | v1 | 0,9083±0,0277 | **0,0381** | **0,0197** |
| RF | v2 | 0,9206±0,0294 | 0,3586 | 0,2203 |
| RF | v2_raw_nan | 0,9090±0,0278 | **0,0614** | **0,0272** |
| RF | **v4_from_v2** | **0,9233±0,0266** | **0,3798** | 0,2234 |
| KNN | v4_from_v2 | 0,9074±0,0302 | 0,0746 | 0,0495 |

**Çarpıcı bulgu — RF, `AL_` doldurma stratejisine aşırı duyarlı.** F1
farkı küçük görünse de (~0,01-0,02), **MCC ve specificity'de uçurum var**:
`v1`/`v2_raw_nan` (ikisi de ham NaN'ı `dense_numeric_features.py`'nin
**medyan**-doldurmasından geçiriyor) MCC≈0,04-0,06 ile neredeyse rastgele
— model benign'i pratikte hiç öğrenmiyor. `v2`/`v4_from_v2` (ikisi de
`fold_versions.py`'nin kendi **sıfır**-doldurmasını miras alıyor, adaptör
yalnızca kalan `EK_` NaN'ını dokunuyor) MCC≈0,36-0,38 ile çok daha iyi.
**Yorum:** zero-inflated, ağır sağa-çarpık `AL_` kolonlarında medyanla
doldurma, RF'in ayrım gücünü ciddi ölçüde bozuyor — muhtemelen "gözlenmedi"
sinyalini (doğal olarak 0'a yakın duran) yapay bir orta-değer etrafına
kümeleyip RF'in split kalitesini düşürüyor. **Sonuç: RF için sıfır-doldurma
zorunlu, medyan-doldurma önerilmiyor.** KNN de zayıf (MCC=0,0746) —
25-boyutlu, zero-inflated bir uzayda mesafe-tabanlı yöntemlerin beklenen
zayıflığıyla tutarlı.

**RF'in en iyi versiyonu: `v4_from_v2`** (F1=0,9233±0,0266) — CatBoost/
v4_from_v2'nin (F1=0,9266±0,0230, Strateji B) hâlâ marjinal gerisinde,
ama aynı ligde; specificity'de (0,2234 vs 0,3767) CatBoost belirgin önde.

### Tier 2 — A/B/C Ağırlıklandırmasının Model Ailesine Göre Farklı Davranışı

**RF/v4_from_v2:**

| Strateji | F1 (paired Δ vs B) | Kazanma oranı | Specificity (paired Δ vs B) | Kazanma oranı |
|---|---|---|---|---|
| A — Ağırlıksız | +0,0038 | %46 | +0,0211 | %32 |
| B — Sabit×2,0 (referans) | — | — | — | — |
| C — Veri-güdümlü SPW | −0,0063 | %32 | **+0,0927** | **%58** |

**CatBoost/v4_from_v2:**

| Strateji | F1 (paired Δ vs B) | Kazanma oranı | Specificity (paired Δ vs B) | Kazanma oranı |
|---|---|---|---|---|
| A — Ağırlıksız | +0,0050 | %54 | **−0,0521** | **%10** |
| B — Sabit×2,0 (referans) | — | — | — | — |
| C — Veri-güdümlü SPW | −0,0160 | %18 | **+0,1123** | **%76** |

**Bulgu — ağırlıklandırma davranışı model ailesine göre nitel olarak
farklı:**

- **CatBoost (boosting): net monoton.** F1'de A>B>C, specificity'de
  C>B>A — E1-EK'in XGBoost bulgusuyla birebir tutarlı, paired kazanma
  oranları da keskin (specificity'de A'nın B'ye karşı kazanma oranı
  yalnızca %10, C'nin %76). Ağırlıklandırma, boosting'in gradyan/hessian
  güncellemesine doğrudan ve öngörülebilir şekilde işliyor.
- **RF (bagging): daha zayıf/az-öngörülebilir.** Ham ortalamalar RF'de de
  benzer bir yön öneriyor gibi görünse de (C'nin specificity'de en iyi
  çıkması ortak), **paired F1 karşılaştırması neredeyse yazı-tura**
  (A vs B kazanma oranı %46) — RF'in bootstrap-örneklemeli yapısının,
  `sample_weight`'i boosting kadar keskin/tutarlı şekilde işlemediğini
  gösteriyor (bilinen bir ML-literatürü gözlemiyle tutarlı: bagging
  yöntemleri örnek ağırlıklandırmaya boosting'den daha az duyarlıdır).

**Nominal en yüksek F1 güncellendi ama pratikte değişmedi:**
`catboost/v4_from_v2/A_no_weight` (F1=0,9316±0,0233), önceki şampiyon
`catboost/v1/B` (F1=0,9313±0,0214) ile **paired karşılaştırmada** ΔF1=
+0,0003, kazanma oranı %42 — istatistiksel olarak **tamamen eşdeğer**,
yeni bir "kazanan" iddiası değil, yalnızca gürültü.

### Tier 3 — Provenance Alt-Grup Performans Kırılımı

Ayrıntılı bulgu `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'nin yeni
ekinde — özet: final adayın (CatBoost/v4_from_v2, Strateji B, **yeniden
eğitilmeden**, `best_params` aynen kullanılarak) 50 dış fold'un tüm
dış-test tahminleri `al_all_missing=0`/`=1` alt-gruplarına bölündü:

| Alt-grup | n | n_benign | F1 | MCC | Specificity |
|---|---|---|---|---|---|
| `al_all_missing=0` | 2800 | 590 | 0,9053 | 0,4525 | 0,3847 |
| `al_all_missing=1` | 890 | 20 | **0,9852** | **−0,0125** | **0,0000** |

**`al_all_missing=1` alt-grubunda model benign'i hiç doğru tahmin
etmiyor** (specificity=0, 20/20 FP) — ama F1 yanıltıcı biçimde yüksek
çıkıyor çünkü bu alt-grup zaten ~%98 patojenik. **Ölçek uyarısı:** bu 20
satır yalnızca **2 benzersiz varyanta** (`VAR_002693`, `VAR_003234`)
karşılık geliyor — çarpıcı ve 10 tekrarın hepsinde tutarlı, ama n=2
düzeyinde bir gözlem, genellenebilir bir istatistik değil.
`VAR_003234` ayrıca bilinen `conflict_group_1` etiket-belirsizliği
grubunun üyesi — bu iki bulgunun (provenance + etiket belirsizliği)
üst üste binmesi, Aşama F'nin adversarial validation'ında ayrıştırılması
gereken ek bir katman.

### Testler

`pytest tests/ -q` → **102/102 geçti**: E1-EK sonrası eklenenler — 4 yeni
(`test_dense_numeric_features_pah.py`, kategori kodlama/medyan doldurmanın
yalnızca train fold'undan geldiğini doğrulayan sızıntı testleri) + 4 yeni
(`test_e2ek_models_pah.py`, biri `nested_cv_evaluate`'in dış-test kümesini
hiperparametre aramasında **hiç** kullanmadığını doğrudan doğrulayan genel
sızıntı testi — RF/KNN dahil bu mekanizmayı kullanan her model ailesine
uygulanıyor).

### Kod / dosyalar

- `src/genova/pah/dense_numeric_features.py` — RF/KNN icin fold-güvenli
  tam-sayısal adaptör (yeni).
- `src/genova/pah/e2ek_models.py` — RF/KNN model sarmalayıcıları + dar
  ızgaralar + A/B/C-enjekte edilebilir RF/CatBoost fabrikaları (yeni).
- `src/genova/pah/e2ek_run.py` — Tier 1+2 orkestrasyonu, `e2_model_
  comparison.csv`'yi genişletir (yeni).
- `src/genova/pah/e2ek_provenance.py` — Tier 3 alt-grup kırılımı (yeni).
- `reports/tables/e2ek_provenance_subgroup_predictions.csv` — Tier 3'ün
  3690 satırlık ham tahmin+alt-grup verisi.
- `tests/test_dense_numeric_features_pah.py`, `tests/test_e2ek_models_
  pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json`'a hiçbir yazma yapılmadı. Hiçbir hiperparametre
aramasının dış-teste baktığı gösterilmedi (yeni genel sızıntı testiyle
doğrulandı). Hiçbir final model kararı verilmedi.

---

## Aşama E3-EK: Genişletilmiş Aday Havuzuyla Kalibrasyon

E2-EK'in genişlettiği aday uzayı (RF × 4 versiyon, KNN, CatBoost/RF ×
A/B/C) E4-E6'ya olduğu gibi taşınırsa kombinasyonel patlama olur — bu
yüzden E3'e başlamadan önce **en fazla 3 final aday**a disiplinli şekilde
daraltıldı.

### E3 Aday Seçimi

1. **LightGBM/v1** — sabit tutuldu. Orijinal E3'te zaten Beta ile kalibre
   edilmişti; bu turda **yeniden kalibre edilmedi**, mevcut sonuç
   (`e3_calibration_metrics.csv`) doğrudan referans olarak kullanıldı.

2. **CatBoost/v4_from_v2 — Strateji B seçildi.** Gerekçe: eşikten
   bağımsız AUPRC'de A/B/C arasında anlamlı fark yok (A=0,9564,
   **B=0,9568**, C=0,9537 — hepsi std≈0,025 içinde; paired kazanma
   oranları A vs B %52, C vs B %34 — kararsız). F1'de de paired
   karşılaştırma istatistiksel bir kazanan göstermiyor (A vs B %54, C vs
   B %18 — C'nin geride kaldığı yönü belirgin ama A/B arasında net bir
   ayrım yok). Karar kuralı gereği (ikisi de belirsizse B varsayılan) ve
   B zaten E2'nin en çok sınanmış/orijinal konfigürasyonu olduğu için
   **B seçildi**. Bu, mevcut CatBoost/v4_from_v2 kalibrasyonunun
   (orijinal E3'te zaten B ile üretilmişti — `models.py::fit_predict_
   catboost` her zaman sabit Strateji B kullanır) **hiçbir yeniden
   hesaplama gerektirmediği** anlamına geliyor; mevcut `e3_calibration_
   metrics.csv` satırları doğrudan kullanıldı.

3. **RF/v4_from_v2 — Strateji B seçildi.** Aynı mantık: AUPRC'de A/B/C
   arasında anlamlı fark yok (A=0,9530, B=0,9541, **C=0,9565** — C'nin
   raw ortalaması en yüksek ama paired kazanma oranı yalnızca %62,
   kesin değil; A vs B %44). F1'de de net bir kazanan yok (paired A vs B
   %46 — neredeyse yazı-tura). Bu, E2-EK'in "RF'de ağırlıklandırma etkisi
   CatBoost'takinden daha az öngörülebilir" bulgusuyla tutarlı — hiçbir
   varyant güçlü bir kanıtla öne çıkmadığı için **B seçildi**
   (muhafazakâr varsayılan). RF/v4_from_v2/B daha önce hiç kalibre
   edilmemişti — bu görevde **yeni** olarak kalibre edildi.

`KNN` ve RF'nin `v1`/`v2`/`v2_raw_nan` versiyonları bu turda ele alınmadı
— E2-EK'te zaten zayıf performans gösterdiler (KNN MCC=0,0746; RF/v1 ve
RF/v2_raw_nan MCC≈0,04-0,06).

### Kalibrasyon Sonuçları (üç aday, 50 dış fold)

| Model | Yöntem | Brier | Log-loss | F1 | Ort. benzersiz olasılık sayısı |
|---|---|---|---|---|---|
| CatBoost/v4_from_v2 | **Beta** | **0,0987±0,0245** | **0,3330±0,0693** | 0,9271±0,0232 | 73,4 |
| CatBoost/v4_from_v2 | Platt | 0,0988±0,0242 | 0,3342±0,0665 | 0,9267±0,0229 | 73,4 |
| CatBoost/v4_from_v2 | Ham | 0,0995±0,0223 | 0,3345±0,0649 | 0,9266±0,0230 | 73,4 |
| CatBoost/v4_from_v2 | Isotonic | 0,1021±0,0247 | 0,3907±0,1364 | 0,9250±0,0228 | 11,3 |
| LightGBM/v1 | **Beta** | **0,1090±0,0282** | **0,3677±0,0795** | 0,9213±0,0280 | 73,5 |
| LightGBM/v1 | Isotonic | 0,1110±0,0293 | 0,4014±0,1237 | 0,9204±0,0271 | 10,4 |
| LightGBM/v1 | Platt | 0,1118±0,0292 | 0,3779±0,0771 | 0,9215±0,0260 | 65,5 |
| LightGBM/v1 | Ham | 0,1184±0,0366 | 0,6418±0,3140 | 0,9199±0,0257 | 68,7 |
| RF/v4_from_v2 | **Beta** | **0,1002±0,0244** | 0,3384±0,0747 | 0,9245±0,0222 | 72,9 |
| RF/v4_from_v2 | Platt | 0,1003±0,0239 | 0,3375±0,0706 | 0,9242±0,0228 | 72,9 |
| RF/v4_from_v2 | Isotonic | 0,1033±0,0266 | 0,4166±0,2012 | 0,9249±0,0231 | 11,5 |
| RF/v4_from_v2 | Ham | 0,1070±0,0226 | 0,3546±0,0597 | 0,9233±0,0266 | 72,9 |

*(RF satırları `src/genova/pah/e3ek_rf_calibration.py` ile bu görevde
üretildi, 50 dış fold'un tamamı; CatBoost/LightGBM satırları orijinal
E3'ten değiştirilmeden alındı.)*

**RF için önerilen kalibratör: Beta — ama Platt'a karşı yalnızca marjinal
üstün.** Paired Brier karşılaştırması (aynı 50 fold): Beta'nın Platt'a
karşı kazanma oranı %58, ortalama fark ihmal edilebilir (−0,0001) —
pratikte eşdeğer, ama Beta yine de bu projedeki **her** model ailesinde
(CatBoost, LightGBM, şimdi RF) tutarlı biçimde en az Platt kadar iyi
çıkıyor, bu yüzden tek-kalibratör tutarlılığı için Beta öneriliyor.

**Isotonic, RF'de de CatBoost/LightGBM'dekiyle aynı aşırı-uyum işaretini
gösteriyor** — ortalama benzersiz kalibre değer sayısı yalnızca 11,5
(Platt/Beta'nın ~73'üne karşı), CatBoost'un 11,3'üne ve LightGBM'in
10,4'üne çok yakın; paired Brier'de Isotonic, Platt'a karşı yalnızca %34
kazanma oranıyla geride. Yani bu bulgu model ailesinden bağımsız, küçük
dış-train boyutunun (n≈295) genel bir sonucu — RF'in farklı olasılık
yapısı (bagging'in oy-oranı tabanlı çıktısı, boosting'in log-odds tabanlı
çıktısından yapısal olarak farklı olsa da) Isotonic'in aşırı-uyum
eğilimini DEĞİŞTİRMEDİ.

### Üç Adayın Özet Tablosu

| Aday | Ağırlıklandırma | Önerilen kalibratör | Brier | Log-loss |
|---|---|---|---|---|
| CatBoost/v4_from_v2 | B (sabit×2,0) | Beta | 0,0987±0,0245 | 0,3330±0,0693 |
| LightGBM/v1 | B (sabit×2,0) | Beta | 0,1090±0,0282 | 0,3677±0,0795 |
| RF/v4_from_v2 | B (sabit×2,0) | Beta | 0,1002±0,0244 | 0,3384±0,0747 |

Üçü de aynı ağırlıklandırma stratejisiyle (B) kalibre edildiği için E4-E6
karşılaştırması tutarlı bir zeminde ilerleyecek.

### Testler

`pytest tests/ -q` çalıştırıldı, sonuç aşağıda. Yeni sızıntı testi
(`test_e3ek_rf_calibration_pah.py`): RF'in çapraz-fit kalibrasyonunun
(`cross_fit_probabilities` + `BetaCalibrator`/`PlattCalibrator`) dış-test
verisine hiç dokunmadan, yalnızca dış-train'in iç örnekleminden fit
edildiğini doğrular (E5/E6'nın sızıntı testleriyle aynı desen).

### Kod / dosyalar

- `src/genova/pah/e3ek_rf_calibration.py` — RF/v4_from_v2/B kalibrasyonu
  (yeni), `e3_calibration_run.py::run_model_calibration`'ı yeniden
  kullanıyor.
- `reports/tables/e3_calibration_metrics.csv`, `e3_calibrated_oof_
  predictions.csv` — mevcut satırlar korunarak RF satırları eklendi.
- `tests/test_e3ek_rf_calibration_pah.py` — yeni sızıntı testi.

Split bankasına, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json`'a hiçbir yazma yapılmadı. E2/E2-EK'in ürettiği hiçbir
tablo değiştirilmedi (yalnızca okundu). CatBoost/v4_from_v2 ve LightGBM/v1
yeniden kalibre edilmedi.

---

## Aşama E4-EK: RF/v4_from_v2 için SLD Önsel Düzeltmesi

RF/v4_from_v2 (Strateji B)'nin Beta-kalibre dış-test olasılıklarına aynı
kapalı-form SLD formülü uygulandı — model yeniden eğitilmedi, `e4_prior_
correction.py`'deki **aynı** `sld_correct` fonksiyonu (aynı `w1=0,343337`,
`w0=4,275449` sabitleri) yeniden kullanıldı, yeni bir formül yazılmadı.
Mevcut CatBoost/v4_from_v2 ve LightGBM/v1 satırları hiç dokunulmadan
korundu (diff ile doğrulandı) — yalnızca RF'in 3690 satırı eklendi.

### Üç adayın SLD-öncesi/sonrası olasılık kayması

| Aday | Ort. proba (Beta, SLD-öncesi) | Ort. proba (SLD-sonrası) |
|---|---|---|
| CatBoost/v4_from_v2 | 0,8240 | 0,4028 |
| LightGBM/v1 | 0,8314 | 0,3708 |
| **RF/v4_from_v2** | **0,8283** | **0,4284** |

RF'in kayması diğer ikisiyle aynı yönde ve büyüklük mertebesinde (~0,83'ten
~0,43'e) — beklenen sonuç, üçü de aynı eğitim/final önsel oranına tabi.

### Elle-doğrulama örneği (RF)

Rastgele seçilen satır: `random_forest/v4_from_v2, repeat=3, outer_fold=1,
VAR_003012`, Beta-kalibre ham olasılık = `0,9054164463613184`.

```
w1 = 0,286/0,833 = 0,343337...
w0 = 0,714/0,167 = 4,275449...
p_yeni = w1·p / (w1·p + w0·(1-p))
       = 0,343337×0,905416 / (0,343337×0,905416 + 4,275449×(1-0,905416))
       = 0,43462157124887923
```

`e4_prior_corrected_probabilities.csv`'deki kayıtlı değer: **aynı satır
için `0,43462157124887923`** — tam kayan-nokta hassasiyetine kadar birebir
eşleşiyor (`tests/test_e4ek_rf_prior_correction_pah.py` ile de otomatik
doğrulanıyor).

### Testler

`pytest tests/ -q` → **107/107 geçti**: 3 yeni test (`test_e4ek_rf_prior_
correction_pah.py`) — elle-hesap doğrulaması, satır sayısı tutarlılığı
(RF'in Beta satır sayısı = SLD satır sayısı), ve RF için ayrı bir formül
yazılmadığının (aynı `sld_correct` fonksiyonu) doğrulanması.

### Kod / dosyalar

- `src/genova/pah/e4ek_rf_prior_correction.py` — RF/v4_from_v2 SLD
  düzeltmesi (yeni), `e4_prior_correction.py::sld_correct`'i aynen
  yeniden kullanıyor.
- `reports/tables/e4_prior_corrected_probabilities.csv` — mevcut 11.070
  satır korunarak RF'in 3690 satırı eklendi (toplam 14.760).
- `tests/test_e4ek_rf_prior_correction_pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json`'a hiçbir yazma yapılmadı. Hiçbir model yeniden
eğitilmedi/arandı.

---

## Aşama E5-EK: Genişletilmiş Üç Aday için Nested Eşik Seçimi

E5'in orijinal nested prosedürü (`process_fold`, `select_threshold`),
RF/v4_from_v2 (Strateji B) için **birebir aynı disiplinle** yeniden
kullanıldı (`src/genova/pah/e5ek_rf_threshold_selection.py`). CatBoost/
v4_from_v2 ve LightGBM/v1 **yeniden hesaplanmadı** — mevcut
`e5_threshold_selection.csv` satırları (orijinal E5'ten, zaten Strateji
B ile) doğrudan kullanıldı; yalnızca RF'in 50 satırı eklendi (diff ile
eski satırların bit-birebir korunduğu doğrulandı).

### Seçilen eşikler

| Aday | Seçilen eşik (ort±std) | Aralık |
|---|---|---|
| CatBoost/v4_from_v2 | 0,3315±0,0731 | 0,13–0,50 |
| LightGBM/v1 | 0,2737±0,0525 | 0,14–0,37 |
| **RF/v4_from_v2** | **0,2949±0,0705** | 0,16–0,47 |

### Karar kuralı (üçünde de: adaptif eşik)

| Aday | Ağırlıklı-F1 (adaptif) | Ağırlıklı-F1 (sabit 0,5) | Ort. fark | %95 GA | Karar |
|---|---|---|---|---|---|
| CatBoost/v4_from_v2 | 0,6187±0,0855 | 0,4931±0,0805 | +0,1256 | [0,0981; 0,1532] | **Adaptif** (iyileşme yarı-GA'nın ~4,6×'i) |
| LightGBM/v1 | 0,5818±0,1057 | 0,2948±0,1767 | +0,2870 | [0,2262; 0,3511] | **Adaptif** (~4,6×) |
| RF/v4_from_v2 | 0,6202±0,1007 | 0,5275±0,0699 | +0,0927 | [0,0629; 0,1193] | **Adaptif** (~3,3×) |

Üçünde de ortalama iyileşme bootstrap %95 GA'nın yarı-genişliğinin kat kat
üzerinde — RF'de de sonuç net: sabit 0,5, SLD-sonrası ölçekte aşırı
tutucu davranıp F1'i çökertiyor (0,6071'e karşı adaptif eşikte 0,8284),
adaptif eşik bunu düzeltiyor.

### Duyarlılık eğrisi (final önsel %20-%38 arası)

| Aday | %20 | %25 | %28,6 | %33 | %38 | Aralık (maks−min) |
|---|---|---|---|---|---|---|
| CatBoost/v4_from_v2 | 0,5463 | 0,5919 | 0,6187 | 0,6466 | 0,6731 | 0,1268 |
| LightGBM/v1 | 0,4901 | 0,5470 | 0,5818 | 0,6189 | 0,6552 | 0,1652 |
| RF/v4_from_v2 | 0,5416 | 0,5907 | 0,6202 | 0,6511 | 0,6810 | 0,1394 |

Üçü de düzgün/monotonik, ani bir uçurum yok — RF de dahil, sabit eşiğin
±5-10 puanlık bir prevalans belirsizliğine karşı **sağlam** olduğu sonucu
genişletilmiş aday havuzunda da doğrulandı.

### Üç adayın nihai karşılaştırması

| Aday | F1 (adaptif eşik, ham) | MCC | Specificity | **Ağırlıklı-F1 (asıl karşılaştırma)** |
|---|---|---|---|---|
| CatBoost/v4_from_v2 | 0,7988±0,0739 | 0,3659±0,0849 | 0,7533±0,1535 | 0,6187±0,0855 |
| LightGBM/v1 | 0,8426±0,0789 | 0,3531±0,1389 | 0,6027±0,1580 | 0,5818±0,1057 |
| RF/v4_from_v2 | 0,8284±0,0595 | 0,3795±0,0894 | 0,7148±0,1577 | **0,6202±0,1007** |

**Nominal en yüksek ağırlıklı-F1 artık RF/v4_from_v2 (0,6202)** — ama
CatBoost/v4_from_v2'yle (0,6187) **paired karşılaştırmada** ortalama fark
yalnızca +0,0015, kazanma oranı %46 — **istatistiksel olarak tamamen
eşdeğer**, yeni bir "kazanan" iddiası değil (E2-EK'in `catboost/v4_
from_v2/A` vs `catboost/v1/B` bulgusuyla aynı desen: nominal sıralama
değişiyor ama gürültü seviyesinde). Üç aday da std'lerin (~0,08-0,11)
içinde örtüşüyor — E6'nın ensemble değerlendirmesinde bu üçü arasında
kesin bir sıralama iddiası hâlâ yok.

### Testler

`pytest tests/ -q` → **109/109 geçti**: 2 yeni test (`test_e5ek_rf_
threshold_selection_pah.py`) — biri dış-test içeriği değiştirilince
`chosen_threshold`'un aynı kaldığını doğrulayan sızıntı testi (E5'in
orijinal testiyle aynı desen), diğeri gerçek split bankasında RF'in
çapraz-fit örnekleminin dış-train'i tam kapladığını doğrulayan entegrasyon
testi.

### Kod / dosyalar

- `src/genova/pah/e5ek_rf_threshold_selection.py` — RF/v4_from_v2 nested
  eşik seçimi (yeni), `e5_threshold_selection.py::process_fold`'u aynen
  yeniden kullanıyor.
- `reports/tables/e5_threshold_selection.csv` — mevcut 150 satır
  korunarak RF'in 50 satırı eklendi (toplam 200).
- `tests/test_e5ek_rf_threshold_selection_pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json`'a hiçbir yazma yapılmadı. CatBoost/v4_from_v2 ve
LightGBM/v1 yeniden hesaplanmadı.

---

## Aşama E6-EK: RF Dahil Ensemble Yeniden Değerlendirmesi (Aşama E'nin Son Adımı)

### Adım 1 — 3×3 OOF Korelasyon Matrisi

Üç adayın (CatBoost/v4_from_v2, LightGBM/v1, RF/v4_from_v2) SLD-düzeltilmiş
dış-test olasılıklarından, `repeat=0`'ın 5 dış fold'unda (369 satır,
sızıntısız) hesaplanan gerçek matris:

| | CatBoost/v4_from_v2 | LightGBM/v1 | RF/v4_from_v2 |
|---|---|---|---|
| **CatBoost/v4_from_v2** | 1,000 | 0,7644 | **0,8764** |
| **LightGBM/v1** | 0,7644 | 1,000 | **0,7397** |
| **RF/v4_from_v2** | 0,8764 | 0,7397 | 1,000 |

**Beklenmedik bulgu — RF↔CatBoost, önceki CatBoost↔LightGBM'den (0,7644)
DAHA yüksek korele (0,8764), daha düşük değil.** Hipotez ("bagging,
boosting'den yapısal olarak farklı, gerçek çeşitlilik sağlar") bu ikili
için **doğrulanmadı** — muhtemelen RF ve CatBoost'un **aynı** `v4_from_v2`
(25-özellik) veri setini kullanması, algoritma ailesi farkından daha
baskın çıkıyor. En düşük korelasyon **LightGBM/v1 ↔ RF/v4_from_v2
(0,7397)** — hem farklı algoritma ailesi hem farklı veri versiyonu (`v1`
vs `v4_from_v2`) kombinasyonu. Bu yüzden Adım 2'nin ensemble denemesi bu
ikiliyle yapıldı; CatBoost+RF (en yüksek korelasyon, orijinal CatBoost+
LightGBM'den bile daha az umut verici) test edilmedi.

### Adım 2 — Ensemble: LightGBM/v1 + RF/v4_from_v2

| Blend yöntemi | Ağırlıklı-F1 (ort±std) | F1 (ham) | Seçilen eşik (ort±std) |
|---|---|---|---|
| **Basit ortalama** | **0,6181±0,0969** | 0,8487±0,0540 | 0,2697±0,0478 |
| Ağırlıklı (kısıtlı) | 0,6129±0,0947 | 0,8364±0,0620 | 0,2834±0,0646 |
| Rank-average | 0,5998±0,0955 | 0,7667±0,0507 | 0,0568±0,0110 |

Kazanan: basit ortalama — ama farkı diğerlerinden küçük. Ağırlıklı blend'in
kendi α seçiminde ortalama α=0,274 (RF'ye ~%73 ağırlık) — 50 fold'un
10'unda (%20) α=0,0 (LightGBM'e sıfır ağırlık, saf RF tercih edildi),
hiçbir fold'da α=1,0 (saf LightGBM) seçilmedi. Bu, RF'in bu ikilide
tutarlı biçimde daha güçlü üye olduğunu gösteriyor.

### Adım 3 — Karar: Ensemble solo modellere karşı paired karşılaştırma

| Blend yöntemi | vs RF solo (Δ, kazanma) | vs CatBoost solo (Δ, kazanma) | Karar |
|---|---|---|---|
| Basit ortalama | −0,0021, %52 | −0,0006, %44 | **Solo** |
| Ağırlıklı | −0,0073, %34 | −0,0059, %42 | **Solo** |
| Rank-average | −0,0204, %42 | −0,0189, %34 | **Solo** |

**Sonuç: net ve tekrarlanan — ensemble yine solo modeli geçemiyor.** Daha
düşük korelasyonlu (0,7397 vs önceki 0,7644), yapısal olarak daha farklı
bir ikili (bagging+boosting) denenmesine rağmen, üç blend yönteminin
**hiçbiri** ne RF'e ne CatBoost'a karşı bootstrap CI'nin yarı-genişliğini
aşan bir iyileşme sağlamıyor — kazanma oranları %34-%52 arası, hepsi
pratikte yazı-tura veya daha kötü. Bu, orijinal E6'nın (CatBoost+LightGBM)
bulgusunu **ikinci kez, farklı bir aday çiftiyle de doğruluyor**: bu
panelde (n=369) ensemble'ın gerçek/tutarlı bir kazancı yok.

### Aşama E'nin Nihai Final Adayı

İki solo aday (RF/v4_from_v2 ağırlıklı-F1=0,6202, CatBoost/v4_from_v2=
0,6187) paired karşılaştırmada istatistiksel olarak eşdeğer (%46 kazanma
oranı, bkz. E5-EK). Ensemble ikisini de geçemedi. Bu durumda nihai seçim,
saf F1 farkından değil şu ek gerekçelerden yapıldı:

1. **CatBoost, tüm E-aşaması boyunca en tutarlı performans gösteren
   aday** — E2'nin orijinal 16-kombinasyonluk taramasından beri hemen her
   alt-karşılaştırmada üst sırada; RF ancak E2-EK'in ek turlarıyla
   istatistiksel paritede.
2. **RF'in belgelenmiş kırılganlığı (E2-EK Tier 1):** `AL_` doldurma
   stratejisine aşırı duyarlı — medyan-doldurmada MCC≈0,04-0,06'ya
   çöküyor, yalnızca sıfır-doldurmada makul (MCC≈0,38). Bu, Aşama F3'ün
   (sağlamlık stres testleri: yeni/görülmemiş kategori, eksiklik oranı
   yapay artışı) doğrudan hedeflediği türden bir kırılganlık sinyali.
3. **CatBoost'un native `NaN`/kategorik desteği**, RF'in gerektirdiği ek
   önişleme katmanından (`dense_numeric_features.py`: frekans kodlama +
   medyan doldurma) yapısal olarak daha az kırılgan — F3'ün stres
   testlerinde ek bir başarısızlık yüzeyi taşımıyor.

**Önerilen nihai aday (onayınıza sunuluyor): CatBoost/v4_from_v2, solo
(ensemble değil), Beta kalibrasyon, SLD önsel düzeltmesi (w1=0,343337,
w0=4,275449), nested dış-fold eşiği (ort±std ≈0,3315±0,0731).**
RF/v4_from_v2 istatistiksel olarak yakın bir alternatif olarak
`e5_threshold_selection.csv`'de tam olarak belgeli kalıyor — F3'ün
sağlamlık sonuçlarına göre bu karar yeniden değerlendirilebilir.

### Testler

`pytest tests/ -q` → **111/111 geçti**: 2 yeni test (`test_e6ek_
ensemble_pah.py`) — ensemble ağırlığının (`alpha`) ve eşiğinin dış-teste
bakmadan seçildiğini doğrulayan sızıntı testi (E6'nın orijinal testiyle
aynı desen) + korelasyon fonksiyonunun `e6_ensemble.py`'den değiştirilmeden
yeniden kullanıldığını doğrulayan test.

### Kod / dosyalar

- `src/genova/pah/e6ek_ensemble.py` — LightGBM/v1 + RF/v4_from_v2
  ensemble değerlendirmesi (yeni); `e6_ensemble.py`'nin `process_
  ensemble_fold`'unu genelleştirilmiş haliyle yeniden kullanıyor
  (`fit_predict_a`/`fit_predict_b` artık parametre, sabit değil),
  `compute_oof_correlation_matrix`'i değiştirmeden içe aktarıyor.
  Orijinal `e6_ensemble.py` **değiştirilmedi**, kendi çıktıları korundu.
- `reports/tables/e6ek_oof_correlation_matrix.csv`, `e6ek_ensemble_
  metrics.csv` — yeni, ayrı dosyalar (orijinal E6 dosyalarının üzerine
  yazılmadı).
- `tests/test_e6ek_ensemble_pah.py` — yeni testler.

Split bankasına, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json`'a hiçbir yazma yapılmadı. E0-E5'in hiçbir tablosu
değiştirilmedi.

---

## AŞAMA E TAMAMLANDI

E0'dan E6-EK'e kadar tüm alt-adımlar tamamlandı ve nested kanıtla
doğrulandı. Önerilen final aday (onay bekliyor): **CatBoost/v4_from_v2,
solo, Beta kalibrasyon, SLD önsel düzeltmesi, nested eşik≈0,3315**.
Bir sonraki adım Aşama F (adversarial validation, bootstrap güven
aralıkları, sağlamlık stres testleri, SHAP açıklanabilirlik, final
paketleme) — ayrı bir onayla başlayacak.

---

## P1 Madde 10: Örnek-Düzeyi Belirsizlik (Final Modelin Kendi OOF'u Üzerinde)

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P1 madde 10.
> Hiçbir model yeniden eğitilmedi; `f0_final_model.py` değiştirilmedi.
> Yeni genel-amaçlı modül: `src/genova/statistics.py` (3 fonksiyon, 12
> birim test). Uygulama: `src/genova/pah/f0_uncertainty_analysis.py`,
> çıktı: `reports/tables/f0_uncertainty_analysis.csv`.
>
> ⚠️ **DÜZELTME NOTU (2026-09-01, "Yanlış Bundle Referansının
> Düzeltilmesi" görevi):** Bu bölümün ilk yazıldığı tarihten beri,
> `f0_uncertainty_analysis.py` yanlışlıkla `final_model_bundle.pkl`'i
> (P1 madde 11 ÖNCESİ, 28-özellikli, resmi OLMAYAN tarihsel bundle)
> yüklüyordu — resmi `final_model_bundle_v2.pkl` DEĞİL. Kaynağından
> düzeltildi (`BUNDLE_OUT` → `BUNDLE_PATH`), aşağıdaki sayılar yeniden
> üretildi. Tam düzeltme geçmişi ve eski (yanlış) sayılar için bkz.
> `reports/10_BOOTSTRAP_GUVEN_ARALIKLARI_PAH.md`'nin kendi düzeltme notu.

**Yöntem notu:** Final bundle'in (`models/pah/final_model_bundle_v2.pkl`)
kendi 5-fold cross-fit OOF'u (Adım 3, P0-5) hiç persist edilmemişti; bu
yüzden bundle'in **kendi dondurulmuş** `pool`/`model_best_params`'ıyla
aynı deterministik `cross_fit_oof` fonksiyonu burada **yeniden çağrıldı**
(bit-bit aynı sonucu verir — bu bir yeniden eğitim değil, zaten bundle'i
üretmiş olan aynı hesabın tekrarı). Kalibratör/önsel-düzeltme/eşik ise
bundle'in kendi fit edilmiş nesnelerinden okundu, hiçbiri yeniden fit
edilmedi.

### 1) Örnek-düzeyi (369 satır, satır-bazlı) vs fold-düzeyi (50 dış-fold) CI

| Metrik | Nokta tahmini | Örnek-düzeyi %95 CI | Genişlik |
|---|---|---|---|
| F1 | 0,8177 | [0,7824 ; 0,8509] | **0,0685** |
| MCC | 0,3921 | [0,2898 ; 0,4902] | 0,2004 |
| Specificity | 0,7869 | [0,6765 ; 0,8909] | 0,2145 |

Karşılaştırma (yalnızca F1, CatBoost/v4_from_v2): mevcut fold-düzeyi CI
(`e5_threshold_selection.csv`'nin 50 dış-fold `f1_chosen` değerinin
FOLD-bazlı bootstrap'ı — denetimin "dar/iyimser" dediği yöntemin ta
kendisi) genişlik **0,0402** veriyor; örnek-düzeyi (doğru birim, 369 satır)
genişlik **0,0685** — **~1,70×** daha geniş.

**Şeffaflık notu:** İki CI'nin nokta tahminleri (0,7988 fold-düzeyi vs
0,8345 örnek-düzeyi) birebir aynı F1 sayısı değil — fold-düzeyi sayı, E5'in
50-dış-fold nested değerlendirmesinden (eski/leaky havuz, P0 düzeltmesi
öncesi, henüz yeniden koşulmadı); örnek-düzeyi sayı, yeni final bundle'ın
kendi 5-fold cross-fit OOF'undan (P0-düzeltmeli havuz) geliyor. Bu
karşılaştırma "aynı sayının iki farklı CI'si" değil, "iki farklı ama
karşılaştırılabilir yöntemin CI GENİŞLİĞİ" karşılaştırmasıdır — asıl mesaj
(fold-düzeyi CI'nin sistematik olarak dar olduğu, yönü) geçerli, ama
genişlik oranı (1,61×) denetimin kaba tahminiyle (≈4×, 0,047→0,179) birebir
eşleşmiyor; burada raporlanan **gerçekten hesaplanmış** sayıdır, tahmin
değil.

### 2) Monte Carlo final-F1 simülasyonu (100 patojenik / 250 benign, eşik=0,35)

| İstatistik | Değer |
|---|---|
| Ortalama F1 | 0,6395 |
| Medyan F1 | 0,6400 |
| %95 aralık | [0,5738 ; 0,7000] |

*(Düzeltme öncesi/yanlış değerler, tarihsel kayıt: ortalama F1=0,6253,
medyan=0,6260, aralık=[0,5645;0,6838], eşik "0,36" — kullanılmamalı.)*

2000 simülasyon, her birinde OOF'tan (yerine koyarak) 100 patojenik + 250
benign satır çekilip final eşik (0,35) uygulandı. Ortalama F1 (0,6395),
eğitim-önseli kompozisyonundaki OOF F1'inden (0,8177) belirgin şekilde
düşük — bu **beklenen bir sonuç**: final kompozisyon eğitimdekinden çok
daha az patojenik-ağırlıklı (%28,6 vs %83,3), F1'in payda bileşeni
(false positive'ler) böyle bir kompozisyonda orantılı olarak büyür. %95
aralığın genişliği (0,126) tek başına, final skorun tek bir nokta tahmini
yerine bir aralık olarak sunulması gerektiğini gösteriyor.

### 3) Nadeau-Bengio düzeltmesi — CatBoost/v4_from_v2 vs RF/v4_from_v2 (E5)

En tartışmalı geçmiş karar yeniden kontrol edildi (`e5_threshold_
selection.csv` yalnızca okundu, yeniden hesaplanmadı):

| Metrik | Ortalama fark (CatBoost−RF) | Naif paired-t p | Nadeau-Bengio p |
|---|---|---|---|
| **weighted_f1 (E5'in resmi karar metriği)** | **−0,0015** | 0,869 | **0,964** |
| f1 (ham, bilgi amaçlı) | −0,0296 | 0,0033 | 0,404 |

Resmi karar metriğinde (weighted_f1) fark zaten naif testte de anlamsızdı
(p=0,869); NB düzeltmesi bunu daha da netleştiriyor (p=0,964) — "istatistiksel
olarak yazı-tura" ifadesi doğrulandı. Ham F1'de naif test yanıltıcı bir
"anlamlı" sonuç verirken (p=0,0033), NB düzeltmesi bunun 50 bağımlı fold'un
varyansını hafife almaktan kaynaklandığını gösteriyor (düzeltilmiş p=0,404)
— NB'nin neden gerekli olduğunun somut bir örneği. **Bu, final model
kararını değiştirmiyor** (CatBoost zaten seçilmişti, gerekçe tie-break
değildi) — yalnızca istatistiksel ifadeyi netleştiriyor.

### Kod / dosyalar

- `src/genova/statistics.py` — `sample_level_bootstrap_ci`,
  `nadeau_bengio_corrected_ttest`, `monte_carlo_final_f1_simulation`
  (panel/model'e özel değil, genel-amaçlı). `tests/test_statistics_pah.py`
  — 12 test, sentetik veri + bilinen analitik sonuçlarla.
- `src/genova/pah/f0_uncertainty_analysis.py` — final bundle'ın OOF'unu
  yeniden üretip (deterministik, yeniden eğitim değil) yukarıdaki 4 analizi
  çalıştırır. `reports/tables/f0_uncertainty_analysis.csv` — sonuçlar.

Split bankasına, `data/processed/pah/`, `f0_final_model.py`, `predict.py`,
`final_model_bundle.pkl`'e hiçbir yazma yapılmadı; E2-E6'nın hiçbir CSV/
model sonucu yeniden hesaplanmadı (`e5_threshold_selection.csv` yalnızca
okundu).

---

## P1 Madde 11: Provenance-Riskli Özelliklerin Çıkarılması (`al_all_missing`/`CAT_1`)

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P1 madde 11.
> Denetimin ablasyon deneyi bu iki özelliği çıkarmanın ölçülebilir bir
> maliyeti olmadığını (AUPRC farkı p=0,83) göstermişti — "bedava sigorta"
> kararı: kaynak-kısayolu riski taşıyan bu iki özellik resmi final modelden
> çıkarıldı. Madde 10'un istatistik araçları (`sample_level_bootstrap_ci`,
> `nadeau_bengio_corrected_ttest`, `monte_carlo_final_f1_simulation`)
> eski/yeni modeli karşılaştırmak için yeniden kullanıldı.

### 1) Özellik listesi

`al_all_missing` ve `CAT_1`, mevcut 28-özellik final havuzunda **ikisi de
gerçekten mevcuttu** — ikisi de çıkarıldı. Eski havuz `f0_final_feature_
pool_ARCHIVED_with_provenance_features.json` olarak arşivlendi (silinmedi);
yeni 26-özellik havuz `f0_final_feature_pool_v2.json`'a yazıldı.

### 2) Yeni final model

`f0_final_model.py`'nin Adım 2-6'sı (Adım 1 — özellik seçimi — bu turda
**tekrar çalıştırılmadı**, havuz elle düzenlendi), aynı fonksiyonlar
(`select_hyperparameters`, `cross_fit_oof`, `select_calibrator`, `select_
prior_and_threshold`, `fit_frozen_v2_preprocessing`, `_fit_catboost_final`
— dosya değiştirilmedi, yalnızca içe aktarılıp yeniden çağrıldı) 26-özellik
havuzuyla tekrarlandı:

| Alan | Eski (28 özellik) | Yeni (26 özellik) |
|---|---|---|
| Hiperparametre | depth=5, lr=0,05 | depth=5, lr=0,05 (aynı) |
| İç-ortalama F1 (Adım 2) | 0,9392 | 0,9333 |
| Kalibratör | Beta (Brier=0,0916) | Beta (Brier=0,0946) |
| Eşik | 0,36 | 0,35 |

`models/pah/final_model_bundle_v2.pkl` olarak kaydedildi. Eski `final_
model_bundle.pkl` **arşivlendi** (`final_model_bundle_ARCHIVED_with_
provenance_features.pkl`, kopya — orijinal silinmedi, `predict.py` artık
onu kullanmıyor ama dosya diskte duruyor).

### 3) Eski/yeni karşılaştırma (madde 10'un araçlarıyla)

**Örnek-düzeyi CI (369 satır, satır-bazlı bootstrap) — örtüşme:**

| Metrik | Eski | Yeni | Örtüşüyor mu |
|---|---|---|---|
| F1 | 0,8345 [0,8007; 0,8679] | 0,8177 [0,7820; 0,8520] | ✅ |
| MCC | 0,3885 [0,2819; 0,4877] | 0,3921 [0,2964; 0,4839] | ✅ |
| Specificity | 0,7377 [0,6207; 0,8448] | 0,7869 [0,6842; 0,8834] | ✅ |

**Nadeau-Bengio (aynı 5-fold iç CV yapısı — iki bundle da aynı `sb.build_
outer_folds(v1_df, seed=F0_SEED+1)` bölmesini paylaşıyor, k=5 — küçük-k
uyarısıyla):**

| Metrik | Ortalama fark (eski−yeni) | NB p-değeri |
|---|---|---|
| F1 | +0,0166 | 0,3147 |
| MCC | −0,0002 | 0,9975 |
| Specificity | −0,0419 | 0,5383 |

**Monte Carlo final-F1 (100 patojenik/250 benign):**

| | Ortalama | %95 aralık |
|---|---|---|
| Eski | 0,6253 | [0,5645; 0,6838] |
| Yeni | 0,6395 | [0,5738; 0,7000] |

Üç aracın hepsi aynı sonuca varıyor: **hiçbir metrikte büyük bir sapma
yok** — tüm örnek-düzeyi CI'ler örtüşüyor, Nadeau-Bengio'da hiçbir metrik
anlamlı çıkmıyor (en küçük p=0,31), Monte Carlo ortalamaları birbirinin
CI'sinin içinde. (k=5 çok küçük bir örneklem olduğu için NB sonuçları
kesin bir kanıt değil, yalnızca destekleyici — asıl kanıt örnek-düzeyi
CI'lerin geniş örtüşmesi.)

### 4) Karar

**Büyük bir sapma yok — beklenen sonuç doğrulandı.** `final_model_bundle_
v2.pkl` yeni resmi final model oldu. `predict.py`'nin `DEFAULT_BUNDLE_
PATH`'i artık `final_model_bundle_v2.pkl`'i gösteriyor (varsayılan
davranış olarak, `--bundle` ile geçersiz kılınabilir). Eski `final_model_
bundle.pkl` diskte duruyor, silinmedi, ama artık kullanılmıyor.

### Kod / dosyalar

- `src/genova/pah/f0_final_model_v2.py` — Adım 2-6'yı 26-özellik havuzuyla
  tekrarlar; `f0_final_model.py` değiştirilmedi, yalnızca içe aktarıldı.
- `src/genova/pah/f0_provenance_ablation_comparison.py` — eski/yeni
  karşılaştırması; `reports/tables/f0_provenance_ablation_comparison.csv`.
- `reports/tables/f0_final_feature_pool_v2.json` (yeni), `f0_final_
  feature_pool_ARCHIVED_with_provenance_features.json` (arşiv).
- `models/pah/final_model_bundle_v2.pkl` (yeni resmi), `final_model_
  bundle_ARCHIVED_with_provenance_features.pkl` (arşiv kopyası).
- `predict.py` — `DEFAULT_BUNDLE_PATH` artık v2'yi gösteriyor.
- `tests/test_f0_final_model_pah.py` — `BUNDLE_PATH` v2'ye güncellendi,
  2 yeni test eklendi (`DEFAULT_BUNDLE_PATH` doğrulaması, havuzda
  provenance-riskli özelliklerin olmadığı doğrulaması).

Split bankasına, `e5_threshold_selection.csv` vb. E2-E6 development
dosyalarına hiçbir yazma yapılmadı; hiçbir dosya silinmedi.

---

## P1 Madde 7: Objective Mismatch Düzeltmesi — ⏸ DURDURULDU, ONAY BEKLİYOR

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P1 madde 7.
> **Bu bölüm bir karar İÇERMİYOR** — büyük bir sapma tespit edildiği için
> süreç kasıtlı olarak durduruldu. `predict.py` ve resmi `final_model_
> bundle_v2.pkl` **değiştirilmedi**.

### Yöntem

`f0_final_model.py`'nin Adım 2'sindeki iç seçim ölçütü (eşik=0,5 ham F1),
yeni bir fonksiyonda (`f0_final_model_v3.py::prior_weighted_inner_score`,
4 birim testle doğrulandı) kapalı-form Bayes önsel düzeltmesi + uyarlanabilir
eşik sonrası önsel-ağırlıklı F1 ile değiştirildi — kalibratör bu iç döngüde
**aranmadı** (bilinçli sadeleştirme, veri açlığı riski). Aynı 4-fold iç CV
bölmesi (aynı `F0_SEED`), aynı `CATBOOST_GRID` (2 aday) kullanıldı — tek
değişken skor formülüydü.

### 1) Sonuç: farklı hiperparametre seçildi

| Ölçüt | Seçilen | İç-ortalama skor |
|---|---|---|
| Eski (eşik=0,5 ham F1) | `depth=5, lr=0,05` | 0,9333 |
| **Yeni (önsel-ağırlıklı F1)** | **`depth=3, lr=0,05`** | 0,6870 |

**Uyarı — marjinal fark:** her iki ölçütte de iki aday arasındaki fark
küçük (yeni ölçüt: 0,6870 vs 0,6774, fark≈0,010; eski ölçüt: 0,9333 vs
0,9253, fark≈0,008) — yalnızca 4 iç-fold ve 2 aday olduğu için bu seçim
gürültüye karşı kırılgan olabilir.

### 2) Eski/yeni karşılaştırma (madde 10 araçları) — BÜYÜK SAPMA

| Metrik | Eski (depth=5) | Yeni (depth=3) | CI örtüşüyor mu |
|---|---|---|---|
| F1 | 0,8177 [0,7820; 0,8520] | 0,9040 [0,8784; 0,9272] | ❌ **HAYIR** |
| MCC | 0,3921 [0,2964; 0,4839] | 0,4749 [0,3564; 0,5855] | ✅ |
| Specificity | 0,7869 [0,6842; 0,8834] | 0,6230 [0,5000; 0,7419] | ✅ |

Nadeau-Bengio (k=5): F1 farkı p=0,0157 (anlamlı), specificity farkı
p=0,0396 (anlamlı, ters yönde — yeni model specificity'de düşük), MCC
farkı p=0,3455 (anlamsız). Monte Carlo (100/250): eski ort=0,6395, yeni
ort=0,6262 (bunlar örtüşüyor — asıl sapma eğitim-önseli kompozisyonundaki
F1'de, final kompozisyonda değil).

**Yorum:** yeni (depth=3) model, F1'de örnek-düzeyi CI'si eskiyle
ÖRTÜŞMEYECEK kadar farklı — bu görevin kendi "büyük sapma" eşiğini aşıyor.
Bu, `depth=3`'ün gerçekten daha mı iyi olduğunu, yoksa (madde 1'deki
marjinal iç-skor farkına bakılırsa) küçük veri setinde gürültülü bir
hiperparametre seçiminin büyütülmüş bir yansıması mı olduğunu netleştirmek
için **daha fazla inceleme gerektiriyor** — otomatik olarak karar
verilmedi.

### 3) Karar: DURDURULDU

`models/pah/final_model_bundle_v3.pkl` **incelemek için kaydedildi** ama
resmi yapılmadı — `final_model_bundle_v2.pkl` arşivlenmedi, `predict.py`
güncellenmedi. Sonuçlar `reports/tables/f0_objective_fix_comparison.csv`'de.
**Bir sonraki adım için kullanıcı onayı/yönlendirmesi bekleniyor.**

### Kod / dosyalar

- `src/genova/pah/f0_final_model_v3.py` — yeni iç seçim ölçütü +
  koşullu yeniden-fit + karşılaştırma; `f0_final_model.py` değiştirilmedi.
- `tests/test_f0_final_model_v3_pah.py` — 4 test (`prior_weighted_inner_
  score`: kusursuz ayrımda F1=1, kompozisyon doğrulaması, eski ölçütten
  farklılığı, girdi hatası).
- `reports/tables/f0_objective_fix_comparison.csv`,
  `models/pah/final_model_bundle_v3.pkl` (resmi DEĞİL, yalnızca inceleme).

---

## P1 Madde 7 — Tekrarlı-Bölme Teşhisi ve NB Son Karar

### Tekrarlı-bölme teşhisi (10 tekrar, `StratifiedKFold`, split bankasından bağımsız)

`depth=3` kazandı: **7/10**, `depth=5` kazandı: **3/10** (ortalama marj
+0,0272, yön sistematik olarak depth=3'e eğik). Bağımsız bir binom-testi
kontrolü bu oranın (7/10) **istatistiksel olarak zayıf** olduğunu
gösterdi (p=0,172, geleneksel anlamlılık eşiğinin üzerinde) — bu projede
tekrar tekrar görülen "nominal fark büyük, testte anlamsız" deseninin bir
tekrarı (bkz. CatBoost/RF, Beta/Platt karşılaştırmaları). Kod:
`src/genova/pah/f0_madde7_stability_check.py`, çıktı: `reports/tables/
f0_madde7_stability_check.csv`.

### Nadeau-Bengio son karar testi — ⚠️ ÇELİŞKİLİ SONUÇ, KARAR ERTELENDİ

`depth=5` (`final_model_bundle_v2.pkl`) vs `depth=3` (`final_model_bundle_
v3.pkl`), aynı 26-özellik havuzu + aynı 5-fold iç CV yapısında (k=5),
dört metrik için ayrı ayrı Nadeau-Bengio testi (`src/genova/pah/f0_madde7_
final_decision.py`, çıktı: `reports/tables/f0_madde7_nb_decision.csv`):

| Metrik | Ortalama fark (eski−yeni) | NB p-değeri | Yön |
|---|---|---|---|
| F1 (ham) | −0,0871 | **0,0157** | depth=3 lehine |
| MCC | −0,0731 | 0,3455 | depth=3 lehine (anlamsız) |
| Specificity | +0,1732 | **0,0396** | **depth=5 lehine** |
| **Weighted-F1 (resmi karar metriği)** | +0,0155 | 0,7793 | depth=5 lehine (anlamsız) |

**Önceden yazılmış karar kuralı** ("herhangi bir ana metrikte p<0,05 VE
yön depth=3 lehine") teknik olarak F1 satırında sağlanıyor (p=0,0157) —
otomasyon bu yüzden `final_model_bundle_v2.pkl`'i `final_model_bundle_
ARCHIVED_pre_objective_fix.pkl` olarak **arşivledi** (orijinal `v2`
dosyası silinmedi, hâlâ yerinde).

**Ama bu noktada durduruldu, `predict.py` GÜNCELLENMEDİ** — çünkü sonuç
göründüğünden daha karmaşık:

- Görevin kendi metni, kararın "**özellikle** önsel-ağırlıklı F1'de"
  anlamlı çıkmasını vurguluyordu — ama tam olarak bu metrik (weighted-F1,
  projede E3'ten beri kullanılan **resmi karar metriği**) anlamsız
  çıkıyor (p=0,78) ve nominal yönü bile depth=3'ü değil depth=5'i
  destekliyor.
- **Specificity aynı anda, ters yönde, anlamlı** (p=0,0396, depth=5
  lehine) — yani "depth=3 daha iyi" sonucunu veren tek metrik ham
  (ağırlıksız) F1, ve bu metrikte kazanç specificity'de eşdeğer bir
  kayıpla geliyor gibi görünüyor.
- Basitçe "4 metrikten 1'i anlamlı çıktı" kuralını mekanik uygulamak,
  görevin kendi ruhuyla (NB'yi zayıf/çelişkili sinyalleri elemek için
  zorunlu kriter yapmak) çelişir — dört metrikten yalnızca biri (ve resmi
  olmayanı) anlamlı, resmi metrik anlamsız, bir metrik de ters yönde
  anlamlı.

**Sonuç: karar kullanıcıya bırakıldı.** `final_model_bundle_v2.pkl`
(depth=5) **hâlâ resmi final** — `predict.py` değiştirilmedi.
`final_model_bundle_v3.pkl` (depth=3) diskte duruyor, silinmedi. Arşiv
kopyası (`final_model_bundle_ARCHIVED_pre_objective_fix.pkl`) zararsız
bir yan etki — orijinal `v2` bozulmadı.

### Kod / dosyalar

- `src/genova/pah/f0_madde7_stability_check.py` — 10 tekrarlı bölme
  teşhisi.
- `src/genova/pah/f0_madde7_final_decision.py` — NB testi + karar
  otomasyonu (arşivleme dahil, `predict.py` güncellemesi HARİÇ —
  kasıtlı olarak ayrı bırakıldı).
- `reports/tables/f0_madde7_stability_check.csv`,
  `f0_madde7_nb_decision.csv`.

Split bankasına dokunulmadı. Hiçbir dosya silinmedi.

### KAPANIŞ KARARI (P1 madde 7 tamamlandı)

**Karar: `depth=5` (`final_model_bundle_v2.pkl`) resmi final olarak
korundu.** `depth=3` adayı (`final_model_bundle_v3.pkl`) yalnızca
resmi-olmayan bir metrikte (ham F1) istatistiksel olarak anlamlı çıktı;
projenin resmi karar metriğinde (weighted-F1) anlamsız ve nominal olarak
`depth=5` lehine, specificity'de de `depth=5` lehine anlamlı. Ham-F1
kazancı specificity kaybıyla eşleniyor — E1'in eşik/önsel-kaynaklı
F1-specificity takasının bir başka örneği. `v3` dosyası silinmedi,
incelenebilir durumda diskte kalıyor.

**Doğrulama (bu kapanış turunda yapıldı):** `final_model_bundle_v2.pkl`
ile arşiv kopyası (`final_model_bundle_ARCHIVED_pre_objective_fix.pkl`)
byte-bit özdeş (`cmp` ile doğrulandı) — arşivleme sırasında hiçbir
bozulma/kayıp olmadı. `predict.py`'nin `DEFAULT_BUNDLE_PATH`'i hâlâ
`final_model_bundle_v2.pkl`'i gösteriyor (madde 7 süreci boyunca hiç
değişmedi). Bundle yeniden yüklenip (`depth=5`, eşik=0,35, kalibratör=
Beta, 26 özellik) bozulmadığı doğrulandı.

**P1 madde 7 TAMAMLANDI.**

---

## P1 Madde 8: Kalibrasyon Yöntem-Seçiminin Fold-İçine Taşınması

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P1 madde 8.
> **E3'ün orijinal `e3_calibration_metrics.csv`'si değiştirilmedi**
> (üzerine yazılmadı) — bu bölüm E3'ün development kaydını yeni,
> ayrı dosyalarla düzeltiyor. Final model (F0) hattı bu maddeden
> **etkilenmiyor** (`f0_final_model.py` zaten kendi OOF'unda Platt/
> Beta'yı bağımsız karşılaştırıp seçiyor).

### Yöntem

Her dış-fold için, **yalnızca o fold'un kendi iç çapraz-fit örnekleminde**
(`cross_fit_probabilities` — E3'ün zaten kurduğu, dış-train'in iç 4-fold'
undan üretilen, dış-teste hiç erişmeyen örneklem) Platt ve Beta ayrı ayrı
fit edilip Brier'e göre karşılaştırıldı (Isotonic atlandı — aşırı-uyumu
E3'te defalarca doğrulanmıştı); kazanan **o fold için** seçilip dış-teste
yalnızca `.transform()` ile uygulandı. Sızıntı testiyle doğrulandı
(`tests/test_e3_fold_local_calibrator_pah.py`, E5/E6 ile aynı desen):
dış-test'in içeriği tamamen değiştirilse bile seçilen yöntem AYNI kalıyor.

### 1) Platt/Beta kaç fold'da kazandı

| Model | Platt | Beta | ECE (ham) | ECE (fold-içi kalibre) |
|---|---|---|---|---|
| CatBoost/v1 | 10/50 | **40/50** | 0,0437 | 0,0244 |
| CatBoost/v4_from_v2 | 1/50 | **49/50** | 0,0339 | 0,0111 |
| LightGBM/v1 | 8/50 | **42/50** | 0,0984 | 0,0146 |
| RandomForest/v4_from_v2 | 9/50 | **41/50** | 0,0555 | 0,0139 |

Dört modelin dördünde de fold-içi kalibrasyon, ham olasılıklara göre ECE'yi
belirgin şekilde düşürüyor (en büyük iyileşme LightGBM/v1'de: 0,098→0,015).

### 2) "Beta üçünde de kazandı" iddiası — nüanslı hâli

Fold-düzeyinde bakıldığında Beta, **dört modelin dördünde de** fold'ların
büyük çoğunluğunda kazanıyor (%80-98 arası) — bu, CatBoost/v1 için bile
geçerli. Bu, E3'ün orijinal **global** bulgusuyla ("üç adayın Brier
ORTALAMASI karşılaştırıldığında CatBoost/v1'de Platt daha iyi çıktı")
yüzeysel olarak çelişiyor gibi görünse de aslında iki farklı soruya cevap
veriyor:

- **Eski (global) soru:** "50 fold'un TAMAMINA TEK bir yöntem
  uygulasaydık, hangisinin ORTALAMA Brier'i daha düşük olurdu?" →
  CatBoost/v1'de cevap Platt'tı (birkaç fold'da Platt'ın büyük
  kazançları, ortalamayı domine ediyor olabilir).
- **Yeni (fold-içi) soru:** "HER fold kendi başına hangisini seçerdi?" →
  CatBoost/v1 dahil her modelde çoğunluk Beta.

Bu iki bulgu birbirini **geçersiz kılmıyor** — global ortalama bir
yöntemin birkaç fold'da büyük kazanç sağlayıp çoğu fold'da hafif kaybettiği
bir senaryoyu gizleyebilir; fold-içi seçim bu heterojenliği ortaya
çıkarıyor. Sonuç: **"Beta üçünde de kazandı" ifadesi artık "Beta, dört
adayın dördünde de fold'ların çoğunluğunda (%80-98) tercih ediliyor, ama
her fold'da değil — global ortalama bazen azınlıktaki Platt-tercihli
fold'lardan etkilenebiliyor" şeklinde daha nüanslı okunmalı.**

### 3) Reliability diagram

`reports/figures/reliability_diagram_fold_local_calibration.png` —
4 modelin fold-içi kalibre edilmiş dış-test olasılıklarının reliability
eğrileri, mükemmel-kalibrasyon referans çizgisiyle birlikte.

### Kod / dosyalar

- `src/genova/pah/e3_fold_local_calibrator.py` — fold-içi kalibratör
  seçimi + ECE + reliability diagram; `e3_calibration_run.py`'nin
  `cross_fit_probabilities`/`_lookup_best_params`/`_variant_ids_in_row_
  order`'ını değiştirmeden yeniden kullanıyor.
- `tests/test_e3_fold_local_calibrator_pah.py` — 5 test (imza kontrolü,
  düşük-Brier seçimi, sızıntı testi [E5/E6 deseni], ECE'nin kusursuz/
  bozuk kalibrasyonda doğru değeri verdiği).
- `reports/tables/e3_fold_local_calibrator_choice.csv` (yeni — fold/model
  başına seçilen yöntem + ECE), `reports/figures/reliability_diagram_
  fold_local_calibration.png` (yeni).

`e3_calibration_metrics.csv`/`e3_calibrated_oof_predictions.csv` (E3'ün
orijinal dosyaları) değiştirilmedi. Split bankasına, `final_model_bundle_
v2.pkl`'e, `predict.py`'ye, `f0_final_model.py`'ye dokunulmadı.

**P1 madde 8 TAMAMLANDI.**

---

## P1 Madde 9: A/B/C Ağırlıklandırmasının Tüm Adaylara Yayılması

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P1 madde 9.
> Amaç yeni bir final aday bulmak değil, XGBoost/LightGBM/Elastic-Net'in
> yalnızca Strateji B ile elenmesinin adil olup olmadığını doğrulamak.
> Yeni hiperparametre araması yapılmadı — `e2_model_comparison.csv`'nin
> zaten kaydettiği (Strateji B) `best_params` aynen yeniden kullanıldı.

### 1) A/B/C sonuçları (ortalama F1, 50 dış-fold)

| Model/versiyon | A (ağırlıksız) | B (sabit×2,0) | C (veri-güdümlü SPW) | En iyi |
|---|---|---|---|---|
| XGBoost/v1 | **0,9199** | 0,9186 | 0,8696 | A |
| LightGBM/v1 | **0,9220** | 0,9199 | 0,9174 | A |
| Elastic-Net/v3 | **0,9077** | 0,8989 | 0,8184 | A |
| Elastic-Net/v4_from_v3 | **0,9091** | 0,9001 | 0,7864 | A |

Dört kombinasyonun dördünde de **A (ağırlıksız)** kazandı — C (veri-güdümlü
SPW) her seferinde en kötüsü. Bu, RF/CatBoost'ta daha önce görülen deseni
tekrarlıyor değil, tam tersi bir yön: bu dört model için ekstra
ağırlıklandırma karmaşıklığının F1'e hiçbir katkısı yok, hatta zararlı.
KNN, `sample_weight` desteği olmadığı için atlandı (zaten E2-EK'te MCC=0,07
ile çok zayıftı) — bu bir sınırlama olarak not düşülüyor, zorla uydurulmadı.

### 2) Kritik soru — elenen adaylar şimdi rekabetçi mi?

**Tasarım notu (görev metninden bilinçli bir sapma, şeffaf belgelendi):**
Görev, karşılaştırmayı adayın ham F1'i ile final adayın **weighted-F1**'i
arasında yapmayı istiyordu. Ama bu iki metrik aynı ölçekte değil (ham F1
eğitim-önseli altında tipik ~0,90 civarı; weighted-F1, kalibrasyon+önsel
düzeltmesi+uyarlanabilir eşik SONRASI tipik ~0,62 civarı — final adayın
kendisi: ham F1=0,9266, weighted-F1=0,6187). Bu karşılaştırmayı NB'ye
doğrudan vermek, **her adayı** yalnızca ölçek farkından dolayı sahte-anlamlı
"üstün" çıkarırdı — nitekim ikincil/bilgi-amaçlı olarak hesaplandı ve
**dördü de p≈0,0000** çıktı (ölçek uyumsuzluğunun doğrudan kanıtı, gerçek
bir üstünlük değil). Bunu karar için kullanmak, P1'in tüm ruhuyla (NB'yi
zayıf/yanlış sinyalleri elemek için zorunlu kriter yapmak) taban tabana
zıt olurdu.

Bunun yerine **birincil karşılaştırma**, iki tarafta da AYNI metriği
kullandı: ham F1 (eşik=0,5, kalibrasyonsuz) — bu, adayların **gerçekten
elendiği** E2 ölçütünün ta kendisi, dolayısıyla "eleme adil miydi"
sorusuna geçerli bir cevap.

| Model/versiyon | En iyi varyant | Fark (aday−final, ham F1) | NB p-değeri | Aday lehine anlamlı mı |
|---|---|---|---|---|
| XGBoost/v1 | A | −0,0067 | 0,4322 | Hayır |
| LightGBM/v1 | A | −0,0045 | 0,6113 | Hayır |
| Elastic-Net/v3 | A | −0,0188 | 0,0680 | Hayır |
| Elastic-Net/v4_from_v3 | A | −0,0175 | 0,0795 | Hayır |

### 3) Karar

**Dördünde de p≥0,05 — hiçbiri final adaya karşı istatistiksel olarak
anlamlı bir üstünlük göstermiyor.** Önceden belirlenmiş karar kuralına
göre: **mevcut eleme kararı sağlam. Final model değişmiyor.** XGBoost'un
(ve LightGBM/Elastic-Net'in) elenmesi, model ailesi farkıyla
ağırlıklandırma farkının karıştığı bir adaletsizlik değildi — en iyi
ağırlıklandırma varyantlarıyla bile final adayı geçemiyorlar.

### Kod / dosyalar

- `src/genova/pah/e2_madde9_weighting_extension.py` — A/C ağırlıklandırma
  (mevcut `best_params`, yeni arama yok) XGBoost/LightGBM/Elastic-Net'e;
  `reports/tables/e2_madde9_weighting_extension.csv` (yeni, `e2_model_
  comparison.csv` değiştirilmedi).
- `src/genova/pah/e2_madde9_nb_decision.py` — NB karar otomasyonu (birincil
  + ikincil/bilgi-amaçlı karşılaştırma); `reports/tables/e2_madde9_nb_
  decision.csv`.

Split bankasına dokunulmadı; `final_model_bundle_v2.pkl`/`predict.py`
değiştirilmedi.

**P1 madde 9 TAMAMLANDI.**

---

## P2 Madde 16: CatBoost Izgarasının Genişletilmesi

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P2 madde 16.
> Madde 7'nin iki dersi burada da uygulandı: (1) doğru ölçüt (önsel-
> düzeltilmiş + uyarlanabilir eşikli weighted-F1, eşik=0,5 ham F1 değil),
> (2) herhangi bir yeni aday kazanırsa tekrarlı-bölme teşhisi + NB
> zorunlu-kriteri, tek bölünmeye güvenilmeden.

### 1) Genişletilmiş ızgarada hangi aday kazandı

`depth∈{3,5} × learning_rate∈{0,03; 0,05; 0,1}` = 6 aday, `iterations=100`
sabit + `early_stopping_rounds=20` (eğitim verisinin içinden ayrılan
%15'lik stratified bir dilimle, fold-güvenli), tek 4-fold bölünmede,
önsel-ağırlıklı iç-F1 ölçütüyle:

| Aday | Önsel-ağırlıklı iç-F1 |
|---|---|
| depth=3, lr=0,03 | 0,6886 |
| depth=3, lr=0,05 | 0,6700 |
| depth=3, lr=0,1 | 0,6878 |
| depth=5, lr=0,03 | 0,6865 |
| **depth=5, lr=0,05** | **0,6945** |
| depth=5, lr=0,1 | 0,6814 |

**Mevcut model (`depth=5, lr=0,05`) 6 adayın en iyisi olarak kazandı.**

### 2) Mevcut modelle aynı mı farklı mı

**Aynı.** Genişletilmiş ızgara (3 kata çıkan `learning_rate` çeşitliliği
+ early stopping) mevcut hiperparametre seçimini değiştirmedi.

### 3) Tekrarlı-bölme + NB testi

**Gerekmedi** — Adım 2'nin kazananı mevcut modelle aynı çıktığı için görev
tanımı gereği burada durulu.

### 4) Nihai durum

**Mevcut model (`final_model_bundle_v2.pkl`, `depth=5, lr=0,05`) korundu
— değişiklik yapılmadı.** Bu, modelin yalnızca 2 adaylık dar bir ızgaraya
karşı değil, `learning_rate`'in de arandığı ve early stopping'in dahil
edildiği daha geniş bir aday havuzuna karşı da sağlam olduğunu doğruluyor.

### Kod / dosyalar

- `src/genova/pah/f0_madde16_grid_search.py` — genişletilmiş ızgara +
  early-stopping'li CatBoost fit fonksiyonu + (gerekirse) tekrarlı-bölme/
  NB otomasyonu (bu turda tetiklenmedi).
- `reports/tables/f0_madde16_grid_search.csv` (6 aday × skor).

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı.

**P2 madde 16 TAMAMLANDI.**

---

## P2 Madde 17: RF Eksik-Değer Doldurma Adaleti

> Kaynak: `reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`, P2 madde 17.
> RF final model değil (CatBoost kazandı) — bu görev bir final model
> kararı değiştirmiyor, yalnızca E2'nin karşılaştırma kaydının adilliğini
> netleştiriyor.

### Yöntem

`dense_numeric_features.py`'ye yeni bir fonksiyon eklendi:
`to_dense_numeric_al_zero_fill` (mevcut `to_dense_numeric` **değiştirilmedi**)
— `AL_` kolonlarındaki `NaN`'ı, eski TRAIN-medyanı yerine **sıfır** ile
dolduruyor (`v2`'nin kendi `ConstantFillImputer(strategy="zero")`
stratejisiyle aynı); `EK_` kolonları yine medyanla dolduruluyor. RF/v1 ve
RF/v2_raw_nan, mevcut `best_params` yeniden kullanılarak (yeni arama yok),
50 dış-fold'da bu yeni doldurmayla yeniden ölçüldü.

### Sonuç — eski (medyan) vs yeni (sıfır) doldurma

| Versiyon | Doldurma | F1 | MCC | Specificity |
|---|---|---|---|---|
| v1 | medyan (eski) | 0,9083±0,028 | **0,0381±0,099** | 0,0197±0,037 |
| v1 | **sıfır (yeni)** | 0,9187±0,028 | **0,3340±0,166** | 0,2093±0,112 |
| v2_raw_nan | medyan (eski) | 0,9090±0,028 | **0,0614±0,113** | 0,0272±0,040 |
| v2_raw_nan | **sıfır (yeni)** | 0,9173±0,028 | **0,3124±0,146** | 0,1888±0,106 |

**Sıfır-doldurma, MCC'yi ~5-9× artırıyor** (0,038→0,334 ve 0,061→0,312),
specificity'yi ~7-10× artırıyor; F1'de de küçük bir iyileşme var
(0,908→0,919 civarı).

### Sonuç: doldurma-artefaktı mı, gerçek kısıt mı?

**Büyük ölçüde doldurma-artefaktıydı.** RF'nin E2-EK'te görülen çöküşü
(MCC≈0,04-0,06), gerçekten büyük ölçüde adaletsiz bir karşılaştırmanın
sonucuydu — CatBoost/XGBoost/LightGBM `AL_`'de hiç doldurma görmezken,
RF'e medyan-doldurma dayatılmıştı. Sıfır-doldurmayla RF'nin MCC'si
`v2`/`v4_from_v2`'deki rekabetçi seviyesine (≈0,31-0,33, önceki ölçülen
0,36-0,39'a yakın) yaklaşıyor.

**Ama tamamen değil, kısmi bir kısıt da var:** düzeltilmiş MCC (≈0,31-0,33)
hâlâ final aday CatBoost/v4_from_v2'nin MCC'sinin (≈0,39) altında — yani
RF'nin bir miktar geride kalması yalnızca doldurma stratejisinden değil,
muhtemelen native-NaN'ın taşıdığı ek bilgiden (hangi değerlerin eksik
OLDUĞU bilgisi, RF'in dense-matris temsilinde `al_all_missing` göstergesi
dışında kayboluyor) de kaynaklanıyor. **Bu görev bir final model kararı
değiştirmiyor** (final aday zaten CatBoost) — yalnızca E2'nin eski
karşılaştırmasının kısmen adaletsiz olduğunu belgeliyor.

**KNN notu:** KNN de aynı adalet sorununu taşıyor (`to_dense_numeric`
üzerinden medyan-doldurma) ama zaten en zayıf aday olduğu için
(MCC=0,07) bu turda yeniden ölçülmedi, önceliklendirilmedi.

### Kod / dosyalar

- `src/genova/pah/dense_numeric_features.py` — yeni fonksiyon
  `to_dense_numeric_al_zero_fill` eklendi (mevcut `to_dense_numeric`
  değiştirilmedi); `tests/test_dense_numeric_features_pah.py`'ye 3 yeni
  test (sıfır-doldurma AL_'de, medyan EK_'te korunuyor, NaN-siz/float).
- `src/genova/pah/e2_madde17_rf_imputation_fairness.py` — RF/v1,v2_raw_nan
  yeniden ölçümü; `reports/tables/rf_imputation_fairness_check.csv`
  (eski+yeni, karşılaştırmalı).

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı; `e2_model_comparison.csv` değiştirilmedi.

**P2 madde 17 TAMAMLANDI.**

---

## Aşama F1: Adversarial Validation (Provenance/Kısayol Testi) — ⏸ ONAY BEKLİYOR

> Kapsam notu: F1'in orijinal hedefi (`al_all_missing`/`CAT_1`'in
> provenance riski taşıyıp taşımadığı) madde 11'de bu ikisi zaten
> çıkarıldığı için kısmen aşıldı — asıl soru artık **"final 26-özellik
> havuzunda (post-madde-11) başka bir provenance sinyali kaldı mı?"**
> oldu. Cevap: **evet, buldu.**

### 1) Kaynak-1/2 testlerinin AUC'leri

Yöntem notu: veri setinde açık bir "ClinVar" bayrağı yok — Kaynak-1,
`CAT_1`'in dolu (gnomAD popülasyon-kanıtı var) vs boş (gnomAD-kanıtı yok,
ClinVar-ağırlıklı bir alt-küme için makul bir **proxy**, kesin değil)
olmasına dayanıyor; bu açıkça bir proxy olarak belirtiliyor. Her iki test
de yalnızca **benign sınıf içinde** (n=61), yardımcı bir LightGBM
sınıflandırıcıyla (final model AİLESİ DEĞİL), grup-farkında 5-fold CV ile:

| Test | n (poz/neg) | CV AUC |
|---|---|---|
| Kaynak-1 (`CAT_1` dolu vs boş, proxy) | 61 (39/22) | **1,0000 ± 0,0000** |
| Kaynak-2 (`CAT_2`/AllofUs dolu vs boş) | 61 (20/41) | 0,5810 ± 0,2058 |

`CAT_2` havuzda **girdi olarak yok** — yalnızca dışlanmış hedef değişken
olarak kullanıldı. Kaynak-2'de sinyal yok (AUC≈0,58, gürültü düzeyinde);
**Kaynak-1'de mükemmel ayrım** — final havuzun (26 özellik) hâlâ `CAT_1`
provenance'ını neredeyse kusursuz tahmin edebildiğini gösteriyor.

### 2) Özellik-bazlı tarama — `>0,75` geçen 4 özellik bulundu

26 özelliğin tek-değişkenli adversarial AUC'si (Kaynak-1/2'ye karşı, yön-
bağımsız):

| Özellik | AUC (Kaynak-1) | AUC (Kaynak-2) |
|---|---|---|
| `AL_26` | **1,0000** | 0,655 |
| `AL_12` | **0,9487** | 0,651 |
| `AL_7` | **0,9487** | 0,703 |
| `AL_49` | **0,7692** | 0,734 |
| *(diğer 22 özellik)* | ≤0,667 | ≤0,733 |

**Mekanizma (doğrulandı):** Bu 4 kolonun **ham** (doldurmadan önceki)
NaN deseni, benign alt-kümede `CAT_1` dolu olan **hiçbir** satırda
görülmüyor (`n_finite=0` o grupta) — yani bu kolonlar, `CAT_1` boşken
**sistematik olarak** da boş, `v2`'nin sıfır-doldurmasından sonra bu
"boşluk" `0,0` değerine dönüşüp neredeyse mükemmel bir kaynak-göstergesi
haline geliyor. Bu, allel-frekansı biyolojisinden değil, veri
toplama/pipeline'ından kaynaklanan klasik bir imza.

### 3) Ablasyon maliyeti — DURDUR ile öneri

Bu 4 özellik çıkarılıp (22 özellik kalan havuz, `reports/tables/f1_final_
feature_pool_v3.json`) madde 11'in **aynı** disipliniyle (Adım 2-6 yeniden
çalıştırıldı, `f0_final_model.py` değiştirilmeden) bir aday üretildi
(`models/pah/final_model_bundle_f1_ablation_candidate.pkl`, **resmi
değil**, yalnızca inceleme). Sonuç: `depth=3, lr=0,05` seçildi (eskisi
`depth=5`), eşik=0,24 (eskisi 0,35).

Madde 10'un araçlarıyla karşılaştırma (`reports/tables/f1_ablation_
comparison.csv`):

| Metrik | Eski (26 özellik) | Yeni (22 özellik) | CI örtüşüyor mu | NB p |
|---|---|---|---|---|
| F1 | 0,8177 [0,78;0,85] | 0,8043 [0,77;0,84] | ✅ | 0,570 |
| MCC | 0,3921 | 0,2826 | ✅ | 0,196 |
| Specificity | 0,7869 | 0,6393 | ✅ | 0,074 |
| Monte Carlo F1 (100/250) | 0,6395 | 0,5496 | ✅ (dar) | — |

**Bu, madde 11'inki kadar temiz bir "bedava" sonuç DEĞİL.** CI'ler
teknik olarak örtüşüyor ve NB hiçbir metrikte formel anlamlılığa
ulaşmıyor (k=5'in gücü düşük) — ama madde 11'in tam tersine, **dört
metrik de aynı yönde** (düşüş) hareket ediyor, specificity'de neredeyse
anlamlılık sınırında (p=0,074). Bu, "istatistiksel olarak ayırt
edilemez ama nominal olarak tutarlı bir maliyet var" durumu — net bir
"maliyetsiz" veya "maliyetli" etiketi yapıştırmak yanıltıcı olur.

**Karar otomatik verilmedi — DURDURULDU.** `final_model_bundle_v2.pkl`
değiştirilmedi, `predict.py` güncellenmedi. **Kullanıcı tercihi
gerekiyor: provenance riskini kapatmak için nominal (ama istatistiksel
olarak kanıtlanmamış) bir performans maliyetini kabul etmek mi, yoksa
mevcut performansı korumak mı?**

### 4) Aşama D ile tutarlılık

`reports/tables/univariate_auc_scan.csv`'de bu 4 özelliğin **Label'a
karşı** AUC'si düşük (0,53-0,66) — Aşama A'nın taramasında "şüpheli"
(>0,90) işaretlenmediler. Bu **çelişki değil, tamamlayıcı**: Aşama A
yalnızca özellik↔Label ilişkisine baktı, bu tur özellik↔kaynak
ilişkisine baktı — ikisi farklı sorular. Bu 4 özellik LABEL'ı doğrudan
sızdırmıyor, ama final test setinde kaynak-dağılımı eğitimdekinden
farklıysa (gayet olası — final set muhtemelen farklı bir küratasyon
sürecinden geçecek), bu özellikler üzerinden dolaylı bir dağılım-kayması
riski taşıyorlar.

### 5) `03b`'nin nihai kapanışı

`03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'ye "NİHAİ DEĞERLENDİRME"
bölümü eklendi: toplam kanıt (Cramér's V, Fisher testleri, kısmi
korelasyon, madde 11'in maliyetsiz-çıkarma bulgusu, ve bu turun
neredeyse-mükemmel adversarial AUC bulgusu) **pipeline/kaynak-provenance
artefaktı hipotezini baskın hipotez olarak destekliyor** — saf biyolojik
hipotez büyük ölçüde çürütülmüş sayılır, %100 sıfırlanmamış (kısmi
korelasyon bulgusu küçük bir bağımsız sinyal ihtimalini açık bırakıyor).

### Kod / dosyalar

- `src/genova/pah/f1_adversarial_validation.py` — Kaynak-1/2 testleri +
  özellik-bazlı tarama; `reports/tables/f1_adversarial_source_cv_auc.csv`,
  `f1_adversarial_per_feature_auc.csv`.
- `src/genova/pah/f1_ablation_candidate.py` — 22-özellik adayı üretir
  (madde 11 deseniyle aynı); `reports/tables/f1_final_feature_pool_v3.json`,
  `models/pah/final_model_bundle_f1_ablation_candidate.pkl` (resmi DEĞİL).
- `src/genova/pah/f1_ablation_comparison.py` — madde 10 araçlarıyla
  karşılaştırma; `reports/tables/f1_ablation_comparison.csv`.

Split bankasına dokunulmadı. `final_model_bundle_v2.pkl`/`predict.py`
değiştirilmedi — onay bekleniyor.

### F1 Ek — Tekrarlı-Bölme Ablasyon Testi (KESİN SONUÇ: gerçek maliyet var)

Tek bir ablasyon karşılaştırmasının belirsizliğini (CI örtüşüyor, NB
anlamsız ama tüm metrikler tutarlı düşüyor) çözmek için madde 7'nin
**aynı** deseni (10 farklı rastgele `StratifiedKFold` bölünmesi,
`i=0..9`) uygulandı — her tekrarda hem 26-özellik (mevcut) hem 22-özellik
(4 provenance-riskli özellik çıkarılmış) için Adım 2 (hiperparametre,
düzeltilmiş önsel-ağırlıklı ölçüt) + Adım 3 (cross-fit OOF → Beta →
SLD → eşik) yeniden çalıştırıldı.

**Sonuç: kararsızlık değil, tam tersi — çok net ve tutarlı bir fark.**

| | Değer |
|---|---|
| 26-özellik kazandı | **10/10** |
| 22-özellik kazandı | 0/10 |
| Yön-işaretli ortalama marj (22−26) | **−0,0877 ± 0,0272** |

26-özellik (mevcut), **10 tekrarın 10'unda da**, önemli ve tutarlı bir
farkla (~0,088 önsel-ağırlıklı F1) kazandı — önceki tek-karşılaştırmanın
"belirsiz" görünmesi, o karşılaştırmanın kullandığı tek (k=5) bölünmenin
düşük istatistiksel gücünden kaynaklanan bir yanılsamaymış, gerçek fark
küçük değilmiş.

**Karar kuralına göre: GERÇEK BİR MALİYET VAR.** `AL_26`/`AL_12`/`AL_7`/
`AL_49`, madde 11'in `al_all_missing`/`CAT_1`'inin aksine, **yalnızca**
provenance-kısayolu değil — eğitim dağılımında ölçülebilir, tutarlı bir
gerçek tahmin gücü de taşıyorlar (kaynağı ne olursa olsun). Bu iki
olasılıkla tutarlı: (a) bu 4 kolon gerçek biyolojik sinyal taşıyor ve
kaynak-korelasyonu tesadüfi bir yan etki, veya (b) model bu "kısayolu"
eğitim setinde gerçekten ağır kullanıyor ve final test setinde
(kaynak-dağılımı farklıysa) bu performans avantajı **kaybolabilir** —
ikisi arasında bu testle ayrım yapılamıyor.

**`final_model_bundle_v2.pkl` DEĞİŞTİRİLMEDİ, `predict.py` GÜNCELLENMEDİ
— karar kullanıcıya bırakıldı.** Seçenekler: (1) mevcut modeli (26
özellik, ölçülen performans) koru, provenance riskini bilinçli olarak
kabul et — çünkü final test setinin gerçek kaynak-dağılımı bilinmiyor,
riskin gerçekleşip gerçekleşmeyeceği belirsiz; (2) 22-özellik modelini
(ölçülen ~0,09 performans kaybı ile) benimse, riski tamamen ortadan
kaldır.

### Kod / dosyalar (F1 Ek)

- `src/genova/pah/f1_madde_stability_check.py` — 10 tekrarlı-bölme testi;
  `reports/tables/f1_madde_stability_check.csv`.

**Nihai karar** (model korundu, risk bilinçli kabul edildi, gerekçesiyle)
`03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'nin "Nihai Karar (F1
sonrası)" bölümünde belgelendi.

Split bankasına dokunulmadı.

---

## Aşama F3: Sağlamlık Stres Testleri — ⚠️ KIRILGANLIK BULUNDU, F1'İN KARARI YENİDEN GÖZDEN GEÇİRİLMELİ

> F1'in "risk bilinçli kabul edildi, F3'te izlenmeli" notunun takibi.
> **Sonuç izlemeyle kalmadı — ciddi bir kırılganlık buldu.**

### 1) Benign-ağırlıklı stres taraması — düzgün, alarm yok

| Benign oranı | Ortalama F1 | %95 aralık |
|---|---|---|
| %50,0 | 0,7446 | [0,6946; 0,7906] |
| %60,0 | 0,7052 | [0,6498; 0,7581] |
| %70,0 | 0,6487 | [0,5815; 0,7105] |
| **%71,4 (şartname)** | 0,6395 | [0,5738; 0,7000] |
| %80,0 | 0,5591 | [0,4859; 0,6304] |

Düzgün, monoton bir eğri — beklenen (benign oranı artınca F1'in payda
tarafındaki FP baskısı artar) davranış, ani bir çöküş yok. **Bu adım tek
başına alarm vermiyor.**

### 2) `CAT_2` (AllofUs) kaynak-geçişi — orta düzeyde bozulma

| Yön | n_test (benign) | F1 | MCC | raw AUC |
|---|---|---|---|---|
| train=değil-AllofUs → test=AllofUs | 144 (20) | 0,638 | 0,198 | 0,719 |
| train=AllofUs → test=değil-AllofUs | 225 (41) | 0,800 | 0,258 | 0,727 |

Normal CV'ye (F1=0,818, raw AUC=0,831) göre orta düzeyde düşüş (~0,10-0,18
F1, ~0,10 AUC) — endişe verici ama `>0,15` eşiğini yalnızca ilk yönde
sınırda geçiyor, tek başına "kırılgan" kararını tetiklemiyor.

### 3) Eksiklik-deseni holdout (`al_all_missing=1`) — sürpriz: yüksek kalıyor

| n_test (benign) | F1 | MCC | raw AUC |
|---|---|---|---|
| 89 (2) | 0,885 | 0,282 | 0,963 |

**Küçük örneklem uyarısı:** yalnızca 2 benzersiz benign satır — bu
sonucun genellenebilirliği çok sınırlı (aynı uyarı `e2ek_provenance.py`
Tier 3 bulgusunda da vardı). Şaşırtıcı biçimde F1/AUC burada **düşmedi**
— ama bu, yalnızca 2 benign örneğe dayandığı için güvenilir bir "sağlam"
kanıtı sayılmamalı.

### 4) `CAT_1` doluluk-geçişi — 🔴 KRİTİK ÇÖKÜŞ

**En kritik test, ve en çarpıcı sonucu verdi.**

| Yön | n_test (benign) | F1 | MCC | raw AUC |
|---|---|---|---|---|
| train=CAT_1 dolu → test=CAT_1 boş | 132 (22) | 0,930 | 0,630 | 0,859 |
| **train=CAT_1 boş → test=CAT_1 dolu** | **237 (39)** | **0,059** | **0,072** | **0,573** |

**Normal CV referansı:** F1=0,818, raw AUC=0,831.

İkinci yönde **F1, 0,818'den 0,059'a çöküyor (mutlak düşüş 0,759)** —
görevin `>0,15` eşiğinin **~5 katı**. Bu bir kalibrasyon/eşik artefaktı
değil: **eşiksiz, kalibrasyondan tamamen bağımsız raw AUC de** 0,831'den
**0,573'e** (rastgele-tahmine yakın) çöküyor — doğrulama olarak ayrıca
ölçüldü. Mekanizma F1'in bulgusuyla tam tutarlı: `AL_26/12/7/49`,
`CAT_1`-boş eğitim alt-kümesinde (132 satır) **sabit (=0)** olduğu için
model bu 4 özelliği kullanmayı hiç öğrenemiyor; `CAT_1`-dolu test
kümesinde bu özellikler gerçek (sıfırdan farklı) değerler aldığında
model bunlarla ne yapacağını bilmiyor. **Yön asimetrik** — ters yönde
(zengin kümeden fakire) sorun yok, hatta normal CV'den bile iyi (F1=0,930)
— asıl kırılganlık, eğitim verisinin bu 4 özellikte **çeşitlilik
içermemesi** durumunda ortaya çıkıyor.

### 5) Sentez — karar kuralına göre

**Beş testten biri (Adım 4) `>0,15` eşiğini çok büyük bir farkla aştı
(0,759, eşiğin ~5 katı).** Karar kuralına göre: **model KIRILGAN, F1'in
kararı (riski bilinçli kabul et, modeli koru) YENİDEN GÖZDEN
GEÇİRİLMELİ.**

**`final_model_bundle_v2.pkl`/`predict.py` bu turda DA değiştirilmedi
— DURDURULDU, kullanıcıya bildiriliyor.** F1'in ölçtüğü 22-özellik
alternatifi (o zamanki ölçülen maliyet: ~0,09 weighted-F1, madde 10'un
araçlarıyla) yeniden gündeme gelmeli — F3'ün bu bulgusu, o maliyetin
karşısına artık çok daha somut bir risk koyuyor: eğer final test setinin
kaynak-kompozisyonu eğitim setinden farklıysa (ki bu tam olarak
beklenen bir durum — final set muhtemelen farklı bir küratasyon
sürecinden geçecek), model **rastgele tahminin biraz üzerinde bir
performansa** düşebilir, yalnızca ~0,09'luk bir marjinal kayıp değil.

### Kod / dosyalar

- `src/genova/pah/f3_robustness_stress_tests.py` — 4 stres testi +
  sentez; `reports/tables/f3_benign_weighted_scan.csv`,
  `f3_source_shift_tests.csv`.

Split bankasına dokunulmadı; `final_model_bundle_v2.pkl`/`predict.py`
değiştirilmedi. Hiçbir kalıcı model üretilmedi (tüm yeniden-fit'ler
yalnızca bu testin geçici, bellekte kalan modelleriydi).

---

## F3 Sonrası Karar Matrisi (26 vs 23 vs 22 Özellik) — ⚠️ BEKLENMEDİK BULGU

> F3'ün bulduğu `CAT_1` çöküşünün, gerçekten `AL_26/12/7/49`'dan mı
> kaynaklandığını, yoksa daha derin bir sorunu mu yansıttığını doğrudan
> sınayan takip görevi. **Sonuç, "bu 4 özelliği çıkarırsak sorunu
> çözeriz" varsayımını çürüttü.**

### Üç aday

- **A (mevcut, 26 özellik):** referans.
- **B (23 özellik):** `AL_26, AL_12, AL_7` çıkarıldı, `AL_49` (daha zayıf
  kaynak-sinyali, AUC=0,77, ama Faz 1/3'te bağımsız biyolojik sinyali de
  var) **tutuldu**. Yeni aday bundle üretildi (resmi değil).
- **C (22 özellik):** F1 Ek'in adayı, hepsi çıkarıldı (mevcut).

### 1) Çöküş testi — B de C de çöküşü ÇÖZMÜYOR

| Aday | Kritik yön (train=CAT_1-boş→test=CAT_1-dolu) raw AUC | Ters yön raw AUC |
|---|---|---|
| A (26) | 0,573 (F3'ten, referans) | 0,859 |
| B (23, `AL_49` tutuldu) | **0,5727** | 0,895 |
| C (22, hepsi çıkarıldı) | **0,4937** | 0,879 |

**Şaşırtıcı bulgu: üçü de kritik yönde neredeyse rastgele-tahmin
seviyesinde (~0,49-0,57) — B, A'dan HİÇ farklı değil; C ise A'dan bile
hafifçe daha kötü (0,494 < 0,573, rastgele tahminin altı).** Bu,
çöküşün `AL_26/12/7/49`'un doğrudan bir sonucu OLMADIĞINI, daha derin bir
mekanizmayı yansıttığını gösteriyor — en olası açıklama: `CAT_1`-boş
alt-kümesi (yalnızca 132 satır) zaten küçük ve muhtemelen başka
yönlerden de (bu 4 özellik dışında) `CAT_1`-dolu alt-kümesinden
(237 satır) sistematik olarak farklı bir alt-popülasyon — 3-4 özelliği
çıkarmak bu daha temel dağılım-farkını düzeltmiyor.

### 2) Ortalama maliyet (10-tekrarlı bölme + NB testi)

| Aday | 10-tekrar kazanma | Yön-işaretli marj | NB p-değeri (weighted-F1, sabit 5-fold) |
|---|---|---|---|
| B (23) | 1/10 | −0,0259 ± 0,0328 | **0,8264** (nominal olarak +0,0218, A'dan iyi ama anlamsız) |
| C (22) | 0/10 | −0,0877 ± 0,0272 | 0,1630 (nominal −0,0943) |

İki farklı ölçüm yöntemi (10-tekrarlı `StratifiedKFold` taraması vs sabit
5-fold NB testi) B için **yön konusunda bile anlaşmıyor** (biri hafif
negatif marj, diğeri hafif pozitif) — ikisi de istatistiksel olarak
anlamsız, bu da küçük örneklemde (369 satır) bu tür karşılaştırmaların
ne kadar gürültülü olduğunu bir kez daha gösteriyor.

### 3) Karar Matrisi

| Aday | Kritik yön AUC | 10-tekrar kazanma | Marj | NB p |
|---|---|---|---|---|
| **A (26)** | 0,573 | — | — | — |
| **B (23, `AL_49` tutuldu)** | 0,5727 | 1/10 | −0,0259 | 0,8264 |
| **C (22, hepsi çıkarıldı)** | 0,4937 | 0/10 | −0,0877 | 0,1630 |

**Yorum:** Görev tanımının öngördüğü "B çöküşü büyük ölçüde önlerse VE
C'den belirgin ucuzsa, B iyi bir orta yoldur" senaryosu **gerçekleşmedi**
— B çöküşü hiç önlemiyor (A ile pratikte aynı: 0,5727 vs 0,573).
Dolayısıyla mevcut üç aday arasında **provenance-riskini gerçekten
azaltan hiçbir seçenek yok** — yalnızca performans kaybı (B'de belirsiz/
küçük, C'de nominal ama anlamsız ~−0,09) satın alıp, ölçülen kırılganlığı
düzeltmeden kalıyoruz. Bu, F1/F3'ün orijinal "4 özelliği çıkar" çerçevesini
sorguluyor: sorun muhtemelen özellik-düzeyinde değil, **eğitim setinin
`CAT_1`-boş alt-kümesinin küçüklüğü/temsil gücü** düzeyinde.

**Hiçbir final karar verilmedi — üç seçenek ölçülmüş halde sunuluyor.**
`final_model_bundle_v2.pkl`/`predict.py` değiştirilmedi.

### Kod / dosyalar

- `src/genova/pah/f3_ablation_candidate_b.py` — aday B (23 özellik)
  üretir; `reports/tables/f3_final_feature_pool_b23.json`,
  `models/pah/final_model_bundle_f3_candidate_b23.pkl` (resmi DEĞİL).
- `src/genova/pah/f3_decision_matrix.py` — çöküş testi + tekrarlı-bölme
  + NB testi; `reports/tables/f3_decision_collapse_test.csv`,
  `f3_decision_stability_check.csv`, `f3_decision_nb_test.csv`.

Split bankasına dokunulmadı.

---

## Aşama F4: SHAP Açıklanabilirlik

> Saf açıklama/görselleştirme turu — final model (`final_model_bundle_
> v2.pkl`) hiçbir şekilde değiştirilmedi/yeniden eğitilmedi, yalnızca
> kendi CatBoost'u `shap.TreeExplainer` ile 369 satırın tamamında
> (final fit'in kendi eğitim verisi) açıklandı.

### 1) Global SHAP sıralaması — ilk 10 özellik

| Sıra | Özellik | Ortalama \|SHAP\| |
|---|---|---|
| 1 | `AL_7` | 0,4151 |
| 2 | `EK_7` | 0,3355 |
| 3 | `AL_300` | 0,2871 |
| 4 | `AL_49` | 0,1801 |
| 5 | `AL_330` | 0,1759 |
| 6 | `AL_26` | 0,1756 |
| 7 | `AL_301` | 0,1541 |
| 8 | `EK_9` | 0,1488 |
| 9 | `AL_329` | 0,1138 |
| 10 | `AL_12` | 0,1122 |

### 2) Riskli 4 özelliğin SHAP davranışı — **beklenenin tersine büyük katkı**

**Görev tanımının beklentisi** ("bu 4 özelliğin katkısı küçük olmalı,
F3'ün 'ana suçlu değiller' bulgusuyla tutarlı olarak") **doğrulanmadı —
tam tersi çıktı.** `AL_7`, **modelin en önemli tek özelliği** (sıra 1/26);
`AL_49` sıra 4, `AL_26` sıra 6 — üçü de üst-orta/üst dilimde. Yalnızca
`AL_12` (sıra 10/26) nispeten daha mütevazı.

**Ama `CAT_1`-doluluğuna göre kırılım beklenen deseni doğruluyor** (daha
ölçülü bir biçimde — "neredeyse sıfır" değil, ama tutarlı bir şekilde
**belirgin düşük**):

| Özellik | `CAT_1`-boş ort. \|SHAP\| | `CAT_1`-dolu ort. \|SHAP\| | Oran |
|---|---|---|---|
| `AL_49` | 0,0796 | 0,2361 | **~3,0×** |
| `AL_26` | 0,1099 | 0,2122 | ~1,9× |
| `AL_7` | 0,3347 | 0,4598 | ~1,4× |
| `AL_12` | 0,0829 | 0,1286 | ~1,6× |

**Yorum — F3'ün bulgusuyla nasıl bağlantılı:** SHAP'in gösterdiği "bu 4
özellik `CAT_1`-boş alt-kümede önemli ölçüde daha az kullanılıyor"
deseni, F3'ün "model bunları o alt-kümede öğrenemiyor (çünkü değerleri
orada zaten sıfıra yakın)" mekanizmasıyla **tam tutarlı**. Ama SHAP aynı
zamanda `AL_7`'nin (özellikle) genel olarak **çok** önemli olduğunu
gösteriyor — bu, F3'ün "4 özelliği çıkarmak çöküşü düzeltmedi, hatta
`C` adayı biraz daha kötüydü" bulgusunu **daha iyi açıklıyor**: model
gerçekten bu özelliklere (özellikle `AL_7`'ye) güçlü şekilde dayanıyor,
bu yüzden onları çıkarmak genel performansa zarar veriyor — ama
`CAT_1`-boş alt-kümesinin **kendisi** zaten bu özellikler dışında da
yeterince çeşitlilik taşımadığı için, çıkarmak kırılganlığı çözmüyor.
İki bulgu birlikte, F3'ün "kök neden özellik değil alt-küme yapısı"
sonucunu **güçlendiriyor**, zayıflatmıyor.

### 3) Yerel açıklama örnekleri

In-sample (369 satırın kendisi — SHAP'in standart pratiği): 330 doğru,
39 yanlış. 3 doğru + 2 yanlış örnek için waterfall grafiği üretildi
(`reports/figures/f4_shap_waterfall_*.png`). İlginç bir detay: yanlış
sınıflandırılan `VAR_003238`, projenin daha önce belgelenen
`conflict_group_1` (çelişkili-profil, giderilemez etiket belirsizliği
taşıyan grup) üyesi — bu, modelin bir hatası değil, verinin kendi
belirsizliğinin beklenen bir yansıması.

### 4) Aşama D ile tutarlılık — orta düzeyde, beklenen bir sapma ile

**Spearman sıra-korelasyonu: 0,3989** (orta düzey pozitif). Üst sırada
kısmi uyum var (`EK_7`/`AL_300` ikisinde de üst sıralarda), ama `AL_7`
SHAP'te #1 iken Aşama D'nin (P0-düzeltmeli, fold-lokal) stabilite
sıralamasında yalnızca #16 — çarpıcı bir sapma.

**Yorum:** Bu beklenen bir sonuç, çelişki değil — SHAP **model-spesifik**
(bu belirli CatBoost fit'inin gerçekte neyi kullandığını gösterir),
Aşama D'nin stabilite seçimi ise **yöntem-ortalaması** (4 bağımsız,
çoğunlukla model-agnostik yöntemin ortalaması). Bir özellik, genel
seçim yöntemlerinde orta düzey "stabil" çıkıp, yine de bu belirli
CatBoost fit'i tarafından ağır kullanılabilir (ağaç modelleri
etkileşim/eşik yapıları yakalayabilir, genel yöntemler bunu tam
yansıtmayabilir).

### Üretilen görseller

- `reports/figures/f4_shap_summary_beeswarm.png`, `f4_shap_summary_bar.png`
  — global özet.
- `reports/figures/f4_shap_risky_features_cat1_dependence.png` — riskli
  4 özelliğin `CAT_1`-doluluğuna göre renklendirilmiş dependence grafiği.
- `reports/figures/f4_shap_waterfall_dogru_*.png` (3), `f4_shap_waterfall_
  yanlis_*.png` (2) — yerel açıklama örnekleri.

### Kod / dosyalar

- `src/genova/pah/f4_shap_explainability.py`; `reports/tables/
  f4_shap_feature_importance.csv`, `f4_shap_vs_stage_d_comparison.csv`.

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı. Hiçbir model eğitilmedi.

---

## Revize: AUROC + Sensitivity Eklenmesi, Osman Hoca Formatında Birleşik Karar Tablosu

> Çapraz-panel karşılaştırması (`12_CAPRAZ_PANEL_KARSILASTIRMA_PAH.md`)
> üç boşluk bulmuştu: AUROC hiçbir model karşılaştırma tablosunda yoktu,
> Sensitivity E2-E6 çekirdeğinde eksikti, ve Osman Hoca'nın istediği
> formatta tek, birleşik bir karar tablosu yoktu. Bu bölüm üçünü de
> kapatıyor — **hiçbir hiperparametre/eşik yeniden aranmadı**, yalnızca
> kayıtlı ayarlarla (`best_params`, `chosen_threshold`) ya gerçek bir
> **replay** (tek bir yeniden fit, arama yok) ya da diskte zaten duran
> olasılıklardan **saf aritmetik** yapıldı.

### Yöntem — üç farklı maliyet seviyesi

1. **E1 baseline (50 fold) ve E2'nin "çekirdek 16" kombinasyonu (800
   satır, `reports/tables/e2_model_comparison.csv`):** gerçek replay —
   CSV'de zaten kayıtlı `best_params` okunup **aynen** kullanılarak
   dış-train'de tek bir fit yapıldı, dış-test'te AUROC (`roc_auc_score`)
   ve Sensitivity (eşik=0,5, E2'nin sabit rejimiyle aynı) hesaplandı.
   E2-EK'in RF/KNN/A-C ağırlıklandırma genişletmeleri (450 ek satır) bu
   turun **kapsamı dışında bırakıldı** — o satırlarda `auroc`/
   `sensitivity` kolonları bilinçli olarak `NaN` (testle doğrulanır).
   Yalnızca birleşik tablonun ihtiyaç duyduğu `random_forest/v4_from_v2`
   (Strateji B) için ayrı, küçük bir replay eklendi
   (`reports/tables/g1_rf_v4fromv2_auroc_sensitivity.csv`).
2. **E5'in 4 final adayı (200 satır, `reports/tables/e5_threshold_
   selection.csv`):** **sıfır yeniden fit** — AUROC, `e3_calibrated_
   oof_predictions.csv`'nin zaten diskte duran `method="raw"` (ham,
   kalibrasyonsuz) dış-test olasılıklarından; Sensitivity, `e4_prior_
   corrected_probabilities.csv`'nin SLD-düzeltilmiş olasılıklarının
   zaten kayıtlı `chosen_threshold` ile eşiklenmesinden hesaplandı (aynı
   yöntem `f1_chosen`/`specificity_chosen`'ın hesaplandığı yöntemle
   birebir tutarlı). Bu mümkün çünkü **Beta kalibrasyonu ve SLD önsel
   düzeltmesi ikisi de olasılığın monotonik dönüşümüdür** — AUROC yalnızca
   sıralamaya bağlı olduğu için hangi işlem aşamasındaki olasılıktan
   hesaplandığı sonucu değiştirmez. Final modelin kendi OOF'unda bu
   `roc_auc_score(y_full, oof_proba) == roc_auc_score(y_full, sld)`
   testiyle **doğrudan doğrulandı** (bit-bit eşit).
3. **Final model (`f0_uncertainty_analysis.csv`):** madde 10'un
   deterministik `cross_fit_oof` yeniden çağrısına (`f0_uncertainty_
   analysis.py`, hiçbir model yeniden eğitilmiyor) AUROC, örnek-düzeyi
   bootstrap CI'li bir satır olarak eklendi.

### Güvenlik kontrolü — mevcut sayılar değişmedi mi?

- Her üç replay/hesaplama scripti, kendi içinde **eskisi ile yenisini
  karşılaştıran bir `assert`** taşıyor (mevcut kolonlar bit-bit aynı
  kalmazsa script hata verip durur).
- Bağımsız regresyon testi (`tests/test_g1_metric_expansion_pah.py`,
  7 test): E2'nin 16 kombinasyonunun F1 ortalama/std'si, bu raporun
  **yukarıdaki** "Tam karşılaştırma tablosu"nda zaten yayınlanmış
  sayılarla (örn. CatBoost/v1: 0,9313±0,0214) karşılaştırılıyor — geçici
  bir yedek dosyaya değil, onaylı rapor içeriğine karşı doğrulanıyor.
  E5/F0 için de aynı desen (`chosen_threshold=0,28`/`f0`'ın CI'si).
- **Bağımsız çapraz-doğrulama:** `random_forest/v4_from_v2` için AUROC
  iki **farklı** kod yolundan geçti — (a) gerçek replay (yeniden fit),
  (b) E5'in sıfır-refit yöntemi (kayıtlı olasılıktan okuma). İkisi
  **fold-fold bit-bit aynı** çıktı — hem replay metodolojisinin
  doğruluğunu hem monotoniklik varsayımını bağımsız olarak kanıtlıyor.
- Sonuç: `pytest tests/ -q` → **170/170 geçti** (163 mevcut + 7 yeni).

### Osman Hoca Formatında Birleşik Karar Tablosu

`reports/tables/13_OSMAN_HOCA_FORMATI_KARAR_TABLOSU.csv` — `Preprocessing
| Model | F1 | MCC | Sensitivity | Specificity | AUPRC | AUROC |
Threshold | Std` formatında, 10 satır (E1 baseline, E2'nin 4 en iyi
kombinasyonu, E5'in 4 final adayı, F0 final model).

**Kritik okuma notu — üç satır grubu farklı eşik/kalibrasyon
rejiminde, doğrudan yan yana "kazanan/kaybeden" okunmamalı:**

| Grup | Eşik | Kalibrasyon | Not |
|---|---|---|---|
| E1/E2 | Sabit (0,359 / 0,5) | Yok (ham olasılık) | F1'leri E5/F0'dan yüksek görünür — bu bir "daha iyi model" değil, farklı bir ölçüm rejimi |
| E5 | Nested, dış-fold-bazlı (~0,27-0,33) | Beta + SLD | Final popülasyona (%28,6 patojenik) göre kalibre edilmiş, gerçekçi performans tahmini |
| F0 (final) | Sabit 0,35 | Beta + SLD | Final bundle'ın kendi 369-satırlık tek OOF partisyonu (n=50 fold ortalaması değil) |

**Final model satırı (F0):** F1=0,8177, MCC=0,3921, Sensitivity=0,7208,
Specificity=0,7869, AUROC=0,8309 (%95 CI: [0,7701; 0,8889]), eşik=0,35.
*(DÜZELTME, Kanonik Metrik Referansları Düzeltme Taraması: bu satır bu
turdan önce hâlâ yanlış bundle'ın (`final_model_bundle.pkl`, 28 özellik)
sayılarını taşıyordu — "P1 Madde 10"/"P1 Madde 11" bölümlerindeki
düzeltmenin aksine, bu "Revize: AUROC + Sensitivity Eklenmesi" bölümü
`f0_uncertainty_analysis.py`'nin `BUNDLE_OUT`→`BUNDLE_PATH` düzeltmesi
sırasında gözden kaçmıştı. Doğru bundle'a (`final_model_bundle_v2.pkl`)
ait sayılar `13_FINAL_PERFORMANCE_CARD_PAH.md` ve `10_BOOTSTRAP_GUVEN_
ARALIKLARI_PAH.md` ile birebir tutarlı.)*

**E5'in dört adayı arasında AUROC sıralaması** (kalibrasyon/eşikten
bağımsız, saf ayrım gücü): CatBoost/v4_from_v2 (final aday) **0,8364**
> Random Forest/v4_from_v2 0,8284 > CatBoost/v1 0,7949 > LightGBM/v1
0,7842 — final adayın seçimi yalnızca ağırlıklı-F1'e değil, eşik-bağımsız
ayrım gücüne göre de en iyi adaydır, bu revize turundan önce ölçülmemiş
bir doğrulamaydı.

### Kod / dosyalar

- `src/genova/pah/g1_e1_baseline_auroc_sensitivity.py`,
  `g1_e2_auroc_sensitivity_replay.py`, `g1_rf_v4fromv2_auroc_sensitivity.py`,
  `g1_e5_auroc_sensitivity.py`, `g1_build_osman_hoca_table.py` (hepsi yeni).
- `f0_uncertainty_analysis.py` — AUROC metriği eklendi (Adım 1'e), geri
  kalan mantık değişmedi.
- `reports/tables/e2_model_comparison.csv`, `e5_threshold_selection.csv`,
  `f0_uncertainty_analysis.csv` — yalnızca yeni kolon/satır eklendi.
- `reports/tables/g1_e1_baseline_auroc_sensitivity.csv`,
  `g1_rf_v4fromv2_auroc_sensitivity.csv`, `13_OSMAN_HOCA_FORMATI_
  KARAR_TABLOSU.csv` (yeni).
- `tests/test_g1_metric_expansion_pah.py` (yeni, 7 test).

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı. Hiçbir hiperparametre/eşik yeniden ARANMADI — hepsi kayıtlı
ayarların replay'i veya diskte zaten duran olasılıklardan hesaplama.

---

## Madde 4: Mutlak-Eşik Fold Kırılganlık Sayımı (CFTR Tarzı)

> Çapraz-panel karşılaştırmasının (`12_CAPRAZ_PANEL_KARSILASTIRMA_PAH.md`)
> Eksen 3'ü: CFTR her finalist için "kaç fold'da MCC<0,40,
> Sensitivity<0,85, Specificity<0,70" gibi **mutlak eşik-altı fold
> sayma** metrikleri raporluyor — PAH'ta yalnızca göreli (kazanma
> oranı) sayımlar vardı. Bu bölüm saf bir sayma işi: **hiçbir yeni
> tahmin/eğitim/arama yok**, yalnızca `e5_threshold_selection.csv`'nin
> (Revize turunda eklenen `mcc_chosen`/`sensitivity_chosen`/
> `specificity_chosen` kolonları) ve F0 final modelin kendi 5-fold iç
> OOF'unun (madde 10'un `cross_fit_oof`'uyla birebir aynı, deterministik
> yeniden üretim) fold-düzeyi sayılarının sayılması.
>
> **Bu bir yeni karar kuralı DEĞİL, tanımlayıcı (descriptive) bir
> kırılganlık resmidir** — 50 fold'un aynı 369 satırın tekrarlı
> bölünmesinden geldiği (dolayısıyla bağımsız gözlem olmadığı) daha
> önce defalarca vurgulanmıştı (bkz. madde 7/9/F1-F3'ün tekrarlı-bölme
> disiplini); bu sayım o uyarıyı geçersiz kılmaz.

### İki eşik seti

| Eşik seti | MCC | Sensitivity | Specificity | Anlamı |
|---|---|---|---|---|
| **CFTR'nin kendi eşikleri** | <0,40 | <0,85 | <0,70 | Birebir çapraz-panel karşılaştırması için |
| **PAH-kalibreli** | <0,2805 | <0,6667 | <0,5556 | E5'in 4 final adayının havuzlanmış (4×50=200 gözlem) fold-düzeyi dağılımının **her metrik için ayrı** 25. persentili — "bu adayların kendi tipik performansına göre kötü sayılan alt-çeyrek" |

**Şeffaflık notu:** CFTR'nin eşikleri PAH'ın metrik ölçeğinde ayrım
gücü düşük — örn. MCC<0,40 neredeyse *her* fold'u işaretliyor (E5
adaylarının MCC'si zaten tipik olarak ~0,33-0,38 civarında), bu yüzden
CFTR eşiğiyle "az kırılgan" görünmek yanıltıcı olabilir. PAH-kalibreli
eşikler PAH'ın kendi ölçeğinde anlamlı bir alt-çeyrek ayırıyor; ikisi de
ayrı ayrı raporlanıyor, hiçbiri diğerinin yerine "doğru" eşik olarak
sunulmuyor.

### Sayım tablosu (`reports/tables/14_FOLD_KIRILGANLIK_SAYIMI.csv`)

| Aday | Eşik seti | n_fold | MCC-altı | Sens-altı | Spec-altı | Toplam işaret |
|---|---|---|---|---|---|---|
| CatBoost/v1 (E5) | CFTR | 50 | 36 | 35 | 32 | 103 |
| CatBoost/v1 (E5) | PAH-kalibreli | 50 | 17 | 13 | 16 | 46 |
| **CatBoost/v4_from_v2 (E5, final aday)** | CFTR | 50 | 33 | 42 | 14 | 89 |
| **CatBoost/v4_from_v2 (E5, final aday)** | PAH-kalibreli | 50 | 11 | 19 | 5 | 35 |
| LightGBM/v1 (E5) | CFTR | 50 | 31 | 31 | 35 | 97 |
| LightGBM/v1 (E5) | PAH-kalibreli | 50 | 15 | 7 | 19 | 41 |
| Random Forest/v4_from_v2 (E5-EK) | CFTR | 50 | 29 | 39 | 23 | 91 |
| Random Forest/v4_from_v2 (E5-EK) | PAH-kalibreli | 50 | **7** | **10** | **8** | **25** |
| [FINAL] CatBoost (F0) | CFTR | 5 | 2 | 5 | 1 | 8 |
| [FINAL] CatBoost (F0) | PAH-kalibreli | 5 | 1 | 0 | 1 | 2 |

*(F0 satırı n=5 — final bundle'ın kendi tek, sabit 5-fold iç OOF
partisyonu; E5'in n=50'siyle (10-tekrarlı 5-fold) doğrudan oransal
kıyaslanamaz, oranlar farklı ölçüm birimleri — bkz. madde 10/F2'nin
"fold-düzeyi vs örnek-düzeyi" uyarısı.)*

### Yorum — dürüst bir nüans, E5/E6 kararını geçersiz kılmıyor

**PAH-kalibreli eşikte final aday (CatBoost/v4_from_v2) 4 adayın "en az
kırılganı" DEĞİL** — Random Forest/v4_from_v2, hem CFTR eşiğinde (91 vs
89, ihmal edilebilir fark) hem özellikle PAH-kalibreli eşikte (25 vs 35)
daha az eşik-altı fold'a sahip. Bu dürüstçe raporlanıyor, gizlenmiyor.

**Ama bu, E5/E6'nın kararını sorgulamıyor — üç nedenle:**

1. **Zaten biliniyordu ve istatistiksel olarak ölçülmüştü.** Madde 10
   Adım 4'ün Nadeau-Bengio testi, CatBoost/v4_from_v2 ile RF/v4_from_v2
   arasında weighted-F1'de **anlamlı fark bulamamıştı**
   (mean_diff=-0,0015, NB p=0,9642 — pratikte eşdeğer). Bu yeni sayım,
   aynı "ikisi istatistiksel olarak ayırt edilemez" sonucuna **farklı
   bir açıdan** ulaşıyor — çelişki değil, aynı bulgunun tekrarı.
2. **CatBoost başka eksenlerde hâlâ üstün.** Revize turunun AUROC
   sıralamasında CatBoost/v4_from_v2 (0,8364) RF'den (0,8284) daha
   yüksek — eşik-bağımsız ayrım gücünde CatBoost öne çıkıyor.
3. **Bu tek bir yeni, tanımlayıcı metrik** — E5/E6'nın kararı çok daha
   geniş bir kanıt kümesine (nested ağırlıklı-F1, OOF korelasyonu,
   ensemble testi, tekrarlı-bölme+NB) dayanıyordu. Bu sayım o kümeye
   **eklenen** bir gözlem, onun **yerine geçen** bir karar kuralı değil.

**Sonuç: model değiştirilmiyor, karar korunuyor** — ama RF'in bu
tanımlayıcı eksende biraz daha "tutarlı" görünmesi, ileride (örn. panel
birleştirme sonrası yeniden değerlendirme gerekirse) hatırlanması
gereken bir nüans olarak kayda geçiriliyor.

**F0 (deploy edilen model), her iki eşik setinde de görece az
işaretleniyor** (CFTR: 8/15, PAH-kalibreli: 2/15) — ama n=5 fold'un
istatistiksel gücü çok düşük, bu tek başına "sağlam" iddiasının kanıtı
sayılmamalı, yalnızca bir gözlem.

### Kod / dosyalar

- `src/genova/pah/g2_fold_fragility_counting.py` (yeni) —
  `reports/tables/14_FOLD_KIRILGANLIK_SAYIMI.csv` üretir.

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı. Hiçbir yeni tahmin/eğitim/arama yapılmadı — F0'ın 5-fold
OOF'u dahil, her şey mevcut deterministik fonksiyonların (aynı seed,
aynı hiperparametre, aynı kalibratör/eşik) yeniden çağrılmasıyla
üretildi.

---

## Seed-Bagged CatBoost Denemesi — DENENDİ, KAPATILDI (model değişmedi)

> Dış denetim önerisi: final tarifin (26 özellik, `depth=5,lr=0,05`,
> Beta, SLD, uyarlanabilir eşik) aynen korunup, her fold'da **5 farklı
> seed** (11, 23, 42, 71, 101) ile ayrı ayrı eğitilen CatBoost'ların
> olasılık ortalamasının ("seed bagging") varyansı ve `CAT_1`
> kırılganlığını iyileştirip iyileştirmediği. Yeni bir model ailesi
> değil, mevcut kararın küçük bir varyasyonu.

### Sonuçlar (5-fold iç CV, `f0_final_model.py`'nin kendi split'i)

| | TEK-seed (mevcut) | Seed-bagged (5 seed) |
|---|---|---|
| Ağırlıklı-F1 (genel) | 0,6417 | 0,6365 |
| Fold-bazlı ağırlıklı-F1 | [0,579; 0,682; 0,529; 0,687; 0,774] | [0,568; 0,642; 0,597; 0,687; 0,702] |
| **Fold-arası std** | **0,0867** | **0,0510** |
| **En kötü fold** | **0,5286** | **0,5683** |
| raw AUC (genel) | 0,8309 | 0,8330 |

**Adım 3.1 — Ortalama performans (NB testi, weighted-F1, n=5 fold):**
mean_diff(bagged−single) = **−0,0108**, NB p = **0,7743** — istatistiksel
olarak anlamlı fark yok (beklenen, n=5'in gücü zaten düşük).

**Adım 3.2 — Varyans/worst-fold:** Bagging fold-arası std'yi **%41
azaltıyor** (0,0867→0,0510) ve en kötü fold'u **belirgin iyileştiriyor**
(0,5286→0,5683) — bu, bagging'in klasik "varyans azaltma" etkisinin
küçük ölçekte de gözlemlenebilir olduğunu gösteriyor. **İlginç bir
gözlem, ama karar kuralının birincil kriteri değil** (aşağıya bakın).

**Adım 3.3 — `CAT_1` kırılganlığı (F3'ün en kritik testi, kritik yön —
train=CAT_1-boş → test=CAT_1-dolu):**

| Aday | F1 | raw AUC |
|---|---|---|
| TEK-seed (F3'ün kayıtlı referansı) | 0,0588 | **0,5727** |
| **Seed-bagged** | 0,0588 | **0,5677** |

**Bagging, `CAT_1` çöküşünü hafifletmiyor — hatta AUC marjinal olarak
(gürültü seviyesinde) daha da düşük.** Bu beklenen bir sonuçtur: F3'ün
kök-neden analizi (bkz. F3 sonrası teşhis bölümü) çöküşün `AL_26/12/7/49`
gibi belirli özelliklerden değil, `CAT_1`-boş alt-kümenin (132 satır)
`AL_` eksiklik oranındaki yapısal farktan (%91,4 vs %36,6) kaynaklandığını
göstermişti — bu, hangi seed'le eğitildiğinden bağımsız bir veri
kısıtıdır, model varyansı değil. Seed bagging (F1/F3 karar matrisinin
"3-4 özelliği çıkar" denemesi gibi) bu kök nedenin doğasına dokunmuyor.

### Karar

Karar kuralına göre: NB p≥0,05 (✅, 0,7743) **VE** `CAT_1` çöküşünde
belirgin bir iyileşme yok (✅, 0,5727→0,5677, `>0,65` eşiğinin çok
altında) → **bagging'in getirisi yok, 5× hesaplama maliyetini
karşılamıyor. Mevcut model (`final_model_bundle_v2.pkl`) korunuyor.**
Varyans azalması gerçek ama karar kuralının birincil kriterlerini
(ortalama performans + `CAT_1` düzeltmesi) değiştirmiyor.

**Toplam hesaplama süresi: 12 saniye** (35 CatBoost fit — 5+25 OOF
fit'i + 10 `CAT_1`-geçiş fit'i; öngörülen >30dk eşiğinin çok altında,
ayrı bir 2-seed ön-test gerekmedi).

### Kod / dosyalar

- `src/genova/pah/f0_seed_bagging_experiment.py` (yeni) —
  `reports/tables/f0_seed_bagging_per_fold.csv`,
  `f0_seed_bagging_cat1_shift.csv`.
- `f0_final_model.py`, `f3_robustness_stress_tests.py`, `models.py`
  DEĞİŞTİRİLMEDİ — yalnızca import edilip yeniden kullanıldı.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — kazanç bulunmadığı için zaten otomatik bir değişiklik
söz konusu değildi.

---

## Benign-Ağırlıklı + Domain-Dengeli CatBoost Denemesi — DENENDİ, DOĞRULANAMADI (model korunuyor)

> Kullanıcının kendi hesabıyla doğruladığı üç bulguya dayanan deneme:
> `sample_weight(row) = class_weight(Label) × domain_weight(row)^strength`
> — `class_weight` benign satırlara {1,0; 1,25; 1,5; 2,0} (patojenik
> hep 1,0); `domain_weight`, `CAT_1` dolu/boş × `AL_` eksiklik tertili
> (eğitim-fold'unun kendi tertilleri, fold-güvenli) alt-gruplarının
> ters-frekansı, `strength` üssüyle yumuşatılmış. 4×3=12 kombinasyon.
>
> **Not:** Mevcut üretim modeli zaten `class_weight=2,0` kullanıyor
> (`models.py::MINORITY_WEIGHT`) — yani ızgaranın `cw=2,0/str=0,0`
> hücresi **mevcut modelin ta kendisi**. Bu, referansla karşılaştırma
> için bir iç tutarlılık kontrolü olarak da işledi: bu hücrenin NB
> testi `p=1,0000, fark=0,0000` çıktı (beklenen, referansla birebir
> aynı olduğu için) — metodolojinin doğru kurulduğunu doğruluyor.

### Referans (mevcut model, `cross_fit_oof` — değiştirilmedi, import edildi)

Eşik=0,35, sensitivity=0,7208, specificity=0,7869, final-projeksiyon
(100P/250B): F1=0,6397, MCC=0,4783, FP=53,28. Worst-fold ağırlıklı-F1
=0,5286. `CAT_1` kritik-yön raw AUC=0,5727 (F3'ün kayıtlı referansıyla
birebir aynı).

### 12 kombinasyonun tam tablosu

| cw | strength | specificity | NB p | NB fark | `CAT_1` AUC | worst-fold | **kriter** |
|---|---|---|---|---|---|---|---|
| 1,00 | 0,0 | 0,6393 | 0,8470 | +0,0144 | 0,5876 | 0,5812 | ❌ |
| 1,00 | 0,5 | 0,6721 | 0,9238 | +0,0059 | 0,5876 | 0,5782 | ❌ |
| **1,00** | **1,0** | **0,8525** | **0,3332** | **+0,0570** | **0,5876** | **0,5893** | **✅** |
| 1,25 | 0,0 | 0,6557 | 0,8205 | +0,0117 | 0,5765 | 0,5999 | ❌ |
| 1,25 | 0,5 | 0,7541 | 0,8132 | +0,0096 | 0,5765 | 0,5779 | ❌ |
| 1,25 | 1,0 | 0,8361 | 0,4320 | +0,0460 | 0,5765 | 0,5487 | ❌ (spec_Δ=+0,049, eşiğin (0,05) az altında) |
| 1,50 | 0,0 | 0,6393 | 0,9924 | +0,0008 | 0,5898 | 0,5663 | ❌ |
| 1,50 | 0,5 | 0,7213 | 0,9477 | +0,0027 | 0,5898 | 0,5787 | ❌ |
| 1,50 | 1,0 | 0,7541 | 0,6344 | +0,0181 | 0,5898 | 0,5779 | ❌ |
| **2,00 (mevcut)** | 0,0 | 0,7869 | 1,0000 | 0,0000 | 0,5727 | 0,5286 | — (referans) |
| 2,00 | 0,5 | 0,8033 | 0,5437 | +0,0246 | 0,5727 | 0,5683 | ❌ (spec_Δ=+0,016) |
| 2,00 | 1,0 | 0,8197 | 0,5666 | +0,0439 | 0,5727 | 0,5285 | ❌ (worst-fold_Δ=−0,0001, sınıra çok yakın ama spec_Δ=+0,033<0,05) |

(Tam veri, tüm sütunlar: `reports/tables/f0_weighting_experiment_grid.csv`.)

### Karar kuralının üç kriteri — tek aday geçti: `cw=1,0, strength=1,0`

1. **Anlamlı düşüş yok:** NB p=0,3332 (≥0,05) ✅, ayrıca yön zaten
   lehte (+0,0570).
2. **Specificity ≥0,05 artıyor:** 0,7869→0,8525, **Δ=+0,0656** ✅
   (`CAT_1` şartı da denendi ama gerekmedi — Δ=+0,0149, `≥0,10`
   eşiğinin altında, gürültü bandına yakın).
3. **Worst-fold >0,03 kötüleşmiyor:** 0,5286→0,5893, **Δ=+0,0607**
   (kötüleşme yok, tersine iyileşme) ✅.

**Final-projeksiyon etkisi (100P/250B, `f0_weighting_experiment_grid.csv`'den
doğrulandı):** F1 0,6397→**0,7000** (Δ=+0,060), MCC 0,4783→**0,5728**
(Δ=+0,095), FP 53,28→**36,89** (specificity artışının doğrudan sonucu).

**Dikkat çeken nüans:** Bu aday, mevcut modelin `class_weight=2,0`
sabit ağırlığını **1,0'a düşürüp** (yani sınıf-dengesizliği modele
klasik yoldan hiç düzeltmeden), bunun yerine **tüm ağırlığı domain-
dengeye** (`CAT_1`/`AL_`-eksiklik alt-gruplarının ters-frekansı,
`strength=1,0` — tam güç) veriyor. Yani kazanç, "benign'e daha çok
ağırlık ver"den değil, "az temsil edilen `CAT_1`/`AL_`-eksiklik alt-
gruplarını dengele"den geliyor — F1'in/F3'ün bulduğu provenance-temelli
kırılganlıkla aynı eksende bir müdahale, ama `CAT_1` şok-testinde
büyük bir düzelme YARATMIYOR (Δ=+0,015, gürültü bandında) — asıl
kazanç specificity/worst-fold üzerinden geliyor, `CAT_1` fragility'sini
"çözmüyor".

### Tekrarlı-Bölme Doğrulaması (madde 7'nin AYNI disiplini) — SONUÇ: GÜRÜLTÜ

> Tek-5-fold ölçümünün istatistiksel gücü düşük olduğu için (n=5),
> `cw=1,0, strength=1,0` adayı madde 7'nin `depth=3` vs `depth=5`
> teşhisinde kullandığı **aynı** tekrarlı-bölme disipliniyle test edildi:
> 10 farklı `StratifiedKFold(n_splits=5, shuffle=True, random_state=i)`
> (split bankasından bağımsız, grup-farkında değil — bilerek böyle, saf
> bir gürültü/kararlılık testi). Tarif (domain-weight formülü,
> `strength=1,0`) bu turda **yeniden ayarlanmadı**.

**Sonuç: aday 10 tekrarın yalnızca 6'sında kazandı** (mevcut model 4'ünde),
**binom testi p=0,7539** — istatistiksel olarak yazı-tura'dan ayırt
edilemez. Ortalama marj +0,0119 (weighted-F1 üzerinde) teknik olarak
`≥0,01` eşiğini geçse de, 6/10 kazanma oranı `≥7/10` eşiğinin altında
kaldığı için karar kuralı **"gerçek sinyal"** dalına düşmedi.

**En çarpıcı bulgu — özgün ölçümdeki specificity kazancı REPLİKE OLMADI:**
Tek-fold ölçümünde specificity Δ=+0,0656 (güçlü, kriterin ana dayanağı)
görülmüştü. 10 tekrarda specificity_delta **yön bile tutarlı değil**
(4/10 pozitif, 6/10 negatif) ve **ortalaması NEGATİF** (−0,0213) —
yani özgün ölçümdeki specificity kazancı, o **tek** rastgele bölünmenin
bir tesadüfiydi, domain-dengeli ağırlıklandırmanın tutarlı bir özelliği
değil. Worst-fold_delta de benzer şekilde tutarsız (4/10 pozitif, 4/10
negatif, 2 berabere; ortalama +0,0061, ihmal edilebilir).

`CAT_1` geçiş testi (aday-başına tek, seed-bağımsız ölçüm — bkz. kod
notu) zaten tek-fold turunda da küçüktü (Δ=+0,0149) ve bu turda
değişmedi (aynı sayı, ölçüm yöntemi gereği).

| | Aday kazandı | Marj (ort±std) | Binom p | Spec_delta yön | Worst-fold_delta yön |
|---|---|---|---|---|---|
| 10 tekrar | 6/10 | +0,0119±0,0162 | 0,7539 | 4(+)/6(−), ort=−0,0213 | 4(+)/4(−)/2(=), ort=+0,0061 |

**KARAR: GÜRÜLTÜ — mevcut model korunuyor.** Madde 7'nin öğrettiği ders
burada da doğrulandı: tek bir bölünmedeki parlak sonuç (özellikle
specificity kazancı) tekrarlı bölünmede kayboldu. `final_model_bundle_
v2.pkl` **değişmiyor**, resmi aday olarak önerilmiyor. "Domain-dengeli
ağırlıklandırma" fikri denendi, tekrarlı-bölmede doğrulanamadı — bu
turda kapanıyor.

**Toplam hesaplama süresi:**
- Adım 1 (tek-5-fold ızgara): 32 saniye (84 CatBoost fit).
- Adım 2 (10-tekrarlı doğrulama): 46,2 saniye (ölçüldü — 102 CatBoost fit).

### Kod / dosyalar

- `src/genova/pah/f0_weighting_experiment.py` (yeni) —
  `reports/tables/f0_weighting_experiment_grid.csv`,
  `f0_weighting_experiment_per_fold.csv`.
- `src/genova/pah/f0_weighting_repeated_split_check.py` (yeni) —
  `reports/tables/f0_weighting_repeated_split_check.csv`,
  `f0_weighting_repeated_split_pooled_folds.csv`.
- `f0_final_model.py::cross_fit_oof`, `f3_robustness_stress_tests.py`,
  `f0_weighting_experiment.py` DEĞİŞTİRİLMEDİ — yalnızca import edilip
  yeniden kullanıldı; domain-weight tarifi ikinci turda yeniden
  ayarlanmadı.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — tekrarlı-bölme testi kazancın gerçek olmadığını
gösterdiği için zaten otomatik bir değişiklik söz konusu değildi.

---

## Provenance + Missingness-Pattern Holdout Teşhisi — Kırılganlık Haritası (saf teşhis, karar yok)

> F3'ün `CAT_1` boş→dolu geçiş testi (raw AUC 0,831→0,573) **tek bir**
> sentetik senaryoya dayanıyordu. Bu tur, aynı kırılganlığın ne kadar
> **genel** olduğunu üç bağımsız açıdan (kaynak/`CAT_2`, eksiklik-
> yoğunluğu/`AL_` tertile, `CAT_1` çapraz) ölçüyor. **Hiçbir model/
> eşik/dosya değişikliği yok, karar kuralı yok** — bu, bir sonraki
> turun (augmentation / shift-robust aday) tasarımına kanıt sağlayan
> saf bir teşhis.
>
> Metodoloji: her holdout senaryosunda, tutulan grubun kalibrasyona
> **hiç sızmaması** için önce kalan veri üzerinde `cross_fit_oof`
> (değişmedi) ile kalibratör/eşik seçildi, sonra kalan verinin
> tamamıyla TEK bir model fit edilip tutulan grupta test edildi.

### Referans (normal CV, bundle'in kendi `cross_fit_oof`'u)

raw AUC=**0,8309**, ağırlıklı-F1=0,6417, specificity=0,7869,
sensitivity=0,7208 (F3'ün kayıtlı referansıyla birebir aynı).

### Adım 1 — `CAT_2` Leave-One-Category-Out (kaynak-bazlı, 8 kategori)

| Kategori | n_test (benign) | raw AUC | Δ raw AUC | Δ ağırlıklı-F1 | Güç |
|---|---|---|---|---|---|
| **AllofUs_EAS** | 17 (2) | **0,4667** | **−0,3643** | −0,2096 | yeterli |
| AllofUs_AFR | 23 (6) | 0,6275 | −0,2035 | −0,3851 | yeterli |
| EMPTY (AllofUs-dışı) | 225 (41) | 0,7266 | −0,1043 | −0,1322 | yeterli (n büyük) |
| AllofUs_OTH | 19 (3) | 0,7500 | −0,0809 | −0,0841 | yeterli |
| AllofUs_EUR | 62 (5) | 0,7614 | −0,0695 | +0,1230 | yeterli |
| AllofUs_AMR | 13 (3) | 0,8667 | +0,0357 | +0,0249 | **düşük (n<15)** |
| AllofUs_MID | 3 (1) | 1,0000 | +0,1691 | +0,3583 | **düşük (n<15)** |
| AllofUs_SAS | 7 (0) | NaN (tek sınıf) | — | +0,2814 | **düşük (n<15)** |

**En çarpıcı bulgu: `AllofUs_EAS`'ın tamamen dışarıda bırakılması (n=17,
yeterli güç), F3'ün `CAT_1` kritik-yön testinden (Δ=−0,2582) BİLE DAHA
BÜYÜK bir raw AUC düşüşü veriyor (Δ=−0,3643).** `AllofUs_AFR` de (n=23)
belirgin bir düşüş gösteriyor (Δ=−0,2035). Bu, kırılganlığın yalnızca
`CAT_1`'in doluluk durumuna değil, **etnik/kaynak alt-grup kompozisyonuna**
da bağlı olduğunu gösteriyor — F1/F3'ün bulduğu tek-eksenli bir sorun
değil, çok-eksenli bir provenance-duyarlılığı.

### Adım 2 — `AL_` Eksiklik-Yoğunluğu Tertile Holdout

| Tertile | n_test (benign) | raw AUC | Δ raw AUC | Δ ağırlıklı-F1 |
|---|---|---|---|---|
| **düşük** | 123 (16) | **0,6011** | **−0,2299** | −0,1266 |
| orta | 138 (34) | 0,7723 | −0,0586 | −0,1722 |
| yüksek | 108 (11) | 0,9283 | **+0,0974** | +0,1552 |

**Beklenmedik bulgu: en büyük düşüş "düşük eksiklik" tertile'ında,
"yüksek eksiklik" tertile'ında DEĞİL.** Sezgisel beklenti (yüksek
eksiklik = `CAT_1`-boş grubun profiline benzer = daha kırılgan) burada
**tersine döndü** — yüksek-eksiklik holdout'ta model referanstan daha
iyi (+0,0974). Olası açıklama: yüksek `AL_` eksikliği (özellikle
`al_all_missing=1`) kendi başına bilgilendirici bir sinyal olduğu için
(bkz. `CLAUDE.md`: "potansiyel patojenite sinyali"), bu alt-grup
kalan veride de (kısmen `CAT_1`-boş satırlar üzerinden) temsil ediliyor
ve model bu deseni öğrenmeye devam edebiliyor; buna karşın "düşük
eksiklik" (yani `AL_` bilgisinin en zengin/en ayırt edici olduğu satırlar)
elden çıkınca model bu zengin sinyalden mahrum kalıyor.

### Adım 3 — `CAT_1` Çapraz Değerlendirme (bu modülün kendi kod yolundan doğrulama)

| Yön | n_test (benign) | raw AUC | Δ raw AUC | Δ ağırlıklı-F1 |
|---|---|---|---|---|
| train=dolu → test=boş | 132 (22) | 0,8585 | +0,0275 | +0,0929 |
| **train=boş → test=dolu (KRİTİK)** | 237 (39) | **0,5727** | **−0,2582** | **−0,5840** |

F3'ün kayıtlı referansıyla (raw AUC=0,5727) **birebir aynı** — bu
modülün kendi bağımsız kod yolu (farklı orkestrasyon, aynı alt
fonksiyonlar) aynı sonucu üretiyor, F3'ün bulgusunu doğrudan doğruluyor.

### Adım 4 — Kırılganlık Haritası (sentez)

| Test | AUC (referans) | AUC (holdout) | Δ AUC | Δ ağırlıklı-F1 | n (holdout) | Güç |
|---|---|---|---|---|---|---|
| `CAT_1` çapraz (KRİTİK, boş→dolu) | 0,8309 | 0,5727 | **−0,2582** | −0,5840 | 237 | yeterli |
| `CAT_2` LOO — AllofUs_EAS | 0,8309 | 0,4667 | **−0,3643** | −0,2096 | 17 | yeterli |
| `CAT_2` LOO — AllofUs_AFR | 0,8309 | 0,6275 | −0,2035 | −0,3851 | 23 | yeterli |
| `AL_` tertile — düşük | 0,8309 | 0,6011 | −0,2299 | −0,1266 | 123 | yeterli |
| `CAT_2` LOO — EMPTY | 0,8309 | 0,7266 | −0,1043 | −0,1322 | 225 | yeterli |
| `AL_` tertile — orta | 0,8309 | 0,7723 | −0,0586 | −0,1722 | 138 | yeterli |
| `CAT_2` LOO — AllofUs_OTH | 0,8309 | 0,7500 | −0,0809 | −0,0841 | 19 | yeterli |
| `CAT_2` LOO — AllofUs_EUR | 0,8309 | 0,7614 | −0,0695 | +0,1230 | 62 | yeterli |
| `CAT_1` çapraz (dolu→boş) | 0,8309 | 0,8585 | +0,0275 | +0,0929 | 132 | yeterli |
| `AL_` tertile — yüksek | 0,8309 | 0,9283 | +0,0974 | +0,1552 | 108 | yeterli |
| `CAT_2` LOO — AllofUs_AMR | 0,8309 | 0,8667 | +0,0357 | +0,0249 | 13 | **düşük** |
| `CAT_2` LOO — AllofUs_MID | 0,8309 | 1,0000 | +0,1691 | +0,3583 | 3 | **düşük** |
| `CAT_2` LOO — AllofUs_SAS | 0,8309 | NaN | — | +0,2814 | 7 | **düşük** |

(Tam veri: `reports/tables/f3_provenance_holdout_diagnosis.csv`.)

**Genel yorum — en kritik shift türü hangisi?**

Yeterli-güçlü (n≥15) testler arasında en büyük 3 düşüş: **`CAT_2`/
AllofUs_EAS leave-one-out (−0,3643)**, `CAT_1` kritik yön (−0,2582),
`AL_`-düşük tertile (−0,2299). **`CAT_1`'in doluluk durumu tek başına
en kritik shift türü DEĞİL** — kaynak/etnik alt-grup dışlaması
(`CAT_2`) en az bir örnekte (EAS) daha büyük bir çöküşe yol açıyor.
Bu, bir sonraki turun augmentation/shift-robust tasarımının **yalnızca
`CAT_1`'e değil, `CAT_2`'nin (ve dolaylı olarak etnik-alt-grup
kompozisyonunun) tamamına** odaklanması gerektiğini gösteriyor — tek-
eksenli bir düzeltme (yalnızca `CAT_1`'e göre) yeterli olmayabilir.
Küçük-örneklem (`AMR`, `MID`, `SAS`, n<15) sonuçları gösterge
niteliğindedir, kesin bir bulgu olarak okunmamalı.

**Toplam hesaplama süresi: 37 saniye** (13 holdout senaryosu × 6
CatBoost fit + 1 referans × 5 fit = 83 fit; öngörülen >20dk eşiğinin
çok altında).

### Kod / dosyalar

- `src/genova/pah/f3_provenance_holdout_diagnosis.py` (yeni) —
  `reports/tables/f3_provenance_holdout_diagnosis.csv`.
- `f0_final_model.py::cross_fit_oof`/`select_calibrator`/`select_
  prior_and_threshold`, `f3_robustness_stress_tests.py::_score`,
  `f0_weighting_experiment.py::final_projection_metrics` DEĞİŞTİRİLMEDİ
  — yalnızca import edilip yeniden kullanıldı.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — bu saf bir teşhis turuydu, hiçbir karar kuralı
uygulanmadı.

---

## Kırılganlık Haritası Confound Audit — Düzeltme/Ek (saf teşhis, karar yok)

> Yukarıdaki kırılganlık haritasının iki bulgusu doğrulanmayı hak
> ediyordu: (1) `AllofUs_EAS`/`AllofUs_AFR`'nin küçük n'i (17/23) —
> özellikle sınıf dengesizliği göz önüne alınınca — AUC noktasının ne
> kadar güvenilir olduğu belirsizdi; (2) `AL_`-düşük tertile'ının
> beklenmedik kırılganlığının (Δ=−0,2299) `CAT_2`'nin AllofUs alt-
> kategorileriyle bir çakışma (confound) olup olmadığı netleşmemişti.
> **Sonuç: her iki bulgu da kısmen düzeltildi — ne tamamen doğrulandı
> ne tamamen çürütüldü, ikisi de daha nüanslı hale geldi.**

### Adım 1 — Bootstrap AUC CI: EAS/AFR'nin nokta tahmini göründüğünden çok daha belirsiz

| Kategori | n (benign/patojenik) | AUC (nokta) | %95 Bootstrap CI | Geçerli/toplam bootstrap |
|---|---|---|---|---|
| `AllofUs_EAS` | 17 (2/15) | 0,4667 | **[0,1875 ; 0,7638]** | 865/1000 (135 atlandı — tek-sınıf resample) |
| `AllofUs_AFR` | 23 (6/17) | 0,6275 | **[0,3556 ; 0,8556]** | 999/1000 |

**Her iki güven aralığı da çok geniş ve hem "rastgele tahmin" (0,5)
hem "referansa yakın" (0,83) değerleri kapsıyor.** `AllofUs_EAS`'ın
%95 CI'sinin üst ucu (0,764) referansa (0,8309) neredeyse yeterince
yakın ki nokta tahmini (0,4667) **tek başına güvenilir bir "çöküş"
kanıtı sayılamaz** — asıl sorun n=17'nin kendisi değil, yalnızca
**2 benign** satır içermesi (bootstrap'ın %13,5'i tek-sınıf resample
üretti, bu da örneklemin ne kadar ince olduğunun doğrudan kanıtı).

**Metodolojik ders:** Önceki turun "n≥15 → yeterli güç" eşiği **çok
kaba** kaldı — toplam n yeterli olsa bile azınlık sınıf sayısı (burada
2 ve 6) AUC'nin güvenilirliğini asıl belirleyen şey. Diğer küçük
kategoriler (`AMR` n=13 benign=3, `MID` n=3 benign=1, `SAS` n=7
benign=0) zaten düşük-güç etiketliydi, CI hesaplanmadı.

### Adım 2 — `CAT_2` × `AL_`-Tertile Crosstab: Confound EAS/AFR'ye özgü değil, TÜM AllofUs'a geneldir

| `CAT_2` | düşük | orta | yüksek | düşük % |
|---|---|---|---|---|
| AllofUs_AFR | 17 | 5 | 1 | 73,9% |
| AllofUs_AMR | 9 | 3 | 1 | 69,2% |
| AllofUs_EAS | 12 | 4 | 1 | 70,6% |
| AllofUs_EUR | 43 | 18 | 1 | 69,4% |
| AllofUs_MID | 2 | 1 | 0 | 66,7% |
| AllofUs_OTH | 15 | 4 | 0 | 78,9% |
| AllofUs_SAS | 6 | 1 | 0 | 85,7% |
| **EMPTY** | **19** | **102** | **104** | **8,4%** |

**Çarpıcı ve beklenenden geniş bir çakışma:** düşük-eksiklik tertile'ının
(123 satır) **%84,6'sı (104 satır) HERHANGİ bir AllofUs alt-kategorisinden**
— bu yalnızca EAS/AFR'ye özgü değil, **her bir AllofUs alt-kategorisinin
%67-86'sı** düşük tertile'da yoğunlaşıyor (`EMPTY`'nin yalnızca %8,4'üne
karşı). Bu, `CAT_2`'nin (kaynak) ve `AL_` eksiklik-yoğunluğunun **aynı
alttaki provenance sinyalinin** iki farklı kesitten görünümü olabileceğini
gösteriyor — tıpkı `CAT_1`/`AL_` eksikliği arasındaki daha önce kurulan
ilişkiye benzer bir desen.

### Adım 3 — Confound Kontrolü: Kısmi confound, tam değil — nüanslı sonuç

| Senaryo | n_test (benign/patojenik) | raw AUC | Δ AUC (referansa göre) |
|---|---|---|---|
| Orijinal düşük-tertile (tüm satırlar) | 123 | 0,6011 | **−0,2298** |
| **düşük-tertile, EAS+AFR çıkarılmış** | 94 | 0,7119 | **−0,1190** |
| düşük-tertile, TÜM AllofUs çıkarılmış (yalnızca `EMPTY`) | 19 (3/16) | 0,5833 | −0,2476 |

**Görev talimatının kuralına göre (Δ≥0,15 → bağımsız sinyal, Δ<0,05 →
saf confound):** EAS+AFR çıkarılınca düşüş **yarıya iniyor** (−0,2298→
−0,1190) ama **hâlâ 0,05'in üzerinde, 0,15'in altında — "ara bölge"**.
Bu, ne temiz bir "confound'du" ne temiz bir "bağımsız sinyal" kararını
destekliyor.

**Beklenmedik bir ek bulgu:** TÜM AllofUs satırları çıkarılıp yalnızca
`EMPTY`'nin (AllofUs-dışı) düşük-eksiklik satırlarında (n=19, sınırda-
düşük güç) test edilince, düşüş **kaybolmuyor, hatta biraz büyüyor**
(−0,2476). Bu, `AL_`-düşük-eksiklik'in **AllofUs-kaynaklı olmayan
satırlarda bile** kendi başına bir kırılganlık taşıdığını — yani
`CAT_2`/AllofUs confound'unun bulguyu **tam olarak açıklamadığını**
gösteriyor. n=19 (yalnızca 3 benign) küçük olduğu için bu son rakam
gösterge niteliğinde, kesin değil.

### Düzeltilmiş Sentez — Önceki Kırılganlık Haritasının Güncellenmiş Okunuşu

| Bulgu | Önceki durum | Güncellenmiş durum |
|---|---|---|
| `CAT_2`/AllofUs_EAS leave-one-out (Δ=−0,3643) | "Yeterli güç" etiketiyle **en büyük düşüş** olarak sunulmuştu | **Düşürüldü: gösterge niteliğinde.** %95 CI [0,19; 0,76] çok geniş, referansa yakın değerleri de kapsıyor — n=17 toplamı yeterli görünse de yalnızca 2 benign satır güvenilir bir AUC tahmini için yetersiz |
| `CAT_2`/AllofUs_AFR leave-one-out (Δ=−0,2035) | "Yeterli güç" | **Düşürüldü: gösterge niteliğinde.** %95 CI [0,36; 0,86] de benzer şekilde geniş |
| `AL_`-düşük tertile (Δ=−0,2299) | Bağımsız, "beklenmedik" bir bulgu olarak sunulmuştu | **Kısmen düzeltildi: karışık/kısmi confound.** Etkinin ~yarısı `CAT_2`/AllofUs kaynaklı (confound), ama EAS+AFR çıkarılsa bile Δ=−0,119 kalıyor — tam bir confound değil, kalan bir bağımsız bileşen de var |
| `CAT_1` çapraz (KRİTİK, Δ=−0,2582) | En sağlam, yeterli-n'li bulgu | **DEĞİŞMEDİ — hâlâ en güvenilir bulgu.** n=237 (39 benign), bu turda yeniden test edilmedi ama önceki turların (F3 + provenance-holdout modülü, iki bağımsız kod yolu) her ikisinde de birebir tutarlı (0,5727=0,5727) |

**Genel yorum güncellemesi:** Önceki turun "`CAT_2`/AllofUs_EAS
`CAT_1`'den bile kritik" iddiası **bu haliyle sürdürülemez** — o
bulgunun istatistiksel temeli (n=17, 2 benign) çok ince. Buna karşılık,
`AL_`-düşük tertile bulgusu ne tamamen reddedilebilir ne tam
güvenilir — kısmen `CAT_2` ile örtüşen, kısmen bağımsız bir sinyal.
**En sağlam, tek kırılganlık bulgusu hâlâ `CAT_1`'in doluluk-geçişi
testi** (n=237, geniş örneklem, iki bağımsız kod yolunda tutarlı). Bir
sonraki turun augmentation/shift-robust tasarımı öncelikle `CAT_1`
eksenine odaklanmalı; `CAT_2`/AllofUs ve `AL_`-eksiklik eksenleri
**ek, ama daha az kesin** kanıt olarak ikincil önceliktedir.

**Toplam hesaplama süresi: 10 saniye** (20 CatBoost fit + 2000 bootstrap
resample; öngörülen >10dk eşiğinin çok altında).

### Kod / dosyalar

- `src/genova/pah/f3_confound_audit.py` (yeni) —
  `reports/tables/f3_confound_cat2_tertile_crosstab.csv`,
  `f3_confound_check_results.csv`.
- `f3_provenance_holdout_diagnosis.py::evaluate_holdout` DEĞİŞTİRİLMEDİ
  — yalnızca import edilip yeniden kullanıldı.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — bu saf bir teşhis/düzeltme turuydu, hiçbir karar kuralı
uygulanmadı.

---

## `CAT_1`-Boş vs `AL_`-Eksiklik-Deseni Örtüşme Teşhisi (saf teşhis, karar yok)

> İki bulgu — `CAT_1` çapraz kırılganlığı (Δ=−0,2582, n=237) ve
> confound audit'in "AllofUs-bağımsız `AL_`-düşük" bulgusu (Δ=−0,2476,
> n=19) — aynı kök nedenin mi, yoksa iki ayrı mekanizmanın mı
> görünümü? **Sonuç: KARIŞIK — ağırlıklı olarak tek bir baskın
> mekanizma (`CAT_1`-boş ≈ `AL_`-eksikliği-yüksek, yapısal olarak
> neredeyse ayrılmaz), artı ayrı, küçük, açıklanamayan ikincil bir
> sinyal.**

### Adım 1 — Crosstab: Yapısal bulgu, kesin ve tartışmasız

| `CAT_1` | düşük | orta | yüksek |
|---|---|---|---|
| dolu | 123 | 106 | 8 |
| **boş** | **0** | 32 | 100 |

**`CAT_1`-boş satırların %0'ı (0/132) `AL_`-düşük-eksiklik tertile'ında
— bu hücre TAMAMEN BOŞ.** `CAT_1`-boş satırların %75,8'i (`100/132`)
zaten `AL_`-yüksek tertile'ında. Bu, `CAT_1`'in doluluk durumu ile
`AL_` eksiklik-yoğunluğunun **yapısal olarak neredeyse ayrılmaz**
olduğunu gösteriyor — `CAT_1`-boş olmak, pratikte `AL_`-düşük-eksiklikli
olmamayı garanti ediyor (istatistiksel bir eğilim değil, **kesin bir
dışlama**).

**Doğrudan kesişim kontrolü:** Önceki confound audit'in "AllofUs-bağımsız,
`AL_`-düşük kırılgan" grubunun (n=19) **hiçbiri `CAT_1`-boş değil —
19/19'u `CAT_1`-dolu.** Yani bu iki bulgu (CAT_1 çapraz kırılganlığı ve
AllofUs-bağımsız `AL_`-düşük kırılganlığı) **satır düzeyinde hiç
kesişmiyor** — biri `CAT_1`-boş popülasyonunda, diğeri tamamen
`CAT_1`-dolu popülasyonunun içinde bir alt-küme. Bu, ikisinin **aynı
satırların** iki farklı etiketten görünümü olmadığını, gerçekten
**ayrı satır kümelerinde** yaşayan iki ayrı gözlem olduğunu kanıtlıyor.

### Adım 2 — Referans Modelin Alt-Küme Performansı (yeniden fit yok, mevcut OOF'un dilimlenmesi)

> **Metodolojik not:** Bu adım Adım 1'den farklı bir soruya cevap
> veriyor — "referans model (tüm veriyle normal CV ile değerlendirilmiş)
> her alt-grupta ne kadar iyi?" (in-distribution alt-grup performansı),
> "bir grup tamamen dışarıda bırakılırsa transfer nasıl bozulur?"
> sorusuna DEĞİL (o, önceki turların holdout/transfer testleriydi).
> İkisi birbirini **tamamlıyor**, çelişmiyor.

| `CAT_1` | `AL_` tertile | n (benign/pato) | raw AUC | Ağırlıklı-F1 | Güç |
|---|---|---|---|---|---|
| dolu | düşük | 123 (16/107) | 0,7161 | 0,5952 | yeterli |
| dolu | orta | 106 (20/86) | 0,5872 | 0,4080 | yeterli |
| dolu | yüksek | 8 (3/5) | 0,8667 | 0,8889 | **düşük** |
| boş | düşük | **0 — BOŞ HÜCRE** | — | — | — |
| boş | orta | 32 (14/18) | **1,0000** | 1,0000 | yeterli (ama n=32 küçük, mükemmel AUC'ye ihtiyatla bakılmalı) |
| boş | yüksek | 100 (8/92) | 0,8635 | 0,8420 | yeterli |

**Temiz bir 2×2 kurulamadı** çünkü `CAT_1`=boş × `AL_`=düşük hücresi
gerçek veride yok (Adım 1'in doğrudan sonucu). Elde edilebilen
karşılaştırmalar:

- **`AL_` sabit (orta), `CAT_1` değişirken:** AUC 0,5872→1,0000
  (Δ=**+0,4128**, n=106 vs 32, ikisi de yeterli-güçlü) — büyük bir
  fark, ama **CAT_1-boş yönünde daha İYİ**, kırılganlık yönünde değil.
  Bu ölçüm in-distribution performansı yansıtıyor (bkz. yukarıdaki
  metodolojik not), önceki transfer/holdout testleriyle doğrudan
  kıyaslanamaz.
- **`CAT_1` sabit (dolu), `AL_` değişirken (düşük→yüksek):** AUC
  0,7161→0,8667 (Δ=+0,1505) — yüksek hücre n=8 (3 benign), **düşük
  güç**, yorumlanmamalı.

### Sentez — Bir Mekanizma mı, İki mi?

**Cevap: Ağırlıklı olarak TEK bir baskın mekanizma, artı küçük bir
ayrı/açıklanamayan ikincil sinyal:**

1. **`CAT_1`-boş fragility (n=237, en sağlam bulgu) ile `AL_`-eksiklik-
   yoğunluğu YAPISAL OLARAK NEREDEYSE AYNI EKSEN** — Adım 1'in kesin
   kanıtı (0/132 kesişim, `CAT_1`-boş asla `AL_`-düşük değil). Bu
   ikisini **augmentation açısından TEK bir hedef** olarak ele almak
   mantıklı: `CAT_1`-boş satırları oversample/reweight etmek DOLAYLI
   olarak `AL_`-yüksek-eksiklikli satırları da hedeflemiş olur (ve
   tersi) — bunlar aynı satırların çoğunluğu.
2. **AllofUs-bağımsız `AL_`-düşük kırılganlığı (n=19, confound audit'ten)
   AYRI, açıklanamayan bir ikincil sinyal** — bu grubun tamamı
   `CAT_1`-dolu olduğu için `CAT_1`-boş mekanizmasıyla açıklanamıyor.
   Küçük n (19, yalnızca 3 benign) nedeniyle bu turda kök nedeni
   netleştirilemedi — ayrı bir araştırma konusu olarak kalıyor.

### Augmentation Yönlendirmesi (bir sonraki tur için)

**Birincil hedef: `CAT_1`/`AL_`-eksiklik ekseni** (tek, baskın
mekanizma) — augmentation/robustness stratejisi doğrudan bu eksene
odaklanmalı, örneğin:
- `CAT_1`-boş satırları oversample/reweight ETMEK **veya** `AL_` blok-
  maskeleme (eğitim satırlarının bir kısmında `AL_` bloğunu yapay
  olarak tamamen eksik hale getirmek) — ikisi de aynı kökten geldiği
  için **birini yapmak diğerini de büyük ölçüde kapsar**, ikisini ayrı
  ayrı tasarlamaya gerek yok.

**İkincil, ayrı bir açık soru:** AllofUs-bağımsız, `CAT_1`-dolu,
`AL_`-düşük-eksiklikli küçük grubun (n=19) kırılganlığı **birincil
augmentation tasarımının kapsamı dışında** — bu grup çok küçük ve kök
nedeni belirsiz, ayrı, daha büyük bir örneklemle (varsa gelecek veri
turlarında) yeniden değerlendirilmeli.

**Toplam hesaplama süresi: 2 saniye** (5 CatBoost fit — yalnızca
referans modelin `cross_fit_oof`'u, Adım 2 yeniden fit gerektirmedi;
öngörülen >10dk eşiğinin çok altında).

### Kod / dosyalar

- `src/genova/pah/f3_cat1_al_overlap_diagnosis.py` (yeni) —
  `reports/tables/f3_cat1_al_tertile_crosstab.csv`,
  `f3_cat1_al_2x2_cells.csv`.
- `f0_final_model.py::cross_fit_oof` DEĞİŞTİRİLMEDİ — yalnızca import
  edilip yeniden kullanıldı (tek çağrı, yeniden fit yok).

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — bu saf bir teşhis turuydu, hiçbir karar kuralı
uygulanmadı.

---

## `CAT_1`/`AL_`-Eksiklik Ekseni Missingness-Augmentation Denemesi — DENENDİ, KRİTER KARŞILANMADI (model korunuyor)

> Örtüşme teşhisinin bulgusuna dayanan deneme: `CAT_1`-boş kırılganlığı
> ile `AL_`-eksiklik-yoğunluğu yapısal olarak neredeyse aynı eksen
> olduğu için (0/132 kesişim), eğitim sırasında `AL_` sütunlarının
> kontrollü maskelenmesinin (bağımsız-sütun veya blok modu, %15/%30
> oranında) `CAT_1` geçiş kırılganlığını azaltıp azaltmadığı test
> edildi. n=19'luk ikincil sinyal bu turun kapsamı dışında bırakıldı.

### 4 Augmentasyon Adayının Tam Tablosu

| Aday | NB p | NB fark | `CAT_1` kritik AUC | Δ `CAT_1` AUC | worst-fold | **kriter** |
|---|---|---|---|---|---|---|
| mask=0,15 bağımsız-sütun | 0,3110 | −0,0248 | 0,5041 | −0,0686 | 0,5372 | ❌ |
| mask=0,15 blok | 0,8037 | +0,0142 | 0,5372 | −0,0355 | 0,5779 | ❌ |
| mask=0,30 bağımsız-sütun | 0,5572 | −0,0388 | 0,4840 | −0,0887 | 0,5558 | ❌ |
| **mask=0,30 blok (en yakın)** | 0,5881 | −0,0313 | 0,5660 | **−0,0067** | 0,5585 | ❌ |
| Referans (mevcut) | — | — | 0,5727 | — | 0,5286 | — |

**Hiçbir aday `CAT_1` kritik-yön AUC'sini artırmadı — dördü de referansa
(0,5727) eşit ya da altında kaldı.** En yakın aday (`mask=0,30 blok`)
bile Δ=−0,0067 ile pratik olarak **hiç iyileşme yok**, gürültü bandının
(`<0,02`) içinde bile sayılabilir ama yön yine de yanlış (negatif).

### Karar Kuralı — Hiçbir Aday Geçmedi

1. **Anlamlı düşüş yok mu?** 3/4 adayda evet (NB p≥0,05), ama bu kriter
   tek başına yeterli değil.
2. **`CAT_1` shift AUC'si ≥0,10 artıyor mu?** **HİÇBİR adayda hayır** —
   en iyi aday bile Δ=−0,0067 (negatif yönde).
3. Worst-fold kriteri bazı adaylarda sağlanıyor olsa da (2. kriter
   sağlanmadığı için) önemsiz.

**KARAR: Hiçbir aday kriteri karşılamıyor — Adım 4'ün (tekrarlı-bölme
doğrulaması) gerekmesine bile gerek kalmadı, doğrudan DUR.** Mevcut
model korunuyor.

### Yorum — Beklenen ve Tutarlı Bir Sonuç

Bu bulgu, F1/F3'ün daha önceki bulgusuyla (madde "3-4 özelliği çıkarmak
`CAT_1` çöküşünü düzeltmiyor") ve seed-bagging denemesinin sonucuyla
**aynı kalıba** uyuyor: **`CAT_1` çöküşünün kök nedeni belirli
`AL_`-desenlerinin veya özelliklerin varlığı/yokluğu değil,
`CAT_1`-boş alt-kümesinin (132 satır) yapısal olarak farklı/az temsil
edilmiş olmasıdır** (bkz. F3 sonrası teşhis: `AL_` eksiklik oranı
%91,4 vs %36,6). Eğitim sırasında `AL_` maskelemek, modelin `AL_`
desenlerine daha az güvenmesini sağlayabilir (nitekim bazı adaylarda
specificity/worst-fold hafifçe iyileşti), ama `CAT_1`-boş alt-kümesinin
**temsil ettiği örneklem çeşitliliği eksikliğini** telafi etmiyor —
augmentasyon, eksik olan **bilgiyi** üretmiyor, yalnızca mevcut
bilgiyi gürültülü hale getiriyor. Bu, augmentasyon tasarımının veri-
seviyesinde (örn. gerçek `CAT_1`-boş çeşitliliğini artıracak yeni veri
toplama/paylaşımı) çözülmesi gerektiğini, salt maskeleme ile
çözülemeyeceğini gösteriyor.

**Toplam hesaplama süresi: 13 saniye** (4 aday × [5 OOF fit + 2
`CAT_1`-geçiş fit] = 28 CatBoost fit + referans 5+2 fit = 35 fit;
öngörülen >20dk eşiğinin çok altında).

### Kod / dosyalar

- `src/genova/pah/f0_missingness_augmentation.py` (yeni) —
  `reports/tables/f0_missingness_augmentation_grid.csv`.
- `f0_final_model.py::cross_fit_oof`, `f3_robustness_stress_tests.py::_score`,
  `models.py::fit_predict_catboost` DEĞİŞTİRİLMEDİ — yalnızca import
  edilip yeniden kullanıldı.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — hiçbir aday kriteri sağlamadığı için zaten otomatik bir
değişiklik söz konusu değildi.

---

## Final Gün Triyaj Sistemi — bkz. `FINAL_GUN_TRIYAJ_REHBERI.md`

> Yukarıdaki bulgular (`CAT_1`-boş kırılganlığının çözülemediği) artık
> "modeli düzeltme" değil, "kırılganlığı tespit edip şeffaf raporlama"
> gerektiriyor. `scripts/final_day_shift_check.py`'nin ürettiği ham
> metrikler, `scripts/final_day_triage_report.py` ile otomatik bir
> GREEN/YELLOW/RED triyaj raporuna çevrildi — **yalnızca bir rapor**,
> model/eşik/split'i hiçbir koşulda otomatik değiştirmez. Tam eşik
> tablosu, doğrulama senaryoları ve örnek çıktı için proje kökündeki
> `FINAL_GUN_TRIYAJ_REHBERI.md`'ye bakın.

---

## P0-3: Çoklu-Prevalans Eşik Sağlamlığı — ⚠️ DÜŞÜK-PREVALANS UCUNDA KIRILGANLIK, DURDUR/BİLDİR

> Şartnamedeki 100P/250B final kompozisyonu **"yaklaşık"** — gerçek final
> test seti farklı bir oranla gelebilir. Bu tur, frozen final bundle'ın
> (`final_model_bundle_v2.pkl`, eşik=0,35 **SABİT, bu turda hiç
> değiştirilmedi**) bu belirsizliğe karşı ne kadar sağlam olduğunu
> ölçüyor — model/eşik seçimi turu değil, **saf karakterizasyon**.
> Mekanizma: bundle'ın kendi deterministik `cross_fit_oof`'u (`f0_final_
> model.py`, değiştirilmedi) yeniden üretildi, aynı `sld` (kalibre +
> önsel-düzeltilmiş) olasılıklar üzerinde `f0_final_performance_card.py::
> monte_carlo_full_metrics` (değiştirilmedi, zaten `n_pathogenic`/
> `n_benign` parametreli, genel-amaçlı) her prevalans için farklı
> örnekleme oranıyla çağrıldı — yeniden model fit YOK. Toplam hesaplama
> süresi **43 saniye**.

### Adım 1-2 — 5 Prevalans Senaryosu (N=350 sabit, eşik=0,35 sabit, 2000 simülasyon/senaryo)

| Prevalans | n_pato/n_benign | F1 | %95 CI | MCC | Specificity | Sensitivity | Precision | FP (ort.) | FP %95 CI |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| **0,20** | 70/280 | **0,5591** | [0,4859; 0,6304] | 0,4362 | 0,7862 | 0,7196 | 0,4580 | 59,86 | [47; 73] |
| 0,25 | 88/262 | 0,6111 | [0,5463; 0,6769] | 0,4638 | 0,7863 | 0,7197 | 0,5319 | 55,99 | [43; 69] |
| 0,286 (şartname) | 100/250 | 0,6395 | [0,5738; 0,7000] | 0,4785 | 0,7862 | 0,7212 | 0,5753 | 53,45 | [41; 67] |
| 0,33 | 116/234 | 0,6700 | [0,6080; 0,7295] | 0,4922 | 0,7868 | 0,7205 | 0,6271 | 49,90 | [38; 62] |
| 0,38 | 133/217 | 0,6959 | [0,6377; 0,7527] | 0,5009 | 0,7863 | 0,7199 | 0,6745 | 46,37 | [35; 58] |

**Çapraz-doğrulama:** 0,286 satırı (F1=0,6395, CI=[0,5738; 0,7000]) `f0_
uncertainty_analysis.csv`'nin `monte_carlo_100_250` sonucuyla **bit-bit
aynı** (aynı seed=42, aynı resampling şeması, `monte_carlo_full_metrics`
üzerinden bağımsız bir çağrı yoluyla) — bu turun mekanizması doğrulandı.
FP≈53,45 de önceki turun bağımsız FP≈53,28 tahminiyle aynı mertebede
(farklı türetme yolu: burada specificity'den, orada doğrudan sayımdan).

**Sağlık kontrolü:** Specificity (0,786-0,787) ve sensitivity
(0,7196-0,7212) **5 senaryoda da neredeyse sabit** — beklenen, çünkü
ikisi de sınıf-koşullu metrikler, prevalans karışım oranından bağımsız
olmalı. Yalnızca F1/precision (prevalansa doğrudan bağlı) anlamlı
değişiyor — bu, simülasyonun doğru çalıştığının bağımsız bir kanıtı.

### Adım 3 — Ortalama/Worst-Case F1 ve Senaryo-Özel Optimal Eşik

- **Ortalama F1 (5 senaryo): 0,6351**
- **Worst-case F1: 0,5591 (prevalans=0,20)** — en kırılgan senaryo, en
  düşük patojenik oranı (en fazla benign ağırlığı).
- F1 aralığı: 0,5591-0,6959 = **0,1368** (task'ın "sağlam" eşiği olan
  <0,05'in ~2,7 katı).
- **Senaryo-özel optimal eşik (yalnızca karşılaştırma, gerçek eşik
  değişmedi):** 0,25 / 0,286 / 0,33 / 0,38 prevalanslarının **hepsinde**
  optimal eşik = **0,35** — frozen eşikle **birebir aynı** (fark=0).
  Yalnızca **prevalans=0,20'de** optimal eşik **0,48**'e kayıyor
  (fark=**0,13**) — düşük-patojenik-oranlı senaryoda model, sabit
  0,35 eşiğiyle gereğinden fazla "cömert" (pathogenic'e çok kolay karar
  veriyor), bu da precision'ı (0,458) ve dolayısıyla F1'i çöktürüyor.

### Adım 4 — Karar: KARIŞIK, uçta ciddi kırılganlık

**Genel resim iki parçalı:**

1. **Prevalans ≥0,25 aralığında (şartnamenin kendi 0,286'sı dahil, en
   olası aralık): sağlam.** Senaryo-özel optimal eşik 4/4 senaryoda
   frozen 0,35'le birebir örtüşüyor (fark=0) — "bu prevalansı önceden
   bilseydik farklı bir eşik seçer miydik?" sorusunun cevabı bu aralıkta
   **hayır**. F1 bu 4 senaryoda 0,6111-0,6959 arasında, hepsi final-
   benign stres bandının (0,62-0,64) içinde veya üstünde.
2. **Prevalans=0,20 ucunda: ciddi kırılganlık.** Worst-case F1 (0,5591),
   final-benign stres bandının alt sınırından (0,62) **0,061 aşağıda**
   — task'ın kendi "ciddi kırılganlık" örnek eşiğini (>0,05 fark) az
   farkla aşıyor. Aynı noktada eşik-farkı da (0,13) diğer 4 senaryonun
   aksine sıfır değil — bu, tek bir uçta gerçek, ölçülebilir bir
   kalibrasyon-uyumsuzluğu, gürültü değil.

**Sonuç: bu görev tanımının "ciddi kırılganlık" dalına düşüyor, ama
YALNIZCA ızgaranın en uç noktasında (P(Pathogenic)=0,20) — orta/üst
aralık (şartnamenin kendi beklentisi dahil) sağlam.** Talimat gereği
**eşik bu turda DEĞİŞTİRİLMEDİ** (final öncesi son anda risk almamak
için) — bulgu burada belgeleniyor, final günü triyaj sistemine bir not
eklendi (aşağıya bakın). **DURDUR — bu bulgu için kullanıcı onayı/kararı
bekleniyor**, otomatik bir aksiyon alınmadı.

### Final Günü Triyaj Notu (eklendi, bkz. `FINAL_GUN_TRIYAJ_REHBERI.md`)

Mevcut 5-metrik GREEN/YELLOW/RED tablosu **veri-kayması** (`CAT_1`/`AL_`/
`al_all_missing`/adversarial-AUC) ölçüyor, **prevalans-belirsizliği**
ölçmüyor — bu farklı bir risk ekseni, mevcut tabloya yeni bir metrik
olarak eklenmedi (eşik seçimi gerektirir, bu turun kapsamı dışı, ayrı
onay gerektirir). Bunun yerine metinsel bir "Bilinen Sınırlama" notu
eklendi: final test setinde gözlenen patojenik oranı belirgin şekilde
düşükse (kabaca **%20'nin altına** yaklaşıyorsa), frozen 0,35 eşiğinin
o bölgede ölçülebilir biçimde optimal-altı olduğu (P0-3 bulgusu)
hatırlatılıyor — **triyaj sistemi eşiği otomatik DEĞİŞTİRMEZ**, yalnızca
bu bilinen sınırlamayı görünür kılar.

### Kod / dosyalar

- `src/genova/pah/f0_multi_prevalence_threshold_robustness.py` (yeni) —
  `f0_final_model.py::cross_fit_oof`, `f0_final_performance_card.py::
  monte_carlo_full_metrics`, `e5_threshold_selection.py::_prior_weights`/
  `THRESHOLD_GRID`, `genova.metrics::f1_binary_positive_weighted`
  DEĞİŞTİRİLMEDİ — yalnızca import edilip yeniden kullanıldı.
- `reports/tables/f0_multi_prevalence_threshold_robustness.csv` (yeni).
- `FINAL_GUN_TRIYAJ_REHBERI.md` — "Bilinen Sınırlama" notu eklendi.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına dokunulmadı —
eşik dahil hiçbir parametre değiştirilmedi. `pytest tests/ -q` →
182/182 geçti (bu tur test eklemedi, mevcut regresyon korundu).

**P0-3 TAMAMLANDI (bulgu belgelendi, uçtaki kırılganlık için DURDUR/
bildir — aksiyon kullanıcı kararına bırakıldı).**

---

## P0-3 Takip: Minimax Sağlam Eşik — ⚠️ TRADE-OFF KAÇINILMAZ, DURDUR/BİLDİR

> Test verisine **hiç bakılmadı** — yalnızca mevcut eğitim OOF olasılıkları
> + genişletilmiş prevalans ızgarası. Bu, "test görmeden önce donan bir
> karar" kategorisinde, final günü kuralını ihlal etmiyor. `final_model_
> bundle_v2.pkl`'nin eşiği bu turda **DEĞİŞTİRİLMEDİ** — yalnızca
> araştırma/öneri üretildi. Mekanizma: P0-3 ile aynı bundle/OOF/`sld`
> kurulumu + `f0_final_performance_card.py::monte_carlo_full_metrics`
> (değiştirilmedi) — kaba 8×9 ızgara hızlı `n_simulations=400` ile
> tarandı (sıralama için yeterli), sonra yalnızca seçilen aday(lar) P0-3
> ile AYNI hassasiyette (`n_simulations=2000`) yeniden hesaplanıp resmi
> sayılar buradan alındı (kaba taramanın gürültülü sayıları rapora
> yazılmadı). Toplam hesaplama: kaba tarama ~365s + aday doğrulama ~420s
> (iki ayrı çalıştırma, ~13 dk toplam — >10dk sınırını hafifçe aştı,
> bilgi amaçlı not ediliyor).

### Adım 1-2 — 8×9 Izgara Özeti (worst-case F1 her zaman prevalans=0,15'te)

| Eşik | Worst-case F1 (kaba, n=400) | F1 @ 0,286 (kaba) | MCC @ 0,286 (kaba) |
|---|---:|---:|---:|
| 0,30 | 0,4210 | 0,5963 | 0,4039 |
| 0,33 | 0,4366 | 0,6027 | 0,4157 |
| **0,35 (mevcut)** | 0,4910 | 0,6396 | 0,4788 |
| 0,38 | 0,4740 | 0,6179 | 0,4484 |
| 0,40 | 0,4571 | 0,5995 | 0,4231 |
| 0,43 | 0,4718 | 0,6018 | 0,4356 |
| 0,45 | 0,4866 | 0,6031 | 0,4495 |
| **0,48** | **0,5365** | 0,6168 | 0,5042 |
| 0,50 | 0,5129 | 0,5906 | 0,4762 |

İlginç bir yan-gözlem: 0,35, komşuları 0,33/0,38'e göre yerel bir tepe
noktası — ızgara pürüzsüz/monoton değil (kaba tarama gürültüsü + gerçek
F1 doğrusallıksızlığı karışımı olabilir). Genel eğilim yine de net:
**worst-case F1, 0,45'in ötesinde belirgin biçimde yükseliyor.**

### Adım 3 — Minimax Eşik Seçimi (aday doğrulama, n=2000)

1. **Saf minimax:** eşik=**0,48**, worst-case F1=**0,5345** (mevcut
   0,35'in worst-case F1'i 0,4900 — kazanç **+0,0445**).
2. **Dengeli minimax:** ızgaradaki **hiçbir eşik** her iki kriteri birden
   (worst-case kazancı ≥0,03 VE 0,286'daki F1 kaybı ≤0,02) karşılamadı.
   0,48 worst-case kriterini rahatça geçiyor (+0,0445≥0,03) ama 0,286'daki
   F1 kaybı **0,0244** — 0,02 sınırını **0,0044 ile aşıyor**, sınıra çok
   yakın ama teknik olarak "dengeli" tanımına girmiyor.

### Adım 4 — Karar: Dal 3 (trade-off kaçınılmaz)

Hiçbir eşik hem worst-case'i iyileştirip hem 0,286'daki performansı tam
koruyamadığı için **Dal 3** geçerli: aşağıda 0,35 (mevcut) vs 0,48 (saf
minimax aday) için tam trade-off tablosu, **DURDUR — bu bir mühendislik
tercih kararı, otomatik seçilmiyor.**

### Tam Karşılaştırma Tablosu — 0,35 (mevcut) vs 0,48 (aday), 8 senaryo

| Prevalans | F1 (0,35→0,48) | MCC (0,35→0,48) | Specificity (0,35→0,48) | Sensitivity (0,35→0,48) | Precision (0,35→0,48) |
|---|---|---|---|---|---|
| 0,15 | 0,4900→**0,5345** | 0,3988→**0,4544** | 0,7866→**0,9182** | 0,7207→0,5367 | 0,3719→0,5358 |
| 0,18 | 0,5347→**0,5597** | 0,4229→**0,4709** | 0,7864→**0,9179** | 0,7195→0,5350 | 0,4262→0,5904 |
| 0,20 | 0,5591→**0,5735** | 0,4362→**0,4799** | 0,7862→**0,9179** | 0,7196→0,5351 | 0,4580→0,6214 |
| 0,22 | 0,5815→**0,5861** | 0,4487→**0,4880** | 0,7865→**0,9179** | 0,7200→0,5361 | 0,4886→0,6498 |
| 0,25 | **0,6111**→0,6013 | 0,4638→**0,4965** | 0,7863→**0,9183** | 0,7197→0,5355 | 0,5319→0,6891 |
| **0,286 (şartname)** | **0,6395**→0,6151 | 0,4785→**0,5026** | 0,7862→**0,9181** | 0,7212→0,5361 | 0,5753→0,7247 |
| 0,33 | **0,6700**→0,6292 | 0,4922→**0,5065** | 0,7868→**0,9183** | 0,7205→0,5356 | 0,6271→0,7658 |
| 0,38 | **0,6959**→0,6408 | 0,5009→**0,5057** | 0,7863→**0,9180** | 0,7199→0,5354 | 0,6745→0,8009 |

**Kalın** = o hücrede daha iyi olan taraf.

**Kritik ek gözlem — task'ın kendi kriterlerinin ölçmediği bir maliyet:**
0,48'e geçiş **sensitivity'yi 0,72'den ~0,54'e düşürüyor** (mutlak
**~0,18 kayıp, ~%25 göreli kayıp**) — yani gerçek patojenik varyantların
belirgin daha büyük bir kısmı kaçırılıyor, specificity'deki büyük kazanç
(0,786→0,918) karşılığında. MCC hemen hemen her senaryoda 0,48 lehine
(sınıflar arası genel ayrım gücü artıyor), ama F1/sensitivity dengesi
şartnamenin resmi metriği (pozitif-sınıf F1) ve klinik önem (patojenik
kaçırmama) açısından 0,35'i orta/üst prevalans aralığında hâlâ daha
savunulabilir kılıyor. **Bu, Adım 3/4'ün F1-odaklı kriterlerinin
görmediği, kararı etkileyebilecek ek bir boyut — kullanıcıya ayrıca
bildiriliyor.**

**Ayrıca bir kapsam sınırlaması:** worst-case her zaman ızgaranın en
düşük ucunda (P=0,15) — ızgara daha da aşağı genişletilseydi (ör. 0,10)
worst-case muhtemelen daha da kötüleşirdi; bu minimax sonucu 0,15 alt
sınırına göre koşullu, "mutlak" bir worst-case değil.

### Kod / dosyalar

- `src/genova/pah/f0_minimax_threshold_search.py` (yeni) — `f0_final_
  performance_card.py::monte_carlo_full_metrics`, `f0_multi_prevalence_
  threshold_robustness.py::_scenario_sample_sizes`, `f0_final_model.py::
  cross_fit_oof` DEĞİŞTİRİLMEDİ — yalnızca import edilip yeniden
  kullanıldı.
- `reports/tables/f0_minimax_threshold_search.csv` (kaba 8×9 ızgara),
  `reports/tables/f0_minimax_threshold_candidate_comparison.csv` (0,35 vs
  0,48 tam karşılaştırma, n=2000) — yeni.

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına dokunulmadı —
eşik dahil hiçbir parametre değiştirilmedi, yalnızca araştırma yapıldı.
Test verisi bu turda hiç kullanılmadı/görülmedi.

**P0-3 Takip TAMAMLANDI (araştırma bitti, trade-off net biçimde
belgelendi — eşik değişikliği kararı kullanıcıya bırakıldı, DURDUR).**

---

## Persistent-Hard-Benign Hata Analizi (İnceleme Raporu Bölüm 34) — saf teşhis, karar yok

> 61 benign örneğin her biri, mevcut 50-outer-fold split bankasının
> (10 tekrar × 5 dış fold) kendi dış-test'ine düştüğü her fold'da nasıl
> skorlandı — `reports/tables/e4_prior_corrected_probabilities.csv`'nin
> (catboost/v4_from_v2/beta_sld, E3→E4 hattının zaten ürettiği) satırlarından
> **doğrudan okundu**, hiçbir model yeniden fit edilmedi. Sabit final eşik
> (bundle'dan okunur, 0,35) her fold'un SLD-düzeltilmiş olasılığına
> uygulandı. SHAP, F4 ile **aynı mekanizma** (`shap.TreeExplainer`, final-
> fit modeli değiştirilmeden) ile yalnızca persistent-hard satırlar için
> yeniden çalıştırıldı — F4 yalnızca agrege `mean|SHAP|` CSV'sini
> saklamıştı, per-satır ham değerleri değil. Toplam hesaplama süresi
> **8 saniye**.

### Adım 1-2 — Kategori Büyüklükleri

| Kategori | n | Oran |
|---|---:|---:|
| Easy (error_rate≤0,10) | 41 | %67,2 |
| Unstable (0,10<error_rate<0,70) | 11 | %18,0 |
| **Persistent-hard (error_rate≥0,70)** | **9** | **%14,8** |

Easy grubunun `mean_P` aralığı [0,0046; 0,2786] — hepsi eşikten (0,35) rahat
uzakta, beklenen. Toplam 61 = 41+11+9, doğrulandı.

### Adım 3 — Persistent-Hard Grubu (n=9, tam liste)

| Variant_ID | error_rate | mean_P | std_P | CAT_1 | CAT_2 | AL_ tertile | SHAP top-2 (final-fit, in-sample) |
|---|---:|---:|---:|---|---|---|---|
| VAR_002693 | 1,00 | 0,506 | 0,097 | boş | EMPTY | yüksek | EK_7 (−0,65), EK_6 (−0,38) |
| VAR_002776 | 1,00 | 0,494 | 0,083 | dolu | EMPTY | orta | AL_7 (−0,68), AL_330 (−0,42) |
| VAR_002857 | 1,00 | 0,514 | 0,041 | dolu | AllofUs_AFR | düşük | AL_329 (−0,45), AL_26 (−0,41) |
| VAR_002978 | 1,00 | 0,765 | 0,087 | boş | AllofUs_EUR | yüksek | AL_7 (−0,31), AL_300 (+0,29) |
| VAR_003057 | 1,00 | 0,563 | 0,040 | dolu | EMPTY | orta | AL_7 (−0,48), AL_330 (−0,40) |
| VAR_002951 | 0,90 | 0,526 | 0,120 | dolu | EMPTY | orta | AL_7 (−0,72), AL_329 (−0,54) |
| VAR_003072 | 0,90 | 0,425 | 0,081 | dolu | AllofUs_AFR | düşük | AL_7 (−0,68), AL_329 (−0,48) |
| VAR_002554 | 0,80 | 0,454 | 0,098 | dolu | AllofUs_AFR | düşük | EK_3 (−0,66), AL_300 (−0,57) |
| VAR_002586 | 0,80 | 0,453 | 0,094 | dolu | EMPTY | orta | AL_300 (−0,58), AL_7 (+0,38) |

**Not (yorumlama sınırı, F4'ün kendi in-sample uyarısıyla aynı):** SHAP
burada final-fit modelin **kendi eğitim verisindeki** (369 satırın tamamı,
in-sample) açıklaması — bu satırların OOF hata istatistiği ise held-out
(nested, 50-fold) tahminlerden geliyor. İkisi aynı olayın iki farklı
görünümü: SHAP, "model bu satırı tam veriyle görünce neden böyle karar
veriyor"ya cevap veriyor; error_rate, "model bu satırı görmeden bırakılınca
ne kadar sık yanlış tahmin ediyor"ya. Doğrudan bire-bir eşleştirilmemeli.

### Ortak Yapısal Özellik Var mı? — ZAYIF/KARIŞIK, tek bir eksen yok

| Özellik | Persistent-hard (n=9) | Tüm 61 benign (taban oran) | Zenginleşme |
|---|---:|---:|---:|
| CAT_1 = dolu | %77,8 (7/9) | %63,9 | hafif (×1,2) |
| CAT_2 = AllofUs_AFR | **%33,3 (3/9)** | %9,8 | **×3,4 — dikkat çekici ama n=3 çok küçük** |
| AL_ tertile = orta | %44,4 | %55,7 | yok (altında) |
| AL_ tertile = düşük | %33,3 | %26,2 | hafif |
| AL_ tertile = yüksek | %22,2 | %18,0 | hafif |

**Hiçbir kategori %70-80'in üzerinde bir konsantrasyon göstermiyor** —
grup ne tamamen `CAT_1`-boş ne tamamen `CAT_1`-dolu, ne tek bir `CAT_2`
kategorisinden, ne tek bir `AL_` tertile'ından. En dikkat çekici sinyal
`AllofUs_AFR`'nin ×3,4 zenginleşmesi, ama yalnızca 3 satırlık bir dilim
— **gösterge niteliğinde, kesin bir yapısal ayrım değil**.

**SHAP tarafında bir örüntü var:** `AL_7`, 9 satırın **9'unda da** top-5
içinde — ama bu F4'ün zaten kurduğu global sıralamada `AL_7`'nin #1/26
olmasıyla tutarlı (ayrıcalıklı değil, beklenen). Daha dikkat çekici olan,
`AL_300`'ün 7/9'da ve genel olarak `AL_301/317/319/329/330/331/334`
("AL_-3xx ailesi") kümesinin tekrar tekrar birlikte görülmesi — F1/F3'ün
riskli-4 listesinden (`AL_26/12/7/49`) farklı ama örtüşen bir küme. Bu,
ayrı bir araştırma konusu olarak not ediliyor, bu turda ileri sürülmüyor.

### Adım 4 — Sonuç: Ne Tam "Hedefli Müdahale İçin Kanıt" Ne Tam "Genel/Yaygın"

Grup **boyut olarak** (n=9≤10, %14,8) hedefli-müdahale eşiğini karşılıyor
— görevin kendi tanımına göre "küçük" sayılır. Ama **yapısal olarak**
tek bir ortak özellik taşımıyor (yukarıdaki tablo) — bu, task'ın iki net
dalından ("küçük + ortak özellik → hedefli müdahale" / "büyük + dağınık →
genel sorun") **hiçbirine tam uymuyor**, ara bir durum:

- Görevin "hedefli müdahale" senaryosu (küçük grup + net ortak özellik,
  örn. domain-conditional threshold) **bu haliyle desteklenmiyor** —
  `CAT_1` veya tek bir `CAT_2` kategorisine göre koşullu bir eşik, bu 9
  satırın çoğunu kapsamayacaktır (yalnızca 3/9 `AllofUs_AFR`, 7/9 `CAT_1`-
  dolu ama bu zaten çoğunluk grubu).
- Ama tamamen "genel/yaygın" da değil — grup küçük (%14,8, >15 satır/%25
  eşiğinin altında) ve `AllofUs_AFR` zenginleşmesi + `AL_7`/`AL_300`-ailesi
  SHAP tekrarı, körü körüne rastgele değil, izlenmeye değer zayıf sinyaller.

**Pratik sonuç:** bu 9 satır, mevcut modelin "kalıcı olarak yanlış anladığı"
gerçek bir alt-küme (error_rate'leri 0,80-1,00, rastgele gürültüyle
açıklanamayacak kadar tutarlı) — ama tek-eksenli, basit bir domain-
conditional-threshold müdahalesiyle temiz biçimde hedeflenemez. Daha ileri
bir müdahale (varsa) `AllofUs_AFR` alt-grubuna özel bir teşhis ya da
bu 9 satırın kendi özgün profiline (feature-value bazlı, tertile/kategori
bazlı değil) bakan bir yaklaşım gerektirebilir — bu, ayrı bir görev.

### Unstable Grubu (n=11) — `CAT_1`/`AL_` İlişkisi

| Özellik | Unstable (n=11) | Tüm 61 benign |
|---|---:|---:|
| CAT_1 = dolu | **%90,9 (10/11)** | %63,9 — belirgin zenginleşme |
| AL_ tertile = düşük | **%63,6 (7/11)** | %26,2 — belirgin zenginleşme |

Unstable grubu, persistent-hard'ın aksine **net bir örüntü gösteriyor**:
neredeyse tamamı `CAT_1`-dolu VE `AL_`-düşük-eksiklik (en "bilgi-zengin"
alt-küme). Mekanizma açık: bu satırların `mean_P`'si **0,24-0,36**
aralığında — tam olarak final eşiğin (0,35) etrafında, sınırda. Std_P
(0,06-0,13) küçük ama fold-arası kalibrasyon gürültüsü bu dar bandı
eşiğin iki yanına ittirip tekrar getiriyor. Bu, bir domain/provenance
sorunu değil, **beklenen bir eşik-sınırı hassasiyeti** — istikrarsızlığın
kaynağı yapısal değil, olasılık-eşik yakınlığı.

### Kod / dosyalar

- `src/genova/pah/f_persistent_hard_benign_analysis.py` (yeni) —
  `e4_prior_corrected_probabilities.csv`, `f4_shap_explainability.py`
  ile aynı `shap.TreeExplainer`/`fold_versions.build_v4_from_v2`
  mekanizması DEĞİŞTİRİLMEDİ — yalnızca import edilip yeniden kullanıldı.
- `reports/tables/f_persistent_hard_benign_error_stats.csv` (yeni, 61
  satır, tüm kategoriler + yapısal özellikler).

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına dokunulmadı —
bu saf bir teşhis turuydu, hiçbir karar kuralı uygulanmadı.

**Persistent-Hard-Benign Hata Analizi TAMAMLANDI (kanıt toplandı, sonraki
adıma karar bu turda verilmedi — görev tanımına uygun).**

---

## Preprocessing Paketleme Denetimi — saf denetim, karar yok

> `final_model_bundle_v2.pkl` yalnızca **okundu**, hiçbir bayt
> değiştirilmedi. `predict.py` **hiç düzenlenmedi** — yalnızca kendi
> fonksiyonları (`apply_v2_preprocessing`, `predict_dataframe`, `validate_
> predict_schema`) yeni, ayrı bir izleme scriptinden import edilip
> çağrıldı. Bu bir kod-değişikliği turu değil.

### Adım 1 — Bundle Tam Envanteri (17 anahtar)

| Anahtar | Tip | Boyut | Örnek/not |
|---|---|---:|---|
| `v2_steps` | list | 4 | aşağıda detaylı |
| `cat_categories` | dict | 7 | `CAT_1..CAT_5, AA_1, AA_2` → her biri için fit-zamanlı kategori listesi |
| `al_cols` | list | 334 | `AL_1..AL_334` (ham isim listesi, bilgi amaçlı) |
| `ek_cols` | list | 9 | `EK_1..EK_9` |
| `v2_feature_cols` | list | 353 | `al_cols+ek_cols+CAT_ALL+3 türetilmiş` — **predict.py tarafından fiilen kullanılmıyor**, yalnızca provenance/bilgi amaçlı |
| `pool` | list | 26 | final özellik listesi, **zaten alfabetik sıralı** (`AL_12,AL_141,...,EK_9`) |
| `model` | CatBoostClassifier | — | fitted final model (369 satır) |
| `model_cat_features` | list | **0** | final pool'da hiç `CAT_`/`AA_` yok → boş |
| `model_best_params` | dict | 2 | `{depth:5, learning_rate:0.05}` |
| `calibrator_name` | str | — | `"beta"` |
| `calibrator` | BetaCalibrator | — | `.model_` → `betacal.BetaCalibration(parameters="abm")`, fitted |
| `w1` / `w0` | float | — | 0,343337 / 4,275449 |
| `threshold` | float | — | 0,35 |
| `final_pathogenic_prior` / `train_pathogenic_prior` | float | — | 0,286 / 0,833 |
| `metadata` | dict | 8 | model adı, veri versiyonu, ağırlıklandırma, satır/özellik sayısı, `removed_for_provenance_risk` |

`v2_steps` (4 fitted transformer, hepsi `genova.pah.missingness`'ten):

| # | Sınıf | Kolon(lar) | Fitted parametre |
|---|---|---|---|
| 0 | `BlockMissingIndicator` | 334 `AL_` | `name="al_all_missing"` — durumsuz, yapısal |
| 1 | `BlockMissingIndicator` | 11 `EK_` (EK_3 hariç) | `name="ek_cat_block_missing"` — durumsuz, yapısal |
| 2 | `MedianImputerWithIndicator` | `EK_3` | `medians_={"EK_3": 3,4801...}` — **eğitimden öğrenilmiş, bundle'da saklı** |
| 3 | `ConstantFillImputer` | 334 `AL_` | `strategy="zero", fill_value=0.0` — sabit ama nesnenin kendi `__dict__`'inde, `predict.py`'de değil |

### Adım 2 — 10 Maddelik Checklist

| # | Madde | Durum | Not |
|---|---|---|---|
| 1 | Özellik listesi+sırası (26) | **✅ Bundle'da** | `bundle["pool"]`, zaten alfabetik. Hem `f0_final_model_v2.py`'nin eğitim-zamanı `build_v4_from_v2` çağrısı hem `predict.py`'nin `apply_v2_preprocessing`'i **aynı** `[c for c in pool if c in df.columns]` deyimini kullanıyor — sıra tutarlılığı tasarım gereği garanti, tesadüf değil. |
| 2 | `AL_` sütun taksonomisi (log1p/logit) | **⚠️ MADDENİN ÖNCÜLÜ GERÇEK BUNDLE'A UYMUYOR** | Bu ayrım (`classify_al_columns`, log1p/logit) yalnızca **v3/v4_from_v3** dalında var (`fold_versions.py::build_v3_full`) — final bundle **v4_from_v2** dalından geliyor, orada TÜM `AL_` sütunlarına tek tip `ConstantFillImputer(strategy="zero")` uygulanıyor, taksonomi/log-dönüşüm YOK. Yani bu madde için "eksik" değil, "sorulan şey final modelde zaten mevcut değil" — yanlış anlaşılmayı önlemek için burada açıkça düzeltiliyor. |
| 3 | Türetilmiş özellikler (`al_all_missing`, `EK_3_missing`) | **✅ Bundle'da, tekrarlanabilir** | `v2_steps`'in [0]/[2] adımlarında üretiliyor, saf/durumsuz mantık (kod `missingness.py`'de, parametreler nesnenin kendisinde). İkisi de final 26-özellik pool'da **yok** (provenance/seçim nedeniyle), üretilip sonra filtreleniyorlar — israf ama zararsız. |
| 4 | Eksik değer doldurma | **✅ Bundle'da** | `EK_3` medyanı (3,4801...) ve `AL_*` sabit-0 stratejisi ikisi de fitted nesnelerin `__dict__`'inde, serileştirildiğinde bundle'la birlikte taşınıyor. |
| 5 | Kategorik kodlama (`model_cat_features`) | **✅ Bundle'da, tutarlı** | `model_cat_features=[]` çünkü final pool'da hiç `CAT_`/`AA_` yok. Bu liste hem eğitimde (`_fit_catboost_final`) hem `predict.py`'de **aynı kaynaktan** (`_category_columns(X_all)` → bundle'a yazılır → `predict.py` okur) geliyor — ayrık/çelişen bir tanım yok. |
| 6 | Feature pool reduction (madde 11) | **✅ Bundle'da (dolaylı)** | `f0_final_feature_pool_v2.json` yalnızca **eğitim zamanında** (`f0_final_model_v2.py`) okunuyor — sonucu (26 isim) `bundle["pool"]`'a yazılıp orada donuyor. `predict.py` bu JSON dosyasına **hiç bağımlı değil**, yalnızca bundle'ı okuyor — jürinin ortamında bu JSON dosyası olmasa bile `predict.py` çalışır. |
| 7 | Beta kalibrasyon parametreleri | **✅ Bundle'da** | `bundle["calibrator"]` — fitted `BetaCalibrator` nesnesinin kendisi (yalnızca a/b katsayıları değil), `.transform()` doğrudan çağrılabilir. |
| 8 | Bayes önsel-düzeltme (w1/w0) | **✅ Bundle'da** | `bundle["w1"]`/`bundle["w0"]` — `predict.py` `e4_prior_correction.py`'nin modül-seviyesi `W1`/`W0` sabitlerini **kullanmıyor**, yalnızca `sld_correct()` fonksiyonunu (saf formül) import edip bundle'ın kendi sayılarını geçiriyor. |
| 9 | Eşik (0,35) | **✅ Bundle'da** | `bundle["threshold"]`, kodda hiçbir yerde `0.35` hardcoded değil. |
| 10 | Sütun adı/tip beklentisi | **⚠️ KISMEN — yalnızca isim/sayı, tip DEĞİL** | `validate_predict_schema` yalnızca kolon **varlığını/sayısını** (`AL:334,CAT:6,EK:9,AA:2`) kontrol ediyor, **dtype kontrolü yok**. Adım 3'te ampirik test edildi — bkz. aşağıda. |

### Adım 3 — Uçtan Uca İzleme Testi

`f_predict_pipeline_trace.py`, ham eğitim verisinden 5 satırlık gerçek bir
örnekle `predict.py`'nin **kendi** fonksiyonlarını adım adım çağırdı:

```
Adim A (sema):          OK
Adim B (v2_steps):      al_all_missing/ek_cat_block_missing/EK_3_missing uretildi;
                         EK_3 medyanla (3,48...), AL_* sifirla dolduruldu (once 5 NaN -> sonra 0)
Adim C (cat_categories): CAT_1-5/AA_1-2 uygulandi ama HICBIRI final pool'da degil
Adim D (pool filtresi):  cikti (5,26), kolon sirasi bundle['pool']'la BIREBIR AYNI (True)
Adim E (stringify):      model_cat_features=[] -> X degismedi (True)
Adim F (model->esik):    5 satir icin proba/Label uretildi, bundle esigi(0,35)/w1/w0 kullanildi
```

**Beklenen 10-maddelik zincirle birebir eşleşti** — hiçbir sürpriz adım,
atlanan adım veya sırası bozuk adım bulunmadı.

**Ek ampirik dtype-risk testi (madde 10'un doğrudan doğrulaması):**

| Senaryo | Sonuç |
|---|---|
| `AL_7` string ama sayı-görünümlü ("0.123" tarzı) | Hata YOK, sonuç değişmedi — pandas/CatBoost sessizce sayıya çeviriyor |
| Final pool-dışı bir `CAT_` sütunu yanlış tipte (numerik) | Hata YOK — zaten pool'a hiç girmiyor, etkisiz |
| `AL_7` gerçekten sayısal-olmayan çöp string içeriyor | **`CatBoostError: Cannot convert 'gecersiz_deger' to float`** — açık, sert hata |

**Sonuç: gerçek bir "sessizce yanlış sonuç" riski bulunamadı** — dtype
kontrolü olmasa da, gerçekten bozuk bir değer CatBoost katmanında net bir
hatayla durur (silent-wrong-result değil, fail-loud). Risk yalnızca
**kullanılabilirlik**: hata mesajı erken/dostça bir şema hatası yerine
CatBoost'un iç `feature_idx`/`doc_idx` diliyle geliyor.

### Adım 4 — Riskli/Eksik Noktalar ve Öneriler (uygulanmadı, yalnızca öneri)

| # | Bulgu | Risk seviyesi | Öneri |
|---|---|---|---|
| A | Madde 10: `validate_predict_schema` dtype kontrolü yapmıyor | **Düşük** (gerçek bozuk veri zaten CatBoost katmanında sert hatayla duruyor, sessiz yanlış sonuç riski gözlenmedi) | İsteğe bağlı iyileştirme: `AL_`/`EK_` sütunlarının `pd.to_numeric` ile dönüştürülebilir olduğunu erken, dostça bir hata mesajıyla kontrol etmek — kozmetik, acil değil. |
| B | Madde 2: görev talimatının log1p/logit taksonomi öncülü final bundle'a uymuyor | **Risk değil, dokümantasyon netliği** | Bu raporun bu bölümü zaten netleştiriyor; ayrıca `predict.py`'nin docstring'ine tek satırlık bir not ("v2 hattı, log/logit dönüşüm YOK") eklenmesi ileride karışıklığı önler — küçük, isteğe bağlı. |
| C | `v2_feature_cols`/`al_cols`/`ek_cols` bundle'da duruyor ama `predict.py` tarafından hiç okunmuyor | **Risk değil** (ölü/kullanılmayan bilgi, zararsız) | Aksiyon gerekmez — provenance/hata-ayıklama amaçlı tutulmaları makul, temizlik isteğe bağlı ve düşük öncelikli. |

**Genel sonuç: paketleme sağlam.** 10 maddenin 8'i tam bundle-içi, 1'i
(madde 2) görev öncülünün gerçek pipeline'a uymadığı bir yanlış-varsayım
(düzeltildi), 1'i (madde 10) düşük riskli bir kullanılabilirlik boşluğu
— **hiçbiri "jürinin farklı bir ortamda sessizce farklı sonuç alması"**
türünden kritik bir sızıntı değil. Hiçbir kod değişikliği yapılmadı;
madde A/B'nin önerileri onay bekliyor.

### Kod / dosyalar

- `src/genova/pah/f_predict_pipeline_trace.py` (yeni) — `predict.py`
  DEĞİŞTİRİLMEDİ, yalnızca onun fonksiyonları import edilip izlendi.
- `reports/tables/` altına yeni bir dosya YAZILMADI — bu tur saf
  konsol-çıktılı bir denetim, kalıcı bir sayısal sonuç üretmedi.

`final_model_bundle_v2.pkl` yalnızca okundu (hiç `joblib.dump` çağrılmadı),
`predict.py`/split bankasına dokunulmadı.

**Preprocessing Paketleme Denetimi TAMAMLANDI (kod değişikliği yok, iki
küçük/düşük-riskli iyileştirme önerisi onay bekliyor).**

---

## Final Freeze Paketi (Jüri Yeniden-Çalıştırma Hazırlığı)

> Şartname madde 7.5: jüri finalist takımların kodunu yeniden çalıştırıp
> beyan edilen sonuçları doğrulayabilir. Bu tur üç boşluğu kapatıyor:
> madde 11'in reprodüksiyon açığı, eksik final-freeze artefaktları, ve
> **hiç doğrulanmamış** temiz-ortam tutarlılığı. `final_model_bundle_
> v2.pkl`/`predict.py`/split bankası yalnızca **okundu/kopyalandı**,
> orijinalleri değiştirilmedi. Model/eşik/hiperparametre değişikliği
> **yok** — tamamen paketleme/reprodüksiyon işi.

### Adım 1 — `f0_madde11_pool_reduction.py`

Yeni script, kaynağı (`f0_final_feature_pool.json`, 28 özellik) okuyup
`al_all_missing`/`CAT_1`'i çıkarıyor. Çalıştırıldı:

```
kaynak havuz: f0_final_feature_pool.json (28 ozellik)
indirgenmis havuz: 26 ozellik (cikarilan: ['al_all_missing', 'CAT_1'])
DOGRULANDI: uretilen icerik, mevcut f0_final_feature_pool_v2.json ile BIREBIR AYNI.
```

**Sonuç: birebir eşleşti** — reprodüksiyon açığı kapatıldı (`REPRODUCE.md`
güncellendi, bkz. "Reprodüksiyon açığı — KAPATILDI" notu).

### Adım 2 — Temiz Ortam Doğrulaması — ⚠️ GERÇEK BİR BOŞLUK BULUNDU VE DÜZELTİLDİ

Sıfır bir `venv` (Python 3.8.10) oluşturuldu, `pyproject.toml`'un
pinleriyle kuruldu. **İlk sonuç: 20 test dosyası `ModuleNotFoundError:
No module named 'yaml'` ile toplama (collection) hatası verdi** —
`dataset_versions.py`'nin `import yaml`'ı `pyproject.toml`'da hiç
listelenmemiş (mevcut geliştirme ortamında zaten kurulu olduğu için bu
şimdiye kadar hiç fark edilmemişti — tam da bu görevin var olma amacı).

**Düzeltme:** `pyproject.toml`'a `pyyaml==6.0.2` eklendi (mevcut ortamın
kendi sürümüyle aynı). Sonrasında:

| Kontrol | Sonuç |
|---|---|
| Temiz ortam `pytest tests/ -q` | **182 passed** — mevcut ortamla birebir aynı |
| `predict.py` (20 satırlık örnek, iki ortamda) | **0 etiket uyuşmazlığı, maksimum olasılık farkı = 0,0** (bit-bit aynı) |

**Kayıtlı "0,023 fold-başına fark" riski artık pinlerle (+ pyyaml
düzeltmesiyle) kapandı** — bu ölçümde gözlenen fark tam olarak sıfır,
yaklaşık değil.

**İkincil, bloklamayan bir boşluk daha not edildi:** `shap`/`matplotlib`
(yalnızca `f4_shap_explainability.py` için, `pytest`/`predict.py`
yolunda hiç kullanılmıyor) da `pyproject.toml`'da yok. SHAP
açıklanabilirlik scriptini yeniden çalıştırmak isteyen biri bunları ayrıca
kurmalı — **kritik reprodüksiyon yoluna (test+tahmin) dahil değil**, bu
turda düzeltilmedi (kapsam dışı, düşük öncelik).

### Adım 3 — `requirements.lock`

Temiz ortamın `pip freeze` çıktısı (35 paket, transitive dahil)
`final_freeze/requirements.lock`'a yazıldı — `pyproject.toml`'un 10
direkt pinine ek olarak matplotlib/pillow/pyparsing gibi transitive
bağımlılıkların TAM sürümlerini de sabitliyor.

### Adım 4 — `final_freeze/` Klasörü

```
final_freeze/
├── PAH_FINAL_20260902_nogit.pkl   (76.684 bayt)
├── predict.py                      (5.180 bayt)
├── manifest.json                   (~2.3 KB)
├── requirements.lock               (35 satır)
├── README_FINAL.md                 (~3.3 KB)
└── SHA256SUMS                      (5 satır)
```

Repo bir git deposu **değil** (`git rev-parse` başarısız) — dosya adında
`gitsha` yerine **`nogit`** kullanıldı, bu şeffaf biçimde hem dosya
adında hem `manifest.json`'da not edildi.

**Önemli sınırlama, `README_FINAL.md`'de açıkça belirtildi:** bu klasör
**tek başına çalıştırılabilir bir paket değil** — `predict.py`'nin kendi
`sys.path.insert(...parent/"src")` mekanizması, deponun kökündeki `src/`
klasörünün varlığını varsayıyor. Klasör bir **doğrulama enstantanesi**
(hash + repro kanıtı), deploy edilebilir bağımsız bir paket değil — bu
ayrım gizlenmeden README'de en üstte vurgulanıyor.

### Adım 5 — `manifest.json` (özet)

- Model/`predict.py` SHA-256'ları, split bankası manifest referansı
  (model ondan **bağımsız** eğitildi, yalnızca izlenebilirlik amaçlı).
- `pytest` sonucu (182 passed, hem mevcut hem temiz ortamda).
- Temiz ortam doğrulamasının tam kaydı (bulunan/düzeltilen `pyyaml`
  boşluğu, `predict.py` çapraz-ortam karşılaştırması, `shap` notu).
- Python sürümü (3.8.10), OS (`Windows-10-10.0.26200-SP0`).

### Adım 6 — `README_FINAL.md`

1 sayfalık jüri-özeti: model kartı (26 özellik, hepsi `AL_`/`EK_`
— `CAT_`/`AA_` final havuzda yok, önceki denetim turunun bulgusuyla
tutarlı), nasıl çalıştırılır, girdi şeması, bilinen sınırlama (tek
cümle + rapor referansı), `pytest`/`SHA256SUMS` ile doğrulama.

### Adım 7 — `SHA256SUMS`

```
adfd1b693df84d67bd46c2a404b776a006da93548cc4dcf17c9b5b57061f27ab *PAH_FINAL_20260902_nogit.pkl
2a54cde1352063324d28b5beb5638f4ab9ce318ca1f2d113ac5cd3cc326cf65e *predict.py
90f252644070b8a243579dbbf099a15f11f875147d05546c2bb2adc60e2eb666 *manifest.json
37183eb42617f6d0a92001bbb7d030118b4c1c585e496a69fb2c61c5c719d1e8 *requirements.lock
4d41a8ac6453637957b5a9a3becfd945c81234ebb703655d116db110e62dbc58 *README_FINAL.md
```

`sha256sum -c SHA256SUMS` ile kendi kendini doğruladı — 5/5 `OK`.

### Kod / dosyalar

- `src/genova/pah/f0_madde11_pool_reduction.py` (yeni).
- `pyproject.toml` — **`pyyaml==6.0.2` eklendi** (tek satırlık,
  gerçek/doğrulanmış bir boşluğun düzeltmesi — model/kod mantığı
  değişmedi, yalnızca bağımlılık bildirimi tamamlandı).
- `REPRODUCE.md` — "Reprodüksiyon açığı" bölümü "KAPATILDI" olarak
  güncellendi, test sayısı (158→182) düzeltildi.
- `final_freeze/` (yeni klasör, 6 dosya, yukarıda listelendi).

`final_model_bundle_v2.pkl`, `predict.py`, split bankası **orijinalleri**
değiştirilmedi — yalnızca `final_freeze/`'e kopyalandı. Tek kod
değişikliği `pyproject.toml`'daki eksik bağımlılık eklemesi (madde
kısıtlarıyla tutarlı: "model/eşik/hiperparametre değişikliği yok").

**Final Freeze Paketi TAMAMLANDI — gerçek bir reprodüksiyon boşluğu
(`pyyaml`) bulundu ve düzeltildi, çapraz-ortam tutarlılığı artık
kanıtlanmış durumda (bit-bit aynı sonuç).**
