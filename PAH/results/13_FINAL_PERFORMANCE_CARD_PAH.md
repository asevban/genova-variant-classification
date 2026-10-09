# PAH Paneli — Final Performance Card

> **Dosya numarası notu:** Görevin önerdiği `07_...` adı, mevcut
> `07_LITERATUR_TARAMASI_ASAMA_E_ONCESI_PAH.md` ile çakıştığı için
> (`08→10` çakışmasında uygulanan aynı disiplinle) sıradaki boş numara
> (**13**) kullanıldı.

Bu kart, final CatBoost modelinin (`models/pah/final_model_bundle_v2.pkl`
— 26 özellik, `depth=5,lr=0,05`, Beta kalibratör, kapalı-form Bayes önsel
düzeltmesi, eşik=0,35) performansını **tek, kendi başına yeterli bir
kaynakta** toplar. Performans artırmaz, model/eşik/dosyaya dokunmaz.

---

## ⚠️ Bulunan ve Düzeltilen Bir Tutarsızlık (önemli, şeffaf not)

Bu kart hazırlanırken, `reports/tables/f0_uncertainty_analysis.csv`
(ve ondan üretilen `10_BOOTSTRAP_GUVEN_ARALIKLARI_PAH.md`) ile bu
kartın sayıları arasında gerçek bir fark bulundu (örn. F1: 0,8345
[eski] vs 0,8177 [bu kart]). Kök neden araştırıldı:

**`src/genova/pah/f0_uncertainty_analysis.py`, `BUNDLE_OUT`
(`final_model_bundle.pkl` — P1 madde 11 ÖNCESİ, 28 özellikli,
provenance-riskli `al_all_missing`/`CAT_1` DAHİL, artık resmi
OLMAYAN tarihsel bundle) yüklüyordu — resmi `final_model_bundle_v2.pkl`
(26 özellik) DEĞİL.** Bu, `f0_final_model.py`'den `BUNDLE_OUT`'un
(kendi orijinal kaydetme amaçlı sabiti) yanlışlıkla "resmi bundle"
yerine import edilmesinden kaynaklanan, madde 10'un ilk yazıldığı
tarihten beri var olan bir hata.

**Doğrulama:** Bu kartın sayıları (`final_model_bundle_v2.pkl`'den,
`cross_fit_oof` ile deterministik olarak yeniden üretildi) F3'ün
(`06_MODEL_SECIM_RAPORU_PAH.md`, Aşama F3, "normal CV" referansı:
F1=0,8177, MCC=0,3921, specificity=0,7869, sensitivity=0,7208,
raw_AUC=0,8309) sayılarıyla **4 ondalık basamağa kadar birebir aynı**
— bu kartın (ve F3'ün, ve bu oturumdaki tüm diğer deneylerin, hepsi
`final_model_bundle_v2.pkl`'i doğru şekilde kullanıyor) sayılarının
**doğru** olduğunu kanıtlıyor.

**Etki kapsamı (kontrol edildi):** Hata yalnızca
`f0_uncertainty_analysis.py`'ye özgü — `grep` ile doğrulandı, başka
hiçbir script `BUNDLE_OUT`'u yükleme amacıyla kullanmıyor. Bu oturumdaki
diğer tüm deneyler (seed-bagging, ağırlıklandırma, missingness-
augmentation, confound audit, `CAT_1`/`AL_` örtüşme teşhisi, triyaj
sistemi) hep `final_model_bundle_v2.pkl`'i doğru yüklüyordu — bu kart
BUNU teyit ediyor. Bu kart resmi olarak DOĞRU sayıları içeriyor;
**`10_BOOTSTRAP_GUVEN_ARALIKLARI_PAH.md`'nin ve `06_MODEL_SECIM_
RAPORU_PAH.md`'nin P1 madde 10 bölümünün ayrı bir turda düzeltilmesi
gerekiyor** (bu kartın kapsamı dışında — yalnızca bulundu, işaretlendi,
düzeltilmedi).

---

## Blok A — Standart CV (Eğitim Dağılımı)

**Kaynak:** Final bundle'ın kendi 5-fold `cross_fit_oof`'u
(`f0_final_model.py`, değişmedi, deterministik yeniden üretim — yeniden
eğitim değil), 369 satırın **doğal** kompozisyonunda (**%83,5 patojenik**
[308/369] — **not: görev metnindeki "~%71 patojenik" ifadesi
yanlış/muhtemelen bir yazım hatası, gerçek/doğrulanmış oran %83,5'tir**,
bkz. `CLAUDE.md`: "%83,3 patojenik eğitimde"). Kalibratör: Beta.
Önsel düzeltme: kapalı-form Bayes (SLD terminolojisiyle anılır, gerçek
SLD değildir — bkz. `06_MODEL_SECIM_RAPORU_PAH.md` P1 madde 13). Eşik:
**0,35**. Örnek-düzeyi bootstrap (n_boot=1000, 369 satırın satır-bazlı
yeniden örneklemesi).

| Metrik | Değer | %95 CI |
|---|---|---|
| F1 (resmi, pozitif sınıf=Patojenik) | 0,8177 | [0,7831 ; 0,8497] |
| Precision | 0,9447 | [0,9132 ; 0,9739] |
| Sensitivity (Recall) | 0,7208 | [0,6689 ; 0,7674] |
| Specificity | 0,7869 | [0,6735 ; 0,8947] |
| MCC | 0,3921 | [0,2884 ; 0,4881] |
| Balanced Accuracy | 0,7538 | [0,6939 ; 0,8125] |
| AUPRC | 0,9545 | [0,9292 ; 0,9760] |
| AUROC | 0,8309 | [0,7722 ; 0,8884] |
| Brier Score | 0,2664 | [0,2440 ; 0,2901] |
| LogLoss | 0,7545 | [0,6946 ; 0,8174] |

*(Brier/LogLoss, SLD-düzeltilmiş — yani pipeline'ın gerçekte ürettiği
— olasılık üzerinden hesaplandı; bu ölçek eğitim dağılımına değil final
hedef önseline göre kalibre olduğu için Brier/LogLoss'un mutlak değeri
yüksek görünebilir — bu, olasılıkların YANLIŞ olduğu anlamına gelmez,
yalnızca değerlendirilen popülasyonun (eğitim, %83,5 patojenik) SLD'nin
hedeflediği popülasyondan (%28,6 patojenik) farklı olmasından kaynaklanır.)*

## Blok B — Final-Projeksiyon (100 Patojenik / 250 Benign)

**Kaynak:** Aynı final bundle'ın SLD-düzeltilmiş olasılıkları, şartnamenin
gerçek final test kompozisyonunu (100 patojenik + 250 benign, toplam
N=350, **%28,6 varsayılan patojenik oranı**) taklit eden Monte Carlo
yeniden-örnekleme (n_simulations=2000, `genova.statistics.monte_carlo_
final_f1_simulation` ile **AYNI** resampling şeması — F1 için bit-bit
aynı sonucu verdiği doğrudan doğrulandı, bkz. aşağıdaki not). Eşik:
**0,35** (değiştirilmedi — final-projeksiyonda optimal olduğu, 0,20–0,85
aralığında tarama ile daha önce doğrulanmıştı, bkz. bu görevin kendi
Bağlam bölümü ve `06_MODEL_SECIM_RAPORU_PAH.md` Aşama E5).

| Metrik | Değer | %95 CI |
|---|---|---|
| F1 | 0,6395 | [0,5738 ; 0,7000] |
| Precision | 0,5753 | [0,5109 ; 0,6417] |
| Sensitivity (Recall) | 0,7212 | [0,6300 ; 0,8000] |
| Specificity | 0,7862 | [0,7320 ; 0,8360] |
| MCC | 0,4785 | [0,3784 ; 0,5688] |
| Balanced Accuracy | 0,7537 | [0,7020 ; 0,8020] |
| AUPRC | 0,6689 | [0,5788 ; 0,7574] |
| AUROC | 0,8305 | [0,7829 ; 0,8728] |
| Brier Score | 0,1459 | [0,1304 ; 0,1633] |
| LogLoss | 0,4398 | [0,3989 ; 0,4858] |

**Çapraz-doğrulama notu:** Bu tablonun F1 satırı, `genova.statistics.
monte_carlo_final_f1_simulation`'ın (değişmedi, mevcut fonksiyon)
**AYNI seed (42) ve AYNI resampling sırasıyla** ürettiği F1 ortalamasıyla
**bit-bit aynı** çıktı (fark=0,00e+00) — bu kartın Monte Carlo
mekanizmasının, projenin zaten kullandığı resmi araçla tutarlı
olduğunun doğrudan kanıtı.

**AUPRC neden Blok B'de Blok A'dan çok daha düşük?** Bu **beklenen ve
doğru** bir davranış, hata değil — AUPRC, pozitif sınıfın önsel oranına
DUYARLIDIR (AUROC'un aksine). Blok A'da patojenik %83,5 (kolay
ayırt edilebilir çoğunluk), Blok B'de %28,6 (azınlık) — aynı model,
aynı ayırt edicilik (AUROC ~0,83 ikisinde de neredeyse aynı), ama düşük
pozitif-oranlı popülasyonda AUPRC yapısal olarak düşer. Bu ikisini
karşılaştırmak "model kötüleşti" değil, "önsel oranı farklı" demektir.

---

## Bağlam Notları (Adım 4)

| Parametre | Değer | Referans |
|---|---|---|
| Karar eşiği | 0,35 (sabit, her iki blokta da aynı) | `06_MODEL_SECIM_RAPORU_PAH.md` Aşama E5 (nested eşik seçimi) + Revize turu "prior düzeltmesi sınır analizi" (SLD eşiği 0,35 ↔ Beta-uzayı eşiği ≈0,8702, tek bir monotonik dönüşüm) |
| Varsayılan final önsel (patojenik) | %28,6 (100/350) | `CLAUDE.md` — şartname |
| Kalibrasyon yöntemi | Beta (`betacal`) | `06_MODEL_SECIM_RAPORU_PAH.md` Aşama E3 — Isotonic'in n=369'da aşırı-uyum gösterdiği (çok az benzersiz kalibre değer) ampirik olarak doğrulandı, Beta/Platt'a karşı seçildi |
| Önsel düzeltme | Kapalı-form Bayes, w1=0,343337, w0=4,275449 | `06_MODEL_SECIM_RAPORU_PAH.md` Aşama E4 + P1 madde 13 (terminoloji: "SLD" adı tarihsel, gerçek yöntem kapalı-form Bayes) |
| Özellik havuzu | 26 özellik (25 `AL_`/`EK_`/`CAT_1` + `al_all_missing` HARİÇ — provenance-riskli 2 özellik P1 madde 11'de çıkarıldı) | `06_MODEL_SECIM_RAPORU_PAH.md` P1 madde 11 |
| **Bilinen sınırlama** | `CAT_1`-boş alt-kümesinin (132 satır) yapısal az-temsiliyeti, model transferinde ciddi kırılganlık yaratıyor (kritik yön raw AUC 0,831→0,573) — **iki bağımsız düzeltme denemesi (domain-dengeli ağırlıklandırma, missingness-augmentation) ikisi de çözemedi**, veri-seviyesi bir kısıt olarak kabul edildi | `06_MODEL_SECIM_RAPORU_PAH.md` Aşama F3 + sonraki teşhis turları; final günü bu riski **otomatik olarak izleyen** (ama asla otomatik müdahale etmeyen) bir GREEN/YELLOW/RED triyaj sistemi kuruldu — bkz. `FINAL_GUN_TRIYAJ_REHBERI.md` |

---

## Kod / Dosyalar

- `src/genova/pah/f0_final_performance_card.py` (yeni) —
  `reports/tables/f0_final_performance_card.csv`.
- `f0_final_model.py::cross_fit_oof`, `genova.statistics.sample_level_
  bootstrap_ci`, `genova.statistics.monte_carlo_final_f1_simulation`
  DEĞİŞTİRİLMEDİ — yalnızca import edilip yeniden kullanıldı (Monte
  Carlo için AYNI resampling şemasını tekrar eden, TÜM metrikleri
  hesaplayan yeni bir sarmalayıcı fonksiyon yazıldı — küçük, bilinçli
  bir kod tekrarı, önceki tüm turlardaki aynı gerekçeyle).

`final_model_bundle_v2.pkl`, `predict.py`, split bankasına
dokunulmadı — hiçbir model yeniden eğitilmedi, yalnızca mevcut bundle'ın
deterministik OOF'u yeniden üretilip üzerinde istatistik hesaplandı.
