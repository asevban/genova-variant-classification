# PAH Paneli — Aşama F2: Bootstrap Güven Aralıkları (Resmi Kapanış)

> ⚠️ **DÜZELTME NOTU (2026-09-01, Revize turu — "Yanlış Bundle
> Referansının Düzeltilmesi" görevi):** Bu raporun ilk yazıldığı
> tarihten (Aşama F2'nin orijinal tamamlanması) beri,
> `src/genova/pah/f0_uncertainty_analysis.py` **yanlışlıkla
> `final_model_bundle.pkl`'i** (P1 madde 11 ÖNCESİ, 28-özellikli,
> provenance-riskli `al_all_missing`/`CAT_1` DAHİL, artık resmi
> OLMAYAN tarihsel bundle) **yüklüyordu — resmi `final_model_bundle_
> v2.pkl` (26 özellik) DEĞİL**. Bu hata "Jüri İçin Final Performance
> Card" görevinde bulundu, bu görevde (`f0_uncertainty_analysis.py`'nin
> `BUNDLE_OUT` referansı `BUNDLE_PATH`/`final_model_bundle_v2.pkl`
> olarak düzeltildi) kaynağından giderildi ve bu rapor **yeniden
> üretilen doğru sayılarla** güncellendi. Aşağıdaki tüm sayılar artık
> **doğru bundle'a (`final_model_bundle_v2.pkl`)** ait — `13_FINAL_
> PERFORMANCE_CARD_PAH.md` ile satır satır (nokta tahminleri birebir,
> CI sınırları farklı `n_boot`/seed nedeniyle küçük farklarla)
> doğrulandı. **Hata gizlenmiyor, şeffaflık için eski (yanlış) sayılar
> da aşağıda "Eski (YANLIŞ) Değerler" başlığı altında saklı tutuluyor.**
>
> Aşama F'nin kendi raporu — Aşama E'nin `06_MODEL_SECIM_RAPORU_PAH.md`'sine
> yazılmadı, E-F ayrımı net kalsın diye ayrı dosya. Dosya numarası notu:
> `GENOVA_PAH_ClaudeCode_Prompt_AsamaE-F_GUNCEL.md`'nin F2 talimatı bu
> dosyayı `08_BOOTSTRAP_GUVEN_ARALIKLARI_PAH.md` olarak adlandırmayı
> öneriyordu, ama `08` numarası zaten `08_ASAMA_E_SONRASI_DENETIM_
> PAH.md` tarafından kullanılıyor — çakışmayı önlemek için sıradaki boş
> numara (**10**) kullanıldı.

Bu görev **yeni bir hesaplama turu değil** — P1 madde 10'da zaten inşa
edilen `sample_level_bootstrap_ci` (`n_boot=2000`, F2'nin "1000+ tekrar"
şartını aşıyor) ve `f0_uncertainty_analysis.csv`'nin doğrulanması + resmi
F2 raporu + tek bir düzeltme (sensitivity eklenmesi).

---

## 1) Final Modelin Örnek-Düzeyi %95 Bootstrap Güven Aralığı

Final model: CatBoost, `v4_from_v2` (26 özellik), Beta kalibratör, kapalı-
form Bayes önsel düzeltmesi, eşik=0,35 (`models/pah/final_model_bundle_
v2.pkl`). Bundle'ın kendi 5-fold cross-fit OOF'u (369 satır, deterministik,
yeniden eğitim değil) üzerinde, `n_boot=2000` **satır-bazlı** (fold-bazlı
DEĞİL) yeniden örnekleme:

| Metrik | Nokta tahmini | %95 Bootstrap CI | Genişlik |
|---|---|---|---|
| F1 | 0,8177 | [0,7824 ; 0,8509] | 0,0685 |
| MCC | 0,3921 | [0,2898 ; 0,4902] | 0,2004 |
| Specificity | 0,7869 | [0,6765 ; 0,8909] | 0,2145 |
| **Sensitivity (recall)** | **0,7208** | **[0,6699 ; 0,7678]** | **0,0979** |
| AUROC | 0,8309 | [0,7701 ; 0,8889] | 0,1187 |

(Sensitivity F2'nin orijinal turunda eklendi — `src/genova/metrics.py::
sensitivity`, TP/(TP+FN), `specificity`'nin doğrudan karşılığı; 4 birim
testle doğrulandı. AUROC, "AUROC/Sensitivity Ekleme" Revize turunda
eklendi. Bu Düzeltme turunda `f0_uncertainty_analysis.py`'nin bundle
referansı düzeltilip yeniden çalıştırıldı — model yeniden eğitilmedi.)

### Eski (YANLIŞ) Değerler — Tarihsel Kayıt, ARTIK GEÇERSİZ

> Yanlış bundle'dan (`final_model_bundle.pkl`, 28 özellik) hesaplanmıştı,
> yalnızca şeffaflık için saklanıyor — **kullanılmamalı.**

| Metrik | Yanlış nokta tahmini | Yanlış %95 CI |
|---|---|---|
| F1 | 0,8345 | [0,8015 ; 0,8663] |
| MCC | 0,3885 | [0,2842 ; 0,4890] |
| Specificity | 0,7377 | [0,6170 ; 0,8508] |
| Sensitivity | 0,7532 | [0,7065 ; 0,8006] |
| AUROC | 0,8171 | [0,7533 ; 0,8806] |

## 2) 100/250 Monte Carlo Simülasyonu (Şartnamenin Gerçek Final Kompozisyonu)

Madde 10'da zaten üretildi (`reports/tables/f0_uncertainty_analysis.csv`),
burada yeniden okunuyor (bu Düzeltme turunda **doğru bundle'la** yeniden
üretildi):

| İstatistik | Değer |
|---|---|
| Ortalama F1 | 0,6395 |
| Medyan F1 | 0,6400 |
| %95 aralık | [0,5738 ; 0,7000] |

*(Eski/yanlış değer, tarihsel kayıt: ortalama F1=0,6253, medyan=0,6260,
aralık=[0,5645;0,6838] — kullanılmamalı.)*

2000 simülasyon, her birinde OOF'tan (yerine koyarak) 100 patojenik +
250 benign satır çekilip final eşik (**0,35** — doğru bundle'ın kendi
eşiği; eski/yanlış raporda "0,36 civarı" deniyordu, o da yanlış bundle'ın
farklı eşiğinden kaynaklanıyordu) uygulandı.

## 3) CI Neden Bu Kadar Geniş? (Doğru Açıklama)

**Önceki (E-F dokümanının orijinal) açıklaması yanlıştı.** "Final test
setinde yalnızca ~13-20 benign örnek beklendiği için geniş aralık
normaldir" ifadesi şartnameyle **çelişiyor** — şartname final PAH test
setinin **250 benign** (100 patojenik/250 benign) içerdiğini açıkça
belirtiyor. Final test seti küçük değil.

**Gerçek neden:** CI'nin genişliği, **bizim geliştirme/değerlendirme
setimizin küçüklüğünden** kaynaklanıyor — 369 satırlık `v1`'de yalnızca
**61 benzersiz benign örnek** var, ve yukarıdaki tüm CI/Monte Carlo
tahminleri bu 61 örneğe (ve 308 patojenik örneğe, ama benign azınlık
sınıf olduğu için asıl darboğaz o) dayanıyor. Final test setinin kendisi
250 benign içerecek — bizim **ölçme imkanımız**, final setin kendisi
değil, sınırlı. Bu, gizlenmeden açıkça yazılıyor.

## 4) Fold-Düzeyi (Eski, Dar) vs Örnek-Düzeyi (Doğru, Geniş) CI

Madde 10'un bulgusu, buraya taşındı:

| | Genişlik (F1) |
|---|---|
| Fold-düzeyi (50 dış-fold'u bağımsızmış gibi sayan eski yöntem) | 0,0402 |
| **Örnek-düzeyi (369 satırı doğru birim sayan yöntem)** | **0,0685** |

**~1,70× daha geniş.** Fold-düzeyi yöntem, aynı 369 satırın 50 farklı
bölünmesini bağımsız gözlemler gibi ele alarak belirsizliği olduğundan
dar gösteriyordu — bu artık projenin resmi CI yöntemi değil, yalnızca
tarihsel karşılaştırma için burada tutuluyor.

*(Şeffaflık notu: fold-düzeyi ve örnek-düzeyi sayının nokta tahminleri
[0,7988 vs 0,8177] birebir aynı F1'in iki CI'si değil — fold-düzeyi sayı
E5'in eski/leaky-havuz nested değerlendirmesinden, örnek-düzeyi sayı
güncel final bundle'ın kendi OOF'undan geliyor. Karşılaştırma "CI GENİŞLİĞİ
yöntemi" hakkında, iki ayrı F1 sayısının birebir eşleştirilmesi hakkında
değil — bkz. `06_MODEL_SECIM_RAPORU_PAH.md` P1 madde 10 bölümündeki aynı
uyarı.)*

## Kod / dosyalar

- `src/genova/metrics.py::sensitivity` (yeni) — `tests/test_metrics.py`'ye
  4 yeni test.
- `src/genova/pah/f0_uncertainty_analysis.py` — sensitivity eklenerek
  yeniden çalıştırıldı (deterministik, model yeniden eğitilmedi); Düzeltme
  turunda **bundle referansı düzeltildi** (`BUNDLE_OUT` → `BUNDLE_PATH`/
  `final_model_bundle_v2.pkl`).
- `reports/tables/f0_uncertainty_analysis.csv` — sensitivity satırı
  eklendi; Düzeltme turunda doğru bundle'la yeniden üretildi.
- `GENOVA_PAH_ClaudeCode_Prompt_AsamaE-F_GUNCEL.md` — F2 bölümündeki
  yanlış "~13-20 benign" ifadesi düzeltildi (satır 227).
- `reports/13_FINAL_PERFORMANCE_CARD_PAH.md` — bu düzeltmenin kanıtı/
  çapraz-doğrulaması, aynı sayıları bağımsız bir yöntemle (Monte Carlo
  tam-metrik genişletmesi) üretti.

Split bankasına, `final_model_bundle_v2.pkl`'e, `predict.py`'ye
dokunulmadı. Hiçbir model yeniden eğitilmedi (ne orijinal turda ne bu
düzeltmede).

**Aşama F2 TAMAMLANDI (düzeltilmiş sayılarla).**
