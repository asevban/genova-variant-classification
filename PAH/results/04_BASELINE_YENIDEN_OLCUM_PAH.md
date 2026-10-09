# PAH Paneli — Aşama E1: Baseline (PDR) Modelinin Yeni Split Bankasında Yeniden Ölçümü

> **Dosya adı uyarısı:** Bu rapor `reports/04_PREPROCESSING_MERDIVENI_PAH.md`
> ile **farklı** bir dosyadır (o rapor Aşama D'nin ölçekleyici/model-ailesi
> merdiveni deneyidir). İkisi de ayrı ayrı korunur.

## Amaç ve kapsam

Bu adım **yeni bir model eğitmek değil**, PDR'de raporlanan XGBoost(MCC-ayarlı)
modelini, PDR'de belgelenen **sabit hiperparametrelerle** yeniden kurup
`data/splits/pah/` sabit split bankasının **her dış fold'unda** (iç fold'da
hiçbir arama yapılmadan) skorlayarak, Aşama E2'den itibaren her yeni model/
versiyon kombinasyonunun geçmesi gereken adil bir referans noktası kurmaktır.

Orijinal PDR model dosyası/ağırlıkları elde değil (repoda `models/` dizini bu
görev öncesinde yoktu). Bu yüzden burada üretilen model, PDR'de belgelenen
**hiperparametreler ve mimari** kullanılarak sıfırdan yeniden eğitilmiş bir
**"yeniden eğitilmiş tarihsel referans"**tır — "orijinal PDR modeli" değildir.

## Sabit yapılandırma (PDR'den, bu adımda değiştirilmedi)

| Bileşen | Değer |
|---|---|
| Model ailesi | XGBoost (`XGBClassifier`) |
| `n_estimators` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0,05 |
| `min_child_weight` | 5 |
| Azınlık sınıf ağırlığı | ×2,0 |
| Karar eşiği | 0,359 |
| `random_state` | 42 (determinizm için, PDR'de belirtilmemişti) |

Azınlık sınıf ağırlığı, XGBoost'un `scale_pos_weight` parametresiyle **değil**,
her dış-fold'un eğitim satırlarına doğrudan uygulanan `sample_weight` ile
uygulandı: fold'daki azınlık sınıfın (50/50 fold'da da tutarlı biçimde
Label=0/benign çıktı — bkz. aşağıdaki doğrulama) her satırına 2,0, çoğunluk
sınıfına (Label=1/patojenik) 1,0 ağırlık verildi. `scale_pos_weight`
kullanılmadı çünkü o parametre **pozitif** (Label=1) sınıfı ölçeklendirir;
burada azınlık sınıf pozitif değil negatif (benign) olduğu için anlamı ters
düşerdi — "azınlık sınıf ağırlığı×2,0" ifadesinin literal karşılığı
`sample_weight` ile elde edildi.

Hiçbir hiperparametre araması yapılmadı; bu adım saf bir ölçüm turudur.

## Veri versiyonu seçimi: v1 (gerekçeli)

Görev talimatı gereği hem `v1` hem `v2` denendi (aşağıdaki tablo), her ikisi
de `data/splits/pah/`'ın 50 dış fold'unda (10 tekrar × 5 fold), fold-içi
yeniden fit edilerek — mevcut `src/genova/pah/missingness.py` sınıfları
(`BlockMissingIndicator`, `MedianImputerWithIndicator`, `ConstantFillImputer`)
yalnızca train fold'unda `fit()` edilip hem train hem test'e `transform()`
uygulanarak. Kategorik kolonların (`CAT_1/2/3/4/5`, `AA_1/2`) kategori
sözlüğü de yalnızca train fold'undan çıkarıldı; test fold'unda görülmeyen bir
kategori NaN'a eşlendi (XGBoost'un native eksik-değer yoluna düşer) — böylece
kategori sözlüğü de bir sızıntı kaynağı olmadı.

- **v1**: ham `NaN` (`AL_`/`EK_`), native kategorik (`CAT_`/`AA_`), eksiklik
  göstergesi yok.
- **v2**: v1 + `al_all_missing` + `ek_cat_block_missing` blok göstergeleri +
  `EK_3` medyan-doldurma+gösterge + `AL_` sıfır-doldurma (diğer `EK_`
  kolonları ham NaN kalır) — `dataset_versions.py::build_v2` ile aynı
  davranış, yalnızca fold-içi yeniden fit edilerek.

| Versiyon | F1 ort±std | F1 min–maks | MCC ort±std | Specificity ort±std |
|---|---|---|---|---|
| **v1** | **0,9199 ± 0,0242** | 0,8661 – 0,9635 | **0,3668 ± 0,1305** | **0,2608 ± 0,1057** |
| v2 | 0,9164 ± 0,0256 | 0,8480 – 0,9558 | 0,3332 ± 0,1366 | 0,2388 ± 0,1002 |

**Karar: `v1` seçildi.** Gerekçe iki katmanlı:

1. **Ampirik:** `v1`, üç metriğin (F1, MCC, specificity) üçünde de `v2`'den
   marjinal ama tutarlı biçimde daha iyi; iki versiyon arasındaki fark F1'de
   yalnızca 0,0035 (bootstrap gerektirmeyecek kadar küçük, pratikte eşdeğer),
   ama tutarlı yönde.
2. **Tarihsel uygunluk (asıl gerekçe):** `al_all_missing`/`ek_cat_block_missing`
   blok göstergeleri ve `EK_3` medyan-doldurma politikası, bu depodaki Aşama D
   Düzeltme Turu'nun (bkz. `CLAUDE.md`, `reports/00_DUZELTME_OZETI_PAH.md`)
   ürünüdür — PDR zamanında (Aşama A-D'nin bu ayrıntılı eksiklik-mühendisliği
   yapılmadan önce) mevcut değildi. PDR'nin XGBoost modelinin, ham veriye en
   yakın, minimum eksiklik-mühendisliği içeren `v1` spesifikasyonuna (native
   NaN + native kategorik, XGBoost'un kendi eksik-değer yol ayrımına
   bırakılmış) daha yakın olduğu değerlendirildi. `v2`'nin ek göstergeleri,
   PDR'nin görmediği bir bilgi avantajı sağlayarak "adil" bir tarihsel
   yeniden kuruluma zarar verirdi.

Sonuç: **frozen baseline `v1` reçetesiyle kaydedildi** (`models/pah/baseline_frozen.pkl`
içinde her iki versiyonun tam sonuçları da saklandı, ileride karşılaştırma
için).

## Dış-CV sonuçları (50 fold: 10 tekrar × 5 dış fold, `v1`)

| Metrik | Ortalama ± std | Min | Maks |
|---|---|---|---|
| F1 (resmi, pozitif sınıf) | **0,9199 ± 0,0242** | 0,8661 | 0,9635 |
| MCC (ikincil) | 0,3668 ± 0,1305 | −0,0669 | 0,6087 |
| Specificity (benign üzerinde TNR) | 0,2608 ± 0,1057 | 0,0000 | 0,5000 |

Her dış fold'da fold boyutu ~295 train / ~74 test satırı (369 satırlık
dedup-sonrası veri setinin 5-fold bölünmesiyle tutarlı); minority-class
tespiti 50 fold'un **tamamında** Label=0 (benign) çıktı — beklenen %83,3
patojenik/%16,7 benign eğitim dağılımıyla tutarlı, sample-weight mantığının
her fold'da doğru sınıfı hedeflediği doğrulandı.

## PDR'nin iç-test F1'i (0,9323) ile karşılaştırma

Dış-CV F1 ortalaması (0,9199), PDR'nin raporladığı iç-test F1'inden (0,9323)
yalnızca **~0,012 puan düşük** — ilk bakışta beklenenden (raporun öngördüğü
"muhtemelen belirgin daha düşük") daha küçük bir fark. Bunun nedeni, F1'in
yalnızca pozitif sınıf (patojenik) üzerinden hesaplanması ve eğitim
dağılımının (%83,3 patojenik) hem PDR'nin iç test setinde hem de bu dış-CV
fold'larında aynı kalması — yani F1, çoğunluk sınıfın kolay ayrılabilirliğinden
büyük ölçüde besleniyor, protokol farkı (nested dış-CV vs. tek iç-test) F1'i
çok fazla değiştirmiyor.

**Asıl fark specificity'de saklı ve çok daha çarpıcı:** dış-CV specificity
ortalaması yalnızca **0,2608** (bazı fold'larda 0,00 — yani o fold'daki
benign örneklerin **hiçbiri** doğru sınıflandırılamadı). Bu, modelin benign
sınıfı ayırt etme gücünün F1'in iyi görünmesinin arkasında gizlendiğini
gösteriyor: PDR'nin patojenik-ağırlıklı iç test setinde (yalnızca ~%16,7
benign) specificity düşük olsa bile toplam F1 (yalnızca pozitif sınıfa
bakan bir metrik) buna kör kalıyor. Final yarışma test setinde beklenen
dağılım tam tersine dönüyor (%28,6 patojenik / %71,4 benign) — yani benign
sınıfı bu kadar zayıf ayırt eden bir modelin **final F1'i, eğitim/PDR
ortamındakinden çok daha kötü çıkma riski taşıyor**, çünkü yanlış
sınıflandırılan benign'ler (FP) test setinde çok daha büyük bir payı
oluşturacak ve pozitif-sınıf F1'i FP üzerinden doğrudan cezalandıracak.

Bu, beklenen ve istenen bir sonuçtur — "kötü" değil, **E4 (final önsel
düzeltmesi) ve E5'in (eşik seçimi) neden bu projede kritik olduğunun somut
kanıtı**. Sabit eşik 0,359 ve sabit hiperparametreler, PDR'nin iç test
koşullarına (yüksek patojenik oranı) göre kalibre edilmiş görünüyor; final
dağılımına göre yeniden kalibre edilmeden bırakılırsa spesifisite açığı
doğrudan final F1'e yansıyacak. Bu sayı (F1=0,9199 ± 0,0242, specificity=0,2608
± 0,1057), Aşama E2-E6'daki her yeni model/versiyon/kalibrasyon/eşik
kombinasyonunun **geçmesi gereken referans** olarak donduruldu.

## Kaydedilen dosya

`models/pah/baseline_frozen.pkl` — pickle edilmiş sözlük:
- `model`: tam 369 satırda (`v1` reçetesiyle) fit edilmiş `XGBClassifier`.
- `data_version`: `"v1"`.
- `feature_columns`, `categorical_columns`, `threshold` (0,359),
  `hyperparameters`, `minority_class`, `minority_weight`.
- `outer_cv_results_v1` / `outer_cv_results_v2`: 50 satırlık tam dış-CV
  sonuç tabloları (her ikisi de, karşılaştırma izlenebilir kalsın diye).
- `summary_v1` / `summary_v2`: yukarıdaki özet istatistikler.
- `label`: `"yeniden egitilmis tarihsel referans (orijinal PDR model dosyasi degil)"`.

## Kod

`src/genova/pah/e1_baseline.py` — **yeni dosya**, mevcut hiçbir
`src/genova/pah/*.py` dosyası değiştirilmedi. `split_bank`/`dataset_versions`/
`missingness` modüllerindeki mevcut fit/transform sınıfları olduğu gibi,
yalnızca fold-içi çağrılarak yeniden kullanıldı.

---

## Ek — Ağırlıklandırma Stratejisi Karşılaştırması (A/B/C)

### Amaç ve kapsam

Üç bağımsız kaynak (Berra'nın dört-panel raporu, `modelleme_literatür_1.md`,
`PAH_Literatur_ve_Modelleme_Asamasi.md`) tutarlı biçimde "no-weighting
zorunlu bir baseline'dır" ve XGBoost için veri-güdümlü `scale_pos_weight`
formülünün denenmesi gerektiğini söylüyor. Bu ek, E1'in **aynı** sabit-
hiperparametreli XGBoost/v1 modelini (yeniden aranmadı), **aynı** 50 dış
fold'da, **aynı** eşikte (0,359) — yalnızca örnek-ağırlıklandırma
stratejisini değiştirerek — üç varyantla yeniden ölçer:

- **A — Ağırlıksız:** `sample_weight` yok, tüm örnekler eşit.
- **B — Sabit azınlık×2,0 (referans):** yukarıdaki E1 sonucunun ta
  kendisi — burada **yeniden çalıştırılmadı**, `models/pah/baseline_
  frozen.pkl`'den okundu.
- **C — Veri-güdümlü `scale_pos_weight`:** her dış fold için `SPW =
  N_benign,train / N_patojenik,train`, yalnızca o fold'un train
  kısmından hesaplanır (`src/genova/pah/weighting.py::compute_data_
  driven_spw`).

**Tek değişken disiplini:** hiperparametre, eşik, veri versiyonu (`v1`)
hepsi E1'le birebir aynı tutuldu — yalnızca ağırlıklandırma değişti. Bu,
0,359 eşiğinin B'nin ağırlıklandırma şemasına göre (dolaylı olarak)
kalibre olmuş olabileceği, A/C için "en adil" eşik olmayabileceği
anlamına geliyor — E2'nin kendi nested eşik seçimi bu farkı ortadan
kaldıracak; burada amaç yalnızca ağırlıklandırmanın **izole** etkisini
görmek.

### Sonuçlar (50 dış fold, ortalama±std)

| Varyant | F1 | MCC | Specificity | Specificity=0 olan fold sayısı |
|---|---|---|---|---|
| A — Ağırlıksız | 0,9196±0,0247 | 0,3338±0,1208 | 0,1878±0,0771 | 2/50 |
| B — Sabit azınlık×2,0 (referans) | **0,9199±0,0242** | **0,3668±0,1305** | 0,2608±0,1057 | 1/50 |
| C — Veri-güdümlü SPW | 0,9064±0,0282 | 0,3229±0,1373 | **0,3257±0,1163** | **0/50** |

### B'ye karşı paired karşılaştırma (aynı 50 fold)

| Varyant | ΔF1 (ort.) | Kazanma oranı | ΔMCC (ort.) | Kazanma oranı | ΔSpecificity (ort.) | Kazanma oranı |
|---|---|---|---|---|---|---|
| A vs B | −0,0002 | %34 | −0,0331 | %26 | **−0,0730** | **%4** |
| C vs B | −0,0134 | %10 | −0,0439 | %34 | **+0,0649** | **%56** |

### Yorum

**A (ağırlıksız) açıkça en zayıf seçenek.** F1'de B'ye neredeyse eşit
görünse de (fark gürültü seviyesinde) bu yine F1'in yalnızca pozitif
sınıfa bakmasından kaynaklanan bir yanılsama — MCC'de (−0,0331, kazanma
oranı yalnızca %26) ve özellikle **specificity'de** (−0,0730, B **50
fold'un 48'inde** A'yı geçiyor — kazanma oranı yalnızca %4) A belirgin
kötü. A ayrıca en çok specificity=0 fold'una sahip (2/50) — hiçbir
ağırlıklandırma yapmamak, zaten çoğunluk-sınıf lehine yapısal olarak
yanlı olan bir problemde beklenen sonucu veriyor: model benign'i
neredeyse hiç öğrenmiyor. Bu sabit-eşik (0,359) karşılaştırmasında A
daha kötü çıktı; A'nın kendi optimal eşiğiyle henüz test edilmediği için
kesin değil — E2-EK'in RF/A-B-C sonuçları bu soruyu netleştirecek.

**C (veri-güdümlü SPW) net bir F1/specificity takası sunuyor.** F1'de
B'den tutarlı biçimde geride (−0,0134, kazanma oranı %10 — yani B, 50
fold'un 45'inde C'yi F1'de geçiyor) ve MCC'de de hafif geride (−0,0439,
%34 kazanma oranı). Ama **specificity'de tutarlı biçimde daha iyi**
(+0,0649, kazanma oranı %56) ve **hiçbir fold'da specificity=0'a
düşmüyor** (0/50, B'nin 1/50'sine ve A'nın 2/50'sine karşı) — yani en
**tutarlı/en az kırılgan** benign-tespiti sağlayan varyant C. Final test
setinin ağırlıklı benign olacağı göz önüne alındığında, bu tutarlılık
(en kötü-durum riskinin düşüklüğü) F1'deki ortalama kaybından daha değerli
olabilir — ama bu bir "C kazandı" kararı değil, yalnızca gözlem;
nihai karar E2'nin nested eşik/kalibrasyon prosedürüyle (tüm üç
ağırlıklandırmanın kendi optimal eşiğiyle karşılaştırılarak) verilmeli.

**Specificity üzerindeki net sıralama: C > B > A.** Üçü de aynı hiper-
parametre/eşikle çalıştığı için bu sıralama, ağırlıklandırma şemasının
minority-class (benign) tespiti üzerindeki etkisinin literatürün öngörüsüyle
(daha güçlü/veri-güdümlü ağırlıklandırma → daha iyi azınlık tespiti)
tam örtüştüğünü gösteriyor.

### Yeniden kullanılabilir altyapı

`src/genova/pah/weighting.py` — üç stratejiyi de üreten ortak arayüz
(`WEIGHTING_STRATEGIES` sözlüğü: `A_no_weight`, `B_fixed_minority_2x`,
`C_data_driven_spw`, her biri `y_train -> sample_weight` imzasında,
herhangi bir sklearn-uyumlu `.fit(X, y, sample_weight=...)` çağrısına
doğrudan verilebilir). E2, model portföyünü genişletirken bu üç
stratejiyi CatBoost/LightGBM/Elastic-Net'e de uygulayabilir — **hangi
stratejinin hangi model ailesinde işe yaradığına dair karar bu görevin
kapsamında değil, E2'nin kendi işi.**

### Kod / dosyalar

- `src/genova/pah/weighting.py` — üç ağırlıklandırma stratejisi (yeni).
- `src/genova/pah/e1_weighting_comparison.py` — A/C'yi ölçen, B'yi
  `baseline_frozen.pkl`'den okuyan karşılaştırma çatısı (yeni).
- `reports/tables/e1ek_weighting_comparison.csv` — 150 satır (3 varyant
  × 50 fold) ham sonuç.
- `tests/test_weighting_pah.py` — 7 yeni test, dahil: `compute_data_
  driven_spw`'nin yalnızca kendisine verilen `y_train`'den hesapladığını
  (val/test'e sızmadığını) doğrudan doğrulayan sızıntı testi.

`data/splits/pah/`, `data/processed/pah/`, `experiments/`, `v4_final_
feature_pool.json` içine hiçbir yazma yapılmadı; `models/pah/baseline_
frozen.pkl` değiştirilmedi/üzerine yazılmadı (yalnızca okundu). Hiçbir
hiperparametre araması yapılmadı — bu bir ağırlıklandırma ölçümüdür,
E2'nin model portföyü genişletmesi bu ekten sonra, ayrı bir onayla
başlayacak.
