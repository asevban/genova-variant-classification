# PAH Paneli — Preprocessing Merdiveni (Tamamlayıcı Deney Turu, Görev 5+6+9)

> CFTR panelinden öğrenilen, PAH'ta eksik olan yöntem: v1→v2→v3→v4 zincirini **tek değişkenli, sabit-diagnostic-modelli** basamaklar olarak izole edip her bileşenin gerçek katkısını ölçmek. Split bankası (`data/splits/pah/`, 10 tekrar × 5 dış fold = 50 dış fold) ve 3 sabit model (`LogisticRegression(class_weight="balanced", max_iter=1000)`, `ExtraTreesClassifier(n_estimators=200, random_state=42)`, `DummyClassifier(strategy="most_frequent")`) hiç değiştirilmedi — değişen yalnızca ön işleme. **Hiçbir sayı final model performansı değildir**, yalnızca ön işleme kararlarının diagnostic karşılaştırmasıdır. Bu rapor mevcut v1-v4 veri setlerini/pipeline'ları değiştirmez, yalnızca ek kanıt sağlar.
>
> Kod: `src/genova/pah/preprocessing_ladder.py` (fold-güvenli basamak inşası), `src/genova/pah/preprocessing_experiments.py` (koşum + tablo üretimi). Ham çıktılar: `reports/tables/preprocessing_ladder.csv`, `reports/tables/missingness_filter_experiment.csv`.

## Görev 5 — Merdiven Sonuçları

Basamaklar, v1'den başlayıp her seferinde **tek bir bileşen** ekler. Basamak 4/5, Görev 4'te split bankasında ölçülen sonuca göre otomatik seçilen ölçekleyiciyi (`RobustScaler` — **bkz. aşağıdaki uyarı**) kullanır.

| Deney ID | Bileşen (bir önceki basamağa eklenen) | Model | F1 (patojenik) | MCC | Seçim |
|---|---|---|---|---|---|
| M-0 | *(baseline)* v1, ham NaN | ExtraTrees | 0.9097 | 0.1630 | referans |
| M-0 | *(baseline)* v1, ham NaN | LogReg | 0.8129 | 0.1850 | referans |
| M-1 | + blok eksiklik göstergeleri (`al_all_missing`, `EK_3_missing`) — doldurma yok | ExtraTrees | 0.9117 (+0.0020) | 0.1914 | **nötr — küçük pozitif, gürültü sınırında** |
| M-1 | + blok eksiklik göstergeleri | LogReg | 0.8136 (+0.0007) | 0.1879 | **nötr — ölçülebilir katkı yok** |
| M-2 | + `AL_` sıfır-doldurma + `EK_3` medyan-doldurma (≈v2) | ExtraTrees | 0.9127 (+0.0010) | 0.2707 (+0.079) | **KABUL — MCC'de belirgin katkı** |
| M-2 | + `AL_` sıfır-doldurma + `EK_3` medyan-doldurma | LogReg | **0.8707 (+0.0571)** | 0.2206 | **KABUL — merdivenin en büyük tek-adım katkısı** |
| M-3 | + log1p/logit + `EK_` harmonizasyonu, ölçeklemesiz (≈v3 çıplak) | ExtraTrees | 0.9115 (−0.0012) | 0.2539 (−0.017) | **nötr — ağaç modeli için ölçülebilir zarar yok, dağılım düzeltmesi kendi başına katkı sağlamıyor** |
| M-3 | + log1p/logit + `EK_` harmonizasyonu | LogReg | 0.8421 (−0.0286) | 0.1576 | **RİSK — dönüşüm ölçeklemesiz uygulanınca LogReg'i kötüleştiriyor** |
| M-4 | + ölçekleyici (bu koşumda `RobustScaler` — Görev 4'ün otomatik seçimi) (≈v3 tam) | ExtraTrees | 0.9133 (+0.0018) | 0.2716 | **nötr-pozitif** |
| M-4 | + `RobustScaler` | LogReg | **0.6898 (−0.1523)** | 0.0163 | **RED — merdivenin en büyük tek-adım zararı, bkz. uyarı** |
| M-5 | + Aşama D'nin 24-özellik havuzu (≈v4) | ExtraTrees | **0.9153 (+0.0020)** | **0.2825** | **KABUL — merdivenin en yüksek ExtraTrees sonucu** |
| M-5 | + 24-özellik havuzu | LogReg | 0.7058 (+0.0160) | 0.2258 | **kısmi toparlanma, M-2/M-3 seviyesine dönmüyor** |
| — | *(referans)* | Dummy | 0.9093 | 0.0 | değişmez taban çizgisi |

**⚠️ Kritik uyarı — M-4/M-5'in LogReg sonuçları `RobustScaler`'a özgü, "ölçekleme kötü" diye okunmamalı:** `02_ON_ISLEME_KARARLARI_PAH.md` §11'de (Görev 4) ayrı ayrı ölçüldüğü gibi, `StandardScaler` aynı basamakta LogReg için F1=0.8635 verir (yani M-3'ün 0.8421'inden **iyileşme**, M-4'teki −0.1523'lük çöküş yalnızca `RobustScaler`'a özgü). Bu merdiven koşumu, Görev 4'ün otomatik-seçim mantığının (`ExtraTrees`'in marjinal üstünlüğüne göre seçim yapan, model-ailesi ayrımı yapmayan kural) `RobustScaler`'ı seçmesi nedeniyle bu çöküşü doğrudan yansıtıyor — bu, otomatik seçim kuralının kendisinin bir zafiyeti olarak Görev 4'te ayrıca belgelendi.

### En büyük katkı sağlayan basamak: **M-2 (`AL_` sıfır-doldurma + `EK_3` medyan-doldurma)**

LogReg'de +5.71 puan F1, +3.6 puan MCC — merdivenin tek başına en büyük, en tutarlı (her iki model ailesinde de pozitif yönlü) katkısı. Ham NaN'ın model girdisine dönüştürülmesi, dönüşüm/ölçeklemeden çok daha belirleyici.

### En az katkı sağlayan / zararlı basamaklar

- **M-1 (yalnız gösterge, doldurma yok):** ölçülebilir katkı yok (her iki modelde de <0.3 puan) — göstergelerin kendisi değil, göstergenin doldurmayla **birlikte** gelmesi (M-2) değer katıyor.
- **M-4 (`RobustScaler`, bu koşumda):** LogReg için merdivenin en büyük tek-adım zararı (−15.23 puan) — ama bu bir "ölçekleme genel olarak kötü" bulgusu değil, spesifik olarak "`RobustScaler` bu veri yapısında `LogReg` için kötü seçim" bulgusu (bkz. yukarıdaki uyarı ve §11).

## Görev 6 — Yüksek-Eksiklik Filtreleme Denemesi (Ampirik Red)

Basamak 2 üzerine, **yalnızca eğitim fold'undaki** `AL_` eksiklik oranına göre (`train_fold_missing_rate`, sızıntısız) `%70/%80/%90/%95` eşiklerinde kolon filtrelemesi denendi.

| Deney ID | Eşik | Ort. düşen kolon (334 `AL_`'den) | Model | F1 (patojenik) | Seçim |
|---|---|---|---|---|---|
| F-baseline | filtresiz (basamak 2) | 0 | ExtraTrees | 0.9127 | referans |
| F-baseline | filtresiz | 0 | LogReg | 0.8707 | referans |
| F-70 | %70 | 22.06 (min 22, maks 25) | ExtraTrees | 0.9148 (+0.0021) | nötr — gürültü sınırında |
| F-70 | %70 | 22.06 | LogReg | 0.8689 (−0.0018) | nötr — gürültü sınırında |
| F-80 | %80 | 18.00 (min=maks 18) | ExtraTrees | 0.9140 (+0.0013) | nötr |
| F-80 | %80 | 18.00 | LogReg | 0.8643 (−0.0064) | nötr |
| F-90 | %90 | 18.00 (F-80 ile aynı) | ExtraTrees | 0.9140 (+0.0013) | nötr — F-80 ile özdeş |
| F-90 | %90 | 18.00 | LogReg | 0.8643 (−0.0064) | nötr — F-80 ile özdeş |
| F-95 | %95 | 18.00 (F-80/F-90 ile aynı) | ExtraTrees | 0.9140 (+0.0013) | nötr — F-80 ile özdeş |
| F-95 | %95 | 18.00 | LogReg | 0.8643 (−0.0064) | nötr — F-80 ile özdeş |

**Bulgu:** `%80/%90/%95` eşikleri **tamamen özdeş sonuç veriyor** (düşen kolon sayısı fold'lar arasında hep tam 18) — bu, eğitim fold'larındaki hiçbir `AL_` kolonunun eksiklik oranının %80-%95 bandına düşmediğini gösteriyor (kolonlar ya ≤%80 ya da >%95 eksik, arada neredeyse hiç kolon yok). Etkinin kendisi **küçük ve model-ailesine göre yön değiştiriyor**: `ExtraTrees` için tutarlı ama küçük bir iyileşme (+0.13 ila +0.21 puan, std ~0.03 içinde — gürültü seviyesinde), `LogReg` için tutarlı ama küçük bir kötüleşme (−0.18 ila −0.64 puan, std ~0.06-0.07 içinde — yine gürültü seviyesinde). Hiçbir eşikte net, iki model ailesinde de tutarlı bir kazanç yok.

**Karar: Yüksek-eksiklik filtreleme denendi (4 eşik × 2 model × 50 dış fold), ölçülen etki gürültü seviyesinde ve model-ailesine göre yön değiştiriyor (ExtraTrees'te marjinal +, LogReg'te marjinal −) — bu nedenle final pipeline'a alınmadı.** Bu artık varsayım değil, ölçülmüş bir ret: "eksiklik biyolojik sinyal taşıyabilir" gerekçesiyle daha önce hiç denenmeden reddedilen karar, şimdi CFTR'nin ilkesiyle deneyip-reddet olarak kanıtlandı.

---

## Neden Alternatifi Seçmedim

**Neden yüksek-eksiklik filtrelemesi kullanılmadı?**
Görev 6'da %70/80/90/95 eşiklerinde ampirik olarak denendi (yukarıdaki tablo) — ölçülen etki gürültü seviyesinde ve model-ailesine göre yön değiştiriyor, tutarlı bir kazanç yok. Ayrıca A.6/A.9 (EDA)'da zaten belgelendiği gibi, 372 satırın 92'sinde (~%25) **tüm `AL_` bloğunun aynı anda eksik olması** potansiyel bir patojenite sinyali olarak değerlendiriliyor — kolon bazlı silme bu sinyali taşıyan `al_all_missing` göstergesiyle zaten ayrı şekilde yakalanıyor, ek filtreleme yalnızca bilgi kaybı riski taşıyordu.

**Neden agresif korelasyon-pruning yapılmadı?**
Aşama A.7 (`reports/01_EDA_RAPORU_PAH.md`), `AL_` kolonları arasında ciddi çoklu-doğrusallık bulmadı (yüksek korelasyonlu kümeler var ama küçük ve izole, VIF/kümeleme analizinde blokaj oluşturacak seviyede değil). Var olan 12 korelasyon kümesi zaten v3'te grup-özet istatistikleriyle (min/max/median/n_positive) temsil ediliyor (bkz. `02_ON_ISLEME_KARARLARI_PAH.md` §6) — kümelerin kendisini silmek yerine özetlemek, bilgi kaybı olmadan boyut indirgemesi sağlıyor; ayrıca Aşama D.1'in nested özellik seçimi (elastic-net + permütasyon + GBDT üçlü teyidi) zaten fazlalık taşıyan kolonları elemek için ayrı, daha ilkeli bir mekanizma sağlıyor.

**Neden `RobustScaler` sabit bırakıldı (bu turda pipeline'lardan çıkarılmadı)?**
Görev 4/5, `RobustScaler`'ın `LogReg` için `StandardScaler`'a göre belirgin şekilde kötü olduğunu gösterdi (bkz. §11, M-4). Ancak bu turun kuralı **modelleme yapmamak** ve **var olan v1-v4/pipeline'ları bozmamak** — bu yüzden `RobustScaler` kaldırılmadı, bunun yerine "Aşama E'nin nested iç döngüsünde ölçekleyici seçimi model-ailesine göre ayrı bir hiperparametre olarak ele alınmalı" notu bırakıldı. Kalıcı bir değişiklik, ancak Aşama E'nin kendi nested doğrulamasıyla kanıtlanarak yapılmalı — bu turun bulgusu bir öneri, final karar değil.

**Neden basamak 1 (yalnız gösterge, doldurmasız) ayrı bir üretim veri seti olarak korunmadı?**
Ölçülen katkısı her iki modelde de <0.3 puan (M-1 satırları) — pratik olarak basamak 0'dan ayırt edilemez. Yeni bir veri seti versiyonu üretmenin (bakım/dokümantasyon yükü) karşılığı yok; bulgusu yalnızca bu rapor tablosunda kayıt altına alınıyor.
