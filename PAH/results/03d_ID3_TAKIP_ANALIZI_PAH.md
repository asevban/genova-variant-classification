# PAH Paneli — ID3 Bulgusu Takip Kontrolleri: Korelasyon + Etkileşim Yapısı

> **Ekleme türü: 03c'nin küçük, keşif amaçlı bir eki.** İki açık noktayı kapatır: (1) `AL_49`
> (yalnızca ID3'te stabil) `AL_300`'ün (4 mevcut yöntemde de stabil) bir temsilcisi mi, yoksa
> bağımsız bir sinyal mi; (2) ID3 ağaçlarının split yapısında marjinal yöntemlerin (MI, tek
> tek özellik önemleri) yakalayamadığı bir etkileşim/sıralama var mı.
>
> **Değişmeyenler:** `feature_selection.py`, `fold_features.py`, `id3_feature_selection.py`,
> split bankası, `v1`-`v4`, `configs/pah/`, `reports/03c_ID3_YONTEM_EKLEME_PAH.md` ve diğer
> mevcut `reports/` dosyaları — hepsi yalnızca import edilip okundu, hiçbiri değiştirilmedi.
> Kod: `notebooks/id3_followup_analysis.py` (yeni, tek seferlik keşif script'i — kalıcı
> pipeline'a gömülmedi). `id3_feature_selection.py`'den `_iter_folds` ve `CHOSEN_MAX_DEPTH`
> **değiştirilmeden import edildi** — aynı split bankası, aynı seed formülü, aynı aday özellik
> evreni kullanıldı.
>
> **Bu rapor bir karar belgesi değildir, yalnızca bulgu raporudur.**

---

## Adım 1 — `AL_49` ↔ `AL_300` Korelasyon ve Küme Üyeliği

**Küme üyeliği (`AL_correlation_clusters.csv`):** `AL_49` ve `AL_300`, **6 eşiğin (r09.9,
r0.95, r0.9, r0.8, r0.7, r0.5) hiçbirinde aynı kümede değil** — en gevşek eşik olan `r0.5`'te
bile ayrı kümeler (`AL_49`→küme 19, `AL_300`→küme 35). Bu, ikisi arasında en gevşek tanımla
bile (|r|>0.5) bir korelasyon-kümesi ilişkisi olmadığını gösteriyor.

**Doğrudan Spearman korelasyonu (`min_periods=30` konvansiyonu):**

| Kaynak | r | n (ortak dolu satır) |
|---|---|---|
| Mevcut `AL_spearman_corr.csv` (ham veri, 372 satır) | 0.0672 | — |
| Bu turda `v1.parquet` üzerinde yeniden hesaplandı (369 satır, dedup sonrası) | **0.0672** | 147 |

İki hesaplama **birebir aynı sonucu veriyor** (dedup edilen 3 satırın bu çiftin ortak-dolu
alt kümesini etkilemediği anlaşılıyor) — `reports/tables/al_49_al_300_correlation.csv`.

**Yorum:** r=0.0672, pratikte **sıfıra yakın, korelasyonsuz**. `n=147` (334 `AL_` kolonunun
yüksek eksiklik oranı nedeniyle iki kolonun örtüşen-dolu satır sayısı düşük, ama `min_periods=30`
eşiğinin oldukça üzerinde — güvenilir bir tahmin). **Sonuç: `AL_49`, `AL_300`'ün bir
temsilcisi/vekili değil — ID3'ün bulduğu sinyal, marjinal korelasyon açısından gerçekten
bağımsız/tamamlayıcı bir eksen.** Bu, 03c'de "ID3 mi bağımsız yeni sinyal buluyor yoksa aynı
sinyalin farklı bir temsilcisini mi seçiyor" sorusuna net bir cevap: **bağımsız.**

---

## Adım 2 — Ağaç İçi Split Yapısı: Co-occurrence + Derinlik

`CHOSEN_MAX_DEPTH=5`, aynı 50 dış fold, `id3_feature_selection.id3_selected` ile birebir aynı
parametrelerle (`criterion="entropy"`, `class_weight="balanced"`, aynı seed formülü) ağaçlar
yeniden fit edildi — bu sefer `feature_importances_` yerine doğrudan `tree_` yapısı (hangi
özellik hangi derinlikte bölünüyor) okundu.

**Determinizm doğrulaması:** Yeniden fit edilen ağaçlardan hesaplanan seçim oranları, 03c'nin
kaydedilmiş `id3_feature_selection_stability.csv` değerleriyle **birebir eşleşti**:

| Özellik | Bu turda yeniden hesaplanan | 03c'de kayıtlı |
|---|---|---|
| `EK_7` | 0.88 | 0.88 |
| `AL_49` | 0.78 | 0.78 |
| `CAT_1` | 0.68 | 0.68 |

Aynı seed → aynı ağaçlar — beklenen tutarlılık teyit edildi (sklearn `DecisionTreeClassifier`,
sabit `random_state` ile deterministik).

### Kök-bölünmesi frekansı ve tipik derinlik

| Özellik | Kullanıldığı fold (50 üzerinden) | **Kök** olduğu fold | Ort. en sığ derinlik |
|---|---|---|---|
| **`EK_7`** | 44 (%88) | **32 (%64)** | **0.59** |
| **`AL_49`** | 39 (%78) | **0 (%0)** | 1.51 |
| **`CAT_1`** | 34 (%68) | **0 (%0)** | 2.71 |

**Bulgu — net bir hiyerarşi var:** `EK_7`, kullanıldığı fold'ların %73'ünde (32/44) ağacın
**kökü**; `AL_49` ve `CAT_1` ise **hiçbir fold'da kök değil** (0/50) — ikisi de her zaman
`EK_7`'den (veya ara sıra başka bir özellikten) sonra, ikincil/daha derin bir dalda devreye
giriyor. Ortalama en sığ derinlik sıralaması da aynı hiyerarşiyi doğruluyor: `EK_7` (0.59) <
`AL_49` (1.51) < `CAT_1` (2.71).

### İkili "kim önce bölünüyor" karşılaştırması (yalnızca ikisinin de aynı fold'da göründüğü durumlar)

| Çift | Birlikte göründüğü fold | Daha sığ olan | Oran |
|---|---|---|---|
| `EK_7` vs `AL_49` | 36/50 | `EK_7` | **33/36 (%92)** |
| `EK_7` vs `CAT_1` | 30/50 | `EK_7` | **26/30 (%87)** |
| `AL_49` vs `CAT_1` | 26/50 | `AL_49` | **22/26 (%85)** |

Üç karşılaştırma da **tutarlı, tek yönlü bir sıralama** veriyor: **`EK_7` → `AL_49` → `CAT_1`**.
Bu tesadüfi değil — 03c'nin diğer 4 yöntemdeki konsensüs gücü sıralamasıyla da örtüşüyor
(`EK_7` 4/4 diğer yöntem, `AL_49` 3/4, `CAT_1` 2/4 — bkz. `03c` Adım 2 tablosu): ID3'ün
ağaç-içi hiyerarşisi, diğer yöntemlerin bağımsız ölçtüğü sinyal gücü sıralamasını **yapısal
olarak da yeniden üretiyor.**

### Co-occurrence — en güçlü eşleşme `AL_49` ↔ `EK_7`

`reports/tables/id3_split_cooccurrence.csv` (405 özellik arası tüm gözlenen çiftler, azalan
co-occurrence sırasına göre). Üç odak özelliğin birbiriyle ve en yakın komşularıyla ilişkisi:

| Çift | Co-occurrence (50 fold) | Jaccard |
|---|---|---|
| **`AL_49` ↔ `EK_7`** | **36** | **0.766** |
| `CAT_1` ↔ `EK_7` | 30 | 0.625 |
| `AL_300` ↔ `EK_7` | 26 | 0.578 |
| `AL_49` ↔ `CAT_1` | 26 | 0.553 |
| `AL_300` ↔ `AL_49` | 23 | 0.535 |
| `AL_300` ↔ `CAT_1` | 19 | 0.452 |

**`AL_49`-`EK_7` çifti, tablodaki en yüksek co-occurrence/jaccard değerlerinden biri** — `AL_49`
kullanıldığında `EK_7` neredeyse her zaman (36/39 = %92) aynı ağaçta da kullanılıyor. Bu, Adım
1'in "AL_49 bağımsız bir sinyal" bulgusuyla **çelişmiyor, onu tamamlıyor**: `AL_49` ile `AL_300`
arasında korelasyon yok (r=0.067), ama `AL_49` ile `EK_7` arasında güçlü bir **yapısal
birliktelik** var — yani ID3, `AL_49`'u `EK_7`'nin bıraktığı artık/tamamlayıcı bilgiyi
yakalamak için kullanıyor gibi görünüyor (klasik bir "birincil sinyal + onu inceltilen ikincil
sinyal" deseni), `AL_300`'ün yerine geçen bir temsilci olarak değil.

**İlginç yan bulgu — `AL_300` da güçlü bir oyuncu, ama farklı bir rolde:** `AL_300` (ID3'ün
kendi %60 eşiğinin hemen altında, 0.54 oranla) aslında 405 özellik içinde **4. en sık kullanılan**
özellik (27/50 fold) ve 4 fold'da bizzat kök oluyor — `EK_7`'den sonra en sık kök olan ikinci
özellik. `AL_300`, `AL_49` ile yalnızca 23/50 fold'da birlikte görünüyor (jaccard 0.535) —
ikisi çoğunlukla **birbirinin yerine geçen alternatif ikincil-split adayları** gibi davranıyor
(pairwise korelasyonları sıfıra yakın olmasına rağmen), `EK_7` her ikisiyle de benzer sıklıkta
eşleşiyor. Bu, `AL_300`'ün de resmi ID3 havuzuna (eşiğin hemen altında kalsa da) yakın bir aday
olduğunu gösteriyor — ama bu turun kapsamı dışında, yalnızca not ediliyor.

### `CAT_1`'in provenance-confound bulgusuyla ilişkisi

`al_all_missing` ile `CAT_1`/`AL_49`/`EK_7` arasındaki co-occurrence **orta düzeyde**
(`CAT_1`↔`al_all_missing`: 10/50, jaccard 0.256; `AL_49`↔`al_all_missing`: 12/50, jaccard
0.286; `EK_7`↔`al_all_missing`: 12/50, jaccard 0.255) — üçü de birbirine yakın, `al_all_missing`
özellikle bu üçlüden biriyle güçlü/ayrıcalıklı bir eşleşme göstermiyor.

Daha çarpıcı olan, `CAT_1`'in **ağaçtaki konumu**: `CAT_1` hiçbir zaman kök değil ve ortalama en
sığ derinliği (2.71) üçü arasında en yüksek — yani ID3, `CAT_1`'i tipik olarak `EK_7` (ve çoğu
zaman `AL_49`) zaten bir bölünme yaptıktan **sonra**, ikincil/inceltici bir ayırt edici olarak
kullanıyor, birincil/dominant bir ayırt edici olarak değil. Bu, `03b_MISSINGNESS_PROVENANCE_
KONTROLU_PAH.md`'deki confound bulgusunu (`al_all_missing`↔`CAT_1` eksikliği arasında
Cramér's V=0.755) **ne doğruluyor ne çürütüyor** — yalnızca yapısal bir nüans ekliyor:
`CAT_1`'in ayırt ediciliği ID3 açısından gerçek ve tekrar üretilebilir (03c'de zaten
gösterilmişti), ama ağaç hiyerarşisinde **birincil değil ikincil** bir rolde — bu, hem "gerçek
tamamlayıcı biyolojik sinyal" hem de "artık/provenance-kaynaklı ince ayar sinyali"
hipotezleriyle tutarlı olabilir; bu turun analiziyle ayırt edilemiyor. **Aşama F'nin
adversarial validation kontrolüne eklenebilecek somut bir ayrıntı:** `CAT_1`'in etkisi
`EK_7`/`AL_49` zaten hesaba katıldıktan sonraki artık sinyalde aranmalı, tek başına marjinal
etkisinde değil.

---

## Kısa Özet

1. **`AL_49` bağımsız bir sinyal, `AL_300`'ün temsilcisi değil.** Korelasyon sıfıra yakın
   (r=0.067) ve hiçbir korelasyon-kümesi eşiğinde (r0.5'e kadar) aynı kümede değiller. Bu daha
   dikkat çekici olan sonuç: ID3 gerçekten tamamlayıcı bir sinyal buluyor.
2. **ID3'ün 3 stabil özelliği arasında net, tekrar üretilebilir bir hiyerarşi var:**
   `EK_7` (neredeyse her zaman kök/en sığ) → `AL_49` (ikincil) → `CAT_1` (en derin/en son).
   Bu sıralama, diğer 4 yöntemin bağımsız ölçtüğü konsensüs-gücü sıralamasıyla örtüşüyor.
3. **`AL_49`-`EK_7` en güçlü yapısal eşleşme** (36/50 co-occurrence, jaccard 0.766) — birincil
   sinyal + tamamlayıcı ikincil sinyal deseni; klasik bir "aynı bilgiyi tekrarlama" değil.
4. **`CAT_1`, ağaçta her zaman ikincil/derin bir rolde** — provenance-confound riskini ne
   doğruluyor ne çürütüyor, ama Aşama F'nin test tasarımına ("`CAT_1`'in artık etkisi,
   `EK_7`/`AL_49` sonrasında test edilmeli") somut bir ayrıntı ekliyor.

## Teslim Edilen Dosyalar

- `notebooks/id3_followup_analysis.py` (yeni, tek seferlik keşif script'i)
- `reports/tables/al_49_al_300_correlation.csv`
- `reports/tables/id3_split_cooccurrence.csv`
- `reports/tables/id3_split_depth_summary.csv` (destekleyici — kök-frekansı/derinlik özeti)
- `reports/03d_ID3_TAKIP_ANALIZI_PAH.md` (bu dosya)

Mevcut hiçbir dosya (`feature_selection.py`, `fold_features.py`, `id3_feature_selection.py`,
`data/splits/pah/*`, `data/processed/pah/*.parquet`, `configs/pah/*.yaml`,
`reports/03c_ID3_YONTEM_EKLEME_PAH.md`, `reports/tables/feature_selection_stability.csv`,
`reports/tables/id3_feature_selection_stability.csv`, `AL_spearman_corr.csv`,
`AL_correlation_clusters.csv` vb.) değiştirilmedi.
