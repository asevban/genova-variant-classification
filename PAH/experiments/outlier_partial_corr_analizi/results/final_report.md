# Faz 2 — Aykırı Gözlem + Kısmi Korelasyon/Koşullu IG Analizi

> ⚠️ **Bu rapor bağımsız, araştırma amaçlı bir deneyin sonucudur — resmi
> GENOVA PAH pipeline'ının hiçbir parçasını değiştirmez.** `data/`,
> `configs/`, `reports/` (raporlar dahil, `03b` dahil), `src/genova/pah/*.py`
> — hiçbirine yazılmadı (bkz. §3 izolasyon doğrulaması). Bölüm B'nin sonucu
> ne kadar net olursa olsun, `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`'ye
> eklenip eklenmeyeceğine **ayrı bir onay turunda** karar verilecek.

---

## PRECHECK — 7/7 madde PASS

`results/precheck_log.txt`: split bankası 50 dış/4 iç fold doğrulandı,
`build_fold_features` yeniden yazılmadan import edildi, `v4_final_feature_pool.json`
25 özellik verdi, **`AL_correlation_summary.json`'da `|r|≥0.90` eşiğinde
`n_clusters_size_gt1 = 0`** doğrulandı (görev metninin ön-bilgisiyle
birebir eşleşti — bu yüzden Bölüm A, korelasyon-çift analizini değil `03d`'nin
yapısal (co-occurrence/jaccard) bulgusunu kullandı), kaynak-kod taraması
official dosyalara yazma girişimi bulamadı.

---

## Bölüm A — Aykırı Gözlem / Etkili Nokta Analizi

### Yöntem

Faz 1'in Model B'si (`entropy_tree_ensemble_deneme/scripts/{entropy_tree,ensemble,threshold_search}.py`)
**doğrudan import edildi**, yeniden yazılmadı. İki aşama:

1. **Ekstremlik taramasi (betimsel, sızıntısız):** 25-özellik havuzunda her
   369 satırın rank/percentile-tabanlı ekstremlik skoru hesaplandı.
   **Metodolojik not:** İlk denemede robust z-skoru (medyan/MAD) kullanıldı,
   ama `AL_` frekans-tipi kolonların sıfır-şişkinliği (CLAUDE.md, EDA A.6)
   MAD'i sıfıra yaklaştırıp z-skorlarını anlamsız/patlamış değerlere
   (yüzlerce milyon) taşıdı — bu ampirik olarak gözlendi ve **rank-tabanlı,
   ölçek-bağımsız bir ölçüme geçilerek düzeltildi** (`influence_screening.csv`).
2. **Fold-koruyan Leave-One-Out:** Kısa listedeki her aday satır için, split
   bankasının **mevcut fold üyelikleri korunarak** (yeniden split
   üretilmeden) o satır train/test/val'den filtrelendi, Model B'nin tam
   nested prosedürü (iç-fold eşik araması → dış-eğitimde final ensemble →
   dış-test tek seferlik skor) yeniden çalıştırıldı.
   **Tasarım kararı (açıkça belgeleniyor):** Görev metni "split yapısını
   yeniden türet" diyordu — bu, sıfırdan yeni bir `StratifiedGroupKFold`
   çalıştırmak olarak da okunabilirdi. Bu seçilmedi çünkü yeniden-split,
   satırın etkisini fold-atamasının rastgeleliğiyle karıştırırdı (gözlenen
   fark hem satırın yokluğundan hem tamamen farklı bir rastgele bölünmeden
   kaynaklanırdı). Fold-koruyan yaklaşım tek değişkeni satırın kendisi
   yapıyor — split bankasının "yalnızca okunur" kuralıyla da daha tutarlı.

`conflict_group_1` (`VAR_003238`/`VAR_003234`) kısa listeye **girmedi**
(ekstremlik/ortak-kuyruk kriterlerini karşılamadı) — bu yüzden özel durum
fiilen ortaya çıkmadı, ama tasarımın bu durumu nasıl ele alacağı kod
docstring'inde belgelendi: fold-koruyan yaklaşımda tek bir üyenin
çıkarılması bir bütünlük ihlali oluşturmuyor (grup-bütünlüğü kuralı split
*üretilirken* uygulanıyordu, burada fold'lar zaten sabit).

### Sonuçlar — En Etkili 10 Satır

Referans (hiçbir satır çıkarılmamış, Faz 1'in Model B'siyle **birebir aynı
kod yolu**): **F1=0.8538, MCC=0.3742** (50 dış fold) — Faz 1'in kaydedilmiş
sonucuyla eşleşiyor, iç-tutarlılık doğrulandı.

| Variant_ID | Label | ΔF1 (çıkarılınca) | ΔMCC | En ekstrem özellik (percentile) |
|---|---|---|---|---|
| VAR_003118 | 1 | **+0.0079** | +0.0203 | `AL_301` (P=0.997) |
| VAR_003085 | 1 | +0.0062 | +0.0182 | `AL_329` (P=1.000) |
| VAR_002763 | 1 | +0.0058 | +0.0060 | `AL_317` (P=0.999) |
| VAR_003168 | 1 | +0.0046 | +0.0127 | `AL_306` (P=1.000) |
| VAR_003021 | 1 | +0.0045 | +0.0062 | `AL_49` (P=1.000) |
| VAR_002566 | 1 | +0.0033 | +0.0128 | `AL_298` (P=1.000) |
| VAR_003151 | 1 | +0.0033 | +0.0003 | `AL_334` (P=0.999) |
| VAR_002998 | 1 | +0.0032 | +0.0028 | `AL_8` (P=1.000) |
| VAR_002765 | 1 | +0.0031 | +0.0072 | `AL_26` (P=1.000) |
| VAR_002792 | 1 | +0.0030 | +0.0015 | `AL_49` (P=0.986) |

*(Tam 14 adaylık liste ve her satırın en ekstrem 3 özelliği:
`results/loo_top10.csv`.)*

**Bulgu — hiçbir satır orantısız/dominant etki göstermiyor:** Kısa listedeki
**14 adayın 14'ünde de ΔF1 pozitif** (satır çıkarılınca F1 hafifçe
*iyileşiyor*), en büyüğü yalnızca **+0.0079** (referans F1'in ~%0.9'u).
Hiçbir tekil satır F1/MCC'yi çökertmiyor ya da tek başına taşımıyor — bu,
`class_weight="balanced"` + dengeli-bootstrap tasarımının (Faz 1) genuinely
sağlam olduğunun, birkaç aşırı gözleme dayanmadığının ek bir kanıtı.
**Not:** Tüm 10 satırın `Label=1` (patojenik) olması tesadüf değil — ekstrem
`AL_` değerleri ("hiç gözlenmemiş"e yakın frekanslar) EDA'nın zaten
bulduğu "patojenik ile ilişkili nadir-gözlem sinyali"yle tutarlı; bu
liste yalnızca %83 patojenik eğitim setinin doğal bir yansıması da olabilir
(14/14'ünün Label=1 çıkması, taban oranı %83 iken beklenenden daha yüksek
ama küçük örneklemde (n=14) istatistiksel olarak şaşırtıcı değil).

### 3-Senaryo Karşılaştırması

| Senaryo | n çıkarılan | F1 (ort.) | MCC (ort.) |
|---|---|---|---|
| (a) Hiçbiri (referans) | 0 | 0.8538 ± 0.0348 | 0.3742 ± 0.1237 |
| (b) En etkili tekil satır (`VAR_003118`) | 1 | 0.8616 | 0.3945 |
| (c) En etkili ilk 5 satır birlikte | 5 | 0.8624 ± 0.0330 | 0.3837 ± 0.1214 |

Tek satır çıkarılınca değişim (+0.0079 F1) ile 5 satır birlikte çıkarılınca
değişim (+0.0087 F1) **neredeyse aynı** — yani etkiler **toplanmıyor/
süperadditif değil**, bu da satırların birbirinden bağımsız, küçük ve
örtüşen (aynı yönde) katkılar yaptığını gösteriyor; "birlikte kaldırılınca
performans dramatik sıçrar" senaryosu **gözlenmedi**.

**Silme önerisi YOK** (görev kuralı gereği) — bu yalnızca bir ölçüm,
hiçbir satırın veri girişi/ölçüm hatası olduğuna dair kanıt aranmadı/bulunmadı.

---

## Bölüm B — `CAT_1` × `Label` Kısmi İlişki

Tam analiz: `results/cat1_partial_analysis.md`. Özet:

| Ölçüm | Koşulsuz | `al_all_missing=0` alt-grubunda (n=280) |
|---|---|---|
| Spearman r | 0.0552 (p=0.29, anlamsız) | Kısmi r = **0.2612** |
| Information Gain | **0.0000** (tam sıfır) | **0.0381** |

**Beklenmedik desen:** Görev metninin ön-kaydettiği iki yönlü kural
("düşüyorsa confound'u destekler, değişmiyorsa biyolojiyi destekler") bu
sonucu tam karşılamıyor — koşullu IG **düşmüyor, artıyor** (sıfırdan
0.0381'e). Bu, `al_all_missing=1` alt-grubunun (n=89, `CAT_1` bu alt-grupta
kodlama gereği **sabit=0**) havuzlanmış/koşulsuz analize yalnızca gürültü
katıp gerçek ilişkiyi **seyrelttiğinin** işareti — klasik bir Simpson-
paradoksu-benzeri desen. Kısmi korelasyon da aynı yönde (0.055→0.261,
büyüyor). **Sonuç: bu iki ölçüm, biyolojik-sinyal hipotezini hafifçe
destekliyor** — eğer `CAT_1`'in tüm görünür sinyali yalnızca
`al_all_missing`'in bir vekili olsaydı, `al_all_missing`'i sabitleyip
`CAT_1`'in fiilen gözlemlendiği popülasyona bakınca ilişkinin
GÜÇLENMESİ değil kaybolması beklenirdi.

**Temkin payı:** (a) Mutlak IG değerleri küçük (<0.04), (b) koşulsuz IG'nin
tam 0 çıkması `mutual_info_classif`'in (KSG tahminleyici) `CAT_1`'in ağır
bağ-degenerasyonuna (369 satırda yalnızca 16 benzersiz değer) karşı bilinen
hassasiyetinden kaynaklanıyor olabilir, (c) bu "hafifçe destekliyor" —
"kanıtlıyor" değil. **F1 adversarial validation hâlâ asıl karar noktası.**

---

## Genel Değerlendirme

- **Bölüm A:** Final 25-özellik havuzunun nested CV performansı, birkaç
  aşırı/uç gözleme bağımlı değil — bu, mevcut Aşama D.1 sonucunun sağlamlığına
  ek bir güven noktası.
- **Bölüm B:** İki bağımsız istatistiksel yöntem (kısmi korelasyon + koşullu
  IG), `03b`'nin açık bıraktığı soruda **biyolojik-sinyal yönüne hafifçe
  eğiliyor** — ama Faz 1'in `CAT_1` ablasyon bulgusuyla (istatistiksel olarak
  anlamsız, p=0.096) doğrudan çelişmiyor, tamamlayıcı bir açıdan bakıyor:
  Faz 1 "`CAT_1` çıkınca ensemble performansı çökmüyor" derken, Faz 2 "`CAT_1`
  var olan ilişkisi, en azından gözlemlendiği popülasyonda, artefaktla
  tam açıklanamıyor" diyor — ikisi birbirini dışlamıyor (küçük ama gerçek
  bir sinyal, ensemble'ın onu kullanmakta zorlanması ile bir arada var olabilir).
- **İkisi de F1 adversarial validation'ın yerini tutmuyor.** Bu turda hiçbir
  resmi rapora yazılmadı — sonuçların `03b`'ye eklenip eklenmeyeceği ayrı
  bir onay turunda değerlendirilecek.

---

## İzolasyon Doğrulaması

`data/`, `configs/`, `reports/` (tamamı — `reports/tables/` dahil), 
`src/genova/pah/*.py` altındaki **173 dosyanın** mtime'ı deney öncesi/sonrası
karşılaştırıldı — **sıfır fark** (bkz. bir sonraki mesajdaki teyit). Üretilen
tüm dosyalar yalnızca `experiments/outlier_partial_corr_analizi/` altında:
`scripts/{precheck,loo_influence,cat1_partial_corr}.py`,
`results/{precheck_log.txt,influence_screening.csv,al49_ek7_joint_tail_rows.csv,loo_top10.csv,loo_scenario_comparison.csv,loo_influence.log,cat1_partial_analysis.md,final_report.md}`.
