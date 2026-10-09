# ⚠️ Bu deney resmi GENOVA PAH pipeline'ını DEĞİŞTİRMEZ

**Statü: bağımsız, araştırma amaçlı deney (Faz 1: Model Karşılaştırması).**
Resmi pipeline'ın (`v4_final_feature_pool.json`, `data/splits/pah/` split
bankası, `feature_selection.py`'nin ürettiği 25-özellik konsensüs havuzu)
**hiçbir parçası bu klasördeki hiçbir script tarafından değiştirilmedi**.
Bu klasördeki her script yalnızca `experiments/entropy_tree_ensemble_deneme/`
altına yazar; `src/genova/pah/` modüllerini yalnızca **import** eder,
`data/splits/pah/`'ı ve `reports/tables/v4_final_feature_pool.json`'ı
yalnızca **okur**.

**Sonuç iyi çıkarsa:** Bu ensemble ailesi Aşama E2'ye resmi bir aday
model olarak önerilir — bu, ayrı bir onay ve ayrı bir entegrasyon süreci
gerektirir, bu deneyin kendisi bir entegrasyon değildir.

**Sonuç kötü/belirsiz çıkarsa:** Yalnızca bu klasörde kalır, ana projeyi
hiçbir şekilde etkilemez.

---

## Ne test ediliyor?

Entropy-tree (Information Gain mantığını kullanan modern binary-split
decision tree — **"saf ID3" değil**, bkz.
`reports/03c_ID3_YONTEM_EKLEME_PAH.md`'nin terminoloji ayrımı) tabanlı bir
ensemble ailesi, split bankasının aynı 50 dış fold'unda, mevcut 25-özellik
konsensüs havuzuna karşı nasıl performans gösteriyor?

| Model | Özellik seti | Yapı |
|---|---|---|
| A | 25 (mevcut havuz) | Tek entropy tree (basit referans) |
| B | 25 (mevcut havuz) | 21-ağaç, dengeli-bootstrap, hard-voting ensemble (**ana hipotez**) |
| C | 405 (tüm aday evren) | Ağaç başına rastgele 70 özellik, 21-ağaç ensemble |
| D | 24 (Model B − `CAT_1`) | Model B'nin `CAT_1`-ablasyonu |

## Nasıl çalıştırılır

```bash
cd experiments/entropy_tree_ensemble_deneme/scripts
python precheck.py         # ZORUNLU ilk adım -- herhangi bir madde FAIL ise dur
python run_experiment.py   # yalnızca precheck PASS ise çalıştır
```

## Metodoloji özeti

- Split bankası (`data/splits/pah/`, 50 dış fold × 4 iç fold) **olduğu gibi**
  okunur, yeniden üretilmez.
- Eşik seçimi (B/C/D için, 21 oydan 11-19 arası oy-eşiği) **yalnızca iç
  fold'larda** aranır; dış test fold'u yalnızca seçilmiş sabit eşikle,
  tek seferde skorlanır (bkz. `scripts/threshold_search.py`, `run_experiment.py`
  içindeki nested prosedür).
- Ön işleme `genova.pah.fold_features.build_fold_features` ile yapılır —
  `MultiValueFrequencyEncoder`, `al_all_missing`, fold-içi medyan/frekans
  öğrenme dahil, resmi pipeline'daki TÜM düzeltmeler otomatik miras alınır.
- `CAT_1` ablasyonu **yalnızca Model B** üzerinde yapılır — Model B, dış
  fold sonuçlarına bakılıp "en iyisi seçildiği" için değil, **önceden,
  tasarım gereği** ana hipotez olduğu için seçildi (aksi hâlde ablasyon
  gizli bir model-seçim adımından sızıntı almış olurdu).
- "E-F dokümanı"/"E4 formülü" bu repoda **bulunamadı** (arandı, doğrulandı) —
  prevalans-ayarlı metrikler bunun yerine standart, doğrulanabilir bir
  yöntemle (Saerens, Latinne & Decaestecker, 2002 usulü — TPR/TNR'nin sınıf
  önceliğinden bağımsız kabul edilmesi) hesaplandı; formül
  `scripts/run_experiment.py::prevalence_adjust()` içinde açıkça yazılı.

Tam sonuçlar ve nihai değerlendirme: `results/final_report.md`.
