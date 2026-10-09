# ⚠️ Bu deney resmi GENOVA PAH pipeline'ını DEĞİŞTİRMEZ

**Statü: bağımsız, araştırma amaçlı deney (Faz 2: Aykırı Gözlem + Kısmi
Korelasyon/Koşullu IG).** `data/`, `configs/`, `reports/` (raporlar dahil —
`03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md` dahil), `src/genova/pah/*.py`
— hiçbirine bu deneyde yazılmadı. Bölüm B'nin sonucu ne olursa olsun,
`03b`'ye eklenip eklenmeyeceği **ayrı bir onay turunda** karar verilecek.

## Ne test ediliyor?

**Bölüm A:** Final 25-özellik havuzunda tek bir satırın (veya birkaçının)
nested CV performansını orantısız etkileyip etkilemediği — Faz 1'in Model
B'si (21-ağaç entropy ensemble) değerlendirme aracı olarak kullanılarak,
fold-koruyan Leave-One-Out ile.

**Bölüm B:** `CAT_1`'in `Label` ile ilişkisinin, `al_all_missing` kontrol
edildiğinde nasıl değiştiği — kısmi korelasyon + koşullu Information Gain
ile, `03b`'nin provenance-confound sorusuna tamamlayıcı bir açıdan bakarak.

## Nasıl çalıştırılır

```bash
cd experiments/outlier_partial_corr_analizi/scripts
python precheck.py             # ZORUNLU ilk adım
python loo_influence.py        # Bölüm A (~25 dk, 16 nested Model-B kosumu)
python cat1_partial_corr.py    # Bölüm B (saniyeler icinde)
```

## Sonuçlar

Tam sonuçlar ve nihai değerlendirme: `results/final_report.md`.
Bölüm B'nin ayrıntılı yazımı: `results/cat1_partial_analysis.md`.

**Kısa özet:** Bölüm A, hiçbir tekil satırın (14 aday arasında en büyük
etki +0.0079 F1) performansı orantısız etkilemediğini gösterdi — mevcut
25-özellik havuzunun sonucu birkaç aşırı gözleme dayanmıyor. Bölüm B,
beklenmedik bir "seyreltme" (Simpson-paradoksu-benzeri) deseni buldu:
`al_all_missing`'i kontrol edince `CAT_1`↔`Label` ilişkisi zayıflamak yerine
güçleniyor — bu, provenance-confound'dan çok biyolojik-sinyal hipotezini
hafifçe destekliyor (kanıtlamıyor). İkisi de F1 adversarial validation'ın
yerini tutmuyor.
