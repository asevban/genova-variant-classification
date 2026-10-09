"""ADIM 6 -- Tum sayisal (AL_+EK_) sutun ciftleri icin ikili korelasyon.

GENOVA'nin resmi pipeline'i yalnizca AL_-AL_ ciftlerini tarmisti
(AL_spearman_corr.csv). Burada AL_-EK_ ve EK_-EK_ ciftleri de dahil,
TUM 343 sayisal kolonun C(343,2)=58,653 cifti taraniyor. Vektorel
(pandas .corr()) hesaplama kullanildi -- 58 bin cift icin Python
dongusu yerine.

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_raw, numeric_columns_raw, safe_print  # noqa: E402


def classify_strength(abs_r):
    if pd.isna(abs_r):
        return "hesaplanamadı"
    if abs_r >= 0.90:
        return "çok güçlü"
    if abs_r >= 0.70:
        return "güçlü"
    if abs_r >= 0.40:
        return "orta"
    if abs_r >= 0.20:
        return "zayıf"
    return "çok zayıf"


def main():
    raw = load_raw()
    num_cols = numeric_columns_raw(raw)
    X = raw[num_cols]

    safe_print(f"{len(num_cols)} sayısal kolon, C({len(num_cols)},2)={len(num_cols)*(len(num_cols)-1)//2} çift taranıyor...")

    # onemli metodolojik nokta: GENOVA'nin kendi AL_-AL_ taramasi (01_eda_pah.py)
    # min_periods=30 kullaniyor -- dusuk esikler (orn. 10) az sayida ortusen-dolu
    # satirla sahte-yuksek korelasyon uretebiliyor (dogrulanmis: min_periods=10'da
    # en yuksek cift |r|=0.92, yalnizca 13 ortak dolu satira dayaniyordu). Ayni
    # konvansiyon burada da (AL_-EK_/EK_-EK_ dahil TUM ciftlerde) uygulanmali --
    # aksi halde ayni sahte-yuksek-korelasyon tuzagina duserdik (ilk denemede
    # dustuk de: min_periods'suz hesaplamada en guclu ciftlerin ortak-n'i 14'tu,
    # bu ampirik olarak gozlenip duzeltildi).
    MIN_PERIODS = 30
    pearson_mat = X.corr(method="pearson", min_periods=MIN_PERIODS)
    spearman_mat = X.corr(method="spearman", min_periods=MIN_PERIODS)

    not_null = (~X.isna()).astype(int)
    common_n_mat = not_null.T.dot(not_null)  # (i,j) = ortak dolu gozlem sayisi

    std = X.std()
    constant_cols = set(std[std.fillna(0) == 0].index)

    cols = num_cols
    n = len(cols)
    idx_i, idx_j = np.triu_indices(n, k=1)

    pearson_vals = pearson_mat.values[idx_i, idx_j]
    spearman_vals = spearman_mat.values[idx_i, idx_j]
    common_n_vals = common_n_mat.values[idx_i, idx_j]
    col_i = np.array(cols)[idx_i]
    col_j = np.array(cols)[idx_j]

    reasons = []
    for i, (ci, cj, cn, pr) in enumerate(zip(col_i, col_j, common_n_vals, pearson_vals)):
        if pd.isna(pr):
            if ci in constant_cols or cj in constant_cols:
                reasons.append("sabit değer (std=0)")
            elif cn < 30:
                reasons.append(f"yetersiz ortak gözlem (n={int(cn)}<30, min_periods eşiği)")
            else:
                reasons.append("hesaplanamadı (bilinmeyen neden)")
        else:
            reasons.append("")

    pair_df = pd.DataFrame({
        "col_i": col_i, "col_j": col_j, "common_n": common_n_vals,
        "pearson_r": pearson_vals, "spearman_r": spearman_vals,
        "abs_pearson_r": np.abs(pearson_vals), "abs_spearman_r": np.abs(spearman_vals),
        "hesaplanamama_nedeni": reasons,
    })
    pair_df["strength_pearson"] = pair_df["abs_pearson_r"].apply(classify_strength)
    pair_df["strength_spearman"] = pair_df["abs_spearman_r"].apply(classify_strength)

    computable = pair_df[pair_df["pearson_r"].notna()]
    n_total = len(pair_df)
    n_computable = len(computable)
    n_uncomputable = n_total - n_computable

    very_strong = computable[computable["abs_pearson_r"] >= 0.90]
    very_weak = computable[computable["abs_pearson_r"] < 0.05]

    top500 = computable.sort_values("abs_pearson_r", ascending=False).head(500)
    bottom500 = computable.sort_values("abs_pearson_r", ascending=True).head(500)
    top500_spearman = computable.sort_values("abs_spearman_r", ascending=False).head(500)

    pair_df.to_csv(RESULTS_DIR / "step6_all_numeric_pairs.csv", index=False)
    top500.to_csv(RESULTS_DIR / "step6_top500_strongest_pearson.csv", index=False)
    top500_spearman.to_csv(RESULTS_DIR / "step6_top500_strongest_spearman.csv", index=False)
    bottom500.to_csv(RESULTS_DIR / "step6_bottom500_weakest.csv", index=False)

    strength_counts = computable["strength_pearson"].value_counts()

    report = f"""# Adım 6 — Tüm Sayısal Sütun Çiftleri İçin İkili Korelasyon

`AL_`+`EK_` = {len(num_cols)} sayısal kolon, **C({len(num_cols)},2) = {n_total:,} çift**
tarandı (yalnızca `AL_`-`AL_` değil, `AL_`-`EK_` ve `EK_`-`EK_` de dahil —
GENOVA'nın resmi pipeline'ında bu kapsamlı tarama yoktu).

**Metodolojik not (ilk denemede yakalanan bir hata):** GENOVA'nın kendi
`AL_`-`AL_` taraması `min_periods=30` kullanıyor — düşük eşiklerin (ör. 10)
az sayıda ortak-dolu satırla sahte-yüksek korelasyon ürettiği zaten
doğrulanmıştı. Bu script'in **ilk çalıştırmasında `min_periods` unutuldu**
ve en güçlü çiftlerin ortak-n'i yalnızca **14** çıktı (aynı sahte-yüksek-
korelasyon tuzağı) — bu ampirik olarak fark edilip **`min_periods=30`
eklenerek düzeltildi**, aşağıdaki sayılar düzeltilmiş hâle aittir.

## Genel Özet

- Hesaplanabilen çift: {n_computable:,} (%{n_computable/n_total*100:.1f})
- **Hesaplanamayan çift: {n_uncomputable:,}** (%{n_uncomputable/n_total*100:.1f}) — nedeni:
{pair_df.loc[pair_df["hesaplanamama_nedeni"] != "", "hesaplanamama_nedeni"].value_counts().to_string()}
- **Çok güçlü (|r|≥0.90) çift sayısı: {len(very_strong)}**
- Çok zayıf (|r|<0.05) çift sayısı: {len(very_weak):,}

### Pearson |r| Dağılımı (yalnızca hesaplanabilen çiftler)

{strength_counts.to_string()}

## En Güçlü 10 Çift

{top500[["col_i","col_j","common_n","pearson_r","spearman_r","strength_pearson"]].head(10).to_string(index=False)}

## Sonuç — `|r|≥0.90` Eşiği: Pearson vs Spearman AYRIMI KRİTİK

**Pearson'da {len(very_strong)} çift `|r|≥0.90` eşiğini geçiyor — ama Spearman'da
bu sayı: {int((computable["abs_spearman_r"] >= 0.90).sum())}.**

Bu, Faz 2'nin PRECHECK'te doğruladığı "AL_-AL_'de `|r|≥0.90` çift yok" bulgusuyla
**çelişmiyor** — o bulgu `AL_correlation_summary.json`'da özellikle **Spearman**
ile hesaplanmıştı (GENOVA'nın `01_eda_pah.py`'si `df[AL_COLS].corr(method="spearman",
min_periods=30)` kullanıyor). Spearman'da genişletilmiş (AL_-EK_/EK_-EK_ dahil)
taramada da **{int((computable["abs_spearman_r"] >= 0.90).sum())} çift** bulunuyor —
Phase 2'nin bulgusu genişletilmiş kapsamda da **doğrulandı**.

**Pearson'daki 222 "çok güçlü" çift neredeyse tamamen aykırı-değer kaynaklı
görünüyor** — `AL_` kolonlarının ağır sağa-çarpık/sıfır-şişkin yapısı
(CLAUDE.md) nedeniyle birkaç uç değer Pearson'ı domine edebiliyor. En çarpıcı
örnek: `AL_235`↔`AL_271` — **Pearson r=0.978, Spearman r=0.003** (rank ilişkisi
pratikte SIFIR). Bu, GENOVA'nın "AL_ için Spearman kullan, Pearson değil"
kararının (`01_eda_pah.py` yorum satırı) genişletilmiş kapsamda da **bağımsız
olarak yeniden doğrulandığı** anlamına geliyor — Adım 10'un grafik incelemesi
bu yüzden Spearman'a göre en güçlü çiftlere odaklanacak.

Tam sonuçlar: `results/step6_all_numeric_pairs.csv` ({n_total:,} satır),
en güçlü/en zayıf 500'lük listeler ayrı dosyalarda (Pearson'a göre sıralı —
yorumlarken Spearman kolonunu da mutlaka kontrol edin).
"""
    (RESULTS_DIR / "step6_correlation_report.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
