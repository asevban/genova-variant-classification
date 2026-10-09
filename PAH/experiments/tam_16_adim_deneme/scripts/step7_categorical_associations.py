"""ADIM 7 -- Kategorik ozellikler icin uygun iliski olculeri.

GENOVA'nin resmi pipeline'i yalnizca al_all_missing x CAT_1/CAT_2 icin
Cramer's V hesaplamisti (03b). Burada TUM CAT_/AA_ kolonlari icin:
  - Kategorik-kategorik: Cramer's V (tum ciftler)
  - Kategorik-sayisal: Eta (correlation ratio)
  - Kategorik-hedef: Cramer's V
  - Hedef bilgisi: Mutual Information

Kategorileri rastgele tamsayi kodlayip Pearson hesaplama HATASI YAPILMADI --
tum olculer kategorik veri icin uygun (siralama-bagimsiz) yontemlerle
hesaplandi.

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_raw, categorical_columns_raw, numeric_columns_raw, safe_print  # noqa: E402


def cramers_v(x, y):
    """Bias-duzeltilmis Cramer's V (Bergsma 2013 duzeltmesi)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, ct.values.sum()
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = np.sqrt(phi2_corr / denom) if denom > 0 else np.nan
    return v, p, n


def eta_correlation_ratio(categorical, numeric):
    """Eta: kategori-ici varyans / toplam varyans farki (correlation ratio).
    0=iliski yok, 1=kategori numeric'i tam belirliyor."""
    df = pd.DataFrame({"cat": categorical, "num": numeric}).dropna()
    if df["cat"].nunique() < 2 or len(df) < 5:
        return np.nan, len(df)
    grand_mean = df["num"].mean()
    ss_between = sum(len(g) * (g["num"].mean() - grand_mean) ** 2 for _, g in df.groupby("cat"))
    ss_total = ((df["num"] - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return np.nan, len(df)
    return float(np.sqrt(ss_between / ss_total)), len(df)


def main():
    raw = load_raw()
    y = raw["Label"].to_numpy()
    cat_cols = [c for c in categorical_columns_raw(raw) if c != "CAT_6"]
    num_cols = numeric_columns_raw(raw)

    safe_print(f"{len(cat_cols)} kategorik kolon ({cat_cols}), CAT_6 (%100 eksik) elendi.")

    # --- kategorik-kategorik: Cramer's V, tum ciftler ---
    cc_rows = []
    for c1, c2 in combinations(cat_cols, 2):
        v, p, n = cramers_v(raw[c1].fillna("__NA__"), raw[c2].fillna("__NA__"))
        cc_rows.append({"col_i": c1, "col_j": c2, "cramers_v": v, "p_value": p, "n": n})
    cc_df = pd.DataFrame(cc_rows).sort_values("cramers_v", ascending=False)
    cc_df.to_csv(RESULTS_DIR / "step7_categorical_categorical_cramers_v.csv", index=False)

    # --- kategorik-hedef: Cramer's V + MI ---
    target_rows = []
    for c in cat_cols:
        v, p, n = cramers_v(raw[c].fillna("__NA__"), raw["Label"])
        x_encoded = raw[c].fillna("__NA__").astype("category").cat.codes.to_numpy().reshape(-1, 1)
        mi = float(mutual_info_classif(x_encoded, y, discrete_features=True, random_state=42)[0])
        target_rows.append({"column": c, "cramers_v_vs_Label": v, "p_value": p, "n": n, "mutual_info_vs_Label": mi})
    target_df = pd.DataFrame(target_rows).sort_values("cramers_v_vs_Label", ascending=False)
    target_df.to_csv(RESULTS_DIR / "step7_categorical_vs_target.csv", index=False)

    # --- kategorik-sayisal: Eta (tum kategorik x tum sayisal -- 8*343=2744 kombinasyon) ---
    eta_rows = []
    for c_cat in cat_cols:
        for c_num in num_cols:
            eta, n = eta_correlation_ratio(raw[c_cat], raw[c_num])
            eta_rows.append({"categorical": c_cat, "numeric": c_num, "eta": eta, "n": n})
    eta_df = pd.DataFrame(eta_rows).sort_values("eta", ascending=False, na_position="last")
    eta_df.to_csv(RESULTS_DIR / "step7_categorical_numeric_eta.csv", index=False)
    # AYNI sahte-yuksek-korelasyon riski (bkz. Adim 6): dusuk n'de eta da
    # sismis cikabilir -- min_periods=30 konvansiyonu burada da uygulanir.
    MIN_N_ETA = 30
    top_eta_reliable = eta_df[eta_df["n"] >= MIN_N_ETA].dropna(subset=["eta"]).head(15)
    n_low_n_high_eta = ((eta_df["n"] < MIN_N_ETA) & (eta_df["eta"] >= 0.7)).sum()

    # --- CAT_3=CAT_4=CAT_5 capraz-dogrulama (onceki bulgu, bagimsiz tekrar) ---
    v345_34, _, _ = cramers_v(raw["CAT_3"], raw["CAT_4"])
    v345_35, _, _ = cramers_v(raw["CAT_3"], raw["CAT_5"])

    report = f"""# Adım 7 — Kategorik Özellikler İçin Uygun İlişki Ölçüleri

Kategoriler **rastgele tamsayı kodlanıp Pearson hesaplanmadı** — sahte
sıralama riski nedeniyle tüm ölçümler kategorik veri için uygun, sıralama-
bağımsız yöntemlerle yapıldı: Cramér's V (kategorik-kategorik / kategorik-
hedef), Eta/correlation ratio (kategorik-sayısal), Mutual Information
(kategorik-hedef).

## Kategorik-Kategorik: En Güçlü İlişkiler

{cc_df.head(10)[["col_i","col_j","cramers_v","p_value","n"]].to_string(index=False)}

**Çapraz-doğrulama (önceki denetimin bağımsız tekrarı):** `CAT_3`↔`CAT_4`
Cramér's V = **{v345_34:.4f}**, `CAT_3`↔`CAT_5` = **{v345_35:.4f}** — bu üç
kolonun birebir aynı olduğu bulgusu (CLAUDE.md'ye eklenmişti) burada da
{"tam olarak V=1.0 ile doğrulandı" if v345_34 > 0.999 else "V≈1'e çok yakın çıktı"}.

## Kategorik-Hedef İlişkisi (Cramér's V + Mutual Information)

{target_df.to_string(index=False)}

## Kategorik-Sayısal: En Güçlü 15 Eta İlişkisi (n≥{MIN_N_ETA}, güvenilir)

**Metodolojik not (Adım 6'daki aynı tuzağın burada da yakalanması):** Filtre
uygulanmadan önce en yüksek eta değerleri (~0.99) hep **n=14** gibi çok
düşük ortak-gözlem sayılarında çıktı — Adım 6'daki Pearson sahte-yükseklik
sorunuyla birebir aynı desen. **{n_low_n_high_eta} kombinasyon** n<{MIN_N_ETA}
iken eta≥0.7 gösteriyordu, bunlar güvenilmez kabul edilip filtrelendi.
Aşağıdaki tablo yalnızca n≥{MIN_N_ETA} olan, güvenilir sonuçları gösteriyor:

{top_eta_reliable[["categorical","numeric","eta","n"]].to_string(index=False)}

Tam sonuçlar (filtresiz, `n` kolonuyla birlikte — yorumlarken mutlaka
kontrol edin): `results/step7_categorical_categorical_cramers_v.csv` (kategorik-kategorik,
tüm çiftler), `step7_categorical_vs_target.csv` (hedef ilişkisi), `step7_categorical_numeric_eta.csv`
(kategorik-sayısal, {len(eta_df)} kombinasyon).
"""
    (RESULTS_DIR / "step7_categorical_report.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
