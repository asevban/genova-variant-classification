"""ADIM 2-3 -- Tum 353 kolon icin Information Gain (yalnizca kesif) +
hipotez eleme listesi.

DUZELTME 1 (gorev metninde belirtildi): Global IG (ham veri, 372 satirin
TAMAMI uzerinde hesaplanir) hicbir eleme kararina DOGRUDAN girdi olmaz --
yalnizca keşif/rapor amaclidir. Adim 3'un onerdigi eleme adaylari birer
HIPOTEZ olarak isaretlenir; gercek dogrulama Adim 13'un nested prosedurunde
yapilir (bu script'te degil).

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import (
    RESULTS_DIR, load_raw, categorical_columns_raw, numeric_columns_raw,
    information_gain_numeric, information_gain_categorical, safe_print,
)


def main():
    raw = load_raw()
    y = raw["Label"].to_numpy()
    num_cols = numeric_columns_raw(raw)
    cat_cols = [c for c in categorical_columns_raw(raw) if c != "CAT_6"]  # %100 eksik, IG hesaplanamaz

    rows = []
    for c in num_cols:
        ig, thr, gr, miss_info = information_gain_numeric(raw[c], y)
        rows.append({
            "column": c, "type": "numeric", "IG": ig, "gain_ratio": gr,
            "best_threshold": thr, "missing_branch_informative": miss_info,
            "missing_rate": raw[c].isna().mean(), "n_unique": raw[c].nunique(),
        })
    for c in cat_cols:
        ig, gr = information_gain_categorical(raw[c], y)
        rows.append({
            "column": c, "type": "categorical", "IG": ig, "gain_ratio": gr,
            "best_threshold": None, "missing_branch_informative": raw[c].isna().any(),
            "missing_rate": raw[c].isna().mean(), "n_unique": raw[c].nunique(),
        })
    # CAT_6 ayrica not edilir (IG hesaplanamaz, %100 eksik)
    rows.append({
        "column": "CAT_6", "type": "categorical", "IG": np.nan, "gain_ratio": np.nan,
        "best_threshold": None, "missing_branch_informative": False,
        "missing_rate": 1.0, "n_unique": 0,
    })

    ig_df = pd.DataFrame(rows).sort_values("IG", ascending=False, na_position="last")
    ig_df.to_csv(RESULTS_DIR / "id3_feature_importance.csv", index=False)

    top20 = ig_df.head(20)
    zero_ig = ig_df[ig_df["IG"].fillna(0) <= 1e-6]

    report_md = f"""# Adım 2 — Tüm 353 Sütun İçin Information Gain (Yalnızca Keşif)

> **Düzeltme 1 uygulandı:** Bu tablo ham veri (372 satırın tamamı) üzerinde
> hesaplandı — dış-fold'ların da göründüğü bir istatistik, bu yüzden
> **hiçbir eleme kararına doğrudan girdi olarak kullanılmıyor**. Adım 3'ün
> önerdiği eleme adayları birer **hipotez**; gerçek doğrulama yalnızca
> Adım 13'ün nested prosedüründe yapılıyor.

Formül: `IG(Label, X) = H(Label) - H(Label|X)` (log2, bit cinsinden).
Sayısal kolonlarda tüm ara-noktalar eşik adayı olarak denendi (ID3-tarzı);
eksiklik/doluluk ayrımının ayrıca bilgilendirici olup olmadığı da
kaydedildi. Gain Ratio, dal-büyüklüğü entropisine bölünerek hesaplandı.

## En Yüksek IG'li 20 Sütun

{top20[["column", "type", "IG", "gain_ratio", "missing_rate"]].to_string(index=False)}

## Genel İstatistikler

- Toplam taranan kolon: {len(ig_df)}
- IG ≈ 0 (≤1e-6) olan kolon sayısı: {len(zero_ig)}
- Eksiklik/doluluk ayrımı bilgilendirici olan kolon sayısı: {int(ig_df['missing_branch_informative'].sum())}

Tam tablo: `results/id3_feature_importance.csv`.
"""
    (RESULTS_DIR / "TUM_SUTUNLAR_INFORMATION_GAIN_RAPORU.md").write_text(report_md, encoding="utf-8")
    safe_print(report_md)

    # --- Adim 3: eleme hipotezi ---
    raw_features = [c for c in raw.columns if c not in ("Variant_ID", "Label")]
    n_unique = raw.nunique()
    missing_rate = raw.isna().mean()

    elimination_rows = []
    ig_lookup = ig_df.set_index("column")

    for c in raw_features:
        reasons = []
        if n_unique.get(c, 99) <= 1:
            reasons.append("sabit (tek benzersiz değer)")
        if missing_rate.get(c, 0) >= 0.90:
            reasons.append(f"çok yüksek eksiklik (%{missing_rate[c]*100:.1f})")
        ig_val = ig_lookup.loc[c, "IG"] if c in ig_lookup.index else np.nan
        if pd.notna(ig_val) and ig_val <= 1e-6:
            reasons.append("çok düşük IG (~0)")
        if c in ig_lookup.index and ig_lookup.loc[c, "type"] == "numeric":
            thr = ig_lookup.loc[c, "best_threshold"]
            n_present = int((raw[c].notna()).sum())
            if thr is not None and n_present > 0:
                below = (raw[c] <= thr).sum()
                above = n_present - below
                if min(below, above) <= 3:
                    reasons.append(f"genellenemeyen tekil-eşikli bölünme (bir tarafta yalnızca {min(below, above)} satır)")

        if reasons:
            elimination_rows.append({
                "column": c, "IG": ig_val, "missing_rate": missing_rate.get(c, np.nan),
                "n_unique": n_unique.get(c, np.nan), "eleme_nedeni": "; ".join(reasons),
                "yerine_tutulan": "",
            })

    # duplicate kolon ciftleri icin: ikincisini "yerine ilki tutulan" olarak isaretle
    feature_cols = raw_features
    checked = set()
    for i, c1 in enumerate(feature_cols):
        if c1 in checked:
            continue
        for c2 in feature_cols[i + 1:]:
            if c2 in checked:
                continue
            if raw[c1].equals(raw[c2]):
                checked.add(c2)
                existing = [r for r in elimination_rows if r["column"] == c2]
                if existing:
                    existing[0]["eleme_nedeni"] += f"; {c1} ile birebir aynı (yinelenen)"
                    existing[0]["yerine_tutulan"] = c1
                else:
                    elimination_rows.append({
                        "column": c2, "IG": ig_lookup.loc[c2, "IG"] if c2 in ig_lookup.index else np.nan,
                        "missing_rate": missing_rate.get(c2, np.nan), "n_unique": n_unique.get(c2, np.nan),
                        "eleme_nedeni": f"{c1} ile birebir aynı (yinelenen)", "yerine_tutulan": c1,
                    })

    elim_df = pd.DataFrame(elimination_rows).drop_duplicates(subset="column").sort_values("column")
    elim_df.to_csv(RESULTS_DIR / "step3_elimination_hypothesis.csv", index=False)

    reduced_cols = [c for c in raw.columns if c not in set(elim_df["column"]) or c in ("Variant_ID", "Label")]
    raw[reduced_cols].to_csv(RESULTS_DIR / "reduced_dataset_hypothesis.csv", index=False)

    safe_print(f"\nAdım 3 -- eleme HİPOTEZİ: {len(elim_df)} kolon aday (ham {len(raw_features)} kolondan), "
               f"azaltılmış hipotez seti {len(reduced_cols)} kolon (Variant_ID/Label dahil). "
               f"Bu KESİN bir karar DEĞİL -- Adım 13'te nested doğrulanacak.")


if __name__ == "__main__":
    main()
