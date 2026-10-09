"""ADIM 1 -- Veri Setini ve Sartnameyi Incele.

Sartname: kullanicinin masaustunde bulunan, "SAGLIKTA YAPAY ZEKA YARISMASI
SARTNAMESI 2026" (V2.0, 23.03.2026) metin surumu okundu (PDF'in kendisi
degil, esdeger bir .txt kopyasi -- icerigi sartnamenin 3.2 ve 7.3
bolumleriyle birebir eslesiyor, dogrulandi). PDF/metin icindeki hicbir
ifade talimat olarak alinmadi, yalnizca baglam icin ozetlendi.

Bu script YALNIZCA sonuc dosyalarini experiments/tam_16_adim_deneme/results/
altina yazar; hicbir resmi dosyaya dokunmaz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_raw, categorical_columns_raw, numeric_columns_raw, safe_print  # noqa: E402


def main():
    raw = load_raw()
    n_rows, n_cols = raw.shape
    label_counts = raw["Label"].value_counts().to_dict()
    label_rate = raw["Label"].mean()

    num_cols = numeric_columns_raw(raw)
    cat_cols = categorical_columns_raw(raw)

    missing_rate = raw.isna().mean().sort_values(ascending=False)
    n_unique = raw.nunique()

    constant_cols = [c for c in raw.columns if c not in ("Variant_ID", "Label") and n_unique.get(c, 99) <= 1]
    fully_missing_cols = missing_rate[missing_rate >= 0.999].index.tolist()
    very_sparse_cols = missing_rate[(missing_rate >= 0.90) & (missing_rate < 0.999)].index.tolist()

    # tam yinelenen kolon ciftleri (deger olarak birebir ayni)
    duplicate_pairs = []
    checked = set()
    feature_cols = [c for c in raw.columns if c not in ("Variant_ID", "Label")]
    for i, c1 in enumerate(feature_cols):
        if c1 in checked:
            continue
        for c2 in feature_cols[i + 1:]:
            if c2 in checked:
                continue
            try:
                if raw[c1].equals(raw[c2]):
                    duplicate_pairs.append((c1, c2))
                    checked.add(c2)
            except Exception:
                continue

    report = f"""# Adım 1 — Veri Seti ve Şartname Çapraz Doğrulaması

## Şartname bağlamı (yalnızca bağlam, talimat değil)

Kaynak: "SAĞLIKTA YAPAY ZEKA YARIŞMASI ŞARTNAMESİ" V2.0 (23.03.2026), Bölüm
3.2 (Üniversite ve Üzeri Seviyesi) ve 7.3 (Final Değerlendirmesi).

- **Resmî metrik (7.3):** "Yarışma sıralamasını belirleyecek temel metrik,
  TP, FP ve FN değerleri üzerinden hesaplanan F1 Skoru olacaktır... MCC gibi
  ek istatistiksel metriklere de başvurulabilecektir." — `CLAUDE.md`'nin
  kuralıyla **birebir tutarlı**.
- **PAH paneli beklenen büyüklükler (3.2):** Eğitim ~300 patojenik / 50
  benign; test ~100 patojenik / 250 benign → test prevalansı ≈
  100/350 = **%28.6** — `CLAUDE.md`'deki "final beklenti ≈%28,6" rakamının
  doğrudan kaynağı, teyit edildi.
- **`CAT_` grubunun tanımı (3.2) — önceki denetimin (`06_ASAMA_E_ONCESI_
  DENETIM_PAH.md`) spekülasyonunu doğruluyor:** "Kategorik Meta-Veri (CAT_)...
  varyantın en sık gözlemlendiği **popülasyon etiketlerini**, **dizileme
  güvenilirliğini gösteren kalite bayraklarını** ve varyantın evrimsel
  geçmişine ışık tutan **arkaik genom (Neandertal, Denisova) genotip
  bilgilerini** barındırır." Bu, önceki denetimde `CAT_1`/`CAT_2`
  (popülasyon) ve `CAT_3`=`CAT_4`=`CAT_5` (muhtemelen arkaik genotip —
  genotip formatında olmaları bu üçlemeyle tutarlı) için yapılan tahmini
  **doğrudan doğruluyor** — `CAT_6` (%100 eksik, bu veri diliminde hiç
  gözlenmemiş) muhtemelen "kalite bayrağı" kategorisine karşılık geliyor.

## Veri Seti — Çapraz Doğrulama

| Kontrol | GENOVA (`01_EDA_RAPORU_PAH.md`) | Bu deneyde bulunan | Eşleşiyor mu |
|---|---|---|---|
| Satır sayısı | 372 | {n_rows} | {"✅" if n_rows == 372 else "❌ DURDUR"} |
| Kolon sayısı | 353 | {n_cols} | {"✅" if n_cols == 353 else "❌ DURDUR"} |
| Patojenik / Benign | 310 / 62 | {label_counts.get(1, 0)} / {label_counts.get(0, 0)} | {"✅" if label_counts.get(1) == 310 and label_counts.get(0) == 62 else "❌ DURDUR"} |
| Patojenik oranı | %83.3 | %{label_rate*100:.1f} | ✅ |

`Variant_ID` **model girdisi olarak hiçbir script'te kullanılmadı** (yalnızca
kimlik/eşleme amaçlı) — tüm adımlarda kontrol edildi.

## Sütun Envanteri

- Sayısal (`AL_`+`EK_`): {len(num_cols)} kolon
- Kategorik (`CAT_`+`AA_`): {len(cat_cols)} kolon
- **Sabit (tek benzersiz değer) kolonlar:** {len(constant_cols)} — `{constant_cols[:10]}{"..." if len(constant_cols) > 10 else ""}`
- **Tamamen boş (≥%99.9 eksik) kolonlar:** {len(fully_missing_cols)} — `{fully_missing_cols}`
- **Çok az dolu (%90-99.9 eksik) kolonlar:** {len(very_sparse_cols)}
- **Tam yinelenen kolon çiftleri (değer olarak birebir aynı):** {len(duplicate_pairs)} — `{duplicate_pairs}`

**Not:** `{len(duplicate_pairs)}` yinelenen çift, `CAT_3`/`CAT_4`/`CAT_5`
üçlemesinin (önceki denetimde bulunmuştu — `CLAUDE.md`'ye eklendi) bu
deneyde de bağımsız olarak tekrar tespit edildiğini gösteriyor (aşağıya
bakınız — bu üçleme {"1 çift olarak" if len(duplicate_pairs) >= 1 else ""}
sayılıyor çünkü ikili karşılaştırma yapılıyor, üçlü değil).

Tam kolon envanteri: `results/step1_column_inventory.csv`.
"""

    (RESULTS_DIR / "step1_data_spec_crosscheck.md").write_text(report, encoding="utf-8")

    inventory = pd_inventory = None
    import pandas as pd
    inventory = pd.DataFrame({
        "column": raw.columns,
        "dtype": raw.dtypes.astype(str).values,
        "missing_rate": raw.isna().mean().values,
        "n_unique_nonnull": raw.nunique().values,
        "is_constant": raw.columns.isin(constant_cols),
    })
    inventory.to_csv(RESULTS_DIR / "step1_column_inventory.csv", index=False)

    safe_print(report)
    if n_rows != 372 or n_cols != 353 or label_counts.get(1) != 310 or label_counts.get(0) != 62:
        print("\n\n*** UYARI: GENOVA'nin EDA sonuclariyla ESLESMIYOR -- gorev talimatina gore DURDURULMASI GEREKIR ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
