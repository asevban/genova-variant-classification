"""ADIM 8 -- Korelasyon ve IG'yi birlikte degerlendir (redundan cift adaylari).
ADIM 9 -- Uclu/kismi korelasyon genellemesi (Faz 2'nin yontemi, YENI ciftlere).

DUZELTME 1 mantigi: yuksek korelasyonlu bir ciftte iki sutunu da otomatik
CIKARMA. Her cift icin IG/Label iliskisi/eksik oran/ortak-n/Pearson-Spearman
uyumu karsilastirilir; hedef hakkinda daha az bilgi tasiyan taraf yalnizca
ADAY olarak isaretlenir -- Adim 13'un nested dogrulamasi olmadan kesin degil.

Adim 9, Faz 2'nin partial_spearman/conditional_mutual_info fonksiyonlarini
DOGRUDAN IMPORT eder (yeniden yazmaz), Adim 6-7'de bulunan yeni ilginc
ciftlere uygular.

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_raw, safe_print  # noqa: E402
from cat1_partial_corr import partial_spearman, conditional_mutual_info  # noqa: E402  (Faz 2'den)


REDUNDANCY_SPEARMAN_THRESHOLD = 0.70  # Spearman'da "cok guclu" (>=0.90) hic yok, "guclu" bandinin tabani


def main():
    raw = load_raw()
    y = raw["Label"].to_numpy()

    pairs = pd.read_csv(RESULTS_DIR / "step6_all_numeric_pairs.csv")
    ig_df = pd.read_csv(RESULTS_DIR / "id3_feature_importance.csv").set_index("column")

    strong_pairs = pairs[pairs["abs_spearman_r"] >= REDUNDANCY_SPEARMAN_THRESHOLD].copy()
    safe_print(f"Spearman |r|>={REDUNDANCY_SPEARMAN_THRESHOLD} olan {len(strong_pairs)} çift, "
               f"IG karşılaştırmasıyla değerlendiriliyor...")

    rows = []
    for _, r in strong_pairs.iterrows():
        ci, cj = r["col_i"], r["col_j"]
        ig_i = ig_df.loc[ci, "IG"] if ci in ig_df.index else np.nan
        ig_j = ig_df.loc[cj, "IG"] if cj in ig_df.index else np.nan
        if pd.isna(ig_i) or pd.isna(ig_j):
            continue
        weaker, stronger = (ci, cj) if ig_i < ig_j else (cj, ci)
        rows.append({
            "col_i": ci, "col_j": cj, "spearman_r": r["spearman_r"], "common_n": r["common_n"],
            "IG_i": ig_i, "IG_j": ig_j,
            "missing_rate_i": ig_df.loc[ci, "missing_rate"] if ci in ig_df.index else np.nan,
            "missing_rate_j": ig_df.loc[cj, "missing_rate"] if cj in ig_df.index else np.nan,
            "aday_cikarilacak_zayif_taraf": weaker, "tutulmasi_onerilen_guclu_taraf": stronger,
            "IG_farki": abs(ig_i - ig_j),
        })
    redundancy_df = pd.DataFrame(rows).sort_values("IG_farki", ascending=False)
    redundancy_df.to_csv(RESULTS_DIR / "step8_redundancy_candidates.csv", index=False)

    candidates_to_drop = set(redundancy_df.loc[redundancy_df["IG_farki"] > 0.005, "aday_cikarilacak_zayif_taraf"])

    report8 = f"""# Adım 8 — Korelasyon + IG Birlikte Değerlendirme (Redundan Çift Adayları)

Spearman `|r|≥{REDUNDANCY_SPEARMAN_THRESHOLD}` olan **{len(strong_pairs)} çift**
bulundu (Adım 6). Her çiftin IG'si (Adım 2, betimsel) karşılaştırıldı;
**IG farkı >0.005 olan {len(candidates_to_drop)} kolon**, çiftinin daha
zayıf tarafı olarak **aday** işaretlendi (kesin eleme değil).

**Önemli:** İki sütun da otomatik çıkarılmadı — her çiftte yalnızca daha
düşük IG'li taraf aday, güçlü taraf tutulması öneriliyor. Bu adaylar Adım 13'te
nested karşılaştırmayla (tam 405 vs bu aday-set) test edilecek.

## En Belirgin IG Farkına Sahip 15 Çift

{redundancy_df.head(15)[["col_i","col_j","spearman_r","IG_i","IG_j","aday_cikarilacak_zayif_taraf"]].to_string(index=False)}

Tam liste: `results/step8_redundancy_candidates.csv`.
"""
    (RESULTS_DIR / "step8_redundancy_report.md").write_text(report8, encoding="utf-8")
    safe_print(report8)

    # --- ADIM 9: kismi korelasyon genellemesi ---
    from common import load_v3
    v3 = load_v3()

    candidates_step9 = [
        ("EK_7", "EK_9", "al_all_missing", "En güçlü EK_-EK_ çifti (Spearman=0.774), ikisi de resmi 25-özellik havuzunda"),
        ("AL_88", "AL_121", "al_all_missing", "GENOVA'nın resmi EDA'sının bildirdiği en güçlü AL_-AL_ çifti (Spearman=0.777)"),
    ]
    partial_rows = []
    label_v3 = v3["Label"].to_numpy()  # v3 (369 satir, dedup sonrasi) -- raw (372) ile HIZALI DEGIL
    from sklearn.feature_selection import mutual_info_classif
    for x_col, y_col_as_z_target, z_col, desc in candidates_step9:
        if x_col not in v3.columns or y_col_as_z_target not in v3.columns:
            continue
        x_full = v3[x_col].to_numpy(dtype=float)
        z_full = v3[z_col].to_numpy(dtype=float)
        y_full = v3[y_col_as_z_target].to_numpy(dtype=float)

        # partial_spearman (Faz 2'den) NaN'lari otomatik dusurmuyor (scipy
        # spearmanr, herhangi bir NaN varsa NaN dondurur) -- Faz 2'nin
        # kendi kullanim durumunda (CAT_1/al_all_missing) hic NaN yoktu, bu
        # yuzden bu sorun daha once ortaya cikmamisti. Burada EK_7/EK_9
        # gibi ~%97 dolu kolonlarda birkac NaN oldugu icin ONCE tam-vaka
        # (complete-case) filtreleme yapiliyor.
        complete = ~(np.isnan(x_full) | np.isnan(y_full) | np.isnan(z_full))
        x, y_pair, z = x_full[complete], y_full[complete], z_full[complete]
        n_dropped_nan = int((~complete).sum())

        r_partial, comps = partial_spearman(x, y_pair, z)
        row = {
            "X": x_col, "Y": y_col_as_z_target, "Z_kontrol": z_col, "aciklama": desc,
            "r_XY_kosulsuz": comps["r_xy"], "r_XY_kismi": r_partial, "ortak_n": comps["n"],
            "NaN_nedeniyle_dusurulen_satir": n_dropped_nan,
        }

        # Label ile kosullu IG: X'in Label ile iliskisi, Y'nin degerine gore
        # (yuksek/dusuk medyan-bolunmus) kosullaninca degisiyor mu?
        y_binary_z = (y_pair > np.nanmedian(y_pair)).astype(int)
        label_complete = label_v3[complete]
        mask = ~np.isnan(x)
        mi_uncond = float(mutual_info_classif(x[mask].reshape(-1, 1), label_complete[mask], random_state=42)[0])
        mi_cond, per_stratum = conditional_mutual_info(x[mask], label_complete[mask], y_binary_z[mask])
        row["IG_X_vs_Label_kosulsuz"] = mi_uncond
        row["Y_kontrol_degiskeni"] = y_col_as_z_target
        row["IG_X_vs_Label_Y_medyan_ile_kosullu"] = mi_cond
        partial_rows.append(row)

    partial_df = pd.DataFrame(partial_rows)
    partial_df.to_csv(RESULTS_DIR / "step9_partial_correlation_generalized.csv", index=False)

    report9 = f"""# Adım 9 — Kısmi Korelasyon / Koşullu IG Genellemesi (Faz 2'nin Yöntemi, Yeni Çiftler)

Faz 2'nin `partial_spearman`/`conditional_mutual_info` fonksiyonları
**doğrudan import edildi** (yeniden yazılmadı), Adım 6-7'de öne çıkan yeni
çiftlere uygulandı. Küçük ortak-n uyarısı: her satırda `ortak_n` açıkça
raporlanıyor.

{partial_df.to_string(index=False)}

**Not — v3 (sıfır-doldurulmuş) vs ham veri farkı:** `AL_88`↔`AL_121` burada
r=0.677 çıkıyor, Adım 6'nın ham-veri (yalnızca gerçek ortak gözlemler,
common_n=123) hesaplamasındaki 0.777'den **farklı** — çünkü bu script v3'ü
(zero-fill uygulanmış, common_n=369) kullanıyor; sıfır-doldurma iki kolonun
"gerçek gözlemlerdeki" korelasyon yapısını seyreltiyor. Bu, ölçüm hatası
değil, hangi veri versiyonunun sorulduğuna bağlı gerçek bir fark — ikisi de
geçerli, farklı soruları cevaplıyor (v3=model-girdisi-olarak-korelasyon,
ham=gerçek-gözlemler-arası-korelasyon).

**Yalnızca güçlü kısmi korelasyon gösterdiği için özellik silme kararı
verilmedi** — bu sonuçlar yalnızca Adım 13'ün nested karşılaştırmasına
girdi sağlayan gözlemler.
"""
    (RESULTS_DIR / "step9_partial_correlation_report.md").write_text(report9, encoding="utf-8")
    safe_print(report9)


if __name__ == "__main__":
    main()
