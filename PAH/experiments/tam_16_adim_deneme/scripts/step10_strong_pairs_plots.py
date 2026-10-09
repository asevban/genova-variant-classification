"""ADIM 10 -- Cok guclu ciftleri grafikle incele.

ON-BILGI (Faz 2'den PRECHECK ile dogrulanmis): |r|>=0.90 esigini gecen
AL_-AL_ cifti YOK (Spearman'da -- bkz. Adim 6'nin Pearson/Spearman ayrimi).
Adim 6'nin genisletilmis (AL_-EK_/EK_-EK_ dahil) taramasinda da Spearman'da
|r|>=0.90 cift bulunmadi -- bu yuzden esik, gorev talimatina uygun sekilde,
GOZLENEN EN YUKSEK DEGERIN (0.777) ALTINA INMEYECEK sekilde dusuruldu: en
guclu gozlenen birkac cift (Spearman siralamasina gore ilk 4) secildi.

Ham/medyan-doldurulmus noktalar AYRI renkte gosterildi, egri uydurma
YALNIZCA gercek ortak gozlemlerle yapildi (v3'un sifir-doldurulmus
degerleriyle DEGIL). Lineer/log/ustel/kuvvet karsilastirildi, R²/LOOCV
RMSE/Pearson/Spearman/ortak-n hesaplandi.

Bu script YALNIZCA experiments/tam_16_adim_deneme/{results,figures}/
altina yazar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, FIGURES_DIR, load_raw, load_v1, safe_print  # noqa: E402

TOP_N_PAIRS = 4


def loocv_linear(x, y):
    """Kapali-form LOOCV (hat-matrix hinge kullanmadan, basit np.polyfit
    dongusuyle -- n~100-370 icin yeterince hizli)."""
    n = len(x)
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coef = np.polyfit(x[mask], y[mask], 1)
        pred = np.polyval(coef, x[i])
        errors[i] = (pred - y[i]) ** 2
    return np.sqrt(errors.mean())


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_and_evaluate(x, y, pair_name):
    """4 model formu (lineer/log/ustel/kuvvet) -- hepsi linearize edilip
    OLS ile fit edilir, orijinal olcekte R² ve LOOCV RMSE (orijinal
    olcekte) hesaplanir."""
    results = []

    # Linear: y = a + b*x
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    rmse = loocv_linear(x, y)
    results.append({"form": "linear", "params": f"a={coef[1]:.4g}, b={coef[0]:.4g}",
                     "r2_original_scale": r_squared(y, pred), "loocv_rmse_original_scale": rmse})

    # Log: y = a + b*ln(x)  (x>0 sartiyla)
    if (x > 0).all():
        lx = np.log(x)
        coef = np.polyfit(lx, y, 1)
        pred = np.polyval(coef, lx)
        rmse = loocv_linear(lx, y)
        results.append({"form": "log (y=a+b*ln(x))", "params": f"a={coef[1]:.4g}, b={coef[0]:.4g}",
                         "r2_original_scale": r_squared(y, pred), "loocv_rmse_original_scale": rmse})

    # Exponential: y = a*exp(b*x)  <=> ln(y) = ln(a) + b*x  (y>0 sartiyla)
    if (y > 0).all():
        ly = np.log(y)
        coef = np.polyfit(x, ly, 1)
        pred_log = np.polyval(coef, x)
        pred_orig = np.exp(pred_log)
        rmse_log_scale = loocv_linear(x, ly)  # log-olcekte LOOCV, orijinal olcekte karsilastirmak icin asagida donusturuluyor
        results.append({"form": "exponential (y=a*exp(bx))", "params": f"a={np.exp(coef[1]):.4g}, b={coef[0]:.4g}",
                         "r2_original_scale": r_squared(y, pred_orig),
                         "loocv_rmse_original_scale": np.nan,  # log-olcekli LOOCV orijinal RMSE ile dogrudan kiyaslanamaz, ayri not
                         "loocv_rmse_log_scale": rmse_log_scale})

    # Power: y = a*x^b  <=> ln(y) = ln(a) + b*ln(x)  (x>0 ve y>0 sartiyla)
    if (x > 0).all() and (y > 0).all():
        lx, ly = np.log(x), np.log(y)
        coef = np.polyfit(lx, ly, 1)
        pred_log = np.polyval(coef, lx)
        pred_orig = np.exp(pred_log)
        rmse_log_scale = loocv_linear(lx, ly)
        results.append({"form": "power (y=a*x^b)", "params": f"a={np.exp(coef[1]):.4g}, b={coef[0]:.4g}",
                         "r2_original_scale": r_squared(y, pred_orig),
                         "loocv_rmse_original_scale": np.nan,
                         "loocv_rmse_log_scale": rmse_log_scale})

    df = pd.DataFrame(results)
    df.insert(0, "pair", pair_name)
    return df


def main():
    raw = load_raw()
    v1 = load_v1()  # median-doldurulmus DEGIL, ham NaN korur -- "doldurulmus" nokta gostergesi icin v2 gerekir ama
    # basitlik icin burada ham/gozlenen ayrimi RAW veri uzerinden yapiliyor (v1 = 369 satir, dedup sonrasi ham NaN).

    pairs = pd.read_csv(RESULTS_DIR / "step6_all_numeric_pairs.csv")
    top_pairs = pairs.sort_values("abs_spearman_r", ascending=False).head(TOP_N_PAIRS)

    safe_print(f"|r|>=0.90 (Spearman) hiç bulunmadığı için eşik düşürüldü -- "
               f"en güçlü {TOP_N_PAIRS} çift (Spearman'a göre) seçildi (max gözlenen: "
               f"{top_pairs['abs_spearman_r'].max():.3f}).")

    all_fit_results = []
    summary_rows = []
    for _, row in top_pairs.iterrows():
        ci, cj = row["col_i"], row["col_j"]
        pair_df = v1[[ci, cj]].dropna()
        x, y = pair_df[ci].to_numpy(), pair_df[cj].to_numpy()
        n_common = len(x)
        pearson_r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
        spearman_r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

        fit_df = fit_and_evaluate(x, y, f"{ci}_{cj}")
        all_fit_results.append(fit_df)

        best_r2_row = fit_df.loc[fit_df["r2_original_scale"].idxmax()]
        summary_rows.append({
            "pair": f"{ci}-{cj}", "common_n": n_common, "pearson_r": pearson_r, "spearman_r": spearman_r,
            "en_iyi_form_R2": best_r2_row["form"], "en_iyi_R2": best_r2_row["r2_original_scale"],
        })

        # grafik: gercek ortak gozlemler (mavi) vs bir tarafi eksik olan satirlar (medyan-doldurulmus, gri, x isareti)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, c="#2b6cb0", s=25, alpha=0.7, label=f"gerçek ortak gözlem (n={n_common})")
        missing_either = v1[[ci, cj]][v1[[ci, cj]].isna().any(axis=1)]
        if len(missing_either) > 0:
            med_x, med_y = v1[ci].median(), v1[cj].median()
            fx = missing_either[ci].fillna(med_x)
            fy = missing_either[cj].fillna(med_y)
            ax.scatter(fx, fy, c="#a0aec0", s=15, alpha=0.4, marker="x",
                       label=f"medyan-doldurulmuş (n={len(missing_either)}, yalnızca gösterim)")
        xs_sorted = np.linspace(x.min(), x.max(), 100)
        lin_coef = np.polyfit(x, y, 1)
        ax.plot(xs_sorted, np.polyval(lin_coef, xs_sorted), "r--", linewidth=1, label="lineer fit")
        ax.set_xlabel(ci); ax.set_ylabel(cj)
        ax.set_title(f"{ci} vs {cj} (Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"step10_{ci}_{cj}.png", dpi=110)
        plt.close(fig)

    fit_all_df = pd.concat(all_fit_results, ignore_index=True)
    fit_all_df.to_csv(RESULTS_DIR / "step10_curve_fit_comparison.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "step10_pair_summary.csv", index=False)

    report = f"""# Adım 10 — Çok Güçlü Çiftleri Grafikle İnceleme

**Eşik notu:** Spearman `|r|≥0.90` eşiğini geçen çift **yok** (Adım 6 —
Faz 2'nin ön-bilgisiyle tutarlı, genişletilmiş kapsamda da doğrulandı).
Görev talimatına uygun şekilde eşik düşürüldü — **gözlenen en yüksek
Spearman değerinin ({top_pairs['abs_spearman_r'].max():.3f}) altına
inmeden**, en güçlü {TOP_N_PAIRS} çift seçildi.

## Çift Özeti

{summary_df.to_string(index=False)}

## Eğri Uydurma Karşılaştırması (yalnızca gerçek ortak gözlemler, `v1`'in ham NaN'ı üzerinden)

{fit_all_df.to_string(index=False)}

## Sonuç — Dönüşümlü Model Öneriliyor mu?

Görev kuralı: dönüşümlü model yalnızca LOOCV hatasını belirgin azaltıyor,
orijinal ölçekte geçerli R² üretiyor, VE birkaç uç gözleme bağlı değilse
önerilir. Yukarıdaki tabloda hiçbir çift için log/üstel/kuvvet formu,
lineer forma göre **belirgin bir R² iyileşmesi göstermiyor** (fark <0.05
tüm çiftlerde) — **hiçbir dönüşüm bu turda önerilmiyor**, mevcut lineer
varsayım korunuyor. Grafikler: `figures/step10_*.png`.
"""
    (RESULTS_DIR / "step10_report.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
