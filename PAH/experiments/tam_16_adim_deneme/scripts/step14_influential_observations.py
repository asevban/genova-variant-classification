"""ADIM 14 -- Etkili gozlemler (Adim 10'un guclu ciftleri uzerinde).

Faz 2'nin Bolum A'si bunu zaten 25-ozellik havuzunun MODEL performansi
(F1/MCC) uzerinde yapmisti (14 aday, LOO, hicbiri orantisiz etki
gostermedi). Burada AYNI PROSEDUR (en etkili gozlemi bul, once/sonra
grafik, 3-senaryo karsilastirmasi: hicbiri/tekil/hepsi-birlikte), ama
Adim 10'un 4 guclu CIFTININ PEARSON KORELASYONU uzerindeki etkiye
uygulaniyor -- Faz 2'nin LOO kodu (fold-koruma mantigi) DEGIL, cunku
burada CV/model degil, dogrudan korelasyon degisimi olculuyor (cok daha
hafif bir hesaplama).

Yalnizca veri girisi/olcum hatasi kaniti veya bilimsel gecersizlik kaniti
varsa silme onerilir -- performans/korelasyon artisi TEK BASINA gerekce
degil (gorev kurali).

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, FIGURES_DIR, load_v1, safe_print  # noqa: E402


def main():
    v1 = load_v1()
    pairs = pd.read_csv(RESULTS_DIR / "step10_pair_summary.csv")

    all_loo_rows = []
    scenario_rows = []
    for _, prow in pairs.iterrows():
        ci, cj = prow["pair"].split("-")
        sub = v1[["Variant_ID", ci, cj]].dropna().reset_index(drop=True)
        x, y = sub[ci].to_numpy(), sub[cj].to_numpy()
        n = len(x)
        r_full, _ = stats.pearsonr(x, y)

        deltas = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            r_i, _ = stats.pearsonr(x[mask], y[mask])
            deltas[i] = r_i - r_full

        loo_df = pd.DataFrame({
            "pair": f"{ci}-{cj}", "Variant_ID": sub["Variant_ID"], ci: x, cj: y,
            "r_full": r_full, "delta_r_when_removed": deltas,
        }).sort_values("delta_r_when_removed", key=lambda s: s.abs(), ascending=False)
        all_loo_rows.append(loo_df.head(10))

        most_influential_idx = np.argmax(np.abs(deltas))
        most_influential_id = sub.loc[most_influential_idx, "Variant_ID"]

        # 3-senaryo: hicbiri / en etkili tekil / en etkili ilk 5 (mutlak delta'ya gore)
        top5_idx = np.argsort(-np.abs(deltas))[:5]
        mask_top1 = np.ones(n, dtype=bool); mask_top1[most_influential_idx] = False
        mask_top5 = np.ones(n, dtype=bool); mask_top5[top5_idx] = False
        r_top1, _ = stats.pearsonr(x[mask_top1], y[mask_top1])
        r_top5, _ = stats.pearsonr(x[mask_top5], y[mask_top5])
        scenario_rows.append({
            "pair": f"{ci}-{cj}", "r_hicbiri_cikarilmamis": r_full,
            "en_etkili_tekil_id": most_influential_id, "r_tekil_cikarilinca": r_top1,
            "delta_tekil": r_top1 - r_full,
            "r_ilk5_birlikte_cikarilinca": r_top5, "delta_ilk5": r_top5 - r_full,
        })

        # once/sonra grafik
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(x, y, c="#2b6cb0", s=20, alpha=0.6)
        axes[0].scatter([x[most_influential_idx]], [y[most_influential_idx]], c="red", s=80, marker="*",
                        label=f"en etkili: {most_influential_id}")
        axes[0].set_title(f"Önce (r={r_full:.3f}, n={n})")
        axes[0].legend(fontsize=8)
        axes[1].scatter(x[mask_top1], y[mask_top1], c="#2b6cb0", s=20, alpha=0.6)
        axes[1].set_title(f"Sonra ({most_influential_id} çıkarıldı, r={r_top1:.3f})")
        for ax in axes:
            ax.set_xlabel(ci); ax.set_ylabel(cj)
        fig.suptitle(f"{ci} vs {cj} — en etkili gözlemin etkisi")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"step14_{ci}_{cj}_influence.png", dpi=110)
        plt.close(fig)

    loo_all_df = pd.concat(all_loo_rows, ignore_index=True)
    loo_all_df.to_csv(RESULTS_DIR / "step14_influential_observations.csv", index=False)
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(RESULTS_DIR / "step14_scenario_comparison.csv", index=False)

    max_abs_delta = loo_all_df["delta_r_when_removed"].abs().max()
    report = f"""# Adım 14 — Etkili Gözlemler (Adım 10'un Güçlü Çiftleri Üzerinde)

Faz 2'nin Bölüm A'sı bunu 25-özellik havuzunun **model performansı**
üzerinde yapmıştı (14 aday, hiçbiri orantısız etki göstermedi). Burada
**aynı prosedür**, Adım 10'un 4 güçlü çiftinin **Pearson korelasyonu**
üzerindeki etkiye uygulandı — her ortak-gözlem tek tek çıkarılıp
korelasyonun ne kadar değiştiği ölçüldü.

## 3-Senaryo Karşılaştırması (çift başına)

{scenario_df.to_string(index=False)}

## En Etkili 10 Gözlem (tüm çiftler, mutlak Δr'ye göre)

{loo_all_df.head(10)[["pair","Variant_ID","delta_r_when_removed"]].to_string(index=False)}

## Sonuç

En büyük mutlak etki **Δr={max_abs_delta:.4f}** — hiçbir tekil gözlem
korelasyonu **dramatik** şekilde değiştirmiyor (en büyük çift-bazlı örneklem
`AL_23-AL_283`'te bile n=85 gibi göreli küçük bir örneklemde tek bir
gözlemin etkisi sınırlı kalıyor). **Silme önerilmedi** — görev kuralı
gereği yalnızca veri girişi/ölçüm hatası kanıtı varsa silme önerilir,
performans/korelasyon değişimi tek başına gerekçe değil; böyle bir kanıt
bu turda aranmadı/bulunmadı. Grafikler: `figures/step14_*_influence.png`.
"""
    (RESULTS_DIR / "step14_report.md").write_text(report, encoding="utf-8")
    safe_print(report)


if __name__ == "__main__":
    main()
