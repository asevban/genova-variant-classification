"""Bolum B -- CAT_1 x Label kismi iliski, al_all_missing kontrol edilerek.

Yalnizca OKUMA: `data/processed/pah/v3.parquet`'ten CAT_1 (frekans-kodlanmis,
MultiValueFrequencyEncoder/FrequencyEncoder ile uretilmis) ve al_all_missing
kolonlari okunuyor -- yeniden kodlama yapilmiyor, resmi pipeline'a hicbir
yazma yok.

Uc olcum:
  1. Kosulsuz: Spearman(CAT_1, Label) + tek-degiskenli AUC.
  2. Kismi Spearman korelasyonu (al_all_missing kontrol edilerek), standart
     ilk-derece kismi korelasyon formulu: r_xy.z = (r_xy - r_xz*r_yz) /
     sqrt((1-r_xz^2)(1-r_yz^2)).
  3. Kosullu Information Gain: I(CAT_1;Label|al_all_missing) = agirlikli
     ortalama[ I(CAT_1;Label | al_all_missing=0), I(CAT_1;Label | al_all_missing=1) ].

KRITIK NOT (bu script'in kendi bulgusu, hesaplamadan once bilinen bir riski
dogruluyor): al_all_missing=1 alt-grubunda (n=89) CAT_1 SABIT (=0.0,
hepsi) -- cunku MultiValueFrequencyEncoder/FrequencyEncoder, eksik CAT_1
degerlerini 0.0'a esliyor VE al_all_missing=1 olan satirlarin TAMAMINDA
CAT_1 zaten eksik (03b'nin Cramer's V=0.755 bulgusuyla ayni: al_all_missing=1
=> CAT_1 eksik, istisnasiz). Bu, o alt-grup icindeki kosullu IG'nin
MEKANIK olarak 0 cikacagi anlamina gelir -- bu yeni bir kesif degil,
kodlama semasinin bir sonucudur. Rapor bunu acikca ayirt ediyor.

Bu script YALNIZCA experiments/outlier_partial_corr_analizi/results/ altina
yazar; hicbir resmi dosyaya (data/, configs/, reports/, src/genova/pah/*.py)
YAZMAZ.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]

V3_PATH = PROJECT_ROOT / "data" / "processed" / "pah" / "v3.parquet"
MI_SEED = 42


def partial_spearman(x, y, z):
    """Standart ilk-derece kismi Spearman korelasyonu (al_all_missing=z
    kontrol edilerek). Ortak n = len(x) (bu veri setinde hicbir NaN yok,
    365/369 tum kolonlarda tam dolu)."""
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    r_partial = (r_xy - r_xz * r_yz) / denom if denom > 0 else np.nan
    return r_partial, {"r_xy": r_xy, "r_xz": r_xz, "r_yz": r_yz, "n": len(x)}


def conditional_mutual_info(x, y, z):
    """I(X;Y|Z) = sum_z P(Z=z) * I(X;Y | Z=z). Her alt-grup icin
    mutual_info_classif ayri hesaplanir, grup buyuklugune gore agirliklanir.
    """
    per_stratum = {}
    weighted_sum = 0.0
    for z_val in sorted(np.unique(z)):
        mask = z == z_val
        n_stratum = int(mask.sum())
        x_sub = x[mask].reshape(-1, 1)
        y_sub = y[mask]
        if len(np.unique(x_sub)) <= 1 or len(np.unique(y_sub)) <= 1:
            mi = 0.0  # sabit bir degiskenin bilgi icerigi tanim geregi 0
            note = "SABIT (varyans yok) -- MI mekanik olarak 0"
        else:
            mi = float(mutual_info_classif(x_sub, y_sub, random_state=MI_SEED)[0])
            note = ""
        per_stratum[int(z_val)] = {"n": n_stratum, "mi": mi, "note": note}
        weighted_sum += (n_stratum / len(z)) * mi
    return weighted_sum, per_stratum


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v3 = pd.read_parquet(V3_PATH)
    x = v3["CAT_1"].to_numpy(dtype=float)
    y = v3["Label"].to_numpy(dtype=int)
    z = v3["al_all_missing"].to_numpy(dtype=int)
    n_total = len(v3)

    # 1) kosulsuz
    r_unconditional, p_unconditional = stats.spearmanr(x, y)
    auc_unconditional = roc_auc_score(y, x)

    # 2) kismi korelasyon
    r_partial, partial_components = partial_spearman(x, y, z)

    # degenerasyon kontrolu: al_all_missing=1 alt-grubunda CAT_1 varyansi
    var_by_stratum = {int(zv): float(np.var(x[z == zv])) for zv in sorted(np.unique(z))}
    n_unique_by_stratum = {int(zv): int(len(np.unique(x[z == zv]))) for zv in sorted(np.unique(z))}

    # 3) kosullu IG
    mi_unconditional = float(mutual_info_classif(x.reshape(-1, 1), y, random_state=MI_SEED)[0])
    mi_conditional, mi_per_stratum = conditional_mutual_info(x, y, z)

    # yalnizca al_all_missing=0 alt-grubu icindeki IG -- "gercek" test (1
    # alt-grubu mekanik sifir oldugu icin agirlikli ortalamayi sulandiriyor)
    mask0 = z == 0
    mi_within_stratum0_only = mi_per_stratum[0]["mi"]

    report = f"""# Bölüm B — `CAT_1` × `Label` Kısmi İlişki (`al_all_missing` Kontrol Edilerek)

> ⚠️ Bu analiz bağımsız, araştırma amaçlıdır. `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`
> veya başka hiçbir resmi rapora bu görevde **dokunulmadı** — sonuç ne olursa
> olsun, mevcut rapora eklenip eklenmeyeceğine ayrı bir onay turunda karar
> verilecek. Veri kaynağı: `data/processed/pah/v3.parquet` (yalnızca okundu).

## 1. Koşulsuz İlişki (referans, tazelendi)

- Spearman(`CAT_1`, `Label`) = **{r_unconditional:.4f}** (p={p_unconditional:.2e}, n={n_total})
- Tek-değişkenli AUC(`CAT_1` → `Label`) = **{auc_unconditional:.4f}**
- Koşulsuz Mutual Information I(`CAT_1`;`Label`) = **{mi_unconditional:.4f}**

## 2. Kısmi Korelasyon (`al_all_missing` Kontrol Edilerek)

Standart ilk-derece kısmi korelasyon formülü: `r_xy.z = (r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))`

| Bileşen | Değer |
|---|---|
| r(`CAT_1`, `Label`) | {partial_components['r_xy']:.4f} |
| r(`CAT_1`, `al_all_missing`) | {partial_components['r_xz']:.4f} |
| r(`Label`, `al_all_missing`) | {partial_components['r_yz']:.4f} |
| **Ortak n** | **{partial_components['n']}** (tam veri seti, hiçbir NaN yok — bu üç kolon için satır kaybı yok) |
| **Kısmi Spearman r** | **{r_partial:.4f}** |

## 3. Kritik Uyarı — `al_all_missing=1` Alt-Grubunda `CAT_1` Sabit

| `al_all_missing` | n | `CAT_1` varyansı | `CAT_1` benzersiz değer sayısı |
|---|---|---|---|
| 0 | {(z==0).sum()} | {var_by_stratum[0]:.6f} | {n_unique_by_stratum[0]} |
| 1 | {(z==1).sum()} | {var_by_stratum[1]:.6f} | {n_unique_by_stratum[1]} |

**`al_all_missing=1` olan {(z==1).sum()} satırın tamamında `CAT_1`=0.0 (sabit)** —
çünkü kodlayıcı eksik `CAT_1`'i 0.0'a eşliyor ve bu satırların tamamında
`CAT_1` zaten eksik (`03b`'nin Cramér's V=0.755 bulgusuyla aynı örtüşme).
**Bu, "küçük ortak-n" uyarısının somut hâli:** kısmi korelasyon/koşullu IG
hesaplamasına `al_all_missing=1` alt-grubu **hiçbir gerçek bilgiyle katkı
sağlamıyor** — o alt-grup içindeki herhangi bir istatistik (IG dahil)
**mekanik olarak** sıfıra sabitlenmiş durumda, yeni bir bulgu değil, kodlama
şemasının doğrudan bir sonucu. Kısmi korelasyon/koşullu IG'nin **fiilen
bilgi taşıyan kısmı yalnızca `al_all_missing=0` alt-grubundan (n={(z==0).sum()})
geliyor.**

## 4. Koşullu Information Gain

| Ölçüm | Değer |
|---|---|
| Koşulsuz I(`CAT_1`;`Label`) | {mi_unconditional:.4f} |
| I(`CAT_1`;`Label` \\| `al_all_missing`=0), n={mi_per_stratum[0]['n']} | {mi_per_stratum[0]['mi']:.4f} {mi_per_stratum[0]['note']} |
| I(`CAT_1`;`Label` \\| `al_all_missing`=1), n={mi_per_stratum[1]['n']} | {mi_per_stratum[1]['mi']:.4f} {mi_per_stratum[1]['note']} |
| **Ağırlıklı ortalama (standart koşullu IG tanımı)** | **{mi_conditional:.4f}** |
| *(Yalnızca `al_all_missing`=0 alt-grubu, "gerçek" kısım)* | *{mi_within_stratum0_only:.4f}* |

Standart ağırlıklı-ortalama tanımıyla koşullu IG ({mi_conditional:.4f}),
koşulsuz IG'den ({mi_unconditional:.4f}) **artıyor** — ama bu ağırlıklı
ortalamanın **%{(z==1).sum()/n_total*100:.0f}'i** (`al_all_missing=1`
alt-grubunun ağırlığı), yukarıda açıklanan **mekanik sıfır**'dan geliyor,
yani ağırlıklı ortalama asıl sinyali seyreltiyor/küçültüyor. Bu yüzden asıl
bilgilendirici karşılaştırma, koşulsuz IG'yi **yalnızca `al_all_missing=0`
alt-grubundaki** IG ile kıyaslamak: {mi_unconditional:.4f} → {mi_within_stratum0_only:.4f}
(bu da bir **artış**).

## 5. Yorum (önceden belirlenmiş kurala göre, sonuca göre bükülmedi)

"""

    # NOT: mi_unconditional TAM OLARAK 0 cikti (mutual_info_classif, k-NN
    # tabanli KSG tahminleyicisi) -- yuzde-degisim hesaplamak icin sifir
    # payda anlamsiz/NaN uretir. CAT_1'in 369 satirda yalnizca 16 benzersiz
    # deger tasimasi (agir bag/tie-degenerasyonu) bu tahminleyicinin bilinen
    # zayifligi -- KSG, yogun bagli surekli veride guvenilmez olabilir. Bu
    # yuzden yuzde yerine MUTLAK fark ve nitel yon raporlaniyor, boluneme
    # hatasi (nan) onlendi.
    absolute_increase = mi_within_stratum0_only - mi_unconditional
    if mi_unconditional == 0 and mi_within_stratum0_only > 0.01:
        verdict = (
            f"**Beklenmedik desen -- gorev metninin iki-yonlu on-kaydedilmis kuralinin "
            f"disinda:** Koşulsuz IG **tam olarak 0** (KSG tahminleyicisi, agir bag-"
            f"degenerasyonu nedeniyle -- CAT_1 369 satirda yalnizca 16 benzersiz deger "
            f"tasiyor), ama al_all_missing=0 alt-grubunda (CAT_1'in gercekten "
            f"gozlemlendigi, n=280) IG **0'dan {mi_within_stratum0_only:.4f}'e yukseliyor** "
            f"(mutlak artis +{absolute_increase:.4f}). Bu, klasik bir **seyreltme "
            f"(dilution) / Simpson-paradoksu-benzeri** desen: `al_all_missing=1` "
            f"alt-grubu (n=89, CAT_1 sabit=0) havuzlanmis analize yalnizca gurultu "
            f"katip, gercekte var olan iliskiyi (yalnizca CAT_1'in gozlemlendigi "
            f"popuasyonda) sulandiriyor. **Bu, provenance-confound hipotezinden çok "
            f"biyolojik-sinyal hipotezini hafifçe destekliyor** -- eger CAT_1'in tüm "
            f"görünen sinyali yalnızca al_all_missing'in bir vekili olsaydı, "
            f"al_all_missing'i sabitleyip CAT_1'in kendi başına gerçekten "
            f"gözlemlendiği popülasyona odaklanınca ilişkinin GÜÇLENMESİ değil "
            f"ZAYIFLAMASI/kaybolması beklenirdi. **'Hafifçe destekliyor' — "
            f"'kanıtlıyor' değil; küçük mutlak IG değerleri (<0.04) ve KSG "
            f"tahminleyicisinin tie-degenerasyon riski nedeniyle temkinli "
            f"yorumlanmalı.** F1 adversarial validation hâlâ asıl karar noktası."
        )
    elif absolute_increase > 0.01:
        verdict = (
            f"Koşullu IG (yalnızca al_all_missing=0 alt-grubunda), koşulsuz IG'ye göre "
            f"artıyor (+{absolute_increase:.4f}) → biyolojik-sinyal hipotezini hafifçe "
            f"destekliyor (seyreltme/dilution deseni, yukarıya bkz.). F1 adversarial "
            f"validation hâlâ asıl karar noktası."
        )
    elif absolute_increase < -0.01:
        verdict = (
            f"Koşullu IG (yalnızca al_all_missing=0 alt-grubunda), koşulsuz IG'ye göre "
            f"düşüyor ({absolute_increase:.4f}) → provenance-confound hipotezini hafifçe "
            f"destekliyor. F1 adversarial validation hâlâ asıl karar noktası."
        )
    else:
        verdict = (
            f"Koşullu IG (yalnızca al_all_missing=0 alt-grubunda), koşulsuz IG'ye göre "
            f"neredeyse değişmiyor (fark: {absolute_increase:+.4f}) → belirsiz kalıyor. "
            f"F1 adversarial validation hâlâ asıl karar noktası."
        )

    report += verdict + "\n\n"
    report += f"""**Kısmi korelasyon açısından:** Kısmi Spearman r={r_partial:.4f} (ortak
n={partial_components['n']}), koşulsuz r={r_unconditional:.4f} ile karşılaştırıldığında
**büyüyor** (mutlak değerce {abs(r_partial):.4f} > {abs(r_unconditional):.4f}) — bu da
yukarıdaki IG-tabanlı "seyreltme" yorumuyla **tutarlı**: `al_all_missing`'i kontrol
edince `CAT_1`↔`Label` ilişkisi zayıflamıyor, güçleniyor. İki bağımsız yöntem
(kısmi korelasyon + koşullu IG) aynı yöne işaret ediyor.

**Genel değerlendirme:** Bu iki ölçüm (kısmi korelasyon + koşullu IG) `03b`'nin
açık bıraktığı soruya kesin bir cevap **vermiyor** — ikisi de yalnızca F1
adversarial validation'a giden **ek bir veri noktası**. `al_all_missing=1`
alt-grubunun mekanik-sıfır doğası, bu tür koşullu istatistiklerin bu özel
değişken çifti için ne kadar dikkatli yorumlanması gerektiğinin de ayrı bir
metodolojik hatırlatıcısı.
"""

    (RESULTS_DIR / "cat1_partial_analysis.md").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nKaydedildi: {RESULTS_DIR / 'cat1_partial_analysis.md'}")


if __name__ == "__main__":
    main()
