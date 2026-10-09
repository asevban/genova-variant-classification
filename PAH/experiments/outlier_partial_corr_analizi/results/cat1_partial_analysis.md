# Bölüm B — `CAT_1` × `Label` Kısmi İlişki (`al_all_missing` Kontrol Edilerek)

> ⚠️ Bu analiz bağımsız, araştırma amaçlıdır. `03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md`
> veya başka hiçbir resmi rapora bu görevde **dokunulmadı** — sonuç ne olursa
> olsun, mevcut rapora eklenip eklenmeyeceğine ayrı bir onay turunda karar
> verilecek. Veri kaynağı: `data/processed/pah/v3.parquet` (yalnızca okundu).

## 1. Koşulsuz İlişki (referans, tazelendi)

- Spearman(`CAT_1`, `Label`) = **0.0552** (p=2.90e-01, n=369)
- Tek-değişkenli AUC(`CAT_1` → `Label`) = **0.5418**
- Koşulsuz Mutual Information I(`CAT_1`;`Label`) = **0.0000**

## 2. Kısmi Korelasyon (`al_all_missing` Kontrol Edilerek)

Standart ilk-derece kısmi korelasyon formülü: `r_xy.z = (r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))`

| Bileşen | Değer |
|---|---|
| r(`CAT_1`, `Label`) | 0.0552 |
| r(`CAT_1`, `al_all_missing`) | -0.6445 |
| r(`Label`, `al_all_missing`) | 0.2168 |
| **Ortak n** | **369** (tam veri seti, hiçbir NaN yok — bu üç kolon için satır kaybı yok) |
| **Kısmi Spearman r** | **0.2612** |

## 3. Kritik Uyarı — `al_all_missing=1` Alt-Grubunda `CAT_1` Sabit

| `al_all_missing` | n | `CAT_1` varyansı | `CAT_1` benzersiz değer sayısı |
|---|---|---|---|
| 0 | 280 | 0.011276 | 16 |
| 1 | 89 | 0.000000 | 1 |

**`al_all_missing=1` olan 89 satırın tamamında `CAT_1`=0.0 (sabit)** —
çünkü kodlayıcı eksik `CAT_1`'i 0.0'a eşliyor ve bu satırların tamamında
`CAT_1` zaten eksik (`03b`'nin Cramér's V=0.755 bulgusuyla aynı örtüşme).
**Bu, "küçük ortak-n" uyarısının somut hâli:** kısmi korelasyon/koşullu IG
hesaplamasına `al_all_missing=1` alt-grubu **hiçbir gerçek bilgiyle katkı
sağlamıyor** — o alt-grup içindeki herhangi bir istatistik (IG dahil)
**mekanik olarak** sıfıra sabitlenmiş durumda, yeni bir bulgu değil, kodlama
şemasının doğrudan bir sonucu. Kısmi korelasyon/koşullu IG'nin **fiilen
bilgi taşıyan kısmı yalnızca `al_all_missing=0` alt-grubundan (n=280)
geliyor.**

## 4. Koşullu Information Gain

| Ölçüm | Değer |
|---|---|
| Koşulsuz I(`CAT_1`;`Label`) | 0.0000 |
| I(`CAT_1`;`Label` \| `al_all_missing`=0), n=280 | 0.0381  |
| I(`CAT_1`;`Label` \| `al_all_missing`=1), n=89 | 0.0000 SABIT (varyans yok) -- MI mekanik olarak 0 |
| **Ağırlıklı ortalama (standart koşullu IG tanımı)** | **0.0289** |
| *(Yalnızca `al_all_missing`=0 alt-grubu, "gerçek" kısım)* | *0.0381* |

Standart ağırlıklı-ortalama tanımıyla koşullu IG (0.0289),
koşulsuz IG'den (0.0000) **artıyor** — ama bu ağırlıklı
ortalamanın **%24'i** (`al_all_missing=1`
alt-grubunun ağırlığı), yukarıda açıklanan **mekanik sıfır**'dan geliyor,
yani ağırlıklı ortalama asıl sinyali seyreltiyor/küçültüyor. Bu yüzden asıl
bilgilendirici karşılaştırma, koşulsuz IG'yi **yalnızca `al_all_missing=0`
alt-grubundaki** IG ile kıyaslamak: 0.0000 → 0.0381
(bu da bir **artış**).

## 5. Yorum (önceden belirlenmiş kurala göre, sonuca göre bükülmedi)

**Beklenmedik desen -- gorev metninin iki-yonlu on-kaydedilmis kuralinin disinda:** Koşulsuz IG **tam olarak 0** (KSG tahminleyicisi, agir bag-degenerasyonu nedeniyle -- CAT_1 369 satirda yalnizca 16 benzersiz deger tasiyor), ama al_all_missing=0 alt-grubunda (CAT_1'in gercekten gozlemlendigi, n=280) IG **0'dan 0.0381'e yukseliyor** (mutlak artis +0.0381). Bu, klasik bir **seyreltme (dilution) / Simpson-paradoksu-benzeri** desen: `al_all_missing=1` alt-grubu (n=89, CAT_1 sabit=0) havuzlanmis analize yalnizca gurultu katip, gercekte var olan iliskiyi (yalnizca CAT_1'in gozlemlendigi popuasyonda) sulandiriyor. **Bu, provenance-confound hipotezinden çok biyolojik-sinyal hipotezini hafifçe destekliyor** -- eger CAT_1'in tüm görünen sinyali yalnızca al_all_missing'in bir vekili olsaydı, al_all_missing'i sabitleyip CAT_1'in kendi başına gerçekten gözlemlendiği popülasyona odaklanınca ilişkinin GÜÇLENMESİ değil ZAYIFLAMASI/kaybolması beklenirdi. **'Hafifçe destekliyor' — 'kanıtlıyor' değil; küçük mutlak IG değerleri (<0.04) ve KSG tahminleyicisinin tie-degenerasyon riski nedeniyle temkinli yorumlanmalı.** F1 adversarial validation hâlâ asıl karar noktası.

**Kısmi korelasyon açısından:** Kısmi Spearman r=0.2612 (ortak
n=369), koşulsuz r=0.0552 ile karşılaştırıldığında
**büyüyor** (mutlak değerce 0.2612 > 0.0552) — bu da
yukarıdaki IG-tabanlı "seyreltme" yorumuyla **tutarlı**: `al_all_missing`'i kontrol
edince `CAT_1`↔`Label` ilişkisi zayıflamıyor, güçleniyor. İki bağımsız yöntem
(kısmi korelasyon + koşullu IG) aynı yöne işaret ediyor.

**Genel değerlendirme:** Bu iki ölçüm (kısmi korelasyon + koşullu IG) `03b`'nin
açık bıraktığı soruya kesin bir cevap **vermiyor** — ikisi de yalnızca F1
adversarial validation'a giden **ek bir veri noktası**. `al_all_missing=1`
alt-grubunun mekanik-sıfır doğası, bu tür koşullu istatistiklerin bu özel
değişken çifti için ne kadar dikkatli yorumlanması gerektiğinin de ayrı bir
metodolojik hatırlatıcısı.
