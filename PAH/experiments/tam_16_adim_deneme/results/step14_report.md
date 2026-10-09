# Adım 14 — Etkili Gözlemler (Adım 10'un Güçlü Çiftleri Üzerinde)

Faz 2'nin Bölüm A'sı bunu 25-özellik havuzunun **model performansı**
üzerinde yapmıştı (14 aday, hiçbiri orantısız etki göstermedi). Burada
**aynı prosedür**, Adım 10'un 4 güçlü çiftinin **Pearson korelasyonu**
üzerindeki etkiye uygulandı — her ortak-gözlem tek tek çıkarılıp
korelasyonun ne kadar değiştiği ölçüldü.

## 3-Senaryo Karşılaştırması (çift başına)

        pair  r_hicbiri_cikarilmamis en_etkili_tekil_id  r_tekil_cikarilinca  delta_tekil  r_ilk5_birlikte_cikarilinca  delta_ilk5
AL_88-AL_121                0.962621         VAR_002604             0.896526    -0.066095                     0.928840   -0.033781
   EK_7-EK_9                0.842766         VAR_002899             0.847340     0.004574                     0.861663    0.018897
AL_23-AL_283                0.981941         VAR_002604             0.938291    -0.043651                     0.910070   -0.071871
 AL_7-AL_103                0.990019         VAR_002604             0.975003    -0.015015                     0.969231   -0.020788

## En Etkili 10 Gözlem (tüm çiftler, mutlak Δr'ye göre)

        pair Variant_ID  delta_r_when_removed
AL_88-AL_121 VAR_002604             -0.066095
AL_88-AL_121 VAR_002765              0.011295
AL_88-AL_121 VAR_003008              0.002451
AL_88-AL_121 VAR_003151              0.002148
AL_88-AL_121 VAR_002790              0.001669
AL_88-AL_121 VAR_002826              0.001063
AL_88-AL_121 VAR_003162              0.001030
AL_88-AL_121 VAR_003202              0.000843
AL_88-AL_121 VAR_003120              0.000690
AL_88-AL_121 VAR_003191              0.000620

## Sonuç

En büyük mutlak etki **Δr=0.0661** — hiçbir tekil gözlem
korelasyonu **dramatik** şekilde değiştirmiyor (en büyük çift-bazlı örneklem
`AL_23-AL_283`'te bile n=85 gibi göreli küçük bir örneklemde tek bir
gözlemin etkisi sınırlı kalıyor). **Silme önerilmedi** — görev kuralı
gereği yalnızca veri girişi/ölçüm hatası kanıtı varsa silme önerilir,
performans/korelasyon değişimi tek başına gerekçe değil; böyle bir kanıt
bu turda aranmadı/bulunmadı. Grafikler: `figures/step14_*_influence.png`.
