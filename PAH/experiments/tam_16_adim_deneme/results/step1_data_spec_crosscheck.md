# Adım 1 — Veri Seti ve Şartname Çapraz Doğrulaması

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
| Satır sayısı | 372 | 372 | ✅ |
| Kolon sayısı | 353 | 353 | ✅ |
| Patojenik / Benign | 310 / 62 | 310 / 62 | ✅ |
| Patojenik oranı | %83.3 | %83.3 | ✅ |

`Variant_ID` **model girdisi olarak hiçbir script'te kullanılmadı** (yalnızca
kimlik/eşleme amaçlı) — tüm adımlarda kontrol edildi.

## Sütun Envanteri

- Sayısal (`AL_`+`EK_`): 343 kolon
- Kategorik (`CAT_`+`AA_`): 8 kolon
- **Sabit (tek benzersiz değer) kolonlar:** 91 — `['AL_80', 'AL_101', 'AL_104', 'AL_107', 'AL_110', 'AL_113', 'AL_116', 'AL_119', 'AL_122', 'AL_125']...`
- **Tamamen boş (≥%99.9 eksik) kolonlar:** 1 — `['CAT_6']`
- **Çok az dolu (%90-99.9 eksik) kolonlar:** 18
- **Tam yinelenen kolon çiftleri (değer olarak birebir aynı):** 81 — `[('CAT_3', 'CAT_4'), ('CAT_3', 'CAT_5'), ('AL_101', 'AL_104'), ('AL_101', 'AL_107'), ('AL_101', 'AL_110'), ('AL_101', 'AL_113'), ('AL_101', 'AL_116'), ('AL_101', 'AL_119'), ('AL_101', 'AL_122'), ('AL_101', 'AL_125'), ('AL_101', 'AL_128'), ('AL_101', 'AL_131'), ('AL_101', 'AL_134'), ('AL_101', 'AL_137'), ('AL_101', 'AL_140'), ('AL_101', 'AL_143'), ('AL_101', 'AL_146'), ('AL_101', 'AL_149'), ('AL_101', 'AL_152'), ('AL_101', 'AL_155'), ('AL_101', 'AL_158'), ('AL_101', 'AL_161'), ('AL_101', 'AL_164'), ('AL_101', 'AL_167'), ('AL_101', 'AL_170'), ('AL_101', 'AL_173'), ('AL_101', 'AL_176'), ('AL_101', 'AL_179'), ('AL_101', 'AL_182'), ('AL_191', 'AL_195'), ('AL_191', 'AL_197'), ('AL_191', 'AL_200'), ('AL_191', 'AL_204'), ('AL_191', 'AL_208'), ('AL_191', 'AL_212'), ('AL_191', 'AL_220'), ('AL_191', 'AL_227'), ('AL_191', 'AL_231'), ('AL_191', 'AL_233'), ('AL_191', 'AL_236'), ('AL_191', 'AL_240'), ('AL_191', 'AL_244'), ('AL_191', 'AL_248'), ('AL_191', 'AL_256'), ('AL_191', 'AL_263'), ('AL_191', 'AL_267'), ('AL_191', 'AL_269'), ('AL_191', 'AL_272'), ('AL_191', 'AL_276'), ('AL_191', 'AL_280'), ('AL_191', 'AL_284'), ('AL_191', 'AL_292'), ('AL_192', 'AL_196'), ('AL_192', 'AL_201'), ('AL_192', 'AL_205'), ('AL_192', 'AL_209'), ('AL_192', 'AL_213'), ('AL_192', 'AL_221'), ('AL_192', 'AL_222'), ('AL_192', 'AL_228'), ('AL_192', 'AL_232'), ('AL_192', 'AL_237'), ('AL_192', 'AL_245'), ('AL_192', 'AL_249'), ('AL_192', 'AL_257'), ('AL_192', 'AL_264'), ('AL_192', 'AL_268'), ('AL_192', 'AL_273'), ('AL_192', 'AL_281'), ('AL_192', 'AL_285'), ('AL_192', 'AL_293'), ('AL_299', 'AL_303'), ('AL_299', 'AL_307'), ('AL_299', 'AL_309'), ('AL_299', 'AL_312'), ('AL_299', 'AL_316'), ('AL_299', 'AL_320'), ('AL_299', 'AL_324'), ('AL_299', 'AL_332'), ('AL_304', 'AL_308'), ('AL_304', 'AL_321')]`

**Not:** `81` yinelenen çift, `CAT_3`/`CAT_4`/`CAT_5`
üçlemesinin (önceki denetimde bulunmuştu — `CLAUDE.md`'ye eklendi) bu
deneyde de bağımsız olarak tekrar tespit edildiğini gösteriyor (aşağıya
bakınız — bu üçleme 1 çift olarak
sayılıyor çünkü ikili karşılaştırma yapılıyor, üçlü değil).

Tam kolon envanteri: `results/step1_column_inventory.csv`.
