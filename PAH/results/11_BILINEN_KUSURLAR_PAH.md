# PAH Paneli — Bilinen, Düzeltilmemiş Kusurlar

> F6 hazırlık denetiminde (bkz. `09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md`)
> tespit edilen, bilinçli olarak kapsam dışı bırakılmış kusurları
> belgeler. Final modelin doğruluğunu etkilemeyen kusurlar burada
> yaşar — etkileyen bir kusur bulunursa bu dosyaya değil, ilgili
> model/rapor dosyasına acil düzeltme olarak işlenir.

## `RankQuantileHarmonizer` — Tüm-`EK_`-Eksik Satırlarda Kusur

`src/genova/pah/transforms.py:71-85`'teki `RankQuantileHarmonizer`,
`transform()` sırasında bir kolonun maskesi (`out[col].notna()`) o
toplu işlemde tamamen boş kalırsa (yani işlenen satırların hepsinde o
`EK_` kolonu eksikse) `QuantileTransformer`'ın "0 sample" hatasına
karşı korumasız.

**Tetiklenme koşulu:** Gerçek veride (`v1.parquet`) bunu tetikleyecek
**4 satır** var — hem tüm-`AL_` hem tüm-`EK_`-sınırsız (`EK_1,2,7,8,9`)
kolonlarında eksik: `VAR_003238`, `VAR_002977`, `VAR_003234`,
`VAR_002796`. Bu satırlar tek başına (veya yalnızca kendi aralarında)
`pah_pipeline_transformed.joblib`'e (`serialize_pipelines.py`) verilirse
hata fırlatılır.

**Final modeli etkilemiyor:** `f0_final_model.py` (ve dolayısıyla
`predict.py`'nin yüklediği `final_model_bundle_v2.pkl`) yalnızca
`fold_versions.py::_v2_steps` (v2-hattı, sıfır-doldurmalı ağaç-modeli
hattı) kullanıyor — `RankQuantileHarmonizer` yalnızca v3/`pah_pipeline_
transformed.joblib`'de (Elastic-Net için tasarlanmış, resmi pipeline'da
kullanılmayan ölçeksiz hat) mevcut. `predict.py` ile üretilen hiçbir
tahmin bu koddan geçmiyor.

**Durum:** Düşük öncelik, düzeltilmedi — bilinçli bir kapsam dışı
bırakma. `test_serialize_pipelines_pah.py::test_transformed_pipeline_
transforms_row_with_all_al_columns_missing`'in yorumu bu dosyaya işaret
ediyor.
