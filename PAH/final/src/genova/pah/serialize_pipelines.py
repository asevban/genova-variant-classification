"""Tamamlayıcı Deney Turu — Görev 7: Aşama B'nin fit/transform sınıflarını
tek bir sklearn `Pipeline` nesnesinde birleştirip `joblib` ile kaydeder.

İki ayrı pipeline üretilir:
  - `pah_pipeline_tree.joblib`       : v2 hattına karşılık gelir (ağaç
    modelleri için -- kategorikler native `category` dtype, ölçekleme yok).
  - `pah_pipeline_transformed.joblib`: v3'ün çekirdek dönüşümlerine karşılık
    gelir (log1p/logit/harmonizasyon + tam kategorik kodlama). ÖLÇEKLEME
    İÇERMEZ -- ampirik olarak (bkz. reports/02_ON_ISLEME_KARARLARI_PAH.md
    §11) `RobustScaler`'ın etkisi model-ailesine göre çok farklı (ağaç
    modelleri kayıtsız, LogReg için StandardScaler'a göre 15 puan daha
    kötü F1); ölçekleyici seçimi artık Aşama E'nin nested iç döngüsünde
    model-ailesine göre ayrı bir hiperparametre. v3'ün korelasyon-kümesi
    grup-özet özellikleri (`build_v3` içindeki inline mantık, bir Aşama B
    sınıfı DEĞİL) bu turun kapsamı dışında bırakıldı -- yalnızca sınıflanmış
    Aşama B bileşenleri birleştiriliyor.

Her iki pipeline da yalnızca `fit`/`transform` içerir, hiçbir sınıflandırıcı
YOKTUR -- Tamamlayıcı Deney Turu'nun "modelleme yok" kuralına uygun.

Çalıştırma: python -m genova.pah.serialize_pipelines
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from genova.pah.schema import SchemaGate
from genova.pah.missingness import BlockMissingIndicator, MedianImputerWithIndicator, ConstantFillImputer
from genova.pah.encoding import FrequencyEncoder, MultiValueFrequencyEncoder, NominalOneHotEncoder
from genova.pah.transforms import classify_al_columns, Log1pTransformer, LogitTransformer, RankQuantileHarmonizer
from genova.pah.dataset_versions import (
    NOMINAL_COLS,
    MULTI_VALUE_FREQ_COLS,
    SINGLE_VALUE_FREQ_COLS,
    EK_BLOCK_COLS,
    CAT_BLOCK_COLS,
)

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
OUT_DIR = ROOT / "artifacts" / "preprocessors"
EK_UNBOUNDED_COLS = ["EK_1", "EK_2", "EK_7", "EK_8", "EK_9"]
# v1, ham CSV semasindan farkli: CAT_6 (%100 eksik) build_v1 icinde zaten
# dusuruldu (bkz. dataset_versions.py). Ham semaya gore dogrulama burada
# yanlislikla basarisiz olur -- bu yuzden SchemaGate'e v1'in gercek
# semasini veriyoruz. Ham CSV -> v1 gecisinin kendi semasi zaten
# build_v1 icinde validate_schema ile bir kere dogrulanmisti.
V1_EXPECTED_GROUP_COUNTS = {"AL": 334, "CAT": 5, "EK": 9, "AA": 2}


def build_tree_pipeline(al_columns, al_numeric_cols):
    """v2 hattı: blok göstergeler + doldurma. Kategorikler native dtype
    (v1'de zaten `category`'ye çevrilmiş), ek kodlama YAPILMAZ -- v2'nin
    kendi tasarımıyla birebir aynı (bkz. reports/02_ON_ISLEME_KARARLARI_PAH.md).
    """
    return Pipeline([
        ("schema_gate", SchemaGate(expected_group_counts=V1_EXPECTED_GROUP_COUNTS)),
        ("al_all_missing", BlockMissingIndicator(columns=al_columns, name="al_all_missing")),
        ("ek_cat_block_missing", BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing")),
        ("ek3_impute", MedianImputerWithIndicator(columns=["EK_3"])),
        ("al_zero_fill", ConstantFillImputer(columns=al_numeric_cols, strategy="zero")),
    ])


def build_transformed_pipeline(al_columns, al_types):
    """v3'ün çekirdek hattı: + log1p/logit + EK_ harmonizasyon + tam
    kategorik kodlama. ÖLÇEKLEME İÇERMEZ (bkz. modül docstring'i) --
    model-ailesine göre ölçekleme Aşama E'nin nested iç döngüsünde ayrı
    bir hiperparametre olarak ele alınacak. `CAT_1` çok-değerli (`&`)
    satırlar taşıdığı için `MultiValueFrequencyEncoder`, `CAT_2` düz
    `FrequencyEncoder` kullanır.
    """
    al_numeric_cols = al_types["constant"] + al_types["ratio_type"] + al_types["frequency_type"]
    return Pipeline([
        ("schema_gate", SchemaGate(expected_group_counts=V1_EXPECTED_GROUP_COUNTS)),
        ("al_all_missing", BlockMissingIndicator(columns=al_columns, name="al_all_missing")),
        ("ek_cat_block_missing", BlockMissingIndicator(columns=EK_BLOCK_COLS + CAT_BLOCK_COLS, name="ek_cat_block_missing")),
        ("ek3_impute", MedianImputerWithIndicator(columns=["EK_3"])),
        ("al_zero_fill", ConstantFillImputer(columns=al_numeric_cols, strategy="zero")),
        ("log1p", Log1pTransformer(columns=al_types["frequency_type"])),
        ("logit", LogitTransformer(columns=al_types["ratio_type"])),
        ("ek_harmonize", RankQuantileHarmonizer(columns=EK_UNBOUNDED_COLS)),
        ("cat1_multi_freq", MultiValueFrequencyEncoder(columns=MULTI_VALUE_FREQ_COLS)),
        ("cat2_freq", FrequencyEncoder(columns=SINGLE_VALUE_FREQ_COLS)),
        ("cat_onehot", NominalOneHotEncoder(columns=NOMINAL_COLS)),
    ])


def compute_al_types(v1, al_columns):
    """DUZELTME (P1 madde 14, denetim bulgusu K19): classify_al_columns
    SIFIR-DOLDURMADAN SONRA cagrilmali -- nested CV'de fiilen kullanilan
    build_v3 (hem dataset_versions.py hem fold_versions.py) ikisi de
    siniflandirmayi ZATEN sifir-doldurulmus AL_ degerlerinden turetiyor
    (sifirlar frac_0_or_1'i sisirip kolonlari ratio_type'a kaydiriyor --
    bu, veriye gore DOGRU ya da YANLIS degil, ama TUTARLI olmali). Onceki
    surum burada ham v1 uzerinde (doldurmadan once) cagiriyordu, bu da
    build_v3'unkinden FARKLI bir siniflandirma uretip (90/82/162 vs
    24/309/1) bu dosyanin urettigi pah_pipeline_transformed.joblib'i,
    nested CV'nin gercekte kullandigi v3'ten sapan, "ismi ayni ama farkli"
    ikinci bir pipeline haline getiriyordu.

    Asagidaki gecici doldurma, `_v2_steps`'in (fold_versions.py) kullandigi
    ayni ConstantFillImputer(al_columns, "zero") adimidir -- yalnizca
    siniflandirma icin; asil pipeline'in kendi `al_zero_fill` adimi RAW v1
    uzerinde ayrica calisir (bkz. `build_tree_pipeline`/`build_transformed_
    pipeline`'daki `ConstantFillImputer` adimi).

    `tests/test_serialize_pipelines_pah.py::test_compute_al_types_matches_
    fold_versions_build_v3_full` bu fonksiyonun ciktisinin `fold_versions.
    py::build_v3_full`'un kendi (bagimsiz) hesapladigi siniflandirmayla
    BIREBIR AYNI oldugunu dogruluyor.
    """
    v1_zero_filled = ConstantFillImputer(columns=al_columns, strategy="zero").fit_transform(v1)
    return classify_al_columns(v1_zero_filled, al_columns)


def main():
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    al_types = compute_al_types(v1, al_columns)
    al_numeric_cols = al_types["constant"] + al_types["ratio_type"] + al_types["frequency_type"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tree_pipe = build_tree_pipeline(al_columns, al_numeric_cols)
    tree_pipe.fit(v1)
    tree_out = tree_pipe.transform(v1)
    joblib.dump(tree_pipe, OUT_DIR / "pah_pipeline_tree.joblib")
    print(f"pah_pipeline_tree: fit edildi, transform çıktısı {tree_out.shape}")

    transformed_pipe = build_transformed_pipeline(al_columns, al_types)
    transformed_pipe.fit(v1)
    transformed_out = transformed_pipe.transform(v1)
    joblib.dump(transformed_pipe, OUT_DIR / "pah_pipeline_transformed.joblib")
    print(f"pah_pipeline_transformed: fit edildi, transform çıktısı {transformed_out.shape}")

    for name, pipe in [("tree", tree_pipe), ("transformed", transformed_pipe)]:
        has_classifier = any(hasattr(step, "predict") for _, step in pipe.steps)
        print(f"{name} pipeline siniflandirici iceriyor mu: {has_classifier}")

    print(f"\nKaydedildi: {OUT_DIR / 'pah_pipeline_tree.joblib'}")
    print(f"Kaydedildi: {OUT_DIR / 'pah_pipeline_transformed.joblib'}")


if __name__ == "__main__":
    main()
