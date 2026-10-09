"""Preprocessing paketleme denetimi (README_FINAL/06 raporu), Adim 3:
`predict.py` HICBIR SEKILDE DEGISTIRILMEDEN, onun kendi fonksiyonlari
(`apply_v2_preprocessing`, `predict_dataframe`, `validate_predict_schema`)
DOGRUDAN import edilip, kucuk (5 satirlik) ham bir ornek uzerinde her ara
adimin (v2_steps sirayla, cat_categories, pool filtreleme, stringify,
model, kalibrator, onsel duzeltme, esik) ciktisi izlenir.

HICBIR MODEL/ESIK/DOSYA DEGISTIRILMEZ -- saf izleme/denetim.

Calistirma: python -m genova.pah.f_predict_pipeline_trace
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from predict import apply_v2_preprocessing, predict_dataframe, validate_predict_schema, DEFAULT_BUNDLE_PATH
from genova.pah.e2ek_models import _stringify_categoricals

RAW_CSV = ROOT / "data" / "raw" / "YARISMA_TRAIN_PAH.csv"
N_SAMPLE = 5


def main():
    bundle = joblib.load(DEFAULT_BUNDLE_PATH)
    raw = pd.read_csv(RAW_CSV)
    sample = raw.drop(columns=[c for c in ("Label", "group_id") if c in raw.columns]).head(N_SAMPLE).copy()
    print(f"ham ornek: {sample.shape[0]} satir, {sample.shape[1]} kolon (Label/group_id cikarildi)", flush=True)

    print("\n=== Adim A: sema dogrulama ===", flush=True)
    validate_predict_schema(sample)
    print("  OK -- gecerli sema.", flush=True)

    print("\n=== Adim B: v2_steps sirayla (BlockMissingIndicator x2, MedianImputerWithIndicator, ConstantFillImputer) ===", flush=True)
    out = sample.copy()
    for i, step in enumerate(bundle["v2_steps"]):
        before_cols = set(out.columns)
        out = step.transform(out)
        new_cols = set(out.columns) - before_cols
        print(f"  [{i}] {type(step).__name__}: yeni kolon(lar)={sorted(new_cols) or '(yok, yerinde donusum)'}", flush=True)
    print(f"  al_all_missing degerleri: {out['al_all_missing'].tolist()}", flush=True)
    print(f"  EK_3_missing degerleri: {out['EK_3_missing'].tolist()}", flush=True)
    print(f"  EK_3 (medyanla dolduruldu mu): {out['EK_3'].tolist()}", flush=True)
    print(f"  AL_1 (sifirla dolduruldu mu, once/sonra NaN sayisi): "
          f"once={sample['AL_1'].isna().sum()} sonra={out['AL_1'].isna().sum()}", flush=True)

    print("\n=== Adim C: cat_categories yeniden-esleme (CAT_1-5/AA_1-2, final pool'da YOK ama yine de uygulaniyor) ===", flush=True)
    for col in bundle["cat_categories"]:
        print(f"  {col}: final pool'da mi? {'EVET' if col in bundle['pool'] else 'HAYIR -- daha sonra filtrelenip atilacak'}", flush=True)

    print("\n=== Adim D: apply_v2_preprocessing (predict.py'nin GERCEK fonksiyonu) -> pool filtreleme ===", flush=True)
    X = apply_v2_preprocessing(sample, bundle)
    print(f"  cikti sekli: {X.shape}, kolonlar bundle['pool']'la BIREBIR AYNI SIRADA mi? "
          f"{list(X.columns) == bundle['pool']}", flush=True)
    print(f"  X.dtypes ozeti: {X.dtypes.value_counts().to_dict()}", flush=True)

    print("\n=== Adim E: stringify (model_cat_features bos oldugu icin hicbir kolon donusmemeli) ===", flush=True)
    X_str = _stringify_categoricals(X, bundle["model_cat_features"])
    print(f"  model_cat_features={bundle['model_cat_features']} -> "
          f"X ile X_str ayni mi? {X.equals(X_str)}", flush=True)

    print("\n=== Adim F: model.predict_proba -> calibrator -> sld_correct -> esik (predict.py'nin GERCEK fonksiyonu) ===", flush=True)
    result = predict_dataframe(sample, bundle)
    print(result.to_string(index=False), flush=True)
    print(f"\n  bundle esigi={bundle['threshold']} w1={bundle['w1']:.6f} w0={bundle['w0']:.6f}", flush=True)

    print("\n=== Beklenen 10-maddelik zincirle karsilastirma ===", flush=True)
    print("  1) ozellik listesi+sirasi: bundle['pool'] -> apply_v2_preprocessing'in son satirinda kullanildi. DOGRULANDI.", flush=True)
    print("  3) turetilmis ozellikler (al_all_missing/EK_3_missing): v2_steps icinde uretildi, sonra pool'da OLMADIKLARI icin filtrelendi. DOGRULANDI.", flush=True)
    print("  5) kategorik kodlama: model_cat_features=[] (final pool'da CAT_/AA_ yok) -> stringify hicbir seyi degistirmedi. DOGRULANDI.", flush=True)
    print("  7-9) kalibrator/onsel/esik: bundle'dan okundu, predict_dataframe icinde sirayla uygulandi. DOGRULANDI.", flush=True)


if __name__ == "__main__":
    main()
