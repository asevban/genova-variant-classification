"""PAH paneli -- final model ile tahmin uretir.

Kullanim:
    python predict.py --input test.csv --output submission.csv

Ham CSV'yi okur (Variant_ID + AL_(334)/CAT_(6)/EK_(9)/AA_(2) kolonlari --
Label GEREKMEZ, gorulmez bile), semayi dogrular, final bundle
`models/final_model_bundle_v2.pkl` ile dondurulmus on isleme + ozellik havuzu
(26 ozellik --
P1 madde 11: provenance-riskli `al_all_missing`/`CAT_1` cikarilmis) +
model + kalibrator + onsel duzeltmesi + esigi sirayla uygular,
`{Variant_ID, predicted_label, predicted_probability, panel}` formatinda
bir CSV yazar (F5: cok-panel birlestirmeye hazir arayuz -- bkz.
PAH_SUBMISSION_INTERFACE.md).

Eski `final_model_bundle.pkl` (28 ozellik, provenance-riskli ikisi dahil)
diskte duruyor ama artik KULLANILMIYOR -- yalnizca tarihsel kayit (bkz.
reports/06_MODEL_SECIM_RAPORU_PAH.md, P1 madde 11 bolumu: iki modelin
istatistiksel olarak ayirt edilemedigi, madde 10'un araclariyla dogrulandi).

Kolon SIRASINA guvenilmez -- her adim isim-bazli hizalama yapar.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np
import pandas as pd

from genova.pah.e2ek_models import _stringify_categoricals
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.schema import _group_of, EXPECTED_GROUP_COUNTS

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_PATH = ROOT / "models" / "final_model_bundle_v2.pkl"

NON_SCHEMA_COLS = ("Variant_ID", "Label", "group_id")
PANEL_NAME = "PAH"


def validate_predict_schema(df):
    """Ham girdi semasini dogrular -- `Label` ZORUNLU DEGIL (tahmin edilen
    hedef budur). Gecersizse acik bir `ValueError` firlatir."""
    issues = []
    if "Variant_ID" not in df.columns:
        issues.append("eksik zorunlu kolon: Variant_ID")
    elif not df["Variant_ID"].is_unique:
        issues.append("Variant_ID benzersiz degil")

    group_counts = {}
    for col in df.columns:
        if col in NON_SCHEMA_COLS:
            continue
        group_counts[_group_of(col)] = group_counts.get(_group_of(col), 0) + 1
    for group, expected in EXPECTED_GROUP_COUNTS.items():
        actual = group_counts.get(group, 0)
        if actual != expected:
            issues.append(f"{group}_ grubu {actual} kolon, beklenen {expected}")

    if issues:
        raise ValueError(f"predict.py: girdi semasi gecersiz: {issues}")


def apply_v2_preprocessing(df, bundle):
    """Bundle'in dondurulmus (369 satirin tamaminda fit edilmis) v2-stili
    on isleme adimlarini yalnizca `.transform()` ile uygular -- hicbir
    adim burada YENIDEN FIT EDILMIYOR. Kolon sirasina degil isme gore
    calisir (tum adimlar pandas isim-bazli erisim kullaniyor)."""
    out = df.copy()
    for step in bundle["v2_steps"]:
        out = step.transform(out)

    for col, categories in bundle["cat_categories"].items():
        values = out[col].astype(object)
        values = values.where(values.isin(categories), other=np.nan)
        out[col] = pd.Categorical(values, categories=categories)

    cols = [c for c in bundle["pool"] if c in out.columns]
    return out[cols]


def predict_dataframe(raw_df, bundle):
    """Ham DataFrame -> (Variant_ID, Label, proba) DataFrame'i. Satir
    sirasi `raw_df` ile birebir hizali kalir (hicbir adim yeniden
    siralama/indeksleme yapmiyor)."""
    validate_predict_schema(raw_df)
    X = apply_v2_preprocessing(raw_df, bundle)
    X_str = _stringify_categoricals(X, bundle["model_cat_features"])

    raw_proba = bundle["model"].predict_proba(X_str)[:, 1]
    calibrated = bundle["calibrator"].transform(raw_proba)
    sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
    pred = (sld >= bundle["threshold"]).astype(int)

    return pd.DataFrame({
        "Variant_ID": raw_df["Variant_ID"].to_numpy(),
        "Label": pred,
        "proba": sld,
    })


def format_submission_output(result, panel=PANEL_NAME):
    """`predict_dataframe`'in ic-kullanim ciktisini (Variant_ID, Label,
    proba) F5 submission-arayuzu sozlesmesine cevirir (bkz.
    PAH_SUBMISSION_INTERFACE.md): sert etiket + ham (kalibre, SLD-
    duzeltilmis) olasilik + panel kaynak-isareti, hepsi ayri kolonlarda."""
    return pd.DataFrame({
        "Variant_ID": result["Variant_ID"],
        "predicted_label": result["Label"],
        "predicted_probability": result["proba"],
        "panel": panel,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ham girdi CSV yolu")
    parser.add_argument("--output", required=True, help="cikti CSV yolu")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH), help="final_model_bundle_v2.pkl yolu")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    raw_df = pd.read_csv(args.input)
    result = predict_dataframe(raw_df, bundle)
    submission = format_submission_output(result)
    submission.to_csv(args.output, index=False)
    print(f"kaydedildi: {args.output} ({len(submission)} satir, "
          f"{int(submission['predicted_label'].sum())} patojenik tahmin)")


if __name__ == "__main__":
    main()
