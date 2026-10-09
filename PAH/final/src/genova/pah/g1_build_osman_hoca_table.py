"""Revize turu, Adim 4: Osman Hoca'nin cografi-panel rehberlik notunun
istedigi formatta (Preprocessing | Model | F1 | MCC | Sensitivity |
Specificity | AUPRC | AUROC | Threshold | Std) TEK, birlesik karar
tablosu. HICBIR yeni hesaplama yapmiyor -- yalnizca Adim 1-3'un
zaten uretip diske yazdigi CSV'leri okuyup birlestiriyor.

Onemli notlar (kaynak-farkli eslikleri karistirmayin):
  - E1/E2 satirlari: SABIT esik (E1: 0,359; E2: 0,5), KALIBRASYONSUZ ham
    olasilik.
  - E5 satirlari: NESTED (dis-fold-bazli) uyarlanabilir esik + Beta
    kalibrasyon + SLD onsel duzeltmesi -- bu yuzden F1'leri E1/E2'den
    DUSUK gorunur, bu BEKLENEN bir sonuctur (bkz. 06 raporu E5 bolumu),
    "kotu model" degil.
  - F0 (final, deploy edilen) satiri: final bundle'in KENDI 5-fold
    cross-fit OOF'u uzerinde (369 satirin tamami, deterministik) --
    E5'in 50-dis-fold ortalamasindan farkli bir olcum birimi (n=369 tek
    partisyon vs n=50 fold ortalamasi), yine karistirilmamali.

`Threshold` sutunu: E1/E2 icin sabit deger; E5/F0 icin dis-fold-bazli
uyarlanabilir esigin kendi ort±std'si.
`Std` sutunu: F1'in std'si (fold-arasi, F0 haric -- F0 tek bir OOF
olcumu, "std" yerine "n=369" notu dusuluyor).

Calistirma: python -m genova.pah.g1_build_osman_hoca_table
"""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models" / "pah"

E1_CSV = TAB_DIR / "g1_e1_baseline_auroc_sensitivity.csv"
E2_CSV = TAB_DIR / "e2_model_comparison.csv"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"
F0_CSV = TAB_DIR / "f0_uncertainty_analysis.csv"
RF_CSV = TAB_DIR / "g1_rf_v4fromv2_auroc_sensitivity.csv"
OUT_CSV = TAB_DIR / "13_OSMAN_HOCA_FORMATI_KARAR_TABLOSU.csv"

E2_PREPROCESSING_LABELS = {
    "v1": "v1 (ham NaN, native kategorik)",
    "v4_from_v2": "v4_from_v2 (25 ozellik, sifir-dolu)",
}


def _fmt(mean, std):
    return f"{mean:.4f}", f"{std:.4f}"


def _e1_row():
    df = pd.read_csv(E1_CSV)
    f1_mean, f1_std = df["f1"].mean(), df["f1"].std()
    return {
        "Preprocessing": "v1 (ham NaN, native kategorik)", "Model": "XGBoost (E1 baseline, PDR sabit HP)",
        "F1": f"{f1_mean:.4f}", "MCC": f"{df['mcc'].mean():.4f}", "Sensitivity": f"{df['sensitivity'].mean():.4f}",
        "Specificity": f"{df['specificity'].mean():.4f}", "AUPRC": "n/a", "AUROC": f"{df['auroc'].mean():.4f}",
        "Threshold": "0.359 (sabit)", "Std": f"{f1_std:.4f}",
    }


def _e2_row(e2_df, model_name, version_name, model_label):
    rows = e2_df[(e2_df.model == model_name) & (e2_df.data_version == version_name)
                 & (e2_df.weighting_variant == "B_fixed_minority_2x")]
    assert len(rows) == 50 and rows["auroc"].notna().all(), f"{model_name}/{version_name}: replay eksik"
    return {
        "Preprocessing": E2_PREPROCESSING_LABELS[version_name], "Model": model_label,
        "F1": f"{rows['f1'].mean():.4f}", "MCC": f"{rows['mcc'].mean():.4f}",
        "Sensitivity": f"{rows['sensitivity'].mean():.4f}", "Specificity": f"{rows['specificity'].mean():.4f}",
        "AUPRC": f"{rows['auprc'].mean():.4f}", "AUROC": f"{rows['auroc'].mean():.4f}",
        "Threshold": "0.5 (sabit, kalibrasyonsuz)", "Std": f"{rows['f1'].std():.4f}",
    }


def _e2_rf_row():
    rf = pd.read_csv(RF_CSV)
    e2_df = pd.read_csv(E2_CSV)
    rf_recorded = e2_df[(e2_df.model == "random_forest") & (e2_df.data_version == "v4_from_v2")
                         & (e2_df.weighting_variant == "B_fixed_minority_2x")]
    return {
        "Preprocessing": E2_PREPROCESSING_LABELS["v4_from_v2"], "Model": "Random Forest (E2-EK, Strateji B)",
        "F1": f"{rf_recorded['f1'].mean():.4f}", "MCC": f"{rf_recorded['mcc'].mean():.4f}",
        "Sensitivity": f"{rf['sensitivity'].mean():.4f}", "Specificity": f"{rf_recorded['specificity'].mean():.4f}",
        "AUPRC": f"{rf_recorded['auprc'].mean():.4f}", "AUROC": f"{rf['auroc'].mean():.4f}",
        "Threshold": "0.5 (sabit, kalibrasyonsuz)", "Std": f"{rf_recorded['f1'].std():.4f}",
    }


def _e5_row(e5_df, model_name, version_name, model_label):
    rows = e5_df[(e5_df.model == model_name) & (e5_df.data_version == version_name)]
    assert len(rows) == 50
    return {
        "Preprocessing": E2_PREPROCESSING_LABELS.get(version_name, version_name) + " + Beta kalibrasyon + SLD onsel duzeltmesi",
        "Model": model_label,
        "F1": f"{rows['f1_chosen'].mean():.4f}", "MCC": f"{rows['mcc_chosen'].mean():.4f}",
        "Sensitivity": f"{rows['sensitivity_chosen'].mean():.4f}", "Specificity": f"{rows['specificity_chosen'].mean():.4f}",
        "AUPRC": "n/a", "AUROC": f"{rows['auroc'].mean():.4f}",
        "Threshold": f"{rows['chosen_threshold'].mean():.4f} +/- {rows['chosen_threshold'].std():.4f} (nested, dis-fold-bazli)",
        "Std": f"{rows['f1_chosen'].std():.4f}",
    }


def _f0_row():
    df = pd.read_csv(F0_CSV)
    sub = df[df.analysis == "sample_level_bootstrap"].set_index("metric")
    bundle = joblib.load(MODEL_DIR / "final_model_bundle_v2.pkl")
    return {
        "Preprocessing": "v4_from_v2 (26 ozellik) + Beta kalibrasyon + SLD onsel duzeltmesi",
        "Model": "[FINAL] CatBoost (F0 -- deploy edilen model)",
        "F1": f"{sub.loc['f1', 'point_estimate']:.4f}", "MCC": f"{sub.loc['mcc', 'point_estimate']:.4f}",
        "Sensitivity": f"{sub.loc['sensitivity', 'point_estimate']:.4f}",
        "Specificity": f"{sub.loc['specificity', 'point_estimate']:.4f}", "AUPRC": "n/a",
        "AUROC": f"{sub.loc['auroc', 'point_estimate']:.4f}",
        "Threshold": f"{bundle['threshold']:.4f} (sabit, final)",
        "Std": f"(n=369, tek OOF partisyonu; %95 CI: F1=[{sub.loc['f1','ci_low']:.4f};{sub.loc['f1','ci_high']:.4f}])",
    }


def main():
    e2_df = pd.read_csv(E2_CSV)
    e5_df = pd.read_csv(E5_CSV)

    rows = [
        _e1_row(),
        _e2_row(e2_df, "catboost", "v1", "CatBoost (E2, en iyi ham)"),
        _e2_row(e2_df, "catboost", "v4_from_v2", "CatBoost/v4_from_v2 (E2, final adayin ozellik havuzu)"),
        _e2_row(e2_df, "lightgbm", "v1", "LightGBM (E2)"),
        _e2_rf_row(),
        _e5_row(e5_df, "catboost", "v1", "CatBoost/v1 (E5, 3 adaydan biri)"),
        _e5_row(e5_df, "catboost", "v4_from_v2", "CatBoost/v4_from_v2 (E5, secilen final aday)"),
        _e5_row(e5_df, "lightgbm", "v1", "LightGBM/v1 (E5)"),
        _e5_row(e5_df, "random_forest", "v4_from_v2", "Random Forest/v4_from_v2 (E5-EK)"),
        _f0_row(),
    ]

    out = pd.DataFrame(rows, columns=[
        "Preprocessing", "Model", "F1", "MCC", "Sensitivity", "Specificity", "AUPRC", "AUROC", "Threshold", "Std",
    ])
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
