"""P1 madde 9, Adim 3: her modelin en iyi A/B/C varyantini, mevcut final
aday CatBoost/v4_from_v2 ile Nadeau-Bengio testiyle karsilastirir --
madde 7'nin kurdugu NB-zorunlu-kriter disipliniyle AYNI.

ONEMLI TASARIM NOTU (metin sapmasi, seffaf belgelendi): gorev metni
karsilastirmayi "final_model_bundle_v2.pkl'in weighted-F1'i" ile yapmayi
istiyordu -- ama bu, ADAYLARIN ham (esik=0,5, kalibrasyonsuz) F1'ini,
final adayin TAMAMEN FARKLI bir olcekteki (Beta-kalibre + kapali-form
Bayes onsel duzeltmesi + uyarlanabilir esik SONRASI) weighted-F1'iyle
karsilastirmak anlamina gelirdi -- bu iki metrik ayni olcekte DEGIL
(egitim-onseli altinda ham F1 tipik olarak ~0.85-0.95 civari, weighted-F1
tipik olarak ~0.55-0.65 civari, bkz. final_model_bundle_v2 OOF F1=0.82 vs
weighted-F1~0.62). Bu ikisini dogrudan NB'ye vermek, aday YAPISAL OLARAK
sahte-anlamli "ustun" cikarirdi (metrik olcek farkindan, gercek bir
ustunlukten degil) -- P1'in tum ruhuyla (zayif/yanlis sinyalleri elemek
icin NB'yi zorunlu kriter yapmak) dogrudan celisir.

Bu yuzden PRIMARY karsilastirma, HER IKI tarafta da AYNI metrigi (ham
F1, esik=0,5, kalibrasyonsuz -- E2'nin kendi eleme olcutu, catboost/
v4_from_v2 STRATEJI B ile `e2_model_comparison.csv`'den) kullanir --
bu, adaylarin GERCEKTEN elendigi olcutle ayni, dolayisiyla "eleme adil
miydi" sorusuna gecerli bir cevap. Gorevin literal olarak istedigi
weighted-F1 karsilastirmasi da IKINCIL/bilgi-amacli olarak hesaplanip
raporlanir, ama net bir olcek-uyumsuzlugu uyarisiyla.

Calistirma: python -m genova.pah.e2_madde9_nb_decision
"""
from pathlib import Path

import numpy as np
import pandas as pd

from genova.statistics import nadeau_bengio_corrected_ttest

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
COMPARISON_CSV = TAB_DIR / "e2_model_comparison.csv"
MADDE9_CSV = TAB_DIR / "e2_madde9_weighting_extension.csv"
E5_CSV = TAB_DIR / "e5_threshold_selection.csv"
OUT_CSV = TAB_DIR / "e2_madde9_nb_decision.csv"

B_LABEL = "B_fixed_minority_2x"
CANDIDATES = [("xgboost", "v1"), ("lightgbm", "v1"), ("elasticnet", "v3"), ("elasticnet", "v4_from_v3")]


def _b_rows(comparison_df, model_name, version_name):
    return comparison_df[
        (comparison_df.model == model_name) & (comparison_df.data_version == version_name)
        & (comparison_df.weighting_variant == B_LABEL)
    ].sort_values(["repeat", "outer_fold"]).reset_index(drop=True)


def main():
    comparison_df = pd.read_csv(COMPARISON_CSV)
    madde9_df = pd.read_csv(MADDE9_CSV)
    e5_df = pd.read_csv(E5_CSV)

    final_b = _b_rows(comparison_df, "catboost", "v4_from_v2")
    final_weighted_f1 = e5_df[(e5_df.model == "catboost") & (e5_df.data_version == "v4_from_v2")].sort_values(
        ["repeat", "outer_fold"]).reset_index(drop=True)
    print(f"final aday (catboost/v4_from_v2): ham-F1 ort={final_b.f1.mean():.4f}  weighted-F1 ort={final_weighted_f1.weighted_f1_chosen.mean():.4f}", flush=True)

    rows_out = []
    any_significant_primary = False

    for model_name, version_name in CANDIDATES:
        b_rows = _b_rows(comparison_df, model_name, version_name)
        variant_means = {"B_fixed_minority_2x": b_rows.f1.mean()}
        variant_frames = {"B_fixed_minority_2x": b_rows}
        for label in ("A_no_weight", "C_data_driven_spw"):
            sub = madde9_df[
                (madde9_df.model == model_name) & (madde9_df.data_version == version_name)
                & (madde9_df.weighting_variant == label)
            ].sort_values(["repeat", "outer_fold"]).reset_index(drop=True)
            variant_means[label] = sub.f1.mean()
            variant_frames[label] = sub

        best_label = max(variant_means, key=variant_means.get)
        best_frame = variant_frames[best_label]
        print(f"\n=== {model_name}/{version_name}: A={variant_means['A_no_weight']:.4f} "
              f"B={variant_means['B_fixed_minority_2x']:.4f} C={variant_means['C_data_driven_spw']:.4f} "
              f"-> en iyi={best_label} ===", flush=True)

        assert (best_frame["repeat"].to_numpy() == final_b["repeat"].to_numpy()).all()
        assert (best_frame["outer_fold"].to_numpy() == final_b["outer_fold"].to_numpy()).all()
        n_train, n_test = final_b["n_train"].to_numpy(), final_b["n_test"].to_numpy()

        nb_primary = nadeau_bengio_corrected_ttest(best_frame["f1"].to_numpy(), final_b["f1"].to_numpy(), n_train, n_test)
        favors_candidate_primary = (nb_primary["mean_diff"] > 0) and (nb_primary["p_value"] < 0.05)
        print(f"  PRIMARY (ham F1 vs ham F1, ayni olcek): candidate-final mean_diff={nb_primary['mean_diff']:+.4f} "
              f"p={nb_primary['p_value']:.4f}  aday-lehine-anlamli={favors_candidate_primary}", flush=True)

        # ikincil / bilgi amacli: gorevin literal istedigi (ham F1 vs weighted-F1) -- OLCEK UYUMSUZ, karar icin KULLANILMIYOR
        assert (best_frame["repeat"].to_numpy() == final_weighted_f1["repeat"].to_numpy()).all()
        assert (best_frame["outer_fold"].to_numpy() == final_weighted_f1["outer_fold"].to_numpy()).all()
        nb_secondary = nadeau_bengio_corrected_ttest(
            best_frame["f1"].to_numpy(), final_weighted_f1["weighted_f1_chosen"].to_numpy(), n_train, n_test,
        )
        print(f"  IKINCIL (ham F1 vs weighted-F1, OLCEK UYUMSUZ -- yalnizca bilgi amacli): "
              f"mean_diff={nb_secondary['mean_diff']:+.4f} p={nb_secondary['p_value']:.4f}", flush=True)

        if favors_candidate_primary:
            any_significant_primary = True

        rows_out.append({
            "model": model_name, "data_version": version_name,
            "mean_f1_A": variant_means["A_no_weight"], "mean_f1_B": variant_means["B_fixed_minority_2x"],
            "mean_f1_C": variant_means["C_data_driven_spw"], "best_variant": best_label,
            "nb_primary_mean_diff": nb_primary["mean_diff"], "nb_primary_p_value": nb_primary["p_value"],
            "favors_candidate_primary": favors_candidate_primary,
            "nb_secondary_scale_mismatched_mean_diff": nb_secondary["mean_diff"],
            "nb_secondary_scale_mismatched_p_value": nb_secondary["p_value"],
        })

    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")

    print(f"\n=== KARAR KURALI: herhangi bir aday PRIMARY (ayni-olcek) testte final adaya karsi "
          f"anlamli ustunluk gosteriyor mu: {any_significant_primary} ===")
    if any_significant_primary:
        print("SONUC: DURDURULDU -- en az bir aday anlamli cikti, kullaniciya bildirilmeli, otomatik benimseme YOK.")
    else:
        print("SONUC: mevcut eleme karari SAGLAM -- hicbir aday (A/B/C'nin en iyisi bile) final adaya karsi "
              "istatistiksel olarak anlamli bir ustunluk gostermiyor. Final model DEGISMIYOR.")


if __name__ == "__main__":
    main()
