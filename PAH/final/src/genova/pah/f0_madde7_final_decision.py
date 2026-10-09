"""P1 madde 7 son adim: `depth=5` (`final_model_bundle_v2.pkl`, mevcut
resmi final) vs `depth=3` (`final_model_bundle_v3.pkl`, tekrarli-bolme
teshisinde 10 tekrarin 7'sinde kazanan aday) icin Nadeau-Bengio testini
ZORUNLU karar kriteri yapar -- yalnizca CI ortusmesi/ortusmemesi ya da
nominal fark buyuklugu YETERLI DEGIL (bu projede tekrar tekrar gorulen
"nominal fark buyuk, testte anlamsiz" deseninin bir tekrari olabilir --
bkz. CatBoost/RF, Beta/Platt, ve bizzat bu maddenin kendi binom-test
kontrolu).

Iki bundle da AYNI havuzu (26 ozellik, f0_final_feature_pool_v2.json)
kullaniyor -- Adim 1'de dogrulandi, yeniden uretim gerekmedi. Iki bundle
da AYNI 5-fold ic CV yapisini (`sb.build_outer_folds(v1_df, seed=F0_SEED+1)`)
paylasiyor -- bu, Nadeau-Bengio'nun k=5 PAIRED karsilastirmasini gecerli
kilan sey.

HICBIR MODEL YENIDEN EGITILMIYOR -- `f0_final_model_v3.py::_per_fold_
scores_for_bundle`/`_oof_predictions_for_bundle` (degistirilmedi) aynen
yeniden kullanilip her iki bundle'in KENDI dondurulmus `pool`/`best_
params`/`calibrator`/`threshold`'uyla ayni deterministik cross-fit tekrar
calistiriliyor.

Karar kurali (kesin, onceden belirlenmis): NB p<0,05 (herhangi bir ana
metrikte) VE yon depth=3 lehineyse -> v3 yeni resmi final. Aksi halde
(p>=0,05 herhangi bir metrikte VEYA yon depth=5 lehine donerse) -> depth=5
korunur, v3 SILINMEZ (tarihsel/incelenmis-ama-benimsenmemis aday).

Calistirma: python -m genova.pah.f0_madde7_final_decision
"""
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity, f1_binary_positive_weighted
from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah.models import fit_predict_catboost
from genova.pah.f0_final_model import MODEL_DIR, V1_PATH, F0_SEED
from genova.pah.f0_final_model_v3 import CURRENT_BUNDLE_PATH, BUNDLE_V3_OUT, BUNDLE_ARCHIVED, _oof_predictions_for_bundle
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.e5_threshold_selection import _prior_weights, FINAL_PATHOGENIC_PRIOR
from genova.statistics import sample_level_bootstrap_ci, nadeau_bengio_corrected_ttest

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
OUT_CSV = TAB_DIR / "f0_madde7_nb_decision.csv"
N_BOOT = 2000


def _per_fold_scores_full(v1_df, bundle):
    """`f0_final_model_v3.py::_per_fold_scores_for_bundle` ile AYNI 5-fold
    split ve fit/kalibrasyon/esik mantigi -- ek olarak onsel-agirlikli F1
    de hesaplaniyor (gorevin istedigi 4. metrik)."""
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
            v1_df, fold["train_variant_ids"], fold["test_variant_ids"], bundle["pool"],
        )
        proba = fit_predict_catboost(X_tr, y_tr, X_va, bundle["model_best_params"])
        calibrated = bundle["calibrator"].transform(proba)
        sld = sld_correct(calibrated, w1=bundle["w1"], w0=bundle["w0"])
        pred = (sld >= bundle["threshold"]).astype(int)
        weights = _prior_weights(y_va, FINAL_PATHOGENIC_PRIOR)
        rows.append({
            "fold": fold["fold"], "n_train": len(y_tr), "n_test": len(y_va),
            "f1": f1_binary_positive(y_va, pred),
            "mcc": matthews_correlation_coefficient(y_va, pred),
            "specificity": specificity(y_va, pred),
            "weighted_f1": f1_binary_positive_weighted(y_va, pred, weights),
        })
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    variant_ids = v1_df["Variant_ID"].tolist()

    old_bundle = joblib.load(CURRENT_BUNDLE_PATH)  # depth=5, final_model_bundle_v2.pkl
    new_bundle = joblib.load(BUNDLE_V3_OUT)  # depth=3, final_model_bundle_v3.pkl
    assert old_bundle["pool"] == new_bundle["pool"], "iki bundle farkli havuz kullaniyor -- Adim 1 varsayimi bozuk"
    print(f"eski (depth=5): esik={old_bundle['threshold']}  yeni (depth=3): esik={new_bundle['threshold']}", flush=True)

    print("\n=== Adim 2.1: Nadeau-Bengio (ayni 5-fold ic CV, k=5) ===", flush=True)
    old_folds = _per_fold_scores_full(v1_df, old_bundle)
    new_folds = _per_fold_scores_full(v1_df, new_bundle)
    assert (old_folds["fold"].to_numpy() == new_folds["fold"].to_numpy()).all()
    n_train, n_test = old_folds["n_train"].to_numpy(), old_folds["n_test"].to_numpy()

    nb_results = {}
    for metric in ("f1", "mcc", "specificity", "weighted_f1"):
        nb = nadeau_bengio_corrected_ttest(old_folds[metric].to_numpy(), new_folds[metric].to_numpy(), n_train, n_test)
        nb_results[metric] = nb
        direction = "depth=3 lehine" if nb["mean_diff"] < 0 else ("depth=5 lehine" if nb["mean_diff"] > 0 else "fark yok")
        print(f"  {metric}: mean_diff(eski-yeni)={nb['mean_diff']:+.4f}  p={nb['p_value']:.4f}  yon={direction}", flush=True)

    print("\n=== Adim 2.2: ornek-duzeyi CI (referans) ===", flush=True)
    old_sld, old_pred, old_y = _oof_predictions_for_bundle(v1_df, old_bundle, all_ids)
    new_sld, new_pred, new_y = _oof_predictions_for_bundle(v1_df, new_bundle, all_ids)
    assert np.array_equal(old_y, new_y)

    ci_rows = []
    for metric_name, metric_fn in (
        ("f1", f1_binary_positive), ("mcc", matthews_correlation_coefficient), ("specificity", specificity),
    ):
        old_ci = sample_level_bootstrap_ci(variant_ids, old_y, old_pred, metric_fn, n_boot=N_BOOT, seed=1)
        new_ci = sample_level_bootstrap_ci(variant_ids, new_y, new_pred, metric_fn, n_boot=N_BOOT, seed=1)
        overlap = not (new_ci["ci_high"] < old_ci["ci_low"] or old_ci["ci_high"] < new_ci["ci_low"])
        print(f"  {metric_name}: eski={old_ci['point_estimate']:.4f} [{old_ci['ci_low']:.4f},{old_ci['ci_high']:.4f}]  "
              f"yeni={new_ci['point_estimate']:.4f} [{new_ci['ci_low']:.4f},{new_ci['ci_high']:.4f}]  ortusuyor={overlap}", flush=True)
        ci_rows.append({"metric": metric_name, "old_point": old_ci["point_estimate"], "new_point": new_ci["point_estimate"],
                         "old_ci_low": old_ci["ci_low"], "old_ci_high": old_ci["ci_high"],
                         "new_ci_low": new_ci["ci_low"], "new_ci_high": new_ci["ci_high"], "ci_overlap": overlap})

    # ---------------------------------------------------------- karar kurali
    significant_for_depth3 = any(
        (nb_results[m]["p_value"] < 0.05) and (nb_results[m]["mean_diff"] < 0)  # eski-yeni<0 -> yeni(depth=3) daha yuksek
        for m in ("f1", "mcc", "specificity", "weighted_f1")
    )
    print(f"\n=== KARAR KURALI: herhangi bir ana metrikte p<0,05 VE yon depth=3 lehine mi: {significant_for_depth3} ===", flush=True)

    rows_out = [{"analysis": "nadeau_bengio", "metric": m, "mean_diff_old_minus_new": nb_results[m]["mean_diff"],
                 "p_value": nb_results[m]["p_value"], "t_stat": nb_results[m]["t_stat"], "k": nb_results[m]["k"]}
                for m in nb_results]
    for r in ci_rows:
        rows_out.append({"analysis": "sample_level_bootstrap_ci", "metric": r["metric"],
                          "mean_diff_old_minus_new": r["old_point"] - r["new_point"],
                          "p_value": np.nan, "t_stat": np.nan, "k": np.nan,
                          **{k: v for k, v in r.items() if k != "metric"}})
    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
    print(f"kaydedildi: {OUT_CSV}", flush=True)

    if significant_for_depth3:
        print("\n=== SONUC: depth=3 icin istatistiksel olarak savunulabilir kanit var -- "
              "final_model_bundle_v3.pkl YENI RESMI FINAL yapiliyor. ===", flush=True)
        if not BUNDLE_ARCHIVED.exists():
            shutil.copy2(CURRENT_BUNDLE_PATH, BUNDLE_ARCHIVED)
            print(f"arsivlendi: {CURRENT_BUNDLE_PATH} -> {BUNDLE_ARCHIVED} (orijinal SILINMEDI)", flush=True)
        print("predict.py GUNCELLENMELI (bu betik dosyayi otomatik degistirmiyor, ayrica yapilacak).", flush=True)
    else:
        print("\n=== SONUC: istatistiksel olarak yetersiz kanit (NB p>=0,05 ya da yon depth=5 lehine) -- "
              "depth=5 (final_model_bundle_v2.pkl) KORUNUYOR. final_model_bundle_v3.pkl SILINMEDI "
              "(tarihsel/incelenmis-ama-benimsenmemis aday). predict.py DEGISTIRILMEDI. ===", flush=True)


if __name__ == "__main__":
    main()
