"""P2 madde 17: RF'nin `v1`/`v2_raw_nan`'da zayif cikmasinin (MCC~=0,04-
0,06) gercekten bir model kisiti mi, yoksa yalnizca doldurma stratejisi
artefakti mi oldugunu netlestirir. CatBoost/XGBoost/LightGBM `AL_`
kolonlarinda hicbir doldurma gormuyor (native NaN); RF/KNN NaN kabul
etmedigi icin MUTLAKA bir doldurma gerektiriyor -- eski `to_dense_numeric`
bunu TRAIN MEDYANIYLA yapiyordu. Bu turda `AL_` icin SIFIR-doldurma
(v2'nin kendi stratejisi) denenip karsilastiriliyor.

Yeni hiperparametre aramasi YOK -- `e2_model_comparison.csv`'nin ZATEN
kaydettigi (Strateji B) `best_params` aynen yeniden kullanildi. Final
model RF DEGIL (CatBoost kazandi) -- bu gorev bir final model karari
degistirmiyor, yalnizca E2'nin karsilastirma kaydinin adilligini
dogruluyor/belgeliyor.

KNN icin: ayni adalet sorununu tasiyor ama zaten en zayif aday (MCC=0,07)
oldugu icin bu turda yeniden olculmedi (raporda tek cumlelik not).

Sonuc YENI bir dosyaya yazildi: `e2_model_comparison.csv`'ye EKLENMEDI.

Calistirma: python -m genova.pah.e2_madde17_rf_imputation_fairness
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah.dense_numeric_features import to_dense_numeric_al_zero_fill
from genova.pah.weighting import weights_fixed_minority

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
OUT_CSV = ROOT / "reports" / "tables" / "rf_imputation_fairness_check.csv"

RANDOM_STATE = 42
N_REPEATS = 10
B_LABEL = "B_fixed_minority_2x"


def fit_predict_rf_al_zero_fill(X_train, y_train, X_test, params):
    """`e2ek_models.py::fit_predict_rf` ile AYNI mimari/agirliklandirma
    (Strateji B) -- yalnizca doldurma stratejisi (`to_dense_numeric_
    al_zero_fill`) degisiyor."""
    X_train_d, X_test_d = to_dense_numeric_al_zero_fill(X_train, X_test)
    weights = weights_fixed_minority(y_train)
    model = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=8, **params)
    model.fit(X_train_d, y_train, sample_weight=weights)
    return model.predict_proba(X_test_d)[:, 1]


def _lookup_best_params_b(comparison_df, version_name, repeat_idx, outer_fold_idx):
    mask = (
        (comparison_df.model == "random_forest") & (comparison_df.data_version == version_name)
        & (comparison_df.weighting_variant == B_LABEL)
        & (comparison_df.repeat == repeat_idx) & (comparison_df.outer_fold == outer_fold_idx)
    )
    rows = comparison_df.loc[mask, "best_params"]
    assert len(rows) == 1, f"random_forest/{version_name} repeat={repeat_idx} outer={outer_fold_idx}: {len(rows)} satir"
    return json.loads(rows.iloc[0])


def _score_row(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_binary_positive(y_true, y_pred),
        "mcc": matthews_correlation_coefficient(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
    }


def rescore(v1_df, comparison_df, version_name, builder):
    rows = []
    for repeat_idx in range(N_REPEATS):
        outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer_fold in outer["folds"]:
            fold_idx = outer_fold["fold"]
            best_params = _lookup_best_params_b(comparison_df, version_name, repeat_idx, fold_idx)
            X_train, y_train, X_test, y_test = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            proba = fit_predict_rf_al_zero_fill(X_train, y_train, X_test, best_params)
            scores = _score_row(y_test, proba)
            rows.append({
                "model": "random_forest", "data_version": version_name, "imputation": "al_zero_fill",
                "repeat": repeat_idx, "outer_fold": fold_idx,
                "n_train": len(y_train), "n_test": len(y_test),
                "best_params": json.dumps(best_params), **scores,
            })
        print(f"  [random_forest/{version_name}] repeat={repeat_idx} tamam", flush=True)
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v2_raw_nan_builder(df, tr, te):
        return fv.build_v2_raw_nan(df, tr, te)

    combos = [("v1", v1_builder), ("v2_raw_nan", v2_raw_nan_builder)]

    all_rows = []
    for version_name, builder in combos:
        print(f"=== random_forest x {version_name} (AL_ sifir-doldurma, 50 dis fold, mevcut best_params) ===", flush=True)
        df = rescore(v1_df, comparison_df, version_name, builder)
        all_rows.append(df)
        print(df[["f1", "mcc", "specificity"]].agg(["mean", "std"]), flush=True)

    new_result = pd.concat(all_rows, ignore_index=True)

    old_rows = []
    for version_name in ("v1", "v2_raw_nan"):
        sub = comparison_df[
            (comparison_df.model == "random_forest") & (comparison_df.data_version == version_name)
            & (comparison_df.weighting_variant == B_LABEL)
        ].copy()
        sub["imputation"] = "median_fill_old"
        old_rows.append(sub[["model", "data_version", "imputation", "repeat", "outer_fold", "n_train", "n_test", "best_params", "f1", "mcc", "specificity"]])
    old_result = pd.concat(old_rows, ignore_index=True)

    result = pd.concat([old_result, new_result], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")

    print("\n=== eski (medyan) vs yeni (sifir) karsilastirma ===")
    print(result.groupby(["data_version", "imputation"])[["f1", "mcc", "specificity"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
