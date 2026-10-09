"""P1 madde 9: A/C agirliklandirma karsilastirmasini XGBoost/LightGBM/
Elastic-Net'e uygular -- E2-EK'te yalnizca RF+CatBoost'a uygulanmisti, bu
uc model hic A/C gormedi, yalnizca Strateji B ile E2'de elendi. Amac YENI
bir final aday bulmak DEGIL, mevcut eleme kararinin adil test edilip
edilmedigini dogrulamak (denetim: "model ailesi farki ile agirliklandirma
farki birbirine karisiyor").

Yeni hiperparametre aramasi YOK -- `e2_model_comparison.csv`'nin ZATEN
kaydettigi (Strateji B) `best_params` aynen yeniden kullanildi. Veri
versiyonu E2'nin orijinal eslesmesiyle SINIRLI (gorevin kendi kapsam
karari): v1 (XGBoost/LightGBM), v3+v4_from_v3 (Elastic-Net) -- v2/
v2_raw_nan/v4_from_v2 bu turda test edilmedi.

KNN atlandi: `sample_weight` destegi yok (mesafe-tabanli), zaten
E2-EK'te cok zayif cikmisti (MCC=0.07) -- zorla uydurulmadi.

Sonuc YENI bir dosyaya yazildi: `e2_model_comparison.csv`'ye EKLENMEDI
(uzerine yazilmadi).

Calistirma: python -m genova.pah.e2_madde9_weighting_extension
"""
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah import models as m
from genova.pah.weighting import WEIGHTING_STRATEGIES

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
AL_CLUSTERS_PATH = ROOT / "reports" / "tables" / "AL_correlation_clusters.csv"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"
OUT_CSV = ROOT / "reports" / "tables" / "e2_madde9_weighting_extension.csv"

RANDOM_STATE = m.RANDOM_STATE
N_REPEATS = 10
B_LABEL = "B_fixed_minority_2x"
NEW_LABELS = ("A_no_weight", "C_data_driven_spw")


def make_fit_predict_xgb(weight_fn):
    """`models.py::fit_predict_xgb` ile AYNI mimari/hiperparametreler --
    yalnizca agirlik kaynagi sabit Strateji B yerine parametre."""
    def fit_predict(X_train, y_train, X_test, params):
        weights = weight_fn(y_train)
        model = XGBClassifier(
            n_estimators=200, min_child_weight=5, tree_method="hist",
            enable_categorical=True, random_state=RANDOM_STATE, n_jobs=8, **params,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict_proba(X_test)[:, 1]
    return fit_predict


def make_fit_predict_lgbm(weight_fn):
    """`models.py::fit_predict_lgbm` ile AYNI mimari -- yalnizca agirlik
    kaynagi parametre."""
    def fit_predict(X_train, y_train, X_test, params):
        weights = weight_fn(y_train)
        cat_cols = m._category_columns(X_train)
        model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=8, **params)
        model.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols if cat_cols else "auto")
        return model.predict_proba(X_test)[:, 1]
    return fit_predict


def make_fit_predict_elasticnet(weight_fn):
    """`models.py::fit_predict_elasticnet` ile AYNI mimari -- yalnizca
    agirlik kaynagi parametre."""
    def fit_predict(X_train, y_train, X_test, params):
        scaler = RobustScaler() if params["scaler"] == "robust" else StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        weights = weight_fn(y_train)
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=params["l1_ratio"], C=params["C"],
            max_iter=1000, tol=1e-3, random_state=RANDOM_STATE,
        )
        model.fit(X_train_s, y_train, sample_weight=weights)
        return model.predict_proba(X_test_s)[:, 1]
    return fit_predict


def _lookup_best_params_b(comparison_df, model_name, version_name, repeat_idx, outer_fold_idx):
    mask = (
        (comparison_df.model == model_name) & (comparison_df.data_version == version_name)
        & (comparison_df.weighting_variant == B_LABEL)
        & (comparison_df.repeat == repeat_idx) & (comparison_df.outer_fold == outer_fold_idx)
    )
    rows = comparison_df.loc[mask, "best_params"]
    assert len(rows) == 1, f"{model_name}/{version_name} repeat={repeat_idx} outer={outer_fold_idx}: {len(rows)} satir"
    return json.loads(rows.iloc[0])


def _score_row(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_binary_positive(y_true, y_pred),
        "mcc": matthews_correlation_coefficient(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
    }


def rescore(v1_df, comparison_df, model_name, version_name, builder, fit_predict_factory):
    rows = []
    for label in NEW_LABELS:
        weight_fn = WEIGHTING_STRATEGIES[label]
        fit_predict_fn = fit_predict_factory(weight_fn)
        for repeat_idx in range(N_REPEATS):
            outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
            for outer_fold in outer["folds"]:
                fold_idx = outer_fold["fold"]
                best_params = _lookup_best_params_b(comparison_df, model_name, version_name, repeat_idx, fold_idx)
                X_train, y_train, X_test, y_test = builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
                proba = fit_predict_fn(X_train, y_train, X_test, best_params)
                scores = _score_row(y_test, proba)
                rows.append({
                    "model": model_name, "data_version": version_name, "weighting_variant": label,
                    "repeat": repeat_idx, "outer_fold": fold_idx,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "best_params": json.dumps(best_params), **scores,
                })
            print(f"  [{model_name}/{version_name}/{label}] repeat={repeat_idx} tamam", flush=True)
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV)
    pool = json.loads(POOL_PATH.read_text())["features"]
    al_clusters = pd.read_csv(AL_CLUSTERS_PATH)

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v3_builder(df, tr, te):
        return fv.build_v3_full(df, tr, te, al_clusters)

    def v4v3_builder(df, tr, te):
        return fv.build_v4_from_v3(df, tr, te, pool)

    combos = [
        ("xgboost", "v1", v1_builder, make_fit_predict_xgb),
        ("lightgbm", "v1", v1_builder, make_fit_predict_lgbm),
        ("elasticnet", "v3", v3_builder, make_fit_predict_elasticnet),
        ("elasticnet", "v4_from_v3", v4v3_builder, make_fit_predict_elasticnet),
    ]

    all_rows = []
    for model_name, version_name, builder, factory in combos:
        print(f"=== {model_name} x {version_name} (A/C agirliklandirma, 50 dis fold, mevcut best_params) ===", flush=True)
        df = rescore(v1_df, comparison_df, model_name, version_name, builder, factory)
        all_rows.append(df)
        print(df.groupby("weighting_variant")[["f1", "mcc", "specificity"]].agg(["mean", "std"]), flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
