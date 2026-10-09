"""Asama E2: model portfoyu + nested (ic-fold hiperparametre secimi, dis-
fold olcum) egitim/degerlendirme. Hiperparametre aramasi HER ZAMAN yalnizca
bir dis fold'un kendi ic 4-fold'unda yapilir; dis fold yalnizca son skor
icin kullanilir (KESIN KURALLAR #2).

Ozellik matrisleri `fold_versions.py`'deki fold-guvenli insaacilardan gelir
-- `data/processed/pah/*.parquet` hicbir yerde dogrudan modele beslenmez.

Calistirma: python -m genova.pah.models
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from genova.metrics import f1_binary_positive, matthews_correlation_coefficient, specificity
from genova.pah import fold_versions as fv
from genova.pah.idempotent_io import upsert_csv

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
POOL_PATH = ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
AL_CLUSTERS_PATH = ROOT / "reports" / "tables" / "AL_correlation_clusters.csv"
COMPARISON_CSV = ROOT / "reports" / "tables" / "e2_model_comparison.csv"

N_REPEATS = 10
DECISION_THRESHOLD = 0.5  # E2 = model karsilastirma; esik secimi E5'in isi.
MINORITY_WEIGHT = 2.0
RANDOM_STATE = 42


def _minority_sample_weight(y_train, weight=MINORITY_WEIGHT):
    minority_class = y_train.value_counts().idxmin()
    return np.where(y_train == minority_class, weight, 1.0)


def _category_columns(X):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


# ---------------------------------------------------------------- modeller

def fit_predict_xgb(X_train, y_train, X_test, params):
    weights = _minority_sample_weight(y_train)
    model = XGBClassifier(
        n_estimators=200, min_child_weight=5, tree_method="hist",
        enable_categorical=True, random_state=RANDOM_STATE, n_jobs=8, **params,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict_proba(X_test)[:, 1]


def fit_predict_lgbm(X_train, y_train, X_test, params):
    weights = _minority_sample_weight(y_train)
    cat_cols = _category_columns(X_train)
    model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=8, **params)
    model.fit(X_train, y_train, sample_weight=weights, categorical_feature=cat_cols if cat_cols else "auto")
    return model.predict_proba(X_test)[:, 1]


def fit_predict_catboost(X_train, y_train, X_test, params):
    weights = _minority_sample_weight(y_train)
    cat_cols = _category_columns(X_train)

    def _stringify(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(object).fillna("__MISSING__").astype(str)
        return X

    model = CatBoostClassifier(
        iterations=100, cat_features=cat_cols, random_seed=RANDOM_STATE,
        verbose=False, thread_count=8, allow_writing_files=False, **params,
    )
    model.fit(_stringify(X_train), y_train, sample_weight=weights)
    return model.predict_proba(_stringify(X_test))[:, 1]


def fit_predict_elasticnet(X_train, y_train, X_test, params):
    scaler = RobustScaler() if params["scaler"] == "robust" else StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    weights = _minority_sample_weight(y_train)
    model = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=params["l1_ratio"], C=params["C"],
        max_iter=1000, tol=1e-3, random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train, sample_weight=weights)
    return model.predict_proba(X_test_s)[:, 1]


# -------------------------------------------------------------- nested CV

def _load_outer_fold(repeat_idx, fold_idx):
    outer = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
    return outer["folds"][fold_idx]


def _load_inner_folds(repeat_idx, outer_fold_idx):
    inner = json.loads((SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{outer_fold_idx}.json").read_text())
    return inner["folds"]


def _score_row(y_true, proba, threshold=DECISION_THRESHOLD):
    y_pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_binary_positive(y_true, y_pred),
        "mcc": matthews_correlation_coefficient(y_true, y_pred),
        "auprc": average_precision_score(y_true, proba),
        "specificity": specificity(y_true, y_pred),
    }


def nested_cv_evaluate(v1_df, feature_builder, fit_predict_fn, param_grid, model_name, version_name, n_repeats=N_REPEATS):
    """Her dis fold icin: ic 4-fold'da param_grid'in her adayini dene (ic
    ortalama F1'e gore en iyisini sec), ardindan yalnizca o adayla dis
    train'de fit edip dis test'te olc. `feature_builder(v1_df, train_ids,
    test_ids) -> (X_train, y_train, X_test, y_test)` imzasina uyar.
    """
    rows = []
    for repeat_idx in range(n_repeats):
        n_outer = len(json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())["folds"])
        for outer_fold_idx in range(n_outer):
            t0 = time.time()
            outer_fold = _load_outer_fold(repeat_idx, outer_fold_idx)
            inner_folds = _load_inner_folds(repeat_idx, outer_fold_idx)

            inner_mean_f1 = []
            for params in param_grid:
                f1s = []
                for inner_fold in inner_folds:
                    X_tr, y_tr, X_va, y_va = feature_builder(v1_df, inner_fold["train_variant_ids"], inner_fold["val_variant_ids"])
                    proba = fit_predict_fn(X_tr, y_tr, X_va, params)
                    f1s.append(f1_binary_positive(y_va, (proba >= DECISION_THRESHOLD).astype(int)))
                inner_mean_f1.append(np.mean(f1s))
            best_idx = int(np.argmax(inner_mean_f1))
            best_params = param_grid[best_idx]

            X_tr, y_tr, X_te, y_te = feature_builder(v1_df, outer_fold["train_variant_ids"], outer_fold["test_variant_ids"])
            proba = fit_predict_fn(X_tr, y_tr, X_te, best_params)
            scores = _score_row(y_te, proba)
            rows.append({
                "model": model_name, "data_version": version_name,
                "repeat": repeat_idx, "outer_fold": outer_fold_idx,
                "n_train": len(y_tr), "n_test": len(y_te),
                "best_params": json.dumps(best_params), "inner_best_f1": inner_mean_f1[best_idx],
                "seconds": time.time() - t0, **scores,
            })
            print(f"  [{model_name}/{version_name}] repeat={repeat_idx} outer={outer_fold_idx} "
                  f"f1={scores['f1']:.4f} mcc={scores['mcc']:.4f} auprc={scores['auprc']:.4f} "
                  f"spec={scores['specificity']:.4f} ({rows[-1]['seconds']:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(results_df):
    out = {}
    for metric in ("f1", "mcc", "auprc", "specificity"):
        out[metric + "_mean"] = results_df[metric].mean()
        out[metric + "_std"] = results_df[metric].std()
        out[metric + "_min"] = results_df[metric].min()
        out[metric + "_max"] = results_df[metric].max()
    return out


def append_results(results_df):
    """P2 madde 19: idempotent yazar (`idempotent_io.upsert_csv`) --
    ayni (model,data_version,repeat,outer_fold) icin script iki kez
    calistirilirsa satir COGALMAZ, eskisi yeni sonucla degistirilir."""
    upsert_csv(results_df, COMPARISON_CSV, ["model", "data_version", "repeat", "outer_fold"])


# ------------------------------------------------------- hiperparametre izgaralari (dar)

CATBOOST_GRID = [{"depth": d, "learning_rate": 0.05} for d in (3, 5)]
XGB_GRID = [{"max_depth": d, "learning_rate": lr} for d in (3, 5) for lr in (0.03, 0.1)]
LGBM_GRID = [{"num_leaves": n, "learning_rate": lr} for n in (7, 15) for lr in (0.03, 0.1)]
ELASTICNET_GRID = [
    {"C": c, "l1_ratio": l1, "scaler": s}
    for c in (0.1, 1.0) for l1 in (0.3, 0.7) for s in ("robust", "standard")
]


def main():
    v1_df = pd.read_parquet(V1_PATH)
    pool = json.loads(POOL_PATH.read_text())["features"]
    al_clusters = pd.read_csv(AL_CLUSTERS_PATH)

    def v1_builder(df, tr, te):
        return fv.build_v1(df, tr, te)

    def v2_builder(df, tr, te):
        return fv.build_v2(df, tr, te)

    def v4v2_builder(df, tr, te):
        return fv.build_v4_from_v2(df, tr, te, pool)

    def v3_builder(df, tr, te):
        return fv.build_v3_full(df, tr, te, al_clusters)

    def v4v3_builder(df, tr, te):
        return fv.build_v4_from_v3(df, tr, te, pool)

    tree_versions = [("v1", v1_builder), ("v2", v2_builder), ("v4_from_v2", v4v2_builder)]
    linear_versions = [("v3", v3_builder), ("v4_from_v3", v4v3_builder)]

    combos = []
    for version_name, builder in tree_versions:
        combos.append(("catboost", version_name, builder, fit_predict_catboost, CATBOOST_GRID))
        combos.append(("xgboost", version_name, builder, fit_predict_xgb, XGB_GRID))
        combos.append(("lightgbm", version_name, builder, fit_predict_lgbm, LGBM_GRID))
    for version_name, builder in linear_versions:
        combos.append(("elasticnet", version_name, builder, fit_predict_elasticnet, ELASTICNET_GRID))

    for model_name, version_name, builder, fit_predict_fn, grid in combos:
        print(f"=== {model_name} x {version_name} ({len(grid)} aday, 50 dis fold) ===", flush=True)
        t0 = time.time()
        results = nested_cv_evaluate(v1_df, builder, fit_predict_fn, grid, model_name, version_name)
        append_results(results)
        print(f"--- {model_name} x {version_name} bitti ({time.time() - t0:.0f}s): {summarize(results)}", flush=True)


if __name__ == "__main__":
    main()
