"""Asama E2-EK: Random Forest ve KNN model sarmalayicilari + dar
hiperparametre izgaralari -- `models.py`'nin nested_cv_evaluate'iyle
dogrudan uyumlu (X_train, y_train, X_test, params) -> proba imzasi.

Agirliklandirma: RF Tier 1'de sabit Strateji B (azinlik x2,0,
`weighting.py::weights_fixed_minority` -- `models.py::_minority_sample_
weight` ile matematiksel olarak ayni formul, E2'nin mevcut agac-modeli
konvansiyonuyla birebir tutarli olsun diye burada da ayni kaynaktan
alindi). Tier 2'nin A/B/C karsilastirmasi icin `make_fit_predict_rf` /
`make_fit_predict_catboost` fabrika fonksiyonlari, `weighting.py::
WEIGHTING_STRATEGIES`'ten herhangi bir stratejiyi enjekte edebilir.

KNN, sklearn'de `sample_weight` desteklemez (mesafe-tabanli bir yontem,
ornek-agirlikli kayip fonksiyonu yok) -- bu yuzden Tier 2'nin agirlik
karsilastirmasina hic dahil edilmedi (gorev kapsamiyla tutarli).
"""
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from genova.pah.dense_numeric_features import to_dense_numeric
from genova.pah.weighting import weights_fixed_minority

RANDOM_STATE = 42

RF_GRID = [{"max_depth": d, "min_samples_leaf": m} for d in (5, 10) for m in (1, 5)]
KNN_GRID = [
    {"k": k, "weights": w, "scaler": s}
    for k in (5, 15) for w in ("uniform", "distance") for s in ("standard", "minmax")
]


def fit_predict_rf(X_train, y_train, X_test, params):
    """Tier 1: RF, sabit Strateji B (azinlik x2,0)."""
    return make_fit_predict_rf(weights_fixed_minority)(X_train, y_train, X_test, params)


def make_fit_predict_rf(weight_fn):
    """Tier 2: RF, verilen agirliklandirma stratejisiyle (A/B/C)."""
    def fit_predict(X_train, y_train, X_test, params):
        X_train_d, X_test_d = to_dense_numeric(X_train, X_test)
        weights = weight_fn(y_train)
        model = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=8, **params,
        )
        model.fit(X_train_d, y_train, sample_weight=weights)
        return model.predict_proba(X_test_d)[:, 1]
    return fit_predict


def _category_columns(X):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def _stringify_categoricals(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).fillna("__MISSING__").astype(str)
    return X


def make_fit_predict_catboost(weight_fn):
    """Tier 2: CatBoost, verilen agirliklandirma stratejisiyle (A/B/C) --
    `models.py::fit_predict_catboost` ile ayni mimari/hiperparametreler,
    yalnizca agirlik kaynagi degisiyor."""
    def fit_predict(X_train, y_train, X_test, params):
        weights = weight_fn(y_train)
        cat_cols = _category_columns(X_train)
        model = CatBoostClassifier(
            iterations=100, cat_features=cat_cols, random_seed=RANDOM_STATE,
            verbose=False, thread_count=8, allow_writing_files=False, **params,
        )
        model.fit(_stringify_categoricals(X_train, cat_cols), y_train, sample_weight=weights)
        return model.predict_proba(_stringify_categoricals(X_test, cat_cols))[:, 1]
    return fit_predict


def fit_predict_knn(X_train, y_train, X_test, params):
    """KNN, sample_weight desteklemedigi icin agirliklandirmasiz."""
    X_train_d, X_test_d = to_dense_numeric(X_train, X_test)
    scaler = StandardScaler() if params["scaler"] == "standard" else MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train_d)
    X_test_s = scaler.transform(X_test_d)
    model = KNeighborsClassifier(n_neighbors=params["k"], weights=params["weights"], metric="euclidean")
    model.fit(X_train_s, y_train)
    return model.predict_proba(X_test_s)[:, 1]
