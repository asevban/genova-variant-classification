"""PAH paneli için ölçek-duyarlı dönüşümler: frekans-tipi AL_ kolonları
için log1p, oran-tipi [0,1] AL_ kolonları için logit, native ölçekleri
farklı olan EK_ kolonları için rank/quantile harmonizasyonu + robust
ölçekleme. Bu ayrı bir pipeline kolu -- ağaç modelleri, CLAUDE.md'nin
"ağaç modelleri hattına karıştırılmayacak" kuralı gereği bunun yerine
dönüştürülmemiş (doldurulmuş) kolonları kullanır.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Sağa çarpık, sıfır-şişkin frekans-tipi AL_ kolonları için log1p.
    Fit edilen parametre yok -- log1p sabit bir fonksiyon -- ama pipeline
    uyumluluğu için fit/transform arayüzü korunuyor.

    >>> Log1pTransformer(columns=["AL_1"]).fit_transform(pd.DataFrame({"AL_1": [0.0, np.e - 1]})).round(3)["AL_1"].tolist()
    [0.0, 1.0]
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out[self.columns] = np.log1p(out[self.columns].clip(lower=0))
        return out


class LogitTransformer(BaseEstimator, TransformerMixin):
    """[0, 1] ile sınırlı oran-tipi AL_ kolonları için epsilon-ofsetli logit;
    tam 0/1 değerleri aksi halde log(x/(1-x))'i +/-inf'e gönderirdi.

    >>> LogitTransformer(columns=["AL_1"], eps=1e-3).fit_transform(pd.DataFrame({"AL_1": [0.5]}))["AL_1"].round(3).tolist()
    [0.0]
    """

    def __init__(self, columns, eps=1e-6):
        self.columns = columns
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        clipped = out[self.columns].clip(lower=self.eps, upper=1 - self.eps)
        out[self.columns] = np.log(clipped / (1 - clipped))
        return out


class RankQuantileHarmonizer(BaseEstimator, TransformerMixin):
    """Farklı native ölçeklerde yaşayan (bazıları [0,1], bazıları sınırsız)
    EK_ kolonlarını, eğitim fold'undan öğrenilen kolon-bazlı quantile
    rankleri üzerinden ortak bir uniform [0,1] ölçeğine harmonize eder.

    >>> df_train = pd.DataFrame({"EK_1": [1.0, 2.0, 3.0, 4.0]})
    >>> harmonizer = RankQuantileHarmonizer(columns=["EK_1"]).fit(df_train)
    >>> harmonizer.transform(df_train)["EK_1"].round(2).tolist()
    [0.0, 0.33, 0.67, 1.0]
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.transformers_ = {}
        for col in self.columns:
            n_quantiles = min(1000, X[col].notna().sum())
            qt = QuantileTransformer(n_quantiles=max(n_quantiles, 2), output_distribution="uniform")
            qt.fit(X[[col]].dropna())
            self.transformers_[col] = qt
        return self

    def transform(self, X):
        out = X.copy()
        for col in self.columns:
            mask = out[col].notna()
            out.loc[mask, col] = self.transformers_[col].transform(out.loc[mask, [col]]).ravel()
        return out


class ColumnwiseScaler(BaseEstimator, TransformerMixin):
    """sklearn'in RobustScaler/StandardScaler/QuantileTransformer'ını bir
    DataFrame kolon alt-kümesi üzerine sarmalar; yalnızca eğitim fold'unda
    fit edilir, kolon isimlerini korur.

    >>> df_train = pd.DataFrame({"EK_7": [-1.0, 0.0, 1.0, 10.0]})
    >>> scaler = ColumnwiseScaler(columns=["EK_7"], method="robust").fit(df_train)
    >>> scaler.transform(df_train)["EK_7"].round(2).tolist()
    [-0.43, -0.14, 0.14, 2.71]
    """

    def __init__(self, columns, method="robust"):
        self.columns = columns
        self.method = method

    def fit(self, X, y=None):
        if self.method == "robust":
            self.scaler_ = RobustScaler()
        elif self.method == "standard":
            self.scaler_ = StandardScaler()
        elif self.method == "quantile":
            self.scaler_ = QuantileTransformer(output_distribution="normal")
        else:
            raise ValueError(f"gecersiz method: {self.method}")
        self.scaler_.fit(X[self.columns])
        return self

    def transform(self, X):
        out = X.copy()
        out[self.columns] = self.scaler_.transform(out[self.columns])
        return out


def classify_al_columns(df, al_columns, ratio_type_min_frac_0_or_1=0.05):
    """AL_ kolonlarını, verinin yapısal özelliklerine dayanarak sabit /
    oran-tipi([0,1]) / frekans-tipi diye ayırır (etikete bağlı değil, bu
    yüzden tam veri setinde veya herhangi bir fold'da sızıntısız çağrılabilir).

    Üç kolon-ismi listesi içeren bir dict döner: "constant", "ratio_type",
    "frequency_type".

    >>> df = pd.DataFrame({"AL_1": [1.0, 1.0], "AL_2": [0.0, 1.0], "AL_3": [0.01, 5.0]})
    >>> result = classify_al_columns(df, ["AL_1", "AL_2", "AL_3"])
    >>> result["constant"], result["ratio_type"], result["frequency_type"]
    (['AL_1'], ['AL_2'], ['AL_3'])
    """
    constant, ratio_type, frequency_type = [], [], []
    for col in al_columns:
        series = df[col]
        if series.nunique(dropna=True) <= 1:
            constant.append(col)
            continue
        in_unit_interval = series.min(skipna=True) >= 0 and series.max(skipna=True) <= 1
        frac_0_or_1 = ((series == 0) | (series == 1)).mean()
        if in_unit_interval and frac_0_or_1 > ratio_type_min_frac_0_or_1:
            ratio_type.append(col)
        else:
            frequency_type.append(col)
    return {"constant": constant, "ratio_type": ratio_type, "frequency_type": frequency_type}
