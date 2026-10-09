"""PAH paneli için eksiklik işleme: blok göstergeleri, seçici kolon-bazlı
göstergeler ve doldurucular. Tüm sınıflar sklearn-tarzı (fit/transform)
olduğu için nested CV içinde fold-içi çağrılabilir -- fit yalnızca
kendisine verilen eğitim fold'una bakar, başka hiçbir şeye bakmaz.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BlockMissingIndicator(BaseEstimator, TransformerMixin):
    """"Bu bloktaki tüm kolonlar eksik" için tek bir 0/1 gösterge ekler.

    Kolon doğrulaması dışında durumsuz -- satır-bazlı tüm-NaN olması yapısal
    bir gerçektir, eğitim verisine fit edilecek bir şey değildir.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"AL_1": [1.0, np.nan], "AL_2": [2.0, np.nan]})
    >>> BlockMissingIndicator(columns=["AL_1", "AL_2"], name="al_all_missing").fit_transform(df)["al_all_missing"].tolist()
    [0, 1]
    """

    def __init__(self, columns, name):
        self.columns = columns
        self.name = name

    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"BlockMissingIndicator: kolon(lar) bulunamadi: {missing}")
        self.fitted_ = True  # durumsuz donusturucu icin sklearn check_is_fitted() uyumlulugu
        return self

    def transform(self, X):
        out = X.copy()
        out[self.name] = X[self.columns].isna().all(axis=1).astype(int)
        return out


class SelectedMissingIndicators(BaseEstimator, TransformerMixin):
    """Açıkça verilen bir kolon listesi için kolon-bazlı 0/1 eksiklik
    göstergesi ekler (örn. EDA'nın reports/tables/missingness_label_
    association.csv'de bilgilendirici bulduğu kolonlar) -- asla 334 AL_
    kolonunun tamamına körü körüne değil.

    >>> df = pd.DataFrame({"EK_3": [1.0, np.nan]})
    >>> out = SelectedMissingIndicators(columns=["EK_3"]).fit_transform(df)
    >>> out["EK_3_missing"].tolist()
    [0, 1]
    """

    def __init__(self, columns, suffix="_missing"):
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"SelectedMissingIndicators: kolon(lar) bulunamadi: {missing}")
        return self

    def transform(self, X):
        out = X.copy()
        for col in self.columns:
            out[f"{col}{self.suffix}"] = X[col].isna().astype(int)
        return out


class ConstantFillImputer(BaseEstimator, TransformerMixin):
    """Verilen kolonlardaki NaN'ları bir sabitle (varsayılan 0.0) doldurur,
    veya strategy="none" iken ham NaN olarak bırakır -- v2/v3'ün aynı kod
    yolundan hem sıfır-dolu hem ham-NaN varyantını üretmesini sağlar.

    >>> df = pd.DataFrame({"AL_1": [1.0, np.nan]})
    >>> ConstantFillImputer(columns=["AL_1"], strategy="zero").fit_transform(df)["AL_1"].tolist()
    [1.0, 0.0]
    >>> ConstantFillImputer(columns=["AL_1"], strategy="none").fit_transform(df)["AL_1"].tolist()[1]
    nan
    """

    def __init__(self, columns, strategy="zero", fill_value=0.0):
        self.columns = columns
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        if self.strategy not in ("zero", "none"):
            raise ValueError(f"gecersiz strategy: {self.strategy}")
        self.fitted_ = True  # durumsuz donusturucu icin sklearn check_is_fitted() uyumlulugu
        return self

    def transform(self, X):
        if self.strategy == "none":
            return X.copy()
        out = X.copy()
        out[self.columns] = out[self.columns].fillna(self.fill_value)
        return out


class MedianImputerWithIndicator(BaseEstimator, TransformerMixin):
    """Bir eksiklik göstergesi ekler, ardından verilen kolonları medyanla
    doldurur. Medyan yalnızca fit()'e verilen eğitim fold'undan öğrenilir;
    sızıntısız bir pipeline'da fit'in validation/test verisinde çağrılması
    hiçbir zaman olmaz.

    >>> df_train = pd.DataFrame({"EK_3": [1.0, 3.0, np.nan]})
    >>> imputer = MedianImputerWithIndicator(columns=["EK_3"]).fit(df_train)
    >>> imputer.transform(df_train)["EK_3"].tolist()
    [1.0, 3.0, 2.0]
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.medians_ = {c: X[c].median() for c in self.columns}
        return self

    def transform(self, X):
        out = X.copy()
        for col in self.columns:
            out[f"{col}_missing"] = X[col].isna().astype(int)
            out[col] = out[col].fillna(self.medians_[col])
        return out
