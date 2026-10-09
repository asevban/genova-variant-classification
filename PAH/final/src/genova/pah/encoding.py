"""PAH paneli için kategorik kodlayıcılar. Fit, kategorileri/frekansları
yalnızca eğitim fold'undan öğrenir; transform hiçbir zaman yeniden fit
etmez, bu yüzden fold-içi çağırmak güvenlidir.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class NominalOneHotEncoder(BaseEstimator, TransformerMixin):
    """Düşük kardinaliteli nominal kolonları (CAT_3/4/5, AA_1/2) one-hot
    kodlar. Transform sırasında görülmeyen kategoriler (örn. eğitimde hiç
    görülmemiş bir validation-fold amino asit harfi), hata vermek yerine
    tamamen sıfır bir satıra eşlenir.

    >>> df_train = pd.DataFrame({"CAT_3": ["G/G", "T/T", "G/G"]})
    >>> enc = NominalOneHotEncoder(columns=["CAT_3"]).fit(df_train)
    >>> sorted(enc.transform(df_train).columns.tolist())
    ['CAT_3_G/G', 'CAT_3_T/T']
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", dtype=int)
        self.encoder_.fit(X[self.columns].astype(str))
        self.feature_names_ = self.encoder_.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        encoded = self.encoder_.transform(X[self.columns].astype(str)).toarray()
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        return pd.concat([X.drop(columns=self.columns), encoded_df], axis=1)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Yüksek kardinaliteli kategorikleri (CAT_1, CAT_2) eğitim-fold'daki
    göreli frekanslarına göre kodlar; strategy="native" ise CatBoost'un
    native işleyebilmesi için pandas "category" dtype olarak bırakır.
    Hiçbir zaman etikete-bağımlı bir kodlama fit etmez (y'ye hiç erişmez)
    -- naif target encoding'in sızıntı riskinden kaçınır.

    >>> df_train = pd.DataFrame({"CAT_2": ["AllofUs_EUR", "AllofUs_EUR", "AllofUs_AFR"]})
    >>> enc = FrequencyEncoder(columns=["CAT_2"]).fit(df_train)
    >>> enc.transform(df_train)["CAT_2"].round(3).tolist()
    [0.667, 0.667, 0.333]
    """

    def __init__(self, columns, strategy="frequency"):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy not in ("frequency", "native"):
            raise ValueError(f"gecersiz strategy: {self.strategy}")
        if self.strategy == "frequency":
            self.frequencies_ = {c: X[c].value_counts(normalize=True) for c in self.columns}
        return self

    def transform(self, X):
        out = X.copy()
        if self.strategy == "native":
            for col in self.columns:
                out[col] = out[col].astype("category")
            return out
        for col in self.columns:
            # once category dtype'dan cikar: bir Categorical, 0.0 kendi
            # mevcut kategorilerinden biri degilse .fillna(0.0)'i reddeder.
            out[col] = out[col].astype(object).map(self.frequencies_[col]).fillna(0.0).astype(float)
        return out


class MultiValueFrequencyEncoder(FrequencyEncoder):
    """`CAT_1` gibi bazı satırlarda `separator` ile birleşik çok-değerli
    kategoriler içeren yüksek kardinaliteli kolonlar için `FrequencyEncoder`'ın
    genişletilmiş hali (bkz. reports/01_EDA_RAPORU_PAH.md: `CAT_1`'de 5 satır
    `"gnomADe_AFR&gnomADe_AMR&...&gnomADe_SAS"` gibi birleşik değerler taşıyor).

    Tekil-değerli satırlar tabanda `FrequencyEncoder` ile **birebir aynı**
    davranır. Çok-değerli bir satır için, `separator`'a göre ayrıştırılan her
    bileşen etiketin **eğitim-fold'undaki tekil (single-valued satırlardan
    hesaplanan) frekansının ortalaması** kodlanmış değer olur. Görülmeyen bir
    bileşen 0.0 katkı sağlar (`FrequencyEncoder`'ın unseen-kategori
    davranışıyla tutarlı). `y`'ye hiç erişmez -- sızıntı riski yok.

    >>> df_train = pd.DataFrame({"CAT_1": ["gnomADe_NFE", "gnomADe_NFE", "gnomADe_AFR", "gnomADe_NFE&gnomADe_AFR"]})
    >>> enc = MultiValueFrequencyEncoder(columns=["CAT_1"]).fit(df_train)
    >>> enc.transform(df_train)["CAT_1"].round(3).tolist()
    [0.667, 0.667, 0.333, 0.5]
    """

    def __init__(self, columns, strategy="frequency", separator="&"):
        super().__init__(columns, strategy=strategy)
        self.separator = separator

    def fit(self, X, y=None):
        if self.strategy not in ("frequency", "native"):
            raise ValueError(f"gecersiz strategy: {self.strategy}")
        if self.strategy == "frequency":
            self.frequencies_ = {}
            for col in self.columns:
                values = X[col].astype(object).dropna().astype(str)
                singles = values[~values.str.contains(self.separator, regex=False)]
                self.frequencies_[col] = singles.value_counts(normalize=True)
        return self

    def transform(self, X):
        if self.strategy == "native":
            return super().transform(X)

        def encode(value):
            if pd.isna(value):
                return 0.0
            value = str(value)
            if self.separator in value:
                parts = value.split(self.separator)
                return float(np.mean([freqs.get(part, 0.0) for part in parts]))
            return float(freqs.get(value, 0.0))

        out = X.copy()
        for col in self.columns:
            freqs = self.frequencies_[col]
            out[col] = out[col].astype(object).map(encode).astype(float)
        return out
