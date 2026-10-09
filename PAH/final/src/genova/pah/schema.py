"""PAH paneli ham-CSV şema ve sentinel doğrulaması.

`validate_schema`/`scan_sentinels` saf kontrollerdir -> düz fonksiyonlar.
`SchemaGate`, bu kontrolü bir sklearn `Pipeline`'ın ilk adımı olarak
kullanabilmek için ince bir fit/transform sarmalayıcısıdır (Tamamlayıcı
Deney Turu, Görev 7) -- veriyi değiştirmez, yalnızca geçersiz şemada hata
fırlatır.
"""
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REQUIRED_COLUMNS = ("Variant_ID", "Label")
EXPECTED_GROUP_COUNTS = {"AL": 334, "CAT": 6, "EK": 9, "AA": 2}
SENTINEL_VALUES = (-999, -9999, 9999, 999, -1)


def _group_of(column: str) -> str:
    match = re.match(r"([A-Za-z]+)_", column)
    return match.group(1) if match else column


def validate_schema(df: pd.DataFrame, expected_group_counts=None) -> dict:
    """Zorunlu kolonları, Label değer kümesini ve AL_/CAT_/EK_/AA_ grup sayılarını kontrol eder.

    "ok" bool'u ve bulunan "issues" listesini içeren bir rapor dict'i döner;
    hiçbir zaman hata fırlatmaz, çağıran taraf nasıl devam edeceğine karar
    vermeden önce loglayabilir/inceleyebilir.

    `expected_group_counts` verilmezse ham CSV şeması (`EXPECTED_GROUP_COUNTS`)
    kullanılır. v1 sonrası veri (örn. `CAT_6` düşürülmüş) gibi farklı bir şemayı
    doğrulamak için çağıran taraf kendi beklenen sayılarını verebilir.

    >>> report = validate_schema(pd.DataFrame({"Variant_ID": ["a"], "Label": [1], "AL_1": [0.1]}))
    >>> report["ok"]
    False
    """
    expected_group_counts = expected_group_counts if expected_group_counts is not None else EXPECTED_GROUP_COUNTS
    issues = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        issues.append(f"eksik zorunlu kolon(lar): {missing_required}")

    if "Variant_ID" in df.columns and not df["Variant_ID"].is_unique:
        issues.append("Variant_ID benzersiz degil")

    if "Label" in df.columns:
        bad_labels = set(df["Label"].dropna().unique()) - {0, 1}
        if bad_labels:
            issues.append(f"Label {{0,1}} disinda deger iceriyor: {bad_labels}")
        if df["Label"].isna().any():
            issues.append("Label icinde null deger var")

    group_counts = {}
    for col in df.columns:
        if col in REQUIRED_COLUMNS:
            continue
        group_counts[_group_of(col)] = group_counts.get(_group_of(col), 0) + 1
    for group, expected in expected_group_counts.items():
        actual = group_counts.get(group, 0)
        if actual != expected:
            issues.append(f"{group}_ grubu {actual} kolon, beklenen {expected}")

    return {"ok": len(issues) == 0, "issues": issues, "group_counts": group_counts}


def scan_sentinels(df: pd.DataFrame, sentinels=SENTINEL_VALUES) -> dict:
    """Sayısal kolonları literal sentinel eksik-değer kodları için tarar.

    Sentinel'lerin olmadığını varsaymaz -- gerçekten tarar ve hangi kolonda
    hangi sentinel değerlerinin kaç kez geçtiğini raporlar.

    >>> scan_sentinels(pd.DataFrame({"AL_1": [0.1, -999]}))
    {'AL_1': {-999: 1}}
    """
    numeric_cols = df.select_dtypes(include="number").columns
    hits = {}
    for col in numeric_cols:
        col_hits = {}
        for sentinel in sentinels:
            count = int((df[col] == sentinel).sum())
            if count > 0:
                col_hits[sentinel] = count
        if col_hits:
            hits[col] = col_hits
    return hits


class SchemaGate(BaseEstimator, TransformerMixin):
    """`validate_schema`'yı bir `Pipeline` adımına çevirir: veriyi hiç
    değiştirmez, yalnızca `fit`/`transform` sırasında şema geçersizse
    `ValueError` fırlatır -- pipeline'ın ilk adımı olarak kullanılmak
    üzere tasarlandı (Tamamlayıcı Deney Turu, Görev 7).

    `expected_group_counts` verilmezse ham CSV şeması varsayılır. v1 ve
    sonrasını (CAT_6 zaten düşürülmüş, 6 değil 5 CAT_ kolonu) girdi alan
    pipeline'lar bu parametreyi v1'in gerçek şemasıyla vermelidir --
    aksi halde ham-CSV şemasına göre yanlışlıkla başarısız olur.

    >>> SchemaGate().fit_transform(pd.DataFrame({"Variant_ID": ["a"], "Label": [0]})).shape[0]
    Traceback (most recent call last):
        ...
    ValueError: SchemaGate: sema gecersiz: ['AL_ grubu 0 kolon, beklenen 334', 'CAT_ grubu 0 kolon, beklenen 6', 'EK_ grubu 0 kolon, beklenen 9', 'AA_ grubu 0 kolon, beklenen 2']
    """

    def __init__(self, expected_group_counts=None):
        self.expected_group_counts = expected_group_counts

    def fit(self, X, y=None):
        report = validate_schema(X, expected_group_counts=self.expected_group_counts)
        if not report["ok"]:
            raise ValueError(f"SchemaGate: sema gecersiz: {report['issues']}")
        self.fitted_ = True  # durumsuz donusturucu icin sklearn check_is_fitted() uyumlulugu
        return self

    def transform(self, X):
        return X
