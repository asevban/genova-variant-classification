"""Tamamlayıcı Deney Turu için sabit (tuning'siz) diagnostic modeller ve
metrikler. Bu modüldeki modeller hiçbir zaman final model adayı değildir --
yalnızca "hangi ön işleme daha iyi" karşılaştırmasında kullanılan, split
bankası boyunca hiperparametresi hiç değişmeyen sabit ölçüm araçlarıdır
(Aşama D.1'deki yardımcı LightGBM/elastic-net kullanımıyla aynı mantık).

Hiçbir fonksiyon burada F1/MCC/vb. değerini "final performans" olarak
sunmaz; hepsi bir ön işleme varyantını bir diğeriyle karşılaştırmak için
diagnostiktir.
"""
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


def fixed_models(seed=42):
    """Her çağrıda taze örnekler döner -- fold'lar arası durum sızıntısı olmaz."""
    return {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=seed),
        "Dummy": DummyClassifier(strategy="most_frequent"),
    }


def compute_metrics(y_true, y_pred, y_proba):
    """Pozitif sınıf (Label=1, patojenik) F1'i CLAUDE.md'nin resmi metriği;
    MCC, specificity ve AUPRC diagnostik tamamlayıcı ölçümler.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "f1_pathogenic": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "specificity": specificity,
        "auprc": average_precision_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
    }


def evaluate_variant(build_fn, split_files, variant_name, seed=42):
    """`build_fn(train_ids, test_ids) -> X_train, y_train, X_test, y_test`
    çağrılabilirini split bankasındaki her dış fold için çalıştırır, 3 sabit
    modeli eğitir/değerlendirir, sonucu fold-bazlı ve model-bazlı ortalanmış
    olarak döner.

    `split_files`: `outer_fold_repeat*.json` dosya yollarının listesi.
    """
    rows = []
    for split_file in split_files:
        outer = json.loads(split_file.read_text())
        for fold in outer["folds"]:
            X_train, y_train, X_test, y_test = build_fn(fold["train_variant_ids"], fold["test_variant_ids"])
            for model_name, model in fixed_models(seed).items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                m = compute_metrics(y_test, y_pred, y_proba)
                m.update({"variant": variant_name, "model": model_name,
                           "repeat": outer["repeat"], "fold": fold["fold"]})
                rows.append(m)
    return pd.DataFrame(rows)


def summarize(raw_df):
    """Fold/tekrar boyunca ortalama ± std, varyant × model bazında."""
    metrics = ["f1_pathogenic", "mcc", "specificity", "auprc"]
    g = raw_df.groupby(["variant", "model"])[metrics]
    summary = g.mean().round(4)
    summary.columns = [f"{c}_mean" for c in summary.columns]
    std = g.std().round(4)
    std.columns = [f"{c}_std" for c in std.columns]
    return summary.join(std).reset_index()
