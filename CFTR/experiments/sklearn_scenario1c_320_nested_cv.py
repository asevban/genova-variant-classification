from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
df = pd.read_csv(ROOT / "YARISMA_TRAIN_CFTR_REDUCED.csv")
y = df["Label"].astype(int).reset_index(drop=True)
X0 = df.drop(columns=["Variant_ID", "Label"]).reset_index(drop=True)
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "320"
if SCENARIO not in {"320", "319"}:
    raise ValueError("Senaryo 320 veya 319 olmalıdır")

GROUPS = {
    "GRUP_AL6_AL251": ["AL_6", "AL_251"],
    "GRUP_AL1_AL211": ["AL_1", "AL_211"],
}
REMOVE = ["AL_6", "AL_251", "AL_1", "AL_211"] + (["CAT_6"] if SCENARIO == "319" else [])

outer_source = pd.read_csv(OUT / "catboost_benign_weighted_finalists_predictions.csv")
outer_source = outer_source[outer_source["key"] == "scenario1_robust_groups"]
outer_tests = {(int(rep), int(fold)): group["row"].astype(int).to_numpy()
               for (rep, fold), group in outer_source.groupby(["repeat", "fold"])}


def fit_group_params(frame):
    params = {}
    for new_name, members in GROUPS.items():
        params[new_name] = {}
        for col in members:
            series = pd.to_numeric(frame[col], errors="coerce")
            median = float(series.median())
            scale = float(series.quantile(0.75) - series.quantile(0.25))
            if not np.isfinite(scale) or scale == 0:
                scale = float(series.std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            params[new_name][col] = (median, scale)
    return params


def transform(frame, params):
    result = frame.drop(columns=REMOVE).copy()
    for new_name, members in GROUPS.items():
        values = []
        for col in members:
            median, scale = params[new_name][col]
            values.append((pd.to_numeric(frame[col], errors="coerce") - median) / scale)
        result[new_name] = pd.concat(values, axis=1).mean(axis=1, skipna=True)
    return result


def estimator(name, transformed_frame, seed):
    categorical = [c for c in transformed_frame.columns if c.startswith("CAT_") or c.startswith("AA_")]
    numeric = [c for c in transformed_frame.columns if c not in categorical]
    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    if name == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=seed)
    elif name == "Extra Trees":
        model = ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def fit_predict_probability(name, x_train, y_train, x_test, seed):
    params = fit_group_params(x_train)
    train_320 = transform(x_train, params)
    test_320 = transform(x_test, params)
    pipe = estimator(name, train_320, seed)
    pipe.fit(train_320, y_train)
    return pipe.predict_proba(test_320)[:, 1]


def metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    p1 = tp / (tp + fp) if tp + fp else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * p1 * recall / (p1 + recall) if p1 + recall else 0.0
    f10 = 2 * p0 * specificity / (p0 + specificity) if p0 + specificity else 0.0
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=(tp + tn) / len(actual), specificity=specificity,
                recall=recall, f1=f1, f1_0=f10, macro_f1=(f1 + f10) / 2,
                mcc=matthews_corrcoef(actual, predicted))


def projected(m):
    tp = 20 * m["recall"]
    fp = 100 * (1 - m["specificity"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * m["recall"] / (precision + m["recall"]) if precision + m["recall"] else 0.0
    return f1, fp


def choose_threshold(actual, probability):
    scored = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        m = metrics(actual, (probability >= threshold).astype(int))
        pf1, pfp = projected(m)
        scored.append((pf1, -pfp, m["recall"], threshold))
    return max(scored)[3]


def evaluate(name):
    predictions, fold_rows, inner_rows = [], [], []
    for repeat in range(1, 6):
        for fold in range(1, 6):
            test_idx = outer_tests[(repeat, fold)]
            test_set = set(test_idx)
            train_idx = np.array([i for i in range(len(X0)) if i not in test_set])
            x_outer_train = X0.iloc[train_idx].reset_index(drop=True)
            y_outer_train = y.iloc[train_idx].reset_index(drop=True)
            inner_probability = np.zeros(len(train_idx))
            inner = StratifiedKFold(4, shuffle=True, random_state=83000 + (repeat - 1) * 10 + fold)
            for inner_fold, (fit_idx, val_idx) in enumerate(inner.split(x_outer_train, y_outer_train), start=1):
                inner_probability[val_idx] = fit_predict_probability(
                    name,
                    x_outer_train.iloc[fit_idx], y_outer_train.iloc[fit_idx],
                    x_outer_train.iloc[val_idx],
                    120000 + repeat * 1000 + fold * 10 + inner_fold,
                )
            threshold = choose_threshold(y_outer_train.to_numpy(), inner_probability)
            inner_rows.extend({"model": name, "key": f"scenario1_{SCENARIO}", "repeat": repeat, "fold": fold,
                               "row": int(train_idx[local]), "actual": int(y_outer_train.iloc[local]),
                               "probability": float(inner_probability[local])}
                              for local in range(len(train_idx)))

            probability = fit_predict_probability(
                name, X0.iloc[train_idx], y.iloc[train_idx], X0.iloc[test_idx],
                220000 + repeat * 100 + fold,
            )
            actual = y.iloc[test_idx].to_numpy()
            predicted = (probability >= threshold).astype(int)
            m = metrics(actual, predicted)
            pf1, pfp = projected(m)
            fold_rows.append({"model": name, "key": f"scenario1_{SCENARIO}", "repeat": repeat, "fold": fold,
                              "threshold": threshold, **m, "projected_f1": pf1,
                              "fp_per_100_benign": pfp})
            predictions.extend({"model": name, "key": f"scenario1_{SCENARIO}", "repeat": repeat, "fold": fold,
                                "row": int(row), "actual": int(a), "probability": float(p),
                                "threshold": threshold, "predicted": int(q)}
                               for row, a, p, q in zip(test_idx, actual, probability, predicted))
    pred = pd.DataFrame(predictions)
    m = metrics(pred.actual, pred.predicted)
    pf1, pfp = projected(m)
    thresholds = pred.groupby(["repeat", "fold"])["threshold"].first()
    return ({"model": name, "feature_count": int(SCENARIO), **m, "projected_f1": pf1,
             "fp_per_100_benign": pfp, "threshold_mean": thresholds.mean(),
             "threshold_sd": thresholds.std(ddof=1)}, fold_rows, predictions, inner_rows)


summaries, folds, predictions, inner_predictions = [], [], [], []
for model_name in ["AdaBoost", "Extra Trees"]:
    summary, model_folds, model_predictions, model_inner = evaluate(model_name)
    summaries.append(summary)
    folds.extend(model_folds)
    predictions.extend(model_predictions)
    inner_predictions.extend(model_inner)

scores = pd.DataFrame(summaries).sort_values(["projected_f1", "specificity", "recall"], ascending=False)
prefix = f"sklearn_scenario1_{SCENARIO}_nested"
scores.to_csv(OUT / f"{prefix}_scores.csv", index=False)
pd.DataFrame(folds).to_csv(OUT / f"{prefix}_folds.csv", index=False)
pd.DataFrame(predictions).to_csv(OUT / f"{prefix}_predictions.csv", index=False)
pd.DataFrame(inner_predictions).to_csv(OUT / f"{prefix}_inner_predictions.csv", index=False)
print(scores.to_string(index=False))
