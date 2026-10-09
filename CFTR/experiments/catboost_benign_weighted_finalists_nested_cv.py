from pathlib import Path
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
df = pd.read_csv(ROOT / "YARISMA_TRAIN_CFTR_REDUCED.csv")
y = df["Label"].astype(int).reset_index(drop=True)
X0 = df.drop(columns=["Variant_ID", "Label"]).reset_index(drop=True)
MISSING_INDICATORS = os.environ.get("MISSING_INDICATORS") == "1"
ONLY_319 = os.environ.get("CATBOOST_ONLY_319") == "1"
OUTPUT_TAG = "_missing_indicators" if MISSING_INDICATORS else ""

groups = {
    "GRUP_AL5": ["AL_5", "AL_16", "AL_112", "AL_40", "AL_304"],
    "GRUP_AL6_AL251": ["AL_6", "AL_251"],
    "GRUP_AL4_AL38": ["AL_4", "AL_38"],
    "GRUP_AL1_AL211": ["AL_1", "AL_211"],
}

models = [
    (
        "scenario1_raw_322",
        "Senaryo 1A - 322 özellikli ham azaltılmış veri",
        [],
        {},
    ),
    (
        "scenario1_robust_groups",
        "Senaryo 1C - 320 özellikli güvenilir gruplar",
        ["AL_6", "AL_251", "AL_1", "AL_211"],
        {"GRUP_AL6_AL251": groups["GRUP_AL6_AL251"],
         "GRUP_AL1_AL211": groups["GRUP_AL1_AL211"]},
    ),
    (
        "scenario1_robust_groups_no_cat6",
        "Senaryo 1D - 319 özellikli güvenilir gruplar, CAT_6 çıkarılmış",
        ["CAT_6", "AL_6", "AL_251", "AL_1", "AL_211"],
        {"GRUP_AL6_AL251": groups["GRUP_AL6_AL251"],
         "GRUP_AL1_AL211": groups["GRUP_AL1_AL211"]},
    ),
    (
        "scenario2_composite_314",
        "Senaryo 2B - 314 özellikli bileşik Model 3",
        ["CAT_6", "AL_16", "AL_112", "AL_40", "AL_304", "AL_38",
         "AL_6", "AL_251", "AL_1", "AL_211"],
        {"GRUP_AL6_AL251": groups["GRUP_AL6_AL251"],
         "GRUP_AL1_AL211": groups["GRUP_AL1_AL211"]},
    ),
]
if ONLY_319:
    models = [m for m in models if m[0] == "scenario1_robust_groups_no_cat6"]


def fit_params(X, grp):
    params = {}
    for new, members in grp.items():
        params[new] = {}
        for col in members:
            s = pd.to_numeric(X[col], errors="coerce")
            median = float(s.median())
            scale = float(s.quantile(0.75) - s.quantile(0.25))
            if not np.isfinite(scale) or scale == 0:
                scale = float(s.std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            params[new][col] = (median, scale)
    return params


def transform(X, remove, grp, params):
    Z = X.drop(columns=remove).copy()
    for new, members in grp.items():
        values = []
        for col in members:
            median, scale = params[new][col]
            values.append((pd.to_numeric(X[col], errors="coerce") - median) / scale)
        Z[new] = pd.concat(values, axis=1).mean(axis=1, skipna=True)
    for col in Z.columns:
        if col.startswith("CAT_") or col.startswith("AA_"):
            Z[col] = Z[col].fillna("__MISSING__").astype(str)
    return Z


def metric_values(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision1 = tp / (tp + fp) if tp + fp else 0.0
    precision0 = tn / (tn + fn) if tn + fn else 0.0
    f1_1 = 2 * precision1 * recall / (precision1 + recall) if precision1 + recall else 0.0
    f1_0 = 2 * precision0 * specificity / (precision0 + specificity) if precision0 + specificity else 0.0
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": (tp + tn) / len(actual), "specificity": specificity,
        "recall": recall, "f1_1": f1_1, "f1_0": f1_0,
        "macro_f1": (f1_1 + f1_0) / 2,
        "mcc": matthews_corrcoef(actual, predicted),
    }


def fit_probability(X_train, y_train, X_test, seed):
    if MISSING_INDICATORS:
        X_train, X_test = X_train.copy(), X_test.copy()
        seen = set()
        for col in list(X_train.columns):
            mask = (X_train[col] == "__MISSING__") if X_train[col].dtype == object else X_train[col].isna()
            n_missing = int(mask.sum())
            if n_missing < 5 or len(X_train) - n_missing < 5:
                continue
            signature = tuple(mask.to_numpy(dtype=np.uint8))
            if signature in seen:
                continue
            seen.add(signature)
            name = f"MISS_{col}"
            X_train[name] = mask.astype(np.uint8)
            test_mask = ((X_test[col] == "__MISSING__") if X_test[col].dtype == object else X_test[col].isna())
            X_test[name] = test_mask.astype(np.uint8)
    cat_features = [i for i, c in enumerate(X_train.columns) if c.startswith("CAT_") or c.startswith("AA_")]
    model = CatBoostClassifier(
        iterations=40, depth=4, learning_rate=0.05, l2_leaf_reg=5,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=seed, verbose=False, allow_writing_files=False, thread_count=-1,
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    return model.predict_proba(X_test)[:, 1]


def choose_threshold(actual, probability):
    # Hedef test dağılımı: yaklaşık 100 benign ve 20 patojenik.
    # Önce bu dağılımdaki pozitif-sınıf F1, eşitlikte daha az FP ve daha yüksek recall seçilir.
    candidates = np.round(np.arange(0.20, 0.951, 0.01), 2)
    rows = []
    for threshold in candidates:
        m = metric_values(actual, (probability >= threshold).astype(int))
        projected_tp = 20 * m["recall"]
        projected_fp = 100 * (1 - m["specificity"])
        projected_precision = projected_tp / (projected_tp + projected_fp) if projected_tp + projected_fp else 0.0
        projected_f1 = (2 * projected_precision * m["recall"] / (projected_precision + m["recall"])
                        if projected_precision + m["recall"] else 0.0)
        rows.append((projected_f1, -projected_fp, m["recall"], threshold))
    return max(rows)[3]


def projected_test_metrics(m):
    projected_tp = 20 * m["recall"]
    projected_fp = 100 * (1 - m["specificity"])
    precision = projected_tp / (projected_tp + projected_fp) if projected_tp + projected_fp else 0.0
    f1 = 2 * precision * m["recall"] / (precision + m["recall"]) if precision + m["recall"] else 0.0
    return f1, projected_fp


def evaluate(key, label, remove, grp):
    predictions = []
    fold_rows = []
    inner_rows = []
    for repeat in range(5):
        outer = StratifiedKFold(5, shuffle=True, random_state=73000 + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X0, y), start=1):
            X_outer = X0.iloc[train_idx].reset_index(drop=True)
            y_outer = y.iloc[train_idx].reset_index(drop=True)
            inner_probability = np.zeros(len(train_idx))
            inner = StratifiedKFold(4, shuffle=True, random_state=83000 + repeat * 10 + fold)
            for inner_fold, (fit_idx, val_idx) in enumerate(inner.split(X_outer, y_outer), start=1):
                params = fit_params(X_outer.iloc[fit_idx], grp)
                X_fit = transform(X_outer.iloc[fit_idx], remove, grp, params)
                X_val = transform(X_outer.iloc[val_idx], remove, grp, params)
                inner_probability[val_idx] = fit_probability(
                    X_fit, y_outer.iloc[fit_idx], X_val,
                    930000 + repeat * 1000 + fold * 10 + inner_fold,
                )
            threshold = choose_threshold(y_outer.to_numpy(), inner_probability)
            inner_rows.extend({"key": key, "model": label, "repeat": repeat + 1,
                               "fold": fold, "row": int(train_idx[local_row]),
                               "actual": int(y_outer.iloc[local_row]),
                               "probability": float(inner_probability[local_row])}
                              for local_row in range(len(train_idx)))
            params = fit_params(X_outer, grp)
            X_train = transform(X_outer, remove, grp, params)
            X_test = transform(X0.iloc[test_idx], remove, grp, params)
            probability = fit_probability(X_train, y_outer, X_test, 1030000 + repeat * 100 + fold)
            predicted = (probability >= threshold).astype(int)
            actual = y.iloc[test_idx].to_numpy()
            m = metric_values(actual, predicted)
            fold_rows.append({"key": key, "model": label, "repeat": repeat + 1, "fold": fold,
                              "threshold": threshold, **m})
            predictions.extend({"key": key, "model": label, "repeat": repeat + 1,
                                "fold": fold, "row": int(idx), "actual": int(a),
                                "probability": float(p), "threshold": threshold,
                                "predicted": int(q)}
                               for idx, a, p, q in zip(test_idx, actual, probability, predicted))
    pred_df = pd.DataFrame(predictions)
    m = metric_values(pred_df["actual"], pred_df["predicted"])
    m["projected_f1"], m["fp_per_100_benign"] = projected_test_metrics(m)
    m["threshold_mean"] = pred_df.groupby(["repeat", "fold"])["threshold"].first().mean()
    m["threshold_sd"] = pred_df.groupby(["repeat", "fold"])["threshold"].first().std(ddof=1)
    base_count = transform(X0, remove, grp, fit_params(X0, grp)).shape[1]
    m["feature_count"] = base_count
    return m, fold_rows, predictions, inner_rows


summaries, all_folds, all_predictions, all_inner = [], [], [], []
for spec in models:
    m, folds, predictions, inner_rows = evaluate(*spec)
    summaries.append({"key": spec[0], "model": spec[1], **m})
    all_folds.extend(folds)
    all_predictions.extend(predictions)
    all_inner.extend(inner_rows)

score_df = pd.DataFrame(summaries)
score_df.to_csv(OUT / f"catboost_benign_weighted_finalists_scores{OUTPUT_TAG}.csv", index=False)
pd.DataFrame(all_folds).to_csv(OUT / f"catboost_benign_weighted_finalists_folds{OUTPUT_TAG}.csv", index=False)
pd.DataFrame(all_predictions).to_csv(OUT / f"catboost_benign_weighted_finalists_predictions{OUTPUT_TAG}.csv", index=False)
pd.DataFrame(all_inner).to_csv(OUT / f"catboost_benign_weighted_finalists_inner_predictions{OUTPUT_TAG}.csv", index=False)

winner = score_df.sort_values(["projected_f1", "specificity", "recall"], ascending=False).iloc[0]
lines = [
    "# Benign Ağırlıklı Nihai CatBoost Karşılaştırması", "",
    "320 ve 314 özellikli iki finalist, yaklaşık 100 benign + 20 patojenik hedef dağılımına göre karşılaştırılmıştır. "
    "Dış değerlendirme 5-fold stratified CV × 5 tekrar, karar eşiği seçimi ise her dış eğitim katında "
    "4-fold iç CV ile yapılmıştır. Bileşik özelliklerin medyan/IQR parametreleri de yalnızca ilgili eğitim "
    "katından öğrenilmiştir; dış test verisi eşik veya ön işleme seçiminde kullanılmamıştır.", "",
    "Eşik, iç CV'de önce hedef dağılımdaki projekte F1, eşitlikte daha az FP ve ardından daha yüksek recall ölçütüne göre seçilmiştir.", "",
    "| Model | Özellik | Eşik ort.±SS | Accuracy | Specificity | Recall | Macro-F1 | MCC | Proj. F1 | FP/100 benign |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for _, r in score_df.iterrows():
    lines.append(
        f"| {r['model']} | {int(r['feature_count'])} | {r['threshold_mean']:.3f}±{r['threshold_sd']:.3f} | "
        f"{r['accuracy']:.4f} | {r['specificity']:.4f} | {r['recall']:.4f} | "
        f"{r['macro_f1']:.4f} | {r['mcc']:.4f} | {r['projected_f1']:.4f} | {r['fp_per_100_benign']:.1f} |"
    )
lines += [
    "", "## Sonuç", "",
    f"Benign ağırlıklı karar kuralına göre en iyi aday **{winner['model']}** modelidir.", "",
    "Bu sonuç iç içe çapraz doğrulama ile elde edilen model-seçim sonucudur. Veri seti yalnızca 111 gözlem "
    "içerdiğinden bağımsız test verisi üzerindeki sonuç hâlâ nihai genelleme kanıtıdır; CV sonucu beklenen "
    "performansın tahminidir.",
]
(OUT / "BENIGN_AGIRLIKLI_NIHAI_CATBOOST_RAPORU.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
