"""Leakage-safe baseline modeling benchmark for the MASTER panel.

The benchmark deliberately refits ``MasterPreprocessor`` inside every
evaluation fold.  Materialized scenario CSV files are not used for CV.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - surfaced in runtime metadata
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = repr(exc)
else:
    XGBOOST_IMPORT_ERROR = ""


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    STRATEGIES,
    MasterPreprocessor,
    load_master_csv,
)


OUT_DIR = ROOT / "results" / "modeling"
RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
OUTER_SPLIT_FILE = ROOT / "artifacts" / "master_outer_split.csv"
FOLD_FILE = ROOT / "artifacts" / "master_repeated_cv_folds.csv"
FEATURE_SETS = ["M2_compact", "M3_missing_aware_compact"]
THRESHOLD = 0.50
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    supports_spw: bool = False
    class_weight: str | None = None


MODEL_SPECS = [
    ModelSpec("Dummy_most_frequent", "dummy"),
    ModelSpec("Logistic_L2_scaled", "logistic"),
    ModelSpec("Logistic_L2_scaled_balanced", "logistic", class_weight="balanced"),
    ModelSpec("RandomForest", "random_forest"),
    ModelSpec("XGBoost_no_weight", "xgboost"),
    ModelSpec("XGBoost_fold_spw", "xgboost", supports_spw=True),
]


def _make_model(spec: ModelSpec, seed: int, scale_pos_weight: float | None = None):
    if spec.family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if spec.family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                class_weight=spec.class_weight,
                random_state=seed,
            ),
        )
    if spec.family == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=spec.class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    if spec.family == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError(f"xgboost import failed: {XGBOOST_IMPORT_ERROR}")
        params: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": 4,
        }
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model family: {spec.family}")


def _positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            cls = int(model.classes_[0]) if hasattr(model, "classes_") else 0
            return np.ones(len(x), dtype=float) if cls == 1 else np.zeros(len(x), dtype=float)
        classes = list(model.classes_)
        return proba[:, classes.index(1)]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def _metric_row(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
    row: dict[str, float | int] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        row["auroc"] = float(roc_auc_score(y_true, y_score))
        row["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        row["auroc"] = float("nan")
        row["auprc"] = float("nan")
    return row


def _aggregate(group: pd.DataFrame) -> pd.Series:
    metric_cols = [
        "f1",
        "mcc",
        "specificity",
        "sensitivity",
        "precision",
        "balanced_accuracy",
        "auprc",
        "auroc",
        "fpr",
        "brier",
        "log_loss",
        "fp",
        "fn",
        "output_features",
        "fit_seconds",
    ]
    payload: dict[str, Any] = {
        "folds": int(len(group)),
        "mean_spw": float(group["scale_pos_weight"].mean()),
    }
    for column in metric_cols:
        payload[f"{column}_mean"] = float(group[column].mean())
        payload[f"{column}_std"] = float(group[column].std(ddof=1))
    return pd.Series(payload)


def _optimise_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "mcc",
) -> dict[str, float]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for threshold in thresholds:
        metrics = _metric_row(y_true, y_score, float(threshold))
        rows.append({"threshold": float(threshold), metric: float(metrics[metric])})
    best = max(rows, key=lambda item: (item[metric], -abs(item["threshold"] - 0.5)))
    return {"threshold": float(best["threshold"]), metric: float(best[metric])}


def _robustness_rows(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prevalences = [0.50, 0.60, 0.70, 0.80]
    repeats = 30
    rng = np.random.default_rng(20260815)
    for (feature_set, model), group in oof.groupby(["feature_set", "model"]):
        best = _optimise_threshold(
            group[TARGET_COLUMN].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
            metric="mcc",
        )
        for benign_rate in prevalences:
            per_sample = []
            benign = group.loc[group[TARGET_COLUMN] == 0]
            pathogenic = group.loc[group[TARGET_COLUMN] == 1]
            n_total = min(
                len(group),
                int(len(benign) / benign_rate),
                int(len(pathogenic) / (1.0 - benign_rate)),
            )
            n_benign = int(round(n_total * benign_rate))
            n_pathogenic = n_total - n_benign
            if n_benign <= 0 or n_pathogenic <= 0:
                continue
            for _ in range(repeats):
                b_idx = rng.choice(benign.index.to_numpy(), size=n_benign, replace=False)
                p_idx = rng.choice(
                    pathogenic.index.to_numpy(), size=n_pathogenic, replace=False
                )
                sample = group.loc[np.concatenate([b_idx, p_idx])]
                metrics = _metric_row(
                    sample[TARGET_COLUMN].to_numpy(dtype=int),
                    sample["score"].to_numpy(dtype=float),
                    best["threshold"],
                )
                per_sample.append(metrics)
            sample_frame = pd.DataFrame(per_sample)
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "optimised_metric": "mcc",
                    "threshold": best["threshold"],
                    "benign_rate": benign_rate,
                    "n_total": int(n_total),
                    "f1_mean": float(sample_frame["f1"].mean()),
                    "f1_std": float(sample_frame["f1"].std(ddof=1)),
                    "mcc_mean": float(sample_frame["mcc"].mean()),
                    "mcc_std": float(sample_frame["mcc"].std(ddof=1)),
                    "specificity_mean": float(sample_frame["specificity"].mean()),
                    "specificity_std": float(sample_frame["specificity"].std(ddof=1)),
                    "sensitivity_mean": float(sample_frame["sensitivity"].mean()),
                    "sensitivity_std": float(sample_frame["sensitivity"].std(ddof=1)),
                    "fp_mean": float(sample_frame["fp"].mean()),
                    "fp_std": float(sample_frame["fp"].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def main() -> dict[str, Any]:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_master_csv(RAW_FILE)
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={ID_COLUMN: "string"})

    train_ids = set(outer.loc[outer["partition"].eq("train"), ID_COLUMN])
    train = raw.loc[raw[ID_COLUMN].isin(train_ids)].reset_index(drop=True)
    if len(train) != 2345:
        raise ValueError(f"Unexpected outer-train size: {len(train)}")

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    total_jobs = len(FEATURE_SETS) * len(MODEL_SPECS) * folds["repeat_seed"].nunique() * 5
    job_no = 0

    for feature_set in FEATURE_SETS:
        for seed in sorted(folds["repeat_seed"].unique()):
            seed_folds = folds.loc[folds["repeat_seed"].eq(seed)]
            for fold in sorted(seed_folds["fold"].unique()):
                hold_ids = set(seed_folds.loc[seed_folds["fold"].eq(fold), ID_COLUMN])
                fit_frame = train.loc[~train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)
                hold_frame = train.loc[train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)

                processor = MasterPreprocessor(STRATEGIES[feature_set])
                x_fit = processor.fit_transform(fit_frame)
                x_hold = processor.transform(hold_frame)
                y_fit = fit_frame[TARGET_COLUMN].to_numpy(dtype=int)
                y_hold = hold_frame[TARGET_COLUMN].to_numpy(dtype=int)
                neg = int((y_fit == 0).sum())
                pos = int((y_fit == 1).sum())
                fold_spw = float(neg / pos)

                for spec in MODEL_SPECS:
                    if spec.family == "xgboost" and XGBClassifier is None:
                        continue
                    job_no += 1
                    model_start = time.time()
                    scale_pos_weight = fold_spw if spec.supports_spw else None
                    model = _make_model(spec, int(seed), scale_pos_weight)
                    fitted = clone(model).fit(x_fit, y_fit)
                    score = _positive_probability(fitted, x_hold)
                    fit_seconds = time.time() - model_start
                    metrics = _metric_row(y_hold, score, THRESHOLD)
                    fold_rows.append(
                        {
                            "feature_set": feature_set,
                            "model": spec.name,
                            "repeat_seed": int(seed),
                            "fold": int(fold),
                            "fit_n": int(len(fit_frame)),
                            "holdout_n": int(len(hold_frame)),
                            "output_features": int(x_fit.shape[1]),
                            "scale_pos_weight": scale_pos_weight
                            if scale_pos_weight is not None
                            else 1.0,
                            "fit_seconds": float(fit_seconds),
                            **metrics,
                        }
                    )
                    oof_rows.append(
                        pd.DataFrame(
                            {
                                ID_COLUMN: hold_frame[ID_COLUMN].astype(str).to_numpy(),
                                TARGET_COLUMN: y_hold,
                                "score": score.astype(float),
                                "feature_set": feature_set,
                                "model": spec.name,
                                "repeat_seed": int(seed),
                                "fold": int(fold),
                            }
                        )
                    )
                    print(
                        f"[{job_no}/{total_jobs}] {feature_set} | {spec.name} "
                        f"| seed={seed} fold={fold} f1={metrics['f1']:.4f} "
                        f"mcc={metrics['mcc']:.4f} spec={metrics['specificity']:.4f}",
                        flush=True,
                    )

    fold_results = pd.DataFrame(fold_rows)
    summary = (
        fold_results.groupby(["feature_set", "model"], sort=False)
        .apply(_aggregate, include_groups=False)
        .reset_index()
        .sort_values(
            ["mcc_mean", "f1_mean", "specificity_mean", "auprc_mean"],
            ascending=[False, False, False, False],
        )
    )
    oof = pd.concat(oof_rows, ignore_index=True)
    robustness = _robustness_rows(oof)
    metadata = {
        "status": "completed",
        "feature_sets": FEATURE_SETS,
        "models": [spec.name for spec in MODEL_SPECS],
        "threshold": THRESHOLD,
        "fit_scope": "MasterPreprocessor refit inside each outer-train CV fold",
        "outer_validation_used_for_selection": False,
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "fold_results": "results/modeling/master_modeling_fold_results.csv",
            "summary": "results/modeling/master_modeling_summary.csv",
            "oof_predictions": "results/modeling/master_modeling_oof_predictions.csv",
            "robustness": "results/modeling/master_modeling_robustness.csv",
            "metadata": "results/modeling/master_modeling_metadata.json",
        },
    }

    fold_results.to_csv(OUT_DIR / "master_modeling_fold_results.csv", index=False)
    summary.to_csv(OUT_DIR / "master_modeling_summary.csv", index=False)
    oof.to_csv(OUT_DIR / "master_modeling_oof_predictions.csv", index=False)
    robustness.to_csv(OUT_DIR / "master_modeling_robustness.csv", index=False)
    (OUT_DIR / "master_modeling_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    display_cols = [
        "feature_set",
        "model",
        "folds",
        "f1_mean",
        "mcc_mean",
        "specificity_mean",
        "sensitivity_mean",
        "precision_mean",
        "auprc_mean",
        "auroc_mean",
        "fp_mean",
        "fn_mean",
    ]
    print(summary.loc[:, display_cols].to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
