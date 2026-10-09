"""Tuned MASTER modeling experiments for the final benign-heavy scenario.

This script is a model-development search, not a new preprocessing step.
Outer validation is not read or used.  Every candidate uses the frozen
outer-train fold bank and refits ``MasterPreprocessor`` only on the current
fold-training rows.
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
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


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


RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
OUTER_SPLIT_FILE = ROOT / "artifacts" / "master_outer_split.csv"
FOLD_FILE = ROOT / "artifacts" / "master_repeated_cv_folds.csv"
OUT_DIR = ROOT / "results" / "modeling" / "tuned"
FINAL_PATHOGENIC_RATE = 500 / 3500
DEFAULT_THRESHOLD = 0.50
RANDOM_STATE = 20260815


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str
    branch: str
    params: dict[str, Any]
    spw_multiplier: float | None = None


def xgb_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 350,
        "max_depth": 3,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
    }
    params.update(overrides)
    return params


def rf_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 450,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(overrides)
    return params


def build_candidates() -> list[Candidate]:
    feature_m3 = "M3_missing_aware_compact"
    feature_m2 = "M2_compact"
    candidates: list[Candidate] = [
        Candidate(
            "M3_XGB_NW_baseline_plus",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(n_estimators=300, learning_rate=0.05, colsample_bytree=0.90),
        ),
        Candidate(
            "M3_XGB_NW_shallow_reg",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(
                n_estimators=550,
                max_depth=2,
                learning_rate=0.025,
                min_child_weight=5,
                gamma=0.05,
                reg_alpha=0.10,
                reg_lambda=5.0,
                colsample_bytree=0.80,
            ),
        ),
        Candidate(
            "M3_XGB_NW_medium",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=2,
                reg_lambda=3.0,
                colsample_bytree=0.80,
            ),
        ),
        Candidate(
            "M3_XGB_NW_regularized",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(
                n_estimators=450,
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=6,
                gamma=0.10,
                reg_alpha=0.25,
                reg_lambda=7.0,
                subsample=0.85,
                colsample_bytree=0.75,
            ),
        ),
        Candidate(
            "M3_XGB_NW_depth4",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                min_child_weight=4,
                gamma=0.05,
                reg_lambda=4.0,
                subsample=0.85,
                colsample_bytree=0.75,
            ),
        ),
        Candidate(
            "M3_XGB_NW_low_fp_reg",
            "xgboost",
            feature_m3,
            "xgb_no_weight",
            xgb_params(
                n_estimators=650,
                max_depth=2,
                learning_rate=0.025,
                min_child_weight=8,
                gamma=0.20,
                reg_alpha=0.50,
                reg_lambda=9.0,
                subsample=0.85,
                colsample_bytree=0.70,
            ),
        ),
    ]

    spw_base_configs = [
        ("baseline", xgb_params(n_estimators=300, learning_rate=0.05, colsample_bytree=0.90)),
        (
            "regularized",
            xgb_params(
                n_estimators=500,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=6,
                gamma=0.10,
                reg_alpha=0.25,
                reg_lambda=6.0,
                subsample=0.85,
                colsample_bytree=0.75,
            ),
        ),
    ]
    for base_name, params in spw_base_configs:
        for multiplier in [0.50, 0.75, 1.00, 1.25, 1.50]:
            candidates.append(
                Candidate(
                    f"M3_XGB_SPW_{base_name}_x{multiplier:g}",
                    "xgboost",
                    feature_m3,
                    "xgb_spw_grid",
                    params,
                    spw_multiplier=multiplier,
                )
            )

    rf_candidates = [
        ("RF_baseline", rf_params()),
        ("RF_leaf4", rf_params(n_estimators=550, min_samples_leaf=4, min_samples_split=8)),
        (
            "RF_depth12",
            rf_params(n_estimators=550, max_depth=12, min_samples_leaf=3, min_samples_split=8),
        ),
        (
            "RF_depth8_balanced",
            rf_params(
                n_estimators=650,
                max_depth=8,
                min_samples_leaf=3,
                min_samples_split=8,
                class_weight="balanced_subsample",
            ),
        ),
    ]
    for feature_set in [feature_m3, feature_m2]:
        prefix = "M3" if feature_set == feature_m3 else "M2"
        for name, params in rf_candidates:
            candidates.append(
                Candidate(
                    f"{prefix}_{name}",
                    "random_forest",
                    feature_set,
                    "random_forest",
                    params,
                )
            )
    return candidates


def make_model(candidate: Candidate, fold_spw: float):
    params = dict(candidate.params)
    if candidate.family == "xgboost":
        if candidate.spw_multiplier is not None:
            params["scale_pos_weight"] = float(fold_spw * candidate.spw_multiplier)
        return XGBClassifier(**params)
    if candidate.family == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unknown family: {candidate.family}")


def positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.ones(len(x), dtype=float) if int(model.classes_[0]) == 1 else np.zeros(len(x))
    classes = list(model.classes_)
    return proba[:, classes.index(1)].astype(float)


def metric_row(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
    row = {
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


def final_prevalence_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    pathogenic_rate: float,
) -> dict[str, Any]:
    positive = y_score[y_true == 1]
    negative = y_score[y_true == 0]
    tpr = float((positive >= threshold).mean()) if len(positive) else 0.0
    fpr = float((negative >= threshold).mean()) if len(negative) else 0.0
    pi = pathogenic_rate
    tp = pi * tpr
    fn = pi * (1.0 - tpr)
    fp = (1.0 - pi) * fpr
    tn = (1.0 - pi) * (1.0 - fpr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom else 0.0
    return {
        "threshold": float(threshold),
        "final_f1": float(f1),
        "final_mcc": float(mcc),
        "final_specificity": float(1.0 - fpr),
        "final_sensitivity": float(tpr),
        "final_precision": float(precision),
        "expected_fp_per_3500": float(3500 * fp),
        "expected_fn_per_3500": float(3500 * fn),
    }


def best_final_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.001), 3)
    rows = [
        final_prevalence_metrics(y_true, y_score, float(threshold), FINAL_PATHOGENIC_RATE)
        for threshold in thresholds
    ]
    table = pd.DataFrame(rows)
    best_f1 = table.sort_values(
        ["final_f1", "final_mcc", "final_specificity"], ascending=False
    ).iloc[0]
    best_mcc = table.sort_values(
        ["final_mcc", "final_f1", "final_specificity"], ascending=False
    ).iloc[0]
    return {
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1_final_f1": float(best_f1["final_f1"]),
        "best_f1_final_mcc": float(best_f1["final_mcc"]),
        "best_f1_specificity": float(best_f1["final_specificity"]),
        "best_f1_sensitivity": float(best_f1["final_sensitivity"]),
        "best_f1_precision": float(best_f1["final_precision"]),
        "best_f1_expected_fp_per_3500": float(best_f1["expected_fp_per_3500"]),
        "best_f1_expected_fn_per_3500": float(best_f1["expected_fn_per_3500"]),
        "best_mcc_threshold": float(best_mcc["threshold"]),
        "best_mcc_final_f1": float(best_mcc["final_f1"]),
        "best_mcc_final_mcc": float(best_mcc["final_mcc"]),
        "best_mcc_specificity": float(best_mcc["final_specificity"]),
        "best_mcc_sensitivity": float(best_mcc["final_sensitivity"]),
        "best_mcc_precision": float(best_mcc["final_precision"]),
        "best_mcc_expected_fp_per_3500": float(best_mcc["expected_fp_per_3500"]),
        "best_mcc_expected_fn_per_3500": float(best_mcc["expected_fn_per_3500"]),
    }


def prepare_fold_cache(feature_sets: list[str]) -> dict[str, list[dict[str, Any]]]:
    raw = load_master_csv(RAW_FILE)
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={ID_COLUMN: "string"})
    train_ids = set(outer.loc[outer["partition"].eq("train"), ID_COLUMN])
    train = raw.loc[raw[ID_COLUMN].isin(train_ids)].reset_index(drop=True)
    cache: dict[str, list[dict[str, Any]]] = {feature_set: [] for feature_set in feature_sets}

    for feature_set in feature_sets:
        for seed in sorted(folds["repeat_seed"].unique()):
            seed_folds = folds.loc[folds["repeat_seed"].eq(seed)]
            for fold in sorted(seed_folds["fold"].unique()):
                hold_ids = set(seed_folds.loc[seed_folds["fold"].eq(fold), ID_COLUMN])
                fit_frame = train.loc[~train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)
                hold_frame = train.loc[train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)
                processor = MasterPreprocessor(STRATEGIES[feature_set])
                x_fit = processor.fit_transform(fit_frame).to_numpy(dtype=np.float32)
                x_hold = processor.transform(hold_frame).to_numpy(dtype=np.float32)
                y_fit = fit_frame[TARGET_COLUMN].to_numpy(dtype=int)
                y_hold = hold_frame[TARGET_COLUMN].to_numpy(dtype=int)
                cache[feature_set].append(
                    {
                        "repeat_seed": int(seed),
                        "fold": int(fold),
                        "x_fit": x_fit,
                        "x_hold": x_hold,
                        "y_fit": y_fit,
                        "y_hold": y_hold,
                        "hold_ids": hold_frame[ID_COLUMN].astype(str).to_numpy(),
                        "output_features": int(x_fit.shape[1]),
                        "fold_spw": float((y_fit == 0).sum() / (y_fit == 1).sum()),
                    }
                )
        print(
            f"Prepared {feature_set}: {len(cache[feature_set])} fold datasets",
            flush=True,
        )
    return cache


def aggregate_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "f1",
        "mcc",
        "specificity",
        "sensitivity",
        "precision",
        "balanced_accuracy",
        "auprc",
        "auroc",
        "fp",
        "fn",
        "brier",
        "log_loss",
        "fit_seconds",
    ]
    rows = []
    for keys, group in fold_results.groupby(
        ["candidate", "branch", "feature_set", "family"], sort=False
    ):
        row = dict(zip(["candidate", "branch", "feature_set", "family"], keys))
        row["folds"] = int(len(group))
        row["output_features_mean"] = float(group["output_features"].mean())
        for column in metric_cols:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in oof.groupby(["candidate", "branch", "feature_set", "family"]):
        avg = group.groupby([ID_COLUMN, TARGET_COLUMN], as_index=False)["score"].mean()
        y = avg[TARGET_COLUMN].to_numpy(dtype=int)
        score = avg["score"].to_numpy(dtype=float)
        default_metrics = metric_row(y, score, DEFAULT_THRESHOLD)
        threshold_metrics = best_final_threshold(y, score)
        rows.append(
            {
                "candidate": keys[0],
                "branch": keys[1],
                "feature_set": keys[2],
                "family": keys[3],
                "oof_unique_ids": int(len(avg)),
                "default_oof_f1": default_metrics["f1"],
                "default_oof_mcc": default_metrics["mcc"],
                "default_oof_specificity": default_metrics["specificity"],
                "default_oof_sensitivity": default_metrics["sensitivity"],
                "default_oof_fp": default_metrics["fp"],
                "default_oof_fn": default_metrics["fn"],
                **threshold_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["best_f1_final_f1", "best_f1_final_mcc", "best_f1_specificity"],
        ascending=False,
    )


def build_ensembles(oof: pd.DataFrame, candidate_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_by_branch: dict[str, str] = {}
    for branch in ["xgb_no_weight", "xgb_spw_grid", "random_forest"]:
        eligible = candidate_summary.loc[candidate_summary["branch"].eq(branch)]
        if len(eligible):
            top_by_branch[branch] = str(eligible.iloc[0]["candidate"])

    ensemble_defs: list[tuple[str, dict[str, float]]] = []
    nw = top_by_branch.get("xgb_no_weight")
    spw = top_by_branch.get("xgb_spw_grid")
    rf = top_by_branch.get("random_forest")
    if nw and spw:
        ensemble_defs.append(("ENS_XGB_NW_SPW_mean", {nw: 0.5, spw: 0.5}))
    if nw and rf:
        ensemble_defs.append(("ENS_XGB_NW_RF_mean", {nw: 0.5, rf: 0.5}))
        ensemble_defs.append(("ENS_XGB_NW70_RF30", {nw: 0.7, rf: 0.3}))
    if nw and spw and rf:
        ensemble_defs.append(("ENS_XGB_NW_SPW_RF_mean", {nw: 1 / 3, spw: 1 / 3, rf: 1 / 3}))
        ensemble_defs.append(("ENS_XGB_NW60_SPW20_RF20", {nw: 0.6, spw: 0.2, rf: 0.2}))

    averaged = {
        candidate: group.groupby([ID_COLUMN, TARGET_COLUMN], as_index=False)["score"].mean()
        for candidate, group in oof.groupby("candidate")
    }
    ensemble_oof_rows = []
    ensemble_summary_rows = []
    for name, weights in ensemble_defs:
        base = None
        for candidate, weight in weights.items():
            scores = averaged[candidate].rename(columns={"score": candidate})
            if base is None:
                base = scores
            else:
                base = base.merge(scores, on=[ID_COLUMN, TARGET_COLUMN], how="inner")
            base[candidate] = base[candidate] * weight
        if base is None:
            continue
        score_cols = list(weights)
        base["score"] = base[score_cols].sum(axis=1)
        y = base[TARGET_COLUMN].to_numpy(dtype=int)
        score = base["score"].to_numpy(dtype=float)
        default_metrics = metric_row(y, score, DEFAULT_THRESHOLD)
        threshold_metrics = best_final_threshold(y, score)
        ensemble_summary_rows.append(
            {
                "candidate": name,
                "branch": "ensemble",
                "feature_set": "+".join(sorted(set(candidate_summary.set_index("candidate").loc[list(weights), "feature_set"]))),
                "family": "ensemble",
                "members": json.dumps(weights, ensure_ascii=False),
                "oof_unique_ids": int(len(base)),
                "default_oof_f1": default_metrics["f1"],
                "default_oof_mcc": default_metrics["mcc"],
                "default_oof_specificity": default_metrics["specificity"],
                "default_oof_sensitivity": default_metrics["sensitivity"],
                "default_oof_fp": default_metrics["fp"],
                "default_oof_fn": default_metrics["fn"],
                **threshold_metrics,
            }
        )
        ensemble_oof_rows.append(
            pd.DataFrame(
                {
                    ID_COLUMN: base[ID_COLUMN].astype(str),
                    TARGET_COLUMN: y,
                    "score": score,
                    "candidate": name,
                    "branch": "ensemble",
                    "feature_set": "ensemble",
                    "family": "ensemble",
                }
            )
        )

    ensemble_oof = (
        pd.concat(ensemble_oof_rows, ignore_index=True)
        if ensemble_oof_rows
        else pd.DataFrame()
    )
    ensemble_summary = pd.DataFrame(ensemble_summary_rows)
    if len(ensemble_summary):
        ensemble_summary = ensemble_summary.sort_values(
            ["best_f1_final_f1", "best_f1_final_mcc", "best_f1_specificity"],
            ascending=False,
        )
    return ensemble_oof, ensemble_summary


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    feature_sets = sorted(set(candidate.feature_set for candidate in candidates))
    cache = prepare_fold_cache(feature_sets)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_started = time.time()
        for fold_data in cache[candidate.feature_set]:
            model = make_model(candidate, fold_data["fold_spw"])
            fit_started = time.time()
            model.fit(fold_data["x_fit"], fold_data["y_fit"])
            score = positive_probability(model, fold_data["x_hold"])
            fit_seconds = time.time() - fit_started
            metrics = metric_row(fold_data["y_hold"], score, DEFAULT_THRESHOLD)
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "branch": candidate.branch,
                    "family": candidate.family,
                    "feature_set": candidate.feature_set,
                    "repeat_seed": fold_data["repeat_seed"],
                    "fold": fold_data["fold"],
                    "output_features": fold_data["output_features"],
                    "fold_spw": fold_data["fold_spw"],
                    "spw_multiplier": candidate.spw_multiplier
                    if candidate.spw_multiplier is not None
                    else 1.0,
                    "effective_spw": fold_data["fold_spw"] * candidate.spw_multiplier
                    if candidate.spw_multiplier is not None
                    else 1.0,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )
            oof_rows.append(
                pd.DataFrame(
                    {
                        ID_COLUMN: fold_data["hold_ids"],
                        TARGET_COLUMN: fold_data["y_hold"],
                        "score": score,
                        "candidate": candidate.name,
                        "branch": candidate.branch,
                        "feature_set": candidate.feature_set,
                        "family": candidate.family,
                        "repeat_seed": fold_data["repeat_seed"],
                        "fold": fold_data["fold"],
                    }
                )
            )
        elapsed = time.time() - candidate_started
        print(
            f"[{idx}/{len(candidates)}] {candidate.name} completed in {elapsed:.1f}s",
            flush=True,
        )

    fold_results = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    fold_summary = aggregate_fold_results(fold_results)
    candidate_summary = summarize_oof(oof)
    ensemble_oof, ensemble_summary = build_ensembles(oof, candidate_summary)
    combined_summary = pd.concat([candidate_summary, ensemble_summary], ignore_index=True)
    combined_summary = combined_summary.sort_values(
        ["best_f1_final_f1", "best_f1_final_mcc", "best_f1_specificity"],
        ascending=False,
    )

    fold_results.to_csv(OUT_DIR / "master_tuned_fold_results.csv", index=False)
    fold_summary.to_csv(OUT_DIR / "master_tuned_fold_summary.csv", index=False)
    oof.to_csv(OUT_DIR / "master_tuned_oof_predictions.csv", index=False)
    candidate_summary.to_csv(OUT_DIR / "master_tuned_candidate_summary.csv", index=False)
    ensemble_oof.to_csv(OUT_DIR / "master_tuned_ensemble_oof_predictions.csv", index=False)
    ensemble_summary.to_csv(OUT_DIR / "master_tuned_ensemble_summary.csv", index=False)
    combined_summary.to_csv(OUT_DIR / "master_tuned_combined_summary.csv", index=False)

    metadata = {
        "status": "completed",
        "candidate_count": len(candidates),
        "folds_per_candidate": 25,
        "final_pathogenic_rate_assumption": FINAL_PATHOGENIC_RATE,
        "final_benign_rate_assumption": 1.0 - FINAL_PATHOGENIC_RATE,
        "outer_validation_used": False,
        "fit_scope": "MasterPreprocessor refit inside every fold-training split",
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "fold_results": "results/modeling/tuned/master_tuned_fold_results.csv",
            "fold_summary": "results/modeling/tuned/master_tuned_fold_summary.csv",
            "oof_predictions": "results/modeling/tuned/master_tuned_oof_predictions.csv",
            "candidate_summary": "results/modeling/tuned/master_tuned_candidate_summary.csv",
            "ensemble_summary": "results/modeling/tuned/master_tuned_ensemble_summary.csv",
            "combined_summary": "results/modeling/tuned/master_tuned_combined_summary.csv",
        },
    }
    (OUT_DIR / "master_tuned_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_cols = [
        "candidate",
        "branch",
        "feature_set",
        "best_f1_threshold",
        "best_f1_final_f1",
        "best_f1_final_mcc",
        "best_f1_specificity",
        "best_f1_sensitivity",
        "best_f1_precision",
        "best_f1_expected_fp_per_3500",
        "best_f1_expected_fn_per_3500",
        "default_oof_f1",
        "default_oof_mcc",
    ]
    print("\n=== TOP FINAL-PREVALENCE CANDIDATES ===")
    print(combined_summary.loc[:, display_cols].head(20).round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
