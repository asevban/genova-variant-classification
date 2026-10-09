"""Leakage-safe TabPFN API experiment for the MASTER panel.

This script evaluates TabPFN as an additional candidate only. It does not use
outer validation/final labels, and it refits MASTER preprocessing separately in
each CV fold. Real API runs upload fold train/test matrices to Prior Labs, so
the API path is gated behind --allow-api-upload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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
from tabpfn_client import TabPFNClassifier, set_access_token

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (
    ID_COLUMN,
    STRATEGIES,
    TARGET_COLUMN,
    MasterPreprocessor,
    load_master_csv,
)

RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
OUTER_SPLIT_FILE = ROOT / "artifacts" / "master_outer_split.csv"
FOLD_FILE = ROOT / "artifacts" / "master_repeated_cv_folds.csv"
OUT_DIR = ROOT / "results" / "modeling" / "tabpfn_api"
DEFAULT_FEATURE_SET = "M3_missing_aware_compact"
FINAL_PATHOGENIC_RATE = 500 / 3500
FINAL_TOTAL_N = 3500
DEFAULT_THRESHOLD = 0.50
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.001), 3)


def _parse_repeat_seeds(value: str, available: list[int]) -> list[int]:
    if value.lower() == "all":
        return available
    requested = [int(item.strip()) for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unknown repeat seed(s): {unknown}; available={available}")
    return requested


def _token_available(load_env_file: bool) -> bool:
    if load_env_file:
        load_dotenv(ROOT / ".env")
    token = os.environ.get("TABPFN_TOKEN") or os.environ.get("PRIORLABS_API_KEY")
    if token:
        set_access_token(token)
        return True
    return False


def _positive_probability(model: TabPFNClassifier, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class 1 is absent from fitted classes: {classes}")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def _metric_row(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
    row: dict[str, float | int] = {
        "n": len(y_true),
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


def _final_prevalence_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    positive = y_score[y_true == 1]
    negative = y_score[y_true == 0]
    tpr = float((positive >= threshold).mean()) if len(positive) else 0.0
    fpr = float((negative >= threshold).mean()) if len(negative) else 0.0
    pi = FINAL_PATHOGENIC_RATE
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
        "expected_fp_per_3500": float(FINAL_TOTAL_N * fp),
        "expected_fn_per_3500": float(FINAL_TOTAL_N * fn),
    }


def _threshold_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _final_prevalence_metrics(y_true, y_score, float(threshold))
            for threshold in THRESHOLDS
        ]
    )


def _ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "oof_auroc": float("nan"),
            "oof_auprc": float("nan"),
            "final_weighted_auroc": float("nan"),
            "final_weighted_auprc": float("nan"),
        }
    source_pi = float(y_true.mean())
    sample_weight = np.where(
        y_true == 1,
        FINAL_PATHOGENIC_RATE / source_pi,
        (1.0 - FINAL_PATHOGENIC_RATE) / (1.0 - source_pi),
    )
    return {
        "oof_auroc": float(roc_auc_score(y_true, y_score)),
        "oof_auprc": float(average_precision_score(y_true, y_score)),
        "final_weighted_auroc": float(
            roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        ),
        "final_weighted_auprc": float(
            average_precision_score(y_true, y_score, sample_weight=sample_weight)
        ),
    }


def _select_threshold(
    table: pd.DataFrame,
    f1_floor: float,
    fp_max: float,
    sensitivity_floor: float,
) -> tuple[pd.Series, str]:
    strict = table.loc[
        table["final_f1"].ge(f1_floor)
        & table["expected_fp_per_3500"].le(fp_max)
        & table["final_sensitivity"].ge(sensitivity_floor)
    ].copy()
    if len(strict):
        return (
            strict.sort_values(
                ["final_mcc", "final_f1", "final_specificity", "final_sensitivity"],
                ascending=False,
            ).iloc[0],
            "strict",
        )
    return (
        table.sort_values(["final_mcc", "final_f1", "final_specificity"], ascending=False).iloc[0],
        "mcc_only",
    )


def _aggregate_folds(fold_results: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {"folds": len(fold_results)}
    for column in [
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
        "predict_seconds",
    ]:
        payload[f"{column}_mean"] = float(fold_results[column].mean())
        payload[f"{column}_std"] = float(fold_results[column].std(ddof=1))
    return payload


def _make_model(args: argparse.Namespace, seed: int) -> TabPFNClassifier:
    params: dict[str, Any] = {"random_state": seed}
    if args.model_path:
        params["model_path"] = args.model_path
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.balance_probabilities:
        params["balance_probabilities"] = True
    return TabPFNClassifier(**params)


def _prepare_fold_plan(
    raw: pd.DataFrame,
    outer: pd.DataFrame,
    folds: pd.DataFrame,
    repeat_seeds: list[int],
    max_folds: int | None,
) -> list[dict[str, Any]]:
    train_ids = set(outer.loc[outer["partition"].eq("train"), ID_COLUMN])
    train = raw.loc[raw[ID_COLUMN].isin(train_ids)].reset_index(drop=True)
    if len(train) != 2345:
        raise ValueError(f"Unexpected outer-train size: {len(train)}")

    plan: list[dict[str, Any]] = []
    for seed in repeat_seeds:
        seed_folds = folds.loc[folds["repeat_seed"].eq(seed)]
        selected_folds = sorted(seed_folds["fold"].unique().tolist())
        if max_folds is not None:
            selected_folds = selected_folds[:max_folds]
        for fold in selected_folds:
            hold_ids = set(seed_folds.loc[seed_folds["fold"].eq(fold), ID_COLUMN])
            fit_frame = train.loc[~train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)
            hold_frame = train.loc[train[ID_COLUMN].isin(hold_ids)].reset_index(drop=True)
            plan.append(
                {
                    "repeat_seed": int(seed),
                    "fold": int(fold),
                    "fit_frame": fit_frame,
                    "hold_frame": hold_frame,
                    "fit_n": len(fit_frame),
                    "holdout_n": len(hold_frame),
                }
            )
    return plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_master_csv(RAW_FILE)
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    folds = pd.read_csv(FOLD_FILE, dtype={ID_COLUMN: "string"})
    available_seeds = sorted(folds["repeat_seed"].unique().astype(int).tolist())
    repeat_seeds = _parse_repeat_seeds(args.repeat_seeds, available_seeds)
    max_folds = None if args.max_folds <= 0 else int(args.max_folds)
    fold_plan = _prepare_fold_plan(raw, outer, folds, repeat_seeds, max_folds)
    token_present = _token_available(args.load_env_file)
    dry_run = args.dry_run or not args.allow_api_upload

    plan_frame = pd.DataFrame(
        [
            {
                "feature_set": args.feature_set,
                "repeat_seed": item["repeat_seed"],
                "fold": item["fold"],
                "fit_n": item["fit_n"],
                "holdout_n": item["holdout_n"],
            }
            for item in fold_plan
        ]
    )
    plan_frame.to_csv(OUT_DIR / "tabpfn_api_fold_plan.csv", index=False)

    metadata: dict[str, Any] = {
        "status": "dry_run" if dry_run else "started",
        "candidate": "TabPFN_API",
        "feature_set": args.feature_set,
        "repeat_seeds": repeat_seeds,
        "fold_count": len(fold_plan),
        "max_folds": max_folds,
        "threshold_default": DEFAULT_THRESHOLD,
        "final_pathogenic_rate_assumption": FINAL_PATHOGENIC_RATE,
        "api_upload_allowed": bool(args.allow_api_upload),
        "env_file_loaded": bool(args.load_env_file),
        "token_env_present": bool(token_present),
        "token_value_logged": False,
        "outer_validation_used_for_selection": False,
        "outputs": {
            "fold_plan": "results/modeling/tabpfn_api/tabpfn_api_fold_plan.csv",
            "fold_results": "results/modeling/tabpfn_api/tabpfn_api_fold_results.csv",
            "oof_predictions": "results/modeling/tabpfn_api/tabpfn_api_oof_predictions.csv",
            "threshold_table": "results/modeling/tabpfn_api/tabpfn_api_threshold_table.csv",
            "summary": "results/modeling/tabpfn_api/tabpfn_api_summary.csv",
            "metadata": "results/modeling/tabpfn_api/tabpfn_api_metadata.json",
        },
    }

    if dry_run:
        metadata["elapsed_seconds"] = time.time() - started
        (OUT_DIR / "tabpfn_api_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Dry run only. Add --allow-api-upload after setting TABPFN_TOKEN to run TabPFN.")
        print(plan_frame.to_string(index=False))
        return metadata

    if not token_present:
        metadata["status"] = "blocked_missing_token"
        metadata["elapsed_seconds"] = time.time() - started
        (OUT_DIR / "tabpfn_api_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "Missing TabPFN token. Create ROOT/.env with TABPFN_TOKEN=... or export "
            "TABPFN_TOKEN before running. Do not paste the token into chat."
        )

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    for job_no, item in enumerate(fold_plan, start=1):
        processor = MasterPreprocessor(STRATEGIES[args.feature_set])
        x_fit = processor.fit_transform(item["fit_frame"]).to_numpy(dtype=np.float32)
        x_hold = processor.transform(item["hold_frame"]).to_numpy(dtype=np.float32)
        y_fit = item["fit_frame"][TARGET_COLUMN].to_numpy(dtype=int)
        y_hold = item["hold_frame"][TARGET_COLUMN].to_numpy(dtype=int)

        model = _make_model(args, item["repeat_seed"])
        fit_started = time.time()
        model.fit(
            x_fit,
            y_fit,
            description=(
                f"MASTER TabPFN API {args.feature_set} seed={item['repeat_seed']} "
                f"fold={item['fold']}"
            ),
        )
        fit_seconds = time.time() - fit_started
        predict_started = time.time()
        score = _positive_probability(model, x_hold)
        predict_seconds = time.time() - predict_started
        metrics = _metric_row(y_hold, score, DEFAULT_THRESHOLD)
        fold_rows.append(
            {
                "candidate": "TabPFN_API",
                "feature_set": args.feature_set,
                "repeat_seed": item["repeat_seed"],
                "fold": item["fold"],
                "fit_n": item["fit_n"],
                "holdout_n": item["holdout_n"],
                "output_features": int(x_fit.shape[1]),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                **metrics,
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    ID_COLUMN: item["hold_frame"][ID_COLUMN].astype(str).to_numpy(),
                    TARGET_COLUMN: y_hold,
                    "score": score.astype(float),
                    "candidate": "TabPFN_API",
                    "feature_set": args.feature_set,
                    "repeat_seed": item["repeat_seed"],
                    "fold": item["fold"],
                }
            )
        )
        print(
            f"[{job_no}/{len(fold_plan)}] TabPFN_API | {args.feature_set} "
            f"| seed={item['repeat_seed']} fold={item['fold']} "
            f"f1={metrics['f1']:.4f} mcc={metrics['mcc']:.4f} "
            f"spec={metrics['specificity']:.4f}",
            flush=True,
        )

    fold_results = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    y_true = oof[TARGET_COLUMN].to_numpy(dtype=int)
    y_score = oof["score"].to_numpy(dtype=float)
    threshold_table = _threshold_table(y_true, y_score)
    selected, selection_status = _select_threshold(
        threshold_table,
        f1_floor=args.f1_floor,
        fp_max=args.fp_max,
        sensitivity_floor=args.sensitivity_floor,
    )
    summary_row = {
        "candidate": "TabPFN_API",
        "feature_set": args.feature_set,
        "selection_status": selection_status,
        **_aggregate_folds(fold_results),
        **_ranking_metrics(y_true, y_score),
        **{f"selected_{key}": value for key, value in selected.to_dict().items()},
    }
    summary = pd.DataFrame([summary_row])

    fold_results.to_csv(OUT_DIR / "tabpfn_api_fold_results.csv", index=False)
    oof.to_csv(OUT_DIR / "tabpfn_api_oof_predictions.csv", index=False)
    threshold_table.to_csv(OUT_DIR / "tabpfn_api_threshold_table.csv", index=False)
    summary.to_csv(OUT_DIR / "tabpfn_api_summary.csv", index=False)
    metadata["status"] = "completed"
    metadata["elapsed_seconds"] = time.time() - started
    (OUT_DIR / "tabpfn_api_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== TABPFN API SUMMARY ===")
    display_cols = [
        "candidate",
        "feature_set",
        "folds",
        "selection_status",
        "mcc_mean",
        "f1_mean",
        "specificity_mean",
        "sensitivity_mean",
        "oof_auprc",
        "final_weighted_auprc",
        "selected_threshold",
        "selected_final_mcc",
        "selected_expected_fp_per_3500",
    ]
    print(summary.loc[:, display_cols].to_string(index=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=sorted(STRATEGIES))
    parser.add_argument(
        "--repeat-seeds",
        default="42",
        help="Comma-separated repeat seeds or 'all'. Default runs the 5 folds for seed 42.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=5,
        help="Maximum folds per selected seed; use 0 for all available folds.",
    )
    parser.add_argument("--model-path", default="", help="Optional TabPFN model path/name.")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--balance-probabilities", action="store_true")
    parser.add_argument("--f1-floor", type=float, default=0.530)
    parser.add_argument("--fp-max", type=float, default=400.0)
    parser.add_argument("--sensitivity-floor", type=float, default=0.62)
    parser.add_argument(
        "--allow-api-upload",
        action="store_true",
        help="Actually send fold matrices to Prior Labs. Without this flag the script is dry-run.",
    )
    parser.add_argument(
        "--load-env-file",
        action="store_true",
        help="Load TABPFN_TOKEN from ROOT/.env. Prefer terminal input for secrets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
