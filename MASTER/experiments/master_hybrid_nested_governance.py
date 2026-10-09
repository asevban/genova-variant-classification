"""Nested governance validation for MASTER Hybrid-C/Hybrid-E candidates.

This is not a new model search for final reporting. It is a leakage-safe
governance check designed to answer one question:

Can Hybrid-C/Hybrid-E still challenge CATOPT-A when every weight,
calibration and threshold decision is made inside the outer-training fold?

Protocol:
- Use only the existing MASTER labelled training table.
- Use only M3_missing_aware_compact preprocessing.
- Fit preprocessing separately inside each inner/outer fold.
- Select weights/calibration/threshold on inner OOF scores only.
- Evaluate the locked decision on the outer holdout only once.
- Do not use historical outer validation or final/test data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_catboost_experiments import build_candidates as build_catboost_candidates  # noqa: E402
from master_lightgbm_experiments import build_candidates as build_lightgbm_candidates  # noqa: E402
from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    STRATEGIES,
    TARGET_COLUMN,
    MasterPreprocessor,
    load_master_csv,
)
from master_targeted_mcc_fp_experiments import (  # noqa: E402
    F1_KEEP_TOLERANCE,
    FP_TARGET,
    RELAXED_F1_FLOOR,
    RELAXED_SENSITIVITY_FLOOR,
    STRICT_SENSITIVITY_FLOOR,
    THRESHOLDS,
    TargetModelCandidate,
    build_new_model_candidates,
    calibrate_scores_cv,
    final_prevalence_metrics,
    make_target_model,
    prior_correct_probability,
    prior_sample_weight,
    ranking_metrics,
    sample_weight_for_candidate,
)
from master_tuned_modeling_experiments import (  # noqa: E402
    FINAL_PATHOGENIC_RATE,
    OUTER_SPLIT_FILE,
    RANDOM_STATE,
    RAW_FILE,
    positive_probability,
    rf_params,
    xgb_params,
)


FOLD_FILE = ROOT / "artifacts" / "master_repeated_cv_folds.csv"
OUT_DIR = ROOT / "results" / "modeling" / "nested_governance"
FEATURE_SET = "M3_missing_aware_compact"
FINAL_TOTAL_N = 3500
AUPRC_KEEP_TOLERANCE = 0.001

BASE_MEMBERS = [
    "RF8",
    "XGBSPW",
    "CATB",
    "CATR",
    "LGBMP25",
    "XGBNW",
    "HBRF",
]

BASE_TO_FULL_NAME = {
    "RF8": "M3_RF_depth8_balanced",
    "XGBSPW": "M3_XGB_SPW_regularized_x1.5",
    "CATB": "M3_CatBoost_bernoulli",
    "CATR": "M3_CatBoost_regularized",
    "LGBMP25": "M3_LGBM_prior025",
    "XGBNW": "M3_XGB_NW_shallow_reg",
    "HBRF": "M3_HardBenign_RF_top30_w2_s025",
}


@dataclass(frozen=True)
class SelectedStrategy:
    strategy: str
    method_group: str
    selected_variant: str
    members: list[str]
    weights: np.ndarray
    threshold: float
    calibration_matrix: str = "raw"
    selection_status: str = "unknown"
    inner_metrics: dict[str, float] | None = None
    details: dict[str, Any] | None = None


def _xgb_nw_params(seed: int) -> dict[str, Any]:
    params = xgb_params(
        n_estimators=550,
        max_depth=2,
        learning_rate=0.025,
        min_child_weight=5,
        gamma=0.05,
        reg_alpha=0.10,
        reg_lambda=5.0,
        colsample_bytree=0.80,
    )
    params["random_state"] = seed
    return params


def _xgb_spw_params(seed: int, fold_spw: float) -> dict[str, Any]:
    params = xgb_params(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=6,
        gamma=0.10,
        reg_alpha=0.25,
        reg_lambda=6.0,
        subsample=0.85,
        colsample_bytree=0.75,
    )
    params["random_state"] = seed
    params["scale_pos_weight"] = float(fold_spw * 1.5)
    return params


def _rf8_params(seed: int) -> dict[str, Any]:
    params = rf_params(
        n_estimators=650,
        max_depth=8,
        min_samples_leaf=3,
        min_samples_split=8,
        class_weight="balanced_subsample",
    )
    params["random_state"] = seed
    return params


def _hard_benign_candidate(seed: int) -> TargetModelCandidate:
    candidates = {
        candidate.name: candidate for candidate in build_new_model_candidates()
    }
    candidate = candidates["M3_HardBenign_RF_top30_w2_s025"]
    params = dict(candidate.params)
    params["random_state"] = seed
    return replace(candidate, params=params)


def _catboost_candidate(name: str, seed: int):
    candidates = {candidate.name: candidate for candidate in build_catboost_candidates()}
    candidate = candidates[name]
    params = dict(candidate.params)
    params["random_seed"] = seed
    return CatBoostClassifier(**params)


def _lightgbm_candidate(seed: int):
    candidates = {candidate.name: candidate for candidate in build_lightgbm_candidates()}
    candidate = candidates["M3_LGBM_prior025"]
    params = dict(candidate.params)
    params["random_state"] = seed
    return LGBMClassifier(**params), candidate.prior_strength


def _fit_one_base(
    member: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    fold_spw: float,
    fold_key: tuple[int, int, int],
) -> tuple[Any, np.ndarray | None, int]:
    if member == "RF8":
        return RandomForestClassifier(**_rf8_params(seed)), None, 0
    if member == "XGBSPW":
        return XGBClassifier(**_xgb_spw_params(seed, fold_spw)), None, 0
    if member == "XGBNW":
        return XGBClassifier(**_xgb_nw_params(seed)), None, 0
    if member == "CATB":
        return _catboost_candidate("M3_CatBoost_bernoulli", seed), None, 0
    if member == "CATR":
        return _catboost_candidate("M3_CatBoost_regularized", seed), None, 0
    if member == "LGBMP25":
        model, prior_strength = _lightgbm_candidate(seed)
        weight = prior_sample_weight(y_fit, prior_strength)
        return model, None if np.allclose(weight, 1.0) else weight, 0
    if member == "HBRF":
        candidate = _hard_benign_candidate(seed)
        weight, hard_count = sample_weight_for_candidate(
            candidate,
            x_fit,
            y_fit,
            seed,
            hard_cache={},
            fold_key=fold_key,
        )
        return make_target_model(candidate), weight, hard_count
    raise ValueError(f"Unknown base member: {member}")


def _fit_score_base_frame(
    fit_frame: pd.DataFrame,
    hold_frame: pd.DataFrame,
    seed: int,
    fold_key: tuple[int, int, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processor = MasterPreprocessor(STRATEGIES[FEATURE_SET])
    x_fit = processor.fit_transform(fit_frame).to_numpy(dtype=np.float32)
    x_hold = processor.transform(hold_frame).to_numpy(dtype=np.float32)
    y_fit = fit_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_hold = hold_frame[TARGET_COLUMN].to_numpy(dtype=int)
    fold_spw = float((y_fit == 0).sum() / (y_fit == 1).sum())

    output = pd.DataFrame(
        {
            ID_COLUMN: hold_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: y_hold,
        }
    )
    fit_seconds: dict[str, float] = {}
    hard_counts: dict[str, int] = {}
    for member in BASE_MEMBERS:
        model, sample_weight, hard_count = _fit_one_base(
            member,
            x_fit,
            y_fit,
            seed,
            fold_spw,
            fold_key,
        )
        started = time.time()
        if sample_weight is None:
            model.fit(x_fit, y_fit)
        else:
            model.fit(x_fit, y_fit, sample_weight=sample_weight)
        fit_seconds[member] = time.time() - started
        hard_counts[member] = hard_count
        output[member] = positive_probability(model, x_hold)

    metadata = {
        "fit_rows": int(len(fit_frame)),
        "hold_rows": int(len(hold_frame)),
        "fit_positive_rate": float(y_fit.mean()),
        "hold_positive_rate": float(y_hold.mean()),
        "output_features": int(x_fit.shape[1]),
        "fit_seconds": fit_seconds,
        "hard_counts": hard_counts,
    }
    return output, metadata


def _make_inner_oof(
    outer_train: pd.DataFrame,
    outer_seed: int,
    outer_fold: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE + outer_seed * 10 + outer_fold,
    )
    y = outer_train[TARGET_COLUMN].to_numpy(dtype=int)
    frames = []
    rows = []
    for inner_fold, (fit_idx, hold_idx) in enumerate(splitter.split(outer_train, y)):
        fit_frame = outer_train.iloc[fit_idx].reset_index(drop=True)
        hold_frame = outer_train.iloc[hold_idx].reset_index(drop=True)
        scores, metadata = _fit_score_base_frame(
            fit_frame,
            hold_frame,
            seed=RANDOM_STATE + outer_seed * 100 + outer_fold * 10 + inner_fold,
            fold_key=(outer_seed, outer_fold, inner_fold),
        )
        scores["inner_fold"] = inner_fold
        frames.append(scores)
        rows.append(
            {
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                **metadata,
            }
        )
    inner_oof = pd.concat(frames, ignore_index=True)
    return inner_oof.sort_values(ID_COLUMN).reset_index(drop=True), pd.DataFrame(rows)


def _select_threshold_fast(
    y_true: np.ndarray,
    score: np.ndarray,
    f1_floor: float,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = np.sort(score[y_true == 1])
    negatives = np.sort(score[y_true == 0])
    pos_n = len(positives)
    neg_n = len(negatives)
    tp_counts = pos_n - np.searchsorted(positives, THRESHOLDS, side="left")
    fp_counts = neg_n - np.searchsorted(negatives, THRESHOLDS, side="left")
    tpr = tp_counts / pos_n
    fpr = fp_counts / neg_n

    pi = FINAL_PATHOGENIC_RATE
    tp = pi * tpr
    fn = pi * (1.0 - tpr)
    fp = (1.0 - pi) * fpr
    tn = (1.0 - pi) * (1.0 - fpr)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(
        2.0 * precision * tpr,
        precision + tpr,
        out=np.zeros_like(tp),
        where=(precision + tpr) > 0,
    )
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.divide(
        tp * tn - fp * fn,
        np.sqrt(denom),
        out=np.zeros_like(tp),
        where=denom > 0,
    )
    specificity = 1.0 - fpr
    expected_fp = FINAL_TOTAL_N * fp
    expected_fn = FINAL_TOTAL_N * fn

    mask = (
        (f1 >= f1_floor)
        & (expected_fp <= FP_TARGET)
        & (tpr >= STRICT_SENSITIVITY_FLOOR)
    )
    status = "strict"
    if not mask.any():
        mask = (
            (f1 >= RELAXED_F1_FLOOR)
            & (expected_fp <= FP_TARGET)
            & (tpr >= RELAXED_SENSITIVITY_FLOOR)
        )
        status = "relaxed"
    if not mask.any():
        mask = np.ones_like(f1, dtype=bool)
        status = "mcc_only"

    candidate_indices = np.flatnonzero(mask)
    order = np.lexsort(
        (
            -tpr[candidate_indices],
            -specificity[candidate_indices],
            -f1[candidate_indices],
            -mcc[candidate_indices],
        )
    )
    idx = candidate_indices[order[0]]
    return {
        "status": status,
        "threshold": float(THRESHOLDS[idx]),
        "f1": float(f1[idx]),
        "mcc": float(mcc[idx]),
        "specificity": float(specificity[idx]),
        "sensitivity": float(tpr[idx]),
        "precision": float(precision[idx]),
        "balanced_accuracy": float((specificity[idx] + tpr[idx]) / 2.0),
        "expected_fp_per_3500": float(expected_fp[idx]),
        "expected_fn_per_3500": float(expected_fn[idx]),
    }


def _sort_key(metrics: dict[str, Any]) -> tuple[int, float, float, float]:
    status_rank = {"strict": 2, "relaxed": 1, "mcc_only": 0}.get(
        str(metrics["status"]),
        0,
    )
    return (
        status_rank,
        float(metrics["mcc"]),
        float(metrics["f1"]),
        -float(metrics["expected_fp_per_3500"]),
    )


def _weighted_score(frame: pd.DataFrame, members: list[str], weights: np.ndarray) -> np.ndarray:
    return frame[members].to_numpy(dtype=float) @ np.asarray(weights, dtype=float)


def _ranking_for_score(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return ranking_metrics(y_true, np.asarray(score, dtype=float))


def _matrix_rank_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    ranked = _ranking_for_score(y_true, score)
    return {
        "inner_oof_auroc": float(ranked["oof_auroc"]),
        "inner_oof_auprc": float(ranked["oof_auprc"]),
        "inner_final_weighted_auprc": float(ranked["final_weighted_auprc"]),
    }


def _select_fixed_weight_strategy(
    name: str,
    method_group: str,
    frame: pd.DataFrame,
    members: list[str],
    weights: list[float],
    f1_floor: float,
    calibration_matrix: str = "raw",
    details: dict[str, Any] | None = None,
) -> SelectedStrategy:
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)
    weight_array = np.asarray(weights, dtype=float)
    weight_array = weight_array / weight_array.sum()
    score = _weighted_score(frame, members, weight_array)
    metrics = _select_threshold_fast(y_true, score, f1_floor)
    metrics.update(_matrix_rank_metrics(y_true, score))
    return SelectedStrategy(
        strategy=name,
        method_group=method_group,
        selected_variant=name,
        members=members,
        weights=weight_array,
        threshold=float(metrics["threshold"]),
        calibration_matrix=calibration_matrix,
        selection_status=str(metrics["status"]),
        inner_metrics={k: float(v) for k, v in metrics.items() if k != "status"},
        details=details or {},
    )


def _hybrid_c_weight_grid() -> list[np.ndarray]:
    units = 40
    weights: list[np.ndarray] = []
    for rf_units in range(6, 23):
        for spw_units in range(2, 17):
            for cat_units in range(2, 25):
                for lgbm_units in range(0, 13):
                    for xgbnw_units in range(0, 13):
                        hard_units = (
                            units
                            - rf_units
                            - spw_units
                            - cat_units
                            - lgbm_units
                            - xgbnw_units
                        )
                        if hard_units < 0 or hard_units > 12:
                            continue
                        if lgbm_units + hard_units == 0:
                            continue
                        if rf_units + spw_units + cat_units < 20:
                            continue
                        weights.append(
                            np.asarray(
                                [
                                    rf_units,
                                    spw_units,
                                    cat_units,
                                    lgbm_units,
                                    xgbnw_units,
                                    hard_units,
                                ],
                                dtype=float,
                            )
                            / units
                        )
    return weights


def _hybrid_e_weight_vectors(samples: int) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    prototypes = [
        [0.382, 0.252, 0.318, 0.013, 0.027, 0.007, 0.001],
        [0.122, 0.101, 0.070, 0.152, 0.274, 0.115, 0.165],
        [0.40, 0.30, 0.30, 0.00, 0.00, 0.00, 0.00],
        [0.375, 0.25, 0.325, 0.00, 0.025, 0.00, 0.025],
        [0.35, 0.25, 0.30, 0.00, 0.025, 0.025, 0.05],
        [0.30, 0.25, 0.25, 0.05, 0.10, 0.05, 0.00],
        [0.30, 0.20, 0.20, 0.10, 0.15, 0.10, 0.05],
        [0.25, 0.15, 0.15, 0.10, 0.20, 0.10, 0.05],
    ]
    vectors: list[np.ndarray] = []
    per_prototype = max(1, samples // len(prototypes))
    for prototype in prototypes:
        vector = np.asarray(prototype, dtype=float)
        vector = vector / vector.sum()
        vectors.append(vector)
        alpha = vector * 120.0 + 1.0
        vectors.extend(rng.dirichlet(alpha, size=per_prototype))
    vectors.extend(rng.dirichlet(np.ones(len(BASE_MEMBERS)) * 2.0, size=max(1, samples // 4)))

    deduped: dict[tuple[float, ...], np.ndarray] = {}
    for vector in vectors:
        vector = np.asarray(vector, dtype=float)
        vector = vector / vector.sum()
        if not (0.05 <= vector[0] <= 0.65):
            continue
        if not (0.02 <= vector[1] <= 0.50):
            continue
        if not (0.02 <= vector[2] <= 0.70):
            continue
        if vector[3] > 0.40 or vector[4] > 0.40 or vector[5] > 0.40:
            continue
        if vector[6] > 0.35:
            continue
        rounded = np.round(vector, 3)
        rounded = rounded / rounded.sum()
        deduped[tuple(np.round(rounded, 3))] = rounded
    return list(deduped.values())


def _select_hybrid_c(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    weights_grid: list[np.ndarray],
) -> SelectedStrategy:
    members = ["RF8", "XGBSPW", "CATB", "LGBMP25", "XGBNW", "HBRF"]
    matrix = inner_frame[members].to_numpy(dtype=float)
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    best: tuple[tuple[int, float, float, float], np.ndarray, dict[str, Any], np.ndarray] | None = None
    for weights in weights_grid:
        score = matrix @ weights
        metrics = _select_threshold_fast(y_true, score, f1_floor)
        key = _sort_key(metrics)
        if best is None or key > best[0]:
            best = (key, weights, metrics, score)
    if best is None:
        raise RuntimeError("Hybrid-C weight grid produced no candidate.")
    _, weights, metrics, score = best
    metrics.update(_matrix_rank_metrics(y_true, score))
    nonzero = {
        BASE_TO_FULL_NAME[member]: float(weight)
        for member, weight in zip(members, weights)
        if weight > 0
    }
    return SelectedStrategy(
        strategy="Hybrid-C_nested_selected",
        method_group="Nested Hybrid-C FP-controlled ensemble",
        selected_variant="HYC_inner_selected",
        members=members,
        weights=weights,
        threshold=float(metrics["threshold"]),
        selection_status=str(metrics["status"]),
        inner_metrics={k: float(v) for k, v in metrics.items() if k != "status"},
        details={"members": nonzero, "selection": "inner_fold_weight_grid"},
    )


def _calibrated_inner_matrices(
    inner_frame: pd.DataFrame,
    members: list[str],
) -> dict[str, np.ndarray]:
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    raw_matrix = inner_frame[members].to_numpy(dtype=float)
    matrices = {"raw": raw_matrix}
    for method in ["sigmoid", "isotonic"]:
        for apply_prior in [False, True]:
            columns = [
                calibrate_scores_cv(
                    y_true,
                    raw_matrix[:, idx],
                    method=method,
                    apply_prior_correction=apply_prior,
                )
                for idx in range(raw_matrix.shape[1])
            ]
            matrices[f"{method}_prior{apply_prior}"] = np.column_stack(columns)
    return matrices


def _fit_apply_calibration(
    inner_frame: pd.DataFrame,
    outer_frame: pd.DataFrame,
    members: list[str],
    matrix_name: str,
) -> np.ndarray:
    outer_raw = outer_frame[members].to_numpy(dtype=float)
    if matrix_name == "raw":
        return outer_raw

    method, prior_part = matrix_name.split("_prior")
    apply_prior = prior_part == "True"
    inner_y = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    inner_raw = np.clip(inner_frame[members].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    output = np.zeros_like(outer_raw, dtype=float)
    for idx, member in enumerate(members):
        train_score = inner_raw[:, idx]
        hold_score = np.clip(outer_raw[:, idx], 1e-7, 1 - 1e-7)
        if method == "sigmoid":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(_logit_array(train_score).reshape(-1, 1), inner_y)
            calibrated = model.predict_proba(_logit_array(hold_score).reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(train_score, inner_y)
            calibrated = model.predict(hold_score)
        else:
            raise ValueError(f"Unknown calibration matrix: {matrix_name}")
        if apply_prior:
            calibrated = prior_correct_probability(calibrated, float(inner_y.mean()))
        output[:, idx] = np.clip(calibrated, 1e-7, 1 - 1e-7)
    return output


def _logit_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _remember_top(
    pool: list[tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray]],
    item: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray],
    max_size: int = 500,
) -> None:
    pool.append(item)
    if len(pool) > max_size * 2:
        pool.sort(key=lambda value: value[0], reverse=True)
        del pool[max_size:]


def _select_hybrid_e(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    catopt_inner_metrics: dict[str, float],
    weight_vectors: list[np.ndarray],
) -> list[SelectedStrategy]:
    members = list(BASE_MEMBERS)
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    matrices = _calibrated_inner_matrices(inner_frame, members)

    best_mcc: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray] | None = None
    best_low_fp: tuple[tuple[float, float, float], str, np.ndarray, dict[str, Any], np.ndarray] | None = None
    auprc_pool: list[
        tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray]
    ] = []

    for matrix_name, matrix in matrices.items():
        for weights in weight_vectors:
            score = matrix @ weights
            metrics = _select_threshold_fast(y_true, score, f1_floor)
            sort_key = _sort_key(metrics)
            candidate = (sort_key, matrix_name, weights, metrics, score)
            if best_mcc is None or sort_key > best_mcc[0]:
                best_mcc = candidate
            if metrics["status"] == "strict":
                if float(metrics["mcc"]) >= catopt_inner_metrics["mcc"] - 0.010:
                    _remember_top(auprc_pool, candidate)
                if float(metrics["f1"]) >= catopt_inner_metrics["f1"] - F1_KEEP_TOLERANCE:
                    fp_key = (
                        -float(metrics["expected_fp_per_3500"]),
                        float(metrics["mcc"]),
                        float(metrics["f1"]),
                    )
                    fp_candidate = (fp_key, matrix_name, weights, metrics, score)
                    if best_low_fp is None or fp_key > best_low_fp[0]:
                        best_low_fp = fp_candidate

    if best_mcc is None:
        raise RuntimeError("Hybrid-E search produced no candidate.")

    auprc_pool.sort(key=lambda value: value[0], reverse=True)
    catopt_auprc = catopt_inner_metrics["inner_final_weighted_auprc"]
    best_auprc = None
    for sort_key, matrix_name, weights, metrics, score in auprc_pool:
        rank = _ranking_for_score(y_true, score)
        if rank["final_weighted_auprc"] >= catopt_auprc - AUPRC_KEEP_TOLERANCE:
            best_auprc = (sort_key, matrix_name, weights, metrics, score, rank)
            break
    if best_auprc is None:
        sort_key, matrix_name, weights, metrics, score = auprc_pool[0] if auprc_pool else best_mcc
        best_auprc = (
            sort_key,
            matrix_name,
            weights,
            metrics,
            score,
            _ranking_for_score(y_true, score),
        )

    strategies = []
    selections = [
        ("Hybrid-E_AUPRC_guard_nested", "inner_best_mcc_with_auprc_guard", best_auprc),
        (
            "Hybrid-E_bestMCC_nested",
            "inner_best_strict_mcc",
            (*best_mcc, _ranking_for_score(y_true, best_mcc[4])),
        ),
    ]
    if best_low_fp is not None:
        fp_key, matrix_name, weights, metrics, score = best_low_fp
        selections.append(
            (
                "Hybrid-E_lowFP_nested",
                "inner_lowest_fp_with_f1_preserved",
                (
                    _sort_key(metrics),
                    matrix_name,
                    weights,
                    metrics,
                    score,
                    _ranking_for_score(y_true, score),
                ),
            )
        )
    else:
        sort_key, matrix_name, weights, metrics, score = best_mcc
        selections.append(
            (
                "Hybrid-E_lowFP_nested",
                "inner_lowfp_no_eligible_fallback_best_mcc",
                (
                    sort_key,
                    matrix_name,
                    weights,
                    metrics,
                    score,
                    _ranking_for_score(y_true, score),
                ),
            )
        )

    for strategy_name, selection_rule, selection in selections:
        _sort, matrix_name, weights, metrics, _score, rank = selection
        metrics = dict(metrics)
        metrics.update(
            {
                "inner_oof_auroc": float(rank["oof_auroc"]),
                "inner_oof_auprc": float(rank["oof_auprc"]),
                "inner_final_weighted_auprc": float(rank["final_weighted_auprc"]),
            }
        )
        nonzero = {
            BASE_TO_FULL_NAME[member]: float(weight)
            for member, weight in zip(members, weights)
            if weight > 0
        }
        strategies.append(
            SelectedStrategy(
                strategy=strategy_name,
                method_group="Nested Hybrid-E calibrated ensemble",
                selected_variant=f"{matrix_name}:{selection_rule}",
                members=members,
                weights=weights,
                threshold=float(metrics["threshold"]),
                calibration_matrix=matrix_name,
                selection_status=str(metrics["status"]),
                inner_metrics={k: float(v) for k, v in metrics.items() if k != "status"},
                details={
                    "members": nonzero,
                    "calibration_matrix": matrix_name,
                    "selection_rule": selection_rule,
                },
            )
        )
    return strategies


def _outer_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    f1_floor: float,
) -> dict[str, float | int | bool]:
    final_metrics = final_prevalence_metrics(
        y_true,
        score,
        threshold=threshold,
        pathogenic_rate=FINAL_PATHOGENIC_RATE,
    )
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ranked = _ranking_for_score(y_true, score)
    balanced_accuracy = (
        final_metrics["final_specificity"] + final_metrics["final_sensitivity"]
    ) / 2.0
    strict_pass = (
        final_metrics["final_f1"] >= f1_floor
        and final_metrics["expected_fp_per_3500"] <= FP_TARGET
        and final_metrics["final_sensitivity"] >= STRICT_SENSITIVITY_FLOOR
    )
    return {
        "outer_f1": float(final_metrics["final_f1"]),
        "outer_mcc": float(final_metrics["final_mcc"]),
        "outer_specificity": float(final_metrics["final_specificity"]),
        "outer_sensitivity": float(final_metrics["final_sensitivity"]),
        "outer_precision": float(final_metrics["final_precision"]),
        "outer_balanced_accuracy": float(balanced_accuracy),
        "outer_expected_fp_per_3500": float(final_metrics["expected_fp_per_3500"]),
        "outer_expected_fn_per_3500": float(final_metrics["expected_fn_per_3500"]),
        "outer_oof_auroc": float(ranked["oof_auroc"]),
        "outer_oof_auprc": float(ranked["oof_auprc"]),
        "outer_final_weighted_auprc": float(ranked["final_weighted_auprc"]),
        "outer_actual_tp": int(tp),
        "outer_actual_tn": int(tn),
        "outer_actual_fp": int(fp),
        "outer_actual_fn": int(fn),
        "outer_strict_pass": bool(strict_pass),
    }


def _evaluate_selected_strategy(
    strategy: SelectedStrategy,
    inner_frame: pd.DataFrame,
    outer_frame: pd.DataFrame,
    f1_floor: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = outer_frame[TARGET_COLUMN].to_numpy(dtype=int)
    matrix = _fit_apply_calibration(
        inner_frame,
        outer_frame,
        strategy.members,
        strategy.calibration_matrix,
    )
    score = matrix @ strategy.weights
    metrics = _outer_metrics(y_true, score, strategy.threshold, f1_floor)
    inner_metrics = strategy.inner_metrics or {}
    row = {
        "strategy": strategy.strategy,
        "method_group": strategy.method_group,
        "selected_variant": strategy.selected_variant,
        "calibration_matrix": strategy.calibration_matrix,
        "threshold": strategy.threshold,
        "selection_status": strategy.selection_status,
        "weights_json": json.dumps(
            {
                BASE_TO_FULL_NAME[member]: float(weight)
                for member, weight in zip(strategy.members, strategy.weights)
                if weight > 0
            },
            ensure_ascii=False,
        ),
        "details_json": json.dumps(strategy.details or {}, ensure_ascii=False),
        **{f"inner_{key}": value for key, value in inner_metrics.items()},
        **metrics,
    }
    pred_frame = pd.DataFrame(
        {
            ID_COLUMN: outer_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: y_true,
            "score": score,
            "threshold": strategy.threshold,
            "prediction": (score >= strategy.threshold).astype(int),
            "strategy": strategy.strategy,
            "method_group": strategy.method_group,
            "selected_variant": strategy.selected_variant,
        }
    )
    return row, pred_frame


def _select_all_strategies(
    inner_frame: pd.DataFrame,
    h_c_grid: list[np.ndarray],
    h_e_vectors: list[np.ndarray],
) -> tuple[list[SelectedStrategy], float]:
    catopt_initial = _select_fixed_weight_strategy(
        name="CATOPT-A_nested_reference",
        method_group="Nested fixed CATOPT-A reference",
        frame=inner_frame,
        members=["RF8", "XGBSPW", "CATB"],
        weights=[0.40, 0.30, 0.30],
        f1_floor=RELAXED_F1_FLOOR,
        details={"selection": "fixed_weights_inner_threshold"},
    )
    catopt_metrics = catopt_initial.inner_metrics or {}
    f1_floor = max(RELAXED_F1_FLOOR, catopt_metrics["f1"] - F1_KEEP_TOLERANCE)
    catopt = _select_fixed_weight_strategy(
        name="CATOPT-A_nested_reference",
        method_group="Nested fixed CATOPT-A reference",
        frame=inner_frame,
        members=["RF8", "XGBSPW", "CATB"],
        weights=[0.40, 0.30, 0.30],
        f1_floor=f1_floor,
        details={"selection": "fixed_weights_inner_threshold"},
    )
    catopt_inner = catopt.inner_metrics or {}

    strategies = [catopt]
    strategies.append(
        _select_fixed_weight_strategy(
            name="Hybrid-C_fixed_reported",
            method_group="Nested fixed reported Hybrid-C",
            frame=inner_frame,
            members=["RF8", "XGBSPW", "CATB", "LGBMP25", "HBRF"],
            weights=[0.375, 0.250, 0.325, 0.025, 0.025],
            f1_floor=f1_floor,
            details={"selection": "reported_fixed_weights_inner_threshold"},
        )
    )
    strategies.append(
        _select_fixed_weight_strategy(
            name="Hybrid-E_fixed_AUPRC_reported",
            method_group="Nested fixed reported Hybrid-E",
            frame=inner_frame,
            members=BASE_MEMBERS,
            weights=[0.382, 0.252, 0.318, 0.013, 0.027, 0.007, 0.001],
            f1_floor=f1_floor,
            details={"selection": "reported_fixed_raw_weights_inner_threshold"},
        )
    )

    low_fp_members = BASE_MEMBERS
    low_fp_weights = [
        0.1221221221221221,
        0.1011011011011011,
        0.07007007007007007,
        0.15215215215215214,
        0.27427427427427425,
        0.1151151151151151,
        0.16516516516516516,
    ]
    low_fp_calibrated = inner_frame.copy()
    low_fp_matrix = _calibrated_inner_matrices(inner_frame, low_fp_members)[
        "isotonic_priorTrue"
    ]
    low_fp_calibrated.loc[:, low_fp_members] = low_fp_matrix
    strategies.append(
        _select_fixed_weight_strategy(
            name="Hybrid-E_fixed_lowFP_reported",
            method_group="Nested fixed reported Hybrid-E",
            frame=low_fp_calibrated,
            members=low_fp_members,
            weights=low_fp_weights,
            f1_floor=f1_floor,
            calibration_matrix="isotonic_priorTrue",
            details={"selection": "reported_fixed_isotonic_prior_weights_inner_threshold"},
        )
    )

    strategies.append(_select_hybrid_c(inner_frame, f1_floor, h_c_grid))
    strategies.extend(_select_hybrid_e(inner_frame, f1_floor, catopt_inner, h_e_vectors))
    return strategies, f1_floor


def _load_development_frame() -> pd.DataFrame:
    raw = load_master_csv(RAW_FILE)
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    train_ids = set(outer.loc[outer["partition"].eq("train"), ID_COLUMN].astype(str))
    return raw.loc[raw[ID_COLUMN].astype(str).isin(train_ids)].reset_index(drop=True)


def _parse_outer_seeds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return [42, 52, 62, 72, 82]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _summarize_outer_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "threshold",
        "outer_f1",
        "outer_mcc",
        "outer_specificity",
        "outer_sensitivity",
        "outer_precision",
        "outer_balanced_accuracy",
        "outer_expected_fp_per_3500",
        "outer_expected_fn_per_3500",
        "outer_oof_auroc",
        "outer_oof_auprc",
        "outer_final_weighted_auprc",
    ]
    rows = []
    for strategy, group in results.groupby("strategy", sort=False):
        row: dict[str, Any] = {
            "strategy": strategy,
            "method_group": str(group["method_group"].iloc[0]),
            "outer_folds": int(len(group)),
            "strict_pass_rate": float(group["outer_strict_pass"].mean()),
            "selection_strict_rate": float(group["selection_status"].eq("strict").mean()),
        }
        for column in metric_cols:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "strict_pass_rate",
            "outer_mcc_mean",
            "outer_f1_mean",
            "outer_expected_fp_per_3500_mean",
        ],
        ascending=[False, False, False, True],
    )


def _write_report(
    summary: pd.DataFrame,
    outer_results: pd.DataFrame,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    top = summary.head(12).copy()
    cat = summary.loc[summary["strategy"].eq("CATOPT-A_nested_reference")]
    winner = top.iloc[0]
    lines = [
        "# MASTER Hybrid Nested Governance Raporu",
        "",
        "Bu rapor, Hybrid-C / Hybrid-E adaylarının aynı OOF üzerinde aşırı seçilme riskini kontrol etmek için yapılan nested governance doğrulamasını özetler.",
        "",
        "## Protokol",
        "",
        f"- Preprocessing: `{FEATURE_SET}`",
        f"- Outer seedler: `{metadata['outer_seeds']}`",
        f"- Outer fold sayısı: `{metadata['outer_fold_count']}`",
        f"- Inner fold sayısı: `{metadata['inner_splits']}`",
        "- Her outer fold içinde ağırlık, kalibrasyon ve threshold yalnız inner OOF üzerinde seçildi.",
        "- Outer holdout yalnız seçilmiş kararın final değerlendirmesinde kullanıldı.",
        "- Final/test veri ve historical outer validation seçim için kullanılmadı.",
        "",
        "## Kısa Karar",
        "",
        f"Nested doğrulamada en yüksek özet sıradaki aday: **{winner['strategy']}**.",
    ]
    if not cat.empty:
        cat_row = cat.iloc[0]
        lines.extend(
            [
                "",
                "CATOPT-A referansına göre ortalama farklar:",
                "",
                "| Metrik | Fark |",
                "|---|---:|",
                f"| F1 | {winner['outer_f1_mean'] - cat_row['outer_f1_mean']:+.4f} |",
                f"| MCC | {winner['outer_mcc_mean'] - cat_row['outer_mcc_mean']:+.4f} |",
                f"| FP/3500 | {winner['outer_expected_fp_per_3500_mean'] - cat_row['outer_expected_fp_per_3500_mean']:+.1f} |",
                f"| AUROC | {winner['outer_oof_auroc_mean'] - cat_row['outer_oof_auroc_mean']:+.4f} |",
                f"| Final weighted AUPRC | {winner['outer_final_weighted_auprc_mean'] - cat_row['outer_final_weighted_auprc_mean']:+.4f} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Ortalama Outer Fold Sonuçları",
            "",
            "| Strateji | Strict pass | F1 mean±std | MCC mean±std | FP/3500 mean±std | Sens mean±std | Spec mean±std | AUROC mean±std | AUPRC_final mean±std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            "| {strategy} | {pass_rate:.2f} | {f1:.4f}±{f1s:.4f} | {mcc:.4f}±{mccs:.4f} | {fp:.1f}±{fps:.1f} | {sens:.4f}±{senss:.4f} | {spec:.4f}±{specs:.4f} | {auc:.4f}±{aucs:.4f} | {auprc:.4f}±{auprcs:.4f} |".format(
                strategy=row["strategy"],
                pass_rate=row["strict_pass_rate"],
                f1=row["outer_f1_mean"],
                f1s=row["outer_f1_std"],
                mcc=row["outer_mcc_mean"],
                mccs=row["outer_mcc_std"],
                fp=row["outer_expected_fp_per_3500_mean"],
                fps=row["outer_expected_fp_per_3500_std"],
                sens=row["outer_sensitivity_mean"],
                senss=row["outer_sensitivity_std"],
                spec=row["outer_specificity_mean"],
                specs=row["outer_specificity_std"],
                auc=row["outer_oof_auroc_mean"],
                aucs=row["outer_oof_auroc_std"],
                auprc=row["outer_final_weighted_auprc_mean"],
                auprcs=row["outer_final_weighted_auprc_std"],
            )
        )

    lines.extend(
        [
            "",
            "## Governance Yorumu",
            "",
            "Bu doğrulama, önceki tek OOF üzerinde yapılan geniş weight-grid ve threshold seçiminden doğabilecek selection-overfit riskini ölçmek için tasarlanmıştır. Bir hibrit adayın resmi final pipeline yerine geçebilmesi için yalnız ortalama metriklerinin iyi olması yetmez; strict pass rate, metrik stabilitesi ve CATOPT-A'ya göre anlamlı/kararlı kazanım birlikte değerlendirilmelidir.",
            "",
            "## Çıktılar",
            "",
            "- `results/modeling/nested_governance/nested_outer_fold_results.csv`",
            "- `results/modeling/nested_governance/nested_governance_summary.csv`",
            "- `results/modeling/nested_governance/nested_outer_predictions.csv`",
            "- `results/modeling/nested_governance/nested_selection_details.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-seeds", default="42")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--hye-weight-samples", type=int, default=2500)
    args = parser.parse_args()

    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outer_seeds = _parse_outer_seeds(args.outer_seeds)
    development = _load_development_frame()
    folds = pd.read_csv(FOLD_FILE, dtype={ID_COLUMN: "string"})
    folds[ID_COLUMN] = folds[ID_COLUMN].astype(str)

    h_c_grid = _hybrid_c_weight_grid()
    h_e_vectors = _hybrid_e_weight_vectors(args.hye_weight_samples)
    print(
        f"Prepared nested search grids: Hybrid-C={len(h_c_grid)}, "
        f"Hybrid-E={len(h_e_vectors)}",
        flush=True,
    )

    outer_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    inner_fit_rows: list[pd.DataFrame] = []

    for outer_seed in outer_seeds:
        seed_folds = folds.loc[folds["repeat_seed"].eq(outer_seed)].copy()
        for outer_fold in sorted(seed_folds["fold"].unique()):
            fold_started = time.time()
            hold_ids = set(
                seed_folds.loc[seed_folds["fold"].eq(outer_fold), ID_COLUMN].astype(str)
            )
            outer_train = development.loc[
                ~development[ID_COLUMN].astype(str).isin(hold_ids)
            ].reset_index(drop=True)
            outer_hold = development.loc[
                development[ID_COLUMN].astype(str).isin(hold_ids)
            ].reset_index(drop=True)

            print(
                f"[outer seed={outer_seed} fold={outer_fold}] "
                f"train={len(outer_train)} holdout={len(outer_hold)}",
                flush=True,
            )
            inner_frame, inner_meta = _make_inner_oof(
                outer_train,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                inner_splits=args.inner_splits,
            )
            inner_fit_rows.append(inner_meta)

            strategies, f1_floor = _select_all_strategies(
                inner_frame,
                h_c_grid,
                h_e_vectors,
            )
            outer_score_frame, outer_meta = _fit_score_base_frame(
                outer_train,
                outer_hold,
                seed=RANDOM_STATE + int(outer_seed) * 100 + int(outer_fold),
                fold_key=(int(outer_seed), int(outer_fold), 99),
            )

            for strategy in strategies:
                row, predictions = _evaluate_selected_strategy(
                    strategy,
                    inner_frame,
                    outer_score_frame,
                    f1_floor,
                )
                row.update(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "outer_train_rows": int(len(outer_train)),
                        "outer_hold_rows": int(len(outer_hold)),
                        "selection_f1_floor": float(f1_floor),
                        "outer_output_features": int(outer_meta["output_features"]),
                    }
                )
                outer_rows.append(row)
                predictions["outer_seed"] = int(outer_seed)
                predictions["outer_fold"] = int(outer_fold)
                prediction_frames.append(predictions)
                selection_rows.append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "strategy": strategy.strategy,
                        "selected_variant": strategy.selected_variant,
                        "selection_status": strategy.selection_status,
                        "threshold": strategy.threshold,
                        "selection_f1_floor": float(f1_floor),
                        "calibration_matrix": strategy.calibration_matrix,
                        "weights_json": json.dumps(
                            {
                                BASE_TO_FULL_NAME[member]: float(weight)
                                for member, weight in zip(strategy.members, strategy.weights)
                                if weight > 0
                            },
                            ensure_ascii=False,
                        ),
                        "details_json": json.dumps(strategy.details or {}, ensure_ascii=False),
                    }
                )

            print(
                f"[outer seed={outer_seed} fold={outer_fold}] "
                f"completed in {time.time() - fold_started:.1f}s",
                flush=True,
            )

    outer_results = pd.DataFrame(outer_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    inner_fit = pd.concat(inner_fit_rows, ignore_index=True)
    summary = _summarize_outer_results(outer_results)

    outer_results.to_csv(OUT_DIR / "nested_outer_fold_results.csv", index=False)
    predictions.to_csv(OUT_DIR / "nested_outer_predictions.csv", index=False)
    selections.to_csv(OUT_DIR / "nested_selection_details.csv", index=False)
    inner_fit.to_csv(OUT_DIR / "nested_inner_fit_audit.csv", index=False)
    summary.to_csv(OUT_DIR / "nested_governance_summary.csv", index=False)

    metadata = {
        "status": "completed",
        "outer_seeds": outer_seeds,
        "outer_fold_count": int(
            folds.loc[folds["repeat_seed"].isin(outer_seeds), ["repeat_seed", "fold"]]
            .drop_duplicates()
            .shape[0]
        ),
        "inner_splits": int(args.inner_splits),
        "hye_weight_samples_requested": int(args.hye_weight_samples),
        "hybrid_c_weight_grid_size": len(h_c_grid),
        "hybrid_e_weight_vector_count": len(h_e_vectors),
        "feature_set": FEATURE_SET,
        "outer_validation_used_for_selection": False,
        "historical_outer_validation_used": False,
        "final_or_test_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    (OUT_DIR / "nested_governance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        summary,
        outer_results,
        metadata,
        OUT_DIR / "HYBRID_NESTED_GOVERNANCE_REPORT.md",
    )

    display_cols = [
        "strategy",
        "outer_folds",
        "strict_pass_rate",
        "outer_f1_mean",
        "outer_f1_std",
        "outer_mcc_mean",
        "outer_mcc_std",
        "outer_expected_fp_per_3500_mean",
        "outer_expected_fp_per_3500_std",
        "outer_sensitivity_mean",
        "outer_specificity_mean",
        "outer_oof_auroc_mean",
        "outer_final_weighted_auprc_mean",
    ]
    print("\n=== NESTED GOVERNANCE SUMMARY ===")
    print(summary.loc[:, display_cols].round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
